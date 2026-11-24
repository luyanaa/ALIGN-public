"""Simulated-annealing placer over sequence pairs.

Equivalent to RALF's ``main_RP_placement.py`` path: anneal over the
sequence-pair representation with energy ``E = HPWL + sqrt(congestion)``.
This solver is dependency-free and always available; it also serves as the
fallback for ``--placer rl`` when torch is not installed.
"""

import logging
import math
import random

from .cost_model import (HPWLMetric, RUDYCongestion, raw_components,
                         symmetry_axes)
from . import legalizer
from .sequence_pair import SequencePair

logger = logging.getLogger(__name__)


class SAPackingSolver:
    """Anneal a module placement over the sequence-pair space.

    Energy: ``HPWL + sqrt(RUDY congestion) + w_sym * symmetry_penalty``.

    Penalty weights are magnitude-matched to the HPWL term on the initial
    placement (``w_i = hpwl0 / raw_i0``) and then grown by subgradient
    ascent from the *raw* violation magnitude (IdeaPlaceEx's penalty-weight
    scheme).  Raw sub-terms are tracked separately so termination can be
    driven by constraint satisfaction rather than a fixed iteration count.
    """

    def __init__(self, problem, *, seed=0, t_init=0.5, t_min=0.05,
                 n_steps=10000, weight_congestion=1.0,
                 weight_symmetry=None, update_interval=None,
                 stagnation_checks=5, symmetry_tol_factor=0.05,
                 max_weight_growth=100.0):
        self.problem = problem
        self.rng = random.Random(seed)
        self.t_init = t_init
        self.t_min = t_min
        self.n_steps = n_steps
        self.hpwl = HPWLMetric(problem)
        self.rudy = RUDYCongestion(problem, self.hpwl)
        self.w_cong = weight_congestion
        self.stagnation_checks = stagnation_checks
        self.max_weight_growth = max_weight_growth

        # Subgradient weight updates run every K steps (K = n_steps // 100).
        self.update_interval = (max(n_steps // 100, 1)
                                if update_interval is None
                                else max(update_interval, 1))

        # --- initial-placement sub-terms and magnitude-matched weights ----
        n = problem.n
        init_sp = SequencePair(list(range(n)), list(range(n)))
        init_placement, _ = self._decode_placement(init_sp, [0] * n)
        # Per-group symmetry axes (B2): explicit solver state, seeded from the
        # feasible-axis rule and thereafter pulled by the (axis-explicit)
        # penalty as SA moves the group's members.
        self.sym_axes = [
            self._feasible_axis(init_placement, group)
            for group in problem.symmetry] or []
        init_parts = self.energy_parts(init_sp, [0] * n)
        self.hpwl0 = init_parts.hpwl
        self.raw0 = init_parts

        # Symmetry weight: explicit ctor override wins (tests), otherwise
        # magnitude-match so its initial contribution ~= HPWL.  A zero
        # initial violation cannot be matched (division by zero); fall back
        # to unit scale so the term can still react if violated later.
        if weight_symmetry is not None:
            self.w_sym = weight_symmetry
            self.adapt_symmetry = False
        elif problem.symmetry:
            self.w_sym = self._match_weight(init_parts.hpwl,
                                            init_parts.symmetry)
            self.adapt_symmetry = True
        else:
            self.w_sym = 0.0
            self.adapt_symmetry = False
        self.w_sym_init = self.w_sym
        self.w_sym_cap = self.max_weight_growth * max(abs(self.w_sym), 1.0)

        # Overlap is not part of the sequence-pair energy (decoding is
        # overlap-free) but is tracked raw for the violation-driven stop.
        self.w_overlap = 0.0

        # Violation tolerance mirroring IdeaPlaceEx: a *length* violation
        # <= 0.05 * sqrt(total instance area); symmetry_penalty is squared,
        # so square the tolerance.
        total_area = sum(i['width'] * i['height']
                         for i in problem.instances)
        self.symmetry_tol = ((symmetry_tol_factor
                              * math.sqrt(max(total_area, 1.0))) ** 2)
        self.last_parts = init_parts

    @staticmethod
    def _match_weight(hpwl0, raw0):
        """``lambda = hpwl0 / max(raw0, eps)`` with a zero-violation guard."""
        eps = 1e-9 * max(abs(hpwl0), 1.0)
        if raw0 > eps:
            return hpwl0 / raw0
        # Already satisfied at init: unit-scale weight (no division blowup).
        return max(abs(hpwl0), 1.0)

    def _decode_placement(self, sp, flips):
        insts = self.problem.instances
        widths = [i['width'] for i in insts]
        heights = [i['height'] for i in insts]
        # flips are not geometric rotations; rectangles keep their extents.
        # (ALIGN leaf transforms support mirroring only, not 90-deg rotation.)
        positions, bbox = sp.decode(widths, heights)
        placement = {i: (positions[i][0], positions[i][1], flips[i])
                     for i in range(self.problem.n)}
        return placement, bbox

    def energy_parts(self, sp, rotations):
        """Return the unweighted :class:`CostComponents` for a state.

        The symmetry term uses the explicit per-group axis (B2).  To keep SA
        cheap the axis is re-derived as the median member centreline on each
        call; storing it on ``self.sym_axes`` keeps it as state and lets the
        penalty effectively pull the axis toward feasibility.
        """
        placement, _ = self._decode_placement(sp, rotations)
        axes = symmetry_axes(placement, self.problem)
        if axes is not None:
            self.sym_axes = axes
        parts = raw_components(placement, self.problem, self.hpwl, self.rudy,
                               axes=axes)
        self.last_parts = parts
        return parts

    def _weighted_energy(self, parts):
        return (parts.hpwl
                + self.w_cong * math.sqrt(max(parts.congestion, 0.0))
                + self.w_sym * parts.symmetry
                + self.w_overlap * parts.overlap)

    def energy(self, sp, rotations):
        return self._weighted_energy(self.energy_parts(sp, rotations))

    def _update_weights(self, parts):
        """Subgradient ascent on the symmetry penalty weight.

        ``delta = 0.01 * hpwl0 * (raw / max(raw0, eps))`` so one unit of the
        initial-scale violation contributes ~1% of HPWL per update; growth is
        capped at ``max_weight_growth * max(|w_init|, 1)`` to avoid runaway.
        Congestion is a quality term (fixed weight), not a violation penalty,
        and overlap is unused by the sequence-pair energy.
        """
        if not self.adapt_symmetry:
            return False
        if parts.symmetry <= 0.0:
            return False
        eps = 1e-9 * max(abs(self.hpwl0), 1.0)
        step = 0.01 * self.hpwl0 / max(self.raw0.symmetry, eps)
        new_w = min(self.w_sym + step * parts.symmetry, self.w_sym_cap)
        if new_w == self.w_sym:
            return False
        logger.debug(
            f'SA placer ({self.problem.name}): w_sym {self.w_sym:.4g} -> '
            f'{new_w:.4g} (raw symmetry {parts.symmetry:.4g})')
        self.w_sym = new_w
        return True

    def _violations_ok(self, parts):
        """Raw-constraint feasibility used by the early stop."""
        return (parts.overlap <= 0.0
                and parts.symmetry <= self.symmetry_tol)

    def solve(self):
        n = self.problem.n
        assert n > 0
        sp = SequencePair(list(range(n)), list(range(n)))
        flips = [0] * n

        # geometric cooling schedule matching ALIGN's SA convention
        alpha = math.exp(math.log(self.t_min / self.t_init)
                         / max(self.n_steps, 1))
        T = self.t_init

        cur_parts = self.energy_parts(sp, flips)
        cur_e = self._weighted_energy(cur_parts)
        best = (sp, list(flips), cur_e)
        cur_sp, cur_flp = sp, flips

        K = self.update_interval
        checks = 0
        last_improve = 0
        best_check_e = cur_e
        stopped_early = False

        for step in range(self.n_steps):
            # perturb: either the sequence pair or one flip bit
            if self.rng.random() < 0.8:
                new_sp = cur_sp.perturb(self.rng)
                new_flp = cur_flp
            else:
                new_sp = cur_sp
                new_flp = list(cur_flp)
                i = self.rng.randrange(n)
                if self.problem.instances[i].get('rotatable', True):
                    # {0: N, 1: S(vertical flip)} -- the only orientation
                    # change representable downstream by sX/sY
                    new_flp[i] = 1 - new_flp[i]
            new_e = self.energy(new_sp, new_flp)
            dE = new_e - cur_e
            if dE <= 0 or self.rng.random() < math.exp(-dE / max(T, 1e-12)):
                cur_sp, cur_flp, cur_e = new_sp, new_flp, new_e
                if cur_e < best[2]:
                    best = (cur_sp, list(cur_flp), cur_e)
            T *= alpha

            # ---- periodic penalty-weight update + violation stop --------
            if (step + 1) % K != 0:
                continue
            parts = self.energy_parts(cur_sp, cur_flp)
            if self._update_weights(parts):
                # energy scale changed: re-evaluate the incumbent under the
                # new weights so comparisons remain consistent
                cur_e = self._weighted_energy(parts)
                best = (best[0], best[1],
                        self._weighted_energy(
                            self.energy_parts(best[0], best[1])))
            checks += 1
            if cur_e < best_check_e - 1e-12 * max(abs(best_check_e), 1.0):
                best_check_e = cur_e
                last_improve = checks
            if (self._violations_ok(parts)
                    and checks - last_improve >= self.stagnation_checks):
                stopped_early = True
                logger.info(
                    f'SA placer ({self.problem.name}): early stop at step '
                    f'{step + 1}/{self.n_steps} (raw overlap '
                    f'{parts.overlap:.3g}, raw symmetry {parts.symmetry:.3g} '
                    f'<= tol {self.symmetry_tol:.3g})')
                break

        placement, bbox = self._decode_placement(best[0], best[1])
        if self.problem.symmetry:
            placement = self._legalize_symmetry(placement, best[0], best[1])
            bbox = self._bbox_of(placement)

        final_parts = raw_components(placement, self.problem,
                                     self.hpwl, self.rudy,
                                     axes=(self.sym_axes
                                           if self.problem.symmetry else None))
        self.last_parts = final_parts
        energy = self._weighted_energy(final_parts)
        logger.info(
            f'SA placer ({self.problem.name}): final energy {energy:.2f}, '
            f'die bbox {bbox}, raw hpwl {final_parts.hpwl:.2f}, '
            f'congestion {final_parts.congestion:.2f}, '
            f'overlap {final_parts.overlap:.3g}, '
            f'symmetry {final_parts.symmetry:.3g}, '
            f'w_cong {self.w_cong:.4g}, w_sym {self.w_sym:.4g}, '
            f'w_overlap {self.w_overlap:.4g}'
            + (f', stopped_early={stopped_early}' if stopped_early else ''))
        return placement, bbox, energy

    def _bbox_of(self, placement):
        insts = self.problem.instances
        ux = uy = 0
        for i, (x, y, _) in placement.items():
            ux = max(ux, x + insts[i]['width'])
            uy = max(uy, y + insts[i]['height'])
        return (ux, uy)

    def _legalize_symmetry(self, placement, sp=None, flips=None):
        """Legalise the SA placement via the shared :mod:`.legalizer`.

        Thin wrapper preserving the historical method contract: the
        per-group axes derived from the input placement are stored on
        ``self.sym_axes`` (read by the final cost evaluation), and ``sp`` /
        ``flips`` are accepted but unused (the legaliser operates purely on
        the decoded placement).
        """
        pl, axes = legalizer.legalize(self.problem, placement)
        self.sym_axes = list(axes)
        return pl

    # -- thin wrappers over the shared legaliser -----------------------------
    # Kept on the class for backward compatibility (``sym_axes`` state and the
    # ``_median_centerline`` / ``_feasible_axis`` contract used by tests); the
    # implementation lives in :mod:`.legalizer`.

    def _snap_group(self, pl, group, axis):
        legalizer.snap_group(self.problem, pl, group, axis)

    def _median_centerline(self, placement, group, direction=None):
        return legalizer.median_centerline(placement, self.problem, group,
                                           direction)

    def _feasible_axis(self, placement, group):
        return legalizer.feasible_axis(placement, self.problem, group)

    def _sym_partners(self, groups):
        return legalizer.sym_partners(groups)

    def _deoverlap(self, placement, groups=None):
        return legalizer.deoverlap(placement, self.problem, groups)
