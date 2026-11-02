"""Simulated-annealing placer over sequence pairs.

Equivalent to RALF's ``main_RP_placement.py`` path: anneal over the
sequence-pair representation with energy ``E = HPWL + sqrt(congestion)``.
This solver is dependency-free and always available; it also serves as the
fallback for ``--placer rl`` when torch is not installed.
"""

import logging
import math
import random

from .cost_model import HPWLMetric, RUDYCongestion, symmetry_penalty
from .sequence_pair import SequencePair

logger = logging.getLogger(__name__)


class SAPackingSolver:
    """Anneal a module placement over the sequence-pair space.

    Energy: ``HPWL + sqrt(RUDY congestion)`` (RALF's annealer energy).
    """

    def __init__(self, problem, *, seed=0, t_init=0.5, t_min=0.05,
                 n_steps=10000, weight_congestion=1.0,
                 weight_symmetry=None):
        self.problem = problem
        self.rng = random.Random(seed)
        self.t_init = t_init
        self.t_min = t_min
        self.n_steps = n_steps
        self.hpwl = HPWLMetric(problem)
        self.rudy = RUDYCongestion(problem, self.hpwl)
        self.w_cong = weight_congestion
        # hard-constraint weight: scale so even a unit deviation dominates
        # a typical HPWL change; overridden only for tests
        if weight_symmetry is None:
            hpwl_scale = max(sum(i['width'] + i['height']
                                 for i in problem.instances), 1)
            weight_symmetry = 10.0 * hpwl_scale if problem.symmetry else 0.0
        self.w_sym = weight_symmetry

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

    def energy(self, sp, rotations):
        placement, _ = self._decode_placement(sp, rotations)
        return (self.hpwl(placement)
                + self.w_cong * math.sqrt(max(self.rudy(placement), 0.0))
                + self.w_sym * symmetry_penalty(placement, self.problem))

    def solve(self):
        n = self.problem.n
        assert n > 0
        sp = SequencePair(list(range(n)), list(range(n)))
        flips = [0] * n

        # geometric cooling schedule matching ALIGN's SA convention
        alpha = math.exp(math.log(self.t_min / self.t_init)
                         / max(self.n_steps, 1))
        T = self.t_init

        best = (sp, list(flips), self.energy(sp, flips))
        cur_sp, cur_flp, cur_e = sp, flips, best[2]

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

        placement, bbox = self._decode_placement(best[0], best[1])
        if self.problem.symmetry:
            placement = self._legalize_symmetry(placement, best[0], best[1])
            bbox = self._bbox_of(placement)
        # report the placement-quality part of the energy (without the
        # hard-constraint symmetry weight, which dwarfs it by design)
        quality = (self.hpwl(placement)
                   + self.w_cong
                   * math.sqrt(max(self.rudy(placement), 0.0)))
        logger.info(f'SA placer ({self.problem.name}): final energy '
                    f'{quality:.2f}, die bbox {bbox}')
        return placement, bbox, best[2]

    def _bbox_of(self, placement):
        insts = self.problem.instances
        ux = uy = 0
        for i, (x, y, _) in placement.items():
            ux = max(ux, x + insts[i]['width'])
            uy = max(uy, y + insts[i]['height'])
        return (ux, uy)

    def _legalize_symmetry(self, placement, sp, flips):
        """Snap symmetric blocks onto a common mirror axis.

        The SA stage gets close but sequence-pair moves rarely land all
        centerlines exactly on one axis.  ALIGN's hard constraint requires
        *exact* centerline equality, so post-process: for each
        SymmetricBlocks group (direction 'V'), compute the median
        centerline of the group's blocks, translate every member
        horizontally to that axis (self-symmetric blocks) or mirror it
        symmetric pairs, then repack vertically to remove any overlap the
        translation introduced.
        """
        from .cost_model import _strict_intersection_area
        insts = self.problem.instances
        pl = dict(placement)
        for group in self.problem.symmetry:
            if group['direction'] != 'V':
                continue  # horizontal-axis symmetry not yet legalized
            axis = self._feasible_axis(pl, group)
            # self-symmetric singles: center on axis
            for pair in group['pairs']:
                if len(pair) == 1:
                    i = pair[0]
                    x, y, o = pl[i]
                    w = insts[i]['width']
                    pl[i] = (int(round(axis - w / 2)), y, o)
                else:
                    i0, i1 = pair
                    w0, w1 = insts[i0]['width'], insts[i1]['width']
                    c = (insts[i0]['height'], insts[i1]['height'])
                    # mirror both about the axis, keeping their shared
                    # y-center (ALIGN also requires aligned y-centers)
                    y0 = pl[i0][1]
                    y1 = pl[i1][1]
                    yc = (y0 + y0 + c[0] + y1 + y1 + c[1]) / 4
                    pl[i0] = (int(round(axis - w0)), int(round(yc - c[0] / 2)), pl[i0][2])
                    pl[i1] = (int(round(axis)), int(round(yc - c[1] / 2)), pl[i1][2])
        # remove overlaps by re-decoding the SP -- the snap may have moved
        # blocks; simplest robust fix is a final greedy de-overlap that
        # pushes colliding blocks apart along x
        pl = self._deoverlap(pl)
        return pl

    def _median_centerline(self, placement, group):
        centers = []
        insts = self.problem.instances
        for pair in group['pairs']:
            for i in pair:
                x, y, _ = placement[i]
                centers.append(x + insts[i]['width'] / 2)
        centers.sort()
        return centers[len(centers) // 2]

    def _feasible_axis(self, placement, group):
        """Median centerline, raised so the widest centered block still
        fits with llx >= 0."""
        axis = self._median_centerline(placement, group)
        insts = self.problem.instances
        for pair in group['pairs']:
            for i in pair:
                axis = max(axis, insts[i]['width'] / 2)
        return axis

    def _deoverlap(self, placement):
        """Greedy horizontal push to remove bbox overlaps (keeps y)."""
        from .cost_model import _strict_intersection_area
        insts = self.problem.instances
        pl = {i: list(v) for i, v in placement.items()}
        for _ in range(200):
            moved = False
            items = sorted(pl.items(), key=lambda kv: kv[1][0])
            boxes = {i: (v[0], v[1], v[0] + insts[i]['width'],
                         v[1] + insts[i]['height']) for i, v in pl.items()}
            for a in range(len(items)):
                for b in range(a + 1, len(items)):
                    i, j = items[a][0], items[b][0]
                    ov = _strict_intersection_area(boxes[i], boxes[j])
                    if ov > 0:
                        # push the right-hand block right past the left one
                        shift = boxes[i][2] - boxes[j][0] + 1
                        pl[j][0] += shift
                        boxes[j] = (pl[j][0], pl[j][1],
                                    pl[j][0] + insts[j]['width'],
                                    pl[j][1] + insts[j]['height'])
                        moved = True
            if not moved:
                break
        return {i: tuple(v) for i, v in pl.items()}
