"""Simulated-annealing placer over sequence pairs.

Equivalent to RALF's ``main_RP_placement.py`` path: anneal over the
sequence-pair representation with energy ``E = HPWL + sqrt(congestion)``.
This solver is dependency-free and always available; it also serves as the
fallback for ``--placer rl`` when torch is not installed.
"""

import logging
import math
import random

from .cost_model import HPWLMetric, RUDYCongestion
from .sequence_pair import SequencePair

logger = logging.getLogger(__name__)


class SAPackingSolver:
    """Anneal a module placement over the sequence-pair space.

    Energy: ``HPWL + sqrt(RUDY congestion)`` (RALF's annealer energy).
    """

    def __init__(self, problem, *, seed=0, t_init=0.5, t_min=0.05,
                 n_steps=10000, weight_congestion=1.0):
        self.problem = problem
        self.rng = random.Random(seed)
        self.t_init = t_init
        self.t_min = t_min
        self.n_steps = n_steps
        self.hpwl = HPWLMetric(problem)
        self.rudy = RUDYCongestion(problem, self.hpwl)
        self.w_cong = weight_congestion

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
        return self.hpwl(placement) + self.w_cong * math.sqrt(
            max(self.rudy(placement), 0.0))

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
        logger.info(f'SA placer ({self.problem.name}): final energy '
                    f'{best[2]:.2f}, die bbox {bbox}')
        return placement, bbox, best[2]
