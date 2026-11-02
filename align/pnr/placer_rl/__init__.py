"""RALF-inspired placer backends for ALIGN.

Implements the placement cost model and solvers described in:
    J. Ratschenberger, H. Pretl, "RALF: A Reinforcement Learning Assisted
    Automated Analog Layout Design Flow", TechRxiv, June 2024.
    DOI: 10.36227/techrxiv.171742468.82130474/v1

Two interchangeable solvers are provided on top of a common cost model:

* ``RLPlacementSolver``  -- sequential placement trained with PPO
  (requires the optional ``torch`` dependency; see ``align[pnn]`` or install
  torch separately). Selected with ``--placer rl``.
* ``SAPackingSolver``    -- simulated-annealing over sequence pairs, matching
  RALF's ``main_RP_placement.py`` baseline. Always available; used as the
  fallback when torch is not installed.

Both produce placements in the scaled-placement-verilog domain so the results
flow into the standard ``3_pnr:route`` stage unchanged.
"""

from .cost_model import PlacementProblem, HPWLMetric, RUDYCongestion  # noqa: F401
from .sa_solver import SAPackingSolver  # noqa: F401
from .driver import rl_placer_driver  # noqa: F401
