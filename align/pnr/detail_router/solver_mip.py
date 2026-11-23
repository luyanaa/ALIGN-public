"""MIP backend: solve a ``RegionProblem`` with ``mip`` (CBC/HiGHS).

Same model as ``solver_sat`` but as an integer program, which lets the
``Prefer`` soft costs participate directly in the objective instead of
a post-pass::

    min  sum_p cost_p * x_p  -  W * sum_p prefer_p * x_p
    s.t. cover, mutex, one-path-per-pair, lazy connectivity cuts
         x_p in {0,1}

Multi-core: ``Model.threads`` is set for solvers that support it
(CBC ignores it; HiGHS honours it when installed via ``mip[highs]``).
"""

import logging

logger = logging.getLogger(__name__)

try:
    import mip
    _MIP = True
except ImportError:                                   # pragma: no cover
    _MIP = False


def mip_available():
    return _MIP


def solve_region(problem, *, solver_name=None, threads=0, max_iters=50,
                 prefer_weight=0.5, verbose=False):
    """Solve one region.  Returns ``set(path_idx)`` or ``None``."""
    if not _MIP:
        raise RuntimeError('mip is not installed')

    n = len(problem.paths)
    m = mip.Model()
    if solver_name:
        m.solver_name = solver_name
    if threads:
        m.threads = threads
    if not verbose:
        m.verbose = 0
    x = [m.add_var(var_type=mip.BINARY) for _ in range(n)]

    # cover each pin
    for (net, pin), ids in problem.paths_touching_pin.items():
        m.add_constr(mip.xsum(x[i] for i in ids) >= 1)
    # at most one path per pin pair
    by_pair = {}
    for i, p in enumerate(problem.paths):
        key = (problem.net_of[i],
               min(p.pin_a, p.pin_b), max(p.pin_a, p.pin_b))
        by_pair.setdefault(key, []).append(i)
    for ids in by_pair.values():
        m.add_constr(mip.xsum(x[i] for i in ids) <= 1)
    # mutexes
    for a, b in problem.mutex:
        m.add_constr(x[a] + x[b] <= 1)

    prefer = {i: w for i, w in problem.prefer}
    m.objective = mip.minimize(
        mip.xsum((problem.paths[i].cost
                  - prefer_weight * prefer.get(i, 0.0)) * x[i]
                 for i in range(n)))

    for it in range(max_iters):
        status = m.optimize()
        if status != mip.OptimizationStatus.OPTIMAL:
            logger.info(f'MIP region infeasible/status={status} '
                        f'(iter {it})')
            return None
        selected = {i for i in range(n) if x[i].x > 0.5}
        bad = problem.check_connectivity(selected)
        if not bad:
            logger.info(f'MIP region solved in {it + 1} iterations, '
                        f'{len(selected)} paths')
            return selected
        cuts = list(problem.connectivity_cuts(selected))
        if not cuts:
            return None
        for _net, bridge in cuts:
            m.add_constr(mip.xsum(x[i] for i in bridge) >= 1)
    logger.warning('MIP region: hit iteration limit')
    return None
