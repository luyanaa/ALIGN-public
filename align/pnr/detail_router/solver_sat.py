"""SAT backend: solve a ``RegionProblem`` with ``python-sat``.

Boolean variable per candidate path (x_p = 1 iff path p is used).

Hard constraints encoded as CNF:
  * cover      -- for every net pin, OR over the paths touching it.
  * mutex      -- for every (a,b) conflict pair, clause (¬x_a ∨ ¬x_b).
  * at most one path per pin *pair* (a net may legitimately pick two
    different routes between the same pins only when making a loop;
    forbidding it is a safe strengthening that shrinks the space).
  * lazy connectivity cuts (at-least-one bridging path) added on each
    unsatisfiable/disconnected iteration.

Soft ``Prefer`` terms are applied as a post-pass: after a feasible
solution exists we re-solve under the assumption that preferred paths
are off, count how many must be on, and keep the cheaper model.
"""

import logging

logger = logging.getLogger(__name__)

try:
    from pysat.solvers import Solver
    _PYSAT = True
except ImportError:                                   # pragma: no cover
    _PYSAT = False


def sat_available():
    return _PYSAT


def solve_region(problem, *, solver='glucose4', max_iters=50,
                 verbose=False):
    """Solve one region.  Returns ``set(path_idx)`` or ``None``.

    ``problem`` is a ``constraints.RegionProblem``.
    """
    if not _PYSAT:
        raise RuntimeError('python-sat is not installed')

    n = len(problem.paths)
    var = list(range(1, n + 1))           # 1-based SAT var per path

    with Solver(name=solver) as s:
        # cover each pin at least once
        for (net, pin), ids in problem.paths_touching_pin.items():
            s.add_clause([var[i] for i in ids])
        # pairwise-path uniqueness per (net, a, b)
        by_pair = {}
        for i, p in enumerate(problem.paths):
            key = (problem.net_of[i],
                   min(p.pin_a, p.pin_b), max(p.pin_a, p.pin_b))
            by_pair.setdefault(key, []).append(i)
        for ids in by_pair.values():
            for a in range(len(ids)):
                for b in range(a + 1, len(ids)):
                    s.add_clause([-var[ids[a]], -var[ids[b]]])
        # spacing mutexes
        for a, b in problem.mutex:
            s.add_clause([-var[a], -var[b]])

        for it in range(max_iters):
            if not s.solve():
                logger.info('SAT region UNSAT '
                            f'(iter {it})')
                return None
            model = set(s.get_model())
            selected = {i for i in range(n) if var[i] in model}
            bad = problem.check_connectivity(selected)
            if not bad:
                logger.info(f'SAT region solved in {it + 1} iterations, '
                            f'{len(selected)} paths')
                return _prefer_prune(problem, selected, s, var)
            cuts = list(problem.connectivity_cuts(selected))
            if not cuts:
                return None
            for _net, bridge in cuts:
                s.add_clause([var[i] for i in bridge])
        logger.warning('SAT region: hit iteration limit')
        return None


def _prefer_prune(problem, selected, solver, var):
    """Cheap soft-preference pass: try to drop selected non-preferred
    paths when an equally feasible model exists without them.

    ``Prefer`` weights mark desirable paths; here we instead *minimize
    wire*: attempt to turn off each selected non-preferred path one by
    one under assumptions, keeping the removal when the problem stays
    satisfiable and connectivity is preserved.
    """
    preferred = {i for i, w in problem.prefer}
    removable = [i for i in selected if i not in preferred]
    for i in removable:
        trial = selected - {i}
        if not solver.solve(assumptions=[-var[i]]):
            continue
        model = set(solver.get_model())
        new_sel = {j for j in range(len(problem.paths))
                   if var[j] in model}
        if not problem.check_connectivity(new_sel):
            selected = new_sel
    return selected
