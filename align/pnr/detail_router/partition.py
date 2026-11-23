"""Spatial partitioning and multi-core routing driver helpers.

Both solver backends are effectively single-threaded (python-sat's
Minisat-family solvers and CBC/HiGHS are sequential).  Multi-core
speedup therefore comes from *partitioning*: group nets whose candidate
footprints are spatially disjoint into regions, then solve each region
concurrently with ``multiprocessing``.

Partitioning strategy
---------------------
1.  Compute a *footprint* per net = bounding box of all its candidate
    cells (expanded by one pitch).
2.  Union-find nets whose footprints overlap on any shared layer.
3.  Each resulting component is an independent region: its candidate
    mutexes only reference nets inside the region, so regions can be
    solved in parallel without sharing state.
"""

import logging
import multiprocessing
from collections import defaultdict

from .candidates import build_region_candidates
from .constraints import build_problem

logger = logging.getLogger(__name__)


def _footprint(grid, cells):
    """Physical bbox of a candidate's cells.

    ``cells`` are global-lattice keys ``(layer, gx, gy)`` in ``(x, y)``
    order; ``cell_anchor`` is the canonical key -> physical map (the
    same point on every layer).
    """
    xs, ys = [], []
    for layer, gx, gy in cells:
        x, y = grid.cell_anchor(layer, (gx, gy))
        xs.append(x)
        ys.append(y)
    return (min(xs), min(ys), max(xs), max(ys))


def _overlap(a, b, margin):
    return not (a[2] + margin < b[0] or b[2] + margin < a[0]
                or a[3] + margin < b[1] or b[3] + margin < a[1])


def partition_nets(net_pins, grid, max_region_nets=20, margin=None):
    """Group ``net_pins`` keys into disjoint spatial regions.

    Returns a list of ``{net: pins}`` region dicts.
    """
    if margin is None:
        margin = max(m.pitch for m in grid.metal_list) * 2
    # candidate footprints come from pin access-point bboxes (cheap
    # approximation); ``pins`` is a list of access-point *sets*
    foot = {}
    for net, pins in net_pins.items():
        pts = [p for pin in pins for p in pin]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        foot[net] = (min(xs), min(ys), max(xs), max(ys))

    nets = list(net_pins)
    parent = list(range(len(nets)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(len(nets)):
        for j in range(i + 1, len(nets)):
            if _overlap(foot[nets[i]], foot[nets[j]], margin):
                union(i, j)

    comp = defaultdict(list)
    for i, n in enumerate(nets):
        comp[find(i)].append(n)

    # split over-large components by simple net-count chunking
    regions = []
    for members in comp.values():
        if len(members) <= max_region_nets:
            regions.append({n: net_pins[n] for n in members})
        else:
            for i in range(0, len(members), max_region_nets):
                chunk = members[i:i + max_region_nets]
                regions.append({n: net_pins[n] for n in chunk})
    logger.info(f'partitioned {len(nets)} nets into {len(regions)} regions')
    return regions


def _solve_one(args):
    """Worker: build candidates + solve one region.  Runs in a child
    process; must be picklable, so everything it needs is passed in."""
    region_nets, grid, guide_map, backend, solver_kw, bnd, prects, \
        reserve = args
    if reserve:
        # pin rects of nets that are *not* being routed here but whose
        # metal must still be respected (symmetric mirror partners).
        prects = dict(prects)
        for net, rects in reserve.items():
            prects.setdefault(net, rects)
    cands = build_region_candidates(region_nets, grid, guide_map,
                                    boundary_pins=bnd,
                                    pin_rects=prects)
    if any(not v for v in cands.values()):
        return None
    n_pins = {net: len(pins) for net, pins in region_nets.items()}
    prob = build_problem(cands, n_pins)
    if backend == 'mip':
        from .solver_mip import solve_region
    else:
        from .solver_sat import solve_region
    sel = solve_region(prob, **solver_kw)
    if sel is None:
        return None
    # return every selected path grouped by net
    out = defaultdict(list)
    for i in sel:
        out[prob.net_of[i]].append(prob.paths[i])
    return dict(out)


def route_parallel(net_pins, grid, guide_map=None, backend='sat',
                   solver_kw=None, n_workers=None,
                   max_region_nets=20, boundary_pins=None,
                   pin_rects=None, reserve_pin_rects=None):
    """Route all nets, partitioned into regions solved in parallel.

    ``boundary_pins`` is ``{net: pin_index}`` marking the synthetic
    die-edge escape pin each net may carry; it is forwarded to the
    candidate builder so those pins are not reserved as foreign
    blockages.  ``pin_rects`` is ``{net: [{layer:[rect]}]}`` used for
    precise pin-attachment overlap tests.  ``reserve_pin_rects`` is an
    extra ``{net: [...]}`` map of pin metal that is *not* routed here
    but must still be respected (a symmetric mirror partner whose pins
    stay real while its wires are derived).

    Returns ``{net: [CandidatePath]}`` or ``None`` on any region
    failure.
    """
    solver_kw = solver_kw or {}
    regions = partition_nets(net_pins, grid, max_region_nets)
    if not regions:
        return {}
    bnd = boundary_pins or {}
    prects = pin_rects or {}
    reserve = reserve_pin_rects or {}
    args = [(r, grid, guide_map or {}, backend, solver_kw,
             {n: i for n, i in bnd.items() if n in r},
             {n: prects[n] for n in r if n in prects}, reserve)
            for r in regions]
    if len(regions) == 1 or n_workers == 1:
        results = [_solve_one(a) for a in args]
    else:
        with multiprocessing.Pool(
                processes=n_workers
                or min(len(regions), multiprocessing.cpu_count())) as p:
            results = p.map(_solve_one, args)
    if any(r is None for r in results):
        return None
    out = defaultdict(list)
    for r in results:
        for net, paths in r.items():
            out[net].extend(paths)
    return dict(out)
