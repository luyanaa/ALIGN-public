"""Candidate-path generation for the detailed router.

Each net's routing is a *tree* of pairwise pin-to-pin paths.  Rather
than enumerate atomic cell segments (which makes the solver
responsible for connectivity), we enumerate complete *candidate paths*
between every pair of pins of a net.  A path is a small legal wire/via
construction on the routing grid (straight run, via stack, or an
L/Z route using one perpendicular leg on a neighbouring layer).
Choosing any set of paths that touches every pin yields a routable
net; a lazy connectivity loop in the solver prunes choices that leave
the tree disconnected.

This is the same modelling idea Intel's AMSR uses (``Route``/``Tie``
candidate objects feeding SAT clauses) but implemented with plain
Python data structures and no legacy file formats.
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PathPiece:
    """One wire run or via inside a candidate path."""
    kind: str                     # 'wire' or 'via'
    layer: str                    # metal layer (for a via: lower layer)
    cells: frozenset              # (layer, ti, si) grid cells occupied
    bbox: tuple = ()              # (x0,y0,x1,y1) grid units, wires only


@dataclass(frozen=True)
class CandidatePath:
    """A complete candidate route between two pins of one net."""
    net: str
    pin_a: int
    pin_b: int
    pieces: tuple                 # tuple[PathPiece]
    cost: float = 0.0             # wirelength + via-count heuristic
    cells: frozenset = frozenset()  # union of piece cells (for mutexes)


class CandidateGenerator:
    """Enumerate candidate paths for one net on a ``RoutingGrid``."""

    def __init__(self, grid, max_layer=6):
        self.grid = grid
        # only route on the lower metals by default (leaf-level signal)
        self.layers = [m for m in grid.metal_list
                       if int(m.name[1:]) <= max_layer]
        self.v_layers = [m.name for m in self.layers
                         if m.direction == 'V']
        self.h_layers = [m.name for m in self.layers
                         if m.direction == 'H']

    # -- geometry helpers --------------------------------------------------
    def cell_center(self, layer, ti, si):
        m = self.grid.metals[layer]
        if m.direction == 'V':
            return (self.grid.track_coord(layer, ti),
                    self.grid.seg_coord(layer, si))
        return (self.grid.seg_coord(layer, si),
                self.grid.track_coord(layer, ti))

    def cell_at(self, layer, x, y):
        m = self.grid.metals[layer]
        ti = self.grid.track_index(layer, x if m.direction == 'V' else y)
        si = self.grid.seg_index(layer, y if m.direction == 'V' else x)
        return (ti, si)

    def _wire_cells_v(self, layer, x, y0, y1):
        """Cells of a vertical wire at ``x`` covering ``y0``..``y1``."""
        ti, sa = self.cell_at(layer, x, y0)
        _, sb = self.cell_at(layer, x, y1)
        lo, hi = sorted((sa, sb))
        return frozenset((layer, ti, s) for s in range(lo, hi + 1))

    def _wire_cells_h(self, layer, y, x0, x1):
        ti, sa = self.cell_at(layer, x0, y)
        _, sb = self.cell_at(layer, x1, y)
        lo, hi = sorted((sa, sb))
        return frozenset((layer, ti, s) for s in range(lo, hi + 1))

    def _free(self, cells):
        return all((c[1], c[2]) in self.grid.allowed[c[0]]
                   for c in cells)

    def _wire_bbox(self, layer, cells):
        m = self.grid.metals[layer]
        hw = m.width // 2
        tis = [t for (_l, t, s) in cells]
        sis = [s for (_l, t, s) in cells]
        c0 = self.grid.track_coord(layer, min(tis))
        c1 = self.grid.track_coord(layer, max(tis))
        s0 = self.grid.seg_coord(layer, min(sis))
        s1 = self.grid.seg_coord(layer, max(sis)) + m.pitch
        if m.direction == 'V':
            return (c0 - hw, s0, c1 + hw, s1)
        return (s0, c0 - hw, s1, c1 + hw)

    def _via_stack(self, layer_lo, layer_hi, x, y):
        """Cells of a via chain between ``layer_lo`` and ``layer_hi``.

        Returns the frozenset of occupied cells on every intermediate
        layer (inclusive), or ``None`` if any layer lacks a free cell.
        """
        names = [m.name for m in self.grid.metal_list]
        i0, i1 = sorted((names.index(layer_lo), names.index(layer_hi)))
        cells = set()
        for name in names[i0:i1 + 1]:
            ti, si = self.cell_at(name, x, y)
            if (ti, si) not in self.grid.allowed[name]:
                return None
            cells.add((name, ti, si))
        return frozenset(cells)

    # -- path constructors --------------------------------------------------
    def _mkpath(self, net, a, b, pieces):
        cells = frozenset().union(*(p.cells for p in pieces))
        n_vias = sum(1 for p in pieces if p.kind == 'via')
        n_wire = sum(len(p.cells) for p in pieces if p.kind == 'wire')
        return CandidatePath(net, a, b, tuple(pieces),
                             float(n_wire + 5 * n_vias), cells)

    def pin_access_cells(self, pin_pts, max_cells=4):
        """Return up to ``max_cells`` ``(layer, ti, si)`` access cells
        for a pin whose ``pin_pts`` is a set of ``(x, y, layer)``
        access points.  Points on the pin's own layer come first, then
        one layer above/below so a via can escape."""
        names = [m.name for m in self.grid.metal_list]
        legal = {m.name for m in self.layers}
        out = []
        seen = set()
        for x, y, layer in sorted(pin_pts):
            if layer not in names:
                continue
            base = names.index(layer)
            for d in (0, 1, -1, 2, -2):
                li = base + d
                if not (0 <= li < len(names)):
                    continue
                name = names[li]
                if name not in legal:
                    continue
                m = self.grid.metals[name]
                ti = self.grid.track_index(
                    name, x if m.direction == 'V' else y)
                si = self.grid.seg_index(
                    name, y if m.direction == 'V' else x)
                key = (name, ti, si)
                if key not in seen and (ti, si) in self.grid.allowed[name]:
                    seen.add(key)
                    out.append(key)
        return out[:max_cells]

    def _vias_between(self, lx, ly, x, y):
        """Via-stack piece connecting two different layers, or []."""
        if lx == ly:
            return []
        cells = self._via_stack(lx, ly, x, y)
        if cells is None:
            return None
        lo = min(lx, ly, key=lambda l: int(l[1:]))
        return [PathPiece('via', lo, cells, ())]

    def straight(self, net, a, b, A, B):
        """A and B are ``(x, y, layer)`` pins.  Straight or via paths."""
        ax, ay, la = A
        bx, by, lb = B
        out = []
        # pure via stack: same x,y on different layers
        if abs(ax - bx) < 1 and abs(ay - by) < 1 and la != lb:
            v = self._vias_between(la, lb, ax, ay)
            if v is not None:
                out.append(self._mkpath(net, a, b, v))
            return out
        # vertical straight: same x column
        if abs(ax - bx) < 1:
            v_choices = {la, lb} & set(self.v_layers)
            for lay in v_choices:
                cells = self._wire_cells_v(lay, ax, ay, by)
                if not self._free(cells):
                    continue
                via_a = self._vias_between(la, lay, ax, ay)
                via_b = self._vias_between(lb, lay, bx, by)
                if via_a is None or via_b is None:
                    continue
                piece = PathPiece('wire', lay, cells,
                                  self._wire_bbox(lay, cells))
                out.append(self._mkpath(net, a, b,
                                        via_a + [piece] + via_b))
        # horizontal straight: same y row
        if abs(ay - by) < 1:
            h_choices = {la, lb} & set(self.h_layers)
            for lay in h_choices:
                cells = self._wire_cells_h(lay, ay, ax, bx)
                if not self._free(cells):
                    continue
                via_a = self._vias_between(la, lay, ax, ay)
                via_b = self._vias_between(lb, lay, bx, by)
                if via_a is None or via_b is None:
                    continue
                piece = PathPiece('wire', lay, cells,
                                  self._wire_bbox(lay, cells))
                out.append(self._mkpath(net, a, b,
                                        via_a + [piece] + via_b))
        return out

    def l_routes(self, net, a, b, A, B, alt_layers=True):
        """L-shaped routes via two Manhattan corners."""
        ax, ay, la = A
        bx, by, lb = B
        out = []
        v_opts = ([la] if la in self.v_layers else []) + \
            [l for l in self.v_layers if l != la]
        h_opts = ([lb] if lb in self.h_layers else []) + \
            [l for l in self.h_layers if l != lb]
        if not alt_layers:
            v_opts, h_opts = v_opts[:1], h_opts[:1]
        corners = ((ax, by, 'v_first'), (bx, ay, 'h_first'))
        for cx, cy, mode in corners:
            for lv in v_opts[:2]:
                for lh in h_opts[:2]:
                    if lv == lh:
                        continue
                    pieces = []
                    if mode == 'v_first':
                        # V leg from A to corner, H leg corner to B
                        c_v = self._wire_cells_v(lv, ax, ay, by)
                        c_h = self._wire_cells_h(lh, by, ax, bx)
                        jx, jy = ax, by
                    else:
                        c_h = self._wire_cells_h(lh, ay, ax, bx)
                        c_v = self._wire_cells_v(lv, bx, ay, by)
                        jx, jy = bx, ay
                    if not (self._free(c_v) and self._free(c_h)):
                        continue
                    va = self._vias_between(
                        la, lv if mode == 'v_first' else lh, ax, ay)
                    vb = self._vias_between(
                        lb, lv if mode == 'h_first' else lh, bx, by)
                    vj = self._vias_between(lv, lh, jx, jy)
                    if va is None or vb is None or vj is None:
                        continue
                    pw_v = PathPiece('wire', lv, c_v,
                                     self._wire_bbox(lv, c_v))
                    pw_h = PathPiece('wire', lh, c_h,
                                     self._wire_bbox(lh, c_h))
                    out.append(self._mkpath(
                        net, a, b,
                        va + [pw_v] + vj + [pw_h] + vb))
        return out

    # -- public -------------------------------------------------------------
    def _pin_centroid(self, pin_pts):
        xs = [p[0] for p in pin_pts]
        ys = [p[1] for p in pin_pts]
        return (sum(xs) // len(xs), sum(ys) // len(ys))

    def _choose_pairs(self, pin_pts_list, k=4):
        """Select pin-pairs to connect: MST over pin centroids plus
        each pin's ``k`` nearest neighbours (bounded for many-pin
        nets)."""
        n = len(pin_pts_list)
        if n <= 2:
            return [(0, 1)] if n == 2 else []
        cents = [self._pin_centroid(p) for p in pin_pts_list]
        dist = [[abs(cents[i][0] - cents[j][0])
                 + abs(cents[i][1] - cents[j][1])
                 for j in range(n)] for i in range(n)]
        # Prim's MST
        in_tree = {0}
        edges = set()
        while len(in_tree) < n:
            best = None
            for i in in_tree:
                for j in range(n):
                    if j in in_tree:
                        continue
                    if best is None or dist[i][j] < best[0]:
                        best = (dist[i][j], i, j)
            _, i, j = best
            edges.add((min(i, j), max(i, j)))
            in_tree.add(j)
        # k-nearest for robustness
        for i in range(n):
            nbrs = sorted(range(n), key=lambda j: dist[i][j])
            for j in nbrs[1:k + 1]:
                edges.add((min(i, j), max(i, j)))
        return sorted(edges)

    def candidates_for_net(self, net, pins, cell_lists=None):
        """``pins`` = ``[set of (x,y,layer) pts, ...]`` -> ``[CandidatePath]``.

        ``cell_lists`` optionally supplies precomputed access cells per
        pin (from ``pin_access_cells``) so callers that also need the
        cells (e.g. foreign-pin reservation) don't recompute them.
        """
        cell_lists = cell_lists or [self.pin_access_cells(p) for p in pins]
        out = []
        for a, b in self._choose_pairs(pins):
            for (la, ta, sa) in cell_lists[a]:
                for (lb, tb, sb) in cell_lists[b]:
                    if (la, ta, sa) == (lb, tb, sb):
                        continue
                    pa = self.cell_center(la, ta, sa)
                    pb = self.cell_center(lb, tb, sb)
                    A = (pa[0], pa[1], la)
                    B = (pb[0], pb[1], lb)
                    out.extend(self.straight(net, a, b, A, B))
                    out.extend(self.l_routes(net, a, b, A, B))
        # dedupe identical cell sets, keep cheapest
        best = {}
        for p in out:
            if p.cells not in best or p.cost < best[p.cells].cost:
                best[p.cells] = p
        return list(best.values())


def build_region_candidates(net_pins, grid, guide_map=None):
    """``{net: [set-of-(x,y,layer) pins, ...]}`` -> ``{net: [CandidatePath]}``.

    While generating one net's candidates, the access cells of *other*
    nets' pins are temporarily reserved so no candidate may run through
    a foreign pin.  Candidates sharing a cell still produce a mutex in
    the solver, but the reservation prunes obviously-illegal paths.
    """
    gen = CandidateGenerator(grid)
    guide_map = guide_map or {}

    # access cells per pin (needed both for reservation and by callers)
    pin_cells = {}
    # *literal* cells each pin's shapes occupy (own layer only) -- these
    # are what we reserve when generating other nets' candidates
    lit_cells = {}
    for net, pins in net_pins.items():
        pin_cells[net] = [gen.pin_access_cells(p) for p in pins]
        lit_cells[net] = []
        for pin in pins:
            cells = set()
            for x, y, layer in pin:
                if layer in grid.metals:
                    m = grid.metals[layer]
                    ti = grid.track_index(
                        layer, x if m.direction == 'V' else y)
                    si = grid.seg_index(
                        layer, y if m.direction == 'V' else x)
                    cells.add((layer, ti, si))
            lit_cells[net].append(cells)

    result = {}
    for net, pins in net_pins.items():
        foreign = []
        for other, plists in lit_cells.items():
            if other == net:
                continue
            for plist in plists:
                foreign.extend(plist)
        removed = grid.reserve_cells(foreign)
        try:
            cands = gen.candidates_for_net(net, pins,
                                           cell_lists=pin_cells[net])
        finally:
            grid.unreserve_cells(removed)
        guides = guide_map.get(net)
        if guides:
            boosted = []
            for c in cands:
                if any(_cells_in_rect(grid, c.cells, rect)
                       for _layer, rect in guides):
                    c = CandidatePath(c.net, c.pin_a, c.pin_b, c.pieces,
                                      c.cost * 0.8, c.cells)
                boosted.append(c)
            cands = boosted
        result[net] = cands
        logger.debug(f'net {net}: {len(cands)} candidate paths, '
                     f'{len(pins)} pins')
    return result


def _cells_in_rect(grid, cells, rect):
    x0, y0, x1, y1 = rect
    for layer, ti, si in cells:
        m = grid.metals[layer]
        if m.direction == 'V':
            x = grid.track_coord(layer, ti)
            y = grid.seg_coord(layer, si)
        else:
            y = grid.track_coord(layer, ti)
            x = grid.seg_coord(layer, si)
        if not (x0 <= x <= x1 and y0 <= y <= y1):
            return False
    return True
