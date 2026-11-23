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

    def pin_access_cells(self, pin_pts, max_cells=4, pin_rects=None):
        """Return up to ``max_cells`` ``(layer, ti, si)`` access cells
        for a pin whose ``pin_pts`` is a set of ``(x, y, layer)``
        access points.

        Cells on the pin's *own* layers are strongly preferred: a path
        endpoint there physically terminates on pin metal (the via
        stack the path builder inserts reaches down to the pin layer).

        ``pin_rects`` optionally supplies ``{layer: [rect,...]}`` -- the
        pin's real shapes.  When given, an access cell is only kept if
        a wire strip on its track (``track_coord +- width/2``)
        geometrically overlaps at least one pin rect on that layer.
        This rejects cells whose track quantisation landed just beside
        the pin (which the point-sampled test could not distinguish).
        """
        names = [m.name for m in self.grid.metal_list]
        legal = {m.name for m in self.layers}
        out = []
        seen = set()
        pts = sorted(pin_pts)
        pin_rects = pin_rects or {}

        def _touches(name, ti, si):
            """Does the cell's centre point land on (or within one
            wire half-width of) one of the pin's real rects on
            ``name``?  Checks both grid axes -- a wire/via ending at
            the cell centre must geometrically meet pin metal, not
            merely share a track."""
            if name not in pin_rects:
                # no rect info: accept if the point is on this layer
                return any(p[2] == name for p in pts)
            m = self.grid.metals[name]
            hw = m.width // 2
            cx, cy = self.cell_center(name, ti, si)
            return any(r[0] - hw <= cx <= r[2] + hw
                       and r[1] - hw <= cy <= r[3] + hw
                       for r in pin_rects[name])

        for deltas in ((0,), (1, -1), (2, -2)):
            for d in deltas:
                for x, y, layer in pts:
                    if layer not in names:
                        continue
                    li = names.index(layer) + d
                    if not (0 <= li < len(names)):
                        continue
                    name = names[li]
                    if name not in legal:
                        continue
                    # a cell on a non-pin layer has no shape to touch;
                    # accept it only in the relaxed rounds
                    if d != 0 and name not in pin_rects:
                        continue
                    m = self.grid.metals[name]
                    ti = self.grid.track_index(
                        name, x if m.direction == 'V' else y)
                    si = self.grid.seg_index(
                        name, y if m.direction == 'V' else x)
                    # try the quantised cell and its along-track
                    # neighbours: seg quantisation can shift the cell
                    # centre half a pitch off the pin rect
                    for si2 in (si - 1, si, si + 1):
                        if si2 < 0 or not _touches(name, ti, si2):
                            continue
                        key = (name, ti, si2)
                        if key not in seen and \
                                (ti, si2) in self.grid.allowed[name]:
                            seen.add(key)
                            out.append(key)
            if out:
                break
        # final fallback: if nothing touched (pin off-grid), accept any
        # free own-layer cell so the net still routes
        if not out:
            for x, y, layer in pts:
                if layer not in legal:
                    continue
                m = self.grid.metals.get(layer)
                if m is None:
                    continue
                ti = self.grid.track_index(
                    layer, x if m.direction == 'V' else y)
                si = self.grid.seg_index(
                    layer, y if m.direction == 'V' else x)
                key = (layer, ti, si)
                if key not in seen and \
                        (ti, si) in self.grid.allowed[layer]:
                    seen.add(key)
                    out.append(key)
        return out[:max_cells]

    def _vias_between(self, lx, ly, x, y):
        """Via pieces connecting two different layers, or [].

        Returns one ``PathPiece`` per intermediate via layer (so an
        M1->M3 stack yields V1 + V2 pieces); each via piece records the
        *via layer name* (``Vn``) so the emitted terminal matches the
        C++ router's schema.
        """
        if lx == ly:
            return []
        cells = self._via_stack(lx, ly, x, y)
        if cells is None:
            return None
        names = [m.name for m in self.grid.metal_list]
        i0, i1 = sorted((names.index(lx), names.index(ly)))
        pieces = []
        w = min(m.width for m in self.grid.metal_list) // 2
        vbox = (x - w, y - w, x + w, y + w)
        # one via piece per bridged adjacent-metal pair; ``bbox`` holds
        # the true junction rect (grid cell centres of the two bridged
        # layers can quantise half a pitch apart, so do NOT re-derive
        # the via position from cells at emission time)
        for k in range(i0, i1):
            lo_name, hi_name = names[k], names[k + 1]
            via_name = self.grid.via_name(lo_name, hi_name)
            vcells = frozenset(
                c for c in cells if c[0] in (lo_name, hi_name))
            pieces.append(PathPiece('via', via_name, vcells, vbox))
        return pieces

    def _junction_on_pin(self, jx, jy, rects, layer=None, hw=40):
        """Does via junction ``(jx, jy)`` land on one of the pin's real
        metal rects?  ``rects`` is ``{layer: [rect]}`` or ``None`` (no
        info -> accept).  ``layer`` restricts the test to the pin's
        shapes on that layer -- required when the endpoint via bottoms
        out on a specific pin layer, since touching an upper-layer
        projection of the pin does not connect it.  ``hw`` is a
        generous tolerance covering the via's half-size."""
        if not rects:
            return True
        for lname, rs in rects.items():
            if layer is not None and lname != layer:
                continue
            for r in rs:
                if r[0] - hw <= jx <= r[2] + hw and \
                        r[1] - hw <= jy <= r[3] + hw:
                    return True
        return False

    def straight(self, net, a, b, A, B, rect_a=None, rect_b=None):
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
                # land the via on the wire's own track, not the raw
                # endpoint coordinate (half-pitch quantisation gap)
                wx = self.grid.track_coord(
                    lay, next(iter({c[1] for c in cells})))
                if not (self._junction_on_pin(wx, ay, rect_a, la)
                        and self._junction_on_pin(wx, by, rect_b, lb)):
                    continue
                via_a = self._vias_between(la, lay, wx, ay)
                via_b = self._vias_between(lb, lay, wx, by)
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
                wy = self.grid.track_coord(
                    lay, next(iter({c[1] for c in cells})))
                if not (self._junction_on_pin(ax, wy, rect_a, la)
                        and self._junction_on_pin(bx, wy, rect_b, lb)):
                    continue
                via_a = self._vias_between(la, lay, ax, wy)
                via_b = self._vias_between(lb, lay, bx, wy)
                if via_a is None or via_b is None:
                    continue
                piece = PathPiece('wire', lay, cells,
                                  self._wire_bbox(lay, cells))
                out.append(self._mkpath(net, a, b,
                                        via_a + [piece] + via_b))
        return out

    def l_routes(self, net, a, b, A, B, alt_layers=True,
                 rect_a=None, rect_b=None):
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
        def _v_cells(lv, ti, y0, y1):
            """Vertical wire cells on explicit track ``ti``."""
            sa = self.cell_at(lv, 0, y0)[1]
            sb = self.cell_at(lv, 0, y1)[1]
            lo, hi = sorted((sa, sb))
            return frozenset((lv, ti, s) for s in range(lo, hi + 1))

        def _h_cells(lh, ti, x0, x1):
            sa = self.cell_at(lh, x0, 0)[1]
            sb = self.cell_at(lh, x1, 0)[1]
            lo, hi = sorted((sa, sb))
            return frozenset((lh, ti, s) for s in range(lo, hi + 1))

        def _variants(layer, coord):
            """Track indices near ``coord`` -- the nearest track plus
            its neighbours, so a leg can slide onto a track that
            actually reaches the pin/junction geometry."""
            t0 = self.grid.track_index(layer, coord)
            return {t0 - 1, t0, t0 + 1} - {-1}

        for mode in ('v_first', 'h_first'):
            for lv in v_opts[:2]:
                for lh in h_opts[:2]:
                    if lv == lh:
                        continue
                    if mode == 'v_first':
                        ti_vs = _variants(lv, ax)
                        ti_hs = _variants(lh, by)
                    else:
                        ti_hs = _variants(lh, ay)
                        ti_vs = _variants(lv, bx)
                    for tiv in ti_vs:
                        wx = self.grid.track_coord(lv, tiv)
                        for tih in ti_hs:
                            wy = self.grid.track_coord(lh, tih)
                            # legs run junction -> endpoint on their own
                            # tracks so the L physically connects
                            if mode == 'v_first':
                                c_v = _v_cells(lv, tiv, ay, wy)
                                c_h = _h_cells(lh, tih, wx, bx)
                                jax, jay = wx, ay    # via A: pin -> V
                                jbx, jby = bx, wy    # via B: pin -> H
                            else:
                                c_h = _h_cells(lh, tih, ax, wx)
                                c_v = _v_cells(lv, tiv, wy, by)
                                jax, jay = ax, wy    # via A: pin -> H
                                jbx, jby = wx, by    # via B: pin -> V
                            if not (self._free(c_v)
                                    and self._free(c_h)):
                                continue
                            # endpoint vias must physically land on pin
                            # metal on the pin's own layer, not just a
                            # quantised neighbour
                            if not (self._junction_on_pin(
                                    jax, jay, rect_a, la)
                                    and self._junction_on_pin(
                                        jbx, jby, rect_b, lb)):
                                continue
                            va = self._vias_between(
                                la, lv if mode == 'v_first' else lh,
                                jax, jay)
                            vb = self._vias_between(
                                lb, lv if mode == 'h_first' else lh,
                                jbx, jby)
                            vj = self._vias_between(lv, lh, wx, wy)
                            if va is None or vb is None or vj is None:
                                continue
                            pw_v = PathPiece(
                                'wire', lv, c_v,
                                self._wire_bbox(lv, c_v))
                            pw_h = PathPiece(
                                'wire', lh, c_h,
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

    def candidates_for_net(self, net, pins, cell_lists=None,
                           rect_lists=None):
        """``pins`` = ``[set of (x,y,layer) pts, ...]`` -> ``[CandidatePath]``.

        ``cell_lists`` optionally supplies precomputed access cells per
        pin (from ``pin_access_cells``) so callers that also need the
        cells (e.g. foreign-pin reservation) don't recompute them.
        ``rect_lists`` is the matching ``net_pin_rects`` entry used by
        ``pin_access_cells`` for precise overlap tests.
        """
        if cell_lists is None:
            rect_lists = rect_lists or [{}] * len(pins)
            cell_lists = [self.pin_access_cells(p, pin_rects=r)
                          for p, r in zip(pins, rect_lists)]
        rect_lists = rect_lists or [None] * len(pins)
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
                    out.extend(self.straight(
                        net, a, b, A, B,
                        rect_a=rect_lists[a], rect_b=rect_lists[b]))
                    out.extend(self.l_routes(
                        net, a, b, A, B,
                        rect_a=rect_lists[a], rect_b=rect_lists[b]))
        # dedupe identical cell sets, keep cheapest
        best = {}
        for p in out:
            if p.cells not in best or p.cost < best[p.cells].cost:
                best[p.cells] = p
        return list(best.values())


def build_region_candidates(net_pins, grid, guide_map=None,
                            boundary_pins=None, pin_rects=None):
    """``{net: [set-of-(x,y,layer) pins, ...]}`` -> ``{net: [CandidatePath]}``.

    While generating one net's candidates, the access cells of *other*
    nets' pins are temporarily reserved so no candidate may run through
    a foreign pin.  Candidates sharing a cell still produce a mutex in
    the solver, but the reservation prunes obviously-illegal paths.

    ``boundary_pins`` optionally maps ``net -> pin_index`` for the
    synthetic die-edge escape pin the driver appends.  Boundary pins
    are routing targets, not fixed metal shapes: they are never
    reserved as foreign blockages (adjacent nets' boundary pins
    quantise to neighbouring tracks and would lock each other out).

    ``pin_rects`` optionally supplies ``{net: [{layer: [rect]}]}`` --
    the real pin shapes, forwarded to ``pin_access_cells`` so access
    cells are only kept when their track overlaps actual pin metal.
    """
    gen = CandidateGenerator(grid)
    guide_map = guide_map or {}
    boundary_pins = boundary_pins or {}
    pin_rects = pin_rects or {}

    # access cells per pin (needed both for reservation and by callers)
    pin_cells = {}
    # *literal* cells each pin's shapes occupy (own layer only) -- these
    # are what we reserve when generating other nets' candidates
    lit_cells = {}
    for net, pins in net_pins.items():
        rect_lists = pin_rects.get(net) or [{}] * len(pins)
        pin_cells[net] = [gen.pin_access_cells(p, pin_rects=r)
                          for p, r in zip(pins, rect_lists)]
        lit_cells[net] = []
        for pi, pin in enumerate(pins):
            if pi == boundary_pins.get(net):
                continue
            cells = set()
            rects = (pin_rects.get(net) or [None] * len(pins))[pi]
            if rects:
                # reserve every grid cell the pin's real shapes cover,
                # not just the sampled access points -- a long pin bar
                # must block the whole track span
                for layer, rs in rects.items():
                    if layer not in grid.metals:
                        continue
                    for r in rs:
                        cells.update(_cells_covered_by_rect(grid, layer, r))
            else:
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
            cands = gen.candidates_for_net(
                net, pins, cell_lists=pin_cells[net],
                rect_lists=pin_rects.get(net))
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


def _cells_covered_by_rect(grid, layer, rect):
    """Grid cells ``(layer, ti, si)`` whose wire footprint overlaps
    ``rect`` -- the same coverage ``RoutingGrid.block_region`` uses,
    returned as a set so the cells can be *reserved* (and restored)
    rather than permanently blocked."""
    m = grid.metals[layer]
    r_x0, r_y0, r_x1, r_y1 = rect
    hw = m.width // 2
    if m.direction == 'V':
        ti0 = grid.track_index(layer, r_x0 - hw)
        ti1 = grid.track_index(layer, r_x1 + hw)
        si0 = grid.seg_index(layer, r_y0)
        si1 = grid.seg_index(layer, r_y1)
    else:
        ti0 = grid.track_index(layer, r_y0 - hw)
        ti1 = grid.track_index(layer, r_y1 + hw)
        si0 = grid.seg_index(layer, r_x0)
        si1 = grid.seg_index(layer, r_x1)
    return {(layer, ti, si)
            for ti in range(ti0, ti1 + 1)
            for si in range(si0, si1 + 1)}


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
