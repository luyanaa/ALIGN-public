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


def _ceil_div(a, b):
    """Integer ceiling division (``b > 0``), exact for negatives too."""
    return -((-a) // b)


def _rect_overlap_area(a, b):
    """Area of the intersection of two ``(x0, y0, x1, y1)`` rects."""
    dx = min(a[2], b[2]) - max(a[0], b[0])
    dy = min(a[3], b[3]) - max(a[1], b[1])
    return dx * dy if dx > 0 and dy > 0 else 0


@dataclass(frozen=True)
class PathPiece:
    """One wire run or via inside a candidate path."""
    kind: str                     # 'wire' or 'via'
    layer: str                    # metal layer (for a via: lower layer)
    cells: frozenset              # (layer, gx, gy) lattice cells occupied
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
    # Cells are keyed ``(layer, gx, gy)`` in the *unified global lattice*
    # frame: ``(gx, gy)`` always means physical ``(gx*g, gy*g)`` on every
    # layer (see ``RoutingGrid``).  No ``(track, seg)`` order appears
    # below -- the axis order is always ``(x, y)``.
    def cell_center(self, layer, gx, gy):
        """Physical anchor ``(x, y)`` of lattice cell ``(gx, gy)``.

        The lattice key is layer-independent, so the answer is the same
        on every layer (``cell_anchor``)."""
        return self.grid.cell_anchor(layer, (gx, gy))

    def cell_at(self, layer, x, y):
        """Containing cell key ``(gx, gy)`` for physical ``(x, y)``."""
        return self.grid.cell_key_at(layer, x, y)

    def _seg_step(self, layer):
        """Lattice step between consecutive along-track cells."""
        return self.grid.metals[layer].pitch // self.grid.g

    def _wire_cells_v(self, layer, x, y0, y1):
        """Cells of a vertical wire at ``x`` covering ``y0``..``y1``."""
        gx, ga = self.cell_at(layer, x, y0)
        _, gb = self.cell_at(layer, x, y1)
        # ``gx``/``ga``..``gb`` are global-lattice keys; the along-track
        # cells are one layer pitch (``pitch // g`` lattice steps) apart.
        step = self._seg_step(layer)
        lo, hi = sorted((ga, gb))
        return frozenset((layer, gx, s)
                         for s in range(lo, hi + 1, step))

    def _wire_cells_h(self, layer, y, x0, x1):
        gx0, gy = self.cell_at(layer, x0, y)
        gx1, _ = self.cell_at(layer, x1, y)
        step = self._seg_step(layer)
        lo, hi = sorted((gx0, gx1))
        return frozenset((layer, s, gy)
                         for s in range(lo, hi + 1, step))

    def _free(self, cells):
        return all((c[1], c[2]) in self.grid.allowed[c[0]]
                   for c in cells)

    def _wire_bbox(self, layer, cells):
        m = self.grid.metals[layer]
        hw = m.width // 2
        xs = [gx for (_l, gx, gy) in cells]
        ys = [gy for (_l, gx, gy) in cells]
        # ``cell_anchor`` is the canonical lattice-key -> physical
        # ``(x, y)`` map (exact, not an approximation); no ``track``/
        # ``seg`` reordering is needed because keys are already
        # ``(x, y)``.
        x0, y0 = self.grid.cell_anchor(layer, (min(xs), min(ys)))
        x1, y1 = self.grid.cell_anchor(layer, (max(xs), max(ys)))
        if m.direction == 'V':
            # along the preferred axis (y) a cell is one pitch long;
            # across the track (x) it is ``width`` wide.
            return (x0 - hw, y0, x1 + hw, y1 + m.pitch)
        return (x0, y0 - hw, x1 + m.pitch, y1 + hw)

    def _via_stack(self, layer_lo, layer_hi, x, y):
        """Cells of a via chain between ``layer_lo`` and ``layer_hi``.

        Returns the frozenset of occupied cells on every intermediate
        layer (inclusive), or ``None`` if any layer lacks a free cell.
        Each layer resolves the *same physical* ``(x, y)`` to its own
        containing cell via ``cell_key_at`` -- the via point itself is
        kept exact in the piece bbox.
        """
        names = [m.name for m in self.grid.metal_list]
        i0, i1 = sorted((names.index(layer_lo), names.index(layer_hi)))
        cells = set()
        for name in names[i0:i1 + 1]:
            gx, gy = self.grid.cell_key_at(name, x, y)
            if (gx, gy) not in self.grid.allowed[name]:
                return None
            cells.add((name, gx, gy))
        return frozenset(cells)

    # -- path constructors --------------------------------------------------
    def _mkpath(self, net, a, b, pieces):
        cells = frozenset().union(*(p.cells for p in pieces))
        n_vias = sum(1 for p in pieces if p.kind == 'via')
        n_wire = sum(len(p.cells) for p in pieces if p.kind == 'wire')
        return CandidatePath(net, a, b, tuple(pieces),
                             float(n_wire + 5 * n_vias), cells)

    # -- access-point geometry ---------------------------------------------
    def _cell_footprint(self, layer, key):
        """Pitch-cell footprint of a lattice key (physical rect).

        Across the track the cell spans one pitch (nearest-track
        rounding); along the preferred direction it spans one pitch from
        the segment start.  This is the same footprint the grid uses to
        decide covered cells, so an access cell "touches" a pin rect
        exactly when these rects intersect.
        """
        m = self.grid.metals[layer]
        p = m.pitch
        x, y = self.grid.cell_anchor(layer, key)
        if m.direction == 'V':
            return (x - p // 2, y, x + p // 2, y + p)
        return (x, y - p // 2, x + p, y + p // 2)

    def _cells_inside_rect(self, layer, rect):
        """Lattice keys whose *anchor point* lies inside the pin box on
        ``layer``.

        For a V-layer rect this is the set of tracks with
        ``track_coord in [x0, x1]`` crossed with the segment cells whose
        anchor ``seg_coord`` is inside ``[y0, y1]``.  H layers are
        symmetric.

        Anchoring inside the box (rather than merely overlapping the
        box with the cell's pitch extent) is what makes the emitted
        attachment geometric: candidate endpoints/vias are placed at the
        cell anchor, so an anchor just outside the pin would produce a
        via/wire that never touches pin metal even though the cell's
        span clips the box.
        """
        m = self.grid.metals[layer]
        p = m.pitch
        x0, y0, x1, y1 = rect
        ox, oy = self.grid.x0, self.grid.y0
        if m.direction == 'V':
            # cross axis = x (tracks), preferred axis = y (segments)
            ti0 = _ceil_div(x0 - ox, p)
            ti1 = (x1 - ox) // p
            si0 = _ceil_div(y0 - oy, p)
            si1 = (y1 - oy) // p
        else:
            # cross axis = y (tracks), preferred axis = x (segments)
            ti0 = _ceil_div(y0 - oy, p)
            ti1 = (y1 - oy) // p
            si0 = _ceil_div(x0 - ox, p)
            si1 = (x1 - ox) // p
        if ti0 > ti1 or si0 > si1:
            return []
        nt, ns = self.grid.n_tracks(layer), self.grid.n_segs(layer)
        out = []
        for ti in range(max(0, ti0), min(nt - 1, ti1) + 1):
            for si in range(max(0, si0), min(ns - 1, si1) + 1):
                out.append(self.grid.cell_key(layer, ti, si))
        return out

    def _containing_cells(self, layer, rect):
        """Fallback when a pin is smaller than one pitch cell: the
        containing cells of the rect's two corners and centre.  Keeps a
        routable (but sub-cell-sized) pin from getting zero access."""
        x0, y0, x1, y1 = rect
        out = []
        seen = set()
        for x, y in ((x0, y0), (x1, y1),
                     ((x0 + x1) // 2, (y0 + y1) // 2)):
            key = self.grid.cell_key_at(layer, x, y)
            if key not in seen:
                seen.add(key)
                out.append(key)
        return out

    def _escape_rects(self, layer, key):
        """The 1-2 wire strips the first segment would occupy leaving
        an access cell along the layer's preferred direction."""
        m = self.grid.metals[layer]
        p, w = m.pitch, m.width
        x, y = self.grid.cell_anchor(layer, key)
        length = p + w
        if m.direction == 'V':
            return [(x - w // 2, y, x + w // 2, y + length),
                    (x - w // 2, y - length, x + w // 2, y)]
        return [(x, y - w // 2, x + length, y + w // 2),
                (x - length, y - w // 2, x, y + w // 2)]

    def _escape_overlap(self, layer, rect, other_pin_rects):
        """Overlap area of an escape rect against blocked grid cells
        (obstacles, which are already removed from ``grid.allowed``) and
        against foreign / other-pin rects."""
        m = self.grid.metals[layer]
        p = m.pitch
        x0, y0, x1, y1 = rect
        ox, oy = self.grid.x0, self.grid.y0
        allowed = self.grid.allowed[layer]
        area = 0
        if m.direction == 'V':
            ti0 = _ceil_div(x0 - p // 2 - ox, p)
            ti1 = (x1 + p // 2 - ox) // p
            si0 = (y0 - oy) // p
            si1 = (y1 - oy) // p
        else:
            ti0 = _ceil_div(y0 - p // 2 - oy, p)
            ti1 = (y1 + p // 2 - oy) // p
            si0 = (x0 - ox) // p
            si1 = (x1 - ox) // p
        nt, ns = self.grid.n_tracks(layer), self.grid.n_segs(layer)
        for ti in range(max(0, ti0), min(nt - 1, ti1) + 1):
            for si in range(max(0, si0), min(ns - 1, si1) + 1):
                if ti < 0 or si < 0:
                    continue
                key = self.grid.cell_key(layer, ti, si)
                if key not in allowed:
                    area += _rect_overlap_area(
                        rect, self._cell_footprint(layer, key))
        for ly, r in other_pin_rects:
            if ly == layer:
                area += _rect_overlap_area(rect, r)
        return area

    def _access_score(self, layer, key, other_pin_rects):
        """Score an access cell: the *least*-obstructed escape direction
        (anaroute keeps zero-overlap directions, else the minimum)."""
        return min(self._escape_overlap(layer, e, other_pin_rects)
                   for e in self._escape_rects(layer, key))

    def pin_access_cells(self, pin_pts, max_cells=4, pin_rects=None,
                         other_pin_rects=None):
        """Return up to ``max_cells`` ``(layer, gx, gy)`` access cells
        for a pin whose ``pin_pts`` is a set of ``(x, y, layer)``
        access points (``gx``/``gy`` are unified lattice indices).

        ``pin_rects`` optionally supplies ``{layer: [rect,...]}`` -- the
        pin's real shapes.  Access cells are the lattice tracks/segments
        whose *anchor* falls inside the pin box (see
        :meth:`_cells_inside_rect`), so the via/wire endpoint emitted at
        that anchor is guaranteed to overlap pin metal on its own layer.
        A pin smaller than one cell falls back to the cells containing
        its corners/centre -- never zero access for a routable pin.

        ``other_pin_rects`` optionally supplies additional
        ``(layer, rect)`` shapes (other nets' pins and the same net's
        other pins) used to score escape directions, so an access cell
        whose only way out is over foreign pin metal ranks last.
        """
        legal = {m.name for m in self.layers}
        pin_rects = pin_rects or {}
        other_pin_rects = list(other_pin_rects or ())
        cands = []                      # (score, key)
        seen = set()

        def _add(layer, key):
            if key in seen:
                return
            if key[1:] not in self.grid.allowed[layer]:
                return
            seen.add(key)
            cands.append(
                (self._access_score(layer, key[1:], other_pin_rects),
                 key))

        for layer, rects in pin_rects.items():
            if layer not in legal:
                continue
            for rect in rects:
                keys = self._cells_inside_rect(layer, rect)
                if not any(k in self.grid.allowed[layer] for k in keys):
                    # pin smaller than a cell, or landed between tracks
                    keys = self._containing_cells(layer, rect)
                for gx, gy in keys:
                    _add(layer, (layer, gx, gy))

        # no usable rects at all (callers without pin geometry): map the
        # supplied access points to their containing cells
        if not cands:
            for x, y, layer in sorted(pin_pts):
                if layer not in legal:
                    continue
                gx, gy = self.grid.cell_key_at(layer, x, y)
                _add(layer, (layer, gx, gy))

        cands.sort(key=lambda t: (t[0], t[1]))
        return [key for _score, key in cands[:max_cells]]

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
        # the exact junction rect.  ``cell_key_at`` resolves the same
        # physical point to each layer's containing cell, but those
        # cells' anchors can still sit at different lattice points, so
        # do NOT re-derive the via position from cells at emission time.
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
                wx = self.grid.lattice_coord(
                    next(iter({c[1] for c in cells})))
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
                wy = self.grid.lattice_coord(
                    next(iter({c[2] for c in cells})))
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
        def _v_cells(lv, tk, y0, y1):
            """Vertical wire cells on explicit track lattice key ``tk``."""
            _, ga = self.cell_at(lv, 0, y0)
            _, gb = self.cell_at(lv, 0, y1)
            step = self._seg_step(lv)
            lo, hi = sorted((ga, gb))
            return frozenset((lv, tk, s)
                             for s in range(lo, hi + 1, step))

        def _h_cells(lh, tk, x0, x1):
            ga, _ = self.cell_at(lh, x0, 0)
            gb, _ = self.cell_at(lh, x1, 0)
            step = self._seg_step(lh)
            lo, hi = sorted((ga, gb))
            return frozenset((lh, s, tk)
                             for s in range(lo, hi + 1, step))

        def _variants(layer, coord):
            """Track lattice keys near ``coord`` -- the nearest track
            plus its neighbours, so a leg can slide onto a track that
            actually reaches the pin/junction geometry."""
            step = self._seg_step(layer)
            t0 = self.grid.track_key(layer, coord)
            return {t0 - step, t0, t0 + step} - \
                {k for k in (t0 - step, t0, t0 + step) if k < 0}

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
                        wx = self.grid.lattice_coord(tiv)
                        for tih in ti_hs:
                            wy = self.grid.lattice_coord(tih)
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
            cell_lists = []
            for pi, (p, r) in enumerate(zip(pins, rect_lists)):
                others = [(ly, rc)
                          for pj, r2 in enumerate(rect_lists)
                          if pj != pi and r2
                          for ly, rs in r2.items() for rc in rs]
                cell_lists.append(self.pin_access_cells(
                    p, pin_rects=r, other_pin_rects=others))
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
    cells are the exact pin-box ∩ lattice set.  Other nets' pin rects
    (and the same net's other pins) are forwarded as escape-scoring
    shapes, so falsely-blocked attachment points rank last.
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
        # shapes of *other* nets: an escape direction running over
        # foreign pin metal is a bad first move, so score it out
        foreign = []
        for other, olists in pin_rects.items():
            if other == net:
                continue
            for rl in olists:
                for ly, rs in (rl or {}).items():
                    foreign.extend((ly, r) for r in rs)
        net_cells = []
        for pi, (p, r) in enumerate(zip(pins, rect_lists)):
            others = list(foreign)
            for pj, r2 in enumerate(rect_lists):
                if pj == pi or not r2:
                    continue
                for ly, rs in r2.items():
                    others.extend((ly, rc) for rc in rs)
            net_cells.append(gen.pin_access_cells(
                p, pin_rects=r, other_pin_rects=others))
        pin_cells[net] = net_cells
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
                        gx, gy = grid.cell_key_at(layer, x, y)
                        cells.add((layer, gx, gy))
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
    """Grid cells ``(layer, gx, gy)`` whose wire footprint overlaps
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
    cells = set()
    for ti in range(ti0, ti1 + 1):
        for si in range(si0, si1 + 1):
            gx, gy = grid.cell_key(layer, ti, si)
            cells.add((layer, gx, gy))
    return cells


def _cells_in_rect(grid, cells, rect):
    x0, y0, x1, y1 = rect
    for layer, gx, gy in cells:
        x, y = grid.cell_anchor(layer, (gx, gy))
        if not (x0 <= x <= x1 and y0 <= y <= y1):
            return False
    return True
