"""Track-aligned routing grid for the detailed router.

Unlike the coarse tile grid in ``wire_planner`` (used for congestion
planning), this grid models *routing tracks*: on each metal layer wires
may only occupy discrete track lines spaced by the PDK ``Pitch``, at the
PDK ``Width``, running along the layer's preferred ``Direction``.  A
route is a connected set of wire segments on tracks plus vias between
adjacent metal layers.

The grid is *sparse*: only grid points that are not fully blocked are
materialized.  Nodes are ``(layer, ix, iy)`` where ``ix`` indexes the
layer's track positions on the cross-axis and ``iy`` the position along
the preferred direction (so a segment is a run of consecutive ``iy`` at
fixed ``ix`` -- matching how routers actually lay down track-aligned
metal).

Units: all coordinates are in the scaled-placement domain (the same
integer units the ``*.scaled_placement_verilog.json`` and ``3_pnr``
collateral use; 1 unit = PDK grid unit, typically 0.5 nm * ScaleFactor).
"""

import logging
from math import gcd

logger = logging.getLogger(__name__)


class LayerInfo:
    """Per-metal-layer routing parameters extracted from layers.json."""

    def __init__(self, name, direction, pitch, width, end_to_end,
                 gds_layer=None):
        self.name = name
        self.direction = direction        # 'H' or 'V' (preferred routing)
        self.pitch = pitch                # center-to-center track spacing
        self.width = width                # default wire width
        self.end_to_end = end_to_end or 0  # min end-to-end spacing
        self.gds_layer = gds_layer

    @property
    def min_space(self):
        """Lateral spacing between parallel tracks = pitch - width."""
        return max(self.pitch - self.width, 0)

    def __repr__(self):
        return f'LayerInfo({self.name} {self.direction} p={self.pitch} w={self.width})'


class ViaInfo:
    """Inter-layer via connecting two adjacent metal layers."""

    def __init__(self, name, lower, upper, space_x, space_y):
        self.name = name
        self.lower = lower
        self.upper = upper
        self.space_x = space_x
        self.space_y = space_y

    def __repr__(self):
        return f'ViaInfo({self.name} {self.lower}->{self.upper})'


def load_pdk_layers(pdk_json):
    """Parse metal/via routing layers from a layers.json Abstraction list.

    ``pdk_json`` is the parsed ``layers.json`` dict.  Returns
    ``(metal_infos, via_infos)`` where ``metal_infos`` is a list of
    ``LayerInfo`` for M1..Mn (routing-capable metal layers in order) and
    ``via_infos`` maps via name -> ``ViaInfo`` keyed by the lower metal.
    """
    metals = {}
    vias = {}
    for e in pdk_json['Abstraction']:
        name = e['Layer']
        if name.startswith('M') and 'Pitch' in e:
            idx = int(name[1:])
            metals[idx] = LayerInfo(
                name, e.get('Direction', 'H'), e['Pitch'],
                e.get('Width', e['Pitch'] // 3),
                e.get('EndToEnd'), gds_layer=e.get('GdsLayerNo'))
        elif name.startswith('V') and 'Stack' in e:
            lo, hi = e['Stack']
            if lo and hi:
                vias[(lo, hi)] = ViaInfo(
                    name, lo, hi,
                    e.get('SpaceX', 0), e.get('SpaceY', 0))
    metal_infos = [metals[k] for k in sorted(metals)]
    return metal_infos, vias


class RoutingGrid:
    """Sparse track-aligned routing grid over the die bbox.

    **Unified index frame.**  Every cell is identified by
    ``(layer_name, gx, gy)`` where ``(gx, gy)`` are *global lattice*
    indices with the same physical meaning on every layer::

        gx = x // g,   gy = y // g

    with ``g = gcd(all metal pitches)``.  So ``(gx, gy)`` always means
    the physical point ``(gx * g, gy * g)`` -- a cell key on M1 and the
    same key on M2 denote the *same* physical location (the earlier
    per-layer ``(ti, si)`` frame did not: ``ti``/``si`` were per-layer
    track/segment counts with mixed rounding, so the same tuple meant
    different physical points on different layers).

    The lattice is *not* the routing pitch: tracks still sit at each
    layer's own ``Pitch`` (no density loss).  Cell keys are simply the
    lower-left ``(track, segment-start)`` corner of the pitch cell
    expressed in common ``g`` units.

    ``allowed`` is a per-layer ``set`` of ``(gx, gy)`` cells that are
    free for routing (not inside an obstacle).  Occupancy is the
    solver's job, not the grid's.
    """

    def __init__(self, bbox, metal_infos, via_infos):
        self.x0, self.y0, self.x1, self.y1 = bbox
        self.metals = {m.name: m for m in metal_infos}
        self.metal_list = list(metal_infos)      # ordered M1..Mn
        self.vias = via_infos                    # {(lo,hi): ViaInfo}
        # gcd of all metal pitches: the common lattice step.  For
        # FinFET14nm_Mock (80/84) this is 4.
        g = 0
        for m in metal_infos:
            g = gcd(g, m.pitch)
        self.g = g or 1
        assert self.g >= 1
        # Track/segment coords are ``origin + index * pitch``; the
        # lattice mapping is exact only when the bbox origin itself
        # lands on the lattice.  Fall back to a consistent floor-divide
        # (still monotone and layer-independent) if it does not.
        self._origin_aligned = (self.x0 % self.g == 0
                                and self.y0 % self.g == 0)
        if not self._origin_aligned:
            logger.warning(
                'bbox origin (%s, %s) is not a multiple of lattice g=%s; '
                'cell keys use floor division and physical anchors may '
                'be off by <1 lattice step', self.x0, self.y0, self.g)
        # allowed[layer] = set of (gx, gy) free cells
        self.allowed = {m.name: set() for m in metal_infos}
        self._init_free_cells()

    def via_name(self, lower, upper):
        """Name of the via layer bridging two adjacent metal layers."""
        v = self.vias.get((lower, upper))
        if v is not None:
            return v.name
        # fall back to V<lower-index> convention when the PDK does not
        # name the via pair explicitly
        try:
            return f'V{int(lower[1:])}'
        except (ValueError, IndexError):
            return f'V{lower}_{upper}'

    # -- coordinate <-> index --------------------------------------------
    def track_index(self, layer, coord):
        """Index of the track nearest ``coord`` on the layer's cross axis."""
        m = self.metals[layer]
        origin = self.x0 if m.direction == 'V' else self.y0
        return max(0, int(round((coord - origin) / m.pitch)))

    def track_coord(self, layer, ti):
        m = self.metals[layer]
        origin = self.x0 if m.direction == 'V' else self.y0
        return origin + ti * m.pitch

    def seg_index(self, layer, coord):
        """Index of the cell along the preferred direction."""
        m = self.metals[layer]
        origin = self.y0 if m.direction == 'V' else self.x0
        return max(0, int((coord - origin) // m.pitch))

    def seg_coord(self, layer, si):
        m = self.metals[layer]
        origin = self.y0 if m.direction == 'V' else self.x0
        return origin + si * m.pitch

    # -- unified lattice cell keys ---------------------------------------
    def lattice_coord(self, k):
        """Physical coordinate of lattice index ``k`` (i.e. ``k * g``)."""
        return k * self.g

    def track_key(self, layer, coord):
        """Lattice index of the track nearest ``coord`` on the cross axis.

        Rounds to the nearest track (as the old ``track_index`` did), then
        expresses the track position in global lattice units.
        """
        return self.track_coord(layer, self.track_index(layer, coord)) \
            // self.g

    def seg_key(self, layer, coord):
        """Lattice index of the segment *containing* ``coord``.

        Floors to the segment start along the preferred direction, then
        expresses it in global lattice units.  Together with
        :meth:`track_key` this is the only quantisation used to map a
        physical point to a cell.
        """
        return self.seg_coord(layer, self.seg_index(layer, coord)) \
            // self.g

    def cell_key(self, layer, ti, si):
        """Global lattice key ``(gx, gy)`` of cell ``(ti, si)``.

        The axis order is always ``(x, y)`` -- *not* ``(track, seg)`` --
        so the key has the same physical meaning on every layer.
        """
        m = self.metals[layer]
        if m.direction == 'V':
            return (self.track_coord(layer, ti) // self.g,
                    self.seg_coord(layer, si) // self.g)
        return (self.seg_coord(layer, si) // self.g,
                self.track_coord(layer, ti) // self.g)

    def cell_key_at(self, layer, x, y):
        """Canonical map: physical point -> containing cell key.

        This is the *single* containment lookup; nothing else should mix
        :meth:`track_index`/:meth:`seg_index` quantisation.  A V-layer
        cell rounds ``x`` to the nearest track and floors ``y`` to the
        containing segment; an H-layer cell is symmetric.
        """
        m = self.metals[layer]
        if m.direction == 'V':
            return (self.track_key(layer, x), self.seg_key(layer, y))
        return (self.seg_key(layer, x), self.track_key(layer, y))

    def cell_anchor(self, layer, key):
        """Physical ``(x, y)`` of a cell key's ``(track, seg-start)``.

        Since a key is a global lattice coordinate pair, the answer is
        layer-independent: ``(gx * g, gy * g)``.
        """
        gx, gy = key
        return (gx * self.g, gy * self.g)

    def n_tracks(self, layer):
        m = self.metals[layer]
        span = (self.x1 - self.x0) if m.direction == 'V' else (self.y1 - self.y0)
        return int(span // m.pitch) + 1

    def n_segs(self, layer):
        m = self.metals[layer]
        span = (self.y1 - self.y0) if m.direction == 'V' else (self.x1 - self.x0)
        return int(span // m.pitch) + 1

    # -- free cells ------------------------------------------------------
    def _init_free_cells(self):
        for m in self.metal_list:
            self.allowed[m.name] = {
                self.cell_key(m.name, ti, si)
                for ti in range(self.n_tracks(m.name))
                for si in range(self.n_segs(m.name))
            }

    def reserve_cells(self, cells):
        """Temporarily remove ``cells`` ``(layer, gx, gy)`` from
        ``allowed`` and return them so callers can restore with
        ``unreserve_cells``.  Used to hide other nets' pin cells while
        generating candidates for one net."""
        removed = []
        for layer, gx, gy in cells:
            if (gx, gy) in self.allowed[layer]:
                self.allowed[layer].discard((gx, gy))
                removed.append((layer, gx, gy))
        return removed

    def unreserve_cells(self, cells):
        for layer, gx, gy in cells:
            self.allowed[layer].add((gx, gy))

    def block_region(self, layer, rect):
        """Mark cells whose track rectangle intersects ``rect`` as blocked.

        A cell on ``layer`` spans ``width`` across the track and ``pitch``
        along it; we block cells whose expanded footprint overlaps rect.
        """
        m = self.metals[layer]
        if m is None:
            return
        r_x0, r_y0, r_x1, r_y1 = rect
        hw = m.width // 2
        if m.direction == 'V':
            ti0 = self.track_index(layer, r_x0 - hw - m.pitch)
            ti1 = self.track_index(layer, r_x1 + hw + m.pitch)
            si0 = self.seg_index(layer, r_y0 - m.pitch)
            si1 = self.seg_index(layer, r_y1 + m.pitch) + 1
        else:
            ti0 = self.track_index(layer, r_y0 - hw - m.pitch)
            ti1 = self.track_index(layer, r_y1 + hw + m.pitch)
            si0 = self.seg_index(layer, r_x0 - m.pitch)
            si1 = self.seg_index(layer, r_x1 + m.pitch) + 1
        cells = self.allowed[layer]
        for ti in range(ti0, ti1 + 1):
            for si in range(si0, si1 + 1):
                cells.discard(self.cell_key(layer, ti, si))

    # -- adjacency -------------------------------------------------------
    def neighbors(self, node):
        """Yield adjacent grid nodes.

        ``node`` is ``(layer, gx, gy)``.  Along-track neighbours move by
        ``pitch // g`` lattice steps along the preferred direction;
        cross-track neighbours move by the same step across it.  A via
        neighbour is the cell on an adjacent layer that *contains* the
        same physical point (resolved through :meth:`cell_key_at`, so the
        via is physically exact even though adjacent layers quantise the
        point to different cell keys).
        """
        layer, gx, gy = node
        m = self.metals[layer]
        step = m.pitch // self.g
        for dx, dy in ((-step, 0), (step, 0), (0, -step), (0, step)):
            nbx, nby = gx + dx, gy + dy
            if nbx < 0 or nby < 0:
                continue
            if (nbx, nby) in self.allowed[layer]:
                yield (layer, nbx, nby)
        names = [x.name for x in self.metal_list]
        li = names.index(layer)
        px, py = self.cell_anchor(layer, node)
        for dl in (-1, 1):
            nl = li + dl
            if 0 <= nl < len(names):
                other = self.metal_list[nl].name
                okey = self.cell_key_at(other, px, py)
                if okey in self.allowed[other]:
                    yield (other, *okey)
