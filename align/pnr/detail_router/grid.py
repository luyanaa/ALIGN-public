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
from collections import defaultdict

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

    Nodes are ``(layer_name, ti, si)``:
        ``ti`` -- track index on the *cross* axis (perpendicular to the
                  preferred direction),
        ``si`` -- segment index along the preferred direction, quantized
                  to the layer pitch.

    ``allowed`` is a per-layer ``defaultdict(set)`` of ``(ti, si)`` cells
    that are free for routing (not inside an obstacle).  Occupancy is the
    solver's job, not the grid's.
    """

    def __init__(self, bbox, metal_infos, via_infos):
        self.x0, self.y0, self.x1, self.y1 = bbox
        self.metals = {m.name: m for m in metal_infos}
        self.metal_list = list(metal_infos)      # ordered M1..Mn
        self.vias = via_infos                    # {(lo,hi): ViaInfo}
        # allowed[layer] = set of (ti, si) free cells
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
                (ti, si)
                for ti in range(self.n_tracks(m.name))
                for si in range(self.n_segs(m.name))
            }

    def reserve_cells(self, cells):
        """Temporarily remove ``cells`` ``(layer, ti, si)`` from
        ``allowed`` and return them so callers can restore with
        ``unreserve_cells``.  Used to hide other nets' pin cells while
        generating candidates for one net."""
        removed = []
        for layer, ti, si in cells:
            if (ti, si) in self.allowed[layer]:
                self.allowed[layer].discard((ti, si))
                removed.append((layer, ti, si))
        return removed

    def unreserve_cells(self, cells):
        for layer, ti, si in cells:
            self.allowed[layer].add((ti, si))

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
                cells.discard((ti, si))

    # -- adjacency -------------------------------------------------------
    def neighbors(self, node):
        """Yield adjacent grid nodes (along-track ±1 seg, or via up/down)."""
        layer, ti, si = node
        m = self.metals[layer]
        for d in (-1, 1):
            nb = (layer, ti, si + d)
            if nb[2] >= 0 and nb in self.allowed[layer]:
                yield nb
            nb = (layer, ti + d, si)
            if nb[1] >= 0 and nb in self.allowed[layer]:
                yield nb
        li = [x.name for x in self.metal_list].index(layer)
        for dl in (-1, 1):
            nl = li + dl
            if 0 <= nl < len(self.metal_list):
                other = self.metal_list[nl].name
                nb = (other, ti, si)
                if nb in self.allowed[other]:
                    yield nb
