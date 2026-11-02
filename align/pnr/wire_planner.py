"""RALF-style tile-grid wire planner (negotiated congestion global router).

Re-implements the wire-planning stage of RALF's two-stage router
(Routing_v2): nets are routed on a coarse tile grid with a PathFinder-like
negotiation loop -- each iteration rips up and re-routes nets ordered by
degrading cost, while a history penalty accumulates on overused tiles.

Scope: this is a *planning/preview* engine.  The ALIGN C++ detail router
does not consume external route guides (GcellDetailRouter builds its grid
only from the in-memory GcellGlobalRouter object), so the output is:

* ``<TOP>_wireplan.json`` -- route guides in the WriteGcellGlobalRoute
  ``{"wires": [...]}`` schema, for inspection and for any future guide
  injection hook;
* ``<TOP>_wireplan_report.json`` -- per-tile demand/supply congestion map.

Enabled with ``--wire_planner`` on ``schematic2layout.py``; runs after
placement inside ``3_pnr:route`` for each variant.
"""

import heapq
import json
import logging
from collections import defaultdict
from itertools import count

logger = logging.getLogger(__name__)


class TileGrid:
    """Coarse multi-layer tile grid over the die.

    Per-node ``supply`` is the tile's routing capacity; demand per net
    visiting the tile is 1 unit (normalized -- RALF uses wire width + min
    spacing, which maps to our normalized unit once a PDK pitch is
    chosen).  Tiles inside a block bbox have supply 0 unless exempted
    (pin tiles and their carved escape channels).
    """

    DIRS = ('E', 'W', 'N', 'S')

    def __init__(self, bbox, gcell, layers, obstacles=(), via_supply=4):
        self.x0, self.y0, x1, y1 = bbox
        self.gcell = gcell
        self.nx = max(1, (x1 - self.x0 + gcell - 1) // gcell)
        self.ny = max(1, (y1 - self.y0 + gcell - 1) // gcell)
        self.layers = list(layers)
        self.via_supply = via_supply
        # supply[(l,i,j)] = node capacity; demand added by nets
        self.supply = defaultdict(lambda: via_supply)
        self._exempt = set()
        self._init_blockages(obstacles)

    def _init_blockages(self, obstacles):
        """Zero the supply of tiles covered by a block bbox.

        Tiles containing a registered pin are exempted so nets can reach
        their terminals (pins sit inside their own block's bbox).
        """
        self._exempt = set()
        for (llx, lly, urx, ury) in obstacles:
            i0 = max(0, (llx - self.x0) // self.gcell)
            i1 = min(self.nx - 1, (urx - 1 - self.x0) // self.gcell)
            j0 = max(0, (lly - self.y0) // self.gcell)
            j1 = min(self.ny - 1, (ury - 1 - self.y0) // self.gcell)
            for l in self.layers:
                for i in range(i0, i1 + 1):
                    for j in range(j0, j1 + 1):
                        if (i, j) in self._exempt:
                            continue
                        self.supply[(l, i, j)] = 0

    def exempt_tile(self, i, j):
        """Keep tile (i,j) traversable (e.g. it contains a pin)."""
        if not (0 <= i < self.nx and 0 <= j < self.ny):
            return
        self._exempt.add((i, j))
        for l in self.layers:
            if self.supply[(l, i, j)] == 0:
                del self.supply[(l, i, j)]

    def in_bounds(self, l, i, j):
        return 0 <= i < self.nx and 0 <= j < self.ny and l in self.layers

    def tile_bbox(self, i, j):
        return (self.x0 + i * self.gcell, self.y0 + j * self.gcell,
                self.x0 + (i + 1) * self.gcell, self.y0 + (j + 1) * self.gcell)

    def tile_of(self, x, y):
        i = min(max((x - self.x0) // self.gcell, 0), self.nx - 1)
        j = min(max((y - self.y0) // self.gcell, 0), self.ny - 1)
        return i, j

    def neighbors(self, node):
        """Yield (neighbor_node, edge_key) for traversable neighbors."""
        l, i, j = node
        li = self.layers.index(l)
        for di, d in enumerate(self.DIRS):
            di_, dj_ = [(1, 0), (-1, 0), (0, 1), (0, -1)][di]
            ni, nj = i + di_, j + dj_
            if 0 <= ni < self.nx and 0 <= nj < self.ny:
                yield (l, ni, nj), (l, ni, nj)
        for dl in (-1, 1):
            nl = li + dl
            if 0 <= nl < len(self.layers):
                yield (self.layers[nl], i, j), (self.layers[nl], i, j)


class Net:
    __slots__ = ('name', 'pins', 'path', 'demand')

    def __init__(self, name, pins):
        self.name = name
        self.pins = pins          # list of (layer, i, j) tile nodes
        self.path = []            # list of tile nodes, current plan
        self.demand = 1.0


class WirePlanner:
    """Negotiated-congestion tile router.

    Per-node cost follows RALF's TileRouter: overflow is the fractional
    demand-over-supply on the entered tile plus the accumulated tile
    ``penalty`` (history).  History update per iteration:
    ``p <- p + max(usage/supply-1, 0)`` on every used tile (forgetting
    factor 1, matching RALF's default).
    """

    def __init__(self, grid):
        self.grid = grid
        self.penalty = defaultdict(float)   # history per tile node
        self.usage = defaultdict(float)     # current demand per tile node

    def _edge_key(self, a, b):
        """Key of the node being entered (demand accrues on the target tile)."""
        return b

    def _cost(self, a, b):
        key = self._edge_key(a, b)
        supply = self.grid.supply[key]
        if supply <= 0:
            return float('inf')
        overflow = max(self.usage[key] / supply - 1.0, 0.0)
        return (overflow + self.penalty[key]) * 100 + 1

    def route_net(self, net):
        """A* over tiles from pins[0] to the remaining pins (sequentially
        joined to the growing tree, i.e. Steiner-approx via shared nodes)."""
        tree = set()
        frontier = []
        tie = count()

        def push(node, cost, path):
            heapq.heappush(frontier, (cost, next(tie), node, path))

        # start from first pin; then connect each remaining pin to the tree
        start = net.pins[0]
        tree.add(start)
        full_path = [start]

        for target in net.pins[1:]:
            if target in tree:
                continue
            frontier.clear()
            for node in tree:
                push(node, 0, [node])
            best = {}
            found = None
            while frontier:
                c, _, node, path = heapq.heappop(frontier)
                if node == target:
                    found = path
                    break
                if node in best and best[node] <= c:
                    continue
                best[node] = c
                for nb, _ in self.grid.neighbors(node):
                    ec = self._cost(node, nb)
                    if ec == float('inf'):
                        continue
                    # manhattan heuristic on tile indices (same layer)
                    h = abs(nb[1] - target[1]) + abs(nb[2] - target[2])
                    if nb[0] != target[0]:
                        h += 1
                    push(nb, c + ec + h, path + [nb])
            if found is None:
                logger.warning(f'wire plan: net {net.name} unroutable to '
                               f'{target}')
                continue
            for n in found:
                if n not in tree:
                    full_path.append(n)
                    tree.add(n)
        net.path = full_path

    def _commit(self, net, sign):
        nodes = net.path
        for a, b in zip(nodes, nodes[1:]):
            self.usage[self._edge_key(a, b)] += sign * net.demand

    def plan(self, nets, iterations=20, order_by_fom=True):
        """Run the negotiation loop over ``nets`` (list of ``Net``)."""
        for net in nets:
            self.route_net(net)
            self._commit(net, +1)

        for it in range(iterations):
            def fom(n):
                used = [self.usage[self._edge_key(a, b)]
                        / max(self.grid.supply[self._edge_key(a, b)], 1e-9)
                        for a, b in zip(n.path, n.path[1:])]
                return len(n.path) + sum(max(u - 1, 0) for u in used)

            order = sorted(nets, key=fom, reverse=True) if order_by_fom else nets
            for net in order:
                self._commit(net, -1)
                self.route_net(net)
                self._commit(net, +1)
            # history update once per iteration
            overflow_total = 0.0
            for key, use in self.usage.items():
                supply = self.grid.supply[key]
                if supply > 0 and use > 0:
                    ov = max(use / supply - 1.0, 0.0)
                    self.penalty[key] += ov
                    overflow_total += ov
            logger.debug(f'wire plan iter {it}: total overflow '
                         f'{overflow_total:.2f}')
            if overflow_total < 1e-6:
                break
        return nets

    def congestion_report(self):
        return {str(k): {'usage': u, 'supply': self.grid.supply[k],
                         'penalty': self.penalty[k]}
                for k, u in self.usage.items() if u > 0}

    def emit_guides(self, nets):
        """Emit the WriteGcellGlobalRoute 'wires' schema."""
        wires = []
        for net in nets:
            for a, b in zip(net.path, net.path[1:]):
                la, ia, ja = a
                lb, ib, jb = b
                box_a = self.grid.tile_bbox(ia, ja)
                box_b = self.grid.tile_bbox(ib, jb)
                rect = [min(box_a[0], box_b[0]), min(box_a[1], box_b[1]),
                        max(box_a[2], box_b[2]), max(box_a[3], box_b[3])]
                wires.append({'layer': str(la if la == lb else f'{la}-{lb}'),
                              'net_name': net.name, 'width': 0,
                              'rect': rect, 'connected_pins': []})
        return {'wires': wires}


def plan_from_placement(scaled_placement_verilog_d, leaf_collateral,
                        *, gcell=None, layers=('M1', 'M2'), iterations=20,
                        target_tiles=64):
    """Build tile pins from a placed variant and run the planner.

    ``scaled_placement_verilog_d`` is a dict (or pydantic model) in the
    scaled-placement-verilog domain; ``leaf_collateral`` is the dict from
    ``load_leaf_collateral``.  Returns ``(guides, report)``.

    ``gcell`` is the tile pitch in layout units.  When None (default) it is
    chosen so the longest die side spans about ``target_tiles`` tiles --
    the A* search is quadratic in tile count, so a fixed small gcell would
    explode on real dies.
    """
    d = (scaled_placement_verilog_d.dict()
         if hasattr(scaled_placement_verilog_d, 'dict')
         else scaled_placement_verilog_d)

    # the top-level module is the one no other module instantiates
    inst_ctns = {i['concrete_template_name'] for m in d['modules']
                 for i in m['instances']}
    top = next(m for m in d['modules']
               if m['concrete_name'] not in inst_ctns)
    leaf_map = {l['concrete_name']: l for l in (d.get('leaves') or [])}

    llx, lly, urx, ury = top['bbox']
    if gcell is None:
        gcell = max(1, max(urx - llx, ury - lly) // target_tiles)
    obstacles = []
    nets = defaultdict(list)

    for inst in top['instances']:
        ctn = inst['concrete_template_name']
        tr = inst['transformation']
        leaf = leaf_map.get(ctn)
        if leaf is None:
            # sub-module instance: obstacle + pins approximated by the
            # instance bbox edge midpoints (we don't know internal pins).
            # Look up the placed submodule bbox from modules.
            sub = next((m for m in d['modules']
                        if m.get('concrete_name') == ctn), None)
            ox, oy, sx, sy = tr['oX'], tr['oY'], tr['sX'], tr['sY']
            if sub is not None:
                bw = sub['bbox'][2] - sub['bbox'][0]
                bh = sub['bbox'][3] - sub['bbox'][1]
            else:
                bw = bh = 0
            ilx = ox if sx > 0 else ox - bw
            ily = oy if sy > 0 else oy - bh
            obstacles.append((ilx, ily, ilx + bw, ily + bh))
            fa = {f['formal']: f['actual'] for f in inst['fa_map']}
            # pin stub: route to the tile just outside the block's east
            # edge -- interior of the block is obstacle-covered.
            stub_x = ilx + bw + 1
            stub_y = ily + bh // 2
            for pin, actual in fa.items():
                nets[actual].append((stub_x, stub_y, layers[0],
                                     (ilx, ily, ilx + bw, ily + bh)))
            continue
        bw = leaf['bbox'][2] - leaf['bbox'][0]
        bh = leaf['bbox'][3] - leaf['bbox'][1]
        ox, oy, sx, sy = tr['oX'], tr['oY'], tr['sX'], tr['sY']
        ilx = ox if sx > 0 else ox - bw
        ily = oy if sy > 0 else oy - bh
        obstacles.append((ilx, ily, ilx + bw, ily + bh))
        # terminals -> tile pins
        col = leaf_collateral.get(ctn, {})
        terms = col.get('terminals', {})
        fa = {f['formal']: f['actual'] for f in inst['fa_map']}
        for pin, rects in terms.items():
            actual = fa.get(pin)
            if actual is None:
                continue
            for r in rects:
                rx0, ry0, rx1, ry1 = r['rect']
                cx, cy = (rx0 + rx1) / 2, (ry0 + ry1) / 2
                gx = ox + sx * cx
                gy = oy + sy * cy
                nets[actual].append((int(gx), int(gy), r['layer'],
                                     (ilx, ily, ilx + bw, ily + bh)))

    grid = TileGrid((llx, lly, urx, ury), gcell, layers,
                    obstacles=obstacles)
    net_objs = []
    for name, pts in nets.items():
        nodes = []
        for x, y, layer, blk in pts:
            i, j = grid.tile_of(int(x), int(y))
            # exempt the pin tile and carve plus-shaped escape channels to
            # all four edges of its own block so a buried pin can reach
            # routing tracks
            grid.exempt_tile(i, j)
            bi0, bj0, bi1, bj1 = blk
            gi0 = max(0, (bi0 - grid.x0) // grid.gcell)
            gi1 = min(grid.nx - 1, (bi1 - 1 - grid.x0) // grid.gcell)
            gj0 = max(0, (bj0 - grid.y0) // grid.gcell)
            gj1 = min(grid.ny - 1, (bj1 - 1 - grid.y0) // grid.gcell)
            for gi in range(gi0, gi1 + 1):
                grid.exempt_tile(gi, j)
            for gj in range(gj0, gj1 + 1):
                grid.exempt_tile(i, gj)
            l = layer if layer in layers else layers[0]
            node = (l, i, j)
            if node not in nodes:
                nodes.append(node)
        if len(nodes) > 1:
            net_objs.append(Net(name, nodes))

    planner = WirePlanner(grid)
    planner.plan(net_objs, iterations=iterations)
    return planner.emit_guides(net_objs), planner.congestion_report()
