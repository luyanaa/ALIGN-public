"""Driver: route one module's scaled placement with the Python
detailed router and emit ALIGN router-output JSON.

Input contract (same files the C++ router consumes):
  * ``<top>.scaled_placement_verilog.json``  -- placed instance
    transformations + leaf bboxes/terminals (scaled units)
  * leaf collateral ``<ctn>.json``           -- pin rects + blockages
  * ``layers.json``                          -- PDK pitches/directions
  * optional ``<top>_wireplan.json``         -- wire-planner guides,
    used as ``Prefer`` soft costs

Output: ``{bbox, terminals:[{netName,layer,rect,netType}], ...}`` in
the same schema the C++ router's ``<top>_<i>.json`` uses, so
downstream ``gen_viewer_json``/GDS steps consume it unchanged.
"""

import json
import logging
import pathlib

from .grid import RoutingGrid, load_pdk_layers
from .candidates import build_region_candidates
from .constraints import build_problem
from .partition import route_parallel, partition_nets

logger = logging.getLogger(__name__)

# metal/via layers we emit routes on; everything else in leaf collateral
# is copied through as 'drawing'.
ROUTE_LAYERS = {'M1', 'M2', 'M3', 'M4', 'M5', 'M6'}


def _transform_rect(tr, rect):
    """Apply an instance transformation to a leaf-local rect."""
    x0, y0, x1, y1 = rect
    pts = [(x0, y0), (x1, y1)]
    sx, sy = tr['sX'], tr['sY']
    ox, oy = tr['oX'], tr['oY']
    out = []
    for x, y in pts:
        out.append((sx * x + ox, sy * y + oy))
    (ax, ay), (bx, by) = out
    return (min(ax, bx), min(ay, by), max(ax, bx), max(ay, by))


def collect_nets_and_obstacles(module, leaf_collateral):
    """Flatten a placed module into per-net pin shapes + obstacles.

    ``module`` is a module dict from the scaled placement verilog;
    ``leaf_collateral`` is ``{ctn: {'.json': path}}`` (the map built by
    ``gen_leaf_collateral``).

    Returns ``(net_pins, obstacles, boundary_nets)``:
      * ``net_pins``:    ``{net: [pin, ...]}`` where each *pin* is a set
        of ``(x, y, layer)`` access points.  One pin == one
        ``(instance, formal)`` pair -- all metal shapes of that pin on
        every layer are internally connected inside the leaf, so the
        router only needs to touch any one of them.
      * ``obstacles``:   ``{layer: [rect,...]}`` shapes on routing
        layers that block cells (instance metal not on a net's pin)
      * ``boundary_nets``: set of nets that are top-level ports
    """
    leaf_json = {}
    for ctn, files in leaf_collateral.items():
        p = files.get('.json')
        if p and pathlib.Path(p).is_file():
            leaf_json[ctn] = json.loads(pathlib.Path(p).read_text())

    params = set(module.get('parameters', []))
    net_pins = {}
    pin_of = {}                     # (net, inst, formal) -> pin index
    obstacles = {}
    boundary_nets = set()

    for inst in module['instances']:
        ctn = inst['concrete_template_name']
        tr = inst['transformation']
        leaf = leaf_json.get(ctn)
        if leaf is None:
            continue
        fa = {f['formal']: f['actual'] for f in inst['fa_map']}
        for term in leaf.get('terminals', []):
            layer = term['layer']
            net_local = term.get('netName')
            rect = _transform_rect(tr, term['rect'])
            if layer not in ROUTE_LAYERS:
                continue
            net = fa.get(net_local)
            # every metal rect tagged with a net name is an access
            # shape of that net's pin on this instance ('pin' marks the
            # primary access; 'drawing' copies of the same pin still
            # expose the net on that layer)
            if net_local is not None and net:
                key = (net, inst['instance_name'], net_local)
                if key not in pin_of:
                    pin_of[key] = len(net_pins.setdefault(net, []))
                    net_pins[net].append(set())
                pin = net_pins[net][pin_of[key]]
                for pt in _sample_rect_points(rect, layer):
                    pin.add(pt)
                if net in params:
                    boundary_nets.add(net)
            else:
                obstacles.setdefault(layer, []).append(rect)
    return net_pins, obstacles, boundary_nets


def _sample_rect_points(rect, layer, n=3):
    """Up to ``n`` access points spread along a pin rect's long axis."""
    x0, y0, x1, y1 = rect
    pts = set()
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    pts.add((cx, cy, layer))
    if x1 - x0 > y1 - y0:                       # long horizontal strip
        pts.add((x0, cy, layer))
        pts.add((x1, cy, layer))
    elif y1 - y0 > x1 - x0:                     # long vertical strip
        pts.add((cx, y0, layer))
        pts.add((cx, y1, layer))
    return pts


def _boundary_pin_for(net, pins, bbox):
    """Escort a net to the nearest die edge so it has an external
    connection point (top-level ports and power/ground rails both need
    one).  ``pins`` is a list of access-point *sets*; returns a new
    single-point pin set."""
    x0, y0, x1, y1 = bbox
    pts = [pt for pin in pins for pt in pin]
    cx = sum(p[0] for p in pts) // len(pts)
    cy = sum(p[1] for p in pts) // len(pts)
    dists = {'W': cx - x0, 'E': x1 - cx, 'S': cy - y0, 'N': y1 - cy}
    edge = min(dists, key=dists.get)
    if edge in 'WE':
        return {(x0 if edge == 'W' else x1, cy, 'M4')}
    return {(cx, y0 if edge == 'S' else y1, 'M4')}


def detail_route_module(module, leaf_collateral, pdk_json,
                        wire_guides=None, backend='sat',
                        n_workers=None, solver_kw=None):
    """Route one placed module.

    Returns ``(terminals, report)`` where ``terminals`` is a list of
    router-output shapes and ``report`` carries stats.
    """
    metal_infos, via_infos = load_pdk_layers(pdk_json)
    bbox = tuple(module['bbox'])
    grid = RoutingGrid(bbox, metal_infos, via_infos)

    net_pins, obstacles, boundary_nets = collect_nets_and_obstacles(
        module, leaf_collateral)

    # every net gets an edge-escape pin so power/ground rails and
    # top-level ports both terminate at the die boundary
    for net in list(net_pins):
        net_pins[net].append(
            _boundary_pin_for(net, net_pins[net], bbox))

    # block obstacles on their layers
    for layer, rects in obstacles.items():
        for r in rects:
            grid.block_region(layer, r)

    guide_map = {}
    if wire_guides:
        for w in wire_guides:
            guide_map.setdefault(w['net'], []).append(
                (w['layer'], tuple(w['rect'])))

    selected = route_parallel(net_pins, grid, guide_map,
                              backend=backend, n_workers=n_workers,
                              solver_kw=solver_kw or {})
    if selected is None:
        return None, {'status': 'failed'}

    terminals = paths_to_terminals(selected, grid)
    report = {'status': 'ok',
              'nets': len(selected),
              'terminals': len(terminals)}
    return terminals, report


def paths_to_terminals(selected_paths, grid):
    """Convert chosen ``CandidatePath`` pieces to router-output rects.

    ``selected_paths`` is ``{net: [CandidatePath, ...]}``."""
    terminals = []
    for net, paths in selected_paths.items():
        for path in paths:
            for piece in path.pieces:
                if piece.kind == 'wire':
                    terminals.append({
                        'netName': net,
                        'layer': piece.layer,
                        'rect': list(piece.bbox),
                        'netType': 'drawing'})
                else:
                    terminals.append({
                        'netName': net,
                        'layer': piece.layer,
                        'rect': _via_rect(grid, piece),
                        'netType': 'drawing'})
    return terminals


def _via_rect(grid, via_piece):
    """Small square at the via cell center."""
    xs, ys = [], []
    for layer, ti, si in via_piece.cells:
        m = grid.metals[layer]
        if m.direction == 'V':
            xs.append(grid.track_coord(layer, ti))
            ys.append(grid.seg_coord(layer, si))
        else:
            ys.append(grid.track_coord(layer, ti))
            xs.append(grid.seg_coord(layer, si))
    cx, cy = xs[0], ys[0]
    w = min(m.width for m in grid.metal_list) // 2
    return [cx - w, cy - w, cx + w, cy + w]
