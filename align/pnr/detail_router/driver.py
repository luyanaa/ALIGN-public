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


def collect_nets_and_obstacles(module, leaf_collateral,
                               input_dir=None):
    """Flatten a placed module into per-net pin access points.

    ``module`` is a module dict from the scaled placement verilog.
    ``leaf_collateral`` is ``{ctn: {'bbox','terminals'}}`` as returned
    by ``placer_rl.leaf_collateral.load_leaf_collateral`` --
    ``terminals`` is already grouped ``{pin_name: [{layer,rect}]}`` so
    each pin's multi-layer access shapes collapse to one logical pin.
    ``input_dir`` (optional) additionally reads ``<ctn>.json`` to
    collect *blockage* shapes (terminals with no pin name) as routing
    obstacles.

    Returns ``(net_pins, obstacles, boundary_nets)``:
      * ``net_pins``:    ``{net: [set-of-(x,y,layer) pts, ...]}`` -- one
        pin == one ``(instance, formal)`` pair
      * ``obstacles``:   ``{layer: [rect,...]}`` blockage shapes on
        routing layers (requires ``input_dir``)
      * ``boundary_nets``: set of nets that are top-level ports
    """
    params = set(module.get('parameters', []))
    net_pins = {}
    pin_of = {}                     # (net, inst, formal) -> pin index
    obstacles = {}
    boundary_nets = set()

    # optional second pass for blockage shapes
    blockages = {}
    if input_dir is not None:
        idir = pathlib.Path(input_dir)
        for inst in module['instances']:
            ctn = inst['concrete_template_name']
            fn = idir / f'{ctn}.json'
            if fn.is_file() and ctn not in blockages:
                blockages[ctn] = json.loads(fn.read_text())\
                    .get('terminals', [])

    for inst in module['instances']:
        ctn = inst['concrete_template_name']
        tr = inst['transformation']
        leaf = leaf_collateral.get(ctn)
        if leaf is None:
            continue
        fa = {f['formal']: f['actual'] for f in inst['fa_map']}
        # ``leaf['terminals']`` is {pin_name: [{layer,rect}, ...]}
        for pin_name, rects in leaf.get('terminals', {}).items():
            net = fa.get(pin_name)
            if net is None:
                continue
            key = (net, inst['instance_name'], pin_name)
            if key not in pin_of:
                pin_of[key] = len(net_pins.setdefault(net, []))
                net_pins[net].append(set())
            pin = net_pins[net][pin_of[key]]
            for r in rects:
                if r['layer'] not in ROUTE_LAYERS:
                    continue
                rect = _transform_rect(tr, r['rect'])
                for pt in _sample_rect_points(rect, r['layer']):
                    pin.add(pt)
            if net in params:
                boundary_nets.add(net)
        # blockages: shapes in the raw leaf json with no pin name
        for term in blockages.get(ctn, []):
            if (term.get('pin') or term.get('netName')) is not None:
                continue
            layer = term['layer']
            if layer in ROUTE_LAYERS:
                obstacles.setdefault(layer, []).append(
                    _transform_rect(tr, term['rect']))
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
                        n_workers=None, solver_kw=None,
                        input_dir=None):
    """Route one placed module.

    Returns ``(terminals, report)`` where ``terminals`` is a list of
    router-output shapes and ``report`` carries stats.
    """
    metal_infos, via_infos = load_pdk_layers(pdk_json)
    bbox = tuple(module['bbox'])
    grid = RoutingGrid(bbox, metal_infos, via_infos)

    net_pins, obstacles, boundary_nets = collect_nets_and_obstacles(
        module, leaf_collateral, input_dir=input_dir)

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
