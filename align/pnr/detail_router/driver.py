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

    Returns ``(net_pins, obstacles, boundary_nets, net_pin_rects)``:
      * ``net_pins``:    ``{net: [set-of-(x,y,layer) pts, ...]}`` -- one
        pin == one ``(instance, formal)`` pair
      * ``obstacles``:   ``{layer: [rect,...]}`` blockage shapes on
        routing layers (requires ``input_dir``)
      * ``boundary_nets``: set of nets that are top-level ports
      * ``net_pin_rects``: ``{net: [{layer: [rect,...]}, ...]}`` -- the
        actual transformed pin rectangles per logical pin, used for
        precise access-cell overlap tests
    """
    params = set(module.get('parameters', []))
    net_pins = {}
    net_pin_rects = {}
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
                net_pin_rects.setdefault(net, []).append({})
            pin = net_pins[net][pin_of[key]]
            prects = net_pin_rects[net][pin_of[key]]
            for r in rects:
                if r['layer'] not in ROUTE_LAYERS:
                    continue
                rect = _transform_rect(tr, r['rect'])
                prects.setdefault(r['layer'], []).append(rect)
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
    return net_pins, obstacles, boundary_nets, net_pin_rects


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
    single-point pin set on M4.

    The caller is responsible for de-conflicting the returned
    coordinate against other nets' boundary pins (track quantisation
    can collapse distinct centroids onto one track)."""
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

    net_pins, obstacles, boundary_nets, net_pin_rects = \
        collect_nets_and_obstacles(module, leaf_collateral,
                                   input_dir=input_dir)

    # every net gets an edge-escape pin so power/ground rails and
    # top-level ports both terminate at the die boundary.  Stagger pins
    # that quantise to the same M4 track cell (different y coords can
    # share a track); also record each boundary pin's index so the
    # candidate builder knows not to reserve it as a foreign blockage.
    taken_cells = set()
    boundary_pins = {}
    for net in list(net_pins):
        boundary_pins[net] = len(net_pins[net])
        pin = _boundary_pin_for(net, net_pins[net], bbox)
        (x, y, layer), = pin
        # quantise the along-edge coordinate to its M4 track and bump it
        # until no other boundary pin claims the same track cell
        if x in (bbox[0], bbox[2]):     # W/E edge: y picks the track
            ti = grid.track_index(layer, y)
            while (x, ti) in taken_cells:
                ti += 1
            taken_cells.add((x, ti))
            y = grid.track_coord(layer, ti)
        else:                           # N/S edge: x picks the track
            ti = grid.track_index(layer, x)
            while (ti, y) in taken_cells:
                ti += 1
            taken_cells.add((ti, y))
            x = grid.track_coord(layer, ti)
        net_pins[net].append({(x, y, layer)})
        net_pin_rects.setdefault(net, []).append(
            {layer: [(x, y, x, y)]})

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
                              solver_kw=solver_kw or {},
                              boundary_pins=boundary_pins,
                              pin_rects=net_pin_rects)
    if selected is None:
        return None, {'status': 'failed'}

    terminals = paths_to_terminals(selected, grid)
    # emit the pin access shapes the router connected to, marked
    # 'pin' like the C++ router -- they both document the attachment
    # point and close the physical connectivity of a multi-access pin
    terminals += pin_terminals(net_pins, selected, grid, leaf_collateral,
                               module, input_dir)
    report = {'status': 'ok',
              'nets': len(selected),
              'terminals': len(terminals)}
    return terminals, report


def pin_terminals(net_pins, selected_paths, grid, leaf_collateral,
                  module, input_dir):
    """Emit ``netType='pin'`` terminals for the leaf metal shapes that
    the routed nets attach to.

    The routed wires connect to a pin at one of its access points; the
    pin's own metal (inside the leaf) is already connected, so echoing
    the pin rect as a 'pin' terminal both documents the attachment and
    lets downstream connectivity checks treat the pin as part of the
    net's geometry.  We rebuild the rects from the raw ``<ctn>.json``
    shapes for the formal pins the net maps to.
    """
    if input_dir is None:
        return []
    # (inst, formal) -> net, and the transformed pin rects per formal
    formal_net = {}
    pin_rects = {}
    leaf_cache = {}
    idir = pathlib.Path(input_dir)
    for inst in module['instances']:
        ctn = inst['concrete_template_name']
        fa = {f['formal']: f['actual'] for f in inst['fa_map']}
        if ctn not in leaf_cache:
            fn = idir / f'{ctn}.json'
            leaf_cache[ctn] = json.loads(fn.read_text()) \
                if fn.is_file() else {'terminals': []}
        for term in leaf_cache[ctn]['terminals']:
            formal = term.get('pin') or term.get('netName')
            if formal is None:
                continue
            net = fa.get(formal)
            if net not in selected_paths:
                continue
            layer = term['layer']
            if layer not in ROUTE_LAYERS:
                continue
            rect = _transform_rect(inst['transformation'], term['rect'])
            key = (net, inst['instance_name'], formal)
            pin_rects.setdefault(key, []).append((layer, rect))
    out = []
    for (net, inst_name, formal), items in pin_rects.items():
        for layer, rect in items:
            out.append({'netName': net, 'layer': layer,
                        'rect': list(rect), 'netType': 'pin',
                        'pin': f'{inst_name}/{formal}'})
    return out


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
                        'rect': list(piece.bbox)
                        if piece.bbox else _via_rect(grid, piece),
                        'netType': 'drawing'})
    return terminals


def _via_rect(grid, via_piece):
    """Small square at the via cell center.

    ``via_piece.cells`` holds the metal cells the via touches; its
    ``layer`` is the via layer name (``Vn``) which has no grid entry,
    so derive geometry from the bridged metal cells.
    """
    xs, ys, ws = [], [], []
    for layer, ti, si in via_piece.cells:
        m = grid.metals[layer]
        ws.append(m.width)
        if m.direction == 'V':
            xs.append(grid.track_coord(layer, ti))
            ys.append(grid.seg_coord(layer, si))
        else:
            ys.append(grid.track_coord(layer, ti))
            xs.append(grid.seg_coord(layer, si))
    # cells from different layers quantise to slightly different
    # centres; average them so the via lands on the true junction
    cx = sum(xs) // len(xs)
    cy = sum(ys) // len(ys)
    w = min(ws) // 2
    return [cx - w, cy - w, cx + w, cy + w]
