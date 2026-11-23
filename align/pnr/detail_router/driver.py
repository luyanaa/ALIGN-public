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


def find_symmetric_net_pairs(module):
    """Return ``[(net1, net2, direction), ...]`` from the module's
    ``SymmetricNets`` constraints.  Only vertical-axis pairs (``V``) are
    currently useful to this router: the mirror is a reflection in x.
    """
    pairs = []
    for c in module.get('constraints', []):
        if c.get('constraint') != 'SymmetricNets':
            continue
        n1, n2 = c.get('net1'), c.get('net2')
        if not n1 or not n2 or n1 == n2:
            continue
        pairs.append((n1, n2, c.get('direction', 'V')))
    return pairs


def _instance_center_x(inst, leaf_collateral):
    """Physical x-centre of a placed instance, or ``None`` if unknown."""
    if inst is None:
        return None
    leaf = leaf_collateral.get(inst.get('concrete_template_name'))
    if not leaf or 'bbox' not in leaf:
        return None
    b = leaf['bbox']
    tr = inst['transformation']
    x0 = tr['sX'] * b[0] + tr['oX']
    x1 = tr['sX'] * b[2] + tr['oX']
    return (x0 + x1) / 2.0


def derive_symmetry_axis_x(module, leaf_collateral, bbox):
    """Derive the vertical mirror axis (an x coordinate).

    Preference order:
      1. ``SymmetricBlocks`` with direction ``V``: a self-symmetric
         single block sits centred on the axis; a paired block's axis is
         the midpoint of the two instance centres.  Median over groups
         (robust against a stray self-symmetric block).
      2. The module bbox vertical centreline.
      3. ``None`` (symmetry disabled).
    """
    by_name = {inst['instance_name']: inst
               for inst in module.get('instances', [])}
    estimates = []
    for c in module.get('constraints', []):
        if c.get('constraint') != 'SymmetricBlocks':
            continue
        if c.get('direction', 'V') != 'V':
            continue
        for pair in c.get('pairs', []):
            centres = [_instance_center_x(by_name.get(name), leaf_collateral)
                       for name in pair]
            centres = [c for c in centres if c is not None]
            if len(pair) == 1 and centres:
                estimates.append(centres[0])
            elif len(pair) == 2 and len(centres) == 2:
                estimates.append((centres[0] + centres[1]) / 2.0)
    if estimates:
        estimates.sort()
        return int(round(estimates[len(estimates) // 2]))
    if bbox:
        return (bbox[0] + bbox[2]) // 2
    return None


def mirror_rect_x(rect, axis_x):
    """Reflect a rect in the vertical line ``x == axis_x``.

    The image of ``[x0, x1]`` is ``[2*axis - x1, 2*axis - x0]`` -- the
    endpoints swap roles so ``x0 < x1`` is preserved."""
    x0, y0, x1, y1 = rect
    nx0 = int(round(2 * axis_x - x1))
    nx1 = int(round(2 * axis_x - x0))
    return [min(nx0, nx1), y0, max(nx0, nx1), y1]


def _resolve_net(name, net_pins):
    """Case-insensitive net lookup (constraints are upper-cased)."""
    if name in net_pins:
        return name
    up = name.upper()
    for net in net_pins:
        if net.upper() == up:
            return net
    return None


def _rects_overlap(a, b):
    """Closed-rect overlap: boundary contact counts (a short)."""
    return not (a[2] < b[0] or b[2] < a[0]
                or a[3] < b[1] or b[3] < a[1])


def _mirror_is_legal(partner, mirrored, existing, net_pins, bbox, tol=0):
    """Re-check a mirrored terminal set before committing to it.

    * every rect lies inside the die bbox (an overhang no larger than
      ``tol`` is allowed: an edge-escape via legitimately overhangs the
      die by half a wire width, on the primary side too),
    * no rect overlaps a foreign net's terminal on the same layer,
    * every partner pin is touched by a mirrored terminal,
    * the mirrored terminals form one connected component.

    Returns ``(ok, reason)``.
    """
    from .validate import check_connectivity, check_pin_coverage

    for t in mirrored:
        x0, y0, x1, y1 = t['rect']
        if (x0 < bbox[0] - tol or y0 < bbox[1] - tol
                or x1 > bbox[2] + tol or y1 > bbox[3] + tol):
            return False, 'outside die bbox'
    for t in mirrored:
        for e in existing:
            if e['netName'] == partner:
                continue
            if e['layer'] == t['layer'] and \
                    _rects_overlap(t['rect'], e['rect']):
                return False, f"overlaps {e['netName']} on {t['layer']}"
    if check_pin_coverage(mirrored, {partner: net_pins.get(partner, [])}):
        return False, 'does not reach partner pins'
    comps = check_connectivity(mirrored)
    if comps.get(partner, 0) != 1:
        return False, f"partner has {comps.get(partner)} components"
    return True, ''


def _route_with_symmetry(net_pins, grid, bbox, guide_map, backend,
                         n_workers, solver_kw, boundary_pins,
                         net_pin_rects, sym_pairs, axis_x,
                         leaf_collateral, module, input_dir):
    """Try to route only the primary net of each symmetric pair and
    mirror its terminal set into the partner.

    Returns ``(terminals, report)`` on success or ``None`` to signal the
    caller to fall back to the completely normal (non-symmetric) route.
    Any infeasibility degrades the whole attempt, so the normal path
    stays byte-for-byte unchanged.
    """
    active = []
    partners = set()
    for n1_name, n2_name, direction in sym_pairs:
        if direction != 'V':
            logger.info('symmetric pair %s/%s is %s-axis; not supported',
                        n1_name, n2_name, direction)
            continue
        n1 = _resolve_net(n1_name, net_pins)
        n2 = _resolve_net(n2_name, net_pins)
        if n1 is None or n2 is None or n1 == n2:
            continue
        if n1 in partners or n2 in partners:
            continue  # keep pairs disjoint (safety cap)
        if n1 not in net_pins or n2 not in net_pins:
            continue
        active.append((n1, n2))
        partners.add(n2)
    if not active:
        return None

    route_pins = {n: p for n, p in net_pins.items() if n not in partners}
    if not route_pins:
        return None
    route_bnd = {n: i for n, i in boundary_pins.items()
                 if n not in partners}
    route_rects = {n: r for n, r in net_pin_rects.items()
                   if n not in partners}
    reserve_rects = {n: net_pin_rects[n] for n in partners
                     if n in net_pin_rects}
    # the partners are not routed normally, but their real pin metal is
    # still foreign to every mirrored net
    partner_pin_terms = pin_terminals(
        net_pins, {n2: True for _, n2 in active}, grid,
        leaf_collateral, module, input_dir)

    selected = route_parallel(
        route_pins, grid, guide_map, backend=backend,
        n_workers=n_workers, solver_kw=solver_kw,
        boundary_pins=route_bnd, pin_rects=route_rects,
        reserve_pin_rects=reserve_rects)
    if selected is None:
        logger.info('symmetry attempt: primary-side routing failed')
        return None

    terminals = paths_to_terminals(selected, grid)
    terminals += pin_terminals(net_pins, selected, grid, leaf_collateral,
                               module, input_dir)

    mirrored_pairs = []
    tol = max((m.width for m in grid.metal_list), default=0) // 2
    for n1, n2 in active:
        if n1 not in selected:
            logger.info('symmetry attempt: primary net %s unrouted', n1)
            return None
        mirrored = [dict(t) for t in
                    paths_to_terminals({n1: selected[n1]}, grid)]
        for t in mirrored:
            t['netName'] = n2
            t['rect'] = mirror_rect_x(t['rect'], axis_x)
            t['mirroredFrom'] = n1
        # partner's *real* pin metal (not a mirror): the placement
        # decides where the partner's pins actually are
        mirrored += pin_terminals(net_pins, {n2: True}, grid,
                                  leaf_collateral, module, input_dir)
        ok, reason = _mirror_is_legal(
            n2, mirrored, terminals + partner_pin_terms,
            net_pins, bbox, tol)
        if not ok:
            logger.info('symmetry attempt: mirroring %s -> %s rejected '
                        '(%s); routing normally', n1, n2, reason)
            return None
        terminals.extend(mirrored)
        mirrored_pairs.append([n1, n2])

    report = {
        'status': 'ok',
        'nets': len(selected) + len(mirrored_pairs),
        'terminals': len(terminals),
        'symmetric': {
            'axis_x': axis_x,
            'mirrored_pairs': mirrored_pairs,
        },
    }
    return terminals, report


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

    # Symmetric-net routing (A3): route the primary net of each
    # SymmetricNets pair, then mirror its wire/via set into the partner.
    # Gated on (a) at least one explicit pair and (b) a derivable axis;
    # any failure falls through to the unchanged normal route below.
    sym_pairs = find_symmetric_net_pairs(module)
    if sym_pairs:
        axis_x = derive_symmetry_axis_x(module, leaf_collateral, bbox)
        if axis_x is not None:
            attempt = _route_with_symmetry(
                net_pins, grid, bbox, guide_map, backend, n_workers,
                solver_kw or {}, boundary_pins, net_pin_rects,
                sym_pairs, axis_x, leaf_collateral, module, input_dir)
            if attempt is not None:
                return attempt
            logger.info('symmetric-net mirror routing degraded; '
                        'using normal routing for every net')
        else:
            logger.info('no symmetry axis derivable; symmetry disabled')

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
    for layer, gx, gy in via_piece.cells:
        m = grid.metals[layer]
        ws.append(m.width)
        # unified lattice key -> physical point (layer-independent)
        x, y = grid.cell_anchor(layer, (gx, gy))
        xs.append(x)
        ys.append(y)
    # cells from different layers quantise to slightly different
    # centres; average them so the via lands on the true junction
    cx = sum(xs) // len(xs)
    cy = sum(ys) // len(ys)
    w = min(ws) // 2
    return [cx - w, cy - w, cx + w, cy + w]
