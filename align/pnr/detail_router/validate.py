"""Post-route validation: check that a detail_py result is physically
correct enough to trust.

Three checks:

* ``check_shorts``      -- no two terminals of *different* nets have
  overlapping rects on the same metal layer (a via/short).
* ``check_connectivity``-- every net's terminals form a single
  electrically-connected component, where terminals on the same layer
  connect by rect-overlap, and a via rect connects the two metal
  layers it bridges.
* ``check_pin_coverage``-- every pin access shape the placement
  declared is touched by >= 1 routed terminal of its net.

These are deliberately geometric, independent of the solver's own
model, so they catch modelling bugs (e.g. pin-index-only connectivity)
that a solver-internal check would miss.
"""

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

ROUTE_LAYERS = {'M1', 'M2', 'M3', 'M4', 'M5', 'M6'}


def _overlap(a, b):
    # boundary-touching rects count as connected: a wire ending exactly
    # where a via begins shares a metal edge, which conducts.  (This
    # also matches how grid-quantised endpoints naturally land.)
    return not (a[2] < b[0] or b[2] < a[0]
                or a[3] < b[1] or b[3] < a[1])


def _layer_kind(layer):
    return 'via' if layer.startswith('V') else 'metal'


def check_shorts(terminals):
    """Return a list of ``(netA, netB, layer, rect)`` overlapping pairs
    of different nets on the same layer."""
    bad = []
    by_layer = defaultdict(list)
    for t in terminals:
        if _layer_kind(t['layer']) == 'metal':
            by_layer[t['layer']].append(t)
    for layer, ts in by_layer.items():
        for i in range(len(ts)):
            for j in range(i + 1, len(ts)):
                a, b = ts[i], ts[j]
                if a['netName'] == b['netName']:
                    continue
                if _overlap(a['rect'], b['rect']):
                    bad.append((a['netName'], b['netName'], layer,
                                a['rect']))
    return bad


def _via_layers(via_layer, metal_order):
    """Metal layers a via rect connects: Vn bridges Mn and M(n+1)."""
    try:
        n = int(via_layer[1:])
    except ValueError:
        return []
    return [m for m in metal_order
            if int(m[1:]) in (n, n + 1)]


def check_connectivity(terminals, metal_order=None):
    """Return ``{net: n_components}`` -- should be 1 for every net.

    Rules:
      * same-layer rect overlap -> connected
      * a via rect connects any terminal it overlaps on a bridged metal
        layer, and connects to another via whose bridged-layer set
        intersects its own
      * terminals tagged with the same ``pin`` attribute are one node:
        a leaf pin's internal metal is already connected inside the
        cell, so all its access shapes count as a single attachment
    """
    if metal_order is None:
        metal_order = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6']
    by_net = defaultdict(list)
    for t in terminals:
        by_net[t['netName']].append(t)

    result = {}
    for net, ts in by_net.items():
        n = len(ts)
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        # same pin -> one node (leaf-internal connectivity)
        pin_groups = defaultdict(list)
        for i, t in enumerate(ts):
            if t.get('pin'):
                pin_groups[t['pin']].append(i)
        for ids in pin_groups.values():
            for k in ids[1:]:
                union(ids[0], k)

        for i in range(n):
            for j in range(i + 1, n):
                a, b = ts[i], ts[j]
                if not _overlap(a['rect'], b['rect']):
                    continue
                if a['layer'] == b['layer']:
                    union(i, j)
                else:
                    for v, m in ((a, b), (b, a)):
                        if _layer_kind(v['layer']) == 'via':
                            if m['layer'] in _via_layers(
                                    v['layer'], metal_order):
                                union(i, j)
                    if (_layer_kind(a['layer']) == 'via'
                            and _layer_kind(b['layer']) == 'via'):
                        la = set(_via_layers(a['layer'], metal_order))
                        lb = set(_via_layers(b['layer'], metal_order))
                        if la & lb:
                            union(i, j)
        result[net] = len({find(i) for i in range(n)})
    return result


def check_pin_coverage(terminals, net_pins):
    """``net_pins`` is the driver's ``{net: [set-of-pts]}``.  Returns
    pins (as ``(net, pin_idx)``) whose access rect encloses no routed
    terminal point of that net.

    A pin is *covered* when at least one routed metal/via terminal of
    the same net overlaps the pin's access rect.  We rebuild a coarse
    rect per pin from its points' bounding box per layer.
    """
    by_net = defaultdict(list)
    for t in terminals:
        by_net[t['netName']].append(t)

    uncovered = []
    for net, pins in net_pins.items():
        ts = by_net.get(net, [])
        for pi, pts in enumerate(pins):
            # group the pin's points by layer into access rects
            by_layer = defaultdict(list)
            for x, y, layer in pts:
                by_layer[layer].append((x, y))
            covered = False
            for layer, points in by_layer.items():
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                access = (min(xs), min(ys), max(xs), max(ys))
                for t in ts:
                    if _overlap(t['rect'], access):
                        covered = True
                        break
                if covered:
                    break
            if not covered:
                uncovered.append((net, pi))
    return uncovered


def validate(terminals, net_pins=None):
    """Run all checks; returns a ``{check: details}`` report."""
    report = {}
    shorts = check_shorts(terminals)
    report['shorts'] = shorts
    conn = check_connectivity(terminals)
    report['components'] = conn
    bad_nets = {n: c for n, c in conn.items() if c != 1}
    report['disconnected_nets'] = bad_nets
    if net_pins is not None:
        report['uncovered_pins'] = check_pin_coverage(
            terminals, net_pins)
    report['ok'] = not shorts and not bad_nets and \
        not report.get('uncovered_pins')
    return report
