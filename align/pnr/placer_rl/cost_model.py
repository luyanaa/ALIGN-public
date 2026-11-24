"""Placement cost model shared by the RALF-style solvers.

Implements the cost components used by RALF
(Ratschenberger & Pretl, TechRxiv 2024):

* HPWL over module nets.
* RUDY routing-demand congestion
  (Spindler & Johannes, DATE 2007) approximated the same way RALF does:
  each net contributes a uniform wire density over its bounding box and the
  congestion is the pair-wise density-weighted bbox intersection area.

The problem input is a single module (hierarchy level) described in the
scaled-placement-verilog domain: instances have concrete leaf templates with
integer bboxes and pin terminals.  All coordinates are plain integers in the
placement domain; no PDK objects are required (the mean wire width ``p``
collapses to a scaling constant and is absorbed into the congestion weight,
so we keep RALF's ``p = 1`` supply convention).
"""

import logging
import re
from collections import defaultdict
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Names that denote a power/ground rail or module port rather than an
# auto-generated internal net (ALIGN names internal nets ``NET0``, ``NET1``,
# ...).  Kept deliberately broad so the escape-cost helper also works for
# netlists that do not expose ``module['parameters']`` to the placer.
_POWER_NET_RE = re.compile(
    r'vdd|vss|gnd|vcc|power|avdd|dvdd|avss|dvss|vph|vpn', re.IGNORECASE)
_INTERNAL_NET_RE = re.compile(r'NET\d+', re.IGNORECASE)


def _is_port_like(name):
    """Heuristic: does ``name`` look like a module port / power rail?

    ALIGN's placement-verilog parameters are upper-case identifiers
    (``VINN``, ``VBIASN``, ``ID``, ...) while internal nets are ``NET<n>``;
    ground is the literal ``0``.  Lower-case internal aliases such as
    ``n2`` are *not* treated as ports.
    """
    if not name:
        return False
    if name in ('0', '1'):
        return True
    if _POWER_NET_RE.search(name):
        return True
    if _INTERNAL_NET_RE.fullmatch(name):
        return False
    return bool(re.search(r'[A-Z]', name)) and name.upper() == name


def infer_port_nets(nets, parameters=None):
    """Return the set of net names that must reach the die boundary.

    ``parameters`` (the module's port list from ``module['parameters']``)
    wins when supplied; power/ground names present in the netlist are always
    included.  When it is unavailable the port names are inferred from the
    netlist using :func:`_is_port_like`.
    """
    names = {net.get('name') for net in nets if net.get('name') is not None}
    ports = set()
    if parameters:
        params = set(parameters)
        for nm in names:
            if nm in params or nm in ('0', '1') or _POWER_NET_RE.search(nm):
                ports.add(nm)
    else:
        for nm in names:
            if _is_port_like(nm):
                ports.add(nm)
    return ports


class PlacementProblem:
    """Single-module placement problem extracted from placement_verilog data.

    Attributes:
        name: module (abstract) name.
        instances: ordered list of dicts with keys
            ``instance_name``, ``concrete_template_name``, ``width``,
            ``height``, ``rotatable``.
        nets: list of dicts ``{name, pins:[(inst_idx, pin_name)], bbox_fn}``
            where ``bbox_fn(placement)`` is resolved at evaluation time from
            pin geometry.  ``pins`` may instead carry fixed
            ``(x, y)`` terminal coordinates for module pins.
        port_nets: set of net names/indices that must escape to the die
            boundary (module parameters plus power/ground rails).  Inferred
            from the net names when not supplied.
        parameters: optional ``module['parameters']`` port list.
        die: optional ``(die_w, die_h)`` reference rectangle for
            :func:`escape_cost`; solvers that allocate a die set it.
    """

    def __init__(self, name, instances, nets, symmetry=None, *,
                 parameters=None, port_nets=None):
        self.name = name
        self.instances = instances
        self.nets = nets
        # [{pairs: [[inst_idx,...],...], direction: 'V'|'H'}]
        self.symmetry = symmetry or []
        # Module ports / power rails: nets that need a die-boundary exit.
        self.parameters = list(parameters or [])
        if port_nets is not None:
            self.port_nets = set(port_nets)
        else:
            self.port_nets = infer_port_nets(nets, self.parameters)
        # Set by solvers that know the die (``RLEnvironment``); used as the
        # default reference rectangle for :func:`escape_cost`.
        self.die = None

    @property
    def n(self):
        return len(self.instances)

    def placements_to_boxes(self, placement):
        """Map a placement dict {inst_idx: (x, y, orient)} to bboxes."""
        boxes = {}
        for idx, inst in enumerate(self.instances):
            x, y, orient = placement[idx]
            w, h = inst['width'], inst['height']
            # orientations 0/1 are N/S (sY flip); 90-deg rotations are not
            # representable by ALIGN's sX/sY leaf transforms, so the solver
            # only emits orient in {0, 1}.
            boxes[idx] = (x, y, x + w, y + h)
        return boxes


class HPWLMetric:
    """Half-perimeter wire length over net bounding boxes.

    Net bboxes are formed from the bounding boxes of the connected instance
    pins when pin geometry is available, else from the instance bbox center.
    """

    def __init__(self, problem):
        self.problem = problem

    def net_bbox(self, net, boxes):
        llx, lly, urx, ury = None, None, None, None
        for pin in net['pins']:
            if 'fixed' in pin:
                x0, y0 = pin['fixed']
                r = (x0, y0, x0, y0)
            else:
                idx = pin['inst']
                x0, y0, x1, y1 = boxes[idx]
                if 'rect' in pin:
                    # pin offset rect within the (possibly rotated) instance
                    r = pin['rect'](boxes[idx])
                else:
                    r = (x0, y0, x1, y1)
            llx = r[0] if llx is None else min(llx, r[0])
            lly = r[1] if lly is None else min(lly, r[1])
            urx = r[2] if urx is None else max(urx, r[2])
            ury = r[3] if ury is None else max(ury, r[3])
        return (llx, lly, urx, ury)

    def __call__(self, placement):
        boxes = self.problem.placements_to_boxes(placement)
        hpwl = 0.0
        for net in self.problem.nets:
            llx, lly, urx, ury = self.net_bbox(net, boxes)
            hpwl += (urx - llx) + (ury - lly)
        return hpwl


def escape_cost(placement, problem, port_nets=None, die=None):
    """Sum of port-net distances to their nearest die edge.

    For every port net (module ``parameters`` plus power/ground rails) the
    net's pin bounding box is measured against the die rectangle
    ``(0, 0, die_w, die_h)`` and the *nearest* edge distance is accumulated::

        escape = sum_over_port_nets min( x_lo, y_lo,
                                         die_w - x_hi, die_h - y_hi )

    A pin/net already touching an edge contributes ``0``; an interior net
    contributes its positive distance.  Distances are clamped at zero so a
    net that hangs slightly outside the die does not earn a negative cost.
    Only pins of instances present in ``placement`` are considered, which
    makes the helper usable on partial (in-progress) RL placements.

    ``port_nets`` may contain net names or net indices; when omitted the
    problem's :attr:`PlacementProblem.port_nets` is used.  ``die`` defaults
    to ``problem.die`` and, failing that, the placement's own bbox.
    """
    if port_nets is None:
        port_nets = getattr(problem, 'port_nets', None) or set()
    if not port_nets:
        return 0.0
    port_nets = set(port_nets)
    boxes = {}
    for idx, inst in enumerate(problem.instances):
        if idx not in placement:
            continue
        x, y, _ = placement[idx]
        boxes[idx] = (x, y, x + inst['width'], y + inst['height'])
    if not boxes:
        return 0.0
    if die is None:
        die = getattr(problem, 'die', None)
    if die is None:
        die = (max(b[2] for b in boxes.values()),
               max(b[3] for b in boxes.values()))
    die_w, die_h = die
    total = 0.0
    for ni, net in enumerate(problem.nets):
        if net.get('name') not in port_nets and ni not in port_nets:
            continue
        llx = lly = urx = ury = None
        for pin in net['pins']:
            if 'fixed' in pin:
                x0, y0 = pin['fixed']
                r = (x0, y0, x0, y0)
            else:
                idx = pin['inst']
                if idx not in boxes:
                    continue  # partial placement: instance not placed yet
                if 'rect' in pin:
                    r = pin['rect'](boxes[idx])
                else:
                    r = boxes[idx]
            llx = r[0] if llx is None else min(llx, r[0])
            lly = r[1] if lly is None else min(lly, r[1])
            urx = r[2] if urx is None else max(urx, r[2])
            ury = r[3] if ury is None else max(ury, r[3])
        if llx is None:
            continue
        d = min(llx, lly, die_w - urx, die_h - ury)
        total += max(0.0, d)
    return total


class RUDYCongestion:
    """RALF-style RUDY congestion estimate.

    Wire density of a net: ``d = p * L / (w * h)`` with ``L = w + h`` the
    net-bbox half-perimeter and ``p`` the mean wire width (default 1, the
    supply-normalized convention used in RALF).  Total congestion is the
    sum over ordered net pairs of ``d2 * area(bbox1 ∩ bbox2)``.
    """

    def __init__(self, problem, hpwl_metric, wire_width=1.0):
        self.problem = problem
        self.hpwl = hpwl_metric
        self.p = wire_width

    @staticmethod
    def _intersection_area(r1, r2):
        x_min = max(r1[0], r2[0])
        x_max = min(r1[2], r2[2])
        y_min = max(r1[1], r2[1])
        y_max = min(r1[3], r2[3])
        if x_min > x_max or y_min > y_max:
            return 0.0
        return (x_max - x_min + 1.0) * (y_max - y_min + 1.0)

    def __call__(self, placement):
        boxes = self.problem.placements_to_boxes(placement)
        net_boxes = []
        for net in self.problem.nets:
            b = self.hpwl.net_bbox(net, boxes)
            w = (b[2] - b[0]) + 1
            h = (b[3] - b[1]) + 1
            d = self.p * (w + h) / (w * h)
            net_boxes.append((d, b))
        c = 0.0
        for i, (d1, b1) in enumerate(net_boxes):
            for j, (d2, b2) in enumerate(net_boxes):
                if i != j:
                    c += d2 * self._intersection_area(b1, b2)
        return c


def _strict_intersection_area(r1, r2):
    """Geometric intersection area (edge-touching boxes do NOT overlap)."""
    dx = min(r1[2], r2[2]) - max(r1[0], r2[0])
    dy = min(r1[3], r2[3]) - max(r1[1], r2[1])
    return dx * dy if dx > 0 and dy > 0 else 0.0


def overlap_area(placement, problem):
    """Total pairwise overlap area of placed instance bboxes (should be 0)."""
    boxes = problem.placements_to_boxes(placement)
    area = 0.0
    items = list(boxes.items())
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            area += _strict_intersection_area(items[i][1], items[j][1])
    return area


def group_axis(placement, problem, group):
    """Median centreline of one symmetry group (the explicit axis ``a_g``).

    IdeaPlaceEx models the mirror axis as a decision variable; we keep the
    same notion -- a per-group axis in *true centreline* units -- but derive
    it cheaply from the current placement (median of the member centrelines
    along the mirror direction).  Deriving it inside the energy keeps SA
    cheap while still letting the penalty pull the placement (and therefore
    the axis) toward feasibility, since translating a group shifts its
    median.

    Direction 'V' mirrors about a vertical axis (x centreline); 'H' about a
    horizontal one (y centreline).
    """
    direction = group.get('direction', 'V')
    centers = []
    for pair in group['pairs']:
        for i in pair:
            x, y, _ = placement[i]
            if direction == 'V':
                centers.append(x + problem.instances[i]['width'] / 2)
            else:
                centers.append(y + problem.instances[i]['height'] / 2)
    if not centers:
        return 0.0
    centers.sort()
    return centers[len(centers) // 2]


def symmetry_axes(placement, problem):
    """Per-group derived axis (``None`` when the problem has no symmetry)."""
    sym = getattr(problem, 'symmetry', None)
    if not sym:
        return None
    return [group_axis(placement, problem, group) for group in sym]


def symmetry_penalty(placement, problem, axes=None):
    """Squared violation of the module's ``SymmetricBlocks`` constraints.

    Mirrors the semantics of ``align.schema.constraint.SymmetricBlocks``:
    for direction 'V' (vertical mirror axis) every pair contributes the
    x-coordinates of its centerline; self-symmetric single instances count
    their own centerline.  Paired (non-self) blocks must additionally align
    their perpendicular centerline within a quarter of the block height.

    Axis handling (B2):

    * ``axes is None`` -- legacy/backward-compatible behaviour: the group's
      first pair defines the reference centerline-sum, exactly as before.
    * ``axes`` supplied -- one explicit axis ``a_g`` per group (index-aligned
      with ``problem.symmetry``; a ``None`` entry is derived with
      :func:`group_axis`).  Each pair is measured against its group axis:
      the pair's centerline sum (``2 * centre`` per member) must equal
      ``2 * len(pair) * a_g``, e.g. ``(x_i+w_i) + (x_j+w_j) = 4*a_g`` for a
      two-element V pair -- IdeaPlaceEx's ``x_i + x_j + w = 2a`` form,
      doubled to match this module's existing centerline-sum convention.

    The penalty is the squared L2 deviation so it scales like the HPWL
    term (quadratic in length units).  ``problem.symmetry`` is a list of
    ``{'pairs': [[inst_idx, ...], ...], 'direction': 'V'|'H'}`` entries.
    """
    sym = getattr(problem, 'symmetry', None)
    if not sym:
        return 0.0
    boxes = problem.placements_to_boxes(placement)
    pen = 0.0
    for gidx, group in enumerate(sym):
        direction = group['direction']
        if axes is None:
            axis = None
            ref = None
        else:
            axis = axes[gidx]
            if axis is None:
                axis = group_axis(placement, problem, group)
            ref = None
        for pair in group['pairs']:
            # centerline sum along the mirror direction (2 * true centre
            # per member, matching the original convention)
            cl = 0.0
            for idx in pair:
                b = boxes[idx]
                cl += (b[0] + b[2]) if direction == 'V' else (b[1] + b[3])
            if axis is None:
                # legacy: first pair sets the reference
                if ref is None:
                    ref = cl
                else:
                    pen += (cl - ref) ** 2
            else:
                target = 2.0 * len(pair) * axis
                pen += (cl - target) ** 2
            if len(pair) == 2:
                # perpendicular centerlines must match within
                # quarter-block-size of each block
                i0, i1 = pair
                b0, b1 = boxes[i0], boxes[i1]
                if direction == 'V':
                    c0, c1 = b0[1] + b0[3], b1[1] + b1[3]
                    h0 = b0[3] - b0[1]
                    h1 = b1[3] - b1[1]
                else:
                    c0, c1 = b0[0] + b0[2], b1[0] + b1[2]
                    h0 = b0[2] - b0[0]
                    h1 = b1[2] - b1[0]
                # violation beyond the 1/4-height slack (both bounds must
                # hold, so the violation is the larger of the two excesses)
                v0 = max(abs(2 * (c0 - c1)) - h0, 0)
                v1 = max(abs(2 * (c0 - c1)) - h1, 0)
                pen += max(v0, v1) ** 2
    return pen


@dataclass(frozen=True)
class CostComponents:
    """Unweighted (raw) energy sub-terms for one placement.

    Keeping the raw, unweighted values separate from the scalar energy lets
    callers

    * initialise penalty weights by magnitude-matching to ``hpwl`` (B1),
    * drive subgradient weight updates from the raw violation magnitude, and
    * run a violation-driven early stop on the raw values.

    ``congestion`` is the raw RUDY estimate (the energy uses its square root,
    matching RALF); ``overlap`` is total pairwise instance-bbox overlap area;
    ``symmetry`` is :func:`symmetry_penalty` (squared centreline deviation).
    """

    hpwl: float
    congestion: float
    overlap: float
    symmetry: float


def raw_components(placement, problem, hpwl_metric=None, rudy_metric=None,
                   axes=None):
    """Return the four unweighted :class:`CostComponents` for ``placement``.

    ``hpwl_metric``/``rudy_metric`` may be supplied to reuse already-built
    metric objects (and their cached problem reference); otherwise they are
    constructed here.  ``axes`` (one per symmetry group) selects the
    axis-explicit symmetry penalty (B2); when omitted the symmetry term keeps
    its legacy first-pair-reference behaviour.
    """
    hpwl_metric = hpwl_metric or HPWLMetric(problem)
    rudy_metric = rudy_metric or RUDYCongestion(problem, hpwl_metric)
    return CostComponents(
        hpwl=hpwl_metric(placement),
        congestion=rudy_metric(placement),
        overlap=overlap_area(placement, problem),
        symmetry=symmetry_penalty(placement, problem, axes=axes),
    )


def make_cost(problem, weight_congestion=1.0, weight_overlap=1.0):
    """Return ``cost(placement) = HPWL + wc*sqrt(congestion) + wo*overlap``.

    Matches RALF's annealer energy ``E = HPWL + sqrt(congestion)`` plus a
    hard overlap penalty used by the sequential (RL) formulation where
    intermediate states may overlap.
    """
    hpwl = HPWLMetric(problem)
    rudy = RUDYCongestion(problem, hpwl)

    def cost(placement):
        import math
        parts = raw_components(placement, problem, hpwl, rudy)
        return (parts.hpwl
                + weight_congestion * math.sqrt(max(parts.congestion, 0.0))
                + weight_overlap * parts.overlap)

    cost.hpwl = hpwl       # exposed for diagnostics
    cost.rudy = rudy
    return cost
