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
from collections import defaultdict

logger = logging.getLogger(__name__)


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
    """

    def __init__(self, name, instances, nets, symmetry=None):
        self.name = name
        self.instances = instances
        self.nets = nets
        # [{pairs: [[inst_idx,...],...], direction: 'V'|'H'}]
        self.symmetry = symmetry or []

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


def symmetry_penalty(placement, problem):
    """Squared violation of the module's ``SymmetricBlocks`` constraints.

    Mirrors the semantics of ``align.schema.constraint.SymmetricBlocks``:
    for direction 'V' (vertical mirror axis) every pair contributes the
    x-coordinates of its centerline; self-symmetric single instances count
    their own centerline.  All pairs must share the same centerline
    ``reference``, and paired (non-self) blocks must additionally align
    their perpendicular centerline within a quarter of the block height.

    The penalty is the squared L2 deviation so it scales like the HPWL
    term (quadratic in length units).  ``problem.symmetry`` is a list of
    ``{'pairs': [[inst_idx, ...], ...], 'direction': 'V'|'H'}`` entries.
    """
    sym = getattr(problem, 'symmetry', None)
    if not sym:
        return 0.0
    boxes = problem.placements_to_boxes(placement)
    insts = problem.instances
    pen = 0.0
    for group in sym:
        direction = group['direction']
        ref = None
        for pair in group['pairs']:
            # centerline sum along the mirror direction
            cl = 0.0
            for idx in pair:
                b = boxes[idx]
                cl += (b[0] + b[2]) if direction == 'V' else (b[1] + b[3])
            if ref is None:
                ref = cl
            else:
                pen += (cl - ref) ** 2
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
        return (hpwl(placement)
                + weight_congestion * math.sqrt(max(rudy(placement), 0.0))
                + weight_overlap * overlap_area(placement, problem))

    cost.hpwl = hpwl       # exposed for diagnostics
    cost.rudy = rudy
    return cost
