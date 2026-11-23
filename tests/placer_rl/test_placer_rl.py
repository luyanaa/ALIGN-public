import math
import random

import pytest

from align.pnr.placer_rl.cost_model import (
    PlacementProblem, HPWLMetric, RUDYCongestion, overlap_area, make_cost,
    raw_components, CostComponents, symmetry_penalty)
from align.pnr.placer_rl.sequence_pair import SequencePair
from align.pnr.placer_rl.sa_solver import SAPackingSolver
from align.pnr.placer_rl.driver import build_problem


def small_problem():
    instances = [
        {'instance_name': 'A', 'concrete_template_name': 'L1',
         'width': 10, 'height': 20, 'rotatable': True},
        {'instance_name': 'B', 'concrete_template_name': 'L2',
         'width': 10, 'height': 10, 'rotatable': True},
        {'instance_name': 'C', 'concrete_template_name': 'L3',
         'width': 30, 'height': 10, 'rotatable': True},
    ]
    nets = [
        {'name': 'n1', 'pins': [{'inst': 0, 'pin': 'p'},
                                {'inst': 1, 'pin': 'p'}]},
        {'name': 'n2', 'pins': [{'inst': 1, 'pin': 'p'},
                                {'inst': 2, 'pin': 'p'}]},
    ]
    return PlacementProblem('M', instances, nets)


def test_sequence_pair_decode_nonoverlap():
    rng = random.Random(0)
    widths = [10, 20, 30, 15]
    heights = [10, 10, 10, 10]
    for _ in range(50):
        gp = list(range(4)); rng.shuffle(gp)
        gn = list(range(4)); rng.shuffle(gn)
        sp = SequencePair(gp, gn)
        pos, (W, H) = sp.decode(widths, heights)
        boxes = [(pos[i][0], pos[i][1],
                  pos[i][0] + widths[i], pos[i][1] + heights[i])
                 for i in range(4)]
        for i in range(4):
            for j in range(i + 1, 4):
                ox = min(boxes[i][2], boxes[j][2]) - max(boxes[i][0], boxes[j][0])
                oy = min(boxes[i][3], boxes[j][3]) - max(boxes[i][1], boxes[j][1])
                assert ox <= 0 or oy <= 0, (i, j, boxes)


def test_hpwl_adjacent_zero():
    p = small_problem()
    # A at (0,0), B at (10,0) touching A's right edge
    placement = {0: (0, 0, 0), 1: (10, 0, 0), 2: (0, 30, 0)}
    hpwl = HPWLMetric(p)
    # instance-bbox based nets: n1 bbox = union(A,B) = (0,0,20,20) -> 40
    # n2 = union(B,C): (0,0,30,40) -> 70; total 110
    assert hpwl(placement) == 110


def test_rudy_positive_and_zero_for_identical():
    # two nets on disjoint instance clusters => congestion 0
    p = PlacementProblem('M', [
        {'instance_name': 'A', 'concrete_template_name': 'L',
         'width': 10, 'height': 10, 'rotatable': True},
        {'instance_name': 'B', 'concrete_template_name': 'L',
         'width': 10, 'height': 10, 'rotatable': True},
        {'instance_name': 'C', 'concrete_template_name': 'L',
         'width': 10, 'height': 10, 'rotatable': True},
        {'instance_name': 'D', 'concrete_template_name': 'L',
         'width': 10, 'height': 10, 'rotatable': True}],
        [{'name': 'n1', 'pins': [{'inst': 0, 'pin': 'p'},
                                 {'inst': 1, 'pin': 'p'}]},
         {'name': 'n2', 'pins': [{'inst': 2, 'pin': 'p'},
                                 {'inst': 3, 'pin': 'p'}]}])
    placement = {0: (0, 0, 0), 1: (10, 0, 0), 2: (100, 0, 0), 3: (110, 0, 0)}
    rudy = RUDYCongestion(p, HPWLMetric(p))
    assert rudy(placement) == 0.0
    # pull C into n1's bbox: congestion positive
    placement[2] = (5, 5, 0)
    assert rudy(placement) > 0.0


def test_overlap_area():
    p = small_problem()
    placement = {0: (0, 0, 0), 1: (5, 0, 0), 2: (0, 30, 0)}
    assert overlap_area(placement, p) > 0
    placement[1] = (10, 0, 0)
    assert overlap_area(placement, p) == 0.0


def test_sa_solver_produces_legal_placement():
    p = small_problem()
    solver = SAPackingSolver(p, seed=1, n_steps=500)
    placement, bbox, energy = solver.solve()
    assert len(placement) == 3
    assert overlap_area(placement, p) == 0.0
    assert energy > 0


def test_build_problem_from_verilog_module():
    module = {
        'abstract_name': 'TOP',
        'instances': [
            {'instance_name': 'I1', 'abstract_template_name': 'NA',
             'concrete_template_name': 'N_X1',
             'fa_map': [{'formal': 'D', 'actual': 'n1'}]},
            {'instance_name': 'I2', 'abstract_template_name': 'NB',
             'concrete_template_name': 'N_X2',
             'fa_map': [{'formal': 'D', 'actual': 'n1'}]},
        ],
    }
    leaf_sizes = {'N_X1': (10, 10), 'N_X2': (20, 10)}
    prob = build_problem(module, leaf_sizes)
    assert prob.n == 2
    assert len(prob.nets) == 1
    assert prob.instances[1]['width'] == 20


def symmetric_problem():
    """Four symmetric V-axis pairs (enough for a non-trivial deviation)."""
    sizes = [(10, 20), (10, 20), (30, 10), (30, 10),
             (8, 12), (8, 12), (14, 6), (14, 6)]
    instances = [
        {'instance_name': f'I{i}', 'concrete_template_name': 'L',
         'width': w, 'height': h, 'rotatable': True}
        for i, (w, h) in enumerate(sizes)]
    nets = [
        {'name': 'n1', 'pins': [{'inst': 0, 'pin': 'p'},
                                {'inst': 2, 'pin': 'p'},
                                {'inst': 4, 'pin': 'p'}]},
        {'name': 'n2', 'pins': [{'inst': 1, 'pin': 'p'},
                                {'inst': 3, 'pin': 'p'},
                                {'inst': 5, 'pin': 'p'}]},
        {'name': 'n3', 'pins': [{'inst': 6, 'pin': 'p'},
                                {'inst': 7, 'pin': 'p'}]},
    ]
    symmetry = [{'pairs': [[0, 1], [2, 3], [4, 5], [6, 7]],
                 'direction': 'V'}]
    return PlacementProblem('SYM', instances, nets, symmetry)


def test_raw_components_are_unweighted():
    p = small_problem()
    placement = {0: (0, 0, 0), 1: (10, 0, 0), 2: (0, 30, 0)}
    parts = raw_components(placement, p)
    assert isinstance(parts, CostComponents)
    assert parts.hpwl == HPWLMetric(p)(placement)
    assert parts.congestion == RUDYCongestion(p, HPWLMetric(p))(placement)
    assert parts.overlap == overlap_area(placement, p)
    assert parts.symmetry == 0.0  # small_problem has no symmetry constraints
    # make_cost stays backward compatible with its original call signature
    cost = make_cost(p, weight_congestion=1.0, weight_overlap=1.0)
    expected = (parts.hpwl + math.sqrt(max(parts.congestion, 0.0))
                + parts.overlap)
    assert abs(cost(placement) - expected) < 1e-9


def test_adaptive_weights_beat_fixed_weight_on_symmetry():
    """Adaptive subgradient growth reaches lower raw symmetry than the
    same initial weight held fixed (no adaptation)."""
    p = symmetric_problem()

    adaptive = SAPackingSolver(p, seed=0, n_steps=1500)
    # quantify the SA trajectory itself, before the symmetry legaliser
    adaptive._legalize_symmetry = lambda pl, sp, fl: pl
    # init magnitude-matching: lambda * raw0 ~= HPWL0
    assert adaptive.raw0.symmetry > 0
    assert abs(adaptive.w_sym_init * adaptive.raw0.symmetry
               - adaptive.hpwl0) < 1e-6 * max(adaptive.hpwl0, 1.0)
    a_pl, _, _ = adaptive.solve()

    fixed_w = adaptive.w_sym_init
    fixed = SAPackingSolver(p, seed=0, n_steps=1500,
                            weight_symmetry=fixed_w)
    fixed._legalize_symmetry = lambda pl, sp, fl: pl
    f_pl, _, _ = fixed.solve()

    # explicit override wins and disables adaptation
    assert fixed.w_sym == fixed_w
    assert fixed.adapt_symmetry is False
    # violation-driven growth is real, and it lowers the raw violation
    assert adaptive.w_sym > adaptive.w_sym_init
    assert (symmetry_penalty(a_pl, p) <= symmetry_penalty(f_pl, p) + 1e-9)
