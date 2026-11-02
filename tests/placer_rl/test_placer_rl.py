import math
import random

import pytest

from align.pnr.placer_rl.cost_model import (
    PlacementProblem, HPWLMetric, RUDYCongestion, overlap_area, make_cost)
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
