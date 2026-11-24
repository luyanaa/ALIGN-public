"""Tests for the RALF-style GAT+decoder RL placer.

All tests are torch-gated: the RL solver requires the optional torch
dependency, while the rest of ``placer_rl`` (SA path) stays torch-free.
"""

import math

import pytest

torch = pytest.importorskip('torch')

from align.pnr.placer_rl.cost_model import (  # noqa: E402
    PlacementProblem, symmetry_penalty)
from align.pnr.placer_rl.driver import build_problem  # noqa: E402
from align.pnr.placer_rl.rl_solver import (  # noqa: E402
    RLEnvironment, RLPlacementSolver, parse_device_type)


def small_problem():
    instances = [
        {'instance_name': 'A', 'concrete_template_name': 'NMOS_S_1',
         'width': 10, 'height': 20, 'rotatable': True},
        {'instance_name': 'B', 'concrete_template_name': 'PMOS_S_1',
         'width': 10, 'height': 10, 'rotatable': True},
        {'instance_name': 'C', 'concrete_template_name': 'CAP_1',
         'width': 30, 'height': 10, 'rotatable': True},
    ]
    nets = [
        {'name': 'VDD', 'pins': [{'inst': 0, 'pin': 'p'},
                                 {'inst': 1, 'pin': 'p'}]},
        {'name': 'n2', 'pins': [{'inst': 1, 'pin': 'p'},
                                {'inst': 2, 'pin': 'p'}]},
    ]
    return PlacementProblem('M', instances, nets)


def symmetric_problem():
    sizes = [(10, 20), (10, 20), (30, 10), (30, 10),
             (8, 12), (8, 12), (14, 6), (14, 6)]
    instances = [
        {'instance_name': f'I{i}', 'concrete_template_name': 'L',
         'width': w, 'height': h, 'rotatable': True}
        for i, (w, h) in enumerate(sizes)]
    nets = [
        {'name': 'VSS', 'pins': [{'inst': 0, 'pin': 'p'},
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


# ---------------------------------------------------------------------------
# (d) device-type parsing
# ---------------------------------------------------------------------------

def test_parse_device_type_buckets():
    assert parse_device_type('NMOS_S_1') == 'NMOS'
    assert parse_device_type('DP_NMOS_B_3') == 'NMOS'
    assert parse_device_type('SCM_PMOS_0') == 'PMOS'
    assert parse_device_type('RES_X') == 'RES'
    assert parse_device_type('CAP_X') == 'CAP'
    # unparseable -> OTHER
    assert parse_device_type('WEIRD_BLOCK') == 'OTHER'
    # falls back to the abstract name when the concrete name has no token
    assert parse_device_type('X1', 'NMOS_S_2') == 'NMOS'


def test_build_problem_sets_dev_type_and_sym_group():
    module = {
        'abstract_name': 'TOP',
        'instances': [
            {'instance_name': 'I1', 'abstract_template_name': 'NA',
             'concrete_template_name': 'NMOS_S_1',
             'fa_map': [{'formal': 'D', 'actual': 'n1'}]},
            {'instance_name': 'I2', 'abstract_template_name': 'NB',
             'concrete_template_name': 'PMOS_S_1',
             'fa_map': [{'formal': 'D', 'actual': 'n1'}]},
        ],
        'constraints': [
            {'constraint': 'SymmetricBlocks',
             'pairs': [['I1', 'I2']], 'direction': 'V'},
        ],
    }
    prob = build_problem(module, {'NMOS_S_1': (10, 10),
                                  'PMOS_S_1': (20, 10)})
    assert [i['dev_type'] for i in prob.instances] == ['NMOS', 'PMOS']
    assert [i['sym_group'] for i in prob.instances] == [0, 0]


# ---------------------------------------------------------------------------
# (a) encoder produces finite embeddings
# ---------------------------------------------------------------------------

def test_gat_encoder_finite_embeddings():
    p = small_problem()
    solver = RLPlacementSolver(p, seed=0, n_episodes=1, grid=8)
    h_i, g = solver._encode()
    assert h_i.shape == (p.n, solver.policy.hidden)
    assert g.shape == (2 * solver.policy.hidden,)
    assert torch.isfinite(h_i).all()
    assert torch.isfinite(g).all()
    # non-trivial: embeddings are not all identical
    assert h_i.std(dim=0).sum().item() > 0


def test_gat_encoder_handles_no_nets():
    instances = [
        {'instance_name': 'A', 'concrete_template_name': 'NMOS_S_1',
         'width': 10, 'height': 10, 'rotatable': True},
        {'instance_name': 'B', 'concrete_template_name': 'PMOS_S_1',
         'width': 10, 'height': 10, 'rotatable': True},
    ]
    p = PlacementProblem('EMPTY', instances, [])
    solver = RLPlacementSolver(p, seed=0, n_episodes=1, grid=4)
    h_i, g = solver._encode()
    assert torch.isfinite(h_i).all()
    assert torch.isfinite(g).all()


# ---------------------------------------------------------------------------
# (b) solve() completes and returns a full placement
# ---------------------------------------------------------------------------

def test_solve_returns_complete_placement():
    p = small_problem()
    solver = RLPlacementSolver(p, seed=0, n_episodes=3, grid=8)
    placement, bbox, energy = solver.solve()
    assert placement is not None
    assert set(placement) == set(range(p.n))
    assert bbox is not None and bbox[0] > 0 and bbox[1] > 0
    assert math.isfinite(energy)


def test_temperature_kwarg_accepted():
    p = small_problem()
    solver = RLPlacementSolver(p, seed=0, n_episodes=1, grid=8,
                               temperature=0.5)
    placement, _, _ = solver.solve()
    assert set(placement) == set(range(p.n))


# ---------------------------------------------------------------------------
# (c) dense shaping fires before the terminal step
# ---------------------------------------------------------------------------

def test_step_rewards_nonzero_before_final():
    p = small_problem()
    env = RLEnvironment(p, grid=32, seed=0)
    env.reset()
    env.order = list(range(p.n))  # deterministic order
    rewards = []
    for k, inst_idx in enumerate(env.order):
        _, reward, done = env.step(inst_idx, k * 3, 0, 0)
        rewards.append(reward)
    assert len(rewards) == p.n
    # at least one non-terminal step earned a shaped (non-zero) reward
    assert any(abs(r) > 1e-9 for r in rewards[:-1])
    assert env.hpwl_accum > 0


# ---------------------------------------------------------------------------
# (e) terminal reward includes the symmetry penalty
# ---------------------------------------------------------------------------

def _run_env(env, sym_weight, actions):
    env.reset()
    env.order = list(range(env.problem.n))
    env.w_sym = sym_weight
    rewards = []
    for inst_idx, (xb, yb, rot) in zip(env.order, actions):
        _, r, _ = env.step(inst_idx, xb, yb, rot)
        rewards.append(r)
    return rewards[-1], dict(env.placement)


def test_terminal_reward_includes_symmetry_term():
    p = symmetric_problem()
    env = RLEnvironment(p, grid=32, seed=0)
    # magnitude-matching gives a positive weight for a symmetric problem
    assert env.w_sym > 0
    assert env.raw_sym0 > 0

    n = p.n
    actions = [((k * 5) % 16, (k * 3) % 16, k % 2) for k in range(n)]
    r0, placement = _run_env(env, 0.0, actions)
    W = 2.5
    r1, placement2 = _run_env(env, W, actions)
    assert placement == placement2  # identical actions -> identical placement

    pen = symmetry_penalty(placement, p)
    assert pen > 0
    assert abs(r1 - (r0 - W * pen)) < 1e-6 * max(abs(r0), 1.0)


# ---------------------------------------------------------------------------
# (f) solve() legalises the grid scatter onto the symmetry axes
# ---------------------------------------------------------------------------

def test_solve_legalises_symmetric_pairs_exactly():
    """The RL policy emits a grid scatter; ``solve()`` must run the shared
    legaliser so symmetric pairs come back exactly mirrored about one axis
    per group (to integer rounding), unlike the raw episode placement."""
    p = symmetric_problem()
    solver = RLPlacementSolver(p, seed=0, n_episodes=2, grid=8)
    placement, bbox, energy = solver.solve()
    assert set(placement) == set(range(p.n))
    assert bbox is not None and math.isfinite(energy)

    for group in p.symmetry:
        sums = []
        for pair in group['pairs']:
            assert len(pair) == 2
            i, j = pair
            xi, yi, _ = placement[i]
            xj, yj, _ = placement[j]
            wi, hi = p.instances[i]['width'], p.instances[i]['height']
            wj, hj = p.instances[j]['width'], p.instances[j]['height']
            # doubled centreline sum: 2*(2*centre) == 4*axis for each pair,
            # so every pair in a group shares one (doubled) axis value.
            sums.append((2 * xi + wi) + (2 * xj + wj))
            # perpendicular (y) centres of the pair are aligned
            assert abs((2 * yi + hi) - (2 * yj + hj)) <= 1
        # exact mirroring: pair centreline sums agree up to the integer
        # rounding of the partner derivation (<= 2 in the doubled sum).
        assert max(sums) - min(sums) <= 2

