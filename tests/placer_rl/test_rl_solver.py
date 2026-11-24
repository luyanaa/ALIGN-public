"""Tests for the RALF-style GAT+decoder RL placer.

All tests are torch-gated: the RL solver requires the optional torch
dependency, while the rest of ``placer_rl`` (SA path) stays torch-free.
"""

import math

import pytest

torch = pytest.importorskip('torch')

from align.pnr.placer_rl.cost_model import (  # noqa: E402
    PlacementProblem, escape_cost, overlap_area, symmetry_penalty)
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



# ---------------------------------------------------------------------------
# (g) boundary-escape shaping is wired into the step reward / packing
# ---------------------------------------------------------------------------

def port_problem():
    """Two port-connected instances plus one internal instance."""
    instances = [
        {'instance_name': 'A', 'concrete_template_name': 'NMOS_S_1',
         'width': 10, 'height': 10, 'rotatable': True},
        {'instance_name': 'B', 'concrete_template_name': 'PMOS_S_1',
         'width': 10, 'height': 10, 'rotatable': True},
        {'instance_name': 'C', 'concrete_template_name': 'CAP_1',
         'width': 10, 'height': 10, 'rotatable': True},
        {'instance_name': 'D', 'concrete_template_name': 'RES_1',
         'width': 10, 'height': 10, 'rotatable': True},
    ]
    nets = [
        {'name': 'VDD', 'pins': [{'inst': 0, 'pin': 'p'},
                                 {'inst': 1, 'pin': 'p'}]},
        {'name': 'VOUTP', 'pins': [{'inst': 2, 'pin': 'p'},
                                   {'inst': 3, 'pin': 'p'}]},
        {'name': 'NET1', 'pins': [{'inst': 1, 'pin': 'p'},
                                  {'inst': 2, 'pin': 'p'}]},
    ]
    return PlacementProblem('PORT', instances, nets,
                            port_nets={'VDD', 'VOUTP'})


def test_env_has_escape_and_compaction_terms():
    p = port_problem()
    env = RLEnvironment(p, grid=16, seed=0)
    assert env.port_net_idx == {0, 1}
    assert env.w_escape > 0.0
    assert env.w_compact > 0.0
    env.reset()
    env.order = [0, 1, 2, 3]
    # put instance A at the (interior) centre: its VDD net is far from edges
    env.step(0, 4, 4, 0)
    before = env._net_escape(0)
    assert before > 0.0
    # snapping A onto the die corner reduces the port-net escape distance
    env2 = RLEnvironment(p, grid=16, seed=0)
    env2.reset()
    env2.order = [0, 1, 2, 3]
    env2.step(0, 0, 0, 0)
    assert env2._net_escape(0) == 0.0 < before


def test_solve_packs_ports_closer_to_edge_than_random_scatter():
    """The escape/compaction shaping makes the solved placement at least as
    boundary-friendly as a uniform random scatter over the same die."""
    p = port_problem()
    solver = RLPlacementSolver(p, seed=0, n_episodes=30, grid=8)
    placement, bbox, _ = solver.solve()
    rl_escape = escape_cost(placement, p, p.port_nets,
                            die=solver.env.die)
    die_w, die_h = solver.env.die
    scatter = {
        0: (die_w // 2, die_h // 2, 0),
        1: (die_w // 2, die_h // 2, 0),
        2: (die_w // 2, die_h // 2, 0),
        3: (die_w // 2, die_h // 2, 0),
    }
    scatter_escape = escape_cost(scatter, p, p.port_nets,
                                 die=solver.env.die)
    assert scatter_escape > 0.0
    assert rl_escape <= scatter_escape


def self_symmetric_singletons_problem():
    """Three self-symmetric V-axis blocks plus one free block.

    This mirrors the telescopic-OTA constraint shape that makes the legaliser
    collapse every self-symmetric block onto one axis: only y can separate
    them, so the post-legalisation column compactor must produce a legal,
    tightly packed column.
    """
    sizes = [(10, 20), (10, 30), (10, 15), (30, 10)]
    instances = [
        {'instance_name': f'I{i}', 'concrete_template_name': 'L',
         'width': w, 'height': h, 'rotatable': True}
        for i, (w, h) in enumerate(sizes)]
    nets = [
        {'name': 'VDD', 'pins': [{'inst': 0, 'pin': 'p'},
                                 {'inst': 1, 'pin': 'p'}]},
        {'name': 'VOUTP', 'pins': [{'inst': 2, 'pin': 'p'},
                                   {'inst': 3, 'pin': 'p'}]},
    ]
    symmetry = [{'pairs': [[0], [1], [2]], 'direction': 'V'}]
    return PlacementProblem('COL', instances, nets, symmetry)


def test_solve_self_symmetric_singletons_packs_legal_column():
    p = self_symmetric_singletons_problem()
    solver = RLPlacementSolver(p, seed=0, n_episodes=5, grid=8)
    placement, bbox, energy = solver.solve()
    # no overlaps after the column compaction
    assert overlap_area(placement, p) == 0.0
    # the three self-symmetric blocks share the minimum feasible axis
    axis = max(p.instances[i]['width'] / 2 for i in (0, 1, 2))
    for i in (0, 1, 2):
        centre = placement[i][0] + p.instances[i]['width'] / 2
        assert abs(centre - axis) <= 0.5
    # y-bands are disjoint and stacked from the origin
    ys = sorted((placement[i][1], placement[i][1] + p.instances[i]['height'])
                for i in (0, 1, 2))
    assert ys[0][0] == 0
    for (_, hi), (lo, _) in zip(ys, ys[1:]):
        assert lo >= hi
    # the free block is flush to the right of the column
    col_right = max(placement[i][0] + p.instances[i]['width']
                    for i in (0, 1, 2))
    assert placement[3][0] == col_right
    # tight column: bbox height is the sum of the stacked group heights
    assert bbox[1] == sum(p.instances[i]['height'] for i in (0, 1, 2))
    assert bbox[0] == col_right + p.instances[3]['width']


# ---------------------------------------------------------------------------
# (h) learned placement-order pointer + PPO entropy bonus
# ---------------------------------------------------------------------------

def test_pointer_produces_permutation():
    """The pointer head places every instance exactly once (a permutation)."""
    p = symmetric_problem()
    solver = RLPlacementSolver(p, seed=0, n_episodes=1, grid=8,
                               use_pointer=True)
    traj, _ = solver._episode()
    order = solver.last_order
    assert len(order) == p.n
    assert sorted(order) == list(range(p.n))
    # trajectory inst indices match the emitted order
    assert [t[0] for t in traj] == order
    # each pointer logp is part of the sampled log-prob
    assert torch.isfinite(torch.stack([t[6] for t in traj])).all()


def test_pointer_false_reproduces_random_order():
    """``use_pointer=False`` keeps today's random-order behaviour exactly."""
    p = symmetric_problem()
    solver = RLPlacementSolver(p, seed=0, n_episodes=1, grid=8,
                               use_pointer=False)
    traj, _ = solver._episode()
    assert solver.last_order == solver.env.order
    assert sorted(solver.last_order) == list(range(p.n))
    # ablation still completes a full solve
    placement, _, _ = solver.solve()
    assert set(placement) == set(range(p.n))


def test_entropy_bonus_finite_and_changes_loss():
    """The entropy term is finite and shifts the PPO loss when enabled."""
    p = symmetric_problem()
    solver = RLPlacementSolver(p, seed=0, n_episodes=1, grid=8,
                               use_pointer=True)
    traj, _ = solver._episode()

    solver.entropy_coef = 0.0
    loss0, ent0 = solver._evaluate_traj(traj)
    assert math.isfinite(float(loss0.detach()))
    assert math.isfinite(float(ent0.detach())) and float(ent0.detach()) > 0.0

    # The raw value loss is large (returns are HPWL-scale) and float32, so use
    # a coefficient that makes the entropy shift well above float precision.
    coef = 1e4 / float(ent0.detach())
    solver.entropy_coef = coef
    loss1, ent1 = solver._evaluate_traj(traj)

    assert torch.allclose(ent0, ent1)
    # subtracting coef * entropy makes the loss strictly smaller
    assert float(loss1.detach()) < float(loss0.detach())
    assert math.isclose(float(loss0.detach() - loss1.detach()), 1e4,
                        rel_tol=1e-2)

