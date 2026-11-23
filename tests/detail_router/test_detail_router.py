"""Unit + integration tests for align.pnr.detail_router."""

import json
import pathlib

import pytest

from align.pnr.detail_router.grid import (LayerInfo, RoutingGrid,
                                          load_pdk_layers)
from align.pnr.detail_router.candidates import (CandidateGenerator,
                                                build_region_candidates)
from align.pnr.detail_router.constraints import build_problem
from align.pnr.detail_router.driver import (_transform_rect,
                                            collect_nets_and_obstacles)
from align.pnr.detail_router.solver_sat import sat_available, \
    solve_region as solve_sat
from align.pnr.detail_router.solver_mip import mip_available, \
    solve_region as solve_mip

DATA = pathlib.Path(__file__).parent.parent.parent / \
    'pdks' / 'FinFET14nm_Mock_PDK'


def _grid(x1=800, y1=800):
    m1 = LayerInfo('M1', 'V', 80, 32, 48)
    m2 = LayerInfo('M2', 'H', 84, 32, 48)
    m3 = LayerInfo('M3', 'V', 80, 40, 48)
    return RoutingGrid((0, 0, x1, y1), [m1, m2, m3], {})


def test_track_index_roundtrip():
    g = _grid()
    assert g.track_coord('M1', 0) == 0
    assert g.track_coord('M1', 1) == 80
    assert g.track_index('M1', 240) == 3


def test_block_region_removes_cells():
    g = _grid()
    before = len(g.allowed['M1'])
    g.block_region('M1', (200, 0, 240, 800))   # vertical bar on track 3
    after = len(g.allowed['M1'])
    assert before - after > 0
    # track 3 (x=240) cells must be gone
    assert all(t != 3 for (t, _s) in g.allowed['M1'])


def test_candidates_two_pin_net():
    g = _grid()
    gen = CandidateGenerator(g)
    # pins are *sets* of (x,y,layer) access points
    pins = [{(100, 100, 'M2')}, {(400, 400, 'M2')}]
    cands = gen.candidates_for_net('N1', pins)
    assert cands
    # at least one candidate uses two metal layers (a via)
    assert any(len({c[0] for c in p.cells}) > 1 for p in cands)


def test_candidates_same_row_straight():
    g = _grid()
    gen = CandidateGenerator(g)
    pins = [{(100, 168, 'M2')}, {(400, 168, 'M2')}]
    cands = gen.candidates_for_net('N2', pins)
    assert any(all(c[0] == 'M2' for c in p.cells) for p in cands)


def test_mutex_between_nets():
    g = _grid()
    net_pins = {
        'A': [{(100, 100, 'M2')}, {(400, 100, 'M2')}],
        'B': [{(100, 100, 'M1')}, {(100, 400, 'M1')}],
    }
    cands = build_region_candidates(net_pins, g)
    prob = build_problem(cands)
    assert len(prob.paths) > 0


@pytest.mark.skipif(not sat_available(), reason='python-sat missing')
def test_sat_solves_small():
    g = _grid()
    net_pins = {
        'A': [{(80, 168, 'M2')}, {(400, 168, 'M2')}],
        'B': [{(240, 80, 'M1')}, {(240, 400, 'M1')}],
    }
    cands = build_region_candidates(net_pins, g)
    prob = build_problem(cands)
    sel = solve_sat(prob)
    assert sel is not None
    assert not prob.check_connectivity(sel)


@pytest.mark.skipif(not mip_available(), reason='mip missing')
def test_mip_solves_small():
    g = _grid()
    net_pins = {
        'A': [{(80, 168, 'M2')}, {(400, 168, 'M2')}],
        'B': [{(240, 80, 'M1')}, {(240, 400, 'M1')}],
    }
    cands = build_region_candidates(net_pins, g)
    prob = build_problem(cands)
    sel = solve_mip(prob)
    assert sel is not None
    assert not prob.check_connectivity(sel)


def test_transform_rect():
    tr = {'oX': 100, 'oY': 50, 'sX': -1, 'sY': 1}
    assert _transform_rect(tr, (0, 0, 10, 20)) == (90, 50, 100, 70)


@pytest.mark.skipif(not DATA.is_dir(), reason='mock PDK missing')
def test_load_pdk_layers():
    pdk = json.loads((DATA / 'layers.json').read_text())
    metals, vias = load_pdk_layers(pdk)
    names = [m.name for m in metals]
    assert 'M1' in names and 'M2' in names
    assert vias[('M1', 'M2')].name == 'V1'


E2E = pathlib.Path(__file__).parent.parent.parent / \
    'work' / 'e2e_ftota'


def _ftota_module():
    scaled = json.loads(
        (E2E / '3_pnr/inputs/FIVE_TRANSISTOR_OTA'
         '.scaled_placement_verilog.json').read_text())
    return scaled['modules'][0]


def _collateral():
    from align.pnr.placer_rl.leaf_collateral import load_leaf_collateral
    primitives = json.loads(
        (E2E / '2_primitives/__primitives__.json').read_text())
    return load_leaf_collateral(E2E / '3_pnr/inputs', primitives)


@pytest.mark.skipif(
    not (E2E / '3_pnr/inputs').is_dir(),
    reason='e2e_ftota run artifacts missing')
@pytest.mark.skipif(not sat_available(), reason='python-sat missing')
def test_ftota_sat_routes_all_nets():
    from align.pnr.detail_router.driver import detail_route_module
    pdk = json.loads((E2E / '3_pnr/inputs/layers.json').read_text())
    terminals, report = detail_route_module(
        _ftota_module(), _collateral(), pdk, backend='sat', n_workers=1,
        input_dir=E2E / '3_pnr/inputs')
    assert report['status'] == 'ok'
    nets = {t['netName'] for t in terminals}
    assert nets >= {'VIN', 'VIP', 'VOP', 'VON', 'VDD', 'VSS', 'TAIL'}


@pytest.mark.skipif(
    not (E2E / '3_pnr/inputs').is_dir(),
    reason='e2e_ftota run artifacts missing')
@pytest.mark.skipif(not mip_available(), reason='mip missing')
def test_ftota_mip_routes_all_nets():
    from align.pnr.detail_router.driver import detail_route_module
    pdk = json.loads((E2E / '3_pnr/inputs/layers.json').read_text())
    terminals, report = detail_route_module(
        _ftota_module(), _collateral(), pdk, backend='mip', n_workers=1,
        input_dir=E2E / '3_pnr/inputs')
    assert report['status'] == 'ok'
