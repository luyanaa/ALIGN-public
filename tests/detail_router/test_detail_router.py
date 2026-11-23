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
    # track 3 (x=240) cells must be gone; keys are unified lattice
    # indices, so x=240 is gx = 240 // g.g = 60
    assert all(gx != 60 for (gx, _gy) in g.allowed['M1'])


def _cell_footprint(grid, layer, key):
    """Pitch-cell footprint of a key: full pitch across the cross axis
    (nearest-track rounding) and one pitch along the segment axis."""
    gx, gy = key
    m = grid.metals[layer]
    x, y = grid.cell_anchor(layer, key)
    p = m.pitch
    if m.direction == 'V':
        return (x - p // 2, y, x + p // 2, y + p)
    return (x, y - p // 2, x + p, y + p // 2)


def test_cell_key_at_contains_physical_point():
    """``cell_key_at`` is the canonical containment map: the cell it
    returns physically contains the query point on *every* layer."""
    g = _grid()
    for layer in ('M1', 'M2', 'M3'):
        for x, y in ((240, 168), (100, 100), (400, 400), (80, 84)):
            key = g.cell_key_at(layer, x, y)
            r = _cell_footprint(g, layer, key)
            assert r[0] <= x <= r[2] and r[1] <= y <= r[3], \
                (layer, (x, y), key, r)


def test_cell_key_same_physical_point_all_layers():
    """A cell key denotes the same physical point on every layer -- the
    core A1 invariant."""
    g = _grid()
    assert g.g == 4
    for layer in ('M1', 'M2', 'M3'):
        assert g.cell_anchor(layer, (60, 42)) == (240, 168)
        # a junction at physical (240, 168) resolves to that same
        # lattice point wherever the layer's tracks/segments fall on it
        key = g.cell_key_at(layer, 240, 168)
        px, py = g.cell_anchor(layer, key)
        fr = _cell_footprint(g, layer, key)
        assert fr[0] <= 240 <= fr[2] and fr[1] <= 168 <= fr[3]


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


def test_wire_bbox_orientation():
    """Wire bboxes are physical rects: long along the layer's preferred
    direction, exactly ``width`` across the track.  Keys are (x, y) on
    every layer."""
    g = _grid()
    gen = CandidateGenerator(g)

    cells = gen._wire_cells_h('M2', 168, 100, 400)
    x0, y0, x1, y1 = gen._wire_bbox('M2', cells)
    assert x1 - x0 > y1 - y0                 # H layer: long in x
    assert y1 - y0 == g.metals['M2'].width
    assert y0 <= 168 <= y1

    cells = gen._wire_cells_v('M1', 240, 100, 400)
    x0, y0, x1, y1 = gen._wire_bbox('M1', cells)
    assert y1 - y0 > x1 - x0                 # V layer: long in y
    assert x1 - x0 == g.metals['M1'].width
    assert x0 <= 240 <= x1


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
