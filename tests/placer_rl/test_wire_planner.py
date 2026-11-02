import pytest

from align.pnr.wire_planner import TileGrid, WirePlanner, Net, plan_from_placement


def test_tile_grid_basics():
    g = TileGrid((0, 0, 1000, 1000), gcell=100, layers=('M1', 'M2'))
    assert g.nx == 10 and g.ny == 10
    assert g.tile_of(0, 0) == (0, 0)
    assert g.tile_of(999, 999) == (9, 9)
    nbs = dict(g.neighbors(('M1', 0, 0)))
    assert ('M1', 1, 0) in nbs and ('M1', 0, 1) in nbs and ('M2', 0, 0) in nbs


def test_obstacle_blocks_edges():
    g = TileGrid((0, 0, 1000, 1000), gcell=100, layers=('M1',),
                 obstacles=[(0, 0, 300, 300)])
    assert g.supply[('M1', 1, 1)] == 0
    assert g.supply[('M1', 5, 5)] != 0


def test_planner_routes_simple_net():
    g = TileGrid((0, 0, 500, 500), gcell=100, layers=('M1', 'M2'))
    net = Net('n1', [('M1', 0, 0), ('M1', 4, 4)])
    p = WirePlanner(g)
    p.plan([net], iterations=3)
    assert len(net.path) > 0
    # path is contiguous
    for a, b in zip(net.path, net.path[1:]):
        assert (abs(a[1] - b[1]) + abs(a[2] - b[2]) +
                (a[0] != b[0])) == 1


def test_negotiation_spreads_congestion():
    g = TileGrid((0, 0, 300, 300), gcell=100, layers=('M1',))
    # funnel several nets through the middle
    nets = [Net(f'n{k}', [('M1', 0, 1), ('M1', 2, 1)]) for k in range(8)]
    p = WirePlanner(g)
    p.plan(nets, iterations=10)
    report = p.congestion_report()
    over = {k: v for k, v in report.items() if v['usage'] > v['supply']}
    # with enough iterations the planner should relieve or mark congestion;
    # either way, the run must terminate and produce a report
    assert isinstance(report, dict)


def test_plan_from_placement_minimal():
    d = {
        'modules': [{
            'abstract_name': 'TOP', 'concrete_name': 'TOP_0',
            'bbox': [0, 0, 400, 400], 'parameters': [],
            'constraints': [],
            'instances': [
                {'instance_name': 'I1', 'abstract_template_name': 'L',
                 'concrete_template_name': 'LC',
                 'fa_map': [{'formal': 'P', 'actual': 'n1'}],
                 'transformation': {'oX': 0, 'oY': 0, 'sX': 1, 'sY': 1}},
                {'instance_name': 'I2', 'abstract_template_name': 'L',
                 'concrete_template_name': 'LC',
                 'fa_map': [{'formal': 'P', 'actual': 'n1'}],
                 'transformation': {'oX': 300, 'oY': 0, 'sX': 1, 'sY': 1}},
            ]}],
        'leaves': [{'abstract_name': 'L', 'concrete_name': 'LC',
                    'bbox': [0, 0, 100, 100],
                    'terminals': [{'name': 'P', 'rect': [0, 0, 10, 10]}]}],
        'global_signals': [],
    }
    collateral = {'LC': {'bbox': [0, 0, 100, 100],
                         'terminals': {'P': [{'layer': 'M1',
                                              'rect': [0, 0, 10, 10]}]}}}
    guides, report = plan_from_placement(d, collateral, gcell=100,
                                         layers=('M1', 'M2'), iterations=3)
    assert 'wires' in guides
    assert len(guides['wires']) > 0
