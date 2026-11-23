import logging
import pathlib
import pytest
from align.schema.graph import Graph
from align.schema import constraint
from align.schema.types import set_context
from align.compiler.util import get_ports_weight
from align.compiler.compiler import compiler_input, annotate_library
from align.compiler.find_constraint import add_or_revert_const, symmnet_device_pairs, recursive_start_points, add_symmetry_const, constraint_generator, self_sym_expand
from align.compiler.motif_symmetry import (
    classify_mos,
    collect_mos_devices,
    pair_pattern,
    find_motif_groups,
    DIFF_SOURCE,
    LOAD,
    INVALID,
)
from align.compiler.gen_abstract_name import PrimitiveLibrary
import textwrap

from utils import clean_data, build_example, ota_six, get_test_id, ota_dummy, comparator, comparator_hier

align_home = pathlib.Path(__file__).resolve().parent.parent.parent
pdk_path = align_home / "pdks" / "FinFET14nm_Mock_PDK"
config_path = pathlib.Path(__file__).resolve().parent.parent / "files"
out_path = pathlib.Path(__file__).resolve().parent / "Results"


def test_symm_net():
    name = f'ckt_{get_test_id()}'
    netlist = ota_six(name)
    constraints = [
        {"constraint": "ConfigureCompiler", "is_digital": True}
    ]
    example = build_example(name, netlist, constraints)
    ckt_library, _ = compiler_input(example, name, pdk_path, config_path)
    ckt = ckt_library.find(name)
    G = Graph(ckt)
    pairs, pinsA, pinsB = symmnet_device_pairs(G, 'VIN', 'VIP', list(), None, True)
    assert pairs == {'VIN': 'VIP', 'MN4': 'MN3'}
    assert pinsA == ['MN4/G', 'VIN']
    assert pinsB == ['MN3/G', 'VIP']
    pairs, pinsA, pinsB = symmnet_device_pairs(G, 'VIN', 'VIP', [{'MN3', 'MN4'}], None)
    assert pairs == {'VIN': 'VIP', 'MN4': 'MN3'}
    pairs, pinsA, pinsB = symmnet_device_pairs(G, 'VIN', 'VIP', ['MN3'], None)
    assert pairs is None
    pairs, pinsA, pinsB = symmnet_device_pairs(G, "VIN", "VIP", ["MN4"], None)
    assert pairs is None
    pairs, pinsA, pinsB = symmnet_device_pairs(G, "VIN", "VIP", list(), ["MN4"])
    assert pairs is None
    pairs, pinsA, pinsB = symmnet_device_pairs(G, "VIN", "VIP", list(), ["MN3"])
    assert pairs is None
    pairs, pinsA, pinsB = symmnet_device_pairs(G, "IBIAS", "TAIL", list(), ["MN3"])
    assert pairs is None
    pairs, pinsA, pinsB = symmnet_device_pairs(G, "VON", "VOP", list(), ["MN3"])
    assert pairs is None
    pairs, pinsA, pinsB = symmnet_device_pairs(G, "VIN", "VON", list(), ["MN3"])
    assert pairs is None
    clean_data(name)


def test_add_symmetry_const():
    name = f'ckt_{get_test_id()}'
    netlist = ota_six(name)
    constraints = [
        {"constraint": "ConfigureCompiler", "is_digital": True}
    ]
    example = build_example(name, netlist, constraints)
    ckt_library, _ = compiler_input(example, name, pdk_path, config_path)
    ckt = ckt_library.find(name)
    with set_context(ckt.constraints):
        x = constraint.SymmetricBlocks(direction="V", pairs=[["MN4", "MN3"]])
    const_pairs = {"MN4": "MN3"}  # skip dictionary element
    with pytest.raises(KeyError):
        add_or_revert_const(const_pairs, ckt.constraints, list())
    assert len(ckt.constraints) == 1
    const_pairs = [["MN4", "MN3"]]
    add_or_revert_const(const_pairs, ckt.constraints, list())
    assert len(ckt.constraints) == 2
    assert ckt.constraints[1] == x
    const_pairs = [["MN4", "MN5"]]  # Skip unequal size
    add_or_revert_const(const_pairs, ckt.constraints, list())
    assert len(ckt.constraints) == 2
    const_pairs = [["VIN", "VIP"]]  # Skip net
    add_or_revert_const(const_pairs, ckt.constraints, list())
    assert len(ckt.constraints) == 2, f"skip nets from symblock constraints {const_pairs}"
    const_pairs = [["MP3", "MP4"]]  # instances do not exist
    add_or_revert_const(const_pairs, ckt.constraints, list())
    assert len(ckt.constraints) == 2, f"skip instances who does not exist {const_pairs}"
    clean_data(name)


def test_match_start_points():
    name = f'ckt_{get_test_id()}'
    netlist = ota_six(name)
    constraints = [
    ]
    example = build_example(name, netlist, constraints)
    ckt_library, _ = compiler_input(example, name, pdk_path, config_path)
    ckt = ckt_library.find(name)
    graph = Graph(ckt)
    match_pairs = dict()
    stop_points = {'IBIAS', 'TAIL', 'VCCX', 'VSSX', 'VON', 'VOP'}
    ports_weight = get_ports_weight(graph)
    recursive_start_points(graph, match_pairs, stop_points, 'VIN', 'VIP', ports_weight)
    assert match_pairs[('VIN', 'VIP')] == {'MN4': 'MN3', 'VIN': 'VIP'}
    clean_data(name)


def test_filter_duplicate_instances():
    name = f'ckt_{get_test_id()}'
    netlist = ota_six(name)
    constraints = []
    example = build_example(name, netlist, constraints)
    ckt_library, _ = compiler_input(example, name, pdk_path, config_path)
    ckt = ckt_library.find(name)
    add = add_symmetry_const(ckt, dict(), set(), [], None)
    pairs = [['MN3', 'MN4'], ['VIN', 'VIP'], ['MN4', 'MN4']]
    mpairs = add.filter_symblock_const(pairs)
    # MN2 is the drain-connected tail device on the shared source net (TAIL) of
    # the MN3/MN4 pair, so it is expanded into the same self-symmetric axis.
    assert mpairs == [['MN3', 'MN4'], ['MN2']]
    clean_data(name)


def test_self_sym_expand():
    name = f'ckt_{get_test_id()}'
    netlist = ota_six(name)
    constraints = []
    example = build_example(name, netlist, constraints)
    ckt_library, _ = compiler_input(example, name, pdk_path, config_path)
    ckt = ckt_library.find(name)
    G = Graph(ckt)
    # MN3/MN4 share their source net TAIL; MN2 drains into TAIL and is the
    # self-symmetric tail/bias device on the same axis.
    assert self_sym_expand(G, ckt, [['MN3', 'MN4']]) == [('MN2', ('MN3', 'MN4'))]
    # Already-identified instances are not re-added.
    assert self_sym_expand(G, ckt, [['MN3', 'MN4']], exclude=['MN2']) == []
    # A non-device or non-differential pair yields no expansion.
    assert self_sym_expand(G, ckt, [['VIN', 'VIP']]) == []
    clean_data(name)


def test_symmnet_filters():
    name = f'ckt_{get_test_id()}'
    netlist = ota_six(name)
    constraints = []
    example = build_example(name, netlist, constraints)
    ckt_library, _ = compiler_input(example, name, pdk_path, config_path)
    ckt = ckt_library.find(name)
    pairs = {1: {'MN3': 'MN4', 'VIN': 'VIP', 'MN4': 'MN4'}}
    add = add_symmetry_const(ckt, pairs, set(), [], None)
    add.loop_through_pairs()
    # Tail device MN2 shares the symmetry axis of the MN3/MN4 diff pair.
    assert ckt.constraints[0].pairs == [['MN3', 'MN4'], ['MN2']]
    clean_data(name)


def test_reusable_const():
    name = f'ckt_{get_test_id()}'
    netlist = ota_six(name)
    constraints = []
    example = build_example(name, netlist, constraints)
    ckt_library, primitive_library = compiler_input(example, name, pdk_path, config_path)
    annotate_library(ckt_library, primitive_library)
    primitives = PrimitiveLibrary(ckt_library, pdk_path).gen_primitive_collateral()
    constraint_generator(ckt_library)
    constraints = ckt_library.find(name).constraints.dict()['__root__']
    clean_data(name)
    clean_data(f"run_{name}")
    example1 = build_example(name, netlist, constraints)
    ckt_library1, primitive_library1 = compiler_input(example1, name, pdk_path, config_path)
    annotate_library(ckt_library1, primitive_library1)
    primitives1 = PrimitiveLibrary(ckt_library1, pdk_path).gen_primitive_collateral()
    constraint_generator(ckt_library1)
    constraints1 = ckt_library1.find(name).constraints
    assert constraints == constraints1


def test_filter_dummy():
    name = f'ckt_{get_test_id()}'
    netlist = ota_dummy(name)
    constraints = []
    example = build_example(name, netlist, constraints)
    ckt_library, primitive_library = compiler_input(example, name, pdk_path, config_path)
    annotate_library(ckt_library, primitive_library)
    ckt = ckt_library.find(name)
    add = add_symmetry_const(ckt, dict(), set(), [], None)
    pairs = [['X_MN3_MN4', 'X_MN3_MN4'], ['X_MN3_DUMMY', 'X_MN4_DUMMY']]
    mpairs = add.filter_symblock_const(pairs)
    assert mpairs == [['X_MN3_MN4'], ['X_MN3_DUMMY', 'X_MN4_DUMMY']]
    clean_data(name)


def test_disable_auto_constraint_virtual():
    name = f'ckt_{get_test_id()}'
    netlist = comparator(name)

    # Sanity check: compler should auto-generate for cmp_inp
    constraints = [
        {"constraint": "PowerPorts", "ports": ["vccx"]},
        {"constraint": "GroundPorts", "ports": ["vssx"]},
        {"constraint": "GroupBlocks", "instances": ["mn0", "mn1", "mn2", "mn3", "mn4"], "instance_name": "x0", "template_name": "cmp_inp"},
        {"constraint": "ConfigureCompiler", "auto_constraint": True, "propagate": True}
    ]
    example = build_example(name, netlist, constraints)
    ckt_library, primitive_library = compiler_input(example, name, pdk_path, config_path)
    annotate_library(ckt_library, primitive_library)
    primitives = PrimitiveLibrary(ckt_library, pdk_path).gen_primitive_collateral()
    constraint_generator(ckt_library)
    assert len([sckt for sckt in ckt_library if 'CMP_INP' in sckt.name]) == 1, f"no CMP_INP hierarchy found {[sckt.name for sckt in ckt_library]}"
    ckt = [sckt for sckt in ckt_library if 'CMP_INP' in sckt.name][0]
    constraints = {c.constraint for c in ckt.constraints}
    assert "SymmetricNets" in constraints or "SymmetricBlocks" in constraints

    # Test disabling
    constraints = [
        {"constraint": "PowerPorts", "ports": ["vccx"]},
        {"constraint": "GroundPorts", "ports": ["vssx"]},
        {"constraint": "GroupBlocks", "instances": ["mn0", "mn1", "mn2", "mn3", "mn4"], "instance_name": "x0", "template_name": "cmp_inp"},
        {"constraint": "ConfigureCompiler", "auto_constraint": False, "propagate": True}
    ]
    example = build_example(name, netlist, constraints)
    ckt_library, primitive_library = compiler_input(example, name, pdk_path, config_path)
    annotate_library(ckt_library, primitive_library)
    primitives = PrimitiveLibrary(ckt_library, pdk_path).gen_primitive_collateral()
    constraint_generator(ckt_library)
    ckt = [sckt for sckt in ckt_library if 'CMP_INP' in sckt.name][0]
    constraints = {c.constraint for c in ckt.constraints}
    assert "SymmetricNets" not in constraints
    assert "SymmetricBlocks" not in constraints
    clean_data(name)


def test_disable_auto_constraint_physical():
    name = f'ckt_{get_test_id()}'
    netlist = comparator_hier(name)

    # Sanity check: compler should auto-generate for cmp_inp
    constraints = [
        {"constraint": "PowerPorts", "ports": ["vccx"]},
        {"constraint": "GroundPorts", "ports": ["vssx"]},
        {"constraint": "ConfigureCompiler", "auto_constraint": True, "propagate": True}
    ]
    example = build_example(name, netlist, constraints)
    ckt_library, primitive_library = compiler_input(example, name, pdk_path, config_path)
    annotate_library(ckt_library, primitive_library)
    primitives = PrimitiveLibrary(ckt_library, pdk_path).gen_primitive_collateral()
    constraint_generator(ckt_library)
    ckt = ckt_library.find("COMPARATOR")
    constraints = {c.constraint for c in ckt.constraints}
    assert "SymmetricNets" in constraints or "SymmetricBlocks" in constraints

    # Test disabling
    constraints = [
        {"constraint": "PowerPorts", "ports": ["vccx"]},
        {"constraint": "GroundPorts", "ports": ["vssx"]},
        {"constraint": "ConfigureCompiler", "auto_constraint": False, "propagate": True}
    ]
    example = build_example(name, netlist, constraints)
    ckt_library, primitive_library = compiler_input(example, name, pdk_path, config_path)
    annotate_library(ckt_library, primitive_library)
    primitives = PrimitiveLibrary(ckt_library, pdk_path).gen_primitive_collateral()
    constraint_generator(ckt_library)
    ckt = ckt_library.find("COMPARATOR")
    constraints = {c.constraint for c in ckt.constraints}
    assert "SymmetricNets" not in constraints
    assert "SymmetricBlocks" not in constraints
    clean_data(name)


def test_group_with_constraint():
    name = f'ckt_{get_test_id()}'
    netlist = textwrap.dedent(
        f"""\
        .subckt {name} vi vo vccx vssx
        mp0 vo vo vccx vccx p w=360e-9 nf=2 m=1
        mn1 vo vi vssx vssx n w=360e-9 nf=2 m=1
        mn2 vo vi vssx vssx n w=360e-9 nf=2 m=1
        mn3 vo vi vssx vssx n w=360e-9 nf=2 m=1
        mn4 vo vi vssx vssx n w=360e-9 nf=2 m=1
        .ends {name}
    """)
    constraints = [
        {"constraint": "ConfigureCompiler", "auto_constraint": False, "propagate": True, "merge_parallel_devices": False},
        {"constraint": "PowerPorts", "ports": ["vccx"]},
        {"constraint": "GroundPorts", "ports": ["vssx"]},
        {"constraint": "DoNotRoute", "nets": ["vssx", "vccx"]},
        {"constraint": "GroupBlocks",
            "instances": ["mn1", "mn2", "mn3", "mn4"],
            "instance_name": "xm",
            "template_name": "mygroup",
            "constraints": [
                {"constraint": "DoNotIdentify", "instances": ["mn1", "mn2", "mn3", "mn4"]},
                {"constraint": "Floorplan", "order": True, "symmetrize": True, "regions": [["mn1", "mn2", "mn3"], ["mn4"]]}
            ]}
    ]

    example = build_example(name, netlist, constraints)
    ckt_library, primitive_library = compiler_input(example, name, pdk_path, config_path)
    annotate_library(ckt_library, primitive_library)

    ckt = ckt_library.find(name)
    constraints = {c.constraint for c in ckt.constraints}
    assert "Floorplan" not in constraints, f"{ckt.constraints}"

    ckt = [ckt for ckt in ckt_library if ckt.name.startswith("MYGROUP")][0]
    constraints = {c.constraint for c in ckt.constraints}
    assert "Floorplan" in constraints, f"{ckt.constraints}"
    assert "DoNotIdentify" in constraints, f"{ckt.constraints}"

    clean_data(name)


def test_motif_classifier():
    # Pure connection classifier (ConstGen MosType).
    assert classify_mos(('n1', 'g', 'n1')) == 'CAP'      # S == D
    assert classify_mos(('n1', 'g', 'g')) == 'DIODE'     # G == D
    assert classify_mos(('n1', 'n1', 'n2')) == 'DUMMY'   # G == S
    assert classify_mos(('n1', 'g', 'n2')) == 'DIFF'     # all distinct

    name = f'ckt_{get_test_id()}'
    example = build_example(name, ota_six(name), [])
    ckt_library, _ = compiler_input(example, name, pdk_path, config_path)
    ckt = ckt_library.find(name)
    devices = collect_mos_devices(ckt)
    assert set(devices) == {'MN1', 'MN2', 'MN3', 'MN4', 'MP5', 'MP6'}
    assert devices['MN1']['mtype'] == 'DIODE'
    assert devices['MP5']['mtype'] == 'DIODE'
    assert devices['MN3']['mtype'] == 'DIFF'
    assert devices['MP6']['mtype'] == 'DIFF'

    # Pair motifs: input diff pair and mirror load.
    assert pair_pattern(devices['MN3'], devices['MN4']) == DIFF_SOURCE
    assert pair_pattern(devices['MP5'], devices['MP6']) == LOAD

    # Hard matchedType+matchedSize gate: a different width means INVALID.
    resized = dict(devices['MN3'])
    resized['params'] = dict(devices['MN3']['params'], W='7.2E-07')
    assert pair_pattern(resized, devices['MN4']) == INVALID
    clean_data(name)


def test_motif_seed_dfs_group():
    name = f'ckt_{get_test_id()}'
    example = build_example(name, ota_six(name), [])
    ckt_library, _ = compiler_input(example, name, pdk_path, config_path)
    ckt = ckt_library.find(name)

    groups = find_motif_groups(ckt, set())
    assert len(groups) == 1
    group = groups[0]
    # Seed: MN3/MN4 share source TAIL (two-device signal net).
    assert set(group['seed']) == {'MN3', 'MN4'}
    # DFS walks both drains to the shared-gate mirror load MP5/MP6.
    assert {tuple(sorted(pair)) for pair in group['pairs']} == {
        ('MN3', 'MN4'), ('MP5', 'MP6')}

    # A source net explicitly marked as a stop point is not a seed.
    assert find_motif_groups(ckt, {'TAIL'}) == []
    clean_data(name)


def test_conflict_resolution_logs_and_skips(caplog):
    name = f'ckt_{get_test_id()}'
    example = build_example(name, ota_six(name), [])
    ckt_library, _ = compiler_input(example, name, pdk_path, config_path)
    ckt = ckt_library.find(name)
    caplog.set_level(logging.DEBUG, logger='align.compiler.find_constraint')

    # MN3/MN4 is already claimed: the new group keeps only MP5/MP6, and the
    # conflict is logged rather than raising from the schema validator.
    written = [{'MN3', 'MN4'}]
    add_or_revert_const([['MN3', 'MN4'], ['MP5', 'MP6']], ckt.constraints, written)
    assert len(ckt.constraints) == 1
    assert ckt.constraints[0].pairs == [['MP5', 'MP6']]
    assert 'conflict resolution: dropping candidate pair' in caplog.text
    assert 'already claimed by an existing symmetry group' in caplog.text

    # A genuine schema rejection is still skipped, but now logs the reason.
    caplog.clear()
    add_or_revert_const([['MN3', 'MN1']], ckt.constraints, [])
    assert len(ckt.constraints) == 1  # mismatched parameters rejected
    assert 'skipping match' in caplog.text
    assert 'Parameters of the symmetry pair' in caplog.text
    clean_data(name)
