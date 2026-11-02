"""Driver bridging ALIGN datastructures to the RALF-style solvers.

Extracts one ``PlacementProblem`` per module from the ``1_topology`` verilog
plus ``2_primitives`` leaf collateral, runs the selected solver bottom-up,
and emits scaled ``placement_verilog`` alternatives compatible with the
``__placer_dump__.json`` contract consumed by ``3_pnr:route``.

Used when ``schematic2layout.py`` is called with ``--placer rl`` (PPO, needs
torch) or ``--placer sa_pack`` (RALF's sequence-pair annealer, pure Python).
"""

import logging
import math
import random
from collections import defaultdict

from .cost_model import PlacementProblem
from .sa_solver import SAPackingSolver
from .rl_solver import RLPlacementSolver, torch_available

logger = logging.getLogger(__name__)


def _leaf_dims(primitives, concrete_name):
    """Return (width, height) of a leaf concrete template in hN units.

    ``primitives`` entries carry the generator metadata; the authoritative
    size comes from the LEF collateral where available.  Fall back to the
    primitive ``bbox`` recorded in ``__primitives__.json``.
    """
    p = primitives.get(concrete_name)
    if p is None:
        return None
    if 'bbox' in p:
        b = p['bbox']
        return b[2] - b[0], b[3] - b[1]
    if 'x_cells' in p and 'y_cells' in p and 'unit_size' in p:
        return p['x_cells'] * p['unit_size'][0], p['y_cells'] * p['unit_size'][1]
    return None


def _modname(module):
    return module.get('abstract_name', module.get('name'))


def build_problem(module, leaf_sizes, ctn_of=None):
    """Build a ``PlacementProblem`` for one verilog module dict.

    ``ctn_of`` maps ``abstract_template_name -> concrete_template_name``
    (from ``__primitives__.json``); instances in the topology-stage verilog
    carry only the abstract name.  Defaults to identity.
    """
    if ctn_of is None:
        ctn_of = {}
    instances = []
    idx_of = {}
    for inst in module['instances']:
        atn = inst['abstract_template_name']
        ctn = inst.get('concrete_template_name', ctn_of.get(atn, atn))
        dims = leaf_sizes.get(ctn)
        if dims is None:
            logger.warning(f'No size for {ctn}; using unit square')
            dims = (1, 1)
        idx_of[inst['instance_name']] = len(instances)
        instances.append({
            'instance_name': inst['instance_name'],
            'concrete_template_name': ctn,
            'width': dims[0],
            'height': dims[1],
            'rotatable': True,
        })

    nets = defaultdict(list)
    for inst in module['instances']:
        for fa in inst['fa_map']:
            nets[fa['actual']].append(
                {'inst': idx_of[inst['instance_name']], 'pin': fa['formal']})

    net_list = [{'name': n, 'pins': p} for n, p in nets.items() if len(p) > 1]

    # translate SymmetricBlocks constraints to instance indices
    symmetry = []
    for c in module.get('constraints', []):
        if c.get('constraint') == 'SymmetricBlocks':
            pairs = []
            ok = True
            for pair in c['pairs']:
                idx_pair = []
                for name in pair:
                    if name in idx_of:
                        idx_pair.append(idx_of[name])
                    else:
                        ok = False
                        break
                if not ok:
                    break
                pairs.append(idx_pair)
            if ok and pairs:
                symmetry.append({'pairs': pairs,
                                 'direction': c.get('direction', 'V')})

    return PlacementProblem(_modname(module), instances, net_list,
                            symmetry=symmetry)


def _bottom_up_order(modules, top_name):
    """Return modules sorted so children are placed before parents."""
    mods = {_modname(m): m for m in modules}
    order = []

    def visit(nm):
        if nm not in mods or nm in order:
            return
        for inst in mods[nm]['instances']:
            atn = inst['abstract_template_name']
            if atn in mods:
                visit(atn)
        order.append(nm)

    visit(top_name)
    return order


def solve_hierarchy(verilog_d, leaf_sizes, top_name, *,
                    engine='sa_pack', seed=0, nvariants=1,
                    sa_steps=10000, rl_episodes=200, grid=32,
                    ctn_of=None):
    """Place every module bottom-up; return per-module placement results.

    Returns ``{abstract_name: (placement, bbox, energy)}`` where placement is
    ``{inst_name: (x, y, orient)}``.  Sub-module placements register their
    own bboxes into ``leaf_sizes`` so parents see them as blocks.
    """
    mods = {_modname(m): m for m in verilog_d['modules']}
    results = {}
    for nm in _bottom_up_order(verilog_d['modules'], top_name):
        problem = build_problem(mods[nm], leaf_sizes, ctn_of=ctn_of)
        if problem.n == 0:
            continue
        if engine == 'rl' and torch_available():
            solver = RLPlacementSolver(problem, seed=seed,
                                       n_episodes=rl_episodes, grid=grid)
        else:
            if engine == 'rl':
                logger.warning('torch not installed; falling back to '
                               'sequence-pair SA placer')
            solver = SAPackingSolver(problem, seed=seed, n_steps=sa_steps)
        placement, bbox, energy = solver.solve()
        # convert inst_idx keys to names
        named = {problem.instances[i]['instance_name']: v
                 for i, v in placement.items()}
        results[nm] = (named, bbox, energy)
        # parents see this module's concrete variant as a block with the
        # produced outline; register it under its concrete name
        leaf_sizes[f'{nm}_0'] = bbox
        # also set the ctn in the parent's instances
        for parent in verilog_d['modules']:
            for inst in parent['instances']:
                if inst['abstract_template_name'] == nm:
                    inst['concrete_template_name'] = f'{nm}_0'
    return results


def emit_placement_verilog(verilog_d, results, leaf_dims_map, top_name,
                           variant=0, leaf_collateral=None, ctn_of=None):
    """Convert solver output into a scaled_placement_verilog-style module.

    Only emits the geometry the router needs: module bbox + per-instance
    transformations.  Solver ``orient`` values are flips (0 -> N, 1 -> S):
    90-deg rotations are not expressible in ALIGN's sX/sY leaf-transform
    scheme, so the solvers only emit orient in {0, 1}.
    """
    if ctn_of is None:
        ctn_of = {}
    new_modules = []
    for m in verilog_d['modules']:
        nm = _modname(m)
        if nm not in results:
            continue
        placement, bbox, _ = results[nm]
        new_m = dict(m)
        new_m['abstract_name'] = nm
        new_m['concrete_name'] = f'{nm}_{variant}'
        new_m.pop('name', None)
        new_m['bbox'] = [0, 0, int(bbox[0]), int(bbox[1])]
        new_insts = []
        for inst in new_m['instances']:
            x, y, rot = placement[inst['instance_name']]
            inst2 = dict(inst)
            atn = inst['abstract_template_name']
            ctn = inst2.get('concrete_template_name',
                            ctn_of.get(atn, atn))
            inst2['concrete_template_name'] = ctn
            w, h = leaf_dims_map.get(ctn, (1, 1))
            # sX/sY encode flips only. orient 1 -> 'S' (vertical flip):
            # C++ setPlacementInfoFromJson subtracts height when sY<0,
            # so pass the unmirrored origin (x, y+h) for flipped cells.
            sx, sy = (1, 1) if rot == 0 else (1, -1)
            inst2['transformation'] = {
                'oX': int(x),
                'oY': int(y if sy > 0 else y + h),
                'sX': sx, 'sY': sy,
            }
            new_insts.append(inst2)
        new_m['instances'] = new_insts
        new_modules.append(new_m)

    out = dict(verilog_d)
    out['modules'] = new_modules

    # Leaves used by the placed modules; the router needs their terminals.
    # Sub-module instances reference concrete module names (``ATN_<sel>``)
    # which are modules, not leaves, so only real leaf ctns are listed.
    used_ctns = set()
    for m in new_modules:
        for inst in m['instances']:
            ctn = inst['concrete_template_name']
            if ctn in (leaf_collateral or {}):
                used_ctns.add(ctn)
    leaves = []
    for ctn in sorted(used_ctns):
        col = leaf_collateral[ctn]
        terminals = [{'name': pin, 'rect': r['rect']}
                     for pin, rects in col['terminals'].items()
                     for r in rects]
        leaves.append({'abstract_name': ctn, 'concrete_name': ctn,
                       'bbox': col['bbox'], 'terminals': terminals})
    out['leaves'] = leaves
    return out


def rl_placer_driver(verilog_d, primitives, leaf_sizes, top_name, *,
                     engine='sa_pack', seed=0, nvariants=1, sa_steps=10000,
                     rl_episodes=200, grid=32, leaf_collateral=None):
    """Top-level entry: returns scaled placement_verilog alternatives dict."""
    top_name = top_name.upper()
    # atn -> first concrete variant from __primitives__.json
    ctn_of = {}
    for k, v in primitives.items():
        ctn_of.setdefault(v['abstract_template_name'],
                          v['concrete_template_name'])
    alts = {}
    for v in range(nvariants):
        sizes = dict(leaf_sizes)
        results = solve_hierarchy(
            verilog_d, sizes, top_name, engine=engine,
            seed=seed + v, nvariants=1, sa_steps=sa_steps,
            rl_episodes=rl_episodes, grid=grid, ctn_of=ctn_of)
        alts[f'{top_name}_{v}'] = emit_placement_verilog(
            verilog_d, results, sizes, top_name, variant=v,
            leaf_collateral=leaf_collateral, ctn_of=ctn_of)
    return alts
