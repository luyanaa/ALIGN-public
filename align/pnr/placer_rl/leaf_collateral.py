"""Load leaf-cell collateral (sizes + pin rects) for the RALF solvers.

Reads the ``<ctn>.json`` files produced in ``2_primitives/`` -- the same
files copied into ``3_pnr/inputs/`` by ``generate_pnr``.  Coordinates are in
the layer-JSON unit grid (same units the scaled placement verilog uses).
"""

import json
import logging
import pathlib

logger = logging.getLogger(__name__)


def _read_lef_size(fp):
    """Minimal LEF parse: SIZE w BY h in microns (DATABASE MICRONS UNITS 1)."""
    import re
    for line in fp:
        m = re.search(r'\bSIZE\s+([\d.]+)\s+BY\s+([\d.]+)', line)
        if m:
            return float(m.group(1)), float(m.group(2))
    return None


def load_leaf_collateral(primitive_dir, primitives, lef_um_per_unit=None):
    """Return ``{concrete_name: {'bbox': [...], 'terminals': {...}}}``.

    ``primitives`` is the ``__primitives__.json`` dict.  Prefers the
    ``<ctn>.json`` collateral; falls back to the ``SIZE`` field of
    ``<ctn>.lef`` (microns) when the JSON is missing.  ``lef_um_per_unit``
    converts LEF microns into placement units (typically 2000 for the mock
    PDKs, i.e. 1000 DBU/um x the hN 2x factor).
    """
    if lef_um_per_unit is None:
        lef_um_per_unit = 2000
    out = {}
    pdir = pathlib.Path(primitive_dir)
    for key, prim in primitives.items():
        ctn = prim['concrete_template_name']
        fn = pdir / f'{ctn}.json'
        if fn.is_file():
            with fn.open('rt') as fp:
                d = json.load(fp)
            terminals = {}
            for t in d.get('terminals', []):
                pin = t.get('pin') or t.get('netName')
                if pin is None:
                    continue  # blockage/drawing shapes have no pin
                terminals.setdefault(pin, []).append(
                    {'layer': t['layer'], 'rect': t['rect']})
            out[ctn] = {'bbox': d['bbox'], 'terminals': terminals}
        else:
            lfn = pdir / f'{ctn}.lef'
            if lfn.is_file():
                with lfn.open('rt') as fp:
                    sz = _read_lef_size(fp)
                if sz is not None:
                    w = int(round(sz[0] * lef_um_per_unit))
                    h = int(round(sz[1] * lef_um_per_unit))
                    out[ctn] = {'bbox': [0, 0, w, h], 'terminals': {}}
                    continue
            logger.warning(f'No collateral for leaf {ctn} in {pdir}')
    return out


def leaf_sizes(collateral):
    """``{ctn: (w, h)}`` for the placer."""
    return {c: (v['bbox'][2] - v['bbox'][0], v['bbox'][3] - v['bbox'][1])
            for c, v in collateral.items()}
