# -*- coding: utf-8 -*-
"""Motif-driven symmetry detection.

This module is an additive, pure-function flavoured port of ConstGen's
functional-motif symmetry finder (``src/sym_detect/Pattern.cpp`` and
``SymDetect.cpp``).  It deliberately does *not* touch constraints or the
compiler graph: it only classifies MOS devices by how they are connected and
returns candidate symmetry groups.  ``find_constraint.add_symmetry_const``
decides how to vote/merge those candidates with the existing port-seeded
channel.

The key ideas borrowed from ConstGen:

* A MOS device has a *mos type* derived from its three terminals::

      S == D -> CAP
      G == D -> DIODE
      G == S -> DUMMY
      else   -> DIFF

* Two devices form a motif only when they have the same base model **and**
  identical width/length/nf (``Pattern::matchedType`` + ``matchedSize``).

* Symmetry is seeded from ``DIFF_SOURCE`` / ``CROSS_LOAD`` pairs that share a
  source net, then expanded by a *pairwise* lock-step DFS: both sides must
  arrive through the same pin type (S or D; GATE is never traversed).  The DFS
  terminates at a new diff pair, at a ``LOAD``/``CASCODE`` pair, and when the
  two walks meet on a shared net.

All functions operate on small plain dictionaries (see :func:`device_info`) so
the classifier and the DFS can be unit-tested without a compiler run.
"""

from __future__ import annotations

import logging
import re
from itertools import combinations

from .util import get_base_model

logger = logging.getLogger(__name__)

#: MOS connection classes (see module docstring).
MOS_TYPES = ("CAP", "DIODE", "DUMMY", "DIFF")

#: Pair motifs.  ``INVALID`` means "not a matched pair at all".
DIFF_SOURCE = "DIFF_SOURCE"
CROSS_LOAD = "CROSS_LOAD"
LOAD = "LOAD"
CASCODE = "CASCODE"
PASSIVE = "PASSIVE"
INVALID = "INVALID"

#: Pair motifs that can seed a symmetry group.
SEED_PATTERNS = (DIFF_SOURCE, CROSS_LOAD)

#: Nets whose names look like supplies are never treated as signal nets.  The
#: compiler may not have explicit PowerPorts/GroundPorts constraints (especially
#: in unit tests), so a name-based guard keeps shared rails from being mistaken
#: for differential source nodes.
_POWER_RE = re.compile(r"(VSS|VDD|VCC|GND|VNW|VPW|VBB|VSUB)", re.IGNORECASE)


def is_power_net(net, stop_points=()):
    """Return True when *net* is a supply/stop net, not a signal net."""
    if net is None:
        return True
    if stop_points and net in stop_points:
        return True
    return bool(_POWER_RE.search(str(net)))


def _pin_roles(pins):
    """Return ``(S, G, D)`` nets, with a positional fallback.

    MOS models in ALIGN use pin names ``D``/``G``/``S``/``B``.  For primitive
    wrappers without named roles we fall back to the conventional netlist pin
    order ``(D, G, S, B)`` used throughout the compiler.
    """
    if not pins:
        return None
    if all(k in pins for k in ("S", "G", "D")):
        return pins["S"], pins["G"], pins["D"]
    ordered = list(pins.values())
    if len(ordered) >= 3:
        return ordered[2], ordered[1], ordered[0]
    return None


def classify_mos(nets):
    """Classify a MOS from its ``(S, G, D)`` nets (ConstGen ``MosType``)."""
    if not nets:
        return INVALID
    source, gate, drain = nets
    if source == drain:
        return "CAP"
    if gate == drain:
        return "DIODE"
    if gate == source:
        return "DUMMY"
    return "DIFF"


def device_info(subckt, name):
    """Build the plain dict describing MOS *name* in *subckt*.

    Keys: ``name``, ``model`` (base model), ``nets`` (pin -> net),
    ``params`` (upper-cased parameter names), ``mtype``.
    """
    element = subckt.get_element(name)
    if element is None:
        return None
    nets = dict(element.pins or {})
    roles = _pin_roles(nets)
    params = {str(k).upper(): v for k, v in (element.parameters or {}).items()}
    return {
        "name": name,
        "model": get_base_model(subckt, name),
        "nets": nets,
        "roles": roles,
        "params": params,
        "mtype": classify_mos(roles),
    }


def collect_mos_devices(subckt):
    """Return ``{name: device_info}`` for all NMOS/PMOS instances."""
    devices = {}
    for element in getattr(subckt, "elements", []):
        name = element.name
        try:
            base = get_base_model(subckt, name)
        except AssertionError:
            continue
        if base not in ("NMOS", "PMOS"):
            continue
        info = device_info(subckt, name)
        if info is not None:
            devices[name] = info
    return devices


def _size_key(dev):
    """Width/length/nf tuple used by the hard ``matchedSize`` gate."""
    params = dev.get("params", {})
    # W is the canonical width; NFIN is a usable fallback in this PDK.
    width = params["W"] if "W" in params else params.get("NFIN")
    length = params.get("L")
    nf = params.get("NF", 1)
    return (str(width), str(length), str(nf))


def matched_device(dev1, dev2):
    """Hard gate: same base model and equal W/L/nf (ConstGen Pattern)."""
    if dev1 is None or dev2 is None:
        return False
    return dev1["model"] == dev2["model"] and _size_key(dev1) == _size_key(dev2)


def _ok_types(*devs):
    return all(dev is not None and dev["mtype"] in ("DIFF", "DIODE") for dev in devs)


def is_diff_source(dev1, dev2):
    """``DIFF_SOURCE``: differential input pair sharing a source/drain net."""
    if not matched_device(dev1, dev2):
        return False
    if dev1["mtype"] != "DIFF" or dev2["mtype"] != "DIFF":
        return False
    s1, g1, d1 = dev1["roles"]
    s2, g2, d2 = dev2["roles"]
    if s1 == s2 and g1 != g2 and d1 != d2:
        return True
    if s1 != s2 and g1 != g2 and d1 == d2:
        return True
    return False


def is_cross_load(dev1, dev2):
    """``CROSS_LOAD``: cross-coupled pair (gate of one tied to drain of other)."""
    if not matched_device(dev1, dev2):
        return False
    if dev1["mtype"] != "DIFF" or dev2["mtype"] != "DIFF":
        return False
    s1, g1, d1 = dev1["roles"]
    s2, g2, d2 = dev2["roles"]
    return s1 == s2 and g1 == d2 and d1 == g2


def is_cascode(dev1, dev2):
    """``CASCODE``: same gate, different source and drain (stacked pair)."""
    if not matched_device(dev1, dev2) or not _ok_types(dev1, dev2):
        return False
    s1, g1, d1 = dev1["roles"]
    s2, g2, d2 = dev2["roles"]
    return s1 != s2 and g1 == g2 and d1 != d2


def is_load(dev1, dev2):
    """``LOAD``: same source and gate, different drain (mirror load)."""
    if not matched_device(dev1, dev2) or not _ok_types(dev1, dev2):
        return False
    s1, g1, d1 = dev1["roles"]
    s2, g2, d2 = dev2["roles"]
    return s1 == s2 and g1 == g2 and d1 != d2


def is_passive(dev1, dev2):
    """``PASSIVE``: two same-sized MOS devices wired as capacitors."""
    return matched_device(dev1, dev2) and dev1["mtype"] == "CAP" and dev2["mtype"] == "CAP"


def pair_pattern(dev1, dev2):
    """Classify a device pair, mirroring ConstGen ``Pattern::pattern`` order."""
    if not matched_device(dev1, dev2):
        return INVALID
    if is_cross_load(dev1, dev2):
        return CROSS_LOAD
    if is_cascode(dev1, dev2):
        return CASCODE
    if is_load(dev1, dev2):
        return LOAD
    if is_diff_source(dev1, dev2):
        return DIFF_SOURCE
    if is_passive(dev1, dev2):
        return PASSIVE
    return INVALID


def _build_net_index(devices):
    """Map net -> list of ``(device_name, pin)`` for S/D/G connections."""
    index = {}
    for name, dev in devices.items():
        for pin in ("S", "D", "G"):
            net = dev["nets"].get(pin)
            if net is None:
                continue
            index.setdefault(net, []).append((name, pin))
    return index


def _devices_on(net_index, net, pin):
    return sorted(name for name, p in net_index.get(net, []) if p == pin)


def _find_seeds(devices, net_index, stop_points):
    """Find DIFF_SOURCE/CROSS_LOAD seed pairs on signal source nets.

    Conservative gate: the shared source net must be a signal net and must have
    exactly two source-connected MOS devices.  A net shared by many sources is
    an array/rail, not a differential pair.
    """
    seeds = []
    for net in sorted(net_index):
        if is_power_net(net, stop_points):
            continue
        source_devs = _devices_on(net_index, net, "S")
        if len(source_devs) != 2:
            continue
        for a, b in combinations(source_devs, 2):
            pattern = pair_pattern(devices[a], devices[b])
            if pattern in SEED_PATTERNS:
                seeds.append(((a, b), pattern))
    # Stable order: by pair name, then pattern.
    seeds.sort(key=lambda item: (item[0], item[1]))
    return seeds


def _expand_pair(devices, net_index, start):
    """Pairwise lock-step DFS from one seed pair, returning a group of pairs."""
    visited = set()
    group = []
    stack = [(start[0], start[1], None)]  # (a, b, pin type used to reach pair)

    while stack:
        a, b, via = stack.pop()
        key = (a, b) if a <= b else (b, a)
        if key in visited:
            continue
        visited.add(key)

        pattern = pair_pattern(devices[a], devices[b])
        if pattern == INVALID:
            continue
        group.append((a, b))

        # Termination rules (ConstGen ``endSrch``):
        #  * LOAD/CASCODE are leaves.
        #  * A new diff pair reached by expansion must not be merged further
        #    (the seed itself, ``via is None``, is allowed to expand).
        if pattern in (LOAD, CASCODE):
            continue
        if via is not None and pattern in SEED_PATTERNS:
            continue
        if pattern == PASSIVE:
            continue

        for pin in ("S", "D"):
            net_a = devices[a]["nets"].get(pin)
            net_b = devices[b]["nets"].get(pin)
            if net_a is None or net_b is None:
                continue
            if net_a == net_b:
                # The two walks met on a shared net: shared-net terminal.
                continue
            cand_a = [name for name in _devices_on(net_index, net_a, pin)
                      if name not in (a, b)]
            cand_b = [name for name in _devices_on(net_index, net_b, pin)
                      if name not in (a, b)]
            for next_a in cand_a:
                for next_b in cand_b:
                    if next_a == next_b:
                        continue
                    next_key = (next_a, next_b) if next_a <= next_b else (next_b, next_a)
                    if next_key in visited:
                        continue
                    if pair_pattern(devices[next_a], devices[next_b]) == INVALID:
                        continue
                    stack.append((next_a, next_b, pin))

    return group


def find_motif_groups(subckt, stop_points=()):
    """Return motif symmetry groups as ``[{'seed': (a, b), 'pairs': [...]}]``.

    Groups are deterministic (sorted seeds, deterministic DFS order).  A seed
    that already appears in an earlier group is skipped, mirroring ConstGen's
    "invalidate consumed DFS sources" behaviour.
    """
    devices = collect_mos_devices(subckt)
    if not devices:
        return []
    net_index = _build_net_index(devices)
    seeds = _find_seeds(devices, net_index, set(stop_points or ()))

    groups = []
    claimed = set()
    for (a, b), _pattern in seeds:
        key = (a, b) if a <= b else (b, a)
        if key in claimed:
            logger.debug(f"motif seed {a},{b} already consumed by an earlier group")
            continue
        pairs = _expand_pair(devices, net_index, (a, b))
        if not pairs:
            continue
        for x, y in pairs:
            claimed.add((x, y) if x <= y else (y, x))
        groups.append({"seed": (a, b), "pairs": pairs})
        logger.debug(f"motif group from seed {a},{b}: {pairs}")
    return groups
