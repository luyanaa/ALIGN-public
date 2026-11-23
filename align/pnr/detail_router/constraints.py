"""Backend-agnostic constraint model for the detailed router.

The routing problem is expressed over boolean "is this candidate path
used" variables.  Both solver backends (SAT, MIP) consume this exact
model and only differ in how they encode and solve it.

Model
-----
* ``CandidatePath`` (see ``candidates.py``) -- a complete legal route
  between two pins of a net.
* Constraints:
    - ``CoverPins(net)``   -- every pin of the net must be touched by
      >= 1 selected path of that net.
    - ``Mutex(a, b)``      -- paths of *different* nets that share a
      grid cell may not both be selected.
    - ``Prefer(path, w)``  -- soft reward for selecting this path
      (e.g. it lies inside the wire-plan guide).  Used directly by the
      MIP backend; the SAT backend applies it as a post-pass tiebreak.
    - Connectivity is enforced lazily by ``check_connectivity`` /
      ``connectivity_cuts``: after each solve, if a net's selected
      paths do not form one connected component covering all its pins,
      a cut is added forcing at least one more path that bridges the
      disconnected parts.
"""

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class RegionProblem:
    """All candidate paths and constraints for one routing region."""

    def __init__(self):
        self.paths = []                  # list[CandidatePath]
        self.net_of = []                 # path idx -> net name
        self.paths_by_net = defaultdict(list)   # net -> [path idx]
        self.paths_touching_pin = defaultdict(list)  # (net,pin)->[idx]
        self.mutex = []                  # list[(idx_a, idx_b)]
        self.prefer = []                 # list[(idx, weight)]

    def add_candidates(self, net, cand_paths, n_pins):
        for c in cand_paths:
            idx = len(self.paths)
            self.paths.append(c)
            self.net_of.append(net)
            self.paths_by_net[net].append(idx)
            self.paths_touching_pin[(net, c.pin_a)].append(idx)
            self.paths_touching_pin[(net, c.pin_b)].append(idx)
        self._n_pins = getattr(self, '_n_pins', {})
        self._n_pins[net] = n_pins

    def n_pins(self, net):
        return self._n_pins.get(net, 0)

    def build_mutexes(self, same_net_ok=True):
        """Cell-sharing conflicts between paths of different nets."""
        cell_map = defaultdict(list)
        for i, p in enumerate(self.paths):
            for cell in p.cells:
                cell_map[cell].append(i)
        for cell, ids in cell_map.items():
            if len(ids) < 2:
                continue
            for a in range(len(ids)):
                for b in range(a + 1, len(ids)):
                    ia, ib = ids[a], ids[b]
                    if same_net_ok and self.net_of[ia] == self.net_of[ib]:
                        continue
                    self.mutex.append((ia, ib))

    # -- lazy connectivity ----------------------------------------------------
    def check_connectivity(self, selected):
        """``selected`` is a set of path indices.  Returns the list of
        nets whose selected paths do not connect all their pins."""
        bad = []
        for net, ids in self.paths_by_net.items():
            pins = self._n_pins.get(net, 0)
            if pins <= 1:
                continue
            sel = [i for i in ids if i in selected]
            if not sel:
                bad.append(net)
                continue
            # union-find over pins connected by selected paths
            parent = list(range(pins))

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            for i in sel:
                pa, pb = self.paths[i].pin_a, self.paths[i].pin_b
                ra, rb = find(pa), find(pb)
                if ra != rb:
                    parent[ra] = rb
            roots = {find(p) for p in range(pins)}
            if len(roots) > 1:
                bad.append(net)
        return bad

    def connectivity_cuts(self, selected):
        """Yield ``(net, [path indices])`` clauses: at least one of these
        additional paths must be selected to connect the net's forest.

        For each disconnected net we return every candidate path that
        bridges the current components (different root sets), expressed
        as an at-least-one constraint *of not-yet-selected* paths.
        """
        for net in self.check_connectivity(selected):
            pins = self._n_pins[net]
            sel = {i for i in self.paths_by_net[net] if i in selected}
            parent = list(range(pins))

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            for i in sel:
                p = self.paths[i]
                ra, rb = find(p.pin_a), find(p.pin_b)
                if ra != rb:
                    parent[ra] = rb
            bridge = [i for i in self.paths_by_net[net]
                      if i not in sel and
                      find(self.paths[i].pin_a) != find(self.paths[i].pin_b)]
            if bridge:
                yield (net, bridge)


def build_problem(region_candidates, n_pins=None):
    """``{net: [CandidatePath]}`` -> ``RegionProblem``.

    ``n_pins`` optionally supplies ``{net: pin_count}`` (needed when a
    net has pins no candidate path could reach)."""
    prob = RegionProblem()
    n_pins = n_pins or {}
    for net, cands in region_candidates.items():
        pins = max((max(c.pin_a, c.pin_b) for c in cands), default=-1) + 1
        pins = max(pins, n_pins.get(net, 0))
        prob.add_candidates(net, cands, pins)
    prob.build_mutexes()
    return prob
