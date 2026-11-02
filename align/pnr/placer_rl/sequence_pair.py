"""Sequence-pair representation and O(n^2) decode to a packing floorplan.

Sequence-pair semantics (Murata et al., IEEE TCAD 1996):
    ``(i before j in G+) and (i before j in G-)``  =>  i is left of j
    ``(i after  j in G+) and (i before j in G-)``  =>  i is above j

Decode uses the oblique-grid construction with longest-path evaluation over
the horizontal/vertical constraint DAGs, matching the approach used by
RALF (itself based on the rectangle-packing-solver by Kotaro Terada,
Apache-2.0).  Implemented here with a plain topological pass to avoid a
graphlib dependency per decode call.
"""

from collections import defaultdict, deque


class SequencePair:
    def __init__(self, gp, gn):
        assert len(gp) == len(gn)
        assert sorted(gp) == sorted(gn) == list(range(len(gp)))
        self.gp = list(gp)
        self.gn = list(gn)
        self.n = len(gp)
        # position of each id in each sequence
        pos_p = [0] * self.n
        pos_n = [0] * self.n
        for i, v in enumerate(self.gp):
            pos_p[v] = i
        for i, v in enumerate(self.gn):
            pos_n[v] = i
        self.pos_p = pos_p
        self.pos_n = pos_n

    def decode(self, widths, heights, rotations=None):
        """Decode to absolute positions.

        Args:
            widths, heights: per-rectangle extents (unrotated).
            rotations: optional per-rectangle orientation; odd values swap
                width/height.

        Returns:
            (positions, bounding_box) where ``positions[i] = (x, y)`` of the
            lower-left corner and ``bounding_box = (W, H)``.
        """
        n = self.n
        w = list(widths)
        h = list(heights)
        if rotations is not None:
            for i in range(n):
                if rotations[i] % 2 == 1:
                    w[i], h[i] = h[i], w[i]

        # Horizontal constraint graph: edge i->j when j is right of i,
        # i.e. pos_p[i] < pos_p[j] and pos_n[i] < pos_n[j].
        # Longest path gives right-edges; x = dist[i] - w[i].
        edges_h = defaultdict(list)
        indeg_h = [0] * n
        for i in range(n):
            for j in range(n):
                if self.pos_p[i] < self.pos_p[j] and self.pos_n[i] < self.pos_n[j]:
                    edges_h[i].append(j)
                    indeg_h[j] += 1

        dist_h = list(w)
        queue = deque(i for i in range(n) if indeg_h[i] == 0)
        while queue:
            i = queue.popleft()
            for j in edges_h[i]:
                if dist_h[j] < dist_h[i] + w[j]:
                    dist_h[j] = dist_h[i] + w[j]
                indeg_h[j] -= 1
                if indeg_h[j] == 0:
                    queue.append(j)

        # Vertical constraint graph: edge i->j when j is above i,
        # i.e. pos_p[i] > pos_p[j] and pos_n[i] < pos_n[j].
        edges_v = defaultdict(list)
        indeg_v = [0] * n
        for i in range(n):
            for j in range(n):
                if self.pos_p[i] > self.pos_p[j] and self.pos_n[i] < self.pos_n[j]:
                    edges_v[i].append(j)
                    indeg_v[j] += 1

        dist_v = list(h)
        queue = deque(i for i in range(n) if indeg_v[i] == 0)
        while queue:
            i = queue.popleft()
            for j in edges_v[i]:
                if dist_v[j] < dist_v[i] + h[j]:
                    dist_v[j] = dist_v[i] + h[j]
                indeg_v[j] -= 1
                if indeg_v[j] == 0:
                    queue.append(j)

        positions = [(dist_h[i] - w[i], dist_v[i] - h[i]) for i in range(n)]
        return positions, (max(dist_h), max(dist_v))

    def perturb(self, rng):
        """Random sequence-pair perturbation: swap two ids in one sequence
        or move one id to a random position (same move set as RALF/SSP)."""
        which = rng.randrange(2)
        seq = self.gp if which == 0 else self.gn
        n = len(seq)
        if rng.random() < 0.5:
            i, j = rng.randrange(n), rng.randrange(n)
            seq[i], seq[j] = seq[j], seq[i]
        else:
            i, j = rng.randrange(n), rng.randrange(n)
            v = seq.pop(i)
            seq.insert(j, v)
        return SequencePair(self.gp, self.gn)
