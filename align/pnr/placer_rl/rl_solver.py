"""PPO-based sequential placer (RALF's ``main_RL_placement.py`` equivalent).

Environment model: instances of one module are placed one at a time in a
random order.  An action is ``(x, y, rotation)`` on a coarse die grid.

This module follows RALF's actual network structure more closely than the
earlier toy MLP:

* a small **bipartite GAT** encoder over the instance<->net incidence graph
  (hand-rolled additive attention, no torch_geometric/dgl), and
* a **deep RL decoder** (shared 2x128 MLP torso) with categorical action
  heads over the grid and a value head.

Reward is **densely shaped**: every step contributes the local change in
wirelength / congestion for the nets touching the just-placed instance plus
incremental overlap area, and the terminal step adds a symmetry penalty.

Requires the optional ``torch`` dependency.  Use ``SAPackingSolver`` when
torch is unavailable -- the driver selects it automatically.
"""

import logging
import math
import random
import re

from .cost_model import (HPWLMetric, RUDYCongestion, _strict_intersection_area,
                         raw_components, symmetry_penalty)
from . import legalizer

logger = logging.getLogger(__name__)

try:  # pragma: no cover - import guard
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    F = None
    _HAS_TORCH = False


# ---------------------------------------------------------------------------
# Device-type parsing (shared with ``driver.build_problem``)
# ---------------------------------------------------------------------------

# ralF-style leaf names look like ``NMOS_S_...``, ``DP_NMOS_B_...``,
# ``SCM_PMOS_...``, ``RES_...`` etc.  The first recognised token wins.
_DEV_RE = re.compile(r'NMOS|PMOS|CAP|RES|NPN|PNP|IND', re.IGNORECASE)
# The paper's categorical buckets; NPN/PNP/IND collapse to OTHER.
_DEV_CANON = {
    'NMOS': 'NMOS', 'PMOS': 'PMOS', 'CAP': 'CAP', 'RES': 'RES',
    'NPN': 'OTHER', 'PNP': 'OTHER', 'IND': 'OTHER',
}
DEV_TYPES = ['NMOS', 'PMOS', 'CAP', 'RES', 'OTHER']
_DEV_INDEX = {name: i for i, name in enumerate(DEV_TYPES)}

_POWER_RE = re.compile(r'vdd|vss|gnd|vcc|power|avdd|dvdd|avss|dvss|vph|vpn',
                       re.IGNORECASE)


def parse_device_type(name, fallback=''):
    """Map a concrete/abstract template name to a categorical device type.

    Searches ``name`` first, then ``fallback``, for one of
    ``NMOS|PMOS|CAP|RES|NPN|PNP|IND``.  Returns one of :data:`DEV_TYPES`
    (``OTHER`` when nothing matches).
    """
    for candidate in (name, fallback):
        if not candidate:
            continue
        m = _DEV_RE.search(candidate)
        if m:
            return _DEV_CANON[m.group(0).upper()]
    return 'OTHER'


def _is_power_net(name):
    return bool(name) and bool(_POWER_RE.search(name))


def torch_available():
    return _HAS_TORCH


# ---------------------------------------------------------------------------
# Torch modules (defined only when torch is present)
# ---------------------------------------------------------------------------

if _HAS_TORCH:

    def _segment_softmax(scores, index, n_nodes):
        """Softmax of ``scores`` grouped by ``index`` (dest node id)."""
        if scores.numel() == 0:
            return scores
        neg = scores.new_full((n_nodes,), -1e30)
        mx = neg.scatter_reduce(0, index, scores, reduce='amax',
                                include_self=True)
        ex = torch.exp(scores - mx[index])
        den = torch.zeros(n_nodes, dtype=scores.dtype, device=scores.device)
        den = den.index_add(0, index, ex)
        return ex / den[index].clamp_min(1e-12)

    class BipartiteGATLayer(nn.Module):
        """One round of masked additive attention in both directions.

        ``inst -> net``: each net attends over its incident instances.
        ``net -> inst``: each instance attends over its incident nets, using
        the already-updated net embeddings (symmetric message passing).
        Both directions use a residual connection followed by ReLU.
        """

        def __init__(self, hidden):
            super().__init__()
            self.i2n_src = nn.Linear(hidden, hidden, bias=False)
            self.i2n_dst = nn.Linear(hidden, hidden, bias=False)
            self.i2n_a = nn.Parameter(torch.zeros(hidden))
            self.n2i_src = nn.Linear(hidden, hidden, bias=False)
            self.n2i_dst = nn.Linear(hidden, hidden, bias=False)
            self.n2i_a = nn.Parameter(torch.zeros(hidden))
            self.norm_i = nn.LayerNorm(hidden)
            self.norm_n = nn.LayerNorm(hidden)

        def _attend(self, h_src, h_dst, edges, n_dst,
                    proj_src, proj_dst, attn_vec):
            src, dst = edges[0], edges[1]
            s = proj_src(h_src)[src]
            d = proj_dst(h_dst)[dst]
            score = F.leaky_relu((s + d) * attn_vec, negative_slope=0.2)
            score = score.sum(-1)
            alpha = _segment_softmax(score, dst, n_dst)
            msg = s * alpha.unsqueeze(-1)
            agg = torch.zeros(n_dst, s.size(-1), dtype=s.dtype,
                              device=s.device)
            agg = agg.index_add(0, dst, msg)
            return agg

        def forward(self, h_i, h_n, edges_i2n, edges_n2i):
            if edges_i2n.size(1) > 0 and h_n.size(0) > 0:
                agg_n = self._attend(h_i, h_n, edges_i2n, h_n.size(0),
                                     self.i2n_src, self.i2n_dst, self.i2n_a)
                h_n = F.relu(self.norm_n(h_n + agg_n))
            if edges_n2i.size(1) > 0 and h_i.size(0) > 0:
                agg_i = self._attend(h_n, h_i, edges_n2i, h_i.size(0),
                                     self.n2i_src, self.n2i_dst, self.n2i_a)
                h_i = F.relu(self.norm_i(h_i + agg_i))
            return h_i, h_n

    class _PolicyNet(nn.Module):
        """Bipartite-GAT encoder + deep decoder (RALF-style).

        Layout:
          * instance input = [4 continuous features | 8-d device embedding]
          * net input      = [degree_norm, is_power, is_sym_net]
          * ``gat_rounds`` (2) bipartite GAT rounds, 64-d hidden
          * graph embedding ``g`` = mean(h_i) + mean(h_n) (concatenated)
          * per-step state = [h_current, g, placed_frac, running_hpwl_norm]
          * torso = Linear(->128) ReLU Linear(->128) ReLU
          * heads = x_bin(grid), y_bin(grid), rot(2), value(1)

        Placement order is left random (RALF shuffles each episode); a
        learned pointer over unplaced instances is a natural later upgrade.
        """

        N_CONT = 4     # [w/die, h/die, degree_norm, is_sym]
        N_NET = 3      # [degree_norm, is_power, is_sym_net]
        N_ROT = 2      # {N, S}: ALIGN leaf transforms encode flips only
        N_DEV = len(DEV_TYPES)

        def __init__(self, grid, hidden=64, dev_emb=8, gat_rounds=2,
                     torso_hidden=128):
            super().__init__()
            self.grid = grid
            self.hidden = hidden
            self.dev_embed = nn.Embedding(self.N_DEV, dev_emb)
            self.inst_in = nn.Linear(self.N_CONT + dev_emb, hidden)
            self.net_in = nn.Linear(self.N_NET, hidden)
            self.gat = nn.ModuleList(
                [BipartiteGATLayer(hidden) for _ in range(gat_rounds)])
            torso_in = hidden + 2 * hidden + 2  # h_i + g(mean_i,mean_n) + prog
            self.torso = nn.Sequential(
                nn.Linear(torso_in, torso_hidden), nn.ReLU(),
                nn.Linear(torso_hidden, torso_hidden), nn.ReLU(),
            )
            self.x_head = nn.Linear(torso_hidden, grid)
            self.y_head = nn.Linear(torso_hidden, grid)
            self.r_head = nn.Linear(torso_hidden, self.N_ROT)
            self.v_head = nn.Linear(torso_hidden, 1)

        def encode(self, dev_ids, inst_cont, net_feat, edges_i2n, edges_n2i):
            x = torch.cat([self.dev_embed(dev_ids), inst_cont], dim=-1)
            h_i = F.relu(self.inst_in(x))
            h_n = F.relu(self.net_in(net_feat))
            for layer in self.gat:
                h_i, h_n = layer(h_i, h_n, edges_i2n, edges_n2i)
            pool_i = h_i.mean(dim=0) if h_i.size(0) > 0 else \
                h_i.new_zeros(self.hidden)
            pool_n = h_n.mean(dim=0) if h_n.size(0) > 0 else \
                h_n.new_zeros(self.hidden)
            g = torch.cat([pool_i, pool_n], dim=-1)
            return h_i, g

        def decode(self, h_i, g, inst_idx, placed_frac, hpwl_norm):
            prog = torch.tensor([placed_frac, hpwl_norm],
                                dtype=h_i.dtype, device=h_i.device)
            state = torch.cat([h_i[inst_idx], g, prog], dim=-1)
            z = self.torso(state)
            return (self.x_head(z), self.y_head(z), self.r_head(z),
                    self.v_head(z))


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class RLEnvironment:
    """Sequential placement environment for one module.

    Coordinates are on a ``grid`` x ``grid`` lattice covering a square die
    whose side is the sum of the longest instance extents (RALF's sizing
    convention).  Instances may temporarily overlap; the final legalization
    step (compaction) is left to downstream stages -- the caller runs overlap
    checks on the returned placement.

    Reward per step (dense shaping)::

        r_t = -(d_hpwl_local + d_sqrt(rudy_local))
              - overlap_inc * overlap_penalty
              [ - w_sym * symmetry_penalty(placement, problem) at terminal ]

    where ``*_local`` only recomputes the nets touching the just-placed
    instance (net bboxes are cached incrementally, so the rest of the netlist
    is never recomputed) and ``overlap_inc`` is the strict overlap area the
    new instance introduces against already-placed instances.
    """

    def __init__(self, problem, grid=32, overlap_penalty=1.0, seed=0):
        self.problem = problem
        self.grid = grid
        self.overlap_penalty = overlap_penalty
        self.rng = random.Random(seed)
        side = sum(max(i['width'], i['height']) for i in problem.instances)
        side = max(side, 1)
        self.die = (side, side)
        self.hpwl = HPWLMetric(problem)
        self.rudy = RUDYCongestion(problem, self.hpwl)

        # incidence lists (net <-> instance) used for incremental updates
        self.inst_nets = [[] for _ in range(problem.n)]
        for ni, net in enumerate(problem.nets):
            for pin in net['pins']:
                if 'inst' in pin:
                    idx = pin['inst']
                    if 0 <= idx < problem.n:
                        self.inst_nets[idx].append(ni)

        # magnitude-matched symmetry weight, exactly like ``sa_solver``:
        # w_sym = hpwl0 / max(raw_sym0, eps) on a trivial initial placement
        trivial = {i: (0, 0, 0) for i in range(problem.n)}
        self.hpwl0 = self.hpwl(trivial) if problem.n else 0.0
        self.raw_sym0 = 0.0
        if getattr(problem, 'symmetry', None):
            self.raw_sym0 = symmetry_penalty(trivial, problem)
            eps = 1e-9 * max(abs(self.hpwl0), 1.0)
            if self.raw_sym0 > eps:
                self.w_sym = self.hpwl0 / self.raw_sym0
            else:  # already satisfied at init: unit-scale guard
                self.w_sym = max(abs(self.hpwl0), 1.0)
        else:
            self.w_sym = 0.0

    def reset(self):
        self.order = list(range(self.problem.n))
        self.rng.shuffle(self.order)
        self.placement = {}
        self.boxes = {}
        # cached per-net (density, bbox) for incremental congestion
        self.net_state = [None] * len(self.problem.nets)
        self.hpwl_accum = 0.0
        self.last_reward = 0.0
        return self.order[0] if self.order else None

    # -- incremental helpers -------------------------------------------------
    def _net_bbox(self, net, boxes):
        """Net bbox over the pins whose instance is already placed."""
        llx = lly = urx = ury = None
        for pin in net['pins']:
            if 'fixed' in pin:
                x0, y0 = pin['fixed']
                r = (x0, y0, x0, y0)
            else:
                idx = pin['inst']
                if idx not in boxes:
                    continue
                x0, y0, x1, y1 = boxes[idx]
                r = pin['rect'](boxes[idx]) if 'rect' in pin else (x0, y0, x1, y1)
            llx = r[0] if llx is None else min(llx, r[0])
            lly = r[1] if lly is None else min(lly, r[1])
            urx = r[2] if urx is None else max(urx, r[2])
            ury = r[3] if ury is None else max(ury, r[3])
        return None if llx is None else (llx, lly, urx, ury)

    @staticmethod
    def _net_hpwl(state):
        if state is None:
            return 0.0
        b = state[1]
        return (b[2] - b[0]) + (b[3] - b[1])

    def _make_net_state(self, net, boxes):
        b = self._net_bbox(net, boxes)
        if b is None:
            return None
        w = (b[2] - b[0]) + 1
        h = (b[3] - b[1]) + 1
        d = (w + h) / (w * h)  # RALF p=1 convention
        return (d, b)

    def _local_rudy(self, affected):
        """RUDY contribution of all ordered net pairs touching ``affected``.

        Nets outside ``affected`` keep their cached state (updated only when
        one of their instances is placed), so the full netlist is never
        recomputed here.
        """
        affected = set(affected)
        total = 0.0
        for i, si in enumerate(self.net_state):
            if si is None:
                continue
            di, bi = si
            for j, sj in enumerate(self.net_state):
                if i == j or sj is None:
                    continue
                if i not in affected and j not in affected:
                    continue
                dj, bj = sj
                total += dj * RUDYCongestion._intersection_area(bi, bj)
        return total

    # -- stepping ------------------------------------------------------------
    def step(self, inst_idx, x_bin, y_bin, rot):
        cell_w = self.die[0] / self.grid
        cell_h = self.die[1] / self.grid
        x = int(x_bin * cell_w)
        y = int(y_bin * cell_h)
        inst = self.problem.instances[inst_idx]
        new_box = (x, y, x + inst['width'], y + inst['height'])

        # incremental overlap introduced by this placement
        overlap_inc = 0.0
        for other in self.boxes.values():
            overlap_inc += _strict_intersection_area(new_box, other)

        affected = list(self.inst_nets[inst_idx])
        hpwl_before = sum(self._net_hpwl(self.net_state[ni])
                          for ni in affected)
        rudy_before = self._local_rudy(affected)

        # commit placement and refresh only the affected net states
        self.placement[inst_idx] = (x, y, rot)
        self.boxes[inst_idx] = new_box
        for ni in affected:
            self.net_state[ni] = self._make_net_state(
                self.problem.nets[ni], self.boxes)

        hpwl_after = sum(self._net_hpwl(self.net_state[ni])
                         for ni in affected)
        rudy_after = self._local_rudy(affected)

        hpwl_delta = hpwl_after - hpwl_before
        self.hpwl_accum += hpwl_delta
        reward = (-(hpwl_delta
                    + math.sqrt(max(rudy_after, 0.0))
                    - math.sqrt(max(rudy_before, 0.0)))
                  - overlap_inc * self.overlap_penalty)

        done = len(self.placement) == self.problem.n
        if done and self.w_sym:
            reward -= self.w_sym * symmetry_penalty(self.placement,
                                                    self.problem)
        self.last_reward = reward
        return None, reward, done


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class RLPlacementSolver:
    """PPO over ``RLEnvironment``.  Returns the best-seen placement."""

    def __init__(self, problem, *, seed=0, n_episodes=200, grid=32,
                 lr=3e-4, gamma=0.99, clip=0.2, updates_per_iter=4,
                 temperature=1.0):
        if not _HAS_TORCH:
            raise RuntimeError('torch not installed; use SAPackingSolver')
        self.problem = problem
        self.env = RLEnvironment(problem, grid=grid, seed=seed)
        self.n_episodes = n_episodes
        self.gamma = gamma
        self.clip = clip
        self.updates = updates_per_iter
        self.temperature = max(float(temperature), 1e-6)
        torch.manual_seed(seed)
        self.policy = _PolicyNet(grid)
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.hpwl_scale = max(self.env.hpwl0, self.env.die[0], 1.0)
        self._build_graph()

    # -- graph inputs --------------------------------------------------------
    def _build_graph(self):
        """Tensorise the static bipartite incidence graph (once)."""
        problem = self.problem
        n = problem.n
        m = len(problem.nets)
        sym_group = {}
        for gi, group in enumerate(getattr(problem, 'symmetry', None) or []):
            for pair in group['pairs']:
                for idx in pair:
                    sym_group[idx] = gi
        deg = [0] * n
        for net in problem.nets:
            for pin in net['pins']:
                if 'inst' in pin:
                    deg[pin['inst']] += 1

        dev_ids = []
        cont = []
        for idx, inst in enumerate(problem.instances):
            dt = inst.get('dev_type')
            if dt is None:
                dt = parse_device_type(inst.get('concrete_template_name', ''),
                                       inst.get('abstract_template_name', ''))
            dev_ids.append(_DEV_INDEX.get(dt, _DEV_INDEX['OTHER']))
            gidx = inst.get('sym_group', sym_group.get(idx, -1))
            cont.append([
                inst['width'] / self.env.die[0],
                inst['height'] / self.env.die[1],
                deg[idx] / max(len(problem.nets), 1),
                1.0 if gidx is not None and gidx >= 0 else 0.0,
            ])

        net_feat = []
        n_incidence = []
        for ni, net in enumerate(problem.nets):
            insts = [p['inst'] for p in net['pins'] if 'inst' in p]
            n_incidence.append(insts)
            net_feat.append([
                len(insts) / max(n, 1),
                1.0 if _is_power_net(net.get('name', '')) else 0.0,
                0.0,  # is_sym_net placeholder
            ])

        src, dst = [], []
        for ni, insts in enumerate(n_incidence):
            for ii in insts:
                src.append(ii)
                dst.append(ni)
        if src:
            edges_i2n = torch.tensor([src, dst], dtype=torch.long)
        else:
            edges_i2n = torch.zeros((2, 0), dtype=torch.long)
        edges_n2i = edges_i2n.flip(0)

        self.dev_ids = torch.tensor(dev_ids, dtype=torch.long)
        self.inst_cont = torch.tensor(cont, dtype=torch.float32)
        if net_feat:
            self.net_feat = torch.tensor(net_feat, dtype=torch.float32)
        else:
            self.net_feat = torch.zeros((0, _PolicyNet.N_NET),
                                        dtype=torch.float32)
        self.edges_i2n = edges_i2n
        self.edges_n2i = edges_n2i

    def _encode(self):
        return self.policy.encode(self.dev_ids, self.inst_cont, self.net_feat,
                                  self.edges_i2n, self.edges_n2i)

    # -- episode -------------------------------------------------------------
    @staticmethod
    def _cat_logp(lx, ly, lr, ax, ay, ar):
        return (torch.distributions.Categorical(logits=lx).log_prob(ax)
                + torch.distributions.Categorical(logits=ly).log_prob(ay)
                + torch.distributions.Categorical(logits=lr).log_prob(ar))

    def _episode(self):
        self.env.reset()
        h_i, g = self._encode()
        placed = set()
        traj = []
        total_reward = 0.0
        n = self.problem.n
        temp = self.temperature
        while len(placed) < n:
            inst_idx = next(i for i in self.env.order if i not in placed)
            pf = len(placed) / n
            hn = self.env.hpwl_accum / self.hpwl_scale
            lx, ly, lr, v = self.policy.decode(h_i, g, inst_idx, pf, hn)
            lx, ly, lr = lx / temp, ly / temp, lr / temp
            dx = torch.distributions.Categorical(logits=lx)
            dy = torch.distributions.Categorical(logits=ly)
            dr = torch.distributions.Categorical(logits=lr)
            ax, ay, ar = dx.sample(), dy.sample(), dr.sample()
            logp = self._cat_logp(lx, ly, lr, ax, ay, ar)
            _, reward, done = self.env.step(inst_idx, int(ax), int(ay),
                                            int(ar))
            traj.append((inst_idx, pf, hn, ax, ay, ar, logp, v, reward))
            placed.add(inst_idx)
            total_reward += reward
            if done:
                break
        return traj, total_reward

    def solve(self):
        best_placement = None
        best_reward = float('-inf')
        for ep in range(self.n_episodes):
            traj, reward = self._episode()
            if reward > best_reward:
                best_reward = reward
                best_placement = dict(self.env.placement)
            self._update(traj)
            if ep % 50 == 0:
                logger.info(f'RL placer ({self.problem.name}): episode {ep}, '
                            f'return {reward:.2f}, best {best_reward:.2f}')
        if best_placement is not None:
            # The RL policy scatters instances on a coarse grid, so the raw
            # output overlaps heavily and ignores the module's hard symmetry
            # constraints.  Turn it into a routable floorplan with the shared
            # legaliser: pair-locked symmetry snapping + de-overlap (or plain
            # de-overlap when the module has no symmetry).
            if getattr(self.problem, 'symmetry', None):
                best_placement, _ = legalizer.legalize(self.problem,
                                                       best_placement)
            else:
                best_placement = legalizer.deoverlap(best_placement,
                                                     self.problem)
            boxes = self.problem.placements_to_boxes(best_placement)
            bbox = (max(b[2] for b in boxes.values()),
                    max(b[3] for b in boxes.values()))
            # Re-score the legalised placement with the shared cost model.
            parts = raw_components(best_placement, self.problem)
            energy = (parts.hpwl
                      + math.sqrt(max(parts.congestion, 0.0))
                      + parts.overlap)
        else:
            bbox = None
            energy = -best_reward
        return best_placement, bbox, energy

    def _update(self, traj):  # pragma: no cover - exercised via solve()
        if not traj:
            return
        rewards = [t[8] for t in traj]
        R = 0.0
        returns = []
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)
        old_logp = torch.stack([t[6] for t in traj]).detach()
        values = torch.stack([t[7] for t in traj]).squeeze(-1).detach()
        adv = returns - values
        if adv.numel() > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        temp = self.temperature
        for _ in range(self.updates):
            h_i, g = self._encode()
            logps, vals = [], []
            for (inst_idx, pf, hn, ax, ay, ar, _, _, _) in traj:
                lx, ly, lr, v = self.policy.decode(h_i, g, inst_idx, pf, hn)
                lx, ly, lr = lx / temp, ly / temp, lr / temp
                logps.append(self._cat_logp(lx, ly, lr, ax, ay, ar))
                vals.append(v)
            logps = torch.stack(logps)
            vals = torch.stack(vals).squeeze(-1)
            ratio = torch.exp(logps - old_logp)
            s1 = ratio * adv
            s2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv
            loss = (-torch.min(s1, s2).mean()
                    + 0.5 * (returns - vals).pow(2).mean())
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
