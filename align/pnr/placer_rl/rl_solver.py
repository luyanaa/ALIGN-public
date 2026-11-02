"""PPO-based sequential placer (RALF's ``main_RL_placement.py`` equivalent).

Environment model: instances of one module are placed one at a time in a
random order.  An action is ``(x, y, rotation)`` on a coarse die grid.
Reward is given only at the end of an episode:
``reward = -(HPWL + sqrt(RUDY congestion))``, matching RALF's energy.

Requires the optional ``torch`` dependency.  Use ``SAPackingSolver`` when
torch is unavailable -- the driver selects it automatically.
"""

import logging
import math
import random

from .cost_model import HPWLMetric, RUDYCongestion, overlap_area

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None
    _HAS_TORCH = False

if _HAS_TORCH:
    _PolicyNetBase = nn.Module
else:  # pragma: no cover
    _PolicyNetBase = object


def torch_available():
    return _HAS_TORCH


class _PolicyNet(_PolicyNetBase):  # pragma: no cover - exercised via integration
    """Only defined when torch is available.

    Small MLP policy: per-instance feature vector + die summary -> action.

    Action head produces (x-bin, y-bin, rotation) categorical logits on a
    grid of ``grid`` x ``grid`` cells.  A single shared torso feeds three
    heads; this is deliberately simpler than RALF's GAT+D2RL stack and is a
    starting point -- the cost model and episode structure are the
    contributions that matter for parity.
    """

    N_FEAT = 8
    N_ROT = 2  # {N, S}: ALIGN leaf transforms encode flips only (sX, sY)

    def __init__(self, grid):
        if not _HAS_TORCH:
            raise RuntimeError('torch not installed')
        super().__init__()
        self.grid = grid
        hidden = 128
        self.torso = nn.Sequential(
            nn.Linear(self.N_FEAT, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.x_head = nn.Linear(hidden, grid)
        self.y_head = nn.Linear(hidden, grid)
        self.r_head = nn.Linear(hidden, self.N_ROT)
        self.v_head = nn.Linear(hidden, 1)

    def forward(self, feat):
        z = self.torso(feat)
        return self.x_head(z), self.y_head(z), self.r_head(z), self.v_head(z)


class RLEnvironment:
    """Sequential placement environment for one module.

    Coordinates are on a ``grid`` x ``grid`` lattice covering a square die
    whose side is the sum of the longest instance extents (RALF's sizing
    convention).  Instances may temporarily overlap; the final legalization
    step (compaction) is left to downstream stages -- for v1 we simply keep
    the best legal-ish placement seen and let the caller run overlap checks.
    """

    def __init__(self, problem, grid=32, overlap_penalty=1.0, seed=0):
        self.problem = problem
        self.grid = grid
        self.overlap_penalty = overlap_penalty
        self.rng = random.Random(seed)
        side = sum(max(i['width'], i['height']) for i in problem.instances)
        self.die = (side, side)
        self.hpwl = HPWLMetric(problem)
        self.rudy = RUDYCongestion(problem, self.hpwl)

    def reset(self):
        self.order = list(range(self.problem.n))
        self.rng.shuffle(self.order)
        self.placement = {}
        return self._features(self.order[0])

    def _features(self, inst_idx):
        inst = self.problem.instances[inst_idx]
        n_placed = len(self.placement)
        return [
            inst['width'] / self.die[0],
            inst['height'] / self.die[1],
            n_placed / self.problem.n,
            float(inst_idx) / self.problem.n,
            len([net for net in self.problem.nets
                 if any(p.get('inst') == inst_idx for p in net['pins'])])
            / max(len(self.problem.nets), 1),
            0.0, 0.0, 1.0,
        ]

    def step(self, inst_idx, x_bin, y_bin, rot):
        cell_w = self.die[0] / self.grid
        cell_h = self.die[1] / self.grid
        x = int(x_bin * cell_w)
        y = int(y_bin * cell_h)
        self.placement[inst_idx] = (x, y, rot)
        done = len(self.placement) == self.problem.n
        reward = 0.0
        if done:
            reward = -(self.hpwl(self.placement)
                       + math.sqrt(max(self.rudy(self.placement), 0.0))
                       + self.overlap_penalty
                       * overlap_area(self.placement, self.problem)
                       / max(self.die[0], 1))
        next_feat = None
        if not done:
            nxt = next(i for i in self.order if i not in self.placement)
            next_feat = self._features(nxt)
        return next_feat, reward, done


class RLPlacementSolver:
    """PPO over ``RLEnvironment``.  Returns the best-seen placement."""

    def __init__(self, problem, *, seed=0, n_episodes=200, grid=32,
                 lr=3e-4, gamma=0.99, clip=0.2, updates_per_iter=4):
        if not _HAS_TORCH:
            raise RuntimeError('torch not installed; use SAPackingSolver')
        self.problem = problem
        self.env = RLEnvironment(problem, grid=grid, seed=seed)
        self.n_episodes = n_episodes
        self.gamma = gamma
        self.clip = clip
        self.updates = updates_per_iter
        self.policy = _PolicyNet(grid)
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=lr)
        torch.manual_seed(seed)

    def _episode(self):
        obs = self.env.reset()
        traj = []
        total_reward = 0.0
        inst_idx = None
        placed = set()
        while True:
            inst_idx = next(i for i in self.env.order if i not in placed)
            feat = torch.tensor(obs, dtype=torch.float32)
            lx, ly, lr, v = self.policy(feat)
            dx = torch.distributions.Categorical(logits=lx)
            dy = torch.distributions.Categorical(logits=ly)
            dr = torch.distributions.Categorical(logits=lr)
            ax, ay, ar = dx.sample(), dy.sample(), dr.sample()
            logp = dx.log_prob(ax) + dy.log_prob(ay) + dr.log_prob(ar)
            obs, reward, done = self.env.step(inst_idx, int(ax), int(ay), int(ar))
            traj.append((feat, ax, ay, ar, logp, v, reward))
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
                            f'reward {reward:.2f}, best {best_reward:.2f}')
        if best_placement is not None:
            boxes = self.problem.placements_to_boxes(best_placement)
            bbox = (max(b[2] for b in boxes.values()),
                    max(b[3] for b in boxes.values()))
        else:
            bbox = None
        return best_placement, bbox, -best_reward

    def _update(self, traj):  # pragma: no cover
        rewards = [t[6] for t in traj]
        # discounted returns (all reward arrives at the end)
        R = 0.0
        returns = []
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)
        old_logp = torch.stack([t[4] for t in traj]).detach()
        values = torch.stack([t[5] for t in traj]).squeeze(-1)
        adv = returns - values.detach()
        if adv.numel() > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        for _ in range(self.updates):
            logps, vals = [], []
            for feat, ax, ay, ar, _, _, _ in traj:
                lx, ly, lr, v = self.policy(feat)
                lp = (torch.distributions.Categorical(logits=lx).log_prob(ax)
                      + torch.distributions.Categorical(logits=ly).log_prob(ay)
                      + torch.distributions.Categorical(logits=lr).log_prob(ar))
                logps.append(lp)
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
