import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

# ----------------------- optional SB3 imports ---------------------------------
try:
    from stable_baselines3 import PPO
    _HAS_SB3 = True
except Exception:
    _HAS_SB3 = False

try:
    from sb3_contrib import TRPO
    _HAS_TRPO = True
except Exception:
    _HAS_TRPO = False


# ----------------------- helpers ----------------------------------------------
@dataclass
class EpisodeMetrics:
    """Accumulates physical-layer metrics for one episode."""
    steps: int = 0
    throughput_sum: float = 0.0
    success_count: int = 0
    tx_power_sum: float = 0.0
    mcs_cnt: Optional[Dict[int, int]] = None  # keys: 4, 16, 64

    def __post_init__(self):
        if self.mcs_cnt is None:
            self.mcs_cnt = {4: 0, 16: 0, 64: 0}

    def push(self, info: Dict[str, Any]):
        self.steps += 1
        self.throughput_sum += float(info.get("throughput", 0.0))
        self.success_count += 1 if info.get("success", False) else 0
        self.tx_power_sum += float(info.get("tx_power", 0.0))
        m = int(info.get("modulation", 4))
        if m in self.mcs_cnt:
            self.mcs_cnt[m] += 1

    def finalize(self) -> Dict[str, float]:
        T = max(1, self.steps)
        avg_throughput = self.throughput_sum / T           # bits/symbol
        per = 1.0 - (self.success_count / T)               # packet/block error rate
        avg_tx_power = self.tx_power_sum / T
        u4, u16, u64 = self.mcs_cnt[4] / T, self.mcs_cnt[16] / T, self.mcs_cnt[64] / T
        return dict(
            avg_throughput=avg_throughput,
            per=per,
            avg_tx_power=avg_tx_power,
            mcs_usage_qpsk=u4,
            mcs_usage_16qam=u16,
            mcs_usage_64qam=u64,
        )


# ----------------------- policy / critic --------------------------------------
class LinkAdaptationPolicy(nn.Module):
    """
    Actor backbone that outputs a categorical distribution over 3 modulations
    and a Gaussian over power in [0,1] (via sigmoid(mu)).
    """
    def __init__(self, state_dim: int, hidden: int = 128):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.mod_head = nn.Linear(hidden, 3)
        self.pwr_mu   = nn.Linear(hidden, 1)
        self.pwr_logstd = nn.Parameter(torch.tensor([-1.5]))  # exp() ~ 0.22

    def forward(self, s: torch.Tensor):
        z = self.feature(s)
        logits = self.mod_head(z)
        mu = torch.sigmoid(self.pwr_mu(z))
        std = torch.exp(self.pwr_logstd).clamp(0.05, 0.5)
        return logits, mu, std

    def sample(self, s_np: np.ndarray):
        s = torch.as_tensor(s_np, dtype=torch.float32).unsqueeze(0)
        logits, mu, std = self.forward(s)
        mod_dist = Categorical(logits=logits)
        mod = mod_dist.sample()
        pwr_dist = Normal(mu, std)
        p_raw = pwr_dist.sample()
        p = torch.clamp(p_raw, 0.0, 1.0)

        logp = mod_dist.log_prob(mod) + pwr_dist.log_prob(p).sum(dim=-1)
        action = {"modulation": int(mod.item()),
                  "power": p.detach().cpu().numpy().reshape(-1)}
        return action, logp.squeeze()

    def log_prob(self, s: torch.Tensor, action: Dict[str, Any]):
        logits, mu, std = self.forward(s)
        mod_dist = Categorical(logits=logits)
        lp_mod = mod_dist.log_prob(torch.as_tensor(action["modulation"], dtype=torch.int64))
        pwr_dist = Normal(mu, std)
        pwr = torch.as_tensor(action["power"], dtype=torch.float32).view_as(mu)
        lp_pwr = pwr_dist.log_prob(pwr).sum(dim=-1)
        return lp_mod + lp_pwr


class CriticV(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, s: torch.Tensor):
        return self.net(s).squeeze(-1)


# ------------------------------ QPPG Agent ------------------------------------
class QPPG:
    """
    Natural Actor-Critic variant:
      - Actor: Fisher-preconditioned policy gradient via CG + FVP
      - Critic: value baseline V(s) trained with MSE
    """
    def __init__(
        self,
        env,
        hidden=64,
        lr=1e-2,
        gamma=0.99,
        xi=0.5,
        max_grad_norm=10.0,
        fisher_samples=256,
        cg_iters=30,
        cg_tol=1e-6,
        precond_every=1,
        step_scale=0.5,
        entropy_coef=1e-2,
        value_coef=1.0,
        gae_lambda=0.95,
        entropy_coef_end=None,
        entropy_anneal_steps=0,
        entropy_anneal="none",
        log_every=50,
    ):
        self.env = env
        self.gamma = gamma
        self.entropy_coef_start = float(entropy_coef)
        self.entropy_coef_end   = float(entropy_coef if entropy_coef_end is None else entropy_coef_end)
        self.entropy_anneal_steps = int(entropy_anneal_steps)
        self.entropy_anneal = str(entropy_anneal)
        self.xi = xi
        self.gae_lambda = gae_lambda
        self.log_every = log_every

        sd = env.observation_space.shape[0]
        self.policy = LinkAdaptationPolicy(sd, hidden)
        self.critic = CriticV(sd, hidden)

        self.opt  = optim.Adam(self.policy.parameters(), lr=lr)
        self.optC = optim.Adam(self.critic.parameters(), lr=lr)

        self.max_grad_norm = max_grad_norm
        self.fisher_samples = fisher_samples
        self.cg_iters = cg_iters
        self.cg_tol = cg_tol
        self.precond_every = precond_every
        self.step_scale = step_scale

        self.value_coef = value_coef

    # --------------- CG/FVP utilities ----------------
    def _flatten_params(self):
        vec, shapes = [], []
        for p in self.policy.parameters():
            shapes.append(p.shape)
            vec.append(p.grad.view(-1) if p.grad is not None
                       else torch.zeros(p.numel(), dtype=torch.float32, device=p.device))
        return torch.cat(vec), shapes

    def _assign_flat_grad(self, flat_grad, shapes):
        idx = 0
        for p, sh in zip(self.policy.parameters(), shapes):
            n = int(np.prod(sh))
            p.grad = flat_grad[idx:idx+n].view(sh).clone()
            idx += n

    def _per_sample_grad(self, s_np, a_dict):
        s = torch.as_tensor(s_np, dtype=torch.float32).unsqueeze(0)
        lp = self.policy.log_prob(s, a_dict)
        self.policy.zero_grad(set_to_none=True)
        lp.backward()
        g = []
        for p in self.policy.parameters():
            g.append(torch.zeros(p.numel(), dtype=torch.float32, device=p.device)
                     if p.grad is None else p.grad.view(-1).detach())
        return torch.cat(g)

    def _fvp(self, v, states, actions):
        """Fisher-vector product: (1/M) Σ g_i (g_i^T v) + xi * v."""
        M = min(len(states), self.fisher_samples)
        if M == 0:
            return self.xi * v
        idxs = np.random.choice(len(states), size=M, replace=False)
        y = torch.zeros_like(v)
        for k in idxs:
            g_i = self._per_sample_grad(states[k], actions[k])
            y += g_i * torch.dot(g_i, v)
        y /= float(M)
        y += self.xi * v
        return y

    def _entropy_coef_at(self, ep_idx: int):
        if self.entropy_anneal_steps <= 0:
            return self.entropy_coef_start
        T = float(self.entropy_anneal_steps)
        if self.entropy_anneal == "linear":
            t = min(ep_idx, self.entropy_anneal_steps) / T
            return self.entropy_coef_start + t * (self.entropy_coef_end - self.entropy_coef_start)
        if self.entropy_anneal == "exp":
            t = min(ep_idx, self.entropy_anneal_steps) / T
            s, e = max(1e-12, self.entropy_coef_start), max(1e-12, self.entropy_coef_end)
            return s * (e / s) ** t
        return self.entropy_coef_start

    def _cg_solve(self, b, states, actions):
        """Solve (F+xi I) x = b using conjugate gradient with FVP oracle."""
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        rr_old = torch.dot(r, r)
        for _ in range(self.cg_iters):
            Ap = self._fvp(p, states, actions)
            denom = torch.dot(p, Ap) + 1e-12
            alpha = rr_old / denom
            x = x + alpha * p
            r = r - alpha * Ap
            rr_new = torch.dot(r, r)
            if rr_new.sqrt().item() < self.cg_tol:
                break
            beta = rr_new / (rr_old + 1e-12)
            p = r + beta * p
            rr_old = rr_new
        return x

    # -------------------- training --------------------
    def train(self, episodes=500, seed=None, log_every=50, return_metrics: bool = False):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        ep_rewards = []
        if return_metrics:
            ep_metrics_list: List[Dict[str, float]] = []

        for ep in range(episodes):
            s, _ = self.env.reset()
            done, total = False, 0.0
            traj = []  # (s, a, r, logp, info)
            epm = EpisodeMetrics()

            while not done:
                a, lp = self.policy.sample(s)
                s2, r, done, _, info = self.env.step(a)
                traj.append((s, a, r, lp, info))
                epm.push(info)
                s = s2
                total += r

            states, actions, rewards, logps, _infos = zip(*traj)
            S = torch.as_tensor(np.stack(states), dtype=torch.float32)

            # critic + GAE
            V_pred = self.critic(S)         # [T]
            V_det  = V_pred.detach()
            r = torch.tensor(rewards, dtype=torch.float32)
            T = len(r)
            A = torch.zeros(T, dtype=torch.float32)
            gae, gamma, lam = 0.0, self.gamma, self.gae_lambda
            for t in reversed(range(T)):
                v = V_det[t]
                v_next = V_det[t+1] if t+1 < T else torch.tensor(0.0)
                delta = r[t] + gamma * v_next - v
                gae = delta + gamma * lam * gae
                A[t] = gae
            A = (A - A.mean()) / (A.std() + 1e-8)
            R_lambda = A + V_det

            # actor loss w/ entropy
            logp_stack = torch.stack(list(logps))
            loss_actor = -(logp_stack * A).sum()
            logits_b, mu_b, std_b = self.policy.forward(S)
            cat = Categorical(logits=logits_b)
            ent = cat.entropy().mean() + torch.log(std_b).mean()
            cur_ec = self._entropy_coef_at(ep)
            loss_actor = loss_actor - cur_ec * ent

            # critic loss
            loss_critic = self.value_coef * (V_pred - R_lambda.detach()).pow(2).mean()

            # natural step
            self.opt.zero_grad(set_to_none=True)
            loss_actor.backward()
            g_flat, shapes = self._flatten_params()
            if (ep % self.precond_every) == 0:
                try:
                    x = self._cg_solve(g_flat, list(states), list(actions))
                    x = self.step_scale * x
                    self._assign_flat_grad(x, shapes)
                except Exception:
                    pass
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.opt.step()

            # critic step
            self.optC.zero_grad(set_to_none=True)
            loss_critic.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optC.step()

            # bookkeeping
            ep_rewards.append(total)
            if return_metrics:
                ep_metrics_list.append(epm.finalize())

            if (ep + 1) % self.log_every == 0:
                k = self.log_every
                avg_r = np.mean(ep_rewards[-k:])
                print(f"[QPPG-AC] ep {ep+1}: avg_reward={avg_r:.3f}")

        if return_metrics:
            metrics: Dict[str, List[float]] = {}
            for d in ep_metrics_list:
                for k, v in d.items():
                    metrics.setdefault(k, []).append(float(v))
            return ep_rewards, metrics

        return ep_rewards


# ------------------------------ Classical NPG ---------------------------------
class ClassicalNPG(QPPG):
    """Same implementation; kept as a named variant for comparisons."""
    pass


# ------------------------------ Simple Actor-Critic ---------------------------
class QAC:
    """Actor-Critic with TD(0) advantage baseline."""
    def __init__(self, env, hidden=128, lr_actor=2e-3, lr_critic=2e-3, gamma=0.99, entropy_coef=1e-3):
        self.env = env
        sd = env.observation_space.shape[0]
        self.actor = LinkAdaptationPolicy(sd, hidden)
        self.critic = CriticV(sd, hidden)
        self.optA = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optC = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.gamma = gamma
        self.entropy_coef = entropy_coef

    def _discounted_returns(self, rewards: List[float]):
        G, out = 0.0, []
        for r in reversed(rewards):
            G = r + self.gamma * G
            out.append(G)
        out.reverse()
        return torch.as_tensor(out, dtype=torch.float32)

    def train(self, episodes=500, seed=None, log_every=50, return_metrics: bool = False):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        ep_rewards = []
        if return_metrics:
            ep_metrics_list: List[Dict[str, float]] = []

        for ep in range(episodes):
            s, _ = self.env.reset()
            done = False
            traj = []  # (s, a, r, logp, info)
            total = 0.0
            epm = EpisodeMetrics()

            while not done:
                a, lp = self.actor.sample(s)
                s, r, done, _, info = self.env.step(a)
                traj.append((s, a, r, lp, info))
                epm.push(info)
                total += r

            states, actions, rewards, logps, _infos = zip(*traj)
            S = torch.as_tensor(np.stack(states), dtype=torch.float32)
            R = self._discounted_returns(list(rewards))
            V = self.critic(S).view(-1)

            # advantages
            A = (R - V.detach())
            A = (A - A.mean()) / (A.std() + 1e-8)

            # actor loss (entropy)
            logp_stack = torch.stack(list(logps))
            logits_b, mu_b, std_b = self.actor.forward(S)
            cat = Categorical(logits=logits_b)
            ent = cat.entropy().mean() + torch.log(std_b).mean()
            loss_actor = -(logp_stack * A).sum() - self.entropy_coef * ent

            # critic loss
            loss_critic = (V - R).pow(2).mean()

            # steps
            self.optA.zero_grad(set_to_none=True)
            loss_actor.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
            self.optA.step()

            self.optC.zero_grad(set_to_none=True)
            loss_critic.backward()
            self.optC.step()

            ep_rewards.append(total)
            if return_metrics:
                ep_metrics_list.append(epm.finalize())

            if (ep + 1) % log_every == 0:
                k = self.__dict__.get("log_every", log_every)
                print(f"[QAC] ep {ep+1}: avg_reward={np.mean(ep_rewards[-k:]):.3f}")

        if return_metrics:
            metrics: Dict[str, List[float]] = {}
            for d in ep_metrics_list:
                for k, v in d.items():
                    metrics.setdefault(k, []).append(float(v))
            return ep_rewards, metrics

        return ep_rewards


# ------------------------------ SB3 baselines ---------------------------------
def train_ppo_sb3(env, total_timesteps=100_000, seed=0):
    if not _HAS_SB3:
        print("[WARN] stable-baselines3 not found — skipping PPO baseline.")
        return None
    from stable_baselines3.common.env_util import make_vec_env
    vec_env = make_vec_env(lambda: env, n_envs=1, seed=seed)
    model = PPO("MlpPolicy", vec_env, verbose=1, seed=seed)
    model.learn(total_timesteps=total_timesteps)
    return model


def train_trpo_sb3(env, total_timesteps=100_000, seed=0):
    if not _HAS_TRPO:
        print("[WARN] sb3-contrib TRPO not found — skipping TRPO baseline.")
        return None
    from stable_baselines3.common.env_util import make_vec_env
    vec_env = make_vec_env(lambda: env, n_envs=1, seed=seed)
    model = TRPO("MlpPolicy", vec_env, verbose=1, seed=seed)
    model.learn(total_timesteps=total_timesteps)
    return model
