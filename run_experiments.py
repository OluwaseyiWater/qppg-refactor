import argparse, os, csv, json, time, numpy as np
from typing import Dict, Any, List, Tuple
from environment import RayleighFadingChannelEnv, RayleighSB3Wrapper
from agents import QPPG, ClassicalNPG, QAC, train_ppo_sb3, train_trpo_sb3

SCENARIOS = {
    "s1_baseline":      dict(n_antennas=4, pilot_snr_db=10.0, noise_uncertainty_db=0.0),
    "s2_high_dim":      dict(n_antennas=8, pilot_snr_db=10.0, noise_uncertainty_db=0.0),
    "s3_low_csi":       dict(n_antennas=4, pilot_snr_db=5.0,  noise_uncertainty_db=0.0),
    "s4_noise_uncert":  dict(n_antennas=4, pilot_snr_db=10.0, noise_uncertainty_db=5.0),
    "s5_combined":      dict(n_antennas=8, pilot_snr_db=10.0, noise_uncertainty_db=5.0),
}

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_curve(path: str, rewards: List[float]):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "reward"])
        for i, r in enumerate(rewards):
            w.writerow([i + 1, float(r)])

def save_metrics_table(path: str, metrics: Dict[str, List[float]]):
    ensure_dir(os.path.dirname(path))
    keys = sorted(metrics.keys())
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode"] + keys)
        T = max(len(v) for v in metrics.values()) if metrics else 0
        for t in range(T):
            row = [t + 1]
            for k in keys:
                vals = metrics.get(k, [])
                row.append(float(vals[t]) if t < len(vals) else "")
            w.writerow(row)

def run_single_agent(env_kwargs: Dict[str, Any], agent_name: str, episodes: int, seed: int,
                     out_dir: str, agent_kwargs: Dict[str, Any] = None, sb3_factor: int = 50):
    agent_kwargs = agent_kwargs or {}
    env = RayleighFadingChannelEnv(max_steps=20, **env_kwargs)
    steps_per_ep = getattr(env, "max_steps", 20)
    start = time.time()

    meta = {
        "agent": agent_name,
        "seed": seed,
        "env_kwargs": env_kwargs,
        "episodes": episodes,
        "steps_per_episode": steps_per_ep,
        "timestamp_start": start,
    }

    rewards: List[float] = []
    metrics: Dict[str, List[float]] = {}

    if agent_name == "QPPG":
        agent = QPPG(env, **agent_kwargs)
        rewards, metrics = agent.train(episodes=episodes, seed=seed, return_metrics=True)
        meta.update({"xi": agent_kwargs.get("xi", 0.5),
                     "fisher_samples": agent_kwargs.get("fisher_samples", 256),
                     "cg_iters": agent_kwargs.get("cg_iters", 30),
                     "precond_every": agent_kwargs.get("precond_every", 1)})

    elif agent_name == "NPG":
        agent = ClassicalNPG(env, xi=agent_kwargs.get("xi", 0.5),
                             lr=agent_kwargs.get("lr", 1e-2),
                             fisher_samples=agent_kwargs.get("fisher_samples", 256),
                             cg_iters=agent_kwargs.get("cg_iters", 30))
        rewards, metrics = agent.train(episodes=episodes, seed=seed, return_metrics=True)
        meta.update({"xi": agent_kwargs.get("xi", 0.5)})

    elif agent_name == "QAC":
        agent = QAC(env)
        rewards, metrics = agent.train(episodes=episodes, seed=seed, return_metrics=True)

    elif agent_name == "PPO":
        env_sb3 = RayleighSB3Wrapper(env)
        total_timesteps = episodes * steps_per_ep * sb3_factor
        model = train_ppo_sb3(env_sb3, total_timesteps=total_timesteps, seed=seed)
        rewards = []
        if model is not None:
            for ep in range(episodes):
                obs, _ = env_sb3.reset(seed=seed + ep)
                done = False
                tot = 0.0
                while not done:
                    act, _ = model.predict(obs, deterministic=False)
                    obs, r, done, _, _ = env_sb3.step(int(act))
                    tot += r
                rewards.append(tot)
        meta.update({"sb3_total_timesteps": total_timesteps, "sb3_factor": sb3_factor})

    elif agent_name == "TRPO":
        env_sb3 = RayleighSB3Wrapper(env)
        total_timesteps = episodes * steps_per_ep * sb3_factor
        model = train_trpo_sb3(env_sb3, total_timesteps=total_timesteps, seed=seed)
        rewards = []
        if model is not None:
            for ep in range(episodes):
                obs, _ = env_sb3.reset(seed=seed + ep)
                done = False
                tot = 0.0
                while not done:
                    act, _ = model.predict(obs, deterministic=False)
                    obs, r, done, _, _ = env_sb3.step(int(act))
                    tot += r
                rewards.append(tot)
        meta.update({"sb3_total_timesteps": total_timesteps, "sb3_factor": sb3_factor})

    else:
        raise ValueError(f"Unknown agent: {agent_name}")

    dur = time.time() - start
    meta["duration_sec"] = dur
    meta["timestamp_end"] = time.time()

    # Save artifacts
    curve_path   = os.path.join(out_dir, f"{agent_name}_seed{seed}.csv")
    meta_path    = os.path.join(out_dir, f"{agent_name}_seed{seed}.json")
    metrics_path = os.path.join(out_dir, f"{agent_name}_seed{seed}_metrics.csv")

    save_curve(curve_path, rewards)
    if metrics:
        save_metrics_table(metrics_path, metrics)

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return rewards

def run_scenarios(which: List[str], episodes: int, seeds: int, out_root: str, sb3_factor: int):
    AGENTS = ["QPPG", "NPG", "QAC", "PPO", "TRPO"]

    # optional sweep of QPPG entropy start
    sweep = []
    if hasattr(args_ns, "qppg_entropy_sweep") and args_ns.qppg_entropy_sweep:
        sweep = [float(x) for x in args_ns.qppg_entropy_sweep.split(",") if x.strip()]

    for sc_key in which:
        env_kwargs = SCENARIOS[sc_key]
        out_dir = os.path.join(out_root, sc_key)
        ensure_dir(out_dir)
        print(f"\n=== Scenario {sc_key}: {env_kwargs} ===")
        for a in AGENTS:
            for s in range(seeds):
                seed = 10_000 + 97 * s
                if a == "QPPG" and sweep:
                    for ec in sweep:
                        agent_dir = os.path.join(out_dir, f"{a}_ec{ec:g}")
                        ensure_dir(agent_dir)
                        print(f" -> {a}, seed={seed}, entropy_start={ec}")
                        agent_kwargs = dict(
                            xi=args_ns.qppg_xi,
                            lr=args_ns.qppg_lr,
                            step_scale=args_ns.qppg_step_scale,
                            fisher_samples=args_ns.qppg_fisher_samples,
                            cg_iters=args_ns.qppg_cg_iters,
                            entropy_coef=ec,
                            entropy_coef_end=(args_ns.qppg_entropy_end if args_ns.qppg_entropy_end is not None else ec),
                            entropy_anneal_steps=args_ns.qppg_entropy_steps,
                            entropy_anneal=args_ns.qppg_entropy_anneal,
                            value_coef=args_ns.qppg_value_coef,
                            gae_lambda=args_ns.qppg_gae_lambda,
                        )
                        run_single_agent(env_kwargs, a, episodes, seed, agent_dir, agent_kwargs, sb3_factor)
                else:
                    agent_dir = os.path.join(out_dir, a)
                    ensure_dir(agent_dir)
                    print(f" -> {a}, seed={seed}")
                    agent_kwargs = {}
                    if a == "QPPG":
                        agent_kwargs = dict(
                            xi=args_ns.qppg_xi,
                            lr=args_ns.qppg_lr,
                            step_scale=args_ns.qppg_step_scale,
                            fisher_samples=args_ns.qppg_fisher_samples,
                            cg_iters=args_ns.qppg_cg_iters,
                            entropy_coef=args_ns.qppg_entropy_start,
                            entropy_coef_end=(args_ns.qppg_entropy_end
                                              if args_ns.qppg_entropy_end is not None
                                              else args_ns.qppg_entropy_start),
                            entropy_anneal_steps=args_ns.qppg_entropy_steps,
                            entropy_anneal=args_ns.qppg_entropy_anneal,
                            value_coef=args_ns.qppg_value_coef,
                            gae_lambda=args_ns.qppg_gae_lambda,
                        )
                    run_single_agent(env_kwargs, a, episodes, seed, agent_dir, agent_kwargs, sb3_factor)

def run_ablation_xi(episodes: int, seeds: int, out_root: str):
    """Run xi ablation ONLY on s1_baseline."""
    xis = [0.001, 0.01, 0.1, 0.5, 1.0]
    env_kwargs = dict(n_antennas=4, pilot_snr_db=10.0, noise_uncertainty_db=0.0)
    out_dir = os.path.join(out_root, "ablation")
    ensure_dir(out_dir)
    csv_path = os.path.join(out_dir, "xi_sweep.csv")

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["xi", "seed", "final_avg_reward"])
        for xi in xis:
            for s in range(seeds):
                seed = 20_000 + 127 * s
                env = RayleighFadingChannelEnv(max_steps=20, **env_kwargs)
                agent = QPPG(env, xi=xi, fisher_samples=128, cg_iters=20, precond_every=1,
                             entropy_coef=1e-3, value_coef=0.5, hidden=64)
                rew = agent.train(episodes=episodes, seed=seed)
                k = min(25, len(rew))
                avg_final = float(np.mean(rew[-k:]))
                w.writerow([xi, seed, avg_final])
                print(f"[Ablation] xi={xi}, seed={seed}, final_avg_reward={avg_final:.3f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", type=str, default="all", help="comma list or 'all'")
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--seeds", type=int, default=15)
    ap.add_argument("--out", type=str, default="results")
    ap.add_argument("--ablation_only", action="store_true")
    ap.add_argument("--sb3_factor", type=int, default=50, help="Multiplier for SB3 total_timesteps per episode")

    # QPPG knobs
    ap.add_argument("--qppg_entropy_start", type=float, default=1e-3)
    ap.add_argument("--qppg_entropy_end", type=float, default=None)
    ap.add_argument("--qppg_entropy_steps", type=int, default=0)
    ap.add_argument("--qppg_entropy_anneal", type=str, default="none", choices=["none","linear","exp"])
    ap.add_argument("--qppg_entropy_sweep", type=str, default="")
    ap.add_argument("--qppg_value_coef", type=float, default=0.5)
    ap.add_argument("--qppg_gae_lambda", type=float, default=0.95)
    # extras we added
    ap.add_argument("--qppg_lr", type=float, default=1e-2)
    ap.add_argument("--qppg_step_scale", type=float, default=0.5)
    ap.add_argument("--qppg_fisher_samples", type=int, default=256)
    ap.add_argument("--qppg_cg_iters", type=int, default=30)
    ap.add_argument("--qppg_xi", type=float, default=0.5)

    args = ap.parse_args()
    global args_ns
    args_ns = args

    out_root = args.out
    os.makedirs(out_root, exist_ok=True)

    if args.ablation_only:
        run_ablation_xi(args.episodes, args.seeds, out_root)
        return

    which = list(SCENARIOS.keys()) if args.scenarios == "all" else [s.strip() for s in args.scenarios.split(",")]
    run_scenarios(which, args.episodes, args.seeds, out_root, args.sb3_factor)

    if "s1_baseline" in which:
        run_ablation_xi(args.episodes, args.seeds, out_root)

if __name__ == "__main__":
    main()
