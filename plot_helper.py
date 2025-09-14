# plot_helper.py
import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

AGENTS = ["NPG", "PPO", "QAC", "QPPG", "TRPO"]

def _collect_runs(result_dir: str, scenario: str, agent: str) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    """
    Return (reward_dfs, metric_dfs) for all seeds of an agent.
    We explicitly separate *_metrics.csv from the reward curves to avoid KeyErrors.
    """
    agent_dir = os.path.join(result_dir, scenario, agent)
    if not os.path.isdir(agent_dir):
        return [], []

    # List all CSVs and classify by suffix, robustly
    all_csvs = sorted(glob.glob(os.path.join(agent_dir, f"{agent}_seed*.csv")))
    reward_paths = [p for p in all_csvs if not p.endswith("_metrics.csv")]
    metric_paths = [p for p in all_csvs if p.endswith("_metrics.csv")]

    rdfs, mdfs = [], []
    for p in reward_paths:
        try:
            df = pd.read_csv(p)
            # Must contain 'reward'
            if "reward" in df.columns:
                rdfs.append(df)
        except Exception:
            pass

    for p in metric_paths:
        try:
            df = pd.read_csv(p)
            # Must contain 'episode'
            if "episode" in df.columns:
                mdfs.append(df)
        except Exception:
            pass

    return rdfs, mdfs

def _mean_sem_on_key(dfs: List[pd.DataFrame], key: str) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean ± s.e.m. across runs, aligning on min length, skipping dfs missing key."""
    dfs = [df for df in dfs if key in df.columns]
    if not dfs:
        return np.array([]), np.array([])
    T = min(len(df[key]) for df in dfs)
    if T == 0:
        return np.array([]), np.array([])
    arr = np.stack([df[key].values[:T] for df in dfs], axis=0)  # [runs, T]
    mean = arr.mean(0)
    sem  = arr.std(0) / np.sqrt(arr.shape[0] + 1e-8)
    return mean, sem

def plot_rewards(result_dir: str, scenario: str, out_dir: str = "plots"):
    os.makedirs(out_dir, exist_ok=True)
    any_data = False
    plt.figure(figsize=(12,7))
    for agent in AGENTS:
        rdfs, _ = _collect_runs(result_dir, scenario, agent)
        if not rdfs:
            continue
        mean, sem = _mean_sem_on_key(rdfs, "reward")
        if mean.size == 0:
            continue
        x = np.arange(1, len(mean)+1)
        plt.plot(x, mean, label=agent)
        plt.fill_between(x, mean-sem, mean+sem, alpha=0.15)
        any_data = True
    if not any_data:
        plt.close()
        print(f"[warn] No reward curves found for {scenario} in {result_dir}")
        return
    plt.title(f"{scenario} — Episode rewards (mean ± s.e.m.)")
    plt.xlabel("Episode"); plt.ylabel("Reward")
    plt.legend()
    out = os.path.join(out_dir, f"{scenario}_curves.png")
    plt.savefig(out, dpi=140, bbox_inches="tight"); plt.close()
    print(f"[saved] {out}")

def plot_final_bars(result_dir: str, scenario: str, tail: int = 50, out_dir: str = "plots"):
    os.makedirs(out_dir, exist_ok=True)
    means, sems, names = [], [], []
    for agent in AGENTS:
        rdfs, _ = _collect_runs(result_dir, scenario, agent)
        if not rdfs:
            continue
        vals = []
        for df in rdfs:
            if "reward" not in df.columns or len(df) == 0:
                continue
            k = min(tail, len(df))
            vals.append(df["reward"].values[-k:].mean())
        if not vals:
            continue
        means.append(float(np.mean(vals)))
        sems.append(float(np.std(vals) / np.sqrt(len(vals))))
        names.append(agent)
    if not names:
        print(f"[warn] No final reward data for {scenario} in {result_dir}")
        return
    plt.figure(figsize=(11,7))
    x = np.arange(len(names))
    plt.bar(x, means, yerr=sems, capsize=4)
    plt.xticks(x, names)
    plt.ylabel(f"Avg reward (last {tail} eps)")
    plt.title(f"{scenario} — Final performance")
    out = os.path.join(out_dir, f"{scenario}_final_bars.png")
    plt.savefig(out, dpi=140, bbox_inches="tight"); plt.close()
    print(f"[saved] {out}")

def plot_metrics(result_dir: str, scenario: str, out_dir: str = "plots_metrics"):
    """
    Plots new metrics (mean ± s.e.m.) across episodes:
      - avg_throughput (bits/symbol)
      - per (packet error rate)
      - avg_tx_power
      - MCS usage (QPSK/16QAM/64QAM)
    Only for agents that produced *_metrics.csv (our in-house agents).
    """
    os.makedirs(out_dir, exist_ok=True)
    metrics_to_plot = [
        ("avg_throughput", "Avg Throughput (bits/symbol)"),
        ("per", "Packet Error Rate"),
        ("avg_tx_power", "Avg Transmit Power"),
    ]
    # time-series panels
    for key, label in metrics_to_plot:
        plt.figure(figsize=(12,6))
        any_data = False
        for agent in ["NPG", "QAC", "QPPG"]:  # these write metrics by default
            _, mdfs = _collect_runs(result_dir, scenario, agent)
            if not mdfs:
                continue
            mean, sem = _mean_sem_on_key(mdfs, key)
            if mean.size == 0:
                continue
            x = np.arange(1, len(mean)+1)
            plt.plot(x, mean, label=agent)
            plt.fill_between(x, mean-sem, mean+sem, alpha=0.15)
            any_data = True
        if not any_data:
            plt.close()
            continue
        plt.xlabel("Episode"); plt.ylabel(label); plt.title(f"{scenario} — {label}")
        plt.legend()
        out = os.path.join(out_dir, f"{scenario}_{key}.png")
        plt.savefig(out, dpi=140, bbox_inches="tight"); plt.close()
        print(f"[saved] {out}")

    # stacked bar (last 50 episodes) for MCS usage
    names = []; u4s=[]; u16s=[]; u64s=[]
    for agent in ["NPG", "QAC", "QPPG"]:
        _, mdfs = _collect_runs(result_dir, scenario, agent)
        if not mdfs:
            continue
        # ensure metric columns exist and handle short runs
        ok = [df for df in mdfs if {"mcs_usage_qpsk","mcs_usage_16qam","mcs_usage_64qam"}.issubset(df.columns)]
        if not ok:
            continue
        tail = min(50, min(len(df) for df in ok))
        u4  = np.mean([df["mcs_usage_qpsk"].values[-tail:].mean()  for df in ok])
        u16 = np.mean([df["mcs_usage_16qam"].values[-tail:].mean() for df in ok])
        u64 = np.mean([df["mcs_usage_64qam"].values[-tail:].mean() for df in ok])
        names.append(agent); u4s.append(u4); u16s.append(u16); u64s.append(u64)

    if names:
        plt.figure(figsize=(10,6))
        x = np.arange(len(names))
        p1 = plt.bar(x, u4s, label="QPSK")
        p2 = plt.bar(x, u16s, bottom=u4s, label="16-QAM")
        p3 = plt.bar(x, u64s, bottom=(np.array(u4s)+np.array(u16s)), label="64-QAM")
        plt.xticks(x, names); plt.ylabel("Usage fraction"); plt.ylim(0,1.02)
        plt.title(f"{scenario} — MCS usage (avg over last 50 eps)")
        plt.legend()
        out = os.path.join(out_dir, f"{scenario}_mcs_usage.png")
        plt.savefig(out, dpi=140, bbox_inches="tight"); plt.close()
        print(f"[saved] {out}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=str, required=True, help="root folder passed as --out to run_experiments.py")
    ap.add_argument("--scenario", type=str, required=True)
    args = ap.parse_args()
    plot_rewards(args.results, args.scenario)
    plot_final_bars(args.results, args.scenario)
    plot_metrics(args.results, args.scenario)
