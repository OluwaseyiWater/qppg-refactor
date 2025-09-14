# plot_results_multi.py
import argparse, os, csv, glob, json
from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt

AGENT_ORDER = ["NPG", "PPO", "QAC", "QPPG", "TRPO"]
AGENT_COLORS = {
    "NPG":  "#1f77b4",  # blue
    "PPO":  "#ff7f0e",  # orange
    "QAC":  "#2ca02c",  # green
    "QPPG": "#d62728",  # red
    "TRPO": "#9467bd",  # purple
}

# ------------------------------- IO -------------------------------------------

def _read_curve(csv_path: str) -> np.ndarray:
    # CSV format: episode,reward (header row). Returns float array of rewards.
    try:
        data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding=None)
        # data["reward"] can be a scalar if length-1; ensure 1D float array
        rewards = np.atleast_1d(data["reward"]).astype(float)
        return rewards
    except Exception:
        return np.array([], dtype=float)

def scan_root(root: str) -> Dict[str, Dict[str, List[np.ndarray]]]:
    """
    Returns nested dict: scenario -> agent -> list[seed_curve: np.ndarray]
    Expects structure from run_experiments.py:
      {root}/{scenario}/{agent} or {root}/{scenario}/{agent_*}/ *.csv
    """
    out: Dict[str, Dict[str, List[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    if not os.path.isdir(root):
        return out

    # scenarios = subfolders of root
    for sc in sorted(next(os.walk(root))[1]):
        sc_dir = os.path.join(root, sc)
        # agent subfolders (may include sweeps like QPPG_ec0.001)
        for agent_folder in sorted(next(os.walk(sc_dir))[1]):
            # canonical agent name is prefix before first underscore
            agent = agent_folder.split("_", 1)[0]
            if agent not in AGENT_COLORS:
                continue
            af = os.path.join(sc_dir, agent_folder)
            # gather all \*_seedXXXXX.csv
            for csv_path in sorted(glob.glob(os.path.join(af, f"{agent}_seed*.csv"))):
                rewards = _read_curve(csv_path)
                if rewards.size > 0:
                    out[sc][agent].append(rewards)
    return out

# ------------------------------ Aggregation -----------------------------------

def pad_and_aggregate(curves: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (mean, std) per episode, handling variable-length curves with NaNs.
    """
    if not curves:
        return np.array([]), np.array([])
    max_T = max(c.shape[0] for c in curves)
    X = np.full((len(curves), max_T), np.nan, dtype=float)
    for i, c in enumerate(curves):
        X[i, :c.shape[0]] = c
    mean = np.nanmean(X, axis=0)
    std  = np.nanstd(X, axis=0)
    return mean, std

def final_stats(curves: List[np.ndarray]) -> Tuple[float, float]:
    """
    Mean and std of the final available episode across seeds.
    """
    if not curves:
        return np.nan, np.nan
    finals = [c[-1] for c in curves if c.size > 0]
    if not finals:
        return np.nan, np.nan
    return float(np.mean(finals)), float(np.std(finals))

def smooth_series(y: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or y.size == 0:
        return y
    k = int(max(1, k // 2 * 2 + 1))  # odd
    kernel = np.ones(k, dtype=float) / k
    # use 'same' conv with NaN handling: fill NaN by nearest valid via forward/backward fill
    x = y.copy()
    # simple nan fill
    isn = np.isnan(x)
    if np.all(isn):
        return y
    # forward-fill then backward-fill
    idx = np.where(~isn, np.arange(len(x)), 0)
    np.maximum.accumulate(idx, out=idx)
    x[isn] = x[idx[isn]]
    idx = np.where(~isn, np.arange(len(x)), len(x)-1)
    np.minimum.accumulate(idx[::-1], out=idx[::-1])
    x[isn] = x[isn] * 0.5 + x[idx[isn]] * 0.5
    return np.convolve(x, kernel, mode="same")

# ------------------------------ Plotting --------------------------------------

def plot_scenario_overlay(
    scenario: str,
    scanned_roots: List[Dict[str, Dict[str, List[np.ndarray]]]],
    labels: List[str],
    out_dir: str,
    smooth: int = 9,
):
    os.makedirs(out_dir, exist_ok=True)

    # 1) Learning curves overlay (color by agent, linestyle by root)
    linestyles = ["-", "--", ":", "-.", (0, (3, 1, 1, 1))]
    plt.figure(figsize=(12, 7))
    for r_idx, store in enumerate(scanned_roots):
        agents = store.get(scenario, {})
        for agent in AGENT_ORDER:
            curves = agents.get(agent, [])
            if not curves:
                continue
            mean, std = pad_and_aggregate(curves)
            if mean.size == 0:
                continue
            if smooth > 1:
                mean = smooth_series(mean, smooth)
                std  = smooth_series(std, smooth)
            x = np.arange(1, mean.shape[0] + 1)
            color = AGENT_COLORS[agent]
            ls = linestyles[r_idx % len(linestyles)]
            label = f"{agent} ({labels[r_idx]})"
            plt.plot(x, mean, color=color, linestyle=ls, linewidth=2.0, label=label)
            plt.fill_between(x, mean - std, mean + std, color=color, alpha=0.12)

    plt.title(f"Learning Curves — {scenario}", fontsize=16)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.grid(alpha=0.25)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{scenario}_curves_overlay.png"), dpi=160)
    plt.close()

    # 2) Grouped bar chart for final-episode reward
    agents_all = sorted(
        set().union(*[set(scanned_roots[i].get(scenario, {}).keys()) for i in range(len(scanned_roots))]),
        key=lambda a: AGENT_ORDER.index(a) if a in AGENT_ORDER else 999,
    )
    if not agents_all:
        return

    width = 0.8 / max(1, len(scanned_roots))
    x0 = np.arange(len(agents_all))
    plt.figure(figsize=(12, 7))
    for r_idx, store in enumerate(scanned_roots):
        means, stds = [], []
        for agent in agents_all:
            m, s = final_stats(store.get(scenario, {}).get(agent, []))
            means.append(m); stds.append(s)
        offset = (r_idx - (len(scanned_roots)-1)/2) * width
        plt.bar(x0 + offset, means, width=width, yerr=stds, capsize=4,
                color=[AGENT_COLORS.get(a, "#777777") for a in agents_all],
                edgecolor="black", linewidth=0.5, label=labels[r_idx])

    plt.xticks(x0, agents_all)
    plt.ylabel("Reward")
    plt.title(f"Final-Episode Reward — {scenario}")
    plt.legend(title="Run")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{scenario}_final_bars_grouped.png"), dpi=160)
    plt.close()

    # 3) CSV summary for reproducibility
    csv_path = os.path.join(out_dir, f"{scenario}_final_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["agent"]
        for lab in labels:
            header += [f"{lab}_mean", f"{lab}_std"]
        w.writerow(header)
        for agent in agents_all:
            row = [agent]
            for store in scanned_roots:
                m, s = final_stats(store.get(scenario, {}).get(agent, []))
                row += [f"{m:.6f}" if np.isfinite(m) else "", f"{s:.6f}" if np.isfinite(s) else ""]
            w.writerow(row)

# ----------------------------- Orchestrator -----------------------------------

def plot_all_scenarios_multi(roots: List[str], labels: List[str], out_dir: str, smooth: int = 9):
    scanned = [scan_root(r) for r in roots]
    # union of scenarios across all roots
    scenarios = sorted(set().union(*[set(s.keys()) for s in scanned]))
    if not scenarios:
        print("[WARN] No scenarios found in the provided roots.")
        return
    os.makedirs(out_dir, exist_ok=True)
    for sc in scenarios:
        print(f"[plot] {sc}")
        plot_scenario_overlay(sc, scanned, labels, out_dir, smooth=smooth)

# --------------------------------- CLI ----------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True,
                    help="Result roots to compare (e.g., results results_qppg_tuned)")
    ap.add_argument("--labels", nargs="+", required=True,
                    help="Legend labels for each root (same length as --roots)")
    ap.add_argument("--out", type=str, default="plots_compare")
    ap.add_argument("--smooth", type=int, default=9, help="Odd window size for moving average; 1 disables.")
    args = ap.parse_args()

    if len(args.labels) != len(args.roots):
        raise SystemExit("ERROR: --labels must have the same number of items as --roots")

    plot_all_scenarios_multi(args.roots, args.labels, args.out, smooth=args.smooth)

if __name__ == "__main__":
    main()
