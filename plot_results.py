# plot_results.py
import os, re, glob, math
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt


AGENT_DIR_RE = re.compile(r"^(?P<agent>QPPG|NPG|QAC|PPO|TRPO)(?:_(?P<variant>.*))?$")
SEED_FILE_RE = re.compile(r".*_seed(?P<seed>\d+)\.csv$")


def _fast_read_csv(path: str) -> pd.DataFrame:
    """Read a result CSV with robust dtype hints and the fast parser if available."""
    kwargs = dict(dtype={"episode": np.int32, "reward": np.float32})
    # Use pyarrow engine if available; falls back to C/py.
    try:
        return pd.read_csv(path, engine="pyarrow", **kwargs)
    except Exception:
        return pd.read_csv(path, **kwargs)


def collect_runs(out_root: str = "results") -> pd.DataFrame:
    """
    Scan results/<scenario>/<agent_or_variant>/*.csv and return a tidy dataframe:
    columns = [scenario, agent, variant, seed, episode, reward]
    """
    rows = []
    out_root = Path(out_root)

    for scenario_dir in sorted(out_root.iterdir()):
        if not scenario_dir.is_dir():
            continue
        scenario = scenario_dir.name

        for agent_dir in sorted(scenario_dir.iterdir()):
            if not agent_dir.is_dir():
                continue

            m = AGENT_DIR_RE.match(agent_dir.name)
            if not m:
                # skip unknown dirs (like meta folders)
                continue
            agent = m.group("agent")
            variant = m.group("variant") or ""

            csv_paths = glob.glob(str(agent_dir / f"{agent}*_seed*.csv"))
            if not csv_paths:
                # allow sb3 agents that might be named slightly differently
                csv_paths = glob.glob(str(agent_dir / "*_seed*.csv"))
            if not csv_paths:
                continue

            # Bulk read each run and attach metadata
            for p in csv_paths:
                seed_m = SEED_FILE_RE.search(os.path.basename(p))
                if not seed_m:
                    continue
                seed = int(seed_m.group("seed"))
                df = _fast_read_csv(p)
                # normalize columns
                if "episode" not in df.columns or "reward" not in df.columns:
                    continue
                df = df[["episode", "reward"]].copy()
                df["scenario"] = scenario
                df["agent"] = agent
                df["variant"] = variant
                df["seed"] = seed
                rows.append(df)

    if not rows:
        raise RuntimeError(f"No result CSVs found under: {out_root}")

    big = pd.concat(rows, ignore_index=True)
    # Consistent ordering
    big["agent_variant"] = np.where(big["variant"].eq(""), big["agent"], big["agent"] + " (" + big["variant"] + ")")
    return big[["scenario", "agent", "variant", "agent_variant", "seed", "episode", "reward"]]


def _smooth(y: np.ndarray, k: int) -> np.ndarray:
    """Simple moving average with edge handling."""
    if k <= 1:
        return y
    k = int(k)
    kernel = np.ones(k, dtype=np.float32) / k
    return np.convolve(y, kernel, mode="same")


def plot_all_results(
    out_root: str = "results",
    out_dir: str = "plots",
    smooth_window: int = 1,
    show_std: bool = True,
    min_seeds_to_plot: int = 1,
    dpi: int = 140,
):
    """
    Produce:
      - One line plot per scenario with mean across seeds (±std shaded).
      - One bar chart comparing final-episode averages per agent/variant per scenario.

    Args:
      out_root: folder created by run_experiments.py (default "results")
      out_dir: where to save PNGs (default "plots")
      smooth_window: moving-average window (episodes) for curves (default 1 = no smoothing)
      show_std: whether to shade ±std around mean
      min_seeds_to_plot: require at least this many seeds for an agent to appear
      dpi: output figure DPI
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    df = collect_runs(out_root)

    # Filter out very sparse runs
    seed_counts = df.groupby(["scenario", "agent_variant"])["seed"].nunique().reset_index(name="n_seeds")
    df = df.merge(seed_counts, on=["scenario", "agent_variant"], how="left")
    df = df[df["n_seeds"] >= min_seeds_to_plot].copy()

    # === Per-scenario learning curves ===
    for scenario, d in df.groupby("scenario", sort=False):
        # Aggregate mean/std across seeds for each episode
        agg = (d.groupby(["agent_variant", "episode"], sort=True)["reward"]
                 .agg(["mean", "std"]).reset_index())

        # Smooth per-series after pivoting (keeps speed)
        piv_mean = agg.pivot(index="episode", columns="agent_variant", values="mean").sort_index()
        piv_std  = agg.pivot(index="episode", columns="agent_variant", values="std").sort_index()

        if smooth_window > 1:
            piv_mean = piv_mean.apply(lambda x: pd.Series(_smooth(x.values.astype(np.float32), smooth_window), index=x.index))

        plt.figure(figsize=(10, 6))
        for col in piv_mean.columns:
            y = piv_mean[col].values
            x = piv_mean.index.values
            plt.plot(x, y, label=col, linewidth=1.8)
            if show_std and col in piv_std:
                s = piv_std[col].values
                if smooth_window > 1:
                    s = _smooth(s.astype(np.float32), smooth_window)
                plt.fill_between(x, y - s, y + s, alpha=0.15, linewidth=0)

        plt.title(f"Learning Curves — {scenario}")
        plt.xlabel("Episode")
        plt.ylabel("Average Reward")
        plt.legend(loc="best", fontsize=9)
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{scenario}_curves.png"), dpi=dpi)
        plt.close()

    # === Final-episode comparison per scenario ===
    finals = (df.groupby(["scenario", "agent_variant", "seed"])
                .apply(lambda x: x.loc[x["episode"].idxmax(), "reward"])
                .reset_index(name="final_reward"))

    summary = (finals.groupby(["scenario", "agent_variant"])
                    ["final_reward"].agg(["mean", "std", "count"])
                    .reset_index().rename(columns={"count": "n_seeds"}))

    for scenario, dsum in summary.groupby("scenario", sort=False):
        dsum = dsum.sort_values("mean", ascending=False)
        labels = dsum["agent_variant"].tolist()
        y = dsum["mean"].values
        err = dsum["std"].values

        plt.figure(figsize=(10, 6))
        xs = np.arange(len(labels))
        plt.bar(xs, y, yerr=err, capsize=3)
        plt.xticks(xs, labels, rotation=20, ha="right")
        plt.title(f"Final-Episode Reward — {scenario}")
        plt.ylabel("Reward")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{scenario}_final_bars.png"), dpi=dpi)
        plt.close()

    # === Overall CSV summary for quick table viewing ===
    summary_path = os.path.join(out_dir, "final_summary.csv")
    summary.sort_values(["scenario", "mean"], ascending=[True, False]).to_csv(summary_path, index=False)
    print(f"Saved plots to '{out_dir}' and summary CSV to '{summary_path}'.")
    

if __name__ == "__main__":
    # Example CLI usage:
    #   python plot_results.py --root results --out plots --smooth 5
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="results")
    ap.add_argument("--out", type=str, default="plots")
    ap.add_argument("--smooth", type=int, default=1)
    ap.add_argument("--nostd", action="store_true")
    ap.add_argument("--min_seeds", type=int, default=1)
    ap.add_argument("--dpi", type=int, default=140)
    args = ap.parse_args()

    plot_all_results(
        out_root=args.root,
        out_dir=args.out,
        smooth_window=args.smooth,
        show_std=(not args.nostd),
        min_seeds_to_plot=args.min_seeds,
        dpi=args.dpi,
    )
