# make_s4_plots.py
import os, glob, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = "results_s3_1000ep"
SCEN = "s3_low_csi"
SC_DIR = os.path.join(ROOT, SCEN)

def load_rewards(files):
    arrs = []
    for f in files:
        df = pd.read_csv(f)
        arrs.append(df["reward"].to_numpy(dtype=float))
    if not arrs:
        return np.empty((0,0))
    T = max(len(x) for x in arrs)
    out = np.full((len(arrs), T), np.nan)
    for i, x in enumerate(arrs):
        out[i, :len(x)] = x
    return out

agents = []
files = {}
for a in sorted(os.listdir(SC_DIR)):
    a_dir = os.path.join(SC_DIR, a)
    if not os.path.isdir(a_dir):
        continue
    csvs = sorted(glob.glob(os.path.join(a_dir, f"{a}_seed*.csv")))
    if csvs:
        agents.append(a)
        files[a] = csvs

data = {a: load_rewards(files[a]) for a in agents}

# 1) curves
plt.figure(figsize=(8,5))
for a in agents:
    arr = data[a]
    if arr.size == 0: 
        continue
    mean = np.nanmean(arr, axis=0)
    sem  = np.nanstd(arr, axis=0, ddof=1)/math.sqrt(arr.shape[0]) if arr.shape[0] > 1 else np.zeros_like(mean)
    x = np.arange(1, len(mean)+1)
    plt.plot(x, mean, label=a)
    if arr.shape[0] > 1:
        plt.fill_between(x, mean - sem, mean + sem, alpha=0.2)
plt.title(f"{SCEN} — Episode rewards (mean ± s.e.m.)")
plt.xlabel("Episode"); plt.ylabel("Reward"); plt.legend(); plt.tight_layout()
curves_path = os.path.join(ROOT, f"{SCEN}_curves.png")
plt.savefig(curves_path, dpi=140); plt.close()

# 2) final bars over last K episodes
labels, bars, errs = [], [], []
for a in agents:
    arr = data[a]
    if arr.size == 0: 
        continue
    T = arr.shape[1]
    K = min(50, T)
    tail = arr[:, -K:]
    per_seed = np.nanmean(tail, axis=1)
    labels.append(a)
    bars.append(np.nanmean(per_seed))
    errs.append(np.nanstd(per_seed, ddof=1)/math.sqrt(len(per_seed)) if len(per_seed) > 1 else 0.0)

plt.figure(figsize=(7,5))
x = np.arange(len(labels))
plt.bar(x, bars, yerr=errs, capsize=4)
plt.xticks(x, labels, rotation=0)
plt.ylabel(f"Avg reward (last {K} eps)")
plt.title(f"{SCEN} — Final performance")
plt.tight_layout()
bars_path = os.path.join(ROOT, f"{SCEN}_final_bars.png")
plt.savefig(bars_path, dpi=140); plt.close()

print("Saved:")
print(curves_path)
print(bars_path)
