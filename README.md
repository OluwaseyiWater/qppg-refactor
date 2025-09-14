
# QPPG for Link Adaptation — Project Skeleton

This folder contains a clean, modular framework to reproduce **Phase 1** (experiments) and **Phase 2** (paper figures) for:

**Title:** *Quantum Fisher-Preconditioned Reinforcement Learning for Link Adaptation in a Rayleigh Fading Channel*

## Structure
```
qrl_linkadapt/
  environment.py          # Rayleigh env + SB3 wrapper
  agents.py               # QPPG, Classical NPG, QAC, SB3 training helpers
  run_experiments.py      # Runs 5 scenarios × 15 seeds + xi ablation
  plot_results.py         # Builds learning-curve and bar figures + ablation
  results/                # (created at runtime) CSVs
  figures/                # (created at runtime) PDFs for the paper
  fig_framework.mmd       # Mermaid diagram of the QPPG workflow
  paper/
    main.tex              # 5-page IEEE-style draft; includes generated figures
```

## Quickstart
### 1. Install dependencies
```bash
pip install numpy torch gymnasium matplotlib pandas scipy
pip install stable-baselines3 sb3-contrib  # optional: for PPO and TRPO
```

### 2. Run experiments
**Default:** all 5 scenarios, 15 seeds, 500 episodes each.
```bash
cd qrl_linkadapt
python run_experiments.py --episodes 500 --seeds 15 --out results
```
Ablation on $\xi$ (Scenario 1 only) is run automatically at the end; or:
```bash
python run_experiments.py --ablation_only --episodes 500 --seeds 15 --out results
```

### 3. Plot figures
```bash
python plot_results.py
```
This generates PDFs in `figures/` that the LaTeX paper includes.

### 4. Compile the paper
Open `paper/main.tex` in Overleaf or compile locally:
```bash
pdflatex main.tex
```

## Notes
- PPO/TRPO require the SB3 packages. If they are not installed, the script will skip them and proceed with QPPG, NPG, and QAC.
- The SB3 wrapper discretizes transmit power to fit a Discrete action. For precise continuous control, rely on QPPG/NPG/QAC results.
- We use 15 seeds to ensure statistical significance. Plots show 95% CIs.
- The computational complexity section in the paper references wall-clock profiling hooks in the code; feel free to extend with your measured numbers.
