
# QPPG for Link Adaptation in Rayleigh Fading Channels

## Quickstart
### 1. Install dependencies
```bash
pip install numpy torch gymnasium matplotlib pandas scipy
pip install stable-baselines3 sb3-contrib  #for PPO and TRPO
```

### 2. Run experiments
**Default:** all 5 scenarios, 15 seeds, for 500 episodes each.
**Separate Runs** To run each scenario separately with its own log results, follow these steps.
```python
python run_experiments.py --episodes 10000 --seeds 15 --scenarios s1_baseline --out results_s1_10000ep
python run_experiments.py --episodes 10000 --seeds 15 --scenarios s2_high_dim --out results_s2_10000ep
python run_experiments.py --episodes 10000 --seeds 15 --scenarios s3_low_csi --out results_s3_10000ep
python run_experiments.py --episodes 10000 --seeds 15 --scenarios s4_noise_uncert --out results_s4_10000ep
python run_experiments.py --episodes 10000 --seeds 15 --scenarios s5_combined --out results_s5_10000ep
```
Ablation on $\xi$ (Scenario 1 only) is run automatically at the end; or:
```python
python run_experiments.py --ablation_only --episodes 500 --seeds 15 --out results
```

### 3. Plot figures
```python
python plot_results.py
```


## Notes
- PPO/TRPO require the SB3 packages. If they are not installed, the script will skip them and proceed with QPPG, NPG, and QAC.
- The SB3 wrapper discretises transmit power to fit a Discrete action. For precise continuous control, rely on QPPG/NPG/QAC results.
- We use 15 seeds to ensure statistical significance.

