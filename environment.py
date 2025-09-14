
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Tuple


class RayleighFadingChannelEnv(gym.Env):
    """
    Single-user link adaptation over a Rayleigh fading channel.
    Action is a dict: {'modulation': Discrete(3), 'power': Box([0,1])}.
    Observation: [Re(h_hat), Im(h_hat)] for N antennas + estimated noise variance.
    Reward: throughput if SNR >= threshold, penalized by power usage; else small negative.
    """
    metadata = {"render_modes": []}

    def __init__(self,
                 n_antennas: int = 4,
                 block_length: int = 50,
                 pilot_snr_db: float = 10.0,
                 max_power: float = 1.0,
                 noise_uncertainty_db: float = 0.0,
                 max_steps: int = 20,
                 power_penalty: float = 0.1,
                 seed: int = None):
        super().__init__()
        self.n_antennas = n_antennas
        self.block_length = block_length
        self.pilot_snr = 10 ** (pilot_snr_db / 10.0)
        self.max_power = float(max_power)
        self.noise_uncertainty_db = float(noise_uncertainty_db)
        self.max_steps = int(max_steps)
        self.power_penalty = float(power_penalty)
        self.rng = np.random.RandomState(seed)

        # Action: Dict(modulation ∈ {0,1,2}, power ∈ [0,1])
        self.action_space = spaces.Dict({
            "modulation": spaces.Discrete(3),   # maps to {4,16,64}
            "power": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        })
        # Observation: Re(h_hat)[N], Im(h_hat)[N], sigma2_est
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2*self.n_antennas + 1,), dtype=np.float32)

        # Modulations and SNR thresholds (dB)
        self.mod_orders = [4, 16, 64]
        self.snr_thr_db = {4: 7.0, 16: 12.0, 64: 18.0}
        self.snr_thr_lin = {M: 10 ** (v / 10.0) for M, v in self.snr_thr_db.items()}

        # Base noise power; can be changed if desired
        self.noise_power_true = 0.1

        self.current_step = 0
        self.h_true = None
        self.h_est = None
        self.sigma2_true = None
        self.sigma2_est = None

    def _gen_channel(self):
        # Rayleigh fading (CN(0,1/N) normalized total power ~ N)
        z = (self.rng.randn(self.n_antennas) + 1j * self.rng.randn(self.n_antennas)) / np.sqrt(2.0)
        # Normalize average power to N
        pwr = np.sum(np.abs(z) ** 2)
        z = z * np.sqrt(self.n_antennas / max(pwr, 1e-12))
        self.h_true = z
        self.sigma2_true = self.noise_power_true

        # pilot estimate with AWGN
        pilot_noise_std = np.sqrt(self.sigma2_true / max(self.pilot_snr, 1e-12))
        pilot_noise = (self.rng.randn(self.n_antennas) + 1j * self.rng.randn(self.n_antennas)) * pilot_noise_std / np.sqrt(2.0)
        self.h_est = self.h_true + pilot_noise

        # noise variance estimate with uncertainty (uniform in dB window)
        if self.noise_uncertainty_db > 0.0:
            u = self.rng.uniform(-self.noise_uncertainty_db/20.0, self.noise_uncertainty_db/20.0)
            self.sigma2_est = self.sigma2_true * (10 ** u)
        else:
            self.sigma2_est = self.sigma2_true

    def _obs(self):
        return np.concatenate([np.real(self.h_est), np.imag(self.h_est), [self.sigma2_est]]).astype(np.float32)

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        self.current_step = 0
        self._gen_channel()
        return self._obs(), {}

    def step(self, action: Dict[str, Any]):
        mod_idx = int(action["modulation"])
        tx_power = float(action["power"][0]) * self.max_power
        M = self.mod_orders[mod_idx]

        channel_gain = np.sum(np.abs(self.h_true) ** 2)
        snr_true = tx_power * channel_gain / max(self.sigma2_true, 1e-12)

        success = snr_true >= self.snr_thr_lin[M]
        if success:
            throughput = np.log2(M)
            power_eff = 1.0 - self.power_penalty * (tx_power / self.max_power)
            reward = float(throughput * max(power_eff, 0.0))
        else:
            throughput = 0.0
            reward = -0.1

        self.current_step += 1
        done = self.current_step >= self.max_steps

        if not done:
            self._gen_channel()

        info = {
            "success": bool(success),
            "modulation": int(M),
            "tx_power": tx_power,
            "snr_true_db": 10.0 * np.log10(max(snr_true, 1e-12)),
            "snr_thr_db": self.snr_thr_db[M],
            "throughput": throughput
        }
        return self._obs(), reward, done, False, info


class RayleighSB3Wrapper(gym.Env):
    """
    SB3-compatible env: single Discrete action index → (mod, discretized power).
    Useful because SB3 doesn't natively handle Dict action spaces.
    """
    def __init__(self, base_env: RayleighFadingChannelEnv, power_bins: int = 21):
        super().__init__()
        assert isinstance(base_env, RayleighFadingChannelEnv)
        self.base = base_env
        self.power_bins = int(power_bins)
        self.action_space = spaces.Discrete(3 * self.power_bins)  # (mod in {0,1,2}) × (power_bin in {0..P-1})
        self.observation_space = self.base.observation_space

    def _decode(self, a: int):
        mod_idx = a // self.power_bins
        p_idx = a % self.power_bins
        p = p_idx / (self.power_bins - 1)
        return {"modulation": mod_idx, "power": np.array([p], dtype=np.float32)}

    def reset(self, seed=None, options=None):
        return self.base.reset(seed=seed, options=options)

    def step(self, a: int):
        action = self._decode(int(a))
        return self.base.step(action)
