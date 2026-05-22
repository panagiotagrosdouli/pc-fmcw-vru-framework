from dataclasses import dataclass
import numpy as np

@dataclass
class FMCWConfig:
    c: float = 3e8
    fc: float = 193.4e12
    bandwidth: float = 10e9
    chirp_time: float = 10e-6
    data_rate: float = 1e9
    samples_per_chirp: int = 2048
    num_chirps: int = 128
    seed: int = 7

    @property
    def slope(self):
        return self.bandwidth / self.chirp_time

    @property
    def fs(self):
        return self.samples_per_chirp / self.chirp_time

    @property
    def wavelength(self):
        return self.c / self.fc

    @property
    def range_resolution(self):
        return self.c / (2 * self.bandwidth)

CFG = FMCWConfig()
RNG = np.random.default_rng(CFG.seed)
