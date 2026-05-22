import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter
from common.plotting import savefig

def build_rdm(targets=((30, 12, 1.0), (100, -20, 0.8)), n_range=256, n_doppler=128, seed=3):
    '''
    Self-consistent automotive RDM demonstration.

    The paper's optical FMCW numbers give centimeter resolution but impractical
    unambiguous range/velocity for a small FFT display. Here targets are placed
    directly in normalized range/Doppler bins, while the processing remains a
    genuine 2D FFT + CFAR demonstration.
    '''
    rng = np.random.default_rng(seed)
    Rmax, Vmax = 150.0, 30.0
    n = np.arange(n_range)
    m = np.arange(n_doppler)
    X = np.zeros((n_doppler, n_range), dtype=complex)
    for R, V, amp in targets:
        kr = R / Rmax * (n_range - 1)
        kd = (V + Vmax) / (2*Vmax) * (n_doppler - 1)
        phase_r = np.exp(1j * 2*np.pi * kr * n / n_range)
        phase_d = np.exp(1j * 2*np.pi * (kd - n_doppler/2) * m / n_doppler)
        X += amp * phase_d[:, None] * phase_r[None, :]
    noise = (rng.normal(size=X.shape) + 1j*rng.normal(size=X.shape)) * 0.1
    X += noise
    S = np.fft.fftshift(np.fft.fft2(X), axes=0)
    P = np.abs(S)**2
    ranges = np.linspace(0, Rmax, n_range)
    velocities = np.linspace(-Vmax, Vmax, n_doppler)
    return P / P.max(), ranges, velocities

def simple_cfar(P, threshold_rel=0.35):
    local_max = maximum_filter(P, size=5)
    mask = (P == local_max) & (P > threshold_rel)
    return np.argwhere(mask)

def demo_sensing():
    P, ranges, velocities = build_rdm()
    det = simple_cfar(P)
    detected = []
    for d, r in det:
        detected.append((float(ranges[r]), float(velocities[d]), float(P[d, r])))

    plt.figure(figsize=(8,5))
    plt.imshow(10*np.log10(P+1e-8), origin="lower", aspect="auto",
               extent=[ranges[0], ranges[-1], velocities[0], velocities[-1]])
    plt.colorbar(label="Power (dB)")
    for R, V, _ in detected:
        plt.plot(R, V, "rx", markersize=8, mew=2)
    plt.xlabel("Range (m)")
    plt.ylabel("Velocity (m/s)")
    plt.title("Range-Doppler map with CA-CFAR-like peak detections")
    p = savefig("part_a_04_range_doppler_cfar.png")

    return {"num_detections": len(detected), "detections": detected, "figure": str(p)}

if __name__ == "__main__":
    print(demo_sensing())
