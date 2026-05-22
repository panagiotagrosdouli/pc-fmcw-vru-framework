import numpy as np
import matplotlib.pyplot as plt
from common.config import CFG
from common.plotting import savefig

def generate_dpsk_bits(n_bits=512, seed=1):
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=n_bits)
    phase = np.pi * np.cumsum(bits) % (2*np.pi)
    return bits, phase

def pc_fmcw_chirp(target_range=7.5):
    n = CFG.samples_per_chirp
    t = np.arange(n) / CFG.fs
    tau = 2 * target_range / CFG.c
    bits, sym_phase = generate_dpsk_bits(512)
    sym_idx = np.minimum((t * CFG.data_rate).astype(int), len(sym_phase)-1)
    phi = sym_phase[sym_idx]

    tx = np.exp(1j * (2*np.pi*CFG.fc*t + np.pi*CFG.slope*t**2 + phi))
    rx = np.exp(1j * (2*np.pi*CFG.fc*(t-tau) + np.pi*CFG.slope*(t-tau)**2 + phi))
    beat = rx * np.conj(tx)
    return t, tx, rx, beat

def estimate_range_from_beat(beat):
    spec = np.fft.fftshift(np.fft.fft(beat * np.hanning(len(beat))))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(beat), d=1/CFG.fs))
    fb = abs(freqs[np.argmax(np.abs(spec))])
    return CFG.c * fb / (2 * CFG.slope), freqs, spec

def demo_waveform():
    true_r = 7.5
    t, tx, rx, beat = pc_fmcw_chirp(true_r)
    r_hat, freqs, spec = estimate_range_from_beat(beat)

    plt.figure(figsize=(8,4))
    plt.plot(t[:300]*1e6, np.unwrap(np.angle(tx[:300])), label="TX phase")
    plt.plot(t[:300]*1e6, np.unwrap(np.angle(rx[:300])), label="RX phase", alpha=0.8)
    plt.xlabel("Time (µs)")
    plt.ylabel("Unwrapped phase (rad)")
    plt.title("PC-FMCW transmitted and delayed received phase")
    plt.legend()
    p1 = savefig("part_a_01_waveform_phase.png")

    plt.figure(figsize=(8,4))
    plt.plot(freqs/1e6, 20*np.log10(np.abs(spec)/np.max(np.abs(spec))+1e-12))
    plt.xlim(-80, 80)
    plt.xlabel("Beat frequency (MHz)")
    plt.ylabel("Normalized magnitude (dB)")
    plt.title(f"Beat spectrum: true range={true_r:.2f} m, estimated={r_hat:.3f} m")
    p2 = savefig("part_a_02_beat_spectrum.png")

    return {
        "true_range_m": true_r,
        "estimated_range_m": float(r_hat),
        "absolute_error_cm": float(abs(r_hat-true_r)*100),
        "figures": [str(p1), str(p2)]
    }

if __name__ == "__main__":
    print(demo_waveform())
