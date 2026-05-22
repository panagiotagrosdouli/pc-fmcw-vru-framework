import numpy as np
import matplotlib.pyplot as plt
from common.plotting import savefig

def dpsk_ber_sim(snr_db_values=np.arange(0, 15, 2), n_bits=50000, seed=2):
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, n_bits)
    phase = np.pi * np.cumsum(bits)
    symbols = np.exp(1j * phase)
    bers = []
    theory = []
    for snr_db in snr_db_values:
        snr = 10**(snr_db/10)
        noise = (rng.normal(size=n_bits) + 1j*rng.normal(size=n_bits)) / np.sqrt(2*snr)
        y = symbols + noise
        diff = y[1:] * np.conj(y[:-1])
        bhat = (np.real(diff) < 0).astype(int)
        bers.append(np.mean(bhat != bits[1:]))
        theory.append(0.5*np.exp(-snr))
    return np.array(snr_db_values), np.array(bers), np.array(theory)

def demo_communication():
    snr_db, ber, theory = dpsk_ber_sim()
    plt.figure(figsize=(7,4))
    plt.semilogy(snr_db, ber, "o-", label="Python DPSK simulation")
    plt.semilogy(snr_db, theory, "--", label="DPSK theoretical trend")
    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("BER")
    plt.grid(True, which="both", alpha=0.3)
    plt.title("1 Gbps DPSK communication baseline")
    plt.legend()
    p = savefig("part_a_03_dpsk_ber.png")
    return {"min_ber": float(ber[-1]), "figure": str(p)}

if __name__ == "__main__":
    print(demo_communication())
