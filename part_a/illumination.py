import numpy as np
import matplotlib.pyplot as plt
from common.plotting import savefig

def raised_cosine_shadow(theta_grid, center, width=5.0, transition=3.0):
    d = np.abs(theta_grid - center)
    intensity = np.ones_like(theta_grid)
    intensity[d <= width] = 0.05
    trans = (d > width) & (d <= width + transition)
    u = (d[trans] - width) / transition
    intensity[trans] = 0.05 + 0.95 * 0.5 * (1 - np.cos(np.pi*u))
    return intensity

def demo_illumination():
    theta = np.linspace(-30, 30, 600)
    centers = [-8, 10]
    I = np.ones_like(theta)
    for c in centers:
        I = np.minimum(I, raised_cosine_shadow(theta, c))

    plt.figure(figsize=(8,4))
    plt.plot(theta, I)
    for c in centers:
        plt.axvline(c, ls="--", alpha=0.5)
    plt.xlabel("Horizontal angle (deg)")
    plt.ylabel("Normalized beam intensity")
    plt.title("Adaptive driving beam shadow zones")
    p = savefig("part_a_06_adaptive_beam.png")
    return {"shadow_centers_deg": centers, "figure": str(p)}

if __name__ == "__main__":
    print(demo_illumination())
