import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from womd.womd_loader import load_womd_directory
from part_b.prediction import kalman_predict
from common.plotting import savefig


def normalize_track(g):
    g = g.sort_values("t").copy()
    x0, y0 = g[["x", "y"]].iloc[0].values
    g["x"] -= x0
    g["y"] -= y0
    g["x_meas"] -= x0
    g["y_meas"] -= y0
    return g


def add_gaussian(grid, xs, ys, cx, cy, sigma):
    X, Y = np.meshgrid(xs, ys)
    grid += np.exp(-((X-cx)**2 + (Y-cy)**2) / (2*sigma**2))
    return grid


def run_future_occupancy_heatmap(
    womd_dir="data/womd/validation",
    max_files=2,
    max_scenarios=120,
    horizon_s=5.0,
    dt=0.1
):
    df = load_womd_directory(
        womd_dir,
        max_files=max_files,
        max_scenarios=max_scenarios
    )

    xs = np.linspace(-40, 40, 180)
    ys = np.linspace(-20, 80, 220)
    grid = np.zeros((len(ys), len(xs)))

    agents_used = 0

    h_steps = int(1.0 / dt)
    f_steps = int(horizon_s / dt)

    for (sid, aid), g in df.groupby(["scenario_id", "agent_id"]):
        g = normalize_track(g)

        if len(g) < h_steps + f_steps:
            continue

        seg = g.iloc[:h_steps + f_steps]
        if np.max(np.diff(seg.t.values)) > 0.25:
            continue

        hist = seg.iloc[:h_steps]
        fut = seg.iloc[h_steps:]

        try:
            pred, covs = kalman_predict(hist, fut.t.values)
        except Exception:
            continue

        for p, C in zip(pred[::5], covs[::5]):
            sigma = max(0.8, float(np.sqrt(np.trace(C[:2, :2]))))
            grid = add_gaussian(grid, xs, ys, p[0], p[1], sigma)

        agents_used += 1

    if grid.max() > 0:
        grid /= grid.max()

    plt.figure(figsize=(8,7))
    plt.imshow(
        grid,
        origin="lower",
        extent=[xs[0], xs[-1], ys[0], ys[-1]],
        aspect="auto"
    )
    plt.colorbar(label="Normalized future occupancy probability")
    plt.scatter([0], [0], marker="^", s=140, label="ego/headlamp")
    plt.xlabel("relative x (m)")
    plt.ylabel("relative y (m)")
    plt.title("WOMD VRU future occupancy heatmap")
    plt.legend()

    fig_path = savefig("part_b_17_future_occupancy_heatmap.png")

    metrics = {
        "agents_used": int(agents_used),
        "horizon_s": float(horizon_s),
        "max_occupancy": float(grid.max()),
        "mean_occupancy": float(grid.mean()),
        "figure": str(fig_path),
    }

    Path("outputs").mkdir(exist_ok=True)

    with open("outputs/future_occupancy_heatmap.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    metrics = run_future_occupancy_heatmap()
    print(json.dumps(metrics, indent=2))
