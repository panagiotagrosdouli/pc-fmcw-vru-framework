import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from womd.womd_loader import load_womd_directory
from part_b.prediction import kalman_predict
from common.plotting import savefig


def normalize_track(g):

    g = g.sort_values("t").copy()

    x0, y0 = g[["x","y"]].iloc[0].values

    g["x"] -= x0
    g["y"] -= y0

    g["x_meas"] -= x0
    g["y_meas"] -= y0

    return g


def gaussian2d(
    X,
    Y,
    mx,
    my,
    sigma
):

    return np.exp(
        -(
            (X-mx)**2
            + (Y-my)**2
        ) / (2*sigma**2 + 1e-9)
    )


def run_risk_heatmap_forecasting(
    womd_dir="data/womd/validation",
    max_files=2,
    max_scenarios=120,
    horizon_s=5.0,
    dt=0.1,
    grid_extent=25,
    grid_res=250
):

    df = load_womd_directory(
        womd_dir,
        max_files=max_files,
        max_scenarios=max_scenarios
    )

    history_steps = int(1.0 / dt)
    future_steps = int(horizon_s / dt)

    xs = np.linspace(
        -grid_extent,
        grid_extent,
        grid_res
    )

    ys = np.linspace(
        0,
        2*grid_extent,
        grid_res
    )

    X, Y = np.meshgrid(xs, ys)

    heatmap = np.zeros_like(X)

    rows = []

    for (sid, aid), g in df.groupby(
        ["scenario_id","agent_id"]
    ):

        g = normalize_track(g)

        if len(g) < history_steps + future_steps:
            continue

        seg = g.iloc[:history_steps + future_steps]

        if np.max(np.diff(seg.t.values)) > 0.25:
            continue

        hist = seg.iloc[:history_steps]
        fut = seg.iloc[history_steps:]

        try:

            pred, covs = kalman_predict(
                hist,
                fut.t.values
            )

        except Exception:
            continue

        for p, P in zip(pred, covs):

            sigma = np.sqrt(
                np.trace(P[:2,:2])
            )

            dist = np.linalg.norm(p)

            risk = np.exp(
                -dist / 10
            )

            sigma_eff = (
                1.5 + 0.5*sigma
            )

            heatmap += risk * gaussian2d(
                X,
                Y,
                p[0],
                p[1],
                sigma_eff
            )

        max_risk = float(np.max(heatmap))

        rows.append({
            "scenario_id": sid,
            "agent_id": int(aid),
            "object_type": g.object_type.iloc[0],
            "final_distance_m":
                float(np.linalg.norm(pred[-1])),
            "max_risk_contribution":
                max_risk,
        })

    out = pd.DataFrame(rows)

    heatmap = heatmap / np.max(heatmap)

    plt.figure(figsize=(8,7))

    plt.imshow(
        heatmap,
        extent=[
            xs.min(),
            xs.max(),
            ys.min(),
            ys.max()
        ],
        origin="lower",
        aspect="auto"
    )

    plt.colorbar(
        label="Normalized future risk"
    )

    plt.xlabel("Lateral position x (m)")
    plt.ylabel("Longitudinal position y (m)")

    plt.title(
        "Future VRU risk heatmap forecast"
    )

    fig1 = savefig(
        "part_b_42_risk_heatmap.png"
    )

    threshold = 0.6

    hotspot_mask = heatmap > threshold

    hotspot_fraction = float(
        hotspot_mask.mean()
    )

    hotspot_area = float(
        hotspot_fraction
        * (2*grid_extent)
        * (2*grid_extent)
    )

    plt.figure(figsize=(8,7))

    plt.contourf(
        X,
        Y,
        heatmap,
        levels=20
    )

    plt.contour(
        X,
        Y,
        hotspot_mask,
        levels=[0.5],
        linewidths=2
    )

    plt.xlabel("x (m)")
    plt.ylabel("y (m)")

    plt.title(
        "High-risk future occupancy zones"
    )

    fig2 = savefig(
        "part_b_43_risk_hotspots.png"
    )

    metrics = {
        "num_agents": int(len(out)),
        "grid_resolution": int(grid_res),
        "max_heatmap_value":
            float(np.max(heatmap)),
        "mean_heatmap_value":
            float(np.mean(heatmap)),
        "high_risk_threshold":
            threshold,
        "high_risk_area_m2":
            hotspot_area,
        "figures": {
            "risk_heatmap": str(fig1),
            "risk_hotspots": str(fig2),
        }
    }

    Path("outputs").mkdir(exist_ok=True)

    out.to_csv(
        "outputs/risk_heatmap_agent_contributions.csv",
        index=False
    )

    np.save(
        "outputs/risk_heatmap.npy",
        heatmap
    )

    with open(
        "outputs/risk_heatmap_forecasting.json",
        "w"
    ) as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":

    metrics = run_risk_heatmap_forecasting()

    print(json.dumps(metrics, indent=2))
