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

    x0, y0 = g[["x", "y"]].iloc[0].values

    g["x"] -= x0
    g["y"] -= y0
    g["x_meas"] -= x0
    g["y_meas"] -= y0

    return g


def compute_collision_metrics(pred, dt=0.1):

    dists = np.linalg.norm(pred, axis=1)

    idx = np.argmin(dists)

    min_dist = float(dists[idx])

    ttc = float(idx * dt)

    risk = np.exp(-min_dist / 5)

    return min_dist, ttc, risk


def run_near_collision_forecasting(
    womd_dir="data/womd/validation",
    max_files=2,
    max_scenarios=150,
    horizon_s=5.0,
    dt=0.1
):

    df = load_womd_directory(
        womd_dir,
        max_files=max_files,
        max_scenarios=max_scenarios
    )

    history_steps = int(1.0 / dt)
    future_steps = int(horizon_s / dt)

    rows = []

    plt.figure(figsize=(9,8))

    plotted = 0

    for (sid, aid), g in df.groupby(["scenario_id","agent_id"]):

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

        min_dist, ttc, risk = compute_collision_metrics(
            pred,
            dt
        )

        uncertainty = float(
            np.sqrt(np.trace(covs[-1]))
        )

        rows.append({
            "scenario_id": sid,
            "agent_id": int(aid),
            "object_type": g.object_type.iloc[0],
            "minimum_future_distance_m": min_dist,
            "time_to_nearest_approach_s": ttc,
            "collision_risk": risk,
            "final_uncertainty": uncertainty,
        })

        if plotted < 30:

            plt.plot(
                pred[:,0],
                pred[:,1],
                alpha=0.7
            )

            idx = np.argmin(
                np.linalg.norm(pred, axis=1)
            )

            plt.scatter(
                pred[idx,0],
                pred[idx,1],
                marker="x",
                s=80
            )

            plotted += 1

    plt.scatter(
        [0],
        [0],
        marker="^",
        s=150,
        label="ego/headlamp"
    )

    plt.axis("equal")
    plt.xlabel("relative x (m)")
    plt.ylabel("relative y (m)")
    plt.title(
        "Near-collision forecasting for VRUs"
    )

    fig_path = savefig(
        "part_b_18_near_collision_forecasting.png"
    )

    out = pd.DataFrame(rows)

    top_risk = out.sort_values(
        "collision_risk",
        ascending=False
    ).head(10)

    metrics = {
        "num_agents": int(len(out)),
        "mean_minimum_distance_m":
            float(out.minimum_future_distance_m.mean()),
        "mean_time_to_nearest_approach_s":
            float(out.time_to_nearest_approach_s.mean()),
        "max_collision_risk":
            float(out.collision_risk.max()),
        "top_risk_agents":
            top_risk.to_dict(orient="records"),
        "figure":
            str(fig_path),
    }

    Path("outputs").mkdir(exist_ok=True)

    out.to_csv(
        "outputs/near_collision_forecasting.csv",
        index=False
    )

    with open(
        "outputs/near_collision_forecasting.json",
        "w"
    ) as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":

    metrics = run_near_collision_forecasting()

    print(json.dumps(metrics, indent=2))
