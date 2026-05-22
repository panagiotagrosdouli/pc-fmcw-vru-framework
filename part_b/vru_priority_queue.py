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


def angle_deg(x, y):

    return np.degrees(
        np.arctan2(x, y + 1e-6)
    )


def crossing_intent(pred):

    x = pred[:,0]

    return bool(
        np.any(x < -0.5)
        and np.any(x > 0.5)
    )


def compute_priority_score(
    min_distance,
    uncertainty,
    crossing,
    speed,
    angle
):

    distance_score = np.exp(-min_distance / 8)

    uncertainty_score = min(
        uncertainty / 5,
        1.0
    )

    crossing_score = 1.0 if crossing else 0.0

    speed_score = min(speed / 10, 1.0)

    frontal_score = np.exp(
        -abs(angle) / 15
    )

    priority = (
        0.35 * distance_score
        + 0.20 * uncertainty_score
        + 0.20 * crossing_score
        + 0.15 * speed_score
        + 0.10 * frontal_score
    )

    return float(priority)


def run_priority_queue(
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

    history_steps = int(1.0 / dt)
    future_steps = int(horizon_s / dt)

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

        dists = np.linalg.norm(pred, axis=1)

        min_distance = float(np.min(dists))

        uncertainty = float(
            np.sqrt(np.trace(covs[-1]))
        )

        crossing = crossing_intent(pred)

        dx = np.gradient(pred[:,0])
        dy = np.gradient(pred[:,1])

        speed = float(
            np.mean(np.hypot(dx,dy)) / dt
        )

        final_angle = float(
            angle_deg(
                pred[-1,0],
                pred[-1,1]
            )
        )

        priority = compute_priority_score(
            min_distance,
            uncertainty,
            crossing,
            speed,
            final_angle
        )

        rows.append({
            "scenario_id": sid,
            "agent_id": int(aid),
            "object_type": g.object_type.iloc[0],
            "minimum_distance_m": min_distance,
            "uncertainty": uncertainty,
            "crossing_intent": crossing,
            "speed_mps": speed,
            "final_angle_deg": final_angle,
            "priority_score": priority,
        })

    out = pd.DataFrame(rows)

    out = out.sort_values(
        "priority_score",
        ascending=False
    )

    top = out.head(25)

    plt.figure(figsize=(10,6))

    plt.bar(
        np.arange(len(top)),
        top.priority_score
    )

    plt.xticks(
        np.arange(len(top)),
        top.object_type,
        rotation=90
    )

    plt.ylabel("Priority score")
    plt.title(
        "VRU adaptive illumination priority queue"
    )

    fig1 = savefig(
        "part_b_21_vru_priority_queue.png"
    )

    plt.figure(figsize=(7,5))

    plt.scatter(
        out.minimum_distance_m,
        out.priority_score,
        alpha=0.6
    )

    plt.xlabel("Minimum predicted distance (m)")
    plt.ylabel("Priority score")

    plt.title(
        "Distance vs adaptive beam priority"
    )

    fig2 = savefig(
        "part_b_22_priority_vs_distance.png"
    )

    metrics = {
        "num_agents": int(len(out)),
        "highest_priority_agent":
            top.iloc[0].to_dict(),
        "mean_priority":
            float(out.priority_score.mean()),
        "crossing_fraction":
            float(out.crossing_intent.mean()),
        "figures": {
            "priority_queue": str(fig1),
            "priority_vs_distance": str(fig2),
        }
    }

    Path("outputs").mkdir(exist_ok=True)

    out.to_csv(
        "outputs/vru_priority_queue.csv",
        index=False
    )

    with open(
        "outputs/vru_priority_queue.json",
        "w"
    ) as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":

    metrics = run_priority_queue()

    print(json.dumps(metrics, indent=2))
