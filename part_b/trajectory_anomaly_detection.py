import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from womd.womd_loader import load_womd_directory
from part_b.prediction import constant_velocity_predict, kalman_predict, ade_fde
from common.plotting import savefig


def normalize_track(g):
    g = g.sort_values("t").copy()
    x0, y0 = g[["x","y"]].iloc[0].values
    g["x"] -= x0
    g["y"] -= y0
    g["x_meas"] -= x0
    g["y_meas"] -= y0
    return g


def anomaly_features(g):
    x = g.x.values
    y = g.y.values
    dx = np.gradient(x)
    dy = np.gradient(y)

    speed = np.hypot(dx, dy) / 0.1
    accel = np.gradient(speed) / 0.1

    heading = np.unwrap(np.arctan2(dy, dx))
    heading_rate = np.gradient(heading) / 0.1

    jerk = np.gradient(accel) / 0.1

    return {
        "max_speed": float(np.max(speed)),
        "mean_speed": float(np.mean(speed)),
        "max_accel": float(np.max(np.abs(accel))),
        "max_heading_rate": float(np.max(np.abs(heading_rate))),
        "max_jerk": float(np.max(np.abs(jerk))),
    }


def run_anomaly_detection(
    womd_dir="data/womd/validation",
    max_files=2,
    max_scenarios=150,
    horizon_s=5.0,
    dt=0.1
):
    df = load_womd_directory(womd_dir, max_files=max_files, max_scenarios=max_scenarios)

    history_steps = int(1.0 / dt)
    future_steps = int(horizon_s / dt)

    rows = []

    for (sid, aid), g in df.groupby(["scenario_id","agent_id"]):
        g = normalize_track(g)

        if len(g) < history_steps + future_steps:
            continue

        seg = g.iloc[:history_steps + future_steps]
        if np.max(np.diff(seg.t.values)) > 0.25:
            continue

        hist = seg.iloc[:history_steps]
        fut = seg.iloc[history_steps:]
        gt = fut[["x","y"]].values

        try:
            cv = constant_velocity_predict(hist, fut.t.values)
            kf, covs = kalman_predict(hist, fut.t.values)
        except Exception:
            continue

        cv_ade, cv_fde = ade_fde(cv, gt)
        kf_ade, kf_fde = ade_fde(kf, gt)

        if max(cv_fde, kf_fde) > 50:
            continue

        f = anomaly_features(seg)

        prediction_residual = min(cv_fde, kf_fde)

        anomaly_score = (
            0.25 * np.tanh(f["max_accel"] / 5)
            + 0.25 * np.tanh(f["max_heading_rate"] / 3)
            + 0.20 * np.tanh(f["max_jerk"] / 20)
            + 0.30 * np.tanh(prediction_residual / 3)
        )

        rows.append({
            "scenario_id": sid,
            "agent_id": int(aid),
            "object_type": g.object_type.iloc[0],
            "prediction_residual_fde": float(prediction_residual),
            "anomaly_score": float(anomaly_score),
            **f
        })

    out = pd.DataFrame(rows).sort_values("anomaly_score", ascending=False)

    plt.figure(figsize=(8,5))
    plt.hist(out.anomaly_score, bins=30)
    plt.xlabel("Anomaly score")
    plt.ylabel("Count")
    plt.title("Trajectory anomaly score distribution")
    fig1 = savefig("part_b_50_anomaly_distribution.png")

    plt.figure(figsize=(7,5))
    plt.scatter(out.max_heading_rate, out.anomaly_score, alpha=0.7)
    plt.xlabel("Max heading rate")
    plt.ylabel("Anomaly score")
    plt.title("Heading-rate anomaly relation")
    fig2 = savefig("part_b_51_heading_anomaly_relation.png")

    metrics = {
        "num_agents": int(len(out)),
        "mean_anomaly_score": float(out.anomaly_score.mean()),
        "max_anomaly_score": float(out.anomaly_score.max()),
        "top_anomalies": out.head(10).to_dict(orient="records"),
        "figures": {
            "distribution": str(fig1),
            "heading_relation": str(fig2),
        }
    }

    Path("outputs").mkdir(exist_ok=True)
    out.to_csv("outputs/trajectory_anomaly_detection.csv", index=False)

    with open("outputs/trajectory_anomaly_detection.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    metrics = run_anomaly_detection()
    print(json.dumps(metrics, indent=2))
