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


def beam_angle(x, y):

    return np.degrees(
        np.arctan2(x, y + 1e-6)
    )


def exponential_smoothing(x, alpha=0.25):

    y = np.zeros_like(x)

    y[0] = x[0]

    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1-alpha) * y[i-1]

    return y


def jerk_metric(signal):

    vel = np.gradient(signal)
    acc = np.gradient(vel)
    jerk = np.gradient(acc)

    return np.mean(np.abs(jerk))


def run_beam_smoothness(
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

    example_raw = None
    example_smooth = None
    example_t = None

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

        raw_angles = beam_angle(
            pred[:,0],
            pred[:,1]
        )

        smooth_angles = exponential_smoothing(
            raw_angles,
            alpha=0.22
        )

        raw_jitter = jerk_metric(raw_angles)
        smooth_jitter = jerk_metric(smooth_angles)

        jitter_reduction = 100 * (
            raw_jitter - smooth_jitter
        ) / max(raw_jitter, 1e-9)

        rows.append({
            "scenario_id": sid,
            "agent_id": int(aid),
            "object_type": g.object_type.iloc[0],
            "raw_jitter": float(raw_jitter),
            "smooth_jitter": float(smooth_jitter),
            "jitter_reduction_percent": float(jitter_reduction),
            "raw_angle_std_deg": float(np.std(raw_angles)),
            "smooth_angle_std_deg": float(np.std(smooth_angles)),
        })

        if example_raw is None:
            example_raw = raw_angles
            example_smooth = smooth_angles
            example_t = fut.t.values

    out = pd.DataFrame(rows)

    plt.figure(figsize=(9,5))

    plt.plot(
        example_t,
        example_raw,
        label="Raw beam steering"
    )

    plt.plot(
        example_t,
        example_smooth,
        label="Smoothed beam steering",
        linewidth=3
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Beam angle (deg)")
    plt.title("Beam anti-flicker smoothing")
    plt.legend()

    fig1 = savefig(
        "part_b_33_beam_smoothing_example.png"
    )

    plt.figure(figsize=(7,5))

    plt.hist(
        out.raw_jitter,
        bins=30,
        alpha=0.5,
        label="Raw"
    )

    plt.hist(
        out.smooth_jitter,
        bins=30,
        alpha=0.5,
        label="Smoothed"
    )

    plt.xlabel("Beam jitter")
    plt.ylabel("Count")
    plt.title("Beam jitter distribution")
    plt.legend()

    fig2 = savefig(
        "part_b_34_beam_jitter_distribution.png"
    )

    metrics = {
        "num_agents": int(len(out)),
        "mean_raw_jitter": float(out.raw_jitter.mean()),
        "mean_smooth_jitter": float(out.smooth_jitter.mean()),
        "mean_jitter_reduction_percent":
            float(out.jitter_reduction_percent.mean()),
        "best_jitter_reduction_percent":
            float(out.jitter_reduction_percent.max()),
        "figures": {
            "beam_smoothing": str(fig1),
            "jitter_distribution": str(fig2),
        }
    }

    Path("outputs").mkdir(exist_ok=True)

    out.to_csv(
        "outputs/beam_smoothness_metrics.csv",
        index=False
    )

    with open(
        "outputs/beam_smoothness.json",
        "w"
    ) as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":

    metrics = run_beam_smoothness()

    print(json.dumps(metrics, indent=2))
