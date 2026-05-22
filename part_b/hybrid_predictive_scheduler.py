import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from womd.womd_loader import load_womd_directory
from part_b.prediction import (
    constant_velocity_predict,
    kalman_predict,
    ade_fde,
)
from common.plotting import savefig


def normalize_track(g):
    g = g.sort_values("t").copy()

    x0, y0 = g[["x", "y"]].iloc[0].values

    g["x"] -= x0
    g["y"] -= y0
    g["x_meas"] -= x0
    g["y_meas"] -= y0

    return g


def detect_maneuver(hist):
    x = hist.x.values
    y = hist.y.values

    if len(x) < 5:
        return False

    heading = np.unwrap(np.arctan2(np.gradient(y), np.gradient(x)))
    heading_change = np.degrees(np.max(heading) - np.min(heading))

    lateral_motion = np.max(x) - np.min(x)

    return bool(
        heading_change > 20
        or lateral_motion > 3
    )


def hybrid_predict(hist, future_times):

    maneuver = detect_maneuver(hist)

    if maneuver:
        pred, covs = kalman_predict(hist, future_times)
        method = "KF"
    else:
        pred = constant_velocity_predict(hist, future_times)
        covs = [np.eye(4)] * len(pred)
        method = "CV"

    return pred, covs, method


def angle_deg(x, y):
    return np.degrees(np.arctan2(x, y + 1e-6))


def compute_beam_schedule(pred_xy,
                          beam_half_angle_deg=8):

    angles = angle_deg(pred_xy[:,0], pred_xy[:,1])

    inside = np.abs(angles) < beam_half_angle_deg

    if np.any(inside):
        idx = np.where(inside)[0][0]
        return idx
    else:
        return None


def run_scheduler(df,
                  history_s=1.0,
                  horizon_s=5.0,
                  dt=0.1):

    history_steps = int(history_s / dt)
    future_steps = int(horizon_s / dt)

    rows = []

    plt.figure(figsize=(10,8))

    plotted = 0

    for (sid, aid), g in df.groupby(["scenario_id", "agent_id"]):

        g = normalize_track(g)

        if len(g) < history_steps + future_steps:
            continue

        seg = g.iloc[:history_steps + future_steps]

        if np.max(np.diff(seg.t.values)) > 0.25:
            continue

        hist = seg.iloc[:history_steps]
        fut = seg.iloc[history_steps:]

        gt = fut[["x","y"]].values
        future_times = fut.t.values

        try:
            pred, covs, method = hybrid_predict(
                hist,
                future_times
            )
        except Exception:
            continue

        ade, fde = ade_fde(pred, gt)

        if max(ade, fde) > 25:
            continue

        gt_schedule = compute_beam_schedule(gt)
        pred_schedule = compute_beam_schedule(pred)

        if gt_schedule is not None and pred_schedule is not None:
            timing_error = abs(gt_schedule - pred_schedule) * dt
            proactive_gain = max(gt_schedule - pred_schedule, 0) * dt
        else:
            timing_error = None
            proactive_gain = 0.0

        final_unc = float(np.sqrt(np.trace(covs[-1])))

        rows.append({
            "scenario_id": sid,
            "agent_id": int(aid),
            "object_type": g.object_type.iloc[0],
            "predictor": method,
            "ade": float(ade),
            "fde": float(fde),
            "beam_timing_error_s": timing_error,
            "proactive_gain_s": proactive_gain,
            "final_uncertainty": final_unc,
        })

        if plotted < 20:

            plt.plot(gt[:,0], gt[:,1],
                     "k-", alpha=0.35)

            if method == "KF":
                style = "-"
            else:
                style = "--"

            plt.plot(pred[:,0], pred[:,1],
                     style, alpha=0.8)

            plt.scatter(hist.x,
                        hist.y,
                        s=10)

            plotted += 1

    plt.axis("equal")
    plt.xlabel("relative x (m)")
    plt.ylabel("relative y (m)")
    plt.title(
        "Hybrid predictive scheduling for adaptive beam control"
    )

    fig_path = savefig(
        "part_b_14_hybrid_predictive_scheduler.png"
    )

    out = pd.DataFrame(rows)

    valid = out.dropna(subset=["beam_timing_error_s"])

    metrics = {
        "num_agents": int(len(out)),
        "cv_fraction":
            float((out.predictor == "CV").mean()),
        "kf_fraction":
            float((out.predictor == "KF").mean()),
        "mean_ade":
            float(out.ade.mean()),
        "mean_fde":
            float(out.fde.mean()),
        "mean_beam_timing_error_s":
            float(valid.beam_timing_error_s.mean())
            if len(valid) else None,
        "mean_proactive_gain_s":
            float(out.proactive_gain_s.mean()),
        "mean_final_uncertainty":
            float(out.final_uncertainty.mean()),
        "figure":
            str(fig_path),
    }

    Path("outputs").mkdir(exist_ok=True)

    out.to_csv(
        "outputs/hybrid_predictive_scheduler.csv",
        index=False
    )

    with open(
        "outputs/hybrid_predictive_scheduler.json",
        "w"
    ) as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":

    df = load_womd_directory(
        "data/womd/validation",
        max_files=2,
        max_scenarios=150
    )

    metrics = run_scheduler(df)

    print(json.dumps(metrics, indent=2))
