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


def motion_complexity(g):
    x = g.x.values
    y = g.y.values

    dx = np.gradient(x)
    dy = np.gradient(y)

    heading = np.unwrap(np.arctan2(dy, dx))
    heading_change = np.degrees(np.max(heading) - np.min(heading))

    lateral_motion = float(np.max(x) - np.min(x))
    displacement = float(np.hypot(x[-1]-x[0], y[-1]-y[0]))

    return heading_change, lateral_motion, displacement


def run_failure_case_analysis(
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

    plt.figure(figsize=(10,8))

    for (sid, aid), g in df.groupby(["scenario_id","agent_id"]):
        g = normalize_track(g)

        if len(g) < history_steps + future_steps:
            continue

        seg = g.iloc[:history_steps+future_steps]

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

        if max(cv_ade, cv_fde, kf_ade, kf_fde) > 50:
            continue

        heading_change, lateral_motion, displacement = motion_complexity(seg)

        crossing = bool(np.any(seg.x.values < -0.5) and np.any(seg.x.values > 0.5))
        final_uncertainty = float(np.sqrt(np.trace(covs[-1])))

        rows.append({
            "scenario_id": sid,
            "agent_id": int(aid),
            "object_type": g.object_type.iloc[0],
            "cv_ade": float(cv_ade),
            "cv_fde": float(cv_fde),
            "kf_ade": float(kf_ade),
            "kf_fde": float(kf_fde),
            "best_model": "CV" if cv_ade < kf_ade else "KF",
            "heading_change_deg": float(heading_change),
            "lateral_motion_m": float(lateral_motion),
            "displacement_m": float(displacement),
            "crossing": crossing,
            "final_uncertainty": final_uncertainty,
            "worst_error": float(max(cv_fde, kf_fde)),
        })

    out = pd.DataFrame(rows)

    failures = out.sort_values("worst_error", ascending=False).head(20)

    plt.scatter(
        out.heading_change_deg,
        out.worst_error,
        alpha=0.6
    )
    plt.xlabel("Heading change (deg)")
    plt.ylabel("Worst FDE error (m)")
    plt.title("Prediction failure analysis: maneuver complexity vs error")
    fig1 = savefig("part_b_27_failure_complexity_error.png")

    plt.figure(figsize=(8,5))
    out.groupby("object_type").worst_error.mean().plot(kind="bar")
    plt.ylabel("Mean worst FDE (m)")
    plt.title("Failure severity by VRU type")
    fig2 = savefig("part_b_28_failure_by_type.png")

    metrics = {
        "num_agents": int(len(out)),
        "mean_worst_error_m": float(out.worst_error.mean()),
        "max_worst_error_m": float(out.worst_error.max()),
        "cv_best_fraction": float((out.best_model == "CV").mean()),
        "kf_best_fraction": float((out.best_model == "KF").mean()),
        "crossing_failure_mean_error": float(
            out[out.crossing].worst_error.mean()
        ) if out.crossing.any() else None,
        "top_failures": failures.to_dict(orient="records"),
        "figures": {
            "complexity_error": str(fig1),
            "failure_by_type": str(fig2),
        }
    }

    Path("outputs").mkdir(exist_ok=True)

    out.to_csv("outputs/failure_case_analysis.csv", index=False)
    failures.to_csv("outputs/top_prediction_failures.csv", index=False)

    with open("outputs/failure_case_analysis.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    metrics = run_failure_case_analysis()
    print(json.dumps(metrics, indent=2))
