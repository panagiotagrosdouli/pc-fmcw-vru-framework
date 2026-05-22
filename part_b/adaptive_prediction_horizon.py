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


def estimate_motion_class(g):
    x = g.x.values
    y = g.y.values
    dx = np.gradient(x)
    dy = np.gradient(y)

    speed = np.mean(np.hypot(dx, dy)) / 0.1
    lateral_motion = np.max(x) - np.min(x)

    heading = np.unwrap(np.arctan2(dy, dx))
    heading_change = np.degrees(np.max(heading) - np.min(heading))

    if speed < 0.5:
        return "stationary", 2.0

    if speed > 4.0 or heading_change > 40:
        return "high_dynamics", 3.0

    if lateral_motion > 4:
        return "crossing", 4.0

    return "smooth", 5.0


def evaluate_fixed_horizon(g, horizon_s, history_s=1.0, dt=0.1):
    h_steps = int(history_s / dt)
    f_steps = int(horizon_s / dt)

    if len(g) < h_steps + f_steps:
        return None

    seg = g.iloc[:h_steps + f_steps]

    if np.max(np.diff(seg.t.values)) > 0.25:
        return None

    hist = seg.iloc[:h_steps]
    fut = seg.iloc[h_steps:]

    gt = fut[["x","y"]].values

    cv = constant_velocity_predict(hist, fut.t.values)
    kf, covs = kalman_predict(hist, fut.t.values)

    cv_ade, cv_fde = ade_fde(cv, gt)
    kf_ade, kf_fde = ade_fde(kf, gt)

    return {
        "cv_ade": float(cv_ade),
        "cv_fde": float(cv_fde),
        "kf_ade": float(kf_ade),
        "kf_fde": float(kf_fde),
        "best_ade": float(min(cv_ade, kf_ade)),
        "best_fde": float(min(cv_fde, kf_fde)),
        "uncertainty": float(np.sqrt(np.trace(covs[-1]))),
    }


def run_adaptive_horizon(
    womd_dir="data/womd/validation",
    max_files=2,
    max_scenarios=150,
    dt=0.1
):
    df = load_womd_directory(womd_dir, max_files=max_files, max_scenarios=max_scenarios)

    rows = []

    for (sid, aid), g in df.groupby(["scenario_id","agent_id"]):
        g = normalize_track(g)

        if len(g) < 80:
            continue

        motion_class, adaptive_horizon = estimate_motion_class(g.iloc[:10])

        fixed_5 = evaluate_fixed_horizon(g, 5.0)
        adaptive = evaluate_fixed_horizon(g, adaptive_horizon)

        if fixed_5 is None or adaptive is None:
            continue

        if max(fixed_5["best_fde"], adaptive["best_fde"]) > 50:
            continue

        rows.append({
            "scenario_id": sid,
            "agent_id": int(aid),
            "object_type": g.object_type.iloc[0],
            "motion_class": motion_class,
            "adaptive_horizon_s": adaptive_horizon,
            "fixed5_best_ade": fixed_5["best_ade"],
            "fixed5_best_fde": fixed_5["best_fde"],
            "adaptive_best_ade": adaptive["best_ade"],
            "adaptive_best_fde": adaptive["best_fde"],
            "adaptive_uncertainty": adaptive["uncertainty"],
        })

    out = pd.DataFrame(rows)

    out["fde_reduction"] = out.fixed5_best_fde - out.adaptive_best_fde

    summary = out.groupby("motion_class").agg(
        num_agents=("agent_id", "count"),
        mean_horizon=("adaptive_horizon_s", "mean"),
        fixed5_fde=("fixed5_best_fde", "mean"),
        adaptive_fde=("adaptive_best_fde", "mean"),
        fde_reduction=("fde_reduction", "mean"),
    ).reset_index()

    plt.figure(figsize=(8,4))
    x = np.arange(len(summary))
    w = 0.35
    plt.bar(x-w/2, summary.fixed5_fde, w, label="Fixed 5s")
    plt.bar(x+w/2, summary.adaptive_fde, w, label="Adaptive horizon")
    plt.xticks(x, summary.motion_class, rotation=20)
    plt.ylabel("FDE (m)")
    plt.title("Adaptive prediction horizon vs fixed 5s")
    plt.legend()
    fig1 = savefig("part_b_31_adaptive_horizon_fde.png")

    plt.figure(figsize=(7,5))
    plt.scatter(out.adaptive_horizon_s, out.adaptive_best_fde, alpha=0.7)
    plt.xlabel("Selected horizon (s)")
    plt.ylabel("Adaptive FDE (m)")
    plt.title("Selected horizon vs prediction error")
    fig2 = savefig("part_b_32_horizon_vs_error.png")

    Path("outputs").mkdir(exist_ok=True)
    out.to_csv("outputs/adaptive_prediction_horizon_agents.csv", index=False)
    summary.to_csv("outputs/adaptive_prediction_horizon_summary.csv", index=False)

    metrics = {
        "num_agents": int(len(out)),
        "mean_fixed5_fde": float(out.fixed5_best_fde.mean()),
        "mean_adaptive_fde": float(out.adaptive_best_fde.mean()),
        "mean_fde_reduction": float(out.fde_reduction.mean()),
        "summary_by_motion_class": summary.to_dict(orient="records"),
        "figures": {
            "fde_comparison": str(fig1),
            "horizon_vs_error": str(fig2),
        }
    }

    with open("outputs/adaptive_prediction_horizon.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    metrics = run_adaptive_horizon()
    print(json.dumps(metrics, indent=2))
