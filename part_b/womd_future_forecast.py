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
    x0, y0 = g[["x", "y"]].iloc[0].values
    g["x"] -= x0
    g["y"] -= y0
    g["x_meas"] -= x0
    g["y_meas"] -= y0
    return g


def evaluate_forecast_horizon(df, history_s=1.0, horizon_s=3.0, dt=0.1):
    history_steps = int(history_s / dt)
    future_steps = int(horizon_s / dt)

    rows = []

    plt.figure(figsize=(9,8))

    plotted = 0

    for (scenario_id, agent_id), g in df.groupby(["scenario_id", "agent_id"]):
        g = normalize_track(g)

        if len(g) < history_steps + future_steps:
            continue

        hist = g.iloc[:history_steps]
        fut = g.iloc[history_steps:history_steps + future_steps]

        # require continuous trajectory
        if np.max(np.diff(g.t.values[:history_steps + future_steps])) > 0.25:
            continue

        future_times = fut.t.values
        gt = fut[["x", "y"]].values

        try:
            cv = constant_velocity_predict(hist, future_times)
            kf, covs = kalman_predict(hist, future_times)
        except Exception:
            continue

        cv_ade, cv_fde = ade_fde(cv, gt)
        kf_ade, kf_fde = ade_fde(kf, gt)

        if max(cv_ade, cv_fde, kf_ade, kf_fde) > 30:
            continue

        rows.append({
            "scenario_id": scenario_id,
            "agent_id": int(agent_id),
            "object_type": g.object_type.iloc[0],
            "horizon_s": horizon_s,
            "cv_ade": float(cv_ade),
            "cv_fde": float(cv_fde),
            "kf_ade": float(kf_ade),
            "kf_fde": float(kf_fde),
            "final_gt_x": float(gt[-1,0]),
            "final_gt_y": float(gt[-1,1]),
            "final_cv_x": float(cv[-1,0]),
            "final_cv_y": float(cv[-1,1]),
            "final_kf_x": float(kf[-1,0]),
            "final_kf_y": float(kf[-1,1]),
        })

        if plotted < 20:
            plt.plot(gt[:,0], gt[:,1], "k-", alpha=0.45)
            plt.plot(cv[:,0], cv[:,1], "--", alpha=0.65)
            plt.plot(kf[:,0], kf[:,1], "-", alpha=0.8)
            plt.scatter(hist.x, hist.y, s=12, alpha=0.7)
            plotted += 1

    plt.axis("equal")
    plt.xlabel("relative x (m)")
    plt.ylabel("relative y (m)")
    plt.title(f"WOMD VRU trajectory forecasting: {horizon_s:.0f}s horizon")
    fig_path = savefig(f"part_b_10_womd_forecast_{int(horizon_s)}s.png")

    out = pd.DataFrame(rows)

    metrics = {
        "horizon_s": horizon_s,
        "num_agents": int(len(out)),
        "cv_ade": float(out.cv_ade.mean()),
        "cv_fde": float(out.cv_fde.mean()),
        "kf_ade": float(out.kf_ade.mean()),
        "kf_fde": float(out.kf_fde.mean()),
        "figure": str(fig_path),
    }

    return out, metrics


def run_future_forecasting(
    womd_dir="data/womd/validation",
    max_files=2,
    max_scenarios=150
):
    df = load_womd_directory(
        womd_dir,
        max_files=max_files,
        max_scenarios=max_scenarios
    )

    results = {}
    all_rows = []

    for horizon in [3.0, 5.0]:
        rows, metrics = evaluate_forecast_horizon(df, horizon_s=horizon)
        results[f"{int(horizon)}s"] = metrics
        rows["horizon"] = horizon
        all_rows.append(rows)

    all_df = pd.concat(all_rows, ignore_index=True)

    Path("outputs").mkdir(exist_ok=True)

    all_df.to_csv("outputs/womd_3s_5s_forecasts.csv", index=False)

    with open("outputs/womd_3s_5s_forecasts.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    results = run_future_forecasting()
    print(json.dumps(results, indent=2))
