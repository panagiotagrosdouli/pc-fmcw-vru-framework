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


def complexity_features(g):
    x = g.x.values
    y = g.y.values
    dx = np.gradient(x)
    dy = np.gradient(y)

    heading = np.unwrap(np.arctan2(dy, dx))
    heading_change = np.degrees(np.max(heading) - np.min(heading))
    lateral_motion = float(np.max(x) - np.min(x))
    displacement = float(np.hypot(x[-1]-x[0], y[-1]-y[0]))
    crossing = bool(np.any(x < -0.5) and np.any(x > 0.5))

    return heading_change, lateral_motion, displacement, crossing


def run_scenario_difficulty(
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

        if max(cv_ade, cv_fde, kf_ade, kf_fde) > 50:
            continue

        heading_change, lateral_motion, displacement, crossing = complexity_features(seg)
        uncertainty = float(np.sqrt(np.trace(covs[-1])))

        rows.append({
            "scenario_id": sid,
            "agent_id": int(aid),
            "object_type": g.object_type.iloc[0],
            "best_ade": float(min(cv_ade, kf_ade)),
            "best_fde": float(min(cv_fde, kf_fde)),
            "heading_change_deg": float(heading_change),
            "lateral_motion_m": float(lateral_motion),
            "displacement_m": float(displacement),
            "crossing": crossing,
            "uncertainty": uncertainty,
            "is_cyclist": g.object_type.iloc[0] == "cyclist",
        })

    agent_df = pd.DataFrame(rows)

    scenario_rows = []
    for sid, g in agent_df.groupby("scenario_id"):
        difficulty = (
            0.30 * np.tanh(g.best_fde.mean() / 3)
            + 0.20 * np.tanh(g.heading_change_deg.mean() / 90)
            + 0.15 * np.tanh(g.lateral_motion_m.mean() / 10)
            + 0.15 * g.crossing.mean()
            + 0.10 * np.tanh(g.uncertainty.mean() / 5)
            + 0.10 * g.is_cyclist.mean()
        )

        scenario_rows.append({
            "scenario_id": sid,
            "num_agents": int(len(g)),
            "difficulty_score": float(difficulty),
            "mean_best_fde": float(g.best_fde.mean()),
            "mean_heading_change_deg": float(g.heading_change_deg.mean()),
            "crossing_rate": float(g.crossing.mean()),
            "cyclist_fraction": float(g.is_cyclist.mean()),
            "mean_uncertainty": float(g.uncertainty.mean()),
        })

    scenarios = pd.DataFrame(scenario_rows).sort_values("difficulty_score", ascending=False)

    plt.figure(figsize=(10,5))
    plt.bar(np.arange(min(20, len(scenarios))), scenarios.difficulty_score.head(20))
    plt.xlabel("Scenario rank")
    plt.ylabel("Difficulty score")
    plt.title("Top WOMD scenario difficulty scores")
    fig1 = savefig("part_b_29_scenario_difficulty_ranking.png")

    plt.figure(figsize=(7,5))
    plt.scatter(scenarios.mean_best_fde, scenarios.difficulty_score, alpha=0.7)
    plt.xlabel("Mean best FDE (m)")
    plt.ylabel("Difficulty score")
    plt.title("Prediction error vs scenario difficulty")
    fig2 = savefig("part_b_30_error_vs_difficulty.png")

    Path("outputs").mkdir(exist_ok=True)
    agent_df.to_csv("outputs/scenario_difficulty_agents.csv", index=False)
    scenarios.to_csv("outputs/scenario_difficulty_scores.csv", index=False)

    metrics = {
        "num_scenarios": int(len(scenarios)),
        "num_agents": int(len(agent_df)),
        "top_scenarios": scenarios.head(10).to_dict(orient="records"),
        "figures": {
            "ranking": str(fig1),
            "error_vs_difficulty": str(fig2),
        }
    }

    with open("outputs/scenario_difficulty_score.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    metrics = run_scenario_difficulty()
    print(json.dumps(metrics, indent=2))
