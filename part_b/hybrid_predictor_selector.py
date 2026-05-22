import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from womd.womd_loader import load_womd_directory
from part_b.prediction import (
    constant_velocity_predict,
    kalman_predict,
    ade_fde
)
from common.plotting import savefig


def normalize_track(g):

    g = g.sort_values("t").copy()

    x0, y0 = g[["x","y"]].iloc[0].values

    g["x"] -= x0
    g["y"] -= y0

    g["x_meas"] -= x0
    g["y_meas"] -= y0

    return g


def motion_features(g):

    x = g.x.values
    y = g.y.values

    dx = np.gradient(x)
    dy = np.gradient(y)

    speed = np.mean(np.hypot(dx,dy)) / 0.1

    heading = np.unwrap(
        np.arctan2(dy, dx)
    )

    heading_change = np.degrees(
        np.max(heading) - np.min(heading)
    )

    lateral_motion = np.max(x) - np.min(x)

    return {
        "speed": float(speed),
        "heading_change": float(heading_change),
        "lateral_motion": float(lateral_motion),
    }


def choose_predictor(features):

    if (
        features["speed"] > 4.5
        or features["heading_change"] > 35
        or features["lateral_motion"] > 4
    ):
        return "KF"

    return "CV"


def run_hybrid_selector(
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

        gt = fut[["x","y"]].values

        try:

            cv_pred = constant_velocity_predict(
                hist,
                fut.t.values
            )

            kf_pred, covs = kalman_predict(
                hist,
                fut.t.values
            )

        except Exception:
            continue

        cv_ade, cv_fde = ade_fde(
            cv_pred,
            gt
        )

        kf_ade, kf_fde = ade_fde(
            kf_pred,
            gt
        )

        if max(cv_fde, kf_fde) > 50:
            continue

        features = motion_features(hist)

        selected = choose_predictor(features)

        if selected == "CV":
            hybrid_ade = cv_ade
            hybrid_fde = cv_fde
        else:
            hybrid_ade = kf_ade
            hybrid_fde = kf_fde

        oracle_best = min(cv_fde, kf_fde)

        rows.append({
            "scenario_id": sid,
            "agent_id": int(aid),
            "object_type": g.object_type.iloc[0],
            "selected_predictor": selected,
            "speed": features["speed"],
            "heading_change": features["heading_change"],
            "lateral_motion": features["lateral_motion"],
            "cv_fde": float(cv_fde),
            "kf_fde": float(kf_fde),
            "hybrid_fde": float(hybrid_fde),
            "oracle_best_fde": float(oracle_best),
        })

    out = pd.DataFrame(rows)

    out["hybrid_gain_vs_cv"] = (
        out.cv_fde - out.hybrid_fde
    )

    out["hybrid_gain_vs_kf"] = (
        out.kf_fde - out.hybrid_fde
    )

    predictor_usage = (
        out.selected_predictor
        .value_counts()
    )

    plt.figure(figsize=(7,5))

    predictor_usage.plot(kind="bar")

    plt.ylabel("Count")

    plt.title(
        "Hybrid predictor selection"
    )

    fig1 = savefig(
        "part_b_40_hybrid_predictor_usage.png"
    )

    plt.figure(figsize=(8,5))

    plt.hist(
        out.cv_fde,
        bins=30,
        alpha=0.5,
        label="CV"
    )

    plt.hist(
        out.kf_fde,
        bins=30,
        alpha=0.5,
        label="KF"
    )

    plt.hist(
        out.hybrid_fde,
        bins=30,
        alpha=0.5,
        label="Hybrid"
    )

    plt.xlabel("FDE (m)")
    plt.ylabel("Count")

    plt.title(
        "Hybrid predictor performance"
    )

    plt.legend()

    fig2 = savefig(
        "part_b_41_hybrid_fde_distribution.png"
    )

    metrics = {
        "num_agents": int(len(out)),
        "mean_cv_fde":
            float(out.cv_fde.mean()),
        "mean_kf_fde":
            float(out.kf_fde.mean()),
        "mean_hybrid_fde":
            float(out.hybrid_fde.mean()),
        "mean_oracle_best_fde":
            float(out.oracle_best_fde.mean()),
        "hybrid_vs_cv_gain":
            float(out.hybrid_gain_vs_cv.mean()),
        "hybrid_vs_kf_gain":
            float(out.hybrid_gain_vs_kf.mean()),
        "predictor_usage":
            predictor_usage.to_dict(),
        "figures": {
            "usage": str(fig1),
            "fde_distribution": str(fig2),
        }
    }

    Path("outputs").mkdir(exist_ok=True)

    out.to_csv(
        "outputs/hybrid_predictor_selector.csv",
        index=False
    )

    with open(
        "outputs/hybrid_predictor_selector.json",
        "w"
    ) as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":

    metrics = run_hybrid_selector()

    print(json.dumps(metrics, indent=2))
