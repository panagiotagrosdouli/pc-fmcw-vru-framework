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


def compute_iape(
    pred,
    gt,
    covs,
    beam_half_angle_deg=8,
    dt=0.1
):

    # --- trajectory error ---
    ade, fde = ade_fde(pred, gt)

    # --- angular beam error ---
    pred_ang = angle_deg(pred[:,0], pred[:,1])
    gt_ang = angle_deg(gt[:,0], gt[:,1])

    angular_error = np.mean(
        np.abs(pred_ang - gt_ang)
    )

    # --- uncertainty penalty ---
    uncertainty = np.mean([
        np.sqrt(np.trace(P))
        for P in covs
    ])

    # --- beam conflict timing ---
    pred_inside = np.abs(pred_ang) < beam_half_angle_deg
    gt_inside = np.abs(gt_ang) < beam_half_angle_deg

    if np.any(pred_inside):
        pred_t = np.where(pred_inside)[0][0] * dt
    else:
        pred_t = 999

    if np.any(gt_inside):
        gt_t = np.where(gt_inside)[0][0] * dt
    else:
        gt_t = 999

    timing_penalty = abs(pred_t - gt_t)

    # --- final IAPE metric ---
    iape = (
        1.0 * ade
        + 0.5 * fde
        + 0.05 * angular_error
        + 0.3 * uncertainty
        + 0.8 * timing_penalty
    )

    return {
        "ade": float(ade),
        "fde": float(fde),
        "angular_error_deg": float(angular_error),
        "uncertainty": float(uncertainty),
        "timing_penalty_s": float(timing_penalty),
        "iape": float(iape),
    }


def evaluate_predictor(
    predictor_name,
    predictor_fn,
    df,
    horizon_s=5.0,
    history_s=1.0,
    dt=0.1
):

    history_steps = int(history_s / dt)
    future_steps = int(horizon_s / dt)

    rows = []

    for (sid, aid), g in df.groupby(
        ["scenario_id", "agent_id"]
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

            if predictor_name == "CV":

                pred = constant_velocity_predict(
                    hist,
                    fut.t.values
                )

                covs = [np.eye(4)] * len(pred)

            else:

                pred, covs = kalman_predict(
                    hist,
                    fut.t.values
                )

        except Exception:
            continue

        if np.max(np.linalg.norm(pred - gt, axis=1)) > 40:
            continue

        metrics = compute_iape(
            pred,
            gt,
            covs,
            dt=dt
        )

        rows.append({
            "scenario_id": sid,
            "agent_id": int(aid),
            "object_type": g.object_type.iloc[0],
            "predictor": predictor_name,
            **metrics
        })

    return pd.DataFrame(rows)


def run_iape_analysis(
    womd_dir="data/womd/validation",
    max_files=2,
    max_scenarios=120
):

    df = load_womd_directory(
        womd_dir,
        max_files=max_files,
        max_scenarios=max_scenarios
    )

    cv_df = evaluate_predictor(
        "CV",
        constant_velocity_predict,
        df
    )

    kf_df = evaluate_predictor(
        "KF",
        kalman_predict,
        df
    )

    out = pd.concat(
        [cv_df, kf_df],
        ignore_index=True
    )

    summary = (
        out.groupby("predictor")
        .agg({
            "ade":"mean",
            "fde":"mean",
            "angular_error_deg":"mean",
            "uncertainty":"mean",
            "timing_penalty_s":"mean",
            "iape":"mean",
        })
        .reset_index()
    )

    plt.figure(figsize=(7,5))

    plt.bar(
        summary.predictor,
        summary.iape
    )

    plt.ylabel("IAPE")
    plt.title(
        "Illumination-Aware Prediction Error"
    )

    fig1 = savefig(
        "part_b_19_iape_comparison.png"
    )

    plt.figure(figsize=(8,5))

    for name, g in out.groupby("predictor"):

        plt.hist(
            g.iape,
            bins=30,
            alpha=0.5,
            label=name
        )

    plt.xlabel("IAPE")
    plt.ylabel("Count")
    plt.title("IAPE distribution")
    plt.legend()

    fig2 = savefig(
        "part_b_20_iape_distribution.png"
    )

    metrics = {
        "predictor_summary":
            summary.to_dict(orient="records"),
        "best_predictor":
            summary.sort_values("iape")
            .iloc[0]["predictor"],
        "figures": {
            "comparison": str(fig1),
            "distribution": str(fig2),
        }
    }

    Path("outputs").mkdir(exist_ok=True)

    out.to_csv(
        "outputs/iape_per_agent.csv",
        index=False
    )

    summary.to_csv(
        "outputs/iape_summary.csv",
        index=False
    )

    with open(
        "outputs/iape_metrics.json",
        "w"
    ) as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":

    metrics = run_iape_analysis()

    print(json.dumps(metrics, indent=2))
