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


def is_continuous(g, max_gap=0.25):
    if len(g) < 2:
        return False
    return np.max(np.diff(g.t.values)) <= max_gap


def angle_deg(x, y):
    return np.degrees(np.arctan2(x, y + 1e-6))


def run_horizon_ablation(df, horizons=(1,2,3,5,8), history_s=1.0, dt=0.1):
    rows = []

    for horizon_s in horizons:
        h_steps = int(history_s / dt)
        f_steps = int(horizon_s / dt)

        for (sid, aid), g in df.groupby(["scenario_id", "agent_id"]):
            g = normalize_track(g)

            if len(g) < h_steps + f_steps:
                continue

            seg = g.iloc[:h_steps + f_steps]
            if not is_continuous(seg):
                continue

            hist = seg.iloc[:h_steps]
            fut = seg.iloc[h_steps:]

            future_times = fut.t.values
            gt = fut[["x", "y"]].values

            try:
                cv = constant_velocity_predict(hist, future_times)
                kf, covs = kalman_predict(hist, future_times)
            except Exception:
                continue

            cv_ade, cv_fde = ade_fde(cv, gt)
            kf_ade, kf_fde = ade_fde(kf, gt)

            if max(cv_ade, cv_fde, kf_ade, kf_fde) > 50:
                continue

            rows.append({
                "scenario_id": sid,
                "agent_id": int(aid),
                "object_type": g.object_type.iloc[0],
                "horizon_s": float(horizon_s),
                "cv_ade": float(cv_ade),
                "cv_fde": float(cv_fde),
                "kf_ade": float(kf_ade),
                "kf_fde": float(kf_fde),
                "kf_uncertainty_final": float(np.sqrt(np.trace(covs[-1]))),
            })

    out = pd.DataFrame(rows)
    summary = out.groupby("horizon_s").agg(
        num_agents=("agent_id", "count"),
        cv_ade=("cv_ade", "mean"),
        cv_fde=("cv_fde", "mean"),
        kf_ade=("kf_ade", "mean"),
        kf_fde=("kf_fde", "mean"),
        uncertainty=("kf_uncertainty_final", "mean"),
    ).reset_index()

    plt.figure(figsize=(8,4))
    plt.plot(summary.horizon_s, summary.cv_ade, "o-", label="CV ADE")
    plt.plot(summary.horizon_s, summary.kf_ade, "o-", label="KF ADE")
    plt.xlabel("Prediction horizon (s)")
    plt.ylabel("ADE (m)")
    plt.title("WOMD horizon ablation")
    plt.grid(alpha=0.3)
    plt.legend()
    fig1 = savefig("part_b_11_horizon_ablation_ade.png")

    plt.figure(figsize=(8,4))
    plt.plot(summary.horizon_s, summary.uncertainty, "o-")
    plt.xlabel("Prediction horizon (s)")
    plt.ylabel("Final KF uncertainty")
    plt.title("Prediction uncertainty growth over horizon")
    plt.grid(alpha=0.3)
    fig2 = savefig("part_b_12_uncertainty_growth.png")

    return out, summary, {"horizon_ade": str(fig1), "uncertainty_growth": str(fig2)}


def run_crossing_prediction(df, horizon_s=5.0, history_s=1.0, dt=0.1):
    h_steps = int(history_s / dt)
    f_steps = int(horizon_s / dt)
    rows = []

    for (sid, aid), g in df.groupby(["scenario_id", "agent_id"]):
        g = normalize_track(g)

        if len(g) < h_steps + f_steps:
            continue

        seg = g.iloc[:h_steps + f_steps]
        if not is_continuous(seg):
            continue

        hist = seg.iloc[:h_steps]
        fut = seg.iloc[h_steps:]
        future_times = fut.t.values

        gt = fut[["x","y"]].values
        cv = constant_velocity_predict(hist, future_times)
        kf, _ = kalman_predict(hist, future_times)

        gt_cross = bool(np.any(gt[:,0] < -0.5) and np.any(gt[:,0] > 0.5))
        cv_cross = bool(np.any(cv[:,0] < -0.5) and np.any(cv[:,0] > 0.5))
        kf_cross = bool(np.any(kf[:,0] < -0.5) and np.any(kf[:,0] > 0.5))

        rows.append({
            "scenario_id": sid,
            "agent_id": int(aid),
            "object_type": g.object_type.iloc[0],
            "gt_crossing": gt_cross,
            "cv_crossing": cv_cross,
            "kf_crossing": kf_cross,
        })

    out = pd.DataFrame(rows)

    def acc(pred):
        return float((out[pred] == out.gt_crossing).mean()) if len(out) else 0.0

    metrics = {
        "num_agents": int(len(out)),
        "gt_crossing_rate": float(out.gt_crossing.mean()) if len(out) else 0.0,
        "cv_crossing_accuracy": acc("cv_crossing"),
        "kf_crossing_accuracy": acc("kf_crossing"),
    }

    return out, metrics


def run_time_to_beam_conflict(df, beam_half_angle_deg=8.0, horizon_s=5.0, history_s=1.0, dt=0.1):
    h_steps = int(history_s / dt)
    f_steps = int(horizon_s / dt)
    rows = []

    for (sid, aid), g in df.groupby(["scenario_id", "agent_id"]):
        g = normalize_track(g)

        if len(g) < h_steps + f_steps:
            continue

        seg = g.iloc[:h_steps + f_steps]
        if not is_continuous(seg):
            continue

        hist = seg.iloc[:h_steps]
        fut = seg.iloc[h_steps:]
        future_times = fut.t.values

        gt = fut[["x","y"]].values
        kf, covs = kalman_predict(hist, future_times)

        gt_angle = angle_deg(gt[:,0], gt[:,1])
        pred_angle = angle_deg(kf[:,0], kf[:,1])

        gt_conflict_idx = np.where(np.abs(gt_angle) < beam_half_angle_deg)[0]
        pred_conflict_idx = np.where(np.abs(pred_angle) < beam_half_angle_deg)[0]

        gt_ttc = float(gt_conflict_idx[0] * dt) if len(gt_conflict_idx) else None
        pred_ttc = float(pred_conflict_idx[0] * dt) if len(pred_conflict_idx) else None

        if gt_ttc is not None and pred_ttc is not None:
            err = abs(gt_ttc - pred_ttc)
        else:
            err = None

        rows.append({
            "scenario_id": sid,
            "agent_id": int(aid),
            "object_type": g.object_type.iloc[0],
            "gt_time_to_conflict_s": gt_ttc,
            "pred_time_to_conflict_s": pred_ttc,
            "ttc_error_s": err,
            "final_uncertainty": float(np.sqrt(np.trace(covs[-1]))),
        })

    out = pd.DataFrame(rows)
    valid = out.dropna(subset=["ttc_error_s"])

    metrics = {
        "num_agents": int(len(out)),
        "num_conflict_agents": int(len(valid)),
        "mean_ttc_error_s": float(valid.ttc_error_s.mean()) if len(valid) else None,
        "median_ttc_error_s": float(valid.ttc_error_s.median()) if len(valid) else None,
    }

    if len(valid):
        plt.figure(figsize=(7,4))
        plt.hist(valid.ttc_error_s, bins=15, alpha=0.8)
        plt.xlabel("Time-to-beam-conflict error (s)")
        plt.ylabel("Count")
        plt.title("Predicted time-to-beam-conflict error")
        fig = savefig("part_b_13_time_to_beam_conflict.png")
        metrics["figure"] = str(fig)

    return out, metrics


def run_maneuver_detection(df, horizon_s=5.0, history_s=1.0, dt=0.1):
    h_steps = int(history_s / dt)
    f_steps = int(horizon_s / dt)
    rows = []

    for (sid, aid), g in df.groupby(["scenario_id", "agent_id"]):
        g = normalize_track(g)

        if len(g) < h_steps + f_steps:
            continue

        seg = g.iloc[:h_steps + f_steps]
        if not is_continuous(seg):
            continue

        x = seg.x.values
        y = seg.y.values
        heading = np.unwrap(np.arctan2(np.gradient(y), np.gradient(x)))
        heading_change = np.degrees(np.max(heading) - np.min(heading))
        lateral_motion = float(np.max(x) - np.min(x))
        displacement = float(np.hypot(x[-1]-x[0], y[-1]-y[0]))

        maneuver = bool(abs(heading_change) > 25 or lateral_motion > 5)

        rows.append({
            "scenario_id": sid,
            "agent_id": int(aid),
            "object_type": g.object_type.iloc[0],
            "heading_change_deg": float(heading_change),
            "lateral_motion_m": lateral_motion,
            "displacement_m": displacement,
            "maneuver_detected": maneuver,
        })

    out = pd.DataFrame(rows)

    metrics = {
        "num_agents": int(len(out)),
        "maneuver_rate": float(out.maneuver_detected.mean()) if len(out) else 0.0,
        "mean_heading_change_deg": float(out.heading_change_deg.mean()) if len(out) else 0.0,
    }

    return out, metrics


def run_all_advanced(womd_dir="data/womd/validation", max_files=2, max_scenarios=150):
    df = load_womd_directory(womd_dir, max_files=max_files, max_scenarios=max_scenarios)

    horizon_rows, horizon_summary, figs = run_horizon_ablation(df)
    crossing_rows, crossing_metrics = run_crossing_prediction(df)
    ttc_rows, ttc_metrics = run_time_to_beam_conflict(df)
    maneuver_rows, maneuver_metrics = run_maneuver_detection(df)

    Path("outputs").mkdir(exist_ok=True)

    horizon_rows.to_csv("outputs/womd_horizon_ablation_rows.csv", index=False)
    horizon_summary.to_csv("outputs/womd_horizon_ablation_summary.csv", index=False)
    crossing_rows.to_csv("outputs/womd_crossing_prediction.csv", index=False)
    ttc_rows.to_csv("outputs/womd_time_to_beam_conflict.csv", index=False)
    maneuver_rows.to_csv("outputs/womd_maneuver_detection.csv", index=False)

    results = {
        "dataset": {
            "max_files": max_files,
            "max_scenarios": max_scenarios,
            "num_scenarios": int(df.scenario_id.nunique()),
            "num_agents": int(df.groupby(["scenario_id","agent_id"]).ngroups),
            "object_counts": {k:int(v) for k,v in df.object_type.value_counts().items()},
        },
        "horizon_ablation": horizon_summary.to_dict(orient="records"),
        "crossing_prediction": crossing_metrics,
        "time_to_beam_conflict": ttc_metrics,
        "maneuver_detection": maneuver_metrics,
        "figures": figs,
    }

    with open("outputs/womd_advanced_predictions.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    results = run_all_advanced()
    print(json.dumps(results, indent=2))
