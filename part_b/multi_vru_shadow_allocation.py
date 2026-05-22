import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from womd.womd_loader import load_womd_directory
from part_b.prediction import kalman_predict
from part_b.vru_priority_queue import normalize_track, crossing_intent, compute_priority_score, angle_deg
from common.plotting import savefig


def merge_intervals(intervals):
    if not intervals:
        return []

    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [intervals[0]]

    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]

        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    return merged


def run_multi_vru_shadow_allocation(
    womd_dir="data/womd/validation",
    max_files=2,
    max_scenarios=120,
    horizon_s=5.0,
    dt=0.1,
    base_shadow_width_deg=2.0,
    uncertainty_gain=0.6,
    max_shadow_zones=8
):

    df = load_womd_directory(
        womd_dir,
        max_files=max_files,
        max_scenarios=max_scenarios
    )

    history_steps = int(1.0 / dt)
    future_steps = int(horizon_s / dt)

    candidates = []

    for (sid, aid), g in df.groupby(["scenario_id","agent_id"]):

        g = normalize_track(g)

        if len(g) < history_steps + future_steps:
            continue

        seg = g.iloc[:history_steps + future_steps]

        if np.max(np.diff(seg.t.values)) > 0.25:
            continue

        hist = seg.iloc[:history_steps]
        fut = seg.iloc[history_steps:]

        try:
            pred, covs = kalman_predict(hist, fut.t.values)
        except Exception:
            continue

        final = pred[-1]
        final_angle = float(angle_deg(final[0], final[1]))
        uncertainty = float(np.sqrt(np.trace(covs[-1])))

        dists = np.linalg.norm(pred, axis=1)
        min_distance = float(np.min(dists))

        dx = np.gradient(pred[:,0])
        dy = np.gradient(pred[:,1])
        speed = float(np.mean(np.hypot(dx,dy)) / dt)

        crossing = crossing_intent(pred)

        priority = compute_priority_score(
            min_distance,
            uncertainty,
            crossing,
            speed,
            final_angle
        )

        shadow_width = base_shadow_width_deg + uncertainty_gain * uncertainty

        candidates.append({
            "scenario_id": sid,
            "agent_id": int(aid),
            "object_type": g.object_type.iloc[0],
            "priority_score": priority,
            "final_angle_deg": final_angle,
            "shadow_width_deg": shadow_width,
            "uncertainty": uncertainty,
            "min_distance_m": min_distance,
            "crossing_intent": crossing,
        })

    out = pd.DataFrame(candidates)

    if len(out) == 0:
        raise RuntimeError("No valid VRU shadow candidates.")

    selected = (
        out.sort_values("priority_score", ascending=False)
        .head(max_shadow_zones)
        .copy()
    )

    intervals = []

    for _, r in selected.iterrows():
        intervals.append((
            r.final_angle_deg - r.shadow_width_deg,
            r.final_angle_deg + r.shadow_width_deg
        ))

    merged = merge_intervals(intervals)

    total_shadow_width = sum(end-start for start, end in merged)

    coverage_count = len(selected)

    angular_grid = np.linspace(-60, 60, 1000)
    intensity = np.ones_like(angular_grid)

    for start, end in merged:
        mask = (angular_grid >= start) & (angular_grid <= end)
        intensity[mask] = 0.1

    plt.figure(figsize=(10,4))
    plt.plot(angular_grid, intensity)
    for start, end in merged:
        plt.axvspan(start, end, alpha=0.2)

    plt.xlabel("Horizontal beam angle (deg)")
    plt.ylabel("Normalized intensity")
    plt.title("Prediction-aware multi-VRU shadow allocation")
    fig1 = savefig("part_b_25_multi_vru_shadow_profile.png")

    plt.figure(figsize=(8,5))
    plt.scatter(
        selected.final_angle_deg,
        selected.priority_score,
        s=120,
        alpha=0.8
    )

    for _, r in selected.iterrows():
        plt.text(
            r.final_angle_deg,
            r.priority_score,
            r.object_type,
            fontsize=8
        )

    plt.xlabel("Predicted final angle (deg)")
    plt.ylabel("Priority score")
    plt.title("Selected VRUs for shadow allocation")
    fig2 = savefig("part_b_26_selected_shadow_vrus.png")

    metrics = {
        "num_candidates": int(len(out)),
        "num_selected_vrus": int(len(selected)),
        "num_merged_shadow_zones": int(len(merged)),
        "total_shadow_width_deg": float(total_shadow_width),
        "mean_selected_priority": float(selected.priority_score.mean()),
        "merged_shadow_intervals": [
            [float(a), float(b)] for a,b in merged
        ],
        "figures": {
            "shadow_profile": str(fig1),
            "selected_vrus": str(fig2),
        }
    }

    Path("outputs").mkdir(exist_ok=True)

    out.to_csv(
        "outputs/multi_vru_shadow_candidates.csv",
        index=False
    )

    selected.to_csv(
        "outputs/multi_vru_shadow_selected.csv",
        index=False
    )

    with open(
        "outputs/multi_vru_shadow_allocation.json",
        "w"
    ) as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    metrics = run_multi_vru_shadow_allocation()
    print(json.dumps(metrics, indent=2))
