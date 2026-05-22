import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from womd.womd_loader import load_womd_directory
from part_b.vru_priority_queue import normalize_track, compute_priority_score, crossing_intent, angle_deg
from part_b.prediction import kalman_predict
from common.plotting import savefig


def allocate_energy(priority_scores,
                    total_power=1.0,
                    min_power=0.05):

    scores = np.array(priority_scores)

    if len(scores) == 0:
        return np.array([])

    scores = scores - scores.min() + 1e-6

    if scores.sum() == 0:
        weights = np.ones_like(scores) / len(scores)
    else:
        weights = scores / scores.sum()

    powers = min_power + (total_power - min_power * len(scores)) * weights

    return np.clip(powers, min_power, 1.0)


def run_energy_aware_beam(
    womd_dir="data/womd/validation",
    max_files=2,
    max_scenarios=120,
    horizon_s=5.0,
    dt=0.1,
    total_power=1.0
):

    df = load_womd_directory(
        womd_dir,
        max_files=max_files,
        max_scenarios=max_scenarios
    )

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

        try:
            pred, covs = kalman_predict(hist, fut.t.values)
        except Exception:
            continue

        dists = np.linalg.norm(pred, axis=1)

        min_distance = float(np.min(dists))
        uncertainty = float(np.sqrt(np.trace(covs[-1])))
        crossing = crossing_intent(pred)

        dx = np.gradient(pred[:,0])
        dy = np.gradient(pred[:,1])
        speed = float(np.mean(np.hypot(dx,dy)) / dt)

        final_angle = float(angle_deg(pred[-1,0], pred[-1,1]))

        priority = compute_priority_score(
            min_distance,
            uncertainty,
            crossing,
            speed,
            final_angle
        )

        rows.append({
            "scenario_id": sid,
            "agent_id": int(aid),
            "object_type": g.object_type.iloc[0],
            "priority_score": priority,
            "minimum_distance_m": min_distance,
            "uncertainty": uncertainty,
            "crossing_intent": crossing,
            "speed_mps": speed,
            "final_angle_deg": final_angle,
        })

    out = pd.DataFrame(rows)

    if len(out) == 0:
        raise RuntimeError("No valid VRU tracks for energy-aware beam control.")

    out = out.sort_values("priority_score", ascending=False).head(40).copy()

    out["allocated_power"] = allocate_energy(
        out.priority_score.values,
        total_power=total_power
    )

    uniform_power = total_power / len(out)

    out["uniform_power"] = uniform_power

    out["power_gain_over_uniform"] = (
        out.allocated_power / uniform_power
    )

    safety_weighted_power = float(
        np.sum(out.allocated_power * out.priority_score)
    )

    uniform_safety_power = float(
        np.sum(out.uniform_power * out.priority_score)
    )

    improvement = 100 * (
        safety_weighted_power - uniform_safety_power
    ) / max(uniform_safety_power, 1e-9)

    plt.figure(figsize=(10,5))
    plt.bar(np.arange(len(out)), out.allocated_power, label="Risk-aware allocation")
    plt.axhline(uniform_power, ls="--", label="Uniform allocation")
    plt.xlabel("VRU priority rank")
    plt.ylabel("Allocated normalized beam power")
    plt.title("Energy-aware adaptive beam allocation")
    plt.legend()
    fig1 = savefig("part_b_23_energy_aware_allocation.png")

    plt.figure(figsize=(7,5))
    plt.scatter(out.priority_score, out.allocated_power, alpha=0.8)
    plt.xlabel("Priority score")
    plt.ylabel("Allocated beam power")
    plt.title("Priority-to-power beam mapping")
    fig2 = savefig("part_b_24_priority_power_mapping.png")

    metrics = {
        "num_agents_allocated": int(len(out)),
        "total_power_budget": float(total_power),
        "uniform_power_per_agent": float(uniform_power),
        "safety_weighted_power": safety_weighted_power,
        "uniform_safety_power": uniform_safety_power,
        "safety_weighted_improvement_percent": float(improvement),
        "top_power_agent": out.iloc[0].to_dict(),
        "figures": {
            "allocation": str(fig1),
            "priority_power_mapping": str(fig2),
        }
    }

    Path("outputs").mkdir(exist_ok=True)

    out.to_csv("outputs/energy_aware_beam_allocation.csv", index=False)

    with open("outputs/energy_aware_beam.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    metrics = run_energy_aware_beam()
    print(json.dumps(metrics, indent=2))
