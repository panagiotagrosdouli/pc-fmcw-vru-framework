import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from womd.womd_loader import load_womd_directory
from part_b.prediction import kalman_predict
from common.plotting import savefig


def normalize_track(g):

    g = g.sort_values("t").copy()

    x0, y0 = g[["x","y"]].iloc[0].values

    g["x"] -= x0
    g["y"] -= y0

    g["x_meas"] -= x0
    g["y_meas"] -= y0

    return g


def gaussian_collision_probability(
    mean_xy,
    cov_xy,
    collision_radius=2.0
):

    sigma = np.sqrt(
        np.trace(cov_xy)
    )

    dist = np.linalg.norm(mean_xy)

    p = np.exp(
        -(dist**2) /
        (2 * (sigma + collision_radius)**2 + 1e-9)
    )

    return float(np.clip(p, 0, 1))


def time_to_collision(pred, dt=0.1):

    d = np.linalg.norm(pred, axis=1)

    idx = np.where(d < 2.0)[0]

    if len(idx) == 0:
        return None

    return float(idx[0] * dt)


def run_collision_probability_forecasting(
    womd_dir="data/womd/validation",
    max_files=2,
    max_scenarios=120,
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

    example = None

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

        try:

            pred, covs = kalman_predict(
                hist,
                fut.t.values
            )

        except Exception:
            continue

        probs = []

        for p, P in zip(pred, covs):

            prob = gaussian_collision_probability(
                p,
                P[:2,:2]
            )

            probs.append(prob)

        probs = np.array(probs)

        max_prob = float(np.max(probs))
        mean_prob = float(np.mean(probs))

        ttc = time_to_collision(
            pred,
            dt=dt
        )

        rows.append({
            "scenario_id": sid,
            "agent_id": int(aid),
            "object_type": g.object_type.iloc[0],
            "max_collision_probability": max_prob,
            "mean_collision_probability": mean_prob,
            "time_to_collision_s": ttc,
            "final_probability": float(probs[-1]),
        })

        if example is None:

            example = {
                "times": fut.t.values,
                "probs": probs
            }

    out = pd.DataFrame(rows)

    high_risk = out.sort_values(
        "max_collision_probability",
        ascending=False
    )

    plt.figure(figsize=(8,5))

    plt.plot(
        example["times"],
        example["probs"],
        linewidth=3
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Collision probability")

    plt.title(
        "Future collision probability forecast"
    )

    fig1 = savefig(
        "part_b_37_collision_probability_example.png"
    )

    plt.figure(figsize=(7,5))

    plt.hist(
        out.max_collision_probability,
        bins=25
    )

    plt.xlabel("Maximum collision probability")
    plt.ylabel("Count")

    plt.title(
        "Collision probability distribution"
    )

    fig2 = savefig(
        "part_b_38_collision_probability_distribution.png"
    )

    plt.figure(figsize=(7,5))

    valid = out.dropna(
        subset=["time_to_collision_s"]
    )

    if len(valid):

        plt.scatter(
            valid.time_to_collision_s,
            valid.max_collision_probability,
            alpha=0.7
        )

        plt.xlabel("Time-to-collision (s)")
        plt.ylabel("Max collision probability")

        plt.title(
            "TTC vs collision probability"
        )

    fig3 = savefig(
        "part_b_39_ttc_vs_collision_probability.png"
    )

    metrics = {
        "num_agents": int(len(out)),
        "mean_max_collision_probability":
            float(out.max_collision_probability.mean()),
        "highest_collision_probability":
            float(out.max_collision_probability.max()),
        "num_possible_collisions":
            int(out.time_to_collision_s.notna().sum()),
        "top_collision_cases":
            high_risk.head(10).to_dict(orient="records"),
        "figures": {
            "forecast_example": str(fig1),
            "distribution": str(fig2),
            "ttc_relation": str(fig3),
        }
    }

    Path("outputs").mkdir(exist_ok=True)

    out.to_csv(
        "outputs/collision_probability_forecasts.csv",
        index=False
    )

    with open(
        "outputs/collision_probability_forecasting.json",
        "w"
    ) as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":

    metrics = run_collision_probability_forecasting()

    print(json.dumps(metrics, indent=2))
