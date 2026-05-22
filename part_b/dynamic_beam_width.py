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

    return g


def collision_risk(pred):

    d = np.linalg.norm(pred, axis=1)

    return float(
        np.exp(-np.min(d)/5)
    )


def beam_width_policy(
    uncertainty,
    risk,
    interaction_strength
):

    width = (
        2.0
        + 1.2 * uncertainty
        + 4.0 * risk
        + 2.5 * interaction_strength
    )

    return float(
        np.clip(width, 2, 18)
    )


def interaction_strength_estimate(df, sid):

    sdf = df[df.scenario_id == sid]

    if len(sdf.agent_id.unique()) <= 1:
        return 0.0

    density = len(sdf.agent_id.unique()) / 10

    return float(
        np.clip(density, 0, 1)
    )


def run_dynamic_beam_width(
    womd_dir="data/womd/validation",
    max_files=2,
    max_scenarios=100,
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

    width_curves = []

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

        interaction = interaction_strength_estimate(
            df,
            sid
        )

        widths = []

        for p, P in zip(pred, covs):

            uncertainty = np.sqrt(
                np.trace(P[:2,:2])
            )

            risk = np.exp(
                -np.linalg.norm(p)/6
            )

            width = beam_width_policy(
                uncertainty,
                risk,
                interaction
            )

            widths.append(width)

        widths = np.array(widths)

        width_curves.append(widths)

        rows.append({
            "scenario_id": sid,
            "agent_id": int(aid),
            "object_type": g.object_type.iloc[0],
            "mean_beam_width_deg":
                float(np.mean(widths)),
            "max_beam_width_deg":
                float(np.max(widths)),
            "min_beam_width_deg":
                float(np.min(widths)),
            "interaction_strength":
                interaction,
            "collision_risk":
                collision_risk(pred),
        })

    out = pd.DataFrame(rows)

    plt.figure(figsize=(9,5))

    for w in width_curves[:12]:

        plt.plot(
            np.arange(len(w))*dt,
            w,
            alpha=0.5
        )

    plt.xlabel("Time (s)")
    plt.ylabel("Beam width (deg)")

    plt.title(
        "Dynamic beam width evolution"
    )

    fig1 = savefig(
        "part_b_46_dynamic_beam_width.png"
    )

    plt.figure(figsize=(7,5))

    plt.scatter(
        out.collision_risk,
        out.mean_beam_width_deg,
        alpha=0.7
    )

    plt.xlabel("Collision risk")
    plt.ylabel("Mean beam width (deg)")

    plt.title(
        "Risk-aware beam widening"
    )

    fig2 = savefig(
        "part_b_47_risk_vs_beam_width.png"
    )

    metrics = {
        "num_agents":
            int(len(out)),
        "mean_beam_width_deg":
            float(out.mean_beam_width_deg.mean()),
        "max_beam_width_deg":
            float(out.max_beam_width_deg.max()),
        "mean_collision_risk":
            float(out.collision_risk.mean()),
        "figures": {
            "beam_width_evolution": str(fig1),
            "risk_relation": str(fig2),
        }
    }

    Path("outputs").mkdir(exist_ok=True)

    out.to_csv(
        "outputs/dynamic_beam_width.csv",
        index=False
    )

    with open(
        "outputs/dynamic_beam_width.json",
        "w"
    ) as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":

    metrics = run_dynamic_beam_width()

    print(json.dumps(metrics, indent=2))
