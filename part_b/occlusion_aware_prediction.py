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


def angle_deg(x, y):

    return np.degrees(
        np.arctan2(x, y + 1e-9)
    )


def compute_occlusion_probability(
    target_pos,
    other_positions
):

    tx, ty = target_pos

    target_angle = angle_deg(tx, ty)
    target_dist = np.hypot(tx, ty)

    occlusion = 0.0

    for ox, oy in other_positions:

        other_angle = angle_deg(ox, oy)
        other_dist = np.hypot(ox, oy)

        angle_diff = abs(
            target_angle - other_angle
        )

        if (
            angle_diff < 4
            and other_dist < target_dist
        ):

            overlap = np.exp(
                -angle_diff / 2
            )

            depth = np.exp(
                -(target_dist - other_dist)/8
            )

            occlusion += overlap * depth

    return float(
        np.clip(occlusion, 0, 1)
    )


def run_occlusion_aware_prediction(
    womd_dir="data/womd/validation",
    max_files=2,
    max_scenarios=80,
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

    inflation_curves = []

    for sid, sdf in df.groupby(
        "scenario_id"
    ):

        tracks = {}

        for aid, g in sdf.groupby("agent_id"):

            g = normalize_track(g)

            if len(g) < history_steps + future_steps:
                continue

            seg = g.iloc[
                :history_steps + future_steps
            ]

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

            tracks[aid] = {
                "pred": pred,
                "covs": covs,
                "type": g.object_type.iloc[0]
            }

        aids = list(tracks.keys())

        for aid in aids:

            target = tracks[aid]

            pred = target["pred"]
            covs = target["covs"]

            inflation = []

            for t in range(len(pred)):

                others = []

                for oid in aids:

                    if oid == aid:
                        continue

                    others.append(
                        tracks[oid]["pred"][t]
                    )

                occ = compute_occlusion_probability(
                    pred[t],
                    others
                )

                base_unc = np.sqrt(
                    np.trace(covs[t][:2,:2])
                )

                inflated_unc = (
                    base_unc
                    * (1 + 2*occ)
                )

                inflation.append(
                    inflated_unc / (base_unc + 1e-9)
                )

            inflation = np.array(inflation)

            inflation_curves.append(inflation)

            rows.append({
                "scenario_id": sid,
                "agent_id": int(aid),
                "object_type": target["type"],
                "mean_uncertainty_inflation":
                    float(np.mean(inflation)),
                "max_uncertainty_inflation":
                    float(np.max(inflation)),
                "mean_occlusion_probability":
                    float(np.mean(inflation-1)/2),
            })

    out = pd.DataFrame(rows)

    plt.figure(figsize=(9,5))

    for curve in inflation_curves[:12]:

        plt.plot(
            curve,
            alpha=0.5
        )

    plt.xlabel("Prediction step")
    plt.ylabel("Uncertainty inflation")

    plt.title(
        "Occlusion-aware uncertainty inflation"
    )

    fig1 = savefig(
        "part_b_48_occlusion_uncertainty.png"
    )

    plt.figure(figsize=(7,5))

    plt.hist(
        out.mean_occlusion_probability,
        bins=25
    )

    plt.xlabel("Mean occlusion probability")
    plt.ylabel("Count")

    plt.title(
        "VRU occlusion probability distribution"
    )

    fig2 = savefig(
        "part_b_49_occlusion_distribution.png"
    )

    metrics = {
        "num_agents":
            int(len(out)),
        "mean_occlusion_probability":
            float(out.mean_occlusion_probability.mean()),
        "max_occlusion_probability":
            float(out.mean_occlusion_probability.max()),
        "mean_uncertainty_inflation":
            float(out.mean_uncertainty_inflation.mean()),
        "figures": {
            "uncertainty_inflation": str(fig1),
            "occlusion_distribution": str(fig2),
        }
    }

    Path("outputs").mkdir(exist_ok=True)

    out.to_csv(
        "outputs/occlusion_aware_prediction.csv",
        index=False
    )

    with open(
        "outputs/occlusion_aware_prediction.json",
        "w"
    ) as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":

    metrics = run_occlusion_aware_prediction()

    print(json.dumps(metrics, indent=2))
