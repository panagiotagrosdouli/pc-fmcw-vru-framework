import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

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


def ellipse_area(cov):

    vals, _ = np.linalg.eigh(cov[:2,:2])

    vals = np.maximum(vals, 1e-6)

    a = 2 * np.sqrt(vals[1])
    b = 2 * np.sqrt(vals[0])

    return np.pi * a * b


def envelope_overlap(gt, pred, covs):

    hits = []

    for p, g, P in zip(pred, gt, covs):

        err = g - p

        try:
            d2 = err.T @ np.linalg.inv(P[:2,:2]) @ err
        except np.linalg.LinAlgError:
            d2 = 999

        hits.append(d2 < 5.991)

    return np.mean(hits)


def run_safety_envelope_prediction(
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

        gt = fut[["x","y"]].values

        try:
            pred, covs = kalman_predict(
                hist,
                fut.t.values
            )
        except Exception:
            continue

        areas = [
            ellipse_area(P)
            for P in covs
        ]

        overlap = envelope_overlap(
            gt,
            pred,
            covs
        )

        rows.append({
            "scenario_id": sid,
            "agent_id": int(aid),
            "object_type": g.object_type.iloc[0],
            "mean_envelope_area": float(np.mean(areas)),
            "max_envelope_area": float(np.max(areas)),
            "final_envelope_area": float(areas[-1]),
            "coverage_rate": float(overlap),
        })

        if example is None:

            example = {
                "pred": pred,
                "gt": gt,
                "covs": covs
            }

    out = pd.DataFrame(rows)

    pred = example["pred"]
    gt = example["gt"]
    covs = example["covs"]

    plt.figure(figsize=(7,7))

    plt.plot(
        gt[:,0],
        gt[:,1],
        "o-",
        label="Ground truth"
    )

    plt.plot(
        pred[:,0],
        pred[:,1],
        "x--",
        label="Prediction"
    )

    for i in range(0, len(pred), 5):

        P = covs[i][:2,:2]

        vals, vecs = np.linalg.eigh(P)

        vals = np.maximum(vals, 1e-6)

        angle = np.degrees(
            np.arctan2(
                vecs[1,1],
                vecs[0,1]
            )
        )

        width = 4 * np.sqrt(vals[1])
        height = 4 * np.sqrt(vals[0])

        e = Ellipse(
            xy=pred[i],
            width=width,
            height=height,
            angle=angle,
            alpha=0.2
        )

        plt.gca().add_patch(e)

    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title(
        "Future safety envelope prediction"
    )

    plt.legend()

    fig1 = savefig(
        "part_b_35_safety_envelope_example.png"
    )

    plt.figure(figsize=(7,5))

    plt.hist(
        out.coverage_rate,
        bins=20
    )

    plt.xlabel("Envelope coverage rate")
    plt.ylabel("Count")

    plt.title(
        "Safety envelope coverage distribution"
    )

    fig2 = savefig(
        "part_b_36_safety_envelope_coverage.png"
    )

    metrics = {
        "num_agents": int(len(out)),
        "mean_coverage_rate":
            float(out.coverage_rate.mean()),
        "mean_final_envelope_area":
            float(out.final_envelope_area.mean()),
        "max_final_envelope_area":
            float(out.final_envelope_area.max()),
        "figures": {
            "envelope_example": str(fig1),
            "coverage_distribution": str(fig2),
        }
    }

    Path("outputs").mkdir(exist_ok=True)

    out.to_csv(
        "outputs/safety_envelope_prediction.csv",
        index=False
    )

    with open(
        "outputs/safety_envelope_prediction.json",
        "w"
    ) as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":

    metrics = run_safety_envelope_prediction()

    print(json.dumps(metrics, indent=2))
