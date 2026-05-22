import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from womd.womd_loader import load_womd_directory
from common.plotting import savefig


def normalize_track(g):
    g = g.sort_values("t").copy()

    x0, y0 = g[["x", "y"]].iloc[0].values

    g["x"] -= x0
    g["y"] -= y0

    return g


def compute_motion_features(g):

    x = g.x.values
    y = g.y.values

    dx = np.diff(x)
    dy = np.diff(y)

    displacement = np.hypot(
        x[-1] - x[0],
        y[-1] - y[0]
    )

    lateral_motion = np.max(x) - np.min(x)

    heading = np.unwrap(np.arctan2(dy + 1e-6, dx + 1e-6))

    heading_change = np.degrees(
        np.max(heading) - np.min(heading)
    )

    curvature = np.mean(np.abs(np.diff(heading))) \
        if len(heading) > 2 else 0.0

    speed = np.mean(np.hypot(dx, dy)) * 10

    crossing = bool(
        np.any(x < -0.5)
        and np.any(x > 0.5)
    )

    return {
        "displacement": displacement,
        "lateral_motion": lateral_motion,
        "heading_change": heading_change,
        "curvature": curvature,
        "speed": speed,
        "crossing": crossing,
    }


def classify_intention(f):

    if f["displacement"] < 1.0:
        return "stationary"

    if f["crossing"]:
        return "crossing"

    if f["heading_change"] > 35:
        return "turning"

    if f["curvature"] > 0.15:
        return "erratic"

    return "straight"


def run_intention_prediction(
    womd_dir="data/womd/validation",
    max_files=2,
    max_scenarios=120
):

    df = load_womd_directory(
        womd_dir,
        max_files=max_files,
        max_scenarios=max_scenarios
    )

    rows = []

    plt.figure(figsize=(10,8))

    colors = {
        "crossing": "red",
        "turning": "orange",
        "straight": "blue",
        "stationary": "gray",
        "erratic": "green",
    }

    plotted = 0

    for (sid, aid), g in df.groupby(["scenario_id", "agent_id"]):

        if len(g) < 20:
            continue

        g = normalize_track(g)

        f = compute_motion_features(g)

        intention = classify_intention(f)

        rows.append({
            "scenario_id": sid,
            "agent_id": int(aid),
            "object_type": g.object_type.iloc[0],
            "intention": intention,
            **f
        })

        if plotted < 40:

            plt.plot(
                g.x,
                g.y,
                color=colors[intention],
                alpha=0.7
            )

            plotted += 1

    plt.axis("equal")
    plt.xlabel("relative x (m)")
    plt.ylabel("relative y (m)")
    plt.title("WOMD VRU intention prediction")

    handles = []

    for k,v in colors.items():
        handles.append(
            plt.Line2D([0],[0], color=v, lw=3, label=k)
        )

    plt.legend(handles=handles)

    fig_path = savefig(
        "part_b_15_intention_prediction.png"
    )

    out = pd.DataFrame(rows)

    summary = (
        out.groupby(["object_type","intention"])
        .size()
        .reset_index(name="count")
    )

    metrics = {
        "num_agents": int(len(out)),
        "intentions":
            summary.to_dict(orient="records"),
        "figure":
            str(fig_path),
    }

    Path("outputs").mkdir(exist_ok=True)

    out.to_csv(
        "outputs/intention_prediction_agents.csv",
        index=False
    )

    summary.to_csv(
        "outputs/intention_prediction_summary.csv",
        index=False
    )

    with open(
        "outputs/intention_prediction.json",
        "w"
    ) as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":

    metrics = run_intention_prediction()

    print(json.dumps(metrics, indent=2))
