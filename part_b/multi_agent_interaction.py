import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from womd.womd_loader import load_womd_directory
from common.plotting import savefig


def normalize_track(g):

    g = g.sort_values("t").copy()

    x0, y0 = g[["x","y"]].iloc[0].values

    g["x"] -= x0
    g["y"] -= y0

    return g


def interaction_score(a, b):

    ta = a.t.values
    tb = b.t.values

    t_common = np.intersect1d(ta, tb)

    if len(t_common) < 5:
        return None

    pa = (
        a.set_index("t")
        .loc[t_common][["x","y"]]
        .values
    )

    pb = (
        b.set_index("t")
        .loc[t_common][["x","y"]]
        .values
    )

    d = np.linalg.norm(
        pa - pb,
        axis=1
    )

    min_dist = np.min(d)

    va = np.gradient(pa, axis=0)
    vb = np.gradient(pb, axis=0)

    dir_similarity = np.mean(
        np.sum(va*vb, axis=1)
        / (
            np.linalg.norm(va, axis=1)
            * np.linalg.norm(vb, axis=1)
            + 1e-9
        )
    )

    score = (
        np.exp(-min_dist/3)
        * (0.5 + 0.5*dir_similarity)
    )

    return {
        "min_distance": float(min_dist),
        "direction_similarity": float(dir_similarity),
        "interaction_score": float(score),
    }


def run_multi_agent_interaction(
    womd_dir="data/womd/validation",
    max_files=2,
    max_scenarios=80
):

    df = load_womd_directory(
        womd_dir,
        max_files=max_files,
        max_scenarios=max_scenarios
    )

    rows = []

    interaction_matrix = []

    for sid, sdf in df.groupby("scenario_id"):

        agents = []

        for aid, g in sdf.groupby("agent_id"):

            g = normalize_track(g)

            if len(g) < 20:
                continue

            agents.append((aid, g))

        for i in range(len(agents)):

            for j in range(i+1, len(agents)):

                aid1, g1 = agents[i]
                aid2, g2 = agents[j]

                res = interaction_score(
                    g1,
                    g2
                )

                if res is None:
                    continue

                rows.append({
                    "scenario_id": sid,
                    "agent_a": int(aid1),
                    "agent_b": int(aid2),
                    **res
                })

                interaction_matrix.append(
                    res["interaction_score"]
                )

    out = pd.DataFrame(rows)

    top = out.sort_values(
        "interaction_score",
        ascending=False
    )

    plt.figure(figsize=(7,5))

    plt.hist(
        out.interaction_score,
        bins=30
    )

    plt.xlabel("Interaction score")
    plt.ylabel("Count")

    plt.title(
        "VRU interaction score distribution"
    )

    fig1 = savefig(
        "part_b_44_interaction_distribution.png"
    )

    plt.figure(figsize=(7,5))

    plt.scatter(
        out.min_distance,
        out.interaction_score,
        alpha=0.7
    )

    plt.xlabel("Minimum distance (m)")
    plt.ylabel("Interaction score")

    plt.title(
        "Proximity vs interaction strength"
    )

    fig2 = savefig(
        "part_b_45_interaction_proximity.png"
    )

    metrics = {
        "num_interactions":
            int(len(out)),
        "mean_interaction_score":
            float(out.interaction_score.mean()),
        "max_interaction_score":
            float(out.interaction_score.max()),
        "high_interaction_pairs":
            top.head(10).to_dict(orient="records"),
        "figures": {
            "distribution": str(fig1),
            "proximity_relation": str(fig2),
        }
    }

    Path("outputs").mkdir(exist_ok=True)

    out.to_csv(
        "outputs/multi_agent_interactions.csv",
        index=False
    )

    with open(
        "outputs/multi_agent_interaction.json",
        "w"
    ) as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":

    metrics = run_multi_agent_interaction()

    print(json.dumps(metrics, indent=2))
