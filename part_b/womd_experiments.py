import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from womd.womd_loader import load_womd_directory
from part_b.womd_prediction import evaluate_womd
from part_b.womd_beam_eval import evaluate_predictive_beam
from common.plotting import savefig


def summarize_by_type(pred_df):
    rows = []
    for obj_type, g in pred_df.groupby("object_type"):
        rows.append({
            "object_type": obj_type,
            "num_agents": int(len(g)),
            "cv_ade": float(g.cv_ade.mean()),
            "cv_fde": float(g.cv_fde.mean()),
            "kf_ade": float(g.kf_ade.mean()),
            "kf_fde": float(g.kf_fde.mean()),
            "best_ade_method": "CV" if g.cv_ade.mean() < g.kf_ade.mean() else "KF",
        })
    return pd.DataFrame(rows)


def plot_prediction_comparison(summary):
    x = np.arange(len(summary))
    width = 0.35

    plt.figure(figsize=(8,4))
    plt.bar(x - width/2, summary.cv_ade, width, label="CV ADE")
    plt.bar(x + width/2, summary.kf_ade, width, label="KF ADE")
    plt.xticks(x, summary.object_type)
    plt.ylabel("ADE (m)")
    plt.title("WOMD VRU prediction error by object type")
    plt.legend()
    return savefig("part_b_06_womd_ade_by_type.png")


def run_womd_experiments(
    womd_dir="data/womd/validation",
    max_files=1,
    max_scenarios=50
):
    df = load_womd_directory(
        womd_dir,
        max_files=max_files,
        max_scenarios=max_scenarios
    )

    pred_df, pred_metrics = evaluate_womd(df)
    beam_metrics = evaluate_predictive_beam(df)
    summary = summarize_by_type(pred_df)

    fig_summary = plot_prediction_comparison(summary)

    results = {
        "dataset": {
            "source": "Waymo Open Motion Dataset validation subset",
            "max_files": max_files,
            "max_scenarios": max_scenarios,
            "num_scenarios": int(df.scenario_id.nunique()),
            "num_agents_raw": int(df.agent_id.nunique()),
            "num_rows": int(len(df)),
            "object_counts": {
                k: int(v) for k, v in df.object_type.value_counts().items()
            },
        },
        "prediction_overall": pred_metrics,
        "prediction_by_type": summary.to_dict(orient="records"),
        "beam_metrics": beam_metrics,
        "figures": {
            "prediction_by_type": str(fig_summary),
            "womd_prediction": pred_metrics.get("figure"),
            "predictive_beam": beam_metrics.get("figure"),
        }
    }

    Path("outputs").mkdir(exist_ok=True)
    pred_df.to_csv("outputs/womd_prediction_agent_metrics.csv", index=False)
    summary.to_csv("outputs/womd_prediction_by_type.csv", index=False)

    with open("outputs/womd_full_experiments.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    results = run_womd_experiments()
    print(json.dumps(results, indent=2))
