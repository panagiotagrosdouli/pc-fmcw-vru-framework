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

    x0, y0 = g[["x", "y"]].iloc[0].values

    g["x"] -= x0
    g["y"] -= y0

    return g


def simulate_occlusion(g,
                       start_idx=10,
                       duration=8):

    g = g.copy()

    end_idx = min(start_idx + duration, len(g))

    g.loc[g.index[start_idx:end_idx], "x_meas"] = np.nan
    g.loc[g.index[start_idx:end_idx], "y_meas"] = np.nan

    return g


def interpolate_measurements(g):

    g = g.copy()

    g["x_meas"] = g["x_meas"].interpolate()
    g["y_meas"] = g["y_meas"].interpolate()

    return g


def covariance_growth(covs):

    vals = []

    for P in covs:
        vals.append(np.sqrt(np.trace(P)))

    return np.array(vals)


def run_occlusion_prediction(
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

    plotted = 0

    for (sid, aid), g in df.groupby(["scenario_id","agent_id"]):

        if len(g) < 35:
            continue

        g = normalize_track(g)

        g["x_meas"] = g["x"]
        g["y_meas"] = g["y"]

        occ = simulate_occlusion(
            g,
            start_idx=10,
            duration=10
        )

        interp = interpolate_measurements(occ)

        hist = interp.iloc[:15]
        fut = interp.iloc[15:35]

        future_times = fut.t.values
        gt = fut[["x","y"]].values

        try:
            pred, covs = kalman_predict(
                hist,
                future_times
            )
        except Exception:
            continue

        uncertainty = covariance_growth(covs)

        drift = np.mean(
            np.linalg.norm(pred - gt, axis=1)
        )

        beam_width = 2 + 1.5 * uncertainty

        rows.append({
            "scenario_id": sid,
            "agent_id": int(aid),
            "object_type": g.object_type.iloc[0],
            "prediction_drift_m": float(drift),
            "max_uncertainty": float(np.max(uncertainty)),
            "mean_beam_width_deg": float(np.mean(beam_width)),
        })

        if plotted < 20:

            plt.plot(
                gt[:,0],
                gt[:,1],
                "k-",
                alpha=0.4
            )

            plt.plot(
                pred[:,0],
                pred[:,1],
                "--",
                alpha=0.8
            )

            occ_seg = occ.iloc[10:20]

            plt.scatter(
                occ_seg.x,
                occ_seg.y,
                marker="x",
                s=45,
                alpha=0.8
            )

            plotted += 1

    plt.axis("equal")
    plt.xlabel("relative x (m)")
    plt.ylabel("relative y (m)")
    plt.title(
        "Occlusion-aware WOMD trajectory prediction"
    )

    fig_path = savefig(
        "part_b_16_occlusion_aware_prediction.png"
    )

    out = pd.DataFrame(rows)

    metrics = {
        "num_agents": int(len(out)),
        "mean_prediction_drift_m":
            float(out.prediction_drift_m.mean()),
        "mean_max_uncertainty":
            float(out.max_uncertainty.mean()),
        "mean_beam_width_deg":
            float(out.mean_beam_width_deg.mean()),
        "figure":
            str(fig_path),
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

    metrics = run_occlusion_prediction()

    print(json.dumps(metrics, indent=2))
