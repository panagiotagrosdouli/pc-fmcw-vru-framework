import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from part_b.prediction import constant_velocity_predict, kalman_predict, ade_fde
from common.plotting import savefig


def normalize_track(g):
    g = g.sort_values("t").copy()
    x0, y0 = g[["x", "y"]].iloc[0].values
    g["x"] = g["x"] - x0
    g["y"] = g["y"] - y0
    g["x_meas"] = g["x_meas"] - x0
    g["y_meas"] = g["y_meas"] - y0
    return g


def split_history_future(agent_df, history_steps=10, future_steps=20):
    g = agent_df.sort_values("t").copy()

    if len(g) < history_steps + future_steps:
        return None, None

    # keep only first continuous segment
    dt = np.diff(g.t.values)
    if len(dt) and np.max(dt[:history_steps+future_steps-1]) > 0.25:
        return None, None

    g = normalize_track(g)
    hist = g.iloc[:history_steps].copy()
    fut = g.iloc[history_steps:history_steps+future_steps].copy()
    return hist, fut


def evaluate_womd(df):
    rows = []

    plt.figure(figsize=(8, 7))

    for (scenario_id, agent_id), g in df.groupby(["scenario_id", "agent_id"]):
        hist, fut = split_history_future(g)

        if hist is None:
            continue

        future_times = fut.t.values
        gt = fut[["x", "y"]].values

        cv_pred = constant_velocity_predict(hist, future_times)
        kf_pred, covs = kalman_predict(hist, future_times)

        cv_ade, cv_fde = ade_fde(cv_pred, gt)
        kf_ade, kf_fde = ade_fde(kf_pred, gt)

        # remove pathological cases / coordinate discontinuities
        if max(cv_ade, kf_ade, cv_fde, kf_fde) > 25:
            continue

        rows.append({
            "scenario_id": scenario_id,
            "agent_id": int(agent_id),
            "object_type": g.object_type.iloc[0],
            "cv_ade": float(cv_ade),
            "cv_fde": float(cv_fde),
            "kf_ade": float(kf_ade),
            "kf_fde": float(kf_fde),
        })

        plt.plot(gt[:,0], gt[:,1], "k-", alpha=0.35)
        plt.plot(cv_pred[:,0], cv_pred[:,1], "--", alpha=0.65)
        plt.plot(kf_pred[:,0], kf_pred[:,1], "-", alpha=0.8)

    plt.axis("equal")
    plt.xlabel("x relative (m)")
    plt.ylabel("y relative (m)")
    plt.title("WOMD trajectory prediction: CV vs Kalman")
    fig_path = savefig("part_b_04_womd_prediction.png")

    out = pd.DataFrame(rows)

    metrics = {
        "num_agents": int(len(out)),
        "mean_cv_ade": float(out.cv_ade.mean()),
        "mean_cv_fde": float(out.cv_fde.mean()),
        "mean_kf_ade": float(out.kf_ade.mean()),
        "mean_kf_fde": float(out.kf_fde.mean()),
        "figure": str(fig_path),
    }

    return out, metrics
