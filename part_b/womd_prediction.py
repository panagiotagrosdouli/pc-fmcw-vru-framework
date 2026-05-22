import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from part_b.prediction import (
    constant_velocity_predict,
    kalman_predict,
    ade_fde,
)

from common.plotting import savefig


def split_history_future(agent_df, history_steps=10, future_steps=20):
    g = agent_df.sort_values("t")

    if len(g) < history_steps + future_steps:
        return None, None

    hist = g.iloc[:history_steps].copy()
    fut = g.iloc[history_steps:history_steps+future_steps].copy()

    return hist, fut


def evaluate_womd(df):
    rows = []

    plt.figure(figsize=(8, 7))

    for agent_id, g in df.groupby("agent_id"):

        split = split_history_future(g)

        if split[0] is None:
            continue

        hist, fut = split

        future_times = fut.t.values
        gt = fut[["x", "y"]].values

        try:
            cv_pred = constant_velocity_predict(hist, future_times)
            kf_pred, covs = kalman_predict(hist, future_times)
        except Exception:
            continue

        cv_ade, cv_fde = ade_fde(cv_pred, gt)
        kf_ade, kf_fde = ade_fde(kf_pred, gt)

        rows.append({
            "agent_id": int(agent_id),
            "object_type": g.object_type.iloc[0],
            "cv_ade": float(cv_ade),
            "cv_fde": float(cv_fde),
            "kf_ade": float(kf_ade),
            "kf_fde": float(kf_fde),
        })

        plt.plot(gt[:,0], gt[:,1], "k-", alpha=0.4)
        plt.plot(cv_pred[:,0], cv_pred[:,1], "--", alpha=0.7)
        plt.plot(kf_pred[:,0], kf_pred[:,1], "-", alpha=0.9)

        # uncertainty ellipses
        for p, C in zip(kf_pred[::4], covs[::4]):
            eigvals = np.linalg.eigvals(C)
            r = np.sqrt(np.max(eigvals))
            circle = plt.Circle(
                (p[0], p[1]),
                r,
                fill=False,
                alpha=0.25
            )
            plt.gca().add_patch(circle)

    plt.axis("equal")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
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
