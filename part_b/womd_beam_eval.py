import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from part_b.prediction import (
    constant_velocity_predict,
    kalman_predict,
)

from common.plotting import savefig


def angle_of(points):
    return np.degrees(np.arctan2(points[:,1], points[:,0]))


def glare_metric(pred_angles, gt_angles, beam_width=6):
    err = np.abs(pred_angles - gt_angles)
    glare = err > beam_width
    return err.mean(), glare.mean()


def evaluate_predictive_beam(df):

    reactive_errs = []
    predictive_errs = []

    plt.figure(figsize=(8,8))

    for aid, g in df.groupby("agent_id"):

        g = g.sort_values("t")

        if len(g) < 30:
            continue

        hist = g.iloc[:10]
        fut = g.iloc[10:30]

        future_times = fut.t.values
        gt = fut[["x","y"]].values

        try:
            cv_pred = constant_velocity_predict(hist, future_times)
            kf_pred, _ = kalman_predict(hist, future_times)
        except Exception:
            continue

        reactive_angles = angle_of(
            np.repeat(
                hist[["x","y"]].values[-1][None,:],
                len(gt),
                axis=0
            )
        )

        gt_angles = angle_of(gt)
        predictive_angles = angle_of(kf_pred)

        r_err, r_glare = glare_metric(reactive_angles, gt_angles)
        p_err, p_glare = glare_metric(predictive_angles, gt_angles)

        reactive_errs.append((r_err, r_glare))
        predictive_errs.append((p_err, p_glare))

        plt.plot(gt[:,0], gt[:,1], "k-", alpha=0.35)
        plt.plot(kf_pred[:,0], kf_pred[:,1], "--", alpha=0.75)

    plt.axis("equal")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Predictive illumination on WOMD VRU trajectories")
    fig_path = savefig("part_b_05_womd_predictive_beam.png")

    reactive_errs = np.array(reactive_errs)
    predictive_errs = np.array(predictive_errs)

    metrics = {
        "num_agents": int(len(reactive_errs)),
        "reactive_mean_angle_error_deg":
            float(reactive_errs[:,0].mean()),
        "predictive_mean_angle_error_deg":
            float(predictive_errs[:,0].mean()),
        "reactive_glare_rate":
            float(reactive_errs[:,1].mean()),
        "predictive_glare_rate":
            float(predictive_errs[:,1].mean()),
        "relative_glare_reduction_percent":
            float(
                0.0 if reactive_errs[:,1].mean() == 0 else
                100 *
                (
                    reactive_errs[:,1].mean()
                    - predictive_errs[:,1].mean()
                )
                / reactive_errs[:,1].mean()
            ),
        "relative_angular_error_reduction_percent":
            float(
                100 *
                (
                    reactive_errs[:,0].mean()
                    - predictive_errs[:,0].mean()
                )
                / reactive_errs[:,0].mean()
            ),
        "figure": str(fig_path),
    }

    return metrics
