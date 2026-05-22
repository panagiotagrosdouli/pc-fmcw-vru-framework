import numpy as np
import matplotlib.pyplot as plt

from part_b.prediction import kalman_predict
from common.plotting import savefig


def angle_of(points):
    return np.degrees(np.arctan2(points[:,1], points[:,0]))


def covariance_to_beamwidth(covariances,
                            base_width_deg=2.0,
                            gain=1.5):
    widths = []

    for P in covariances:
        pos_cov = P[:2,:2]
        sigma = np.sqrt(np.trace(pos_cov))
        widths.append(base_width_deg + gain * sigma)

    return np.array(widths)


def evaluate_uncertainty_beam(df):

    plt.figure(figsize=(9,8))

    all_widths = []
    all_errors = []

    count = 0

    for (scenario_id, agent_id), g in df.groupby(["scenario_id", "agent_id"]):

        g = g.sort_values("t")

        if len(g) < 30:
            continue

        hist = g.iloc[:10]
        fut = g.iloc[10:30]

        future_times = fut.t.values

        gt = fut[["x","y"]].values

        try:
            pred, covs = kalman_predict(hist, future_times)
        except Exception:
            continue

        beam_widths = covariance_to_beamwidth(covs)

        pred_angles = angle_of(pred)
        gt_angles = angle_of(gt)

        angular_error = np.abs(pred_angles - gt_angles)

        all_widths.extend(beam_widths.tolist())
        all_errors.extend(angular_error.tolist())

        plt.plot(gt[:,0], gt[:,1], "k-", alpha=0.25)

        for i in range(0, len(pred), 4):

            p = pred[i]
            bw = beam_widths[i]

            plt.scatter(p[0], p[1],
                        s=40 + bw*20,
                        alpha=0.25)

        count += 1

        if count > 25:
            break

    plt.axis("equal")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Uncertainty-aware predictive beam adaptation")

    fig_path = savefig(
        "part_b_07_uncertainty_beam_control.png"
    )

    metrics = {
        "num_agents_visualized": count,
        "mean_beam_width_deg":
            float(np.mean(all_widths)),
        "max_beam_width_deg":
            float(np.max(all_widths)),
        "mean_prediction_angle_error_deg":
            float(np.mean(all_errors)),
        "figure": str(fig_path),
    }

    return metrics


if __name__ == "__main__":

    from womd.womd_loader import load_womd_directory

    df = load_womd_directory(
        "data/womd/validation",
        max_files=1,
        max_scenarios=20
    )

    metrics = evaluate_uncertainty_beam(df)

    print(metrics)
