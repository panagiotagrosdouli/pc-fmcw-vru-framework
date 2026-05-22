import numpy as np
import matplotlib.pyplot as plt

from part_b.prediction import kalman_predict
from common.plotting import savefig


def compute_risk(position, velocity, uncertainty,
                 w_dist=0.45, w_angle=0.25, w_speed=0.15, w_unc=0.15):

    x, y = position
    distance = np.hypot(x, y)
    angle = abs(np.degrees(np.arctan2(x, y)))
    speed = np.linalg.norm(velocity)

    dist_score = 1 / (distance + 1e-3)
    angle_score = np.exp(-angle / 20)
    speed_score = np.tanh(speed / 5)
    unc_score = np.tanh(uncertainty / 5)

    risk = (
        w_dist * dist_score
        + w_angle * angle_score
        + w_speed * speed_score
        + w_unc * unc_score
    )

    return risk


def evaluate_risk_aware_beam(df):

    risks = []

    plt.figure(figsize=(8,7))

    for (scenario_id, agent_id), g in df.groupby(["scenario_id", "agent_id"]):

        g = g.sort_values("t")

        if len(g) < 30:
            continue

        hist = g.iloc[:10]
        fut = g.iloc[10:30]

        future_times = fut.t.values

        try:
            pred, covs = kalman_predict(hist, future_times)
        except Exception:
            continue

        velocity = pred[-1] - pred[0]
        uncertainty = np.sqrt(np.trace(covs[-1]))

        risk = compute_risk(
            position=pred[-1],
            velocity=velocity,
            uncertainty=uncertainty
        )

        risks.append({
            "scenario_id": scenario_id,
            "agent_id": int(agent_id),
            "object_type": g.object_type.iloc[0],
            "risk": float(risk),
            "x": float(pred[-1,0]),
            "y": float(pred[-1,1]),
            "uncertainty": float(uncertainty)
        })

    risks = sorted(risks, key=lambda r: r["risk"], reverse=True)

    top = risks[:30]

    xs = [r["x"] for r in top]
    ys = [r["y"] for r in top]
    cs = [r["risk"] for r in top]
    ss = [80 + 500*r["risk"] for r in top]

    sc = plt.scatter(xs, ys, c=cs, s=ss, alpha=0.75)
    plt.colorbar(sc, label="Risk score")

    for r in top[:8]:
        plt.text(r["x"], r["y"], r["object_type"], fontsize=8)

    plt.scatter([0], [0], marker="^", s=120, label="ego/headlamp")
    plt.xlabel("Predicted x (m)")
    plt.ylabel("Predicted y (m)")
    plt.axis("equal")
    plt.title("Risk-aware VRU prioritization for predictive beam control")
    plt.legend()

    fig_path = savefig("part_b_08_risk_aware_beam.png")

    metrics = {
        "num_agents_scored": len(risks),
        "top_risk": risks[0] if risks else None,
        "mean_risk": float(np.mean([r["risk"] for r in risks])),
        "figure": str(fig_path)
    }

    return metrics


if __name__ == "__main__":
    from womd.womd_loader import load_womd_directory

    df = load_womd_directory(
        "data/womd/validation",
        max_files=1,
        max_scenarios=30
    )

    metrics = evaluate_risk_aware_beam(df)
    print(metrics)
