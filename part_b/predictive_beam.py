import numpy as np
import matplotlib.pyplot as plt
from part_b.dataset import generate_vru_dataset
from part_b.prediction import constant_velocity_predict, kalman_predict
from common.plotting import savefig

def bearing_deg(x, y):
    return np.degrees(np.arctan2(x, y))

def simulate_beam(df, method="reactive", current_t=4.0, history_s=1.0, horizon_s=0.8, delay_s=0.5):
    times = sorted(df.t.unique())
    exposures = []
    angular_errors = []
    for t in times:
        if t < current_t:
            continue
        glared_agents = 0
        total_agents = 0
        for aid, g in df.groupby("agent_id"):
            past = g[(g.t >= t-history_s) & (g.t <= t)]
            now = g[np.isclose(g.t, t)]
            if len(past) < 3 or len(now) == 0:
                continue
            total_agents += 1
            true_theta = bearing_deg(now.x.iloc[0], now.y.iloc[0])
            command_time = max(t - delay_s, 0)
            cmd_hist = g[(g.t >= command_time-history_s) & (g.t <= command_time)]
            if len(cmd_hist) < 3:
                cmd_hist = past
            if method == "reactive":
                row = cmd_hist.iloc[-1]
                cmd_theta = bearing_deg(row.x_meas, row.y_meas)
            else:
                future_times = np.array([cmd_hist.t.iloc[-1] + horizon_s])
                pred, _ = kalman_predict(cmd_hist, future_times)
                cmd_theta = bearing_deg(pred[0,0], pred[0,1])
            err = abs(true_theta - cmd_theta)
            angular_errors.append(err)
            if err > 2.2:
                glared_agents += 1
        if total_agents:
            exposures.append(glared_agents / total_agents)
    return np.array(exposures), np.array(angular_errors)

def demo_predictive_beam():
    df = generate_vru_dataset()
    reactive_exp, reactive_err = simulate_beam(df, "reactive")
    predictive_exp, predictive_err = simulate_beam(df, "predictive")
    improvement = 100 * (reactive_exp.mean() - predictive_exp.mean()) / max(reactive_exp.mean(), 1e-9)

    plt.figure(figsize=(8,4))
    plt.plot(reactive_exp*100, label="Reactive")
    plt.plot(predictive_exp*100, label="Predictive KF")
    plt.xlabel("Evaluation frame")
    plt.ylabel("Glare exposure (%)")
    plt.title(f"Predictive beam control reduces glare exposure by {improvement:.1f}%")
    plt.legend()
    p = savefig("part_b_03_predictive_beam.png")
    return {
        "reactive_glare_rate": float(reactive_exp.mean()),
        "predictive_glare_rate": float(predictive_exp.mean()),
        "relative_improvement_percent": float(improvement),
        "reactive_mean_angular_error_deg": float(reactive_err.mean()),
        "predictive_mean_angular_error_deg": float(predictive_err.mean()),
        "figure": str(p)
    }

if __name__ == "__main__":
    print(demo_predictive_beam())
