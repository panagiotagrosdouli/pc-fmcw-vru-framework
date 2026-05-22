import numpy as np
import matplotlib.pyplot as plt
from part_b.dataset import generate_vru_dataset, split_history_future
from common.plotting import savefig

def constant_velocity_predict(hist, future_times):
    h = hist.sort_values("t")
    p1 = h[["x_meas","y_meas"]].iloc[-1].values.astype(float)
    p0 = h[["x_meas","y_meas"]].iloc[0].values.astype(float)
    dt = h.t.iloc[-1] - h.t.iloc[0]
    v = (p1 - p0) / max(dt, 1e-6)
    return np.array([p1 + v*(tt-h.t.iloc[-1]) for tt in future_times])

def kalman_predict(hist, future_times):
    # Constant-velocity Kalman filter implemented directly.
    h = hist.sort_values("t")
    dt = np.median(np.diff(h.t.values))
    x = np.array([h.x_meas.iloc[0], h.y_meas.iloc[0], 0.0, 0.0])
    P = np.eye(4) * 5
    F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], float)
    H = np.array([[1,0,0,0],[0,1,0,0]], float)
    Q = np.eye(4) * 0.04
    R = np.eye(2) * 0.25
    I = np.eye(4)
    for _, row in h.iterrows():
        x = F @ x
        P = F @ P @ F.T + Q
        z = np.array([row.x_meas, row.y_meas])
        y = z - H @ x
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ y
        P = (I - K @ H) @ P

    preds = []
    covs = []
    last_t = h.t.iloc[-1]
    step = dt
    Fstep = F.copy()
    while len(preds) < len(future_times):
        x = Fstep @ x
        P = Fstep @ P @ Fstep.T + Q
        preds.append(x[:2].copy())
        covs.append(P[:2,:2].copy())
    return np.array(preds), covs

def ade_fde(pred, gt):
    dist = np.linalg.norm(pred - gt, axis=1)
    return float(dist.mean()), float(dist[-1])

def demo_prediction():
    df = generate_vru_dataset()
    hist, fut = split_history_future(df)
    rows = []
    plt.figure(figsize=(8,6))
    for aid, h in hist.groupby("agent_id"):
        f = fut[fut.agent_id == aid].sort_values("t")
        future_times = f.t.values
        gt = f[["x","y"]].values
        cv = constant_velocity_predict(h, future_times)
        kf, covs = kalman_predict(h, future_times)
        cv_ade, cv_fde = ade_fde(cv, gt)
        kf_ade, kf_fde = ade_fde(kf, gt)
        rows.append([aid, h.object_type.iloc[0], cv_ade, cv_fde, kf_ade, kf_fde])
        plt.plot(gt[:,0], gt[:,1], "k-", alpha=0.4)
        plt.plot(cv[:,0], cv[:,1], "--", alpha=0.75)
        plt.plot(kf[:,0], kf[:,1], "-", alpha=0.85)
        plt.scatter(h.x_meas, h.y_meas, s=10)
    plt.axis("equal")
    plt.xlabel("x lateral (m)")
    plt.ylabel("y forward (m)")
    plt.title("Constant-velocity and Kalman trajectory prediction")
    p = savefig("part_b_02_prediction.png")
    return rows, {"figure": str(p)}

if __name__ == "__main__":
    rows, meta = demo_prediction()
    print(rows)
