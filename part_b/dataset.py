import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from common.plotting import savefig

def generate_vru_dataset(seed=11, dt=0.1, T=9.0):
    rng = np.random.default_rng(seed)
    times = np.arange(0, T + 1e-9, dt)
    rows = []
    specs = [
        ("pedestrian", 0, lambda t: (-8 + 1.2*t, 28 + 0.2*t)),
        ("pedestrian", 1, lambda t: (10 - 1.0*t, 24 + 0.5*t)),
        ("cyclist", 2, lambda t: (-18 + 4.0*t, 38 - 0.7*t)),
        ("cyclist", 3, lambda t: (-12 + 3.0*t, 22 + 6*np.sin(t/3.2))),
    ]
    for typ, aid, f in specs:
        for t in times:
            x, y = f(t)
            xm = x + rng.normal(0, 0.25)
            ym = y + rng.normal(0, 0.25)
            rows.append([aid, typ, t, x, y, xm, ym])
    return pd.DataFrame(rows, columns=["agent_id","object_type","t","x","y","x_meas","y_meas"])

def split_history_future(df, history_s=1.0, future_s=3.0, current_t=4.0):
    hist = df[(df.t >= current_t-history_s) & (df.t <= current_t)]
    fut = df[(df.t > current_t) & (df.t <= current_t+future_s)]
    return hist, fut

def demo_dataset():
    df = generate_vru_dataset()
    plt.figure(figsize=(7,6))
    for aid, g in df.groupby("agent_id"):
        plt.plot(g.x, g.y, label=f"{g.object_type.iloc[0]} {aid}")
        plt.scatter(g.x_meas[::5], g.y_meas[::5], s=10, alpha=0.45)
    plt.scatter([0], [0], marker="^", s=120, label="ego/headlamp")
    plt.axis("equal")
    plt.xlabel("x lateral (m)")
    plt.ylabel("y forward (m)")
    plt.title("Synthetic VRU trajectories")
    plt.legend()
    p = savefig("part_b_01_vru_dataset.png")
    return df, {"num_agents": df.agent_id.nunique(), "figure": str(p)}

if __name__ == "__main__":
    df, meta = demo_dataset()
    print(meta)
