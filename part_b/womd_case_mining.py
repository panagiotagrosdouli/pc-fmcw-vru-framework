import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from womd.womd_loader import load_womd_directory
from common.plotting import savefig


def normalize_agent_track(g):
    g = g.sort_values("t").copy()

    # Use first valid state as local origin.
    # This removes global Waymo map coordinates and gives relative motion.
    x0 = g.x.iloc[0]
    y0 = g.y.iloc[0]

    g["xr"] = g.x - x0
    g["yr"] = g.y - y0

    return g


def compute_case_features(g):
    g = normalize_agent_track(g)

    x = g.xr.values
    y = g.yr.values
    vx = g.velocity_x.values
    vy = g.velocity_y.values

    dist = np.hypot(x, y)
    speed = np.hypot(vx, vy)
    angle = np.degrees(np.arctan2(x, y + 1e-6))

    displacement = float(np.hypot(x[-1] - x[0], y[-1] - y[0]))
    min_dist = float(np.min(dist[1:])) if len(dist) > 1 else 0.0
    max_speed = float(np.max(speed))
    mean_speed = float(np.mean(speed))
    angle_span = float(np.max(angle) - np.min(angle))

    # crossing-like: lateral sign change in local coordinates
    crossing = bool(np.any(x < -0.5) and np.any(x > 0.5))

    # fast lateral motion
    lateral_motion = float(np.max(x) - np.min(x))

    fast_agent = bool(max_speed > 3.0)
    strong_turn = bool(angle_span > 25)
    moving = bool(displacement > 2.0)

    # collision-risk proxy in local frame:
    # high motion + high angular span + lateral crossing + speed
    risk = (
        0.25 * np.tanh(displacement / 8)
        + 0.25 * np.tanh(lateral_motion / 5)
        + 0.20 * np.tanh(max_speed / 5)
        + 0.20 * np.tanh(angle_span / 60)
        + (0.10 if crossing else 0.0)
    )

    return {
        "displacement_m": displacement,
        "lateral_motion_m": lateral_motion,
        "max_speed_mps": max_speed,
        "mean_speed_mps": mean_speed,
        "angle_span_deg": angle_span,
        "crossing": crossing,
        "fast_agent": fast_agent,
        "strong_turn": strong_turn,
        "moving": moving,
        "case_risk": float(risk),
    }


def mine_cases(df, top_k=15):
    rows = []

    for (scenario_id, agent_id), g in df.groupby(["scenario_id", "agent_id"]):
        if len(g) < 20:
            continue

        f = compute_case_features(g)

        if not (f["moving"] or f["crossing"] or f["fast_agent"] or f["strong_turn"]):
            continue

        rows.append({
            "scenario_id": scenario_id,
            "agent_id": int(agent_id),
            "object_type": g.object_type.iloc[0],
            **f,
        })

    cases = pd.DataFrame(rows)

    if len(cases) == 0:
        return cases

    return cases.sort_values("case_risk", ascending=False).head(top_k)


def plot_top_cases(df, cases):
    plt.figure(figsize=(9,8))

    for _, case in cases.iterrows():
        g = df[
            (df.scenario_id == case.scenario_id)
            & (df.agent_id == case.agent_id)
        ]

        g = normalize_agent_track(g)

        label = f"{case.object_type} {case.agent_id}"
        plt.plot(g.xr, g.yr, marker="o", ms=2, label=label)

        plt.scatter(g.xr.iloc[0], g.yr.iloc[0], marker="s", s=50)
        plt.scatter(g.xr.iloc[-1], g.yr.iloc[-1], marker="x", s=80)

    plt.scatter([0], [0], marker="^", s=150, label="initial VRU position")
    plt.axvline(0, ls="--", alpha=0.4)
    plt.axhline(0, ls="--", alpha=0.4)

    plt.xlabel("relative x (m)")
    plt.ylabel("relative y (m)")
    plt.axis("equal")
    plt.title("Mined special WOMD VRU cases: crossing / turning / fast motion")
    plt.legend(fontsize=7)

    return savefig("part_b_09_womd_high_risk_cases.png")


def run_case_mining(
    womd_dir="data/womd/validation",
    max_files=2,
    max_scenarios=150
):
    df = load_womd_directory(
        womd_dir,
        max_files=max_files,
        max_scenarios=max_scenarios
    )

    cases = mine_cases(df)

    Path("outputs").mkdir(exist_ok=True)

    cases.to_csv("outputs/womd_high_risk_cases.csv", index=False)

    fig = None
    if len(cases):
        fig = plot_top_cases(df, cases)

    result = {
        "num_cases": int(len(cases)),
        "top_cases": cases.to_dict(orient="records"),
        "figure": str(fig) if fig else None,
    }

    with open("outputs/womd_high_risk_cases.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


if __name__ == "__main__":
    result = run_case_mining()
    print(json.dumps(result, indent=2))
