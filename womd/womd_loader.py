from pathlib import Path
import struct
import pandas as pd

from waymo_open_dataset.protos import scenario_pb2


def read_tfrecord_records(path, max_records=None):
    records = []
    with open(path, "rb") as f:
        while True:
            header = f.read(8)
            if not header:
                break
            length = struct.unpack("<Q", header)[0]
            f.read(4)
            payload = f.read(length)
            f.read(4)
            records.append(payload)
            if max_records and len(records) >= max_records:
                break
    return records


def object_type_name(obj_type):
    mapping = {
        1: "vehicle",
        2: "pedestrian",
        3: "cyclist",
        4: "other",
    }
    return mapping.get(obj_type, f"unknown_{obj_type}")


def scenario_to_dataframe(scenario, max_agents=None):
    rows = []
    times = list(scenario.timestamps_seconds)

    for track in scenario.tracks:
        obj_type = object_type_name(track.object_type)

        if obj_type not in {"pedestrian", "cyclist"}:
            continue

        states = track.states
        valid_count = sum(1 for s in states if s.valid)

        if valid_count < 15:
            continue

        for i, state in enumerate(states):
            if not state.valid:
                continue

            t = times[i] if i < len(times) else i * 0.1

            rows.append({
                "scenario_id": scenario.scenario_id,
                "agent_id": track.id,
                "object_type": obj_type,
                "t": float(t),
                "x": float(state.center_x),
                "y": float(state.center_y),
                "x_meas": float(state.center_x),
                "y_meas": float(state.center_y),
                "velocity_x": float(state.velocity_x),
                "velocity_y": float(state.velocity_y),
                "heading": float(state.heading),
                "current_time_index": int(scenario.current_time_index),
            })

        if max_agents and len(set(r["agent_id"] for r in rows)) >= max_agents:
            break

    return pd.DataFrame(rows)


def load_womd_directory(womd_dir, max_files=1, max_scenarios=20):
    womd_dir = Path(womd_dir)
    files = sorted(womd_dir.rglob("*.tfrecord*"))

    if not files:
        raise FileNotFoundError(f"No .tfrecord files found in {womd_dir}")

    dfs = []
    scenarios_read = 0

    for file in files[:max_files]:
        for payload in read_tfrecord_records(file):
            scenario = scenario_pb2.Scenario()
            scenario.ParseFromString(payload)

            df = scenario_to_dataframe(scenario)
            if len(df) > 0:
                dfs.append(df)

            scenarios_read += 1
            if scenarios_read >= max_scenarios:
                break

        if scenarios_read >= max_scenarios:
            break

    if not dfs:
        raise RuntimeError("No pedestrian/cyclist tracks found in selected WOMD scenarios.")

    out = pd.concat(dfs, ignore_index=True)
    return out
