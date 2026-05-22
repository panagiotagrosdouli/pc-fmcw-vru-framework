import argparse, json
from pathlib import Path
from part_b.dataset import demo_dataset
from part_b.prediction import demo_prediction
from part_b.predictive_beam import demo_predictive_beam
from womd.womd_loader import load_womd_directory

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_source", choices=["synthetic", "womd"], default="synthetic")
    parser.add_argument("--womd_dir", default="data/womd")
    args = parser.parse_args()

    if args.data_source == "womd":
        df = load_womd_directory(args.womd_dir, max_files=1, max_scenarios=30)
        dataset_meta = {
            "num_agents": int(df.agent_id.nunique()),
            "num_scenarios": int(df.scenario_id.nunique()),
            "num_rows": int(len(df)),
            "source": "Waymo Open Motion Dataset validation subset"
        }
    else:
        df, dataset_meta = demo_dataset()
    pred_rows, pred_meta = demo_prediction()
    beam = demo_predictive_beam()

    results = {
        "data_source": args.data_source,
        "dataset": dataset_meta,
        "prediction_rows": pred_rows,
        "prediction": pred_meta,
        "predictive_beam": beam,
    }
    out = Path("outputs")
    out.mkdir(exist_ok=True)
    (out/"part_b_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
