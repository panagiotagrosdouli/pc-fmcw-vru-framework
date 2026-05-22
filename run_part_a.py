import json
from pathlib import Path
from part_a.waveform import demo_waveform
from part_a.communication import demo_communication
from part_a.sensing import demo_sensing
from part_a.tracking import demo_tracking
from part_a.illumination import demo_illumination
from part_a.performance import demo_performance

def main():
    results = {
        "waveform": demo_waveform(),
        "communication": demo_communication(),
        "sensing": demo_sensing(),
        "tracking": demo_tracking(),
        "illumination": demo_illumination(),
        "performance": demo_performance(),
    }
    out = Path("outputs")
    out.mkdir(exist_ok=True)
    (out/"part_a_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
