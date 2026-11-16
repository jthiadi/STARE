"""Paper Table 3: action recognition on MECCANO with the final-label method (2,823 annotated test segments, 61 actions,
top-1), recomputed on a CPU from the saved model outputs in system/assistant/output/meccano_action_end/.

The one choice (which of the two MECCANO VideoMAE runs) is read from the saved val result (action_end_val.json); every
other setting is the IndustReal one. Needs the MECCANO annotations (STARE/data/MECCANO, or $MECCANO_ROOT).

    cd system/assistant && python ../../metrics/ar/score_meccano_final_labels.py   # -> results/meccano_final_labels.json
"""
import json
import sys
from pathlib import Path

A = Path(__file__).resolve().parents[2] / "system" / "assistant"
sys.path.insert(0, str(A))
RESULT = Path(__file__).resolve().parent / "results" / "meccano_final_labels.json"


def chosen_vmae(val_result: dict) -> str:
    return val_result["videomae_run"]


def main() -> None:
    from meccano_action_end import ablation, data
    from meccano_action_end.action_end import VAL_RESULT
    vmae = chosen_vmae(json.loads(VAL_RESULT.read_text(encoding="utf-8")))
    val, test = ablation.run("val", vmae), ablation.run("test", vmae)
    report = {"protocol": "MECCANO action recognition: annotated segments, 61 actions, top-1; final-label method, past only",
              "videomae_run": vmae, "val": val["full_recipe"], "test": test["full_recipe"],
              "test_segments": len(data.clips(data.split("test")))}
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
