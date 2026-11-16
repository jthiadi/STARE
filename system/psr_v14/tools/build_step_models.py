"""Rebuild the two step models of v9 from the train + val videos (no test data):
  model/tables.json : count tables of the naive-Bayes step model (which step a fastening action completes)
  model/lr.pkl      : logistic-regression step model (C = 3.0)
Both use ground-truth AR and PSR labels of train + val only (this is training, not inference).

    python tools/build_step_models.py <out_dir>      # e.g. tools_out/ ; compare with model/ via SHA256SUMS
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_tables as bt                       # noqa: E402
from psr.core import list_videos                # noqa: E402
from psr.lr_model import LRGroupModel           # noqa: E402

ITEMS = [("train", v) for v in list_videos("train")] + [("val", v) for v in list_videos("val")]


def samples(items):
    out = []
    for sp, v in items:
        out += [json.loads(s) for s in bt.video_samples(sp, v)[0]]
    return out


if __name__ == "__main__":
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    (out / "tables.json").write_text(json.dumps(bt.build(ITEMS), indent=1))
    LRGroupModel(samples(ITEMS), C=3.0).save(out / "lr.pkl")
    print("written", out / "tables.json", out / "lr.pkl")
