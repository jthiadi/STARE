"""Check that the frozen v14 package is byte-identical to what was frozen before the single test run.

model/SHA256SUMS lists the model files, the psr/ package, eval_v14.py, tools/run_asd.py and the ASD weights
(weights/asd_best_IndustRealandSynthetic.pt, not in git: checked only if present).

    python tools/verify_frozen.py
"""
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
bad = 0
for line in (HERE / "model" / "SHA256SUMS").read_text().splitlines():
    digest, name = line.split(" *", 1)
    f = HERE / name
    if not f.exists():
        print(f"absent   {name}" + (" (download separately, see README)" if name.startswith("weights/") else ""))
        bad += not name.startswith("weights/")
        continue
    ok = hashlib.sha256(f.read_bytes()).hexdigest() == digest
    bad += not ok
    print(f"{'OK  ' if ok else 'DIFF'}     {name}")
print("all frozen files match" if not bad else f"{bad} file(s) differ or are missing")
sys.exit(1 if bad else 0)
