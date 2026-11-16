"""Oracle AR ablation (PREREGISTRATION_ablations.md): the annotated actions, revealed causally as OracleARProvider,
reach v14 D frames late, D = ceil(10 x v14's p95 compute) (no AR model, so no chunk wait and no AR compute).
No imports with side effects: used by retune.py and test_once.py."""
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


def d_oracle() -> int:
    return math.ceil(round(10.0 * json.loads((HERE / "results/latency.json").read_text())["v14_p95_ms"] / 1000, 6))


def shifted_oracle(provider_cls, split: str, video: str, n: int, D: int) -> list:
    """Names of the annotated actions active at frame t - D (OracleARProvider's rule), nothing before D."""
    ar = provider_cls(); ar.reset(split, video, n)
    return [[o.name for o in ar.observe(t - D)] if t >= D else [] for t in range(n)]


def patch(provider_cls, D: int) -> None:
    """Make OracleARProvider.observe(t) return what it returned for t - D (for eval_v14.py --ar gt)."""
    orig = provider_cls.observe
    provider_cls.observe = lambda self, t: orig(self, t - D) if t >= D else []
