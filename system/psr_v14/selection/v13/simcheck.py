import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from v12.common import D0Tracker, H
from analyze_errvids import models, score
from psr.filter_tracker import FilterConfig
import v13.simar as SA
M = models(); CFG = FilterConfig(**dict(H.V9, tau=0.05, theta=0.7)); D0Tracker.d0 = 8.0
t0 = time.time()
r = [(sp, v, *(lambda t: (t.y_hat, t.y_hat_err))(H.run(sp, v, CFG, M[v[:2]], "live", tracker_cls=D0Tracker))) for sp, v in H.items("live")]
print("real live val", H.fmt(score(r)), f"({time.time()-t0:.0f}s)", flush=True)
for seed in range(3):
    r = [(sp, v, *(lambda t: (t.y_hat, t.y_hat_err))(H.run(sp, v, CFG, M[v[:2]], "live", tracker_cls=D0Tracker, obs_fn=lambda v_, sp=sp: SA.simulate(sp, v_, seed)))) for sp, v in H.items("live")]
    print(f"sim seed {seed} val ", H.fmt(score(r)), flush=True)
r = [(sp, v, *(lambda t: (t.y_hat, t.y_hat_err))(H.run(sp, v, CFG, M[v[:2]], "live", tracker_cls=D0Tracker, obs_fn=lambda v_, sp=sp: SA.simulate(sp, v_, 0)))) for sp, v in H.E.ITEMS]
print("sim seed 0 ALL 52", H.fmt(score(r)), f"({time.time()-t0:.0f}s)")
