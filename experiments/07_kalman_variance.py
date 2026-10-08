"""2.4 Kalman-style uncertainty-aware ratings: per-team rating variance (steady state 1)
rises by var_season at each season start and relaxes toward 1 at rate var_decay per game;
updates use K × variance (larger early in the season). var_season=0 is baseline_v2.
Both parameters fit on training seasons."""
from dataclasses import asdict, replace
import itertools

from common import br, ev, evaluate, rs_probs, train_brier

BASE = br.BASELINE_V2


def fit2(seasons):
    best = None
    for v, a in itertools.product([0.0, 0.5, 1.0, 2.0, 3.0, 5.0], [0.02, 0.05, 0.1, 0.2]):
        p = replace(BASE, var_season=v, var_decay=a)
        b = train_brier(rs_probs(p), seasons)
        if best is None or b < best[0]:
            best = (b, p)
    return best[1]


new, new_f2 = fit2(ev.TRAIN["test"]), fit2(ev.TRAIN["fold2"])
print(f"fit: var_season {new.var_season}, var_decay {new.var_decay}; fold2: {new_f2.var_season}, {new_f2.var_decay}")
c = evaluate(BASE, new, new_params_fold2=new_f2)
print(ev.format_compare(c))
ev.log_experiment("07", "2.4 Kalman-style team rating variance", "experiment", "baseline_v2",
                  f"var_season={new.var_season}, var_decay={new.var_decay}", asdict(new), c,
                  notes=f"fold2 fit {new_f2.var_season}/{new_f2.var_decay}")
