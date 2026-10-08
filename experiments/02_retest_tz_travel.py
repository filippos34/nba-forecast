"""Retest (allowed exception): time-zone travel. Elo per hour of time-zone change since
the team's previous game (same season), separately for eastward / westward travel.
Session 4b tried −20/−10 flat penalties with UTC-date rest counts (Δ −0.0004)."""
from dataclasses import asdict, replace

from common import br, ev, evaluate, fit

GRID = [float(v) for v in range(-20, 11, 2)]

base = br.BASELINE_V2
p1, _ = fit(base, "tz_east", GRID, ev.TRAIN["test"])
new, _ = fit(p1, "tz_west", GRID, ev.TRAIN["test"])
new, _ = fit(new, "tz_east", GRID, ev.TRAIN["test"])      # one coordinate-descent pass
f1, _ = fit(base, "tz_east", GRID, ev.TRAIN["fold2"])
new_f2, _ = fit(f1, "tz_west", GRID, ev.TRAIN["fold2"])
c = evaluate(base, new, new_params_fold2=new_f2)
print(ev.format_compare(c))
ev.log_experiment("02", "retest: time-zone travel", "retest", "baseline_v2",
                  f"tz_east={new.tz_east:+.0f}, tz_west={new.tz_west:+.0f} Elo/hour",
                  asdict(new), c, notes=f"fold2 refit east {new_f2.tz_east:+.0f} west {new_f2.tz_west:+.0f}")
