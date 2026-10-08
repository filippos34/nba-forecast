"""Retest (allowed exception): 1-day rest penalty, now on correct arena-local rest days.
Session 9 removed it (bootstrap CI [−21, 0] Elo) using UTC-date rest counts."""
from dataclasses import asdict, replace

from common import br, ev, evaluate, fit

GRID = [float(v) for v in range(-40, 21, 2)]

base = br.BASELINE_V2
new, grid = fit(base, "rest_1day", GRID, ev.TRAIN["test"])
new_f2, _ = fit(base, "rest_1day", GRID, ev.TRAIN["fold2"])
c = evaluate(base, new, new_params_fold2=new_f2)
print(ev.format_compare(c))
print(grid.sort_values("train_brier").head(5).to_string(index=False))
ev.log_experiment("01", "retest: 1-day rest penalty (local rest days)", "retest", "baseline_v2",
                  f"rest_1day={new.rest_1day:+.0f} Elo (fit on 2021-22→2024-25)", asdict(new), c,
                  notes=f"fold2 refit {new_f2.rest_1day:+.0f}")
