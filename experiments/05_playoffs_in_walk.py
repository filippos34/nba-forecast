"""2.3 Playoffs (and play-in) in the rating walk: a separate K for updating the regular-
season ratings from play-in / playoff games (HC 50, no rest). The grid includes K=0
(= option (b), the April behaviour), so the fit chooses between (a) and (b)."""
from dataclasses import asdict, replace

from common import br, ev, evaluate, fit

GRID = [0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 30.0, 40.0, 60.0]
BASE = br.BASELINE_V2
new, grid = fit(BASE, "K_playoff_rs", GRID, ev.TRAIN["test"])
new_f2, _ = fit(BASE, "K_playoff_rs", GRID, ev.TRAIN["fold2"])
print(grid.to_string(index=False))
c = evaluate(BASE, new, new_params_fold2=new_f2)
print(ev.format_compare(c))
ev.log_experiment("05", "2.3 playoffs in the walk (K_playoff_rs, grid incl. 0)", "experiment", "baseline_v2",
                  f"K_playoff_rs={new.K_playoff_rs:g}", asdict(new), c,
                  decision=("KEEP K=0 (option b)" if new.K_playoff_rs == 0 else None),
                  notes=f"fold2 fit K={new_f2.K_playoff_rs:g}")
