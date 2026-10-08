"""
Step 2.0 — bug fixes (not gated by rule 4; the change in Brier is reported).
  a. rest days on the arena-local date; back-to-back penalty refit on the corrected labels
  b. no home court at neutral sites
  c. decay anchored on each game's own season (no change on current data; guarded by tests)
Each step is compared with the previous one and with baseline_v1 (April params, tip order).
The end state is baseline_v2.
"""
from dataclasses import asdict, replace

import numpy as np

from common import br, ev, evaluate, fit

B2B_GRID = [float(v) for v in range(-120, 1, 4)]


def step(name, base, new, new_f2=None, base_f2=None, exp_id=None, notes=""):
    c = evaluate(base, new, new_params_fold2=new_f2, base_params_fold2=base_f2)
    print(f"\n== {name}\n{ev.format_compare(c)}")
    if exp_id:
        ev.log_experiment(exp_id, name, "fix", "baseline_v1" if base == br.LEGACY else "previous step",
                          name, asdict(new), c, decision="FIX", notes=notes)
    return c


def main():
    v1 = br.LEGACY
    print("baseline_v1 = April params, tip-order walk:", asdict(v1))

    a1 = replace(v1, rest_basis="local")
    step("2.0a1 rest days on arena-local dates (B2B −56 kept)", v1, a1, exp_id="00a1")

    a2, grid = fit(a1, "rest_b2b", B2B_GRID, ev.TRAIN["test"])
    a2_f2, _ = fit(a1, "rest_b2b", B2B_GRID, ev.TRAIN["fold2"])
    utc_fit, _ = fit(v1, "rest_b2b", B2B_GRID, ev.TRAIN["test"])
    print(f"  (for reference, refit on the old UTC labels: {utc_fit.rest_b2b})")
    step(f"2.0a2 refit B2B penalty on local labels → {a2.rest_b2b:.0f}", a1, a2, new_f2=a2_f2,
         base_f2=a1, exp_id="00a2",
         notes=f"fold2 refit {a2_f2.rest_b2b:.0f}; old-label refit {utc_fit.rest_b2b:.0f}")

    b = replace(a2, neutral_no_hc=True)
    b_f2 = replace(a2_f2, neutral_no_hc=True)
    step("2.0b no home court at neutral sites", a2, b, new_f2=b_f2, base_f2=a2_f2, exp_id="00b")

    c = replace(b, decay_anchor="2025-26")
    print("\n== 2.0c decay anchored on the game's own season: identical on 2021-22→2025-26 data "
          "(max |Δp| = %.1e); tests guard appending new seasons" %
          np.max(np.abs(br.run_walk(__import__("common").games(), b).probs
                        - br.run_walk(__import__("common").games(), c).probs)))

    total = step("2.0 all fixes vs baseline_v1 → baseline_v2", v1, c, new_f2=replace(b_f2), exp_id="00v2",
                 notes="baseline_v2")
    print("\nbaseline_v2 params:", asdict(c))
    return c, total


if __name__ == "__main__":
    main()
