"""2.1 revised (Gate 2): spread from the win probability, μ = σ·Φ⁻¹(p), σ fit per test
season on the seasons before it (ML on margins). Compared with the standalone margin model
(slope on the Elo difference incl. availability, fit on the same seasons). Both use the
production model (baseline_v2 walk + availability layer). If the spread calibration is
within noise, the derived version is used so moneyline / spread / alt lines agree."""
import numpy as np
import pandas as pd

from common import ROOT, br, ev, games, rs_mask, rs_frame
import availability_model as am
import margin_model as mm

g = games()
adj = am.historical_adjustments(g)
walk = br.run_walk(g, br.PRODUCTION, adj)
m_rs = rs_mask()
rs = rs_frame()
p, diff = walk.probs[m_rs], walk.diffs[m_rs]
margin = (rs["home_pts"] - rs["away_pts"]).to_numpy(float)
rows = []
for test, train in (("2024-25", ["2021-22", "2022-23", "2023-24"]),
                    ("2025-26", ["2021-22", "2022-23", "2023-24", "2024-25"])):
    tr = rs["season"].isin(train).to_numpy(); te = (rs["season"] == test).to_numpy()
    d = mm.fit_sigma(p[tr], margin[tr]); s = mm.fit(diff[tr], margin[tr])
    mu_d, mu_s = d.mu(p[te]), s.mu(diff[te])
    crps_d = mm.crps_normal(margin[te], mu_d, d.sigma); crps_s = mm.crps_normal(margin[te], mu_s, s.sigma)
    rng = np.random.default_rng(0); dd = crps_d - crps_s
    boot = dd[rng.integers(0, len(dd), (2000, len(dd)))].mean(1)
    ll = lambda mu, sg: np.mean(-0.5 * ((margin[te] - mu) / sg) ** 2 - np.log(sg * np.sqrt(2 * np.pi)))
    pit_d = mm.norm.cdf((margin[te] - mu_d) / d.sigma); pit_s = mm.norm.cdf((margin[te] - mu_s) / s.sigma)
    dec = lambda pit: np.histogram(pit, bins=10, range=(0, 1))[0] / len(pit)
    rows.append({"test": test, "sigma_derived": d.sigma, "slope_standalone": s.slope, "sigma_standalone": s.sigma,
                 "crps_derived": crps_d.mean(), "crps_standalone": crps_s.mean(), "crps_delta": dd.mean(),
                 "crps_ci_lo": np.percentile(boot, 2.5), "crps_ci_hi": np.percentile(boot, 97.5),
                 "ll_derived": ll(mu_d, d.sigma), "ll_standalone": ll(mu_s, s.sigma),
                 "cover1sd_derived": np.mean(np.abs(pit_d - 0.5) < 0.3413),
                 "cover1sd_standalone": np.mean(np.abs(pit_s - 0.5) < 0.3413),
                 "pit_max_dev_derived": np.abs(dec(pit_d) - 0.1).max(),
                 "pit_max_dev_standalone": np.abs(dec(pit_s) - 0.1).max(),
                 "mean_abs_mu_gap": np.mean(np.abs(mu_d - mu_s))})
t = pd.DataFrame(rows)
print(t.round(4).T.to_string())
t.to_csv(ROOT / "reports" / "spreads_derived_vs_standalone.csv", index=False)
within = all(r["crps_ci_lo"] <= 0 <= r["crps_ci_hi"] or r["crps_delta"] < 0 for r in rows)
print("\nderived within noise of (or better than) standalone on both seasons:", within)
