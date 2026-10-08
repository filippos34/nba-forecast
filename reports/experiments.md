# Phase 2 experiments

Test = 2025-26 regular season (1,231 games); Apr-7 = the first 1,178 (without the last 53 tanking games); 2024-25 = second fold with parameters refit on 2021-22→2023-24. Δ = new − base Brier (negative = better), paired bootstrap 2,000 resamples. Rule 4: adopt only if the test CI excludes zero or Δ ≤ −0.0085.

| id | experiment | kind | test Brier (base → new) | Δ test [95% CI] | Δ Apr-7 [95% CI] | Δ 2024-25 [95% CI] | log loss (base → new) | acc (base → new) | auto verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| 00a1 | 2.0a1 rest days on arena-local dates (B2B −56 kept) | fix | 0.2057 → 0.2051 | -0.0006 [-0.0018, +0.0007] | -0.0006 [-0.0019, +0.0006] | +0.0010 [-0.0003, +0.0023] | 0.5981 → 0.5967 | 68.7% → 68.8% | FIX | Fixed |
| 00a2 | 2.0a2 refit B2B penalty on local labels → -60 | fix | 0.2051 → 0.2052 | +0.0001 [-0.0000, +0.0002] | +0.0001 [-0.0000, +0.0003] | +0.0000 [-0.0001, +0.0002] | 0.5967 → 0.5969 | 68.8% → 68.7% | FIX | Fixed |
| 00b | 2.0b no home court at neutral sites | fix | 0.2052 → 0.2053 | +0.0001 [-0.0001, +0.0003] | +0.0001 [-0.0001, +0.0003] | -0.0001 [-0.0004, +0.0001] | 0.5969 → 0.5970 | 68.7% → 68.5% | FIX | Fixed |
| 00v2 | 2.0 all fixes vs baseline_v1 → baseline_v2 | fix | 0.2057 → 0.2053 | -0.0004 [-0.0017, +0.0009] | -0.0004 [-0.0017, +0.0009] | +0.0009 [-0.0004, +0.0021] | 0.5981 → 0.5970 | 68.7% → 68.5% | FIX | Fixed |
| 01 | retest: 1-day rest penalty (local rest days) | retest | 0.2053 → 0.2049 | -0.0003 [-0.0009, +0.0002] | -0.0004 [-0.0010, +0.0002] | -0.0001 [-0.0004, +0.0003] | 0.5970 → 0.5964 | 68.5% → 69.0% | REJECT | Rejected |
| 02 | retest: time-zone travel | retest | 0.2053 → 0.2049 | -0.0003 [-0.0007, +0.0001] | -0.0004 [-0.0008, +0.0001] | +0.0004 [-0.0003, +0.0010] | 0.5970 → 0.5962 | 68.5% → 68.2% | REJECT | Rejected |
| 03a | 2.2(a) player availability — actual DNPs (ceiling) | ceiling | 0.2053 → 0.2014 | -0.0039 [-0.0085, +0.0007] | -0.0036 [-0.0083, +0.0011] | -0.0055 [-0.0103, -0.0009] | 0.5970 → 0.5857 | 68.5% → 68.1% | CEILING | Ceiling only |
| 04 | 2.1 margin model Φ(μ/σ) | structural | 0.2053 → 0.2057 | +0.0005 [-0.0010, +0.0020] | +0.0007 [-0.0008, +0.0023] | -0.0001 [-0.0021, +0.0018] | 0.5970 → 0.5991 | 68.5% → 68.5% | REJECT | Adopted for spreads |
| 05 | 2.3 playoffs in the walk (K_playoff_rs, grid incl. 0) | experiment | 0.2053 → 0.2061 | +0.0008 [-0.0002, +0.0019] | +0.0009 [-0.0003, +0.0020] | +0.0001 [-0.0012, +0.0015] | 0.5970 → 0.5989 | 68.5% → 68.3% | REJECT | Rejected |
| 06 | 2.3 preseason roster prior | experiment | 0.2053 → 0.2050 | -0.0003 [-0.0011, +0.0004] | -0.0003 [-0.0011, +0.0005] | -0.0001 [-0.0012, +0.0010] | 0.5970 → 0.5962 | 68.5% → 68.6% | REJECT | Rejected |
| 07 | 2.4 Kalman-style team rating variance | experiment | 0.2053 → 0.2083 | +0.0030 [-0.0008, +0.0068] | +0.0033 [-0.0005, +0.0071] | +0.0010 [-0.0034, +0.0050] | 0.5970 → 0.6039 | 68.5% → 67.2% | REJECT | Rejected |
| 03b | 2.2(b) player availability — pregame reports T−60 | experiment | 0.2053 → 0.2025 | -0.0027 [-0.0076, +0.0021] | -0.0024 [-0.0072, +0.0022] | -0.0050 [-0.0096, -0.0006] | 0.5970 → 0.5882 | 68.5% → 68.4% | REJECT | Rejected |
| 03c | 2.2(b) availability — STRICT: all tuning on 2021-22→2023-24 only | experiment | 0.2053 → 0.2019 | -0.0034 [-0.0083, +0.0013] | -0.0029 [-0.0074, +0.0019] | -0.0054 [-0.0099, -0.0008] | 0.5970 → 0.5872 | 68.5% → 69.0% | REJECT | Adopted |
| 03d | 2.2(b) availability — strict, roster-aware membership | experiment | 0.2053 → 0.2024 | -0.0029 [-0.0082, +0.0022] | -0.0024 [-0.0075, +0.0027] | -0.0036 [-0.0088, +0.0016] | 0.5970 → 0.5886 | 68.5% → 68.9% | REJECT | Rejected |
| 09a | 2.3 playoff K on the adopted model (grid incl. 0) | experiment | 0.2019 → 0.2026 | +0.0007 [-0.0003, +0.0017] | +0.0008 [-0.0003, +0.0018] | +0.0002 [-0.0007, +0.0010] | 0.5872 → 0.5887 | 69.0% → 69.1% | REJECT | Rejected |
| 09b | 2.3 preseason roster prior on the adopted model (grid incl. 0) | experiment | 0.2019 → 0.2018 | -0.0001 [-0.0004, +0.0002] | -0.0001 [-0.0004, +0.0002] | +0.0001 [-0.0003, +0.0005] | 0.5872 → 0.5869 | 69.0% → 68.8% | REJECT | Rejected |
| 03e | 2.2(b) availability — strict, roster-aware membership + 14-day recency | experiment | 0.2053 → 0.2027 | -0.0026 [-0.0081, +0.0026] | -0.0022 [-0.0073, +0.0032] | -0.0037 [-0.0089, +0.0015] | 0.5970 → 0.5890 | 68.5% → 68.9% | REJECT | Adopted (judgment call) |
