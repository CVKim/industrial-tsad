# Results on TSB-AD (public benchmark)

VUS-PR, official eval lists, protocol of [protocol.md](protocol.md). "published" = per-series results released with
TSB-AD or merged into its leaderboard (state of 2026-10-12); "paper" = self-reported in the paper only. Open pull
requests to the leaderboard report higher numbers, some through hosted APIs or compiled code only; they are listed
once merged. Produced by the research scripts this package was refactored from (2026-10); the package reproduces them
(`scripts/verify_reproduction.py`).

## TSB-AD-M (multivariate, 180 eval series)

| method | VUS-PR | note |
|---|---|---|
| **PaAno final + regime gate (AxonAD JEPA term)** | **0.589 ± 0.001** | 3 AxonAD seeds; vs AxonAD published +0.093..+0.095, series and dataset-cluster CIs > 0 for every seed; gain concentrated on GHL |
| **PaAno final** (`paano_final`) | **0.544 ± 0.001** | 3 seeds; vs AxonAD published +0.049 (series CI +0.0005..+0.097) |
| AxonAD (published) | 0.495 | our reruns 0.492 / 0.504 / 0.503 (seeds 2024-2026) |
| VACE (paper) | 0.48 | |
| MSCAD (paper) / our re-implementation | 0.47 / 0.465 | |
| StreamVAE (published) | 0.435 | |
| PaAno official (paper) / our port | 0.43 / 0.422 | |
| xLSTMAD / CHARM / TSPulse-FT (published) | 0.392 / 0.391 / 0.387 | |
| KAN-AD (our port of the official model) | 0.294 | paper table 0.40 |
| TimesNet (published) | 0.185 | |

### Mode-token adapters (winner-take-all modes) on TSB-AD-M

| change | result |
|---|---|
| AxonAD + reconstruction-space modes (K 64) | 0.4959 vs 0.4918 (+0.004, CI −0.011..+0.020) — no gain |
| AxonAD + JEPA-query-space modes (per head, K 4 / 8 / 64; context gate; joint selection K 8 / 64) | fail on tuning: the sustained GHL anomaly is explained away (GHL 0.207 → 0.02-0.10) |
| MSCAD + modes (K 16) | 0.455 vs 0.465 (−0.010, n.s.) |
| PaAno memory bank → mode head | −0.004 (CI −0.007..−0.001) |
| PaAno final + mode-head term | +0.000 |

Choosing the mode with the *observed future* (winner-take-all on the prediction target) explains anomalies away in
predictive models, whatever the selection granularity; in reconstruction space the modes average out.

## TSB-AD-U (univariate, 350 eval series)

| method | VUS-PR | note |
|---|---|---|
| Time-RCD+MAFT (FT) (published) | 0.586 | best entry of the official leaderboard; per-series CSV, no M result |
| MSCAD (our re-implementation) | 0.572 | paper 0.57 — reproduced; vs Time-RCD+MAFT −0.013 (series CI −0.053..+0.027, n.s.) |
| PaAno final (multivariate config as is) | 0.552 | MSCAD − PaAno +0.020, series CI −0.015..+0.057 (n.s.); PaAno ahead on 15 datasets, MSCAD on 6 |
| TSPulse-FT (published) | 0.552 | |
| Time-RCD / CHARM (published) | 0.522 / 0.501 | |
| StreamVAE / AxonAD (published) | 0.451 / 0.434 | |

* **Held-out 472 series** (Eva-Full minus the 350 eval and 48 tuning series; YAHOO, UCR, WSD, Stock):
  MSCAD 0.320, PaAno 0.407 — MSCAD's univariate lead depends on the composition of the official eval list.
  Over all 822 Eva-Full series: MSCAD 0.427, PaAno 0.468.
* **Fusion** rz(PaAno) + 0.5 · rz(MSCAD): 0.620 on the 350 eval series, but it failed the tuning rule (SMAP −0.13) and
  the pre-registered confirmation on the 472 held-out series (+0.008 vs PaAno, CI −0.005..+0.022) → not claimed.
* MSCAD + modes (K 64) on TSB-AD-U: fails the tuning rule (WSD −0.050). Run on the 350 eval series for reporting
  only: 0.576 vs 0.572 (+0.004, series CI −0.006..+0.013, dataset-cluster CI −0.005..+0.010; 7 datasets up, 6 down)
  — no gain.
