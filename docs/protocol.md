# Evaluation protocol

The numbers in this repository follow one protocol, so that a "gain" means the same thing everywhere.

## Benchmark and metric

* **TSB-AD** (Liu & Paparrizos, NeurIPS 2024 D&B): TSB-AD-M (multivariate, 180 eval + 20 tuning series) and
  TSB-AD-U (univariate, 350 eval + 48 tuning series), official file lists.
* Every series is semi-supervised: train on the prefix `[0, train_end)` (encoded in the file name), score every point.
* Ranking metric **VUS-PR** (TSB-AD `generate_curve`, version `opt`, 250 thresholds, sliding window = the dominant
  period of the first channel, `find_length_rank`). AUC-PR / AUC-ROC / VUS-ROC are reported alongside. The PA / event /
  range / affiliation F1 of `get_metrics` are not computed (≈ 70 % of the run time on long series, not used to rank).

## Selection on the tuning list only

1. A change is compared with its reference on the **tuning** list only.
2. **Dataset rule** (`industrial_tsad.evaluators.DatasetRule`): dataset means of the per-series metric; the change passes when
   more datasets go up than down at |d| > 0.01 **and** no dataset drops by more than 0.05.
3. Among passing variants the highest dataset mean is selected (`industrial_tsad.evaluators.select`). If none passes, the best
   variant may still be run on the eval list **for reporting only** — it cannot be claimed.

## One look at the eval list

4. The selected variant runs **once** on the eval list (seeds fixed by the preset's seed policy). It is never
   re-selected on eval.
5. **Paired comparison** with the reference on the same series (`industrial_tsad.evaluators.paired`): mean difference,
   series bootstrap and dataset-cluster bootstrap 95 % intervals (10,000 resamples), Wilcoxon p, datasets up / down.
6. **Claim rule**: an improvement is claimed only if the mean gain is ≥ +0.008 VUS-PR **and** the series-bootstrap
   interval excludes 0. When m different changes are evaluated on the same eval list, the interval is taken at the
   Bonferroni level 0.05 / m (`looks=m`).
7. A held-out confirmation set can settle a result that failed step 2: TSB-AD-U `Eva-Full` contains 472 series outside
   the 350 eval and 48 tuning series.

## Pre-registration

Before an eval run, write the question, the selection rule and the claim rule with the SHA-256 of the code
(`local/prereg.py`, `industrial_tsad.registry`). Later entries (results, deviations such as a bug fix) are appended, never
edited in place.

## Seeds and determinism

* Per-series seed = `seed.base + index in the file list` (PaAno, MSCAD, CATCH, KAN-AD, TimesNet: base 2027);
  AxonAD uses 2024 for every series (as the published run). Robustness to the seed is reported as mean ± sd.
* MSCAD and AxonAD reproduce saved scores bit for bit; PaAno's GPU kernels (convolutions, `index_add`) are not
  bit-deterministic, so its scores vary by ~0.1 % between runs (VUS-PR unchanged) — see
  `scripts/verify_reproduction.py`.

## Practical notes

* At most two heavy workers per 10 GB GPU (under Windows WDDM, memory over-subscription silently slows a worker by
  orders of magnitude instead of raising an error); watch progress per worker and GPU memory.
* Workers share a run through claim files; a killed worker leaves `claims/<file>.claim` behind — delete it before
  resuming.
