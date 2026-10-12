# Detectors

All detectors implement `BaseDetector.fit_score(x, train_end, seed, dataset) -> (T,)` and are built from a config /
preset through `industrial_tsad.create_detector`. External code is imported from local clones at run time and never copied
into this repository (paths: `industrial_tsad/utils/paths.py`).

| type | preset(s) | origin | what is ours | verified |
|---|---|---|---|---|
| `paano` | `paano_official`, `paano_final` | PaAno, ICLR 2026 (MIT) | port of encoder / training / memory bank / kNN; channel-independent term, amplitude-preserving channel term, point-level next-step head | VUS-PR equal to the saved runs; scores within GPU nondeterminism |
| `mscad` | `mscad`, `mscad_modes64` | MSCAD, arXiv 2609.38004 (no code) | full re-implementation from the paper text; mode-token adapter | scores bit-identical to the saved runs |
| `axonad` | `axonad`, `axonad_recon_modes64`, `axonad_query_modes8_joint` | AxonAD, ECML-PKDD 2026 (Apache-2.0) | wrapper exposing the score parts (mse, jepa); mode-token adapters (recon / query, per-head / joint, context gate) | plain model bit-identical to the saved runs |
| `catch` | `catch` | CATCH, ICLR 2025 (no license file → not vendored) | wrapper around the official class; scoring of the series tail; `einops` shim for the six patterns used | shim equal to einops 0.8 |
| `kanad` | `kanad` | KAN-AD, ICML 2025 (MIT) | training / scoring without EasyTSAD; channel aggregation | — |
| `timesnet` | `timesnet` | TimesNet via TSB-AD (Apache-2.0) | thin wrapper of `run_TimesNet` | — |

## Settings

* **PaAno** (official TSB-AD-M run): patch 96, stride 1, 100 iterations, batch 512, lr 1e-4 (cosine to lr / 10), AdamW
  wd 1e-4, per-patch RevIN, positives 1..2 steps away, pretext weight 1 → 0 over the first 20 % of iterations,
  farthest-negative triplet rule of the released code, memory bank = 10 % of the training patches (≥ 500, k-means on
  L2-normalised embeddings), score = mean top-3 cosine distance. `paano_final` adds a channel-independent term
  (weight 1), an amplitude-preserving channel term on continuous channels (weight 1) and a point-level heteroscedastic
  next-step head (weight 0.5), combined as robust-z sums.
* **MSCAD**: window 128, patches 4 / 16 / 64, d 256, 2 encoder layers (4 heads), 2 bidirectional cross-scale blocks,
  Adam 1e-3, cosine annealing over 30 epochs, batch 128, early stopping (patience 5) on the last 10 % of the training
  part, ≤ 20,000 windows per epoch, channels independent (score = channel mean).
* **AxonAD**: window 100, d_model 128, 8 heads, lr 5e-4, kl_tail_k 10, forecast_steps 1, 50 epochs with early
  stopping; score = robust-z MSE + robust-z JEPA tail distance (training median / IQR), centre-padded.
* **CATCH**: the repository's per-dataset scripts for MSL, SMAP, SMD, PSM, SWaT, GECCO, Genesis, CreditCard, class
  defaults otherwise. Two fixes for short training parts (both documented in the module): validation extended
  backwards when the 80 / 20 split leaves fewer than `seq_len + 1` points, and a smaller batch when an epoch would
  have fewer than 10 batches (the official loop divides by zero there). The model is large (≈ 2.1e8 parameters for
  55 channels): one worker per 10 GB GPU.
* **Regime gate** (`industrial_tsad.core.fusion.RegimeGate`): add `w · rz(AxonAD JEPA part)` to the base score only where the
  median JEPA z over the test part exceeds τ (τ = 5, w = 1).
