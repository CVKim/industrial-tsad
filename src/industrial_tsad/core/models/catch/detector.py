"""CATCH (Wu et al., ICLR 2025; https://github.com/decisionintelligence/CATCH) through its official wrapper class, imported
from a local clone at run time (the repository has no license file, so none of its code is copied here).
  fit   : CATCH.detect_fit = 80/20 split of the training part, StandardScaler on the 80 %, windows of seq_len at
          stride 1, reconstruction MSE + dc_lambda * channel-discovery loss + auxi_lambda * frequency loss, Adam +
          OneCycle (mask generator with its own optimiser), early stopping on validation MSE, best checkpoint
  score : the criterion of CATCH.detect_score per time step = channel-mean squared reconstruction error
          + score_lambda * channel-mean frequency criterion, on consecutive non-overlapping windows of seq_len; the
          tail the official loader drops (< seq_len points) is scored by one extra window ending at the last point
Hyper-parameters: the repository's per-dataset script (scripts/multivariate_detection/detect_score/<name>_script/
CATCH.sh) when the dataset exists there and dataset_presets is on, else the class defaults; then config.overrides.
Deviations (short training parts only): when the official 80/20 split leaves fewer than seq_len + 1 validation points,
the validation part is extended backwards into the training part so early stopping has windows to evaluate; when an
epoch would have fewer than 10 batches, the batch size is reduced so that it has 10 (the official loop otherwise
divides by zero: it steps the mask generator every len(loader) // 10 batches).
Runtime shims, nothing installed: einops.rearrange for the six patterns CATCH uses (checked equal to einops 0.8);
an empty parent package for ts_benchmark.baselines.time_series_library (its __init__ imports every TSLib model,
including the uninstalled reformer_pytorch)."""
import json
import re
import sys
import types
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ....utils.paths import get_paths
from ..base import BaseDetector

SCRIPT_NAME = {"MSL": "MSL", "SMAP": "SMAP", "SMD": "SMD", "PSM": "PSM", "SWaT": "SWAT", "GECCO": "GECCO",
               "Genesis": "Genesis", "CreditCard": "Creditcard"}


def rearrange(x, pattern, **s):
    """the six einops.rearrange patterns used by CATCH"""
    p = " ".join(pattern.split())
    if p == "b c d -> (b c d) 1":
        return x.reshape(-1, 1)
    if p == "(b c d) -> b c d":
        return x.reshape(s["b"], s["c"], s["d"])
    if p == "b n (h d) -> b h n d":
        b, n, hd = x.shape
        return x.reshape(b, n, s["h"], hd // s["h"]).permute(0, 2, 1, 3)
    if p == "b h n d -> b n (h d)":
        b, h, n, d = x.shape
        return x.permute(0, 2, 1, 3).reshape(b, n, h * d)
    if p == "b n c p -> (b n) p c":
        b, n, c, q = x.shape
        return x.permute(0, 1, 3, 2).reshape(b * n, q, c)
    if p == "(b n) p c -> b n p c":
        bn, q, c = x.shape
        return x.reshape(s["b"], bn // s["b"], q, c)
    raise NotImplementedError(pattern)


_CACHE = {}


def _catch_modules():
    if _CACHE:
        return _CACHE["module"], _CACHE["criterion"]
    root = get_paths().catch_code
    if not (root / "ts_benchmark").exists():
        raise FileNotFoundError(f"CATCH clone not found at {root} (set ITSAD_CATCH_CODE)")
    if "einops" not in sys.modules:
        m = types.ModuleType("einops")
        m.rearrange = rearrange
        sys.modules["einops"] = m
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import ts_benchmark.baselines  # noqa: F401  (light: an adapter dict)
    tsl = "ts_benchmark.baselines.time_series_library"
    if tsl not in sys.modules:
        pkg = types.ModuleType(tsl)
        pkg.__path__ = [str(root / "ts_benchmark" / "baselines" / "time_series_library")]
        sys.modules[tsl] = pkg
    import ts_benchmark.baselines.catch  # noqa: F401  (its __init__ binds the name CATCH to the class)
    from ts_benchmark.baselines.catch.utils.fre_rec_loss import frequency_criterion
    _CACHE["module"] = sys.modules["ts_benchmark.baselines.catch.CATCH"]
    _CACHE["criterion"] = frequency_criterion
    return _CACHE["module"], _CACHE["criterion"]


def dataset_hyper_params(dataset: Optional[str]) -> dict:
    """the repository's tuned hyper-parameters for this dataset, {} when there is no script for it"""
    name = SCRIPT_NAME.get(dataset or "")
    if not name:
        return {}
    f = get_paths().catch_code / "scripts" / "multivariate_detection" / "detect_score" / f"{name}_script" / "CATCH.sh"
    m = re.search(r"--model-hyper-params '([^']*)'", f.read_text(encoding="utf-8"))
    return json.loads(m.group(1)) if m else {}


class CATCHDetector(BaseDetector):
    def fit_score(self, x: np.ndarray, train_end: int, seed: int = 0, dataset: Optional[str] = None) -> np.ndarray:
        mod, frequency_criterion = _catch_modules()
        hp = dict(dataset_hyper_params(dataset) if self.config.dataset_presets else {})
        hp.update(self.config.overrides)
        probe = mod.CATCH(**hp).config  # resolved hyper-parameters (class defaults + script + overrides)
        n_win = int(len(x[:train_end]) * 0.8) - probe.seq_len + 1
        if n_win > 1 and n_win / probe.batch_size < 10:
            # the official loop steps the mask generator every len(loader) // 10 batches, which is 0 (ZeroDivisionError)
            # for short training parts: use a smaller batch so that one epoch has at least 10 batches
            hp["batch_size"] = max(2, n_win // 10)
        torch.manual_seed(seed)
        np.random.seed(seed)
        clf = mod.CATCH(**hp)
        L = clf.config.seq_len
        official = mod.train_val_split

        def split_with_min_val(train_data, ratio, _seq):
            tr, va = official(train_data, ratio, None)
            if va is None or len(va) < L + 1:
                va = train_data.iloc[max(0, len(train_data) - (L + 1)):]
            return tr, va

        mod.train_val_split = split_with_min_val
        try:
            clf.detect_fit(pd.DataFrame(np.asarray(x[:train_end], np.float64)), None)
        finally:
            mod.train_val_split = official
        return self._score(clf, x, frequency_criterion)

    @torch.no_grad()
    def _score(self, clf, full, frequency_criterion, batch=64):
        cfg = clf.config
        L = cfg.seq_len
        x = clf.scaler.transform(np.asarray(full, np.float64)).astype(np.float32)
        n = len(x)
        if n < L:
            x = np.concatenate([x, np.repeat(x[-1:], L - n, 0)], 0)
        N = len(x)
        starts = list(range(0, N - L + 1, L))
        tail = starts[-1] + L < N
        if tail:
            starts.append(N - L)
        clf.model.load_state_dict(clf.early_stopping.check_point)
        clf.model.to(clf.device).eval()
        mse = nn.MSELoss(reduction="none")
        fc = frequency_criterion(cfg)
        out = np.zeros(N, np.float64)
        for i in range(0, len(starts), batch):
            ss = starts[i:i + batch]
            xb = torch.from_numpy(np.stack([x[s:s + L] for s in ss])).to(clf.device)
            o, _, _ = clf.model(xb)
            sc = (torch.mean(mse(xb, o), dim=-1) + cfg.score_lambda * torch.mean(fc(xb, o), dim=-1)).cpu().numpy()
            for k, s in enumerate(ss):
                if tail and s == N - L:
                    covered = starts[-2] + L
                    out[covered:] = sc[k][L - (N - covered):]
                else:
                    out[s:s + L] = sc[k]
        return out[:n]
