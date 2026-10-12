"""KAN-AD (Zhou et al., ICML 2025; https://github.com/issaccv/KAN-AD, MIT) without its EasyTSAD framework: the model
class KANADModel is imported unchanged from a local clone (kanad/kanad.py); training and scoring re-implement
KANAD.train_valid_phase / test_phase and the EasyTSAD data handling with the repository's settings:
  data     : training part split 80 / 20, z-score fitted on the 80 % (StandardScaler), first-order differencing
  samples  : window 96 -> next value;  model KANADModel(window 96, order 2)
  training : Adam lr 0.01, StepLR(5, 0.75), MSE, batch 1024 shuffled, <= 100 epochs, early stopping on the validation
             loss (patience 3, delta 1e-4; the last model is kept)
  score    : |prediction - actual| of the differenced series; the first 96 points take the first score
Univariate by design: a multivariate series is scored channel by channel (seed + 1000 c) and aggregated (mean / max)."""
import importlib.util
import sys
import types
from typing import Optional

import numpy as np
import torch
from torch import nn, optim

from ....utils.paths import get_paths
from ..base import BaseDetector

_MODEL = {}


def _model_class():
    if "cls" in _MODEL:
        return _MODEL["cls"]
    for name in ["EasyTSAD", "EasyTSAD.DataFactory", "EasyTSAD.Exptools", "EasyTSAD.Methods", "EasyTSAD.DataFactory.TorchDataSet",
                 "EasyTSAD.DataFactory.TorchDataSet.PredictWindow", "torchinfo"]:
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["EasyTSAD.DataFactory"].TSData = object
    sys.modules["EasyTSAD.Exptools"].EarlyStoppingTorch = object
    sys.modules["EasyTSAD.Methods"].BaseMethod = object
    sys.modules["EasyTSAD.DataFactory.TorchDataSet"].PredictWindow = sys.modules["EasyTSAD.DataFactory.TorchDataSet.PredictWindow"]
    sys.modules["EasyTSAD.DataFactory.TorchDataSet.PredictWindow"].UTSOneByOneDataset = object
    src = get_paths().kanad_code / "kanad" / "kanad.py"
    spec = importlib.util.spec_from_file_location("kanad_src", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _MODEL["cls"] = mod.KANADModel
    return _MODEL["cls"]


def _diff1(x):
    return (np.pad(x, (0, 1), "edge") - np.pad(x, (1, 0), "edge"))[:-1]


def _windows(d, w):
    n = len(d) - w
    if n <= 0:
        return torch.zeros(0, w), torch.zeros(0, 1)
    idx = np.arange(w)[None] + np.arange(n)[:, None]
    return torch.as_tensor(d[idx], dtype=torch.float32), torch.as_tensor(d[w:w + n, None], dtype=torch.float32)


class _EarlyStopping:
    def __init__(self, patience=3, delta=1e-4):
        self.patience, self.delta, self.counter, self.best, self.stop = patience, delta, 0, None, False

    def __call__(self, val_loss):
        score = -val_loss
        if self.best is None:
            self.best = score
        elif score < self.best + self.delta:
            self.counter += 1
            self.stop = self.counter >= self.patience
        else:
            self.best, self.counter = score, 0


class KANADDetector(BaseDetector):
    def fit_score(self, x: np.ndarray, train_end: int, seed: int = 0, dataset: Optional[str] = None) -> np.ndarray:
        x = np.asarray(x, np.float64)
        S = np.stack([self.fit_score_1d(x[:train_end, c], x[:, c], seed + 1000 * c) for c in range(x.shape[1])], 1)
        return S.max(1) if self.config.channel_agg == "max" else S.mean(1)

    def fit_score_1d(self, train_part, full, seed=0):
        cfg, device = self.config, self.device
        torch.manual_seed(seed)
        np.random.seed(seed)
        cut = len(train_part) - int(len(train_part) * 0.2)
        tr80, va20 = train_part[:cut], train_part[cut:]
        mu, sd = float(tr80.mean()), float(tr80.std())
        sd = sd if sd > 0 else 1.0

        def prep(v):
            return _diff1((np.asarray(v, np.float64) - mu) / sd)

        Xtr, Ytr = _windows(prep(tr80), cfg.window)
        Xva, Yva = _windows(prep(va20), cfg.window)
        model = _model_class()(window=cfg.window, order=cfg.order).to(device)
        if len(Xtr) > 1:
            opt = optim.Adam(model.parameters(), lr=cfg.lr)
            sched = optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.75)
            lossf = nn.MSELoss()
            es = _EarlyStopping()
            Xtr, Ytr, Xva, Yva = Xtr.to(device), Ytr.to(device), Xva.to(device), Yva.to(device)
            g = torch.Generator(device="cpu").manual_seed(seed)
            for _ in range(cfg.epochs):
                model.train()
                perm = torch.randperm(len(Xtr), generator=g).to(device)
                for i in range(0, len(Xtr), cfg.batch):
                    b = perm[i:i + cfg.batch]
                    if len(b) < 2:  # BatchNorm needs more than one value per channel
                        continue
                    opt.zero_grad()
                    loss = lossf(model(Xtr[b]), Ytr[b])
                    loss.backward()
                    opt.step()
                model.eval()
                with torch.no_grad():
                    vl = [lossf(model(Xva[i:i + cfg.batch]), Yva[i:i + cfg.batch]).item() for i in range(0, len(Xva), cfg.batch)]
                sched.step()
                es(float(np.mean(vl)) if vl else 0.0)
                if es.stop:
                    break
        model.eval()
        X, Y = _windows(prep(full), cfg.window)
        s = np.zeros(len(full), np.float32)
        if len(X):
            with torch.no_grad():
                e = torch.cat([(model(X[i:i + 8192].to(device)) - Y[i:i + 8192].to(device)).abs().cpu()
                               for i in range(0, len(X), 8192)])[:, -1].numpy()
            e[np.isnan(e)] = 1000.0
            s[cfg.window:] = e
            s[:cfg.window] = e[0]
        return s
