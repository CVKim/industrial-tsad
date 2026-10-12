"""AxonAD (ECML-PKDD 2026; https://github.com/iis-esslingen/AxonAD, Apache-2.0) through the TSB_AD implementation in
its clone (not vendored), with the TSB-AD settings (window 100, d_model 128, 8 heads, lr 5e-4, kl_tail_k 10,
forecast_steps 1, 50 epochs with early stopping).
Score = robust-z reconstruction MSE + robust-z JEPA tail query distance, each z-scored by its training median / IQR
(the series is normalised with its own statistics, as TSB-AD's ReconstructDataset does), centre-padded like
TCNTransformer.decision_function. The two parts are kept in `self.parts` ('mse', 'jepa') for fusion / the regime gate.
With an adapter (mode_adapters.py) the model starts from the SAME initial backbone weights as the plain model and the
training draws the same random stream (the adapter's parameters are drawn on the side), so plain vs adapter differ only
by the adapter."""
import math
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from ....utils.external import ensure_tsbad
from ..base import BaseDetector
from .mode_adapters import build_mode_model


def _tsbad():
    ensure_tsbad()
    from TSB_AD.models.AxonAD import TCNTransformer
    from TSB_AD.utils.dataset import ReconstructDataset
    return TCNTransformer, ReconstructDataset


def _pad(z, win):
    pad_l, pad_r = math.ceil((win - 1) / 2), (win - 1) // 2
    return np.r_[[z[0]] * pad_l, z, [z[-1]] * pad_r]


class AxonADDetector(BaseDetector):
    def _hp(self):
        c = self.config
        return {"win_size": c.win_size, "d_model": c.d_model, "num_heads": c.num_heads, "lr": c.lr,
                "kl_tail_k": c.kl_tail_k, "forecast_steps": c.forecast_steps, "epochs": c.epochs}

    def fit_score(self, x: np.ndarray, train_end: int, seed: int = 0, dataset: Optional[str] = None) -> np.ndarray:
        TCNTransformer, _ = _tsbad()
        X = np.asarray(x, np.float64)
        torch.manual_seed(seed)
        np.random.seed(seed)
        hp = self._hp()
        clf = TCNTransformer(feats=X.shape[1], **hp)
        ad = self.config.adapter
        if ad is not None:
            cpu_state = torch.get_rng_state()
            cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            m = build_mode_model(X.shape[1], hp["win_size"], hp["d_model"], hp["num_heads"], hp["forecast_steps"], ad,
                                 hp["kl_tail_k"]).to(clf.device)
            missing, unexpected = m.load_state_dict(clf.model.state_dict(), strict=False)
            assert not unexpected and all(q.startswith(("mode_", "gate_")) for q in missing), (missing, unexpected)
            torch.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
            clf.model = m
            clf.opt = torch.optim.AdamW(clf.model.parameters(), lr=hp["lr"], weight_decay=1e-5)
        if ad is not None and ad.prior == "gate":
            _fit_with_gate(clf, X[:train_end])
            mse, jepa = _gate_parts(clf, X)
        else:
            clf.fit(X[:train_end])
            mse, jepa = self.parts_of(clf, X)
        self.parts = {"mse": mse.astype(np.float32), "jepa": jepa.astype(np.float32)}
        del clf
        torch.cuda.empty_cache()
        return mse + jepa

    @staticmethod
    def parts_of(clf, data, normalize=True):
        """the two z-scored parts of TCNTransformer.decision_function, centre-padded like it"""
        _, ReconstructDataset = _tsbad()
        ds = ReconstructDataset(data, window_size=clf.win_size, normalize=normalize)
        loader = DataLoader(ds, batch_size=clf.batch_size, shuffle=False)
        clf.model.eval()
        zm, zj = [], []
        with torch.inference_mode():
            for d, _ in loader:
                d = d.to(clf.device)
                rec, q_real, q_pred, k, q_tgt, _ = clf.model(d)
                mse = (rec - d).pow(2).sum(dim=-1).mean(dim=1)
                jepa = clf._jepa_tail_dist(q_tgt, q_pred)
                zm.append(((mse - clf.mse_med) / (clf.mse_iqr + 1e-9)).cpu())
                zj.append(((jepa - clf.jepa_med) / (clf.jepa_iqr + 1e-9)).cpu())
        return tuple(_pad(torch.cat(z).numpy(), clf.win_size) for z in (zm, zj))


# ---------------------------------------------------------------------------------------- context-gate (prior) variant
def _fit_with_gate(clf, data):
    """TCNTransformer.fit (same split, loaders, loss, EMA, early stopping on validation reconstruction) + the gate's
    cross-entropy outside the loss balance; gradients clipped separately for the gate"""
    _, ReconstructDataset = _tsbad()
    split = int((1 - clf.validation_size) * len(data))
    tsTrain, tsValid = (data, data[:0]) if split <= clf.win_size else (data[:split], data[split:])
    train_loader = DataLoader(ReconstructDataset(tsTrain, window_size=clf.win_size), batch_size=clf.batch_size, shuffle=True)
    valid_loader = DataLoader(ReconstructDataset(tsValid, window_size=clf.win_size), batch_size=clf.batch_size, shuffle=False)
    gate = [p for n, p in clf.model.named_parameters() if n.startswith("gate_")]
    rest = [p for n, p in clf.model.named_parameters() if not n.startswith("gate_")]
    for _ in range(1, clf.epochs + 1):
        clf.model.train()
        for d, _y in train_loader:
            d = d.to(clf.device)
            rec, q_real, q_pred, k, q_tgt, tmask = clf.model(d)
            s_rec, s_kl = clf.model.s_rec, clf.model.s_kl
            loss = (torch.exp(-s_rec) * clf.mse(rec, d) + s_rec + torch.exp(-s_kl) * clf._jepa_masked_loss(q_tgt, q_pred, tmask) + s_kl)
            clf.opt.zero_grad(set_to_none=True)
            (loss + clf.model.aux["gate_ce"]).backward()
            torch.nn.utils.clip_grad_norm_(rest, 1.0)
            torch.nn.utils.clip_grad_norm_(gate, 1.0)
            clf.opt.step()
            clf.model.ema_update_target()
        if len(valid_loader) > 0:
            clf.model.eval()
            v, n = 0.0, 0
            with torch.no_grad():
                for d, _y in valid_loader:
                    d = d.to(clf.device)
                    v += float(clf.mse(clf.model(d)[0], d).item())
                    n += 1
            clf.early_stopping(v / n, clf.model)
            if clf.early_stopping.early_stop:
                break
    _calibrate_gate(clf, tsTrain)


def _gate_windows(clf, data):
    _, ReconstructDataset = _tsbad()
    loader = DataLoader(ReconstructDataset(data, window_size=clf.win_size), batch_size=clf.batch_size, shuffle=False)
    m, tail = clf.model, clf.kl_tail_k
    m.eval()
    mse, mix = [], []
    with torch.inference_mode():
        for d, _ in loader:
            d = d.to(clf.device)
            rec = m(d)[0]
            cos, logp = m.aux["cos"][:, :, -tail:], m.aux["logp"][:, :, -tail:]
            mse.append((rec - d).pow(2).sum(dim=-1).mean(dim=1).cpu())
            mix.append((-torch.logsumexp(logp + m.kappa * (cos - 1), dim=-1)).mean(dim=(1, 2)).cpu())
    return (torch.cat(mse).numpy().astype(np.float64), torch.cat(mix).numpy().astype(np.float64))


def _calibrate_gate(clf, tsTrain):
    """kappa = vMF ML approximation R (D - R^2) / (1 - R^2) from the training windows' best-mode cosine, then robust
    median / IQR of both parts"""
    _, ReconstructDataset = _tsbad()
    loader = DataLoader(ReconstructDataset(tsTrain, window_size=clf.win_size), batch_size=clf.batch_size, shuffle=False)
    clf.model.eval()
    r, n = 0.0, 0
    with torch.inference_mode():
        for d, _y in loader:
            clf.model(d.to(clf.device))
            c = clf.model.aux["cos"][:, :, -clf.kl_tail_k:].max(-1).values
            r += float(c.double().sum())
            n += c.numel()
    R = min(max(r / max(n, 1), 1e-3), 0.999)
    D = clf.model.d_model // clf.model.attn.num_heads
    clf.model.kappa = R * (D - R * R) / (1 - R * R)
    mse, mix = _gate_windows(clf, tsTrain)
    clf.gate_stats = [clf._robust_fit(v) if len(v) else (0.0, 1.0) for v in (mse, mix)]


def _gate_parts(clf, data):
    out = []
    for v, (med, iqr) in zip(_gate_windows(clf, data), clf.gate_stats):
        out.append(_pad((v - med) / (iqr + 1e-9), clf.win_size))
    return tuple(out)
