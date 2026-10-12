"""MSCAD training and scoring (paper settings): Adam 1e-3, cosine annealing over 30 epochs, batch 128, gradient-norm clip
1.0, early stopping (patience 5) on a 10 % validation hold-out; z-score per channel; channels processed independently
(one shared model, each channel window is a sample); score = channel average; window score = mean of the per-scale
reconstruction errors; point score = average over the windows covering the point.
Our choices (fixed before any run): training windows at stride 1, at most 20,000 random windows per epoch; validation
= the last 10 % of the training part; inference stride 1, or the smallest stride keeping <= 3 M windows per series;
TF32 matmuls on Ampere GPUs."""
import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from ..base import BaseDetector
from .nets import MSCAD, window_errors


class MSCADDetector(BaseDetector):
    def fit_score(self, x: np.ndarray, train_end: int, seed: int = 0, dataset: Optional[str] = None) -> np.ndarray:
        x = np.asarray(x, np.float64)
        return self.fit_score_segments([x[:train_end]], [x], seed)[0]

    def fit_score_segments(self, train_segments: List[np.ndarray], test_series: List[np.ndarray], seed: int = 0,
                           circular: bool = False) -> List[np.ndarray]:
        """one model trained on several normal segments (len_i, C), scoring several series (T, C); circular = the
        series are closed loops (the last W - 1 steps are prepended). TF32 is enabled for this call only (the
        reported runs used it) and the previous global flags are restored afterwards."""
        flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        try:
            return self._fit_score_segments(train_segments, test_series, seed, circular)
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = flags

    def _fit_score_segments(self, train_segments, test_series, seed, circular):
        cfg, device = self.config, self.device
        W = cfg.window
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        rng = np.random.default_rng(seed)
        segs = [np.asarray(s, np.float64) for s in train_segments]
        cat = np.concatenate(segs, 0)
        mu, sd = cat.mean(0), cat.std(0)
        sd = np.where(sd > 1e-8, sd, 1.0)
        z = [((s - mu) / sd).astype(np.float32) for s in segs]
        tr_idx, va_idx = [], []
        for k, s in enumerate(z):  # windows (segment, channel, start); validation = the last 10 % of each segment
            n, C = s.shape
            if n < W:
                continue
            cut = int(n * 0.9)
            for c in range(C):
                starts = np.arange(0, n - W + 1)
                tr_idx.append(np.stack([np.full(len(starts[starts + W <= cut]), k), np.full(len(starts[starts + W <= cut]), c), starts[starts + W <= cut]], 1))
                va_idx.append(np.stack([np.full(len(starts[starts >= cut]), k), np.full(len(starts[starts >= cut]), c), starts[starts >= cut]], 1))
        tr_idx = np.concatenate([t for t in tr_idx if len(t)], 0) if any(len(t) for t in tr_idx) else np.zeros((0, 3), int)
        va_idx = np.concatenate([t for t in va_idx if len(t)], 0) if any(len(t) for t in va_idx) else np.zeros((0, 3), int)
        zt = [torch.as_tensor(s, device=device) for s in z]
        ar = torch.arange(W, device=device)

        def gather(idx):
            out = torch.empty(len(idx), W, device=device)
            for k in np.unique(idx[:, 0]):
                m = np.nonzero(idx[:, 0] == k)[0]
                t = torch.as_tensor(idx[m, 2], device=device)[:, None] + ar[None]
                c = torch.as_tensor(idx[m, 1], device=device)[:, None]
                out[torch.as_tensor(m, device=device)] = zt[k][t, c]
            return out

        model = MSCAD(W, cfg.patches, cfg.d_model, cfg.heads, cfg.layers, cfg.bridge_blocks, cfg.n_modes).to(device)
        if len(tr_idx):
            opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
            va = va_idx if len(va_idx) <= 4096 else va_idx[rng.choice(len(va_idx), 4096, replace=False)]
            best, bad, best_state = math.inf, 0, None
            vx = gather(va) if len(va) else None
            for _ in range(cfg.epochs):
                model.train()
                ep = tr_idx if len(tr_idx) <= cfg.max_windows_per_epoch else tr_idx[rng.choice(len(tr_idx), cfg.max_windows_per_epoch, replace=False)]
                ex = gather(ep[rng.permutation(len(ep))])
                for i in range(0, len(ex), cfg.batch):
                    loss = window_errors(model, ex[i:i + cfg.batch]).mean()
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                sched.step()
                if len(va):
                    model.eval()
                    with torch.no_grad():
                        v = float(torch.cat([window_errors(model, vx[i:i + 1024]) for i in range(0, len(vx), 1024)]).mean())
                    if v < best - 1e-6:
                        best, bad, best_state = v, 0, {k: t.detach().clone() for k, t in model.state_dict().items()}
                    else:
                        bad += 1
                        if bad >= cfg.patience:
                            break
            if best_state is not None:
                model.load_state_dict(best_state)
        model.eval()
        outs = []
        for x in test_series:
            xz = ((np.asarray(x, np.float64) - mu) / sd).astype(np.float32)
            T, C = xz.shape
            if circular:
                xz = np.concatenate([xz[-(W - 1):], xz], 0)
            n = len(xz)
            if n < W:
                outs.append(np.zeros(T, np.float32))
                continue
            nwin = n - W + 1
            stride = max(1, math.ceil(nwin * C / cfg.max_infer_windows))
            starts = np.arange(0, nwin, stride)
            if starts[-1] != nwin - 1:
                starts = np.append(starts, nwin - 1)
            xt = torch.as_tensor(xz, device=device)
            acc = np.zeros(n, np.float64)
            cnt = np.zeros(n, np.float64)
            st = torch.as_tensor(starts, device=device)
            with torch.no_grad():
                for c in range(C):
                    errs = []
                    for i in range(0, len(starts), 2048):  # 2048 windows per batch keeps one worker near 2 GB
                        idx = st[i:i + 2048, None] + ar[None]
                        errs.append(window_errors(model, xt[:, c][idx]).float().cpu().numpy())
                    e = np.concatenate(errs)
                    d = np.zeros(n + 1)  # spread each window's score over its points (difference array)
                    np.add.at(d, starts, e)
                    np.add.at(d, starts + W, -e)
                    acc += np.cumsum(d)[:n]
                    if c == 0:
                        dc = np.zeros(n + 1)
                        np.add.at(dc, starts, 1.0)
                        np.add.at(dc, starts + W, -1.0)
                        cnt = np.cumsum(dc)[:n]
            sc = acc / np.maximum(cnt, 1) / C
            outs.append((sc[W - 1:] if circular else sc).astype(np.float32))
        del model
        torch.cuda.empty_cache()
        return outs
