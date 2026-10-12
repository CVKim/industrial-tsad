"""PaAno detector on one series: official multivariate term (mv), optional channel-independent term (ci), amplitude-
preserving channel term on continuous channels (ci_amp) and the point-level next-step head (point); terms are combined
as weighted robust-z sums. The order of model construction and seeding follows the reported runs exactly (seeding is
part of the result: the point head re-seeds the global generator before the ci terms are built)."""
import math
from typing import Optional

import numpy as np
import torch

from ....evaluators.metrics import robust_z
from ..base import BaseDetector
from .nets import PatchEncoder, build_memory_bank
from .scoring import ci_scores, distribute, knn, point_head_scores
from .training import embed_all, train_ci, train_official


class PaAnoDetector(BaseDetector):
    def fit_score(self, x: np.ndarray, train_end: int, seed: int = 0, dataset: Optional[str] = None) -> np.ndarray:
        a, device, tr = self.config, self.device, int(train_end)
        data = np.asarray(x).astype(np.float32)  # the runs loaded series as float32
        raw = data
        if not a.revin:
            mu, sd = data[:tr].mean(0, keepdims=True), data[:tr].std(0, keepdims=True)
            data = (data - mu) / np.where(sd == 0, 1e-8, sd)
        torch.manual_seed(seed)
        np.random.seed(seed)
        X = torch.as_tensor(data.T.copy(), device=device)
        L, C, T = a.patch, data.shape[1], data.shape[0]
        terms, pt = [], None
        self.parts = {}
        if a.mv:
            model = PatchEncoder(in_channels=C, use_revin=bool(a.revin)).to(device)
            train_official(model, X[:, :tr], L, device, a.iters, a.lr, seed)
            chunk = max(256, min(2048, int(2e8 / (C * L * 4))))
            emb_tr = embed_all(model, X[:, :tr], L, chunk)
            bk = a.bank_k
            if bk < 0:  # k = sqrt(n / 2): the memory grows with the training data
                bk = max(8, int(round(math.sqrt(emb_tr.shape[0] / 2))))
            bank = build_memory_bank(emb_tr, bank_size=bk, frac=0.0 if bk else 0.1, min_cores=bk or 500, seed=42,
                                     metric=a.metric).to(device)
            del emb_tr
            torch.cuda.empty_cache()
            emb_all = embed_all(model, X, L, chunk)
            pt = point_head_scores(emb_all, raw, tr, L, device, seed) if a.point else None
            terms.append(("mv", 1.0, knn(emb_all, bank, a.metric)))
            del emb_all, model, bank
            torch.cuda.empty_cache()
        if a.ci:
            if a.ci_norm == "center":  # series z-scored with training statistics, patches only mean-centred
                mu_t, sd_t = raw[:tr].mean(0, keepdims=True), raw[:tr].std(0, keepdims=True)
                Xci = torch.as_tensor(np.clip((raw - mu_t) / np.where(sd_t < 1e-8, 1.0, sd_t), -50, 50).T.astype(np.float32).copy(),
                                      device=device)
                model = PatchEncoder(in_channels=1, use_revin=False, center_channels=1).to(device)
            else:
                Xci = X
                model = PatchEncoder(in_channels=1, use_revin=bool(a.revin)).to(device)
            train_ci(model, Xci[:, :tr], L, device, a.iters, a.lr, seed)
            bkc = a.bank_k_ci if a.bank_k_ci is not None else a.bank_k
            terms.append(("ci", a.ci, ci_scores(model, Xci, tr, L, a.metric, a.ci_stride, a.ci_agg, seed, bank_k=bkc)))
            del model
            torch.cuda.empty_cache()
        if a.ci_amp:  # amplitude-preserving channel term on continuous channels (>= 10 distinct training values)
            cont = np.array([len(np.unique(raw[:tr, c])) >= 10 for c in range(raw.shape[1])])
            if cont.any():
                mu_t, sd_t = raw[:tr].mean(0, keepdims=True), raw[:tr].std(0, keepdims=True)
                Za = np.clip((raw - mu_t) / np.where(sd_t < 1e-8, 1.0, sd_t), -50, 50)[:, cont]
                Xa = torch.as_tensor(Za.T.astype(np.float32).copy(), device=device)
                model = PatchEncoder(in_channels=1, use_revin=False, center_channels=1).to(device)
                train_ci(model, Xa[:, :tr], L, device, a.iters, a.lr, seed + 11)
                terms.append(("ci_amp", a.ci_amp, ci_scores(model, Xa, tr, L, a.metric, a.ci_stride, a.ci_agg, seed + 11)))
                del model, Xa
                torch.cuda.empty_cache()
        del X
        if not terms:
            raise ValueError("PaAno config without any term (mv, ci, ci_amp all off)")
        ps = terms[0][2] if len(terms) == 1 else np.stack([w * robust_z(s) for _, w, s in terms]).sum(0)
        score = distribute(ps, T, L)
        for name, _, s in terms:
            self.parts[name] = distribute(s, T, L)
        if a.point and a.mv and pt is not None:  # point-level term added at the point level
            score = robust_z(score) + a.point * robust_z(pt[:T])
            self.parts["point"] = pt[:T]
        return score
