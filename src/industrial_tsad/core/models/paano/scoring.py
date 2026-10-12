"""PaAno scoring: kNN distance to the memory bank, patch -> point distribution, channel-independent term and the
point-level next-step head."""
import math

import numpy as np
import torch
import torch.nn.functional as F

from ....evaluators.metrics import robust_z
from .nets import build_memory_bank
from .training import patches_ci


@torch.inference_mode()
def dist_matrix(x, bank, metric):
    """(n, m) distances in PaAno's score unit: 1 - cos (official), or RMS L2 per embedding dimension"""
    if metric == "cosine":
        return 1 - F.normalize(x, dim=1) @ F.normalize(bank, dim=1).T
    d2 = ((x * x).sum(1, keepdim=True) - 2 * x @ bank.T + (bank * bank).sum(1)[None]).clamp_min(0)
    return (d2 / x.shape[1]).sqrt()


@torch.inference_mode()
def knn(q, bank, metric, k=3, chunk=8192):
    """mean of the k smallest distances to the bank"""
    k = min(k, len(bank))
    out = []
    for i in range(0, len(q), chunk):
        d = dist_matrix(q[i:i + chunk], bank, metric)
        out.append(torch.topk(d, k, dim=1, largest=False).values.mean(1))
    return torch.cat(out).cpu().numpy()


def distribute(ps, T, L):
    """point score = mean of the scores of the patches covering the point (non-circular)"""
    N = len(ps)
    cs = np.r_[0.0, np.cumsum(ps, dtype=np.float64)]
    p = np.arange(T)
    hi = np.minimum(p, N - 1)
    lo = np.maximum(0, p - L + 1)
    return ((cs[hi + 1] - cs[lo]) / (hi - lo + 1)).astype(np.float32)


def patch_level_features(Z, L):
    """Z (T, C) -> (T - L + 1, 2C): per-channel mean and std of every patch (what the per-patch RevIN removes)"""
    cs = np.vstack([np.zeros((1, Z.shape[1])), np.cumsum(Z, 0, dtype=np.float64)])
    cs2 = np.vstack([np.zeros((1, Z.shape[1])), np.cumsum(Z.astype(np.float64) ** 2, 0)])
    mean = (cs[L:] - cs[:-L]) / L
    var = np.maximum((cs2[L:] - cs2[:-L]) / L - mean ** 2, 0)
    return np.hstack([mean, np.sqrt(var)]).astype(np.float32)


def point_head_scores(emb_all, raw, tr, L, device, seed, iters=500, minvals=10, hidden=256):
    """point-level heteroscedastic next-step head on the frozen PaAno embedding: for every patch start s the next value
    x[s + L] of each continuous channel (>= minvals distinct training values) is predicted as a change from the last
    value, Laplace location m and scale b from an MLP on [standardised embedding, per-channel patch mean, log patch std,
    last value] (training-z units), Laplace NLL on rows inside the training part. Score = |delta - m| / b, robust z per
    channel, mean of the top-3 channels; the first L points take the median. None when no channel is continuous."""
    mu, sd = raw[:tr].mean(0), raw[:tr].std(0)
    Zg = np.clip((raw - mu) / np.where(sd < 1e-8, 1.0, sd), -50, 50).astype(np.float32)
    cont = np.array([len(np.unique(raw[:tr, c])) >= minvals for c in range(raw.shape[1])])
    if not cont.any():
        return None
    T, C = Zg.shape
    n = T - L
    feat_lv = patch_level_features(Zg, L)[:n]
    feat_lv[:, C:] = np.log(feat_lv[:, C:] + 1e-3)
    last = Zg[L - 1:L - 1 + n]
    e = emb_all[:n].float()
    ntr = max(1, min(n, tr - L))
    e = (e - e[:ntr].mean(0)) / (e[:ntr].std(0) + 1e-6)
    feats = torch.cat([e, torch.as_tensor(np.hstack([feat_lv, last]), device=e.device)], 1)
    delta = torch.as_tensor(Zg[L:L + n][:, cont] - last[:, cont], device=e.device)
    kc = int(cont.sum())
    torch.manual_seed(seed + 7)
    net = torch.nn.Sequential(torch.nn.Linear(feats.shape[1], hidden), torch.nn.GELU(), torch.nn.Linear(hidden, 2 * kc)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    g = torch.Generator(device="cpu").manual_seed(seed + 7)
    for _ in range(iters):
        b_idx = torch.randint(0, ntr, (min(1024, ntr),), generator=g).to(device)
        out = net(feats[b_idx])
        m, b = out[:, :kc], F.softplus(out[:, kc:]) + 1e-3
        loss = ((delta[b_idx] - m).abs() / b + torch.log(b)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.inference_mode():
        errs = []
        for i in range(0, n, 65536):
            out = net(feats[i:i + 65536])
            m, b = out[:, :kc], F.softplus(out[:, kc:]) + 1e-3
            errs.append(((delta[i:i + 65536] - m).abs() / b).cpu().numpy())
    E = robust_z(np.concatenate(errs, 0), axis=0)
    pt = np.sort(E, 1)[:, -min(3, kc):].mean(1)
    return np.r_[np.full(L, np.median(pt)), pt].astype(np.float32)


@torch.inference_mode()  # without it every chunk keeps its autograd graph (tens of GB on long multichannel series)
def ci_scores(model, X, tr, L, metric, stride, agg, seed, bank_max=2000, chunk=4096, bank_k=0):
    """per channel: embeddings of every stride-th patch, bank = up to bank_max random training patches of that channel
    (or bank_k k-means modes), top-3 kNN, robust z over the series; aggregated over channels (max, mean of the top 3,
    or mean of the top max(3, C/10)) -> (N,) on the stride grid, repeated back to every start"""
    C, T = X.shape
    N = T - L + 1
    starts = torch.arange(0, N, stride, device=X.device)
    n_tr = tr - L + 1
    g = torch.Generator(device="cpu").manual_seed(seed)
    zs = np.empty((C, len(starts)), np.float32)
    for c in range(C):
        ch = torch.full((len(starts),), c, device=X.device, dtype=torch.long)
        e = torch.cat([model.embedding(patches_ci(X, ch[i:i + chunk], starts[i:i + chunk], L)).float()
                       for i in range(0, len(starts), chunk)])
        tr_rows = torch.nonzero(starts < n_tr).squeeze(1)
        pick = tr_rows[torch.randperm(len(tr_rows), generator=g)[:bank_max].to(X.device)]
        kc = bank_k if bank_k >= 0 else max(8, int(round(math.sqrt(len(tr_rows) / 2))))
        bank = e[pick] if not kc else build_memory_bank(e[tr_rows], bank_size=kc, frac=0.0, min_cores=kc, seed=seed + c,
                                                        metric=metric).to(X.device)
        zs[c] = robust_z(knn(e, bank, metric))
    if agg == "max":
        z = zs.max(0)
    else:
        kk = max(3, math.ceil(C / 10)) if agg == "topfrac" else 3
        z = np.sort(zs, 0)[-min(kk, C):].mean(0)
    return np.repeat(z, stride)[:N]
