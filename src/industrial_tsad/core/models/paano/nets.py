"""PaAno (Park & Kang, ICLR 2026; https://github.com/jinnnju/PaAno, MIT): patch encoder and coreset memory bank.

Faithful to the official code: Conv1d(k = 7, 5, 3, 3; ch = 128, 256, 128, 64) + BN + ReLU, global average pooling ->
64-d embedding; 2-layer projection head (256-d) for the triplet loss; linear head on concat(h_a, h_b) for the pretext
task; per-patch RevIN. Memory bank: k-means on L2-normalised embeddings, the real sample nearest to each centre kept
(official: sklearn MiniBatchKMeans with k-means++; here k-means++ + Lloyd on the GPU).
Additions: `center_channels` (one shared offset removed per patch instead of RevIN, keeps amplitude), K-mode banks.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RevIN1d(nn.Module):
    """per-sample, per-channel instance normalisation along time (normalisation only)"""

    def __init__(self, eps=1e-5, min_sigma=1e-5):
        super().__init__()
        self.eps = eps
        self.min_sigma = min_sigma

    def forward(self, x):
        mu = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        sigma = (var + self.eps).sqrt().clamp_min(self.min_sigma)
        return (x - mu) / sigma


def center_shared(x, n_ch):
    """subtract one offset per patch: the mean over the first n_ch channels and all time steps"""
    if n_ch <= 0:
        return x
    h = x[:, :n_ch]
    return torch.cat([h - h.mean(dim=(1, 2), keepdim=True), x[:, n_ch:]], dim=1)


class PatchEncoder(nn.Module):
    def __init__(self, in_channels=1, projection_dim=256, layers=(128, 256, 128, 64), kss=(7, 5, 3, 3),
                 use_revin=False, center_channels=0):
        super().__init__()
        self.center_channels = center_channels
        self.revin = RevIN1d() if use_revin else None
        blocks = []
        for i, (c, k) in enumerate(zip(layers, kss)):
            blocks.append(nn.Sequential(
                nn.Conv1d(layers[i - 1] if i > 0 else in_channels, c, kernel_size=k, stride=1, padding=k // 2, bias=False),
                nn.BatchNorm1d(c),
                nn.ReLU(inplace=True)))
        self.convblocks = nn.ModuleList(blocks)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.projection_head = nn.Sequential(nn.Linear(layers[-1], projection_dim), nn.ReLU(),
                                             nn.Linear(projection_dim, projection_dim))
        self.classification_head = nn.Linear(layers[-1] * 2, 1)
        self.embed_dim = layers[-1]

    def embedding(self, x):
        x = center_shared(x, self.center_channels)
        if self.revin is not None:
            x = self.revin(x)
        for block in self.convblocks:
            x = block(x)
        return self.gap(x).flatten(1)

    def projection(self, h):
        return self.projection_head(h)

    def forward(self, x):
        return self.embedding(x)


def _assign(x, c, chunk=None):
    """nearest centre index and squared distance for every row of x (chunked)"""
    chunk = chunk or max(1024, min(32768, 2 ** 25 // max(c.shape[0], 1)))
    idx = torch.empty(x.shape[0], dtype=torch.long, device=x.device)
    dist = torch.empty(x.shape[0], device=x.device)
    cn = (c * c).sum(1)
    for i in range(0, x.shape[0], chunk):
        xb = x[i:i + chunk]
        d = (xb * xb).sum(1, keepdim=True) - 2 * xb @ c.T + cn[None]
        v, a = d.min(1)
        idx[i:i + chunk], dist[i:i + chunk] = a, v.clamp_min(0)
    return idx, dist


def kmeans_torch(x, k, iters=50, seed=42):
    """k-means++ initialisation + Lloyd iterations on x's device; empty clusters re-seeded from the farthest points"""
    g = torch.Generator(device=x.device).manual_seed(seed)
    N = x.shape[0]
    c = torch.empty(k, x.shape[1], device=x.device)
    c[0] = x[torch.randint(0, N, (1,), generator=g, device=x.device)]
    d2 = ((x - c[0]) ** 2).sum(1)
    for j in range(1, k):
        tot = d2.sum()
        nxt = torch.randint(0, N, (1,), generator=g, device=x.device) if tot <= 0 else torch.multinomial(d2 / tot, 1, generator=g)
        c[j] = x[nxt[0]]
        d2 = torch.minimum(d2, ((x - c[j]) ** 2).sum(1))
    for _ in range(iters):
        a, dist = _assign(x, c)
        sums = torch.zeros_like(c).index_add_(0, a, x)
        cnt = torch.bincount(a, minlength=k).float()
        empty = cnt == 0
        c = torch.where(empty[:, None], c, sums / cnt.clamp_min(1)[:, None])
        if empty.any():
            far = torch.topk(dist, int(empty.sum())).indices
            c[empty] = x[far]
    return c


def build_memory_bank(embeddings, bank_size=4096, frac=0.1, min_cores=500, seed=42, iters=50, metric="cosine"):
    """official rule: k = frac * N (>= min_cores) k-means centres on L2-normalised embeddings, then the real sample
    nearest to each centre; capped at bank_size (<= 0: no cap); all embeddings when k >= N - 1"""
    emb = embeddings.detach().float()
    N = emb.shape[0]
    k = max(min(min_cores, N - 1), int(round(frac * N)))
    if bank_size and bank_size > 0:
        k = min(k, bank_size)
    if k >= N - 1:
        return emb.cpu()
    flat = F.normalize(emb, dim=1) if metric == "cosine" else emb
    centers = kmeans_torch(flat, k, iters=iters, seed=seed)
    nearest = torch.empty(k, dtype=torch.long, device=flat.device)
    best = torch.full((k,), float("inf"), device=flat.device)
    chunk = max(1024, min(32768, 2 ** 25 // k))
    for i in range(0, N, chunk):
        xb = flat[i:i + chunk]
        d = (xb * xb).sum(1, keepdim=True) - 2 * xb @ centers.T + (centers * centers).sum(1)[None]
        v, a = d.min(dim=0)
        upd = v < best
        best[upd] = v[upd]
        nearest[upd] = a[upd] + i
    return emb[nearest].cpu()
