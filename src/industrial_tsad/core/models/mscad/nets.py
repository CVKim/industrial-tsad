"""MSCAD (Guo et al., "No Scale Left Behind: Multi-Scale Autoencoder with Bi-directional Attention for Time Series
Anomaly Detection", arXiv 2609.38004) - OUR re-implementation from the paper text (no official code).
From the paper: window W = 128; parallel branches with patch sizes 4 / 16 / 64; per branch a linear patch embedding to
d = 256 and L = 2 Transformer encoder layers (4 heads, GELU FFN 4d), parameters not shared between branches; B = 2
bidirectional cross-scale blocks (each scale attends to the concatenated latents of the other scales; cross-attention +
FFN, residual + LayerNorm; block parameters shared across scales); per-branch linear decoder back to patches.
Our choices (fixed before any run): pre-norm Transformer blocks, learnable positional embeddings.
Mode-token adapter (n_modes = K > 0): K learnable tokens per scale read the window's patch tokens in the branch encoder
and are decoded into patch patterns; the window error then uses, per patch, the pattern that best explains the
residual (winner-take-all). The tokens are created last, so K = 0 builds exactly the plain model."""
import torch
import torch.nn as nn


class Bridge(nn.Module):
    """one bidirectional cross-scale block shared by all scales: z_s <- z_s + CA(z_s, concat(z_t, t != s)); FFN"""

    def __init__(self, d, heads):
        super().__init__()
        self.nq, self.nkv, self.nf = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, zs):
        out = []
        for s, z in enumerate(zs):
            ctx = self.nkv(torch.cat([zs[t] for t in range(len(zs)) if t != s], 1))
            z = z + self.att(self.nq(z), ctx, ctx, need_weights=False)[0]
            out.append(z + self.ff(self.nf(z)))
        return out


class MSCAD(nn.Module):
    def __init__(self, window=128, patches=(4, 16, 64), d=256, heads=4, layers=2, bridge_blocks=2, n_modes=0):
        super().__init__()
        self.W, self.patches = window, tuple(patches)
        self.emb = nn.ModuleList([nn.Linear(p, d) for p in self.patches])
        self.pos = nn.ParameterList([nn.Parameter(torch.zeros(1, window // p, d)) for p in self.patches])

        def layer():
            return nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)

        self.enc = nn.ModuleList([nn.TransformerEncoder(layer(), layers, enable_nested_tensor=False) for _ in self.patches])
        self.bridge = nn.ModuleList([Bridge(d, heads) for _ in range(bridge_blocks)])
        self.dec = nn.ModuleList([nn.Sequential(nn.LayerNorm(d), nn.Linear(d, p)) for p in self.patches])
        for p in self.pos:
            nn.init.normal_(p, std=0.02)
        self.K = n_modes
        if n_modes:
            self.modes = nn.ParameterList([nn.Parameter(torch.randn(1, n_modes, d) * 0.02) for _ in self.patches])

    def forward(self, x):
        """x (B, W) -> per-scale reconstructions (B, W); with modes also per-scale mode patterns (B, K, P_s)"""
        W = self.W
        zs, mos = [], []
        for i, p in enumerate(self.patches):
            tok = self.emb[i](x.view(len(x), W // p, p)) + self.pos[i]
            if self.K:
                tok = torch.cat([tok, self.modes[i].expand(len(x), -1, -1)], 1)
            z = self.enc[i](tok)
            if self.K:
                mos.append(z[:, W // p:])
                z = z[:, :W // p]
            zs.append(z)
        for blk in self.bridge:
            zs = blk(zs)
        recs = [self.dec[i](z).reshape(len(x), W) for i, z in enumerate(zs)]
        if not self.K:
            return recs
        return recs, [self.dec[i](m) for i, m in enumerate(mos)]


def window_errors(model, Xw):
    """(B, W) -> (B,) window score = mean over scales of the reconstruction MSE; with mode tokens, per patch the mode
    pattern that best explains the residual is subtracted first (min_k ||x - (x_hat + m_k)||^2)"""
    out = model(Xw)
    if not getattr(model, "K", 0):
        return torch.stack([((r - Xw) ** 2).mean(1) for r in out], 0).mean(0)
    recs, mps = out
    errs = []
    for (r, m), p in zip(zip(recs, mps), model.patches):
        res = (Xw - r).view(len(Xw), model.W // p, p)
        e = ((res[:, :, None, :] - m[:, None, :, :]) ** 2).mean(-1)
        errs.append(e.min(-1).values.mean(1))
    return torch.stack(errs, 0).mean(0)
