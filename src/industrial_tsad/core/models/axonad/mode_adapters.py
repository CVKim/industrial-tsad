"""Mode-token adapters on AxonAD's TCN-Transformer (the backbone is untouched; adapter parameters are added after it).

K learnable mode tokens, winner-take-all (WTA) residual: the mode that best explains the observation is added to the
model's output with gradient, its selection without gradient.
  RECON  (space = recon): the tokens read the window's hidden states (cross-attention) and are decoded by the model's
         own reconstruction head (minus its bias) into channel patterns; per time step the pattern minimising the L2
         reconstruction residual is added to the reconstruction.
  QUERY  (space = query): the tokens read only the window's history (steps before every JEPA-supervised step, so the
         answer cannot leak into them), are mapped by the online query projection, and the mode maximising
         cos(q_target, q_pred + m_k) is added to the predicted query - per head (K^H combinations per step) or jointly
         over the heads (K combinations). prior = gate adds a gate p(k | causal context) trained by cross-entropy on the
         WTA choice (detached inputs) and scored as the mixture likelihood -log sum_k p_k exp(kappa (cos_k - 1)), kappa
         from the von Mises-Fisher ML approximation on the training windows.
Finding on TSB-AD-M (tuning 20 / eval 180): RECON K64 +0.004 (n.s.); QUERY loses GHL in every variant - choosing the mode
with the observed future explains sustained anomalies away (docs/results_tsbad.md)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....utils.external import ensure_tsbad


def _base_model():
    ensure_tsbad()
    from TSB_AD.models.AxonAD import TCNTransformerModel
    return TCNTransformerModel


def build_mode_model(feats, seq_len, d_model, num_heads, forecast_steps, adapter, tail):
    """TCNTransformerModel subclass with the requested adapter"""
    Base = _base_model()

    class ModeTCN(Base):
        def __init__(self):
            super().__init__(feats, seq_len, d_model=d_model, num_heads=num_heads, forecast_steps=forecast_steps)
            self.K, self.space, self.selection, self.prior = adapter.n_modes, adapter.space, adapter.selection, adapter.prior
            s = max(0, min(self.forecast_steps, seq_len))
            self.ctx = seq_len - max(1, (seq_len - s) // 2) if adapter.history_only else seq_len - tail
            self.kappa = 1.0
            self.aux = {}
            if self.space == "recon":
                self.mode_q = nn.Parameter(torch.randn(1, self.K, d_model) * 0.02)
                self.mode_ln = nn.LayerNorm(d_model)
                self.mode_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
            else:
                self.mode_qj = nn.Parameter(torch.randn(1, self.K, d_model) * 0.02)
                self.mode_lnj = nn.LayerNorm(d_model)
                self.mode_attnj = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
                if self.prior == "gate":
                    self.gate_q = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
                    self.gate_k = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model))

        def forward(self, x):
            B, T, _ = x.shape
            h = self.embed(x) + self.pos
            h_norm = self.ln1(h)
            h_attn, q_real, q_pred, k = self.attn(h_teacher=h_norm, h_hist=h_norm)
            h2 = h + h_attn
            h2 = h2 + self.ffn(self.ln2(h2))
            rec = self.head(h2)
            if self.space == "recon":
                mq = self.mode_q.expand(B, -1, -1)
                ctx = self.ln2(h2)
                mt = mq + self.mode_attn(self.mode_ln(mq), ctx, ctx, need_weights=False)[0]
                mode_out = self.head(mt) - self.head.bias
                with torch.no_grad():
                    kstar = ((x - rec)[:, :, None, :] - mode_out[:, None, :, :]).pow(2).sum(-1).argmin(-1)
                rec = rec + torch.gather(mode_out, 1, kstar[..., None].expand(-1, -1, mode_out.shape[-1]))
            with torch.no_grad():
                ht = self.t_embed(x) + self.t_pos
                q_tgt = self._split_heads(self.t_wq(self.t_ln1(ht)), self.attn.num_heads)
            if self.space == "query":
                q_pred = self._query_modes(B, T, h_norm, q_pred, q_tgt)
            if self.training:
                s = max(0, min(self.forecast_steps, T))
                tmask = self._make_tail_mask(B=B, T=T, s=s, tail_len=max(1, (T - s) // 2), device=x.device)
            else:
                tmask = torch.zeros(B, T, dtype=torch.bool, device=x.device)
            return rec, q_real, q_pred, k, q_tgt, tmask

        def _query_modes(self, B, T, h_norm, q_pred, q_tgt):
            H, K, c = self.attn.num_heads, self.K, self.ctx
            Dh = self.d_model // H
            hist = h_norm[:, :c]
            mq = self.mode_qj.expand(B, -1, -1)
            mt = mq + self.mode_attnj(self.mode_lnj(mq), hist, hist, need_weights=False)[0]
            mh = self.attn.wq(mt).view(B, K, H, Dh).transpose(1, 2)  # (B, H, K, Dh)
            qt, qp = q_tgt[:, :, c:], q_pred[:, :, c:]
            with torch.no_grad():  # cos(q_tgt, q_pred + m_k) for every mode without materialising q_pred + m_k
                d_tp = (qt * qp).sum(-1, keepdim=True)
                d_tm = torch.einsum("bhtd,bhkd->bhtk", qt, mh)
                d_pm = torch.einsum("bhtd,bhkd->bhtk", qp, mh)
                pp = (qp * qp).sum(-1, keepdim=True)
                mm = (mh * mh).sum(-1)[:, :, None, :]
                tn = qt.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                cos = (d_tp + d_tm) / (tn * (pp + 2 * d_pm + mm).clamp_min(1e-12).sqrt())  # (B, H, T - c, K)
                if self.selection == "joint":
                    kj = cos.mean(1).argmax(-1)  # one mode per step for all heads
                    kstar = kj[:, None, :].expand(-1, H, -1)
                else:
                    kstar = cos.argmax(-1)
            sel = torch.gather(mh, 2, kstar[..., None].expand(-1, -1, -1, Dh))
            if self.prior == "gate":
                gq = self.gate_q(q_pred.detach().transpose(1, 2).reshape(B, T, self.d_model)[:, c:])
                gq = gq.view(B, T - c, H, Dh).transpose(1, 2)
                gk = self.gate_k(mt.detach()).view(B, K, H, Dh).transpose(1, 2)
                logp = F.log_softmax(torch.einsum("bhtd,bhkd->bhtk", gq, gk) / math.sqrt(Dh), dim=-1)
                self.aux = ({"gate_ce": F.nll_loss(logp.reshape(-1, K), kstar.reshape(-1))} if self.training
                            else {"cos": cos, "logp": logp})
            else:
                self.aux = {} if self.training else {"cos": cos}
            return torch.cat([q_pred[:, :, :c], qp + sel], dim=2)

    return ModeTCN()
