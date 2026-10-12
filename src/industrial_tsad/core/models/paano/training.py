"""PaAno training objective on one series (non-circular), as the official TSB-AD run: anchors at random patch starts,
positives 1..2 steps away, triplet loss with the released farthest-negative rule on L2-normalised projections, and the
"is b the patch L steps before a?" pretext (BCE, 5 random negatives, weight 1 -> 0 over the first 20 % of iterations);
AdamW (wd 1e-4), cosine LR lr -> lr / 10, checkpoint = lowest batch loss."""
import copy
import math

import torch
import torch.nn.functional as F


def patches(X, idx, L):
    """X (C, T) on device, idx (B,) start positions -> (B, C, L)"""
    return X[:, idx[:, None] + torch.arange(L, device=X.device)].permute(1, 0, 2)


def patches_ci(X, ch, idx, L):
    """X (C, T), ch / idx (B,) -> (B, 1, L): one channel per patch (channel-independent encoder)"""
    return X[ch[:, None], idx[:, None] + torch.arange(L, device=X.device)].unsqueeze(1)


def _cosine_lr(opt, lr, it, num_iter):
    for pg in opt.param_groups:
        pg["lr"] = lr / 10 + (lr - lr / 10) * 0.5 * (1 + math.cos(math.pi * min(it, num_iter) / num_iter))


def train_official(model, Xtr, L, device, num_iter, lr, seed, batch=512):
    g = torch.Generator(device="cpu").manual_seed(seed)
    N = Xtr.shape[1] - L + 1
    radius, num_rand, mu, margin = 2, 5, 10.0, 0.1
    offsets = torch.tensor([*range(-radius, 0), *range(1, radius + 1)])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    best, best_state = float("inf"), copy.deepcopy(model.state_dict())
    model.train()
    for it in range(1, num_iter + 1):
        _cosine_lr(opt, lr, it, num_iter)
        M = min(batch, N)
        a = torch.randint(0, N, (M,), generator=g)
        cand = a[:, None] + offsets[None]
        valid = (cand >= 0) & (cand < N)
        score = torch.where(valid, torch.rand(cand.shape, generator=g), torch.full(cand.shape, -1.0))
        p = cand.gather(1, score.argmax(1, keepdim=True)).squeeze(1)
        p = torch.where(valid.any(1), p, a)
        lam = 1 - it / (num_iter / 5) if it < num_iter / 5 else 0.0
        ad, pd_ = a.to(device), p.to(device)
        if lam > 0:
            pre = a - L
            pmask = pre >= 0
            h = model.embedding(torch.cat([patches(Xtr, ad, L), patches(Xtr, pd_, L), patches(Xtr, pre.clamp_min(0).to(device), L)]))
            h_a, h_p, h_pre = h[:M], h[M:2 * M], h[2 * M:]
        else:
            h = model.embedding(torch.cat([patches(Xtr, ad, L), patches(Xtr, pd_, L)]))
            h_a, h_p = h[:M], h[M:]
        z_a = F.normalize(model.projection(h_a), dim=1)
        z_p = F.normalize(model.projection(h_p), dim=1)
        sim = z_a @ z_p.T
        sim_f = sim.clone()
        sim_f.diagonal().fill_(float("inf"))
        loss = F.relu((1 - sim.diag()) - (1 - sim_f).max(1).values + margin).mean() / mu
        if lam > 0 and pmask.any():
            pm_d = pmask.to(device)
            f_pre = torch.cat([h_a[pm_d], h_pre[pm_d]], 1)
            ai = torch.arange(M, device=device).repeat_interleave(num_rand)
            ui = (ai + torch.randint(1, M, (M * num_rand,), generator=g).to(device)) % M
            f_un = torch.cat([h_a.repeat_interleave(num_rand, 0), h_a[ui]], 1)
            logits = model.classification_head(torch.cat([f_pre, f_un])).squeeze(1)
            lab = torch.cat([torch.ones(len(f_pre), device=device), torch.zeros(len(f_un), device=device)])
            la = bce(logits, lab)
            loss = loss + lam * (la[:len(f_pre)].mean() + la[len(f_pre):].mean())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if loss.item() < best:
            best, best_state = loss.item(), copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()


def train_ci(model, Xtr, L, device, num_iter, lr, seed, batch=512):
    """train_official with channel-independent patches: anchor = (channel, start), positive = same channel 1..2 steps
    away, pretext = same channel L steps before"""
    g = torch.Generator(device="cpu").manual_seed(seed)
    C, N = Xtr.shape[0], Xtr.shape[1] - L + 1
    offsets = torch.tensor([-2, -1, 1, 2])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    best, best_state = float("inf"), copy.deepcopy(model.state_dict())
    model.train()
    for it in range(1, num_iter + 1):
        _cosine_lr(opt, lr, it, num_iter)
        M = min(batch, N * C)
        c = torch.randint(0, C, (M,), generator=g)
        a_ = torch.randint(0, N, (M,), generator=g)
        cand = a_[:, None] + offsets[None]
        valid = (cand >= 0) & (cand < N)
        score = torch.where(valid, torch.rand(cand.shape, generator=g), torch.full(cand.shape, -1.0))
        p = cand.gather(1, score.argmax(1, keepdim=True)).squeeze(1)
        p = torch.where(valid.any(1), p, a_)
        lam = 1 - it / (num_iter / 5) if it < num_iter / 5 else 0.0
        cd, ad, pd_ = c.to(device), a_.to(device), p.to(device)
        parts = [patches_ci(Xtr, cd, ad, L), patches_ci(Xtr, cd, pd_, L)]
        pre = a_ - L
        pmask = pre >= 0
        if lam > 0:
            parts.append(patches_ci(Xtr, cd, pre.clamp_min(0).to(device), L))
        h = model.embedding(torch.cat(parts))
        h_a, h_p = h[:M], h[M:2 * M]
        z_a = F.normalize(model.projection(h_a), dim=1)
        z_p = F.normalize(model.projection(h_p), dim=1)
        sim = z_a @ z_p.T
        sim_f = sim.clone()
        sim_f.diagonal().fill_(float("inf"))
        loss = F.relu((1 - sim.diag()) - (1 - sim_f).max(1).values + 0.1).mean() / 10.0
        if lam > 0 and pmask.any():
            h_pre = h[2 * M:]
            pm_d = pmask.to(device)
            f_pre = torch.cat([h_a[pm_d], h_pre[pm_d]], 1)
            ai = torch.arange(M, device=device).repeat_interleave(5)
            ui = (ai + torch.randint(1, M, (M * 5,), generator=g).to(device)) % M
            f_un = torch.cat([h_a.repeat_interleave(5, 0), h_a[ui]], 1)
            logits = model.classification_head(torch.cat([f_pre, f_un])).squeeze(1)
            lab = torch.cat([torch.ones(len(f_pre), device=device), torch.zeros(len(f_un), device=device)])
            la = bce(logits, lab)
            loss = loss + lam * (la[:len(f_pre)].mean() + la[len(f_pre):].mean())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if loss.item() < best:
            best, best_state = loss.item(), copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()


@torch.inference_mode()
def embed_all(model, X, L, chunk):
    N = X.shape[1] - L + 1
    out = []
    for i in range(0, N, chunk):
        out.append(model.embedding(patches(X, torch.arange(i, min(N, i + chunk), device=X.device), L)).float())
    return torch.cat(out)
