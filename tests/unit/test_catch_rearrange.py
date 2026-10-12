"""the six rearrange patterns CATCH uses, checked against explicit index arithmetic (and einops when installed)"""
import importlib.util

import pytest
import torch

from industrial_tsad.core.models.catch.detector import rearrange


def test_patterns_against_explicit_indexing():
    x = torch.arange(3 * 4 * 5).reshape(3, 4, 5)
    assert torch.equal(rearrange(x, "b c d -> (b c d) 1"), x.reshape(60, 1))
    assert torch.equal(rearrange(x.reshape(60), "(b c d) -> b c d", b=3, c=4, d=5), x)
    q = torch.randn(2, 7, 12)
    r = rearrange(q, "b n (h d) -> b h n d", h=3)
    assert r.shape == (2, 3, 7, 4) and torch.equal(r[1, 2, 5], q[1, 5, 8:12])
    back = rearrange(r, "b h n d -> b n (h d)")
    assert torch.equal(back, q)
    y = torch.randn(2, 5, 3, 8)
    z = rearrange(y, "b n c p -> (b n) p c")
    assert z.shape == (10, 8, 3) and torch.equal(z[7], y[1, 2].T)
    assert torch.equal(rearrange(z, "(b n) p c -> b n p c", b=2), y.permute(0, 1, 3, 2))
    with pytest.raises(NotImplementedError):
        rearrange(x, "b c d -> d c b")


@pytest.mark.skipif(importlib.util.find_spec("einops") is None, reason="einops not installed")
def test_equal_to_einops():
    import einops
    cases = [((3, 4, 5), "b c d -> (b c d) 1", {}), ((60,), "(b c d) -> b c d", dict(b=3, c=4, d=5)),
             ((2, 7, 12), "b n (h d) -> b h n d", dict(h=3)), ((2, 3, 7, 4), "b h n d -> b n (h d)", {}),
             ((2, 5, 3, 8), "b n c p -> (b n) p c", {}), ((10, 8, 3), "(b n) p c -> b n p c", dict(b=2))]
    for shape, pat, kw in cases:
        x = torch.randn(*shape)
        assert torch.equal(rearrange(x, pat, **kw), einops.rearrange(x, pat, **kw))
