"""tiny synthetic series on the CPU: every built-in detector that needs no external clone returns one finite score per
point, and an injected level shift scores above the normal part"""
import numpy as np
import pytest

from industrial_tsad.configs import MSCADConfig, PaAnoConfig
from industrial_tsad.core.models import create_detector


def _series(T=900, C=2, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    x = np.stack([np.sin(2 * np.pi * t / 50 + c) for c in range(C)], 1) + 0.05 * rng.normal(size=(T, C))
    x[700:760, 0] += 3.0
    return x


@pytest.mark.parametrize("cfg", [
    PaAnoConfig(patch=32, iters=3),
    PaAnoConfig(patch=32, iters=3, ci=1.0, ci_amp=1.0, point=0.5),
])
def test_paano_smoke(cfg):
    x = _series()
    det = create_detector(cfg, device="cpu")
    s = det.fit_score(x, 400, seed=1)
    assert s.shape == (len(x),) and np.isfinite(s).all()
    assert s[700:760].mean() > np.median(s[:400])


@pytest.mark.parametrize("modes", [0, 4])
def test_mscad_smoke(modes):
    x = _series()
    cfg = MSCADConfig(window=32, patches=(4, 16), d_model=16, heads=2, layers=1, bridge_blocks=1, epochs=2,
                      max_windows_per_epoch=256, n_modes=modes)
    s = create_detector(cfg, device="cpu").fit_score(x, 400, seed=1)
    assert s.shape == (len(x),) and np.isfinite(s).all()
    assert s[700:760].mean() > np.median(s[:400])
