import numpy as np

from industrial_tsad.core.fusion import RegimeGate, rz_sum
from industrial_tsad.evaluators.metrics import robust_z


def test_robust_z_centres_and_keeps_dtype():
    s = np.r_[np.zeros(50), np.ones(50)].astype(np.float32) + np.linspace(0, 1e-3, 100, dtype=np.float32)
    z = robust_z(s)
    assert z.dtype == np.float32
    assert abs(np.median(z)) < 1e-3


def test_rz_sum_truncates_and_weights():
    a, b = np.arange(10.0), np.arange(12.0)[::-1]
    f = rz_sum([a, b], [1.0, 0.0])
    assert len(f) == 10 and np.allclose(f, robust_z(a))


def test_regime_gate_switch():
    base = np.random.default_rng(0).normal(size=200)
    quiet = np.zeros(200)
    loud = np.r_[np.zeros(100), np.full(100, 9.0)]
    g = RegimeGate(tau=5.0, weight=1.0)
    s, on, stat = g(base, quiet, 100)
    assert not on and stat == 0.0 and np.array_equal(s, base)
    s, on, stat = g(base, loud, 100)
    assert on and stat == 9.0 and len(s) == 200
