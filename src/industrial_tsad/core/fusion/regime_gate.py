"""Regime gate: add an innovation (prediction-mismatch) term only on series whose innovation model reports a regime
mismatch, i.e. S = rz(base) + w * rz(innovation) if median(innovation z over the test part) > tau, else base alone.
On TSB-AD-M the innovation term is AxonAD's JEPA tail distance (robust z by its training median / IQR); tau = 5,
w = 1 (chosen on the 20 tuning series, inside a wide gap of the statistic). PaAno final + gate: eval 0.589 +- 0.001
over three AxonAD seeds (docs/results_tsbad.md)."""
from dataclasses import dataclass

import numpy as np

from ...evaluators.metrics import robust_z


@dataclass(frozen=True)
class RegimeGate:
    tau: float = 5.0
    weight: float = 1.0

    def statistic(self, innovation: np.ndarray, train_end: int) -> float:
        j = np.asarray(innovation, np.float64)
        return float(np.median(j[train_end:])) if train_end < len(j) else 0.0

    def __call__(self, base: np.ndarray, innovation: np.ndarray, train_end: int):
        """-> (score, gate_on, statistic); base is returned unchanged when the gate is off"""
        stat = self.statistic(innovation, train_end)
        if stat <= self.tau:
            return np.asarray(base), False, stat
        n = min(len(base), len(innovation))
        b = np.asarray(base, np.float64)[:n]
        j = np.asarray(innovation, np.float64)[:n]
        return robust_z(b) + self.weight * robust_z(j), True, stat
