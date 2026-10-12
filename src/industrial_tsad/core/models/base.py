from abc import ABC, abstractmethod
from typing import Dict, Optional

import numpy as np
import torch


class BaseDetector(ABC):
    """Semi-supervised detector on one series: train on the prefix x[:train_end] (normal by the TSB-AD convention) and
    return one anomaly score per point of the whole series (higher = more anomalous)."""

    def __init__(self, config, device: Optional[str] = None):
        self.config = config
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.parts: Dict[str, np.ndarray] = {}  # optional named score parts of the last call (e.g. AxonAD mse / jepa)

    @property
    def name(self) -> str:
        return self.config.type

    @abstractmethod
    def fit_score(self, x: np.ndarray, train_end: int, seed: int = 0, dataset: Optional[str] = None) -> np.ndarray:
        """x (T, C) -> (T,) point scores. `dataset` lets detectors with per-dataset presets pick them (CATCH)."""
