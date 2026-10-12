"""TimesNet (Wu et al., ICLR 2023) through TSB-AD's implementation, exactly as TSB_AD.model_wrapper.run_TimesNet:
TimesNet(win_size, enc_in, lr, epochs).fit(train part) and decision_function(whole series)."""
from typing import Optional

import numpy as np
import torch

from ....utils.external import ensure_tsbad
from ..base import BaseDetector


class TimesNetDetector(BaseDetector):
    def fit_score(self, x: np.ndarray, train_end: int, seed: int = 0, dataset: Optional[str] = None) -> np.ndarray:
        ensure_tsbad()
        from TSB_AD.models.TimesNet import TimesNet
        torch.manual_seed(seed)
        np.random.seed(seed)
        x = np.asarray(x, np.float64)
        c = self.config
        clf = TimesNet(win_size=c.win_size, enc_in=x.shape[1], lr=c.lr, epochs=c.epochs)
        clf.fit(x[:train_end])
        score = np.asarray(clf.decision_function(x)).ravel()
        del clf
        torch.cuda.empty_cache()
        return score
