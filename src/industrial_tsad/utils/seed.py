import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """torch (CPU + CUDA), numpy and python seeds; the detectors also seed their own generators explicitly"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
