"""Detector configs (discriminated by ``type``). Defaults = the settings used for the reported TSB-AD runs."""
from typing import Annotated, Dict, Literal, Optional, Tuple, Union

from pydantic import Field

from .base import BaseConfig


class SeedPolicy(BaseConfig):
    """per-series seed = base + index of the series in its file list (per_series) or base for every series"""
    base: int = 2027
    per_series: bool = True


class BaseDetectorConfig(BaseConfig):
    seed: SeedPolicy = Field(default_factory=SeedPolicy)


class PaAnoConfig(BaseDetectorConfig):
    """PaAno (ICLR 2026) on one series + optional terms. Official TSB-AD-M run = defaults; 'final' preset adds
    ci = 1, ci_amp = 1, point = 0.5 (TSB-AD-M eval 0.544 +- 0.001)."""
    type: Literal["paano"] = "paano"
    patch: int = 96
    iters: int = 100
    lr: float = 1e-4
    metric: Literal["cosine", "l2"] = "cosine"
    revin: bool = True
    mv: bool = Field(True, description="official multivariate term")
    ci: float = Field(0.0, description="weight of the channel-independent term (0 = off)")
    ci_stride: int = 2
    ci_agg: Literal["max", "top3", "topfrac"] = "top3"
    ci_norm: Literal["revin", "center"] = "revin"
    ci_amp: float = Field(0.0, description="weight of the amplitude-preserving channel term on continuous channels")
    point: float = Field(0.0, description="weight of the point-level heteroscedastic next-step head")
    bank_k: int = Field(0, description="0 = official memory (10 % of patches, >= 500); K > 0 = K k-means modes")
    bank_k_ci: Optional[int] = None


class ModeAdapterConfig(BaseConfig):
    """mode-token adapter on AxonAD: K learnable mode tokens, winner-take-all residual.
    space = recon : modes decoded into channel patterns and added to the reconstruction (L2 selection per time step)
    space = query : modes in the JEPA query space; selection per head (K^H combinations) or jointly over the heads;
                    prior = gate adds a context gate and scores the mixture likelihood"""
    n_modes: int = 64
    space: Literal["recon", "query"] = "recon"
    selection: Literal["per_head", "joint"] = "joint"
    prior: Literal["none", "gate"] = "none"
    history_only: bool = Field(True, description="query space: modes read only steps before every supervised step")


class AxonADConfig(BaseDetectorConfig):
    """AxonAD (ECML-PKDD 2026) with its TSB-AD settings; score = robust-z reconstruction MSE + robust-z JEPA tail
    distance (centre-padded). Seed 2024 for every series (as the published run)."""
    type: Literal["axonad"] = "axonad"
    seed: SeedPolicy = Field(default_factory=lambda: SeedPolicy(base=2024, per_series=False))
    win_size: int = 100
    d_model: int = 128
    num_heads: int = 8
    lr: float = 5e-4
    kl_tail_k: int = 10
    forecast_steps: int = 1
    epochs: int = 50
    adapter: Optional[ModeAdapterConfig] = None


class MSCADConfig(BaseDetectorConfig):
    """MSCAD (arXiv 2609.38004), our re-implementation from the paper text (no official code)."""
    type: Literal["mscad"] = "mscad"
    window: int = 128
    patches: Tuple[int, ...] = (4, 16, 64)
    d_model: int = 256
    heads: int = 4
    layers: int = 2
    bridge_blocks: int = 2
    lr: float = 1e-3
    epochs: int = 30
    batch: int = 128
    patience: int = 5
    max_windows_per_epoch: int = 20000
    max_infer_windows: int = 3_000_000
    n_modes: int = Field(0, description="mode-token adapter: K learnable mode tokens per scale (0 = plain MSCAD)")


class CATCHConfig(BaseDetectorConfig):
    """CATCH (ICLR 2025) through its official wrapper class; hyper-parameters = the repository's per-dataset script when
    the dataset exists there (dataset_presets), else the class defaults, then `overrides`."""
    type: Literal["catch"] = "catch"
    dataset_presets: bool = True
    overrides: Dict[str, Union[int, float, str, bool]] = Field(default_factory=dict)


class KANADConfig(BaseDetectorConfig):
    """KAN-AD (ICML 2025): model class from the official clone, training/scoring re-implemented; univariate, so a
    multivariate series is scored channel by channel and averaged"""
    type: Literal["kanad"] = "kanad"
    window: int = 96
    order: int = 2
    batch: int = 1024
    epochs: int = 100
    lr: float = 0.01
    channel_agg: Literal["mean", "max"] = "mean"


class TimesNetConfig(BaseDetectorConfig):
    """TimesNet through TSB-AD's implementation and its run_TimesNet settings (win 96, lr 1e-4, 50 epochs)"""
    type: Literal["timesnet"] = "timesnet"
    win_size: int = 96
    lr: float = 1e-4
    epochs: int = 50


DetectorConfig = Annotated[Union[PaAnoConfig, AxonADConfig, MSCADConfig, CATCHConfig, KANADConfig, TimesNetConfig],
                           Field(discriminator="type")]
