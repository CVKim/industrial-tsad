from typing import Literal, Optional

from pydantic import Field

from .base import BaseConfig
from .models import DetectorConfig


class BenchmarkRunConfig(BaseConfig):
    """one detector on one TSB-AD file list. Several worker processes may share a run (claim files)."""
    detector: DetectorConfig
    benchmark: Literal["M", "U"] = "M"
    file_list: str = Field("TSB-AD-M-Tuning.csv", description="name inside the official File_List folder, or a path")
    tag: str
    out_root: Optional[str] = None
    claim: bool = True
    save_scores: bool = True
    save_parts: bool = Field(True, description="detectors with score parts (AxonAD: mse / jepa) save them as .npz")
