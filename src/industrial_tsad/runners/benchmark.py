"""Run one detector over a TSB-AD file list.

Resumable and shareable: every worker appends its rows to its own per_file_w<pid>.csv, skips series already done by
any worker, and claims a series by creating claims/<file>.claim (O_EXCL) when cfg.claim is on, so any number of worker
processes (e.g. two per 10 GB GPU) can share one run. Point scores are saved as scores/<file>.npy, score parts
(AxonAD mse / jepa) as parts/<file>.npz. Seeds: config.seed.base (+ index of the series in its list if per_series)."""
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..configs import BenchmarkRunConfig
from ..core.models import create_detector
from ..datasets import tsbad
from ..evaluators.metrics import threshold_free_metrics
from ..utils.paths import get_paths


def run_dir(cfg: BenchmarkRunConfig) -> Path:
    root = Path(cfg.out_root) if cfg.out_root else get_paths().results / f"tsbad_{cfg.benchmark}"
    return root / cfg.tag


def _done(out: Path):
    parts = sorted(out.glob("per_file*.csv"))
    return set(pd.concat([pd.read_csv(q) for q in parts])["file"]) if parts else set()


def run_benchmark(cfg: BenchmarkRunConfig, device=None, log=print) -> pd.DataFrame:
    out = run_dir(cfg)
    (out / "scores").mkdir(parents=True, exist_ok=True)
    (out / "claims").mkdir(exist_ok=True)
    (out / "config.json").write_text(cfg.model_dump_json(indent=1), encoding="utf-8")
    det = create_detector(cfg.detector, device)
    files = tsbad.file_list(cfg.file_list)
    csv_path = out / f"per_file_w{os.getpid()}.csv"
    seed_cfg = cfg.detector.seed
    t0 = time.time()
    for n, f in enumerate(files):
        if f in _done(out):
            continue
        if cfg.claim:
            try:
                os.close(os.open(out / "claims" / (f[:-4] + ".claim"), os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            except FileExistsError:
                continue
        t1 = time.time()
        s = tsbad.load(f, cfg.benchmark)
        seed = seed_cfg.base + (n if seed_cfg.per_series else 0)
        score = det.fit_score(s.x, s.train_end, seed=seed, dataset=s.dataset)
        if cfg.save_scores:
            np.save(out / "scores" / f"{f[:-4]}.npy", np.asarray(score, np.float32))
        if cfg.save_parts and det.parts:
            (out / "parts").mkdir(exist_ok=True)
            np.savez_compressed(out / "parts" / f"{f[:-4]}.npz", **{k: np.asarray(v, np.float32) for k, v in det.parts.items()})
        m = threshold_free_metrics(score[:len(s.label)], s.label[:len(score)], s.sliding_window)
        row = {"file": f, "dataset": s.dataset, "channels": s.x.shape[1], "seed": seed,
               **{k: m[k] for k in ("VUS-PR", "AUC-PR", "AUC-ROC", "VUS-ROC")}, "sec": round(time.time() - t1, 1)}
        pd.DataFrame([row]).to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
        if cfg.claim:
            (out / "claims" / (f[:-4] + ".claim")).unlink(missing_ok=True)
        log(f"[{n + 1}/{len(files)}] {f}: VUS-PR {row['VUS-PR']:.3f} AUC-PR {row['AUC-PR']:.3f} ({row['sec']}s, total {time.time() - t0:.0f}s)")
        torch.cuda.empty_cache()
    return summarize(out)


def load_run(out: Path) -> pd.DataFrame:
    parts = sorted(Path(out).glob("per_file*.csv"))
    if not parts:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(q) for q in parts]).drop_duplicates("file").set_index("file")


def summarize(out: Path) -> pd.DataFrame:
    d = load_run(out)
    if d.empty:
        return d
    D = d.groupby("dataset")["VUS-PR"].mean()
    summ = {"files": len(d), "VUS-PR": float(d["VUS-PR"].mean()), "AUC-PR": float(d["AUC-PR"].mean()),
            "dataset_mean_VUS-PR": float(D.mean()), "by_dataset": D.round(4).to_dict()}
    (Path(out) / "summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    return d
