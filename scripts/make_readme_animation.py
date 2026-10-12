"""README animations from saved runs, as light and dark GIFs. Scores are computed offline per series; the animations
only reveal them.

  # the regime gate on one TSB-AD-M series: whole series -> zoom -> PaAno score -> JEPA term + gate -> gated score
  python scripts/make_readme_animation.py gate \
      --scores results/tsbad_M/paano_final_eva/scores --parts results/tsbad_M/axonad_eva/parts

  # several detectors on one series, one after another (a folder holds <series>.npy, or <series>.npz whose
  # robust-z parts are summed, e.g. AxonAD's mse + jepa)
  python scripts/make_readme_animation.py tour --series 138_CATSv2_id_1_Sensor_tr_16568_1st_16668.csv \
      --channels bfo2,cso1,bso1 --run "PaAno final=results/tsbad_M/paano_final_eva/scores" \
      --run "MSCAD=results/tsbad_M/mscad_eva/scores" --run "AxonAD=results/tsbad_M/axonad_eva/parts"
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from industrial_tsad.core.fusion import RegimeGate  # noqa: E402
from industrial_tsad.datasets import load  # noqa: E402
from industrial_tsad.evaluators.metrics import threshold_free_metrics  # noqa: E402
from industrial_tsad.utils.paths import get_paths  # noqa: E402

THEMES = {
    "dark": {"bg": "#0d1117", "text": "#e6edf3", "muted": "#9198a1", "grid": "#30363d",
             "sig": ["#79c0ff", "#ffa657", "#7ee787"], "runs": ["#3fc9b3", "#ffa657", "#d2a8ff", "#79c0ff"],
             "paano": "#8b949e", "jepa": "#d2a8ff", "gated": "#3fc9b3", "gt": "#f85149", "gt_a": 0.28,
             "train": "#ffffff", "train_a": 0.06, "box": "#e3b341"},
    "light": {"bg": "#ffffff", "text": "#1f2328", "muted": "#59636e", "grid": "#d8dee4",
              "sig": ["#0969da", "#bc4c00", "#1a7f37"], "runs": ["#0f8b7d", "#bc4c00", "#8250df", "#0969da"],
              "paano": "#6e7781", "jepa": "#8250df", "gated": "#0f8b7d", "gt": "#cf222e", "gt_a": 0.18,
              "train": "#000000", "train_a": 0.05, "box": "#9a6700"},
}
COLS = 700      # pixel columns of the min/max envelope
GROW = 6        # frames to draw a score from left to right


def envelope(v, a, b, cols=COLS):
    """min/max per pixel column of v[a:b] -> x, y drawn as one vertical stroke per column"""
    a, b = int(max(0, a)), int(min(len(v), b))
    cols = max(2, min(cols, b - a))
    edges = np.linspace(a, b, cols + 1).astype(int)
    lo = np.minimum.reduceat(v[a:b], edges[:-1] - a)
    hi = np.maximum.reduceat(v[a:b], edges[:-1] - a)
    return np.repeat((edges[:-1] + edges[1:]) / 2, 2), np.column_stack([lo, hi]).ravel()


def spans(label):
    d = np.diff(np.r_[0, label.astype(int), 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def channels(s, series, names):
    cols = list(pd.read_csv(get_paths().tsbad_data / "TSB-AD-M" / series, nrows=1).columns[:-1])
    out = []
    for n in names:
        v = s.x[:, cols.index(n)]
        z = (v - v[:s.train_end].mean()) / (v[:s.train_end].std() + 1e-9)
        out.append(np.clip(z, -3.5, 3.5))           # one huge spike must not flatten the rest of the channel
    return out


def score_of(folder, stem):
    folder = Path(folder)
    if (folder / f"{stem}.npy").exists():
        return np.load(folder / f"{stem}.npy").astype(np.float64)
    z = np.load(folder / f"{stem}.npz")
    return np.sum([z[k].astype(np.float64) for k in z.files], axis=0)


class Stage:
    """figure with a title, a sensor panel and score panels; draw(k) is filled in by the scene functions"""

    def __init__(self, c, title, subtitle, n_panels, s, gt):
        self.c, self.s, self.gt = c, s, gt
        self.fig = plt.figure(figsize=(8.8, 5.6), dpi=100, facecolor=c["bg"])
        gs = self.fig.add_gridspec(n_panels + 1, 1, height_ratios=[1.5] + [1] * n_panels, hspace=0.55, left=0.03,
                                   right=0.97, top=0.86, bottom=0.06)
        self.axs = [self.fig.add_subplot(gs[i]) for i in range(n_panels + 1)]
        self.fig.text(0.03, 0.955, title, fontsize=13, fontweight="bold", color=c["text"])
        self.fig.text(0.03, 0.915, subtitle, fontsize=9.5, color=c["muted"])
        self.cap = [self.fig.text(0.03, 0, "", fontsize=9.5, color=c["text"]) for _ in self.axs]

    def frame(self, lo, hi, shown):
        c, last = self.c, max(i for i, on in enumerate(shown) if on)
        for i, ax in enumerate(self.axs):
            ax.clear()
            self.cap[i].set_position((0.03, ax.get_position().y1 + 0.012))
            if not shown[i]:                                   # placeholder: what comes next
                ax.axis("off")
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.plot([0, 1], [0.12, 0.12], color=c["grid"], lw=1.0, ls=(0, (4, 4)))
                continue
            ax.axis("on")
            ax.set_facecolor(c["bg"])
            ax.set_xlim(lo, hi)
            ax.set_yticks([])
            ax.tick_params(axis="x", colors=c["muted"], labelsize=8, length=0)
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.grid(axis="x", color=c["grid"], linewidth=0.6)
            ax.axvspan(0, self.s.train_end, color=c["train"], alpha=c["train_a"], lw=0)
            for g0, g1 in self.gt:
                ax.axvspan(g0, g1, color=c["gt"], alpha=c["gt_a"], lw=0)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
            if i != last:
                ax.set_xticklabels([])

    def sensors(self, sig, names, lo, hi):
        for j, v in enumerate(sig):
            x, y = envelope(v + 4.5 * j, lo, hi)
            self.axs[0].plot(x, y, color=self.c["sig"][j % 3], lw=0.7)
        if lo < self.s.train_end:
            self.axs[0].text(lo + 0.004 * (hi - lo), 4.5 * len(sig) - 2.2, "train part (normal)", fontsize=8,
                             color=self.c["muted"], va="top")
        self.cap[0].set_text(f"Sensors ({', '.join(names)})")

    def score(self, i, v, lo, hi, color, grow, text):
        x, y = envelope(v, lo, lo + grow * (hi - lo), int(COLS * grow))
        self.axs[i].plot(x, y, color=color, lw=0.7)
        vv = v[int(lo):int(hi)]
        self.axs[i].set_ylim(np.percentile(vv, 0.5) - 0.05 * np.ptp(vv), vv.max() * 1.08)
        self.cap[i].set_text(text)

    def placeholder(self, i, text):
        self.cap[i].set_text(text)
        self.cap[i].set_color(self.c["muted"])

    def save(self, draw, n_frames, out, fps):
        anim = FuncAnimation(self.fig, draw, frames=n_frames, blit=False)
        anim.save(out, writer=PillowWriter(fps=fps), savefig_kwargs={"facecolor": self.c["bg"]})
        plt.close(self.fig)
        print(out, f"{out.stat().st_size / 1e6:.2f} MB")


def gate_scene(a):
    s = load(a.series, "M")
    stem = a.series[:-4]
    base = score_of(a.scores, stem)
    jepa = np.load(Path(a.parts) / f"{stem}.npz")["jepa"].astype(np.float64)
    gated, gate_on, stat = RegimeGate()(base, jepa, s.train_end)
    win = s.sliding_window
    vus_base = threshold_free_metrics(base, s.label, win)["VUS-PR"]
    vus_gated = threshold_free_metrics(gated, s.label, win)["VUS-PR"]
    print(f"gate {gate_on} statistic {stat:.1f}; VUS-PR base {vus_base:.3f} gated {vus_gated:.3f}")
    names = a.channels.split(",")
    sig = channels(s, a.series, names)
    T, gt = len(s.label), spans(s.label)
    z0, z1 = gt[0][0] - 10000, gt[-1][1] + 8000          # zoom window around the anomalies
    n_over, n_zoom, n_base, n_jepa, n_gate, n_hold = 14, 14, 20, 22, 18, 28
    st = np.cumsum([0, n_over, n_zoom, n_base, n_jepa, n_gate])
    for theme, c in THEMES.items():
        g = Stage(c, "Regime gate on a TSB-AD-M series",
                  f"{a.label} · {s.x.shape[1]} sensors · {T:,} steps · red = labelled anomaly", 3, s, gt)
        panels = [(1, base, c["paano"], "PaAno final score",
                   f"PaAno final score  ·  VUS-PR {vus_base:.3f}: spikes everywhere, the fault does not stand out"),
                  (2, jepa, c["jepa"], "AxonAD JEPA term",
                   f"AxonAD JEPA term  ·  median z over the test part = {stat:.1f} > τ = 5  →  gate opens"),
                  (3, gated, c["gated"], "PaAno + w · rz(JEPA)", f"PaAno + w · rz(JEPA)  ·  VUS-PR {vus_gated:.3f}")]

        def draw(k, g=g, c=c, panels=panels):
            if k < st[1]:
                lo, hi = 0, T
            elif k < st[2]:
                t = (k - st[1]) / (n_zoom - 1)
                t = t * t * (3 - 2 * t)
                lo, hi = t * z0, (1 - t) * T + t * z1
            else:
                lo, hi = z0, z1
            shown = [True, k >= st[2], k >= st[3], k >= st[4]]
            g.frame(lo, hi, shown)
            g.sensors(sig, names, lo, hi)
            if k < st[2]:
                g.axs[0].add_patch(plt.Rectangle((z0, -3), z1 - z0, 4.5 * len(sig) + 2.5, fill=False, ec=c["box"],
                                                 lw=1.4))
            for i, v, col, short, text in panels:
                if shown[i]:
                    g.score(i, v, lo, hi, col, min(1.0, (k - st[i + 1] + 1) / GROW), text)
                    g.cap[i].set_color(col if i == 3 else c["text"])
                else:
                    g.placeholder(i, short)
            return []

        g.save(draw, st[-1] + n_hold, Path(a.out) / f"gate-demo-{theme}.gif", a.fps)


def tour_scene(a):
    s = load(a.series, "M")
    stem = a.series[:-4]
    runs = []
    for spec in a.run:
        name, folder = spec.split("=", 1)
        v = score_of(folder, stem)
        runs.append((name, v, threshold_free_metrics(v, s.label, s.sliding_window)["VUS-PR"]))
        print(f"{name}: VUS-PR {runs[-1][2]:.3f}")
    names = a.channels.split(",")
    sig = channels(s, a.series, names)
    T, gt = len(s.label), spans(s.label)
    n_intro, n_each, n_hold = 12, 18, 30
    st = [n_intro + i * n_each for i in range(len(runs) + 1)]
    best = max(r[2] for r in runs)
    for theme, c in THEMES.items():
        g = Stage(c, "Same series, same interface, different detectors",
                  f"{a.label} · {s.x.shape[1]} sensors · {T:,} steps · red = labelled anomaly", len(runs), s, gt)

        def draw(k, g=g, c=c):
            shown = [True] + [k >= st[i] for i in range(len(runs))]
            g.frame(0, T, shown)
            g.sensors(sig, names, 0, T)
            for i, (name, v, vus) in enumerate(runs):
                col = c["runs"][i % len(c["runs"])]
                if shown[i + 1]:
                    star = "  ← best" if vus == best and k >= st[-1] else ""
                    g.score(i + 1, v, 0, T, col, min(1.0, (k - st[i] + 1) / GROW), f"{name}  ·  VUS-PR {vus:.3f}{star}")
                    g.cap[i + 1].set_color(col)
                else:
                    g.placeholder(i + 1, name)
            return []

        g.save(draw, st[-1] + n_hold, Path(a.out) / f"tour-{theme}.gif", a.fps)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gate")
    g.add_argument("--series", default="034_GHL_id_3_Sensor_tr_50000_1st_122001.csv")
    g.add_argument("--label", default="GHL gasoil heating loop")
    g.add_argument("--channels", default="HT_temperature.T,RT_temperature.T")
    g.add_argument("--scores", required=True, help="folder with <series>.npy of the base run (PaAno final)")
    g.add_argument("--parts", required=True, help="folder with <series>.npz of the AxonAD run (key 'jepa')")
    t = sub.add_parser("tour")
    t.add_argument("--series", default="138_CATSv2_id_1_Sensor_tr_16568_1st_16668.csv")
    t.add_argument("--label", default="CATS controlled-anomalies system")
    t.add_argument("--channels", default="bfo2,cso1,bso1")
    t.add_argument("--run", action="append", required=True, help="NAME=FOLDER, in display order")
    for p in (g, t):
        p.add_argument("--out", default=str(ROOT / "docs" / "assets"))
        p.add_argument("--fps", type=int, default=10)
    a = ap.parse_args()
    (gate_scene if a.cmd == "gate" else tour_scene)(a)


if __name__ == "__main__":
    main()
