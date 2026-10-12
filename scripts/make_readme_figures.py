"""README figures as light and dark SVGs (GitHub picks one through <picture> and prefers-color-scheme).

    python scripts/make_readme_figures.py    # -> docs/assets/{results,gate}-{light,dark}.svg

results: VUS-PR on the TSB-AD eval lists, the numbers of docs/results_tsbad.md.
gate:    the regime gate of the best TSB-AD-M system (industrial_tsad.core.fusion.RegimeGate).
The architecture and protocol diagrams are Archify sources in docs/diagrams/.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets"
FONT = "-apple-system, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif"
THEMES = {
    "light": {"text": "#1f2328", "muted": "#59636e", "grid": "#d1d9e0", "ours": "#0f8b7d", "pub": "#9aa6b2",
              "stroke": "#8c959f", "fill": "#f6f8fa", "gate_fill": "#fff8c5", "gate_stroke": "#9a6700"},
    "dark": {"text": "#e6edf3", "muted": "#9198a1", "grid": "#3d444d", "ours": "#3fc9b3", "pub": "#656c76",
             "stroke": "#6e7681", "fill": "#161b22", "gate_fill": "#2d2400", "gate_stroke": "#d29922"},
}

# ---------------------------------------------------------------- results chart: (label, VUS-PR, ours?)
M = [
    ("PaAno final + regime gate", 0.589, True),
    ("PaAno final", 0.544, True),
    ("AxonAD", 0.495, False),
    ("MSCAD (re-implemented)", 0.465, True),
    ("StreamVAE", 0.435, False),
    ("PaAno official (port)", 0.422, True),
    ("xLSTMAD", 0.392, False),
    ("CHARM", 0.391, False),
    ("TSPulse-FT", 0.387, False),
    ("KAN-AD (port)", 0.294, True),
    ("TimesNet", 0.185, False),
]
U = [
    ("MSCAD (re-implemented)", 0.572, True),
    ("PaAno final", 0.552, True),
    ("TSPulse-FT", 0.552, False),
    ("Time-RCD", 0.522, False),
    ("CHARM", 0.501, False),
]
ROW = 0.34  # inches per bar row


def _panel(ax, rows, title, c):
    labels = [r[0] for r in rows][::-1]
    vals = [r[1] for r in rows][::-1]
    cols = [c["ours"] if r[2] else c["pub"] for r in rows][::-1]
    y = range(len(rows))
    ax.barh(y, vals, height=0.62, color=cols, zorder=2)
    for yi, v in zip(y, vals):
        ax.text(v + 0.008, yi, f"{v:.3f}", va="center", ha="left", fontsize=9.5, color=c["text"])
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=10, color=c["text"])
    ax.set_xlim(0, 0.68)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xticks([0, 0.2, 0.4, 0.6])
    ax.tick_params(axis="x", colors=c["muted"], labelsize=9, length=0)
    ax.tick_params(axis="y", length=0, pad=6)
    ax.grid(axis="x", color=c["grid"], linewidth=0.8, zorder=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, loc="left", fontsize=11.5, color=c["text"], fontweight="bold", pad=10)
    ax.set_xlabel("VUS-PR", fontsize=9.5, color=c["muted"])


def results_chart(theme):
    c = THEMES[theme]
    n = len(M)
    fig_h = n * ROW + 1.25
    fig = plt.figure(figsize=(10.4, fig_h))
    fig.patch.set_alpha(0)
    bottom, top = 0.62 / fig_h, 1 - 0.45 / fig_h
    h_row = (top - bottom) / n
    ax_m = fig.add_axes([0.235, bottom, 0.30, top - bottom])
    ax_u = fig.add_axes([0.735, top - h_row * len(U), 0.23, h_row * len(U)])
    for ax in (ax_m, ax_u):
        ax.set_facecolor("none")
    _panel(ax_m, M, "TSB-AD-M · 180 multivariate series", c)
    _panel(ax_u, U, "TSB-AD-U · 350 univariate series", c)
    # legend and note in the free space under the univariate panel
    x0, y0 = 0.58, top - h_row * len(U) - 0.17
    for i, (name, col) in enumerate([("run with this repository", c["ours"]),
                                     ("published per-series results", c["pub"])]):
        yy = y0 - i * 0.07
        fig.patches.append(plt.Rectangle((x0, yy - 0.012), 0.016, 0.03, transform=fig.transFigure, color=col))
        fig.text(x0 + 0.025, yy, name, fontsize=9.5, color=c["text"], va="center")
    fig.text(x0, y0 - 0.17, "Official eval lists, one look each; the gate's\n± is 0.001 over three AxonAD seeds.",
             fontsize=8.8, color=c["muted"], va="top", linespacing=1.4)
    out = OUT / f"results-{theme}.svg"
    fig.savefig(out, format="svg", transparent=True, metadata={"Date": None})
    plt.close(fig)
    svg = out.read_text(encoding="utf-8")
    out.write_text(svg.replace("'DejaVu Sans'", FONT), encoding="utf-8")
    return out


# ---------------------------------------------------------------- regime gate (hand-drawn SVG)
def _box(x, y, w, h, title, sub, c, stroke=None, fill=None):
    cx = x + w / 2
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" fill="{fill or c["fill"]}" '
            f'stroke="{stroke or c["stroke"]}" stroke-width="1.2"/>'
            f'<text x="{cx}" y="{y + 19}" text-anchor="middle" font-size="13" font-weight="600" fill="{c["text"]}">{title}</text>'
            f'<text x="{cx}" y="{y + 35}" text-anchor="middle" font-size="11" fill="{c["muted"]}">{sub}</text>')


def gate_figure(theme):
    c = THEMES[theme]
    claim = ("Regime gate: the PaAno score always reaches the sum; AxonAD's JEPA term reaches it only when the median "
             "of its robust z over the test part exceeds tau = 5, then with weight w = 1.")
    line = f'stroke="{c["stroke"]}" stroke-width="1.5" fill="none"'
    acc = f'stroke="{c["ours"]}" stroke-width="2.2" fill="none"'
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 880 250" width="880" height="250" role="img" '
        f'aria-label="{claim}" font-family="{FONT.replace(chr(39), "")}">',
        f'<title>{claim}</title>',
        '<defs>'
        f'<marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
        f'<path d="M0,0 L10,5 L0,10 z" fill="{c["stroke"]}"/></marker>'
        f'<marker id="arr-acc" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">'
        f'<path d="M0,0 L10,5 L0,10 z" fill="{c["ours"]}"/></marker>'
        '</defs>',
        # nodes
        _box(20, 38, 120, 44, "series x", "x[:train_end] normal", c),
        _box(210, 38, 210, 44, "PaAno final", "robust-z sum of its terms", c),
        _box(210, 153, 210, 44, "AxonAD", "JEPA tail distance, robust z", c),
        _box(720, 38, 140, 44, "anomaly score", "one per point", c),
        f'<circle cx="640" cy="60" r="17" fill="{c["fill"]}" stroke="{c["stroke"]}" stroke-width="1.2"/>'
        f'<text x="640" y="65.5" text-anchor="middle" font-size="18" fill="{c["text"]}">+</text>',
        f'<polygon points="530,135 600,175 530,215 460,175" fill="{c["gate_fill"]}" stroke="{c["gate_stroke"]}" stroke-width="1.4"/>'
        f'<text x="530" y="171" text-anchor="middle" font-size="11.5" fill="{c["text"]}">median z on</text>'
        f'<text x="530" y="186" text-anchor="middle" font-size="11.5" fill="{c["text"]}">test part &gt; τ ?</text>',
        # edges
        f'<path d="M140,60 H207" {line} marker-end="url(#arr)"/>',
        f'<path d="M80,82 V175 H207" {line} marker-end="url(#arr)"/>',
        f'<path d="M420,60 H620" {line} marker-end="url(#arr)"/>',
        f'<text x="520" y="52" text-anchor="middle" font-size="11" fill="{c["muted"]}">always</text>',
        f'<path d="M657,60 H717" {line} marker-end="url(#arr)"/>',
        f'<path d="M420,175 H457" {line} marker-end="url(#arr)"/>',
        f'<path d="M600,175 H640 V80" {acc} marker-end="url(#arr-acc)"/>',
        f'<text x="652" y="122" font-size="11.5" font-weight="600" fill="{c["ours"]}">yes: + w · rz(JEPA)</text>',
        f'<text x="652" y="138" font-size="11" fill="{c["muted"]}">τ = 5, w = 1</text>',
        f'<path d="M530,215 V236 H560" {line} stroke-dasharray="4 3"/>',
        f'<text x="568" y="240" font-size="11" fill="{c["muted"]}">no: the term is dropped</text>',
        '</svg>',
    ]
    out = OUT / f"gate-{theme}.svg"
    out.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["svg.hashsalt"] = "industrial-tsad"
    for t in THEMES:
        print(results_chart(t))
        print(gate_figure(t))
