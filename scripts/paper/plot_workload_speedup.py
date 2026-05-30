"""Generate the ICDM workload-to-speedup mechanism figure."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "article" / "workload_speedup.pdf"
PDF_METADATA = {
    "Creator": "ComNetX ICDM figure scripts",
    "Producer": "Matplotlib pdf backend",
    "CreationDate": datetime(2026, 1, 1, tzinfo=timezone.utc),
    "ModDate": datetime(2026, 1, 1, tzinfo=timezone.utc),
}

# Audited workload-profile summary used in the main-paper figure.
# edge_pct is |Ebar|/|E| (%), speedup is Base/Local, and uv_pct is |U|/|V| (%).
ROWS = [
    ("Leiden", "dyn_cora", 0.40, 9.6, 11.5),
    ("Leiden", "dyn_pubmed", 0.64, 22.0, 19.9),
    ("Leiden", "arxivmath", 0.79, 43.9, 43.7),
    ("DF-Leiden", "dyn_cora", 0.39, 0.78, 11.4),
    ("DF-Leiden", "dyn_pubmed", 0.64, 0.95, 20.8),
    ("DF-Leiden", "arxivmath", 0.77, 8.4, 48.2),
    ("S$^2$CAG", "dyn_cora", 0.35, 3.0, 9.7),
    ("S$^2$CAG", "dyn_pubmed", 1.04, 3.4, 26.7),
    ("S$^2$CAG", "arxivmath", 1.08, 4.3, 62.8),
]

COLORS = {
    "Leiden": "#1f77b4",
    "DF-Leiden": "#2ca02c",
    "S$^2$CAG": "#9467bd",
}
MARKERS = {
    "dyn_cora": "o",
    "dyn_pubmed": "s",
    "arxivmath": "^",
}
LABELS = {
    "dyn_cora": "C",
    "dyn_pubmed": "P",
    "arxivmath": "A",
}


def bubble_size(uv_pct: float) -> float:
    return 18 + 2.2 * uv_pct


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(3.35, 2.25), constrained_layout=True)
    ax.axhline(1.0, color="#777777", lw=0.7, ls="--", zorder=0)

    for backend in COLORS:
        points = [row for row in ROWS if row[0] == backend]
        points.sort(key=lambda row: row[2])
        ax.plot(
            [row[2] for row in points],
            [row[3] for row in points],
            color=COLORS[backend],
            lw=0.9,
            alpha=0.7,
            zorder=1,
        )
        for _, dataset, edge_pct, speedup, uv_pct in points:
            ax.scatter(
                edge_pct,
                speedup,
                s=bubble_size(uv_pct),
                marker=MARKERS[dataset],
                facecolor=COLORS[backend],
                edgecolor="white",
                linewidth=0.45,
                alpha=0.95,
                zorder=3,
            )
            ax.text(
                edge_pct,
                speedup,
                LABELS[dataset],
                ha="center",
                va="center",
                color="white",
                fontsize=5.5,
                fontweight="bold",
                zorder=4,
            )

    ax.set_yscale("log")
    ax.set_xlim(0.28, 1.16)
    ax.set_ylim(0.55, 58)
    ax.set_xlabel(r"Contracted edge workload, $|\bar E|/|E|$ (%)")
    ax.set_ylabel("Base/Local time")
    ax.set_yticks([0.75, 1, 3, 10, 30, 50])
    ax.set_yticklabels(["0.75x", "1x", "3x", "10x", "30x", "50x"])
    ax.grid(axis="y", color="#d0d0d0", lw=0.45, ls=":", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    backend_handles = [
        Line2D([0], [0], color=COLORS[name], marker="o", lw=0.9, markersize=4, label=name)
        for name in COLORS
    ]
    ax.legend(
        handles=backend_handles,
        loc="upper left",
        frameon=False,
        handlelength=1.6,
        borderpad=0.1,
        labelspacing=0.25,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.015, metadata=PDF_METADATA)


if __name__ == "__main__":
    main()
