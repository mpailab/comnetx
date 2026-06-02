"""Generate the ICDM topology-ablation quality/speedup figure."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
MEASUREMENTS = ROOT / "results" / "icdm-2026-0" / "measurements" / "real_graph_measurements.json"
OUT = ROOT / "article" / "topology_ablation_pareto.pdf"
PDF_METADATA = {
    "Creator": "ComNetX ICDM figure scripts",
    "Producer": "Matplotlib pdf backend",
    "CreationDate": datetime(2026, 1, 1, tzinfo=timezone.utc),
    "ModDate": datetime(2026, 1, 1, tzinfo=timezone.utc),
}

DATASETS = ("dyn_cora", "dyn_pubmed", "arxivmath")
LEVELS = (1, 2, 3, 4)
RADII = (0, 1, 2)

RADIUS_STYLES = {
    0: ("#1f77b4", "o"),
    1: ("#d95f02", "s"),
    2: ("#2ca02c", "^"),
}

X_AXIS = {
    "dyn_cora": {
        "scale": "linear",
        "xlim": (0, 26),
        "xticks": [1, 5, 10, 15, 20, 25],
        "xticklabels": ["1x", "5x", "10x", "15x", "20x", "25x"],
    },
    "dyn_pubmed": {
        "scale": "log",
        "xlim": (0.85, 220),
        "xticks": [1, 10, 100],
        "xticklabels": ["1x", "10x", "100x"],
    },
    "arxivmath": {
        "scale": "log",
        "xlim": (0.85, 720),
        "xticks": [1, 10, 100],
        "xticklabels": ["1x", "10x", "100x"],
    },
}


@dataclass(frozen=True)
class Point:
    dataset: str
    level: int | None
    radius: int | None
    modularity: float
    time: float
    speedup: float
    label: str
    is_full: bool = False


def bundle_order(record: dict) -> int:
    match = re.search(r"(\d+)$", record.get("bundle_record_id", ""))
    return int(match.group(1)) if match else 10**9


def first_matching(records: list[dict], **criteria: object) -> dict:
    matches = [
        record
        for record in records
        if all(str(record.get(key)) == str(value) for key, value in criteria.items())
    ]
    if not matches:
        details = ", ".join(f"{key}={value}" for key, value in sorted(criteria.items()))
        raise RuntimeError(f"missing topology-ablation measurement: {details}")
    return min(matches, key=bundle_order)


def load_points() -> dict[str, list[Point]]:
    records = json.loads(MEASUREMENTS.read_text())["records"]
    by_dataset: dict[str, list[Point]] = {}

    for dataset in DATASETS:
        full = first_matching(
            records,
            method="leidenalg",
            mode="naive",
            base_dataset=dataset,
            batch_strategy="999:10",
            updates=10,
        )
        full_time = float(full["total_time"])
        points = [
            Point(
                dataset=dataset,
                level=None,
                radius=None,
                modularity=float(full["final_modularity"]),
                time=full_time,
                speedup=1.0,
                label="Full",
                is_full=True,
            )
        ]

        for level in LEVELS:
            for radius in RADII:
                record = first_matching(
                    records,
                    method="leidenalg",
                    mode="smart",
                    base_dataset=dataset,
                    batch_strategy="999:10",
                    updates=10,
                    subcoms_depth=level,
                    radius=radius,
                )
                local_time = float(record["total_time"])
                points.append(
                    Point(
                        dataset=dataset,
                        level=level,
                        radius=radius,
                        modularity=float(record["final_modularity"]),
                        time=local_time,
                        speedup=full_time / local_time,
                        label=str(level),
                    )
                )

        by_dataset[dataset] = points

    return by_dataset


def pareto_front(points: list[Point]) -> list[Point]:
    front: list[Point] = []
    for point in points:
        dominated = False
        for other in points:
            at_least_as_good = (
                other.speedup >= point.speedup
                and other.modularity >= point.modularity
            )
            strictly_better = (
                other.speedup > point.speedup
                or other.modularity > point.modularity
            )
            if at_least_as_good and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(point)
    return sorted(front, key=lambda point: point.speedup)


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
            "legend.fontsize": 5.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    by_dataset = load_points()
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 1.75), constrained_layout=True)

    for ax, dataset in zip(axes, DATASETS, strict=True):
        points = by_dataset[dataset]
        front = pareto_front(points)
        ax.plot(
            [point.speedup for point in front],
            [point.modularity for point in front],
            color="#222222",
            lw=0.75,
            zorder=1,
        )

        full = next(point for point in points if point.is_full)
        ax.scatter(
            [full.speedup],
            [full.modularity],
            s=42,
            marker="*",
            color="#222222",
            edgecolor="white",
            linewidth=0.35,
            zorder=4,
        )
        ax.text(
            full.speedup + 0.75 if dataset == "dyn_cora" else full.speedup * 1.08,
            full.modularity,
            "Full",
            ha="left",
            va="center",
            fontsize=5.5,
            color="#222222",
            zorder=5,
        )

        for point in [point for point in points if not point.is_full]:
            color, marker = RADIUS_STYLES[point.radius or 0]
            is_default = point.level == 3 and point.radius == 1
            ax.scatter(
                [point.speedup],
                [point.modularity],
                s=36 if not is_default else 48,
                marker=marker,
                color=color,
                edgecolor="#111111" if is_default else "white",
                linewidth=0.65 if is_default else 0.35,
                alpha=0.96,
                zorder=3 if not is_default else 5,
            )
            ax.text(
                point.speedup,
                point.modularity,
                point.label,
                ha="center",
                va="center",
                color="white",
                fontsize=5,
                fontweight="bold",
                zorder=6,
            )

        modularities = [point.modularity for point in points]
        margin = (max(modularities) - min(modularities)) * 0.12
        ax.set_ylim(min(modularities) - margin, max(modularities) + margin)
        axis = X_AXIS[dataset]
        ax.set_xscale(axis["scale"])
        ax.set_xlim(*axis["xlim"])
        ax.set_xticks(axis["xticks"])
        ax.set_xticklabels(axis["xticklabels"])
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
        ax.set_title(dataset)
        ax.grid(axis="both", color="#d0d0d0", lw=0.4, ls=":", zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel(r"Final modularity $Q$")
    for ax in axes:
        ax.set_xlabel("Base/Local time")

    handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="none",
            markerfacecolor=color,
            markeredgecolor="white",
            markersize=4.3,
            label=rf"$r={radius}$",
        )
        for radius, (color, marker) in RADIUS_STYLES.items()
    ]
    handles.append(Line2D([0], [0], color="#222222", lw=0.75, label="Pareto"))
    axes[-1].legend(
        handles=handles,
        loc="lower left",
        frameon=False,
        handlelength=1.0,
        borderpad=0.1,
        labelspacing=0.2,
        columnspacing=0.6,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.015, metadata=PDF_METADATA)


if __name__ == "__main__":
    main()
