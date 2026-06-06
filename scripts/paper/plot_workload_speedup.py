"""Generate the ICDM workload-to-speedup mechanism figure from icdm-2026-1."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "results" / "icdm-2026-1" / "measurements" / "experiment_measurements.json"
WORKLOAD_PROFILES = ROOT / "results" / "icdm-2026-1" / "measurements" / "workload_profiles.json"
OUT = ROOT / "article" / "workload_speedup.pdf"
PDF_METADATA = {
    "Creator": "ComNetX ICDM figure scripts",
    "Producer": "Matplotlib pdf backend",
    "CreationDate": datetime(2026, 1, 1, tzinfo=timezone.utc),
    "ModDate": datetime(2026, 1, 1, tzinfo=timezone.utc),
}

DATASETS = ("dyn_cora", "dyn_pubmed", "arxivmath")
DATASET_NODES = {
    "dyn_cora": 2708,
    "dyn_pubmed": 19717,
    "arxivmath": 270013,
}
DATASET_DIRECTED_EDGES = {
    "dyn_cora": 2 * 5278,
    "dyn_pubmed": 2 * 44338,
    "arxivmath": 2 * 799745,
}
BASELINE_ALGORITHMS = {
    "leidenalg": "leidenalg-naive",
    "dfleiden": "dfleiden-dynamic",
    "s2cag": "s2cag-i:10-naive-feat:dataset",
}
LOCAL_ALGORITHMS = {
    "leidenalg": "leidenalg-L:3-r:1-gpu",
    "dfleiden": "dfleiden-L:3-r:1-gpu",
    "s2cag": "s2cag-i:10-L:3-r:1-agg:norm-gpu-feat:dataset",
}
BACKEND_LABELS = {
    "leidenalg": "Leiden",
    "dfleiden": "DF-Leiden",
    "s2cag": "S$^2$CAG",
}
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
MIN_PLOTTED_SPEEDUP = 0.01


@dataclass(frozen=True)
class WorkloadPoint:
    backend: str
    dataset: str
    edge_pct: float
    speedup: float
    uv_pct: float


def load_records(path: Path) -> list[dict]:
    return json.loads(path.read_text())["records"]


def mean_time(records: list[dict], algorithm: str, dataset: str) -> float:
    times = [
        float(record["total_time"])
        for record in records
        if record.get("measurement_type") == "experiment"
        and record.get("algorithm") == algorithm
        and record.get("base_dataset") == dataset
        and record.get("batch_strategy") == "999:10"
        and record.get("updates") == 10
    ]
    if not times:
        raise RuntimeError(f"missing paired experiment time: {algorithm}, {dataset}")
    return mean(times)


def profile_record(records: list[dict], method: str, dataset: str) -> dict:
    matches = [
        record
        for record in records
        if record.get("measurement_type") == "workload_profile"
        and record.get("measurement_family") == "article_completion_workload_profiles"
        and record.get("method") == method
        and record.get("base_dataset") == dataset
        and record.get("batch_strategy") == "999:10"
        and record.get("variant") == "full"
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one workload profile for {method}, {dataset}; got {len(matches)}")
    return matches[0]


def final_level_mean(rows: list[dict], key: str) -> float:
    values = []
    for row in rows:
        sequence = row.get(key)
        if isinstance(sequence, list) and sequence:
            values.append(float(sequence[-1]))
    if not values:
        raise RuntimeError(f"missing profile key: {key}")
    return mean(values)


def load_points() -> list[WorkloadPoint]:
    experiments = load_records(EXPERIMENTS)
    profiles = load_records(WORKLOAD_PROFILES)
    points: list[WorkloadPoint] = []

    for method, backend in BACKEND_LABELS.items():
        for dataset in DATASETS:
            profile = profile_record(profiles, method, dataset)
            rows = profile["profile"]["rows"]
            baseline_time = mean_time(experiments, BASELINE_ALGORITHMS[method], dataset)
            local_time = mean_time(experiments, LOCAL_ALGORITHMS[method], dataset)
            edge_pct = final_level_mean(rows, "contracted_edges_by_level") / DATASET_DIRECTED_EDGES[dataset] * 100.0
            uv_pct = final_level_mean(rows, "closure_vertices_by_level") / DATASET_NODES[dataset] * 100.0
            points.append(
                WorkloadPoint(
                    backend=backend,
                    dataset=dataset,
                    edge_pct=edge_pct,
                    speedup=baseline_time / local_time,
                    uv_pct=uv_pct,
                )
            )

    return points


def bubble_size(uv_pct: float) -> float:
    return 18 + 1.15 * uv_pct


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

    points = load_points()
    fig, ax = plt.subplots(figsize=(3.35, 2.25), constrained_layout=True)
    ax.axhline(1.0, color="#777777", lw=0.7, ls="--", zorder=0)

    for backend in COLORS:
        backend_points = sorted(
            [point for point in points if point.backend == backend],
            key=lambda point: point.edge_pct,
        )
        ax.plot(
            [point.edge_pct for point in backend_points],
            [max(point.speedup, MIN_PLOTTED_SPEEDUP) for point in backend_points],
            color=COLORS[backend],
            lw=0.9,
            alpha=0.7,
            zorder=1,
        )
        for point in backend_points:
            plotted_speedup = max(point.speedup, MIN_PLOTTED_SPEEDUP)
            ax.scatter(
                point.edge_pct,
                plotted_speedup,
                s=bubble_size(point.uv_pct),
                marker=MARKERS[point.dataset],
                facecolor=COLORS[backend],
                edgecolor="white",
                linewidth=0.45,
                alpha=0.95,
                zorder=3,
            )
            ax.text(
                point.edge_pct,
                plotted_speedup,
                LABELS[point.dataset],
                ha="center",
                va="center",
                color="white",
                fontsize=5.5,
                fontweight="bold",
                zorder=4,
            )

    ax.set_yscale("log")
    ax.set_xlim(0.25, 2.25)
    ax.set_ylim(MIN_PLOTTED_SPEEDUP * 0.85, 70)
    ax.set_xlabel(r"Contracted edge workload, $|\bar E|/|E|$ (%)")
    ax.set_ylabel("Base/Local time")
    ax.set_yticks([0.01, 0.1, 1, 3, 10, 30, 50])
    ax.set_yticklabels([r"$\leq$0.01x", "0.1x", "1x", "3x", "10x", "30x", "50x"])
    ax.grid(axis="y", color="#d0d0d0", lw=0.45, ls=":", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    backend_handles = [
        Line2D([0], [0], color=name_color, marker="o", lw=0.9, markersize=4, label=name)
        for name, name_color in COLORS.items()
    ]
    ax.legend(
        handles=backend_handles,
        loc="lower right",
        frameon=False,
        handlelength=1.6,
        borderpad=0.1,
        labelspacing=0.25,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.015, metadata=PDF_METADATA)


if __name__ == "__main__":
    main()
