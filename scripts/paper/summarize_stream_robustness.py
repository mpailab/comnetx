"""Generate the paired stream-id robustness rows used in the ICDM paper."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any


DEFAULT_REGISTRY = Path("results/registry/all_results.json")


@dataclass(frozen=True)
class PairSpec:
    backend: str
    dataset: str
    batch_strategy: str
    local_algorithm: str
    baseline_algorithm: str
    force_undirected: bool = True


SECTIONS: list[tuple[str, list[PairSpec]]] = [
    (
        "Leiden, Local versus full recomputation",
        [
            PairSpec("Leiden", "dyn_pubmed", "999:10", "leidenalg-L:3-r:1-gpu", "leidenalg-naive"),
            PairSpec("Leiden", "dyn_pubmed", "999:50", "leidenalg-L:3-r:1-gpu", "leidenalg-naive"),
            PairSpec("Leiden", "dyn_pubmed", "999:100", "leidenalg-L:3-r:1-gpu", "leidenalg-naive"),
            PairSpec("Leiden", "arxivmath", "999:10", "leidenalg-L:3-r:1-gpu", "leidenalg-naive"),
            PairSpec("Leiden", "arxivmath", "999:50", "leidenalg-L:3-r:1-gpu", "leidenalg-naive"),
            PairSpec("Leiden", "arxivmath", "999:100", "leidenalg-L:3-r:1-gpu", "leidenalg-naive"),
        ],
    ),
    (
        "DF-Leiden, Local versus native dynamic backend",
        [
            PairSpec("DF-Leiden", "dyn_pubmed", "999:10", "dfleiden-L:3-r:1-gpu", "dfleiden-dynamic"),
            PairSpec("DF-Leiden", "dyn_pubmed", "999:50", "dfleiden-L:3-r:1-gpu", "dfleiden-dynamic"),
            PairSpec("DF-Leiden", "dyn_pubmed", "999:100", "dfleiden-L:3-r:1-gpu", "dfleiden-dynamic"),
            PairSpec("DF-Leiden", "arxivmath", "999:10", "dfleiden-L:3-r:1-gpu", "dfleiden-dynamic"),
            PairSpec("DF-Leiden", "arxivmath", "999:50", "dfleiden-L:3-r:1-gpu", "dfleiden-dynamic"),
            PairSpec("DF-Leiden", "arxivmath", "999:100", "dfleiden-L:3-r:1-gpu", "dfleiden-dynamic"),
        ],
    ),
    (
        "S$^2$CAG, dataset features",
        [
            PairSpec(
                "S$^2$CAG",
                "dyn_pubmed",
                "999:10",
                "s2cag-i:10-L:3-r:1-agg:norm-gpu-feat:dataset",
                "s2cag-i:10-naive-feat:dataset",
            ),
            PairSpec(
                "S$^2$CAG",
                "dyn_pubmed",
                "999:100",
                "s2cag-i:10-L:3-r:1-agg:norm-gpu-feat:dataset",
                "s2cag-i:10-naive-feat:dataset",
            ),
            PairSpec(
                "S$^2$CAG",
                "dyn_pubmed",
                "99:50",
                "s2cag-i:10-L:3-r:1-agg:norm-gpu-feat:dataset",
                "s2cag-i:10-naive-feat:dataset",
            ),
            PairSpec(
                "S$^2$CAG",
                "arxivmath",
                "999:10",
                "s2cag-i:10-L:3-r:1-agg:norm-gpu-feat:dataset",
                "s2cag-i:10-naive-feat:dataset",
            ),
        ],
    ),
]


def numeric(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def load_experiment_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [item for item in payload if item.get("measurement_type") == "experiment"]


def record_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record.get("algorithm"),
        record.get("base_dataset"),
        record.get("force_undirected"),
        record.get("batch_strategy"),
        record.get("stream_id"),
    )


def index_records(records: list[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
    index: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in records:
        key = record_key(record)
        if key in index:
            first = index[key].get("source_file")
            second = record.get("source_file")
            raise ValueError(f"Duplicate record for {key}: {first} and {second}")
        index[key] = record
    return index


def paired_records(
    spec: PairSpec,
    records: list[dict[str, Any]],
    index: dict[tuple[Any, ...], dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    local_streams = {
        record.get("stream_id")
        for record in records
        if record.get("algorithm") == spec.local_algorithm
        and record.get("base_dataset") == spec.dataset
        and record.get("force_undirected") == spec.force_undirected
        and record.get("batch_strategy") == spec.batch_strategy
    }
    baseline_streams = {
        record.get("stream_id")
        for record in records
        if record.get("algorithm") == spec.baseline_algorithm
        and record.get("base_dataset") == spec.dataset
        and record.get("force_undirected") == spec.force_undirected
        and record.get("batch_strategy") == spec.batch_strategy
    }

    pairs = []
    for stream_id in sorted(local_streams & baseline_streams):
        local_key = (
            spec.local_algorithm,
            spec.dataset,
            spec.force_undirected,
            spec.batch_strategy,
            stream_id,
        )
        baseline_key = (
            spec.baseline_algorithm,
            spec.dataset,
            spec.force_undirected,
            spec.batch_strategy,
            stream_id,
        )
        pairs.append((index[local_key], index[baseline_key]))
    return pairs


def values(pairs: list[tuple[dict[str, Any], dict[str, Any]]], side: int, key: str) -> list[float]:
    out = []
    for pair in pairs:
        value = numeric(pair[side].get(key))
        if value is not None:
            out.append(value)
    return out


def deltas(pairs: list[tuple[dict[str, Any], dict[str, Any]]], key: str) -> list[float]:
    out = []
    for local, baseline in pairs:
        local_value = numeric(local.get(key))
        baseline_value = numeric(baseline.get(key))
        if local_value is not None and baseline_value is not None:
            out.append(local_value - baseline_value)
    return out


def mean_std(items: list[float]) -> tuple[float | None, float | None]:
    if not items:
        return None, None
    return mean(items), stdev(items) if len(items) > 1 else 0.0


def summarize(spec: PairSpec, pairs: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    local_times = values(pairs, 0, "total_time")
    baseline_times = values(pairs, 1, "total_time")
    speedups = []
    for local, baseline in pairs:
        local_time = numeric(local.get("total_time"))
        baseline_time = numeric(baseline.get("total_time"))
        if local_time is not None and baseline_time is not None and local_time != 0:
            speedups.append(baseline_time / local_time)

    summary: dict[str, Any] = {
        "backend": spec.backend,
        "dataset": spec.dataset,
        "batch_strategy": spec.batch_strategy,
        "n": len(pairs),
    }

    metrics = {
        "local_modularity": values(pairs, 0, "final_modularity"),
        "baseline_modularity": values(pairs, 1, "final_modularity"),
        "local_time": local_times,
        "baseline_time": baseline_times,
        "speedup": speedups,
        "nmi_delta": deltas(pairs, "final_nmi"),
    }
    for name, metric_values in metrics.items():
        metric_mean, metric_std = mean_std(metric_values)
        summary[f"{name}_mean"] = metric_mean
        summary[f"{name}_std"] = metric_std
    return summary


def format_pm(summary: dict[str, Any], prefix: str, digits: int) -> str:
    value = summary[f"{prefix}_mean"]
    spread = summary[f"{prefix}_std"]
    if value is None or spread is None:
        return "--"
    return f"{value:.{digits}f} $\\pm$ {spread:.{digits}f}"


def latex_dataset(dataset: str) -> str:
    if dataset == "arxivmath":
        return "arxivmath"
    return dataset.replace("_", "\\_")


def write_latex(summaries_by_section: list[tuple[str, list[dict[str, Any]]]]) -> None:
    for section_index, (section, summaries) in enumerate(summaries_by_section):
        if section_index:
            print("\\addlinespace")
        print(f"\\multicolumn{{9}}{{l}}{{\\textit{{{section}}}}} \\\\")
        for summary in summaries:
            columns = [
                summary["backend"],
                latex_dataset(summary["dataset"]),
                summary["batch_strategy"],
                str(summary["n"]),
                format_pm(summary, "local_modularity", 3),
                format_pm(summary, "baseline_modularity", 3),
                format_pm(summary, "local_time", 2),
                format_pm(summary, "baseline_time", 2),
                format_pm(summary, "speedup", 1),
            ]
            print(" & ".join(columns) + " \\\\")


def write_csv(summaries_by_section: list[tuple[str, list[dict[str, Any]]]]) -> None:
    rows = []
    for section, summaries in summaries_by_section:
        for summary in summaries:
            row = {"section": section}
            row.update(summary)
            rows.append(row)
    fieldnames = sorted({key for row in rows for key in row})
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


def write_nmi_deltas(summaries_by_section: list[tuple[str, list[dict[str, Any]]]]) -> None:
    selected: dict[str, dict[str, dict[str, Any]]] = {}
    for _, summaries in summaries_by_section:
        for summary in summaries:
            if summary["batch_strategy"] == "999:10":
                selected.setdefault(summary["backend"], {})[summary["dataset"]] = summary

    parts = []
    for backend in ["Leiden", "DF-Leiden", "S$^2$CAG"]:
        dyn_pubmed = selected[backend]["dyn_pubmed"]["nmi_delta_mean"]
        arxivmath = selected[backend]["arxivmath"]["nmi_delta_mean"]
        parts.append(f"{backend}: {dyn_pubmed:+.3f}/{arxivmath:+.3f}")
    print("Final-NMI deltas for 999:10 (Local minus baseline): " + "; ".join(parts))


def build_summaries(registry_path: Path) -> list[tuple[str, list[dict[str, Any]]]]:
    records = load_experiment_records(registry_path)
    index = index_records(records)
    summaries_by_section = []
    for section, specs in SECTIONS:
        summaries = []
        for spec in specs:
            pairs = paired_records(spec, records, index)
            if not pairs:
                raise ValueError(f"No paired stream_id records for {spec}")
            summaries.append(summarize(spec, pairs))
        summaries_by_section.append((section, summaries))
    return summaries_by_section


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--format", choices=["latex", "csv"], default="latex")
    parser.add_argument(
        "--nmi-deltas",
        action="store_true",
        help="Also print the final-NMI delta line for the completed 999:10 rows.",
    )
    args = parser.parse_args()

    summaries_by_section = build_summaries(args.registry)
    if args.format == "latex":
        write_latex(summaries_by_section)
    else:
        write_csv(summaries_by_section)

    if args.nmi_deltas:
        if args.format == "latex":
            print()
        write_nmi_deltas(summaries_by_section)


if __name__ == "__main__":
    main()
