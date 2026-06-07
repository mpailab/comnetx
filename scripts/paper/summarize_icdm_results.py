"""Flatten and summarize ComNetX result JSON files for paper tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


def flatten_file(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict] = []

    for algorithm, datasets in data.items():
        if not isinstance(datasets, dict):
            continue
        for dataset, machines in datasets.items():
            if not isinstance(machines, dict):
                continue
            for machine, batches in machines.items():
                if not isinstance(batches, dict):
                    continue
                for batch, entries in batches.items():
                    if not entries:
                        continue
                    total_time = sum(float(item.get("time", 0.0)) for item in entries)
                    final = entries[-1]
                    row = {
                        "source": str(path),
                        "algorithm": algorithm,
                        "dataset": dataset,
                        "machine": machine,
                        "batch": batch,
                        "updates": len(entries),
                        "final_modularity": final.get("modularity"),
                        "total_time": total_time,
                    }
                    for metric in ["ari", "f1", "nmi", "AMI", "NMI", "F1"]:
                        if metric in final:
                            row[metric.lower()] = final[metric]
                    rows.append(row)

    return rows


def read_all(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        rows.extend(flatten_file(path))
    return rows


def numeric(values: list) -> list[float]:
    out = []
    for value in values:
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            out.append(value)
    return out


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (row["algorithm"], row["dataset"], row["batch"])
        groups[key].append(row)

    summary = []
    for (algorithm, dataset, batch), items in sorted(groups.items()):
        mods = numeric([item.get("final_modularity") for item in items])
        times = numeric([item.get("total_time") for item in items])
        nmis = numeric([item.get("nmi") for item in items])
        summary.append(
            {
                "algorithm": algorithm,
                "dataset": dataset,
                "batch": batch,
                "runs": len(items),
                "modularity_mean": mean(mods) if mods else None,
                "modularity_std": stdev(mods) if len(mods) > 1 else 0.0 if mods else None,
                "time_mean": mean(times) if times else None,
                "time_std": stdev(times) if len(times) > 1 else 0.0 if times else None,
                "nmi_mean": mean(nmis) if nmis else None,
                "nmi_std": stdev(nmis) if len(nmis) > 1 else 0.0 if nmis else None,
            }
        )
    return summary


def write_csv(rows: list[dict], path: Path | None) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    handle = path.open("w", newline="", encoding="utf-8") if path else None
    try:
        stream = handle if handle is not None else None
        writer = csv.DictWriter(stream or __import__("sys").stdout, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    finally:
        if handle is not None:
            handle.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", help="Result JSON files.")
    parser.add_argument("--summary", action="store_true", help="Aggregate by algorithm/dataset/batch.")
    parser.add_argument("--output", help="Optional CSV output path.")
    args = parser.parse_args()

    paths = [Path(item) for item in args.results]
    rows = read_all(paths)
    if args.summary:
        rows = summarize(rows)

    write_csv(rows, Path(args.output) if args.output else None)


if __name__ == "__main__":
    main()

