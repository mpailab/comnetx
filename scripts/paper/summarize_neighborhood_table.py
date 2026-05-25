"""Build the paper neighborhood table from raw neighborhood JSON logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


DEFAULT_SOURCES = {
    "dyn_cora": "results/neighborhood/small_b999:10.json",
    "dyn_acm": "results/neighborhood/small_b999:10.json",
    "dyn_citeseer": "results/neighborhood/small_b999:10.json",
    "patent": "results/neighborhood/tgc_b999:10.json",
    "dyn_pubmed": "results/neighborhood/dyn_at_gr_b-:10.json",
    "arxivmath": "results/neighborhood/tgc_b999:10.json",
}


def fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def summarize_dataset(dataset: str, path: Path, strategy: str) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    item = data["datasets"][dataset]
    n = item["info"]["n"]
    rows = item["strategies"][strategy]

    def pct_at(radius: int) -> list[float]:
        return [row[radius] / n * 100.0 for row in rows]

    b1 = pct_at(1)
    return {
        "dataset": dataset,
        "updates": len(rows),
        "affected": mean(row[0] for row in rows),
        "b1_mean": mean(b1),
        "b1_max": max(b1),
        "b2_mean": mean(pct_at(2)),
        "b3_mean": mean(pct_at(3)),
        "b5_mean": mean(pct_at(5)),
    }


def latex_row(row: dict) -> str:
    dataset = row["dataset"].replace("_", "\\_")
    return (
        f"{dataset} & "
        f"{row['updates']} & "
        f"{fmt(row['affected'], 1)} & "
        f"{fmt(row['b1_mean'])} / {fmt(row['b1_max'])} & "
        f"{fmt(row['b2_mean'])} & "
        f"{fmt(row['b3_mean'])} & "
        f"{fmt(row['b5_mean'])} \\\\"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="999:10")
    parser.add_argument(
        "--format",
        choices=["latex", "csv"],
        default="latex",
        help="Output format.",
    )
    args = parser.parse_args()

    rows = [
        summarize_dataset(dataset, Path(path), args.strategy)
        for dataset, path in DEFAULT_SOURCES.items()
    ]

    if args.format == "csv":
        print("dataset,updates,affected,b1_mean,b1_max,b2_mean,b3_mean,b5_mean")
        for row in rows:
            print(
                ",".join(
                    [
                        row["dataset"],
                        str(row["updates"]),
                        fmt(row["affected"], 1),
                        fmt(row["b1_mean"]),
                        fmt(row["b1_max"]),
                        fmt(row["b2_mean"]),
                        fmt(row["b3_mean"]),
                        fmt(row["b5_mean"]),
                    ]
                )
            )
    else:
        for row in rows:
            print(latex_row(row))


if __name__ == "__main__":
    main()
