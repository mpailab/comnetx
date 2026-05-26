"""Import launcher stdout measurement summaries into result JSON files.

Some historical runs are available only as text logs printed by
``scripts/launch.py``. This script reconstructs the launcher-result JSON shape
from those logs so ``collect_results_registry.py`` can ingest the measurements.
It keeps repeated runs when their metrics differ, but skips near-identical rows
already present in the current registry.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any


DATASET_RE = re.compile(r"^Dataset:\s+(?P<dataset>.+?)\s+\((?P<batch>.+?)\s+batches\)\s*$")
BASELINE_RE = re.compile(r"^Baseline:\s+(?P<algorithm>.+?)\s*$")
METRIC_RE = re.compile(r"^(?P<name>Initial modularity|ARI|F1|NMI|Labels modularity|Final modularity|Total time):\s+(?P<value>[-+0-9.eE]+)\s*$")
ERROR_RE = re.compile(r"^Error\s+(?P<message>.+?)\s+on:\s+(?P<dataset>\S+)\s+(?P<batch>\S+)\s+(?P<algorithm>.+?)\s*$")

NEAR_TIME_TOL = 0.02
NEAR_METRIC_TOL = 0.005


def numeric(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def normalize_batch(raw: str) -> str:
    return raw.strip()


def parse_log(path: Path) -> tuple[list[dict[str, Any]], list[list[str]]]:
    records: list[dict[str, Any]] = []
    errors: list[list[str]] = []
    current: dict[str, Any] | None = None

    def flush() -> None:
        nonlocal current
        if not current:
            return
        if "error" in current:
            errors.append(
                [
                    str(current["algorithm"]),
                    str(current["dataset"]),
                    str(current["batch_strategy"]),
                    str(current["error"]),
                ]
            )
        elif {"dataset", "algorithm", "batch_strategy", "Final modularity", "Total time"} <= current.keys():
            records.append(dict(current))
        current = None

    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or set(line) == {"-"}:
                continue

            dataset_match = DATASET_RE.match(line)
            if dataset_match:
                flush()
                current = {
                    "dataset": dataset_match.group("dataset"),
                    "batch_strategy": normalize_batch(dataset_match.group("batch")),
                    "source_log": str(path),
                    "source_line": line_no,
                }
                continue

            baseline_match = BASELINE_RE.match(line)
            if baseline_match and current is not None:
                current["algorithm"] = baseline_match.group("algorithm")
                continue

            error_match = ERROR_RE.match(line)
            if error_match:
                if current is None:
                    current = {}
                current["dataset"] = current.get("dataset") or error_match.group("dataset")
                current["batch_strategy"] = current.get("batch_strategy") or normalize_batch(error_match.group("batch"))
                current["algorithm"] = current.get("algorithm") or error_match.group("algorithm")
                current["error"] = error_match.group("message")
                continue

            metric_match = METRIC_RE.match(line)
            if metric_match and current is not None:
                current[metric_match.group("name")] = float(metric_match.group("value"))

    flush()
    return records, errors


def registry_records(registry_dir: Path) -> list[dict[str, Any]]:
    path = registry_dir / "all_results.json"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return [item for item in data if isinstance(item, dict) and item.get("measurement_type") == "experiment"]


def registry_errors(registry_dir: Path) -> set[tuple[str | None, str | None, str | None, str | None]]:
    path = registry_dir / "errors.json"
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    out = set()
    for item in data:
        if not isinstance(item, dict):
            continue
        out.add(
            (
                item.get("algorithm"),
                item.get("dataset"),
                item.get("batch_strategy"),
                item.get("error_message"),
            )
        )
    return out


def near_equal(a: Any, b: Any, tol: float) -> bool:
    left = numeric(a)
    right = numeric(b)
    if left is None or right is None:
        return False
    return abs(left - right) <= tol


def is_near_existing(record: dict[str, Any], existing: list[dict[str, Any]]) -> bool:
    for item in existing:
        if item.get("algorithm") != record["algorithm"]:
            continue
        if item.get("base_dataset") != record["dataset"] and item.get("dataset") != record["dataset"]:
            continue
        if str(item.get("batch_strategy")) != str(record["batch_strategy"]):
            continue
        if not near_equal(item.get("total_time"), record.get("Total time"), NEAR_TIME_TOL):
            continue
        if not near_equal(item.get("final_modularity"), record.get("Final modularity"), NEAR_METRIC_TOL):
            continue
        item_nmi = item.get("final_nmi")
        if item_nmi is not None and not near_equal(item_nmi, record.get("NMI"), NEAR_METRIC_TOL):
            continue
        return True
    return False


def to_launcher_entry(record: dict[str, Any]) -> dict[str, Any]:
    entry = {
        "modularity": record["Final modularity"],
        "time": record["Total time"],
        "Final modularity": record["Final modularity"],
        "imported_from_text_log": True,
        "source_log": record["source_log"],
        "source_line": record["source_line"],
    }
    for key in ["Initial modularity", "ARI", "F1", "NMI", "Labels modularity"]:
        if key in record:
            entry[key] = record[key]
    return entry


def add_result(db: dict[str, Any], record: dict[str, Any], machine: str) -> None:
    machines = db.setdefault(record["algorithm"], {}).setdefault(record["dataset"], {})
    machine_name = machine
    duplicate_index = 2
    while record["batch_strategy"] in machines.get(machine_name, {}):
        machine_name = f"{machine}#{duplicate_index}"
        duplicate_index += 1
    machines.setdefault(machine_name, {})[record["batch_strategy"]] = [to_launcher_entry(record)]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def import_logs(
    log_paths: list[Path],
    output_dir: Path,
    registry_dir: Path,
    stamp: str,
    machine: str,
) -> dict[str, Any]:
    existing_records = registry_records(registry_dir)
    existing_errors = registry_errors(registry_dir)
    manifest: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "registry_dir": str(registry_dir),
        "output_dir": str(output_dir),
        "machine": machine,
        "sources": [],
        "written_result_files": [],
        "written_error_files": [],
    }

    for log_path in log_paths:
        records, errors = parse_log(log_path)
        db: dict[str, Any] = {}
        skipped_near_duplicates = 0
        imported_records = 0

        for record in records:
            if is_near_existing(record, existing_records):
                skipped_near_duplicates += 1
                continue
            add_result(db, record, machine)
            imported_records += 1

        new_errors = []
        skipped_errors = 0
        for algorithm, dataset, batch_strategy, message in errors:
            key = (algorithm, dataset, str(batch_strategy), message)
            if key in existing_errors:
                skipped_errors += 1
                continue
            new_errors.append([algorithm, dataset, str(batch_strategy), message])

        result_path = output_dir / f"{log_path.stem}_{stamp}.json"
        error_path = output_dir / f"errors_{log_path.stem}_{stamp}.json"
        if db:
            write_json(result_path, db)
            manifest["written_result_files"].append(str(result_path))
        if new_errors:
            write_json(error_path, new_errors)
            manifest["written_error_files"].append(str(error_path))

        manifest["sources"].append(
            {
                "log": str(log_path),
                "parsed_records": len(records),
                "imported_records": imported_records,
                "skipped_near_duplicate_records": skipped_near_duplicates,
                "parsed_errors": len(errors),
                "imported_errors": len(new_errors),
                "skipped_duplicate_errors": skipped_errors,
            }
        )

    manifest_path = output_dir / f"manifest_text_log_import_{stamp}.json"
    write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/imported_logs"))
    parser.add_argument("--registry-dir", type=Path, default=Path("results/registry"))
    parser.add_argument("--stamp", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--machine", default="text-log-import")
    args = parser.parse_args()

    manifest = import_logs(args.logs, args.output_dir, args.registry_dir, args.stamp, args.machine)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
