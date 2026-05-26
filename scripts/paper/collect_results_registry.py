"""Collect all result JSON files into a searchable registry.

The collector understands the repository's launch-result JSON format, launcher
error JSON files, and neighborhood-analysis JSON files. Repeated runs with the
same launch parameters are preserved unless their measured series are exact or
near-exact duplicates, in which case their source paths are attached to one
canonical record.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any


NUMERIC_TOL_DIGITS = 6
TIME_TOL_DIGITS = 3


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def is_error_file(path: Path, data: Any) -> bool:
    if path.name.startswith("errors_"):
        return True
    return isinstance(data, list) and all(isinstance(item, list) for item in data)


def is_neighborhood_file(data: Any) -> bool:
    return (
        isinstance(data, dict)
        and isinstance(data.get("datasets"), dict)
        and isinstance(data.get("parameters"), dict)
        and "generated_at" in data
    )


def parse_algorithm(algorithm: str) -> dict[str, Any]:
    tokens = algorithm.split("-")
    method = tokens[0]
    out: dict[str, Any] = {
        "algorithm": algorithm,
        "method": method,
        "mode": None,
        "iterations": None,
        "subcoms_depth": None,
        "radius": None,
        "aggregation": None,
        "device": None,
        "feature_mode": None,
        "mod_weight": None,
        "resolution": None,
    }

    for token in tokens[1:]:
        if token in {"naive", "dynamic", "raw", "smart"}:
            out["mode"] = token
        elif token in {"gpu", "cpu"}:
            out["device"] = token
        elif token.startswith("i:"):
            out["iterations"] = token.split(":", 1)[1]
        elif token.startswith("L:"):
            out["subcoms_depth"] = token.split(":", 1)[1]
            out["mode"] = out["mode"] or "smart"
        elif token.startswith("r:"):
            out["radius"] = token.split(":", 1)[1]
            out["mode"] = out["mode"] or "smart"
        elif token.startswith("agg:"):
            out["aggregation"] = token.split(":", 1)[1]
        elif token.startswith("feat:"):
            out["feature_mode"] = token.split(":", 1)[1]
        elif token.startswith("mod:"):
            out["mod_weight"] = token.split(":", 1)[1]
        elif token.startswith("res:"):
            out["resolution"] = token.split(":", 1)[1]

    if out["mode"] is None:
        if out["subcoms_depth"] is not None or out["radius"] is not None:
            out["mode"] = "smart"
        else:
            out["mode"] = "unknown"

    return out


def dataset_parts(dataset: str) -> dict[str, Any]:
    force_undirected = dataset.endswith("-sym")
    base = dataset[:-4] if force_undirected else dataset
    return {
        "dataset": dataset,
        "base_dataset": base,
        "force_undirected": force_undirected,
    }


def source_owner(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    parts = rel.parts
    if len(parts) >= 2 and parts[0] == "results":
        return parts[1]
    return "unknown"


def numeric(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(value):
        return value
    return None


def round_measure(key: str, value: Any) -> Any:
    number = numeric(value)
    if number is None:
        return value
    digits = TIME_TOL_DIGITS if key == "time" else NUMERIC_TOL_DIGITS
    return round(number, digits)


def normalized_series(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for entry in entries:
        normalized.append({key: round_measure(key, value) for key, value in sorted(entry.items())})
    return normalized


def series_digest(entries: list[dict[str, Any]]) -> str:
    payload = json.dumps(normalized_series(entries), sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def run_key(record: dict[str, Any], include_machine: bool = False) -> str:
    fields = [
        "measurement_type",
        "algorithm",
        "method",
        "mode",
        "iterations",
        "subcoms_depth",
        "radius",
        "aggregation",
        "device",
        "feature_mode",
        "mod_weight",
        "resolution",
        "base_dataset",
        "force_undirected",
        "batch_strategy",
    ]
    if include_machine:
        fields.append("machine")
    return "|".join(str(record.get(field)) for field in fields)


def flatten_launch_result(path: Path, data: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    owner = source_owner(path, root)

    for algorithm, datasets in data.items():
        if not isinstance(datasets, dict):
            continue
        # Neighborhood files are handled elsewhere. Ordinary launch results have
        # algorithm names at the top level, then datasets, machines, batches.
        if algorithm in {"generated_at", "parameters", "datasets", "merged_from"}:
            continue

        alg = parse_algorithm(algorithm)
        for dataset, machines in datasets.items():
            if not isinstance(machines, dict):
                continue
            ds = dataset_parts(dataset)
            for machine, batches in machines.items():
                if not isinstance(batches, dict):
                    continue
                for batch_strategy, entries in batches.items():
                    if not isinstance(entries, list) or not entries:
                        continue
                    if not all(isinstance(item, dict) for item in entries):
                        continue

                    final = entries[-1]
                    times = [numeric(item.get("time")) for item in entries]
                    times = [item for item in times if item is not None]

                    record = {
                        "measurement_type": "experiment",
                        "source_file": str(path),
                        "source_owner": owner,
                        "algorithm": algorithm,
                        "machine": machine,
                        "batch_strategy": str(batch_strategy),
                        "updates": len(entries),
                        "total_time": sum(times) if times else None,
                        "final_modularity": numeric(final.get("modularity")),
                        "series_digest": series_digest(entries),
                        "series": entries,
                    }
                    record.update(alg)
                    record.update(ds)

                    for key, value in final.items():
                        if key in {"modularity", "time"}:
                            continue
                        number = numeric(value)
                        record[f"final_{key.lower()}"] = number if number is not None else value

                    record["run_key"] = run_key(record, include_machine=False)
                    record["run_key_with_machine"] = run_key(record, include_machine=True)
                    records.append(record)

    return records


def flatten_errors(path: Path, data: list[Any], root: Path) -> list[dict[str, Any]]:
    errors = []
    owner = source_owner(path, root)
    for idx, item in enumerate(data):
        algorithm = dataset = batch_strategy = message = None
        if isinstance(item, list) and len(item) >= 4:
            algorithm, dataset, batch_strategy, message = item[:4]
        else:
            message = repr(item)
        record = {
            "measurement_type": "error",
            "source_file": str(path),
            "source_owner": owner,
            "error_index": idx,
            "algorithm": algorithm,
            "dataset": dataset,
            "batch_strategy": str(batch_strategy) if batch_strategy is not None else None,
            "error_message": message,
        }
        if algorithm:
            record.update(parse_algorithm(str(algorithm)))
        if dataset:
            record.update(dataset_parts(str(dataset)))
        errors.append(record)
    return errors


def flatten_neighborhood(path: Path, data: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    records = []
    owner = source_owner(path, root)
    parameters = data.get("parameters", {})
    for dataset, payload in data.get("datasets", {}).items():
        info = payload.get("info", {})
        for strategy, rows in payload.get("strategies", {}).items():
            if not isinstance(rows, list) or not rows:
                continue
            record = {
                "measurement_type": "neighborhood",
                "source_file": str(path),
                "source_owner": owner,
                "dataset": dataset,
                "base_dataset": dataset,
                "force_undirected": False,
                "batch_strategy": str(strategy),
                "updates": len(rows),
                "nodes": info.get("n"),
                "edges": info.get("m"),
                "directedness": info.get("d"),
                "weightedness": info.get("w"),
                "max_step": parameters.get("max_step"),
                "neighborhood_series": rows,
            }
            if rows and all(isinstance(row, list) and len(row) > 0 for row in rows):
                n = numeric(info.get("n"))
                for radius in range(max(len(row) for row in rows)):
                    vals = [row[radius] for row in rows if len(row) > radius]
                    if not vals:
                        continue
                    record[f"B{radius}_mean"] = mean(vals)
                    record[f"B{radius}_max"] = max(vals)
                    if n:
                        record[f"B{radius}_pct_mean"] = mean(vals) / n * 100.0
                        record[f"B{radius}_pct_max"] = max(vals) / n * 100.0
            records.append(record)
    return records


def dedupe_experiments(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seen: dict[tuple[str, str], dict[str, Any]] = {}
    deduped: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []

    for record in records:
        key = (record["run_key_with_machine"], record["series_digest"])
        if key not in seen:
            record = dict(record)
            record["duplicate_count"] = 0
            record["duplicate_sources"] = []
            seen[key] = record
            deduped.append(record)
            continue

        canonical = seen[key]
        canonical["duplicate_count"] += 1
        canonical["duplicate_sources"].append(record["source_file"])
        duplicates.append(
            {
                "canonical_source": canonical["source_file"],
                "duplicate_source": record["source_file"],
                "run_key_with_machine": record["run_key_with_machine"],
                "series_digest": record["series_digest"],
            }
        )

    return deduped, duplicates


def compact_record(record: dict[str, Any]) -> dict[str, Any]:
    skip = {"series", "neighborhood_series"}
    return {key: value for key, value in record.items() if key not in skip}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize_groups(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("measurement_type") == "experiment":
            groups[record["run_key"]].append(record)

    summaries = []
    for key, items in sorted(groups.items()):
        first = items[0]
        mods = [numeric(item.get("final_modularity")) for item in items]
        mods = [item for item in mods if item is not None]
        times = [numeric(item.get("total_time")) for item in items]
        times = [item for item in times if item is not None]
        nmis = [numeric(item.get("final_nmi")) for item in items]
        nmis = [item for item in nmis if item is not None]
        summaries.append(
            {
                "run_key": key,
                "algorithm": first.get("algorithm"),
                "method": first.get("method"),
                "mode": first.get("mode"),
                "iterations": first.get("iterations"),
                "subcoms_depth": first.get("subcoms_depth"),
                "radius": first.get("radius"),
                "aggregation": first.get("aggregation"),
                "device": first.get("device"),
                "feature_mode": first.get("feature_mode"),
                "mod_weight": first.get("mod_weight"),
                "resolution": first.get("resolution"),
                "base_dataset": first.get("base_dataset"),
                "force_undirected": first.get("force_undirected"),
                "batch_strategy": first.get("batch_strategy"),
                "unique_runs": len(items),
                "duplicate_sources": sum(int(item.get("duplicate_count", 0)) for item in items),
                "machines": ";".join(sorted({str(item.get("machine")) for item in items})),
                "sources": ";".join(sorted({str(item.get("source_file")) for item in items})),
                "modularity_mean": mean(mods) if mods else None,
                "modularity_std": stdev(mods) if len(mods) > 1 else 0.0 if mods else None,
                "time_mean": mean(times) if times else None,
                "time_std": stdev(times) if len(times) > 1 else 0.0 if times else None,
                "nmi_mean": mean(nmis) if nmis else None,
                "nmi_std": stdev(nmis) if len(nmis) > 1 else 0.0 if nmis else None,
            }
        )
    return summaries


def collect(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    experiment_records: list[dict[str, Any]] = []
    error_records: list[dict[str, Any]] = []
    neighborhood_records: list[dict[str, Any]] = []
    unreadable = []

    for path in sorted(input_dir.rglob("*.json")):
        if output_dir in path.parents:
            continue
        try:
            data = load_json(path)
        except Exception as exc:  # pragma: no cover - defensive registry code
            unreadable.append({"source_file": str(path), "error": str(exc)})
            continue

        if is_error_file(path, data):
            error_records.extend(flatten_errors(path, data, input_dir.parent))
        elif is_neighborhood_file(data):
            neighborhood_records.extend(flatten_neighborhood(path, data, input_dir.parent))
        elif isinstance(data, dict):
            experiment_records.extend(flatten_launch_result(path, data, input_dir.parent))
        else:
            unreadable.append({"source_file": str(path), "error": f"Unsupported JSON root: {type(data).__name__}"})

    experiment_records, duplicates = dedupe_experiments(experiment_records)
    all_records = experiment_records + neighborhood_records
    summary = summarize_groups(experiment_records)

    write_json(output_dir / "all_results.json", [compact_record(item) for item in all_records])
    write_json(output_dir / "all_results_with_series.json", all_records)
    write_csv(output_dir / "all_results.csv", [compact_record(item) for item in all_records])
    write_json(output_dir / "summary_by_run_key.json", summary)
    write_csv(output_dir / "summary_by_run_key.csv", summary)
    write_json(output_dir / "errors.json", error_records)
    write_csv(output_dir / "errors.csv", error_records)
    write_json(output_dir / "deduplicated_sources.json", duplicates)
    write_csv(output_dir / "deduplicated_sources.csv", duplicates)
    write_json(output_dir / "unreadable.json", unreadable)

    manifest = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "experiment_records": len(experiment_records),
        "neighborhood_records": len(neighborhood_records),
        "error_records": len(error_records),
        "deduplicated_sources": len(duplicates),
        "unreadable_files": len(unreadable),
        "outputs": {
            "compact_json": str(output_dir / "all_results.json"),
            "full_json": str(output_dir / "all_results_with_series.json"),
            "compact_csv": str(output_dir / "all_results.csv"),
            "summary_csv": str(output_dir / "summary_by_run_key.csv"),
            "errors_csv": str(output_dir / "errors.csv"),
            "deduplicated_sources_csv": str(output_dir / "deduplicated_sources.csv"),
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="results")
    parser.add_argument("--output-dir", default="results/registry")
    args = parser.parse_args()

    manifest = collect(Path(args.input_dir), Path(args.output_dir))
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
