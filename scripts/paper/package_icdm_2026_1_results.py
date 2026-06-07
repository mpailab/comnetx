"""Build the unified full-fidelity ICDM 2026-1 measurement bundle."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.paper.collect_results_registry import (  # noqa: E402
    dataset_parts,
    is_profile_file,
    numeric,
    parse_algorithm,
    series_digest,
    stable_digest,
)


DEFAULT_TAG = "icdm-2026-1"
DEFAULT_INPUT_DIR = Path("results")
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / DEFAULT_TAG
ARTICLE_DATASETS = {
    "dyn_cora",
    "dyn_acm",
    "dyn_citeseer",
    "patent",
    "dyn_pubmed",
    "arxivmath",
}
CORE_METHODS = {"leidenalg", "dfleiden", "s2cag"}
LEGACY_COMPATIBILITY_ALGORITHMS = {
    "flmig": {
        "baseline": "flmig-i:10-naive",
        "local": "flmig-i:10-L:3-r:1-gpu",
        "baseline_label": "Naive",
    },
    "prgpt:infomap": {
        "baseline": "prgpt:infomap-naive",
        "local": "prgpt:infomap-L:3-r:1-gpu",
        "baseline_label": "Naive",
    },
    "prgpt:locale": {
        "baseline": "prgpt:locale-naive",
        "local": "prgpt:locale-L:3-r:1-gpu",
        "baseline_label": "Naive",
    },
    "magi": {
        "baseline": "magi-i:10-naive-feat:dataset",
        "local": "magi-i:10-L:3-r:1-agg:norm-gpu-feat:random",
        "baseline_label": "Naive",
    },
    "dmon": {
        "baseline": "dmon-i:10-naive-feat:dataset",
        "local": "dmon-i:10-L:3-r:1-agg:norm-gpu-feat:random",
        "baseline_label": "Naive",
    },
    "mfc": {
        "baseline": "mfc-i:100-dynamic-feat:dataset",
        "local": "mfc-i:100-L:3-r:1-agg:norm-gpu-feat:random",
        "baseline_label": "Dynamic",
    },
}
LEGACY_EXPECTED_OOM_BASELINES = {
    ("mfc", "arxivmath"),
    ("magi", "arxivmath"),
    ("dmon", "arxivmath"),
}
ARTICLE_NAVIGATION = [
    {
        "article_item": "Compatibility study (tab:main-results)",
        "data": "measurements/experiment_measurements.json",
        "families": ["single_container_real_graph", "article_compatibility_baselines"],
        "methods": [
            "dfleiden",
            "dmon",
            "flmig",
            "leidenalg",
            "magi",
            "mfc",
            "prgpt:infomap",
            "prgpt:locale",
            "s2cag",
        ],
    },
    {
        "article_item": "Repeated 999:10 robustness (tab:stability)",
        "data": "measurements/experiment_measurements.json",
        "families": ["dyn_pubmed_leiden_repeats", "single_container_real_graph"],
        "methods": ["leidenalg", "dfleiden", "s2cag"],
    },
    {
        "article_item": "Empirical neighborhood growth (tab:workload)",
        "data": "measurements/neighborhood_measurements.json",
        "families": ["article_neighborhood_growth"],
    },
    {
        "article_item": "Contracted workload and speedup (fig:workload-speedup)",
        "data": "measurements/workload_profiles.json",
        "families": ["article_completion_workload_profiles", "single_container_workload_profiles"],
        "methods": ["leidenalg", "dfleiden", "s2cag"],
        "note": "Speed ratios use paired records from measurements/experiment_measurements.json.",
    },
    {
        "article_item": "Leiden topology ablation (fig:topology-ablation)",
        "data": "measurements/experiment_measurements.json",
        "families": ["article_completion_real_graph", "single_container_real_graph"],
        "methods": ["leidenalg"],
    },
    {
        "article_item": "Gamma-resolution sensitivity",
        "data": "measurements/experiment_measurements.json",
        "families": ["gamma_sweep"],
        "methods": ["leidenalg"],
    },
    {
        "article_item": "S2CAG feature-mode ablation (tab:feature-ablation)",
        "data": "measurements/experiment_measurements.json",
        "families": ["single_container_real_graph"],
        "methods": ["s2cag"],
    },
    {
        "article_item": "Leiden closure/contraction ablation (tab:closure-ablation)",
        "data": "measurements/workload_profiles.json",
        "families": ["single_container_workload_profiles"],
        "methods": ["leidenalg"],
    },
    {
        "article_item": "Long-horizon topology endpoints (tab:long-horizon)",
        "data": "measurements/experiment_measurements.json",
        "families": ["single_container_real_graph"],
        "methods": ["leidenalg", "dfleiden"],
    },
    {
        "article_item": "Directed-control reviewer check",
        "data": "measurements/experiment_measurements.json",
        "families": ["single_container_directed_control"],
        "methods": ["leidenalg"],
    },
    {
        "article_item": "Conductance / normalized-cut reviewer check",
        "data": "measurements/cut_metrics.json",
        "families": ["leiden_cut_metrics"],
        "methods": ["leidenalg"],
    },
    {
        "article_item": "Controlled DSBM stress test",
        "data": "measurements/experiment_measurements.json",
        "families": ["single_container_dsbm"],
        "methods": ["leidenalg"],
    },
]
OLD_SOURCE_KEYS = {
    "source_artifact",
    "source_file",
    "source_owner",
    "source_root",
    "source_series",
    "stream_key",
    "duplicate_sources",
}
PUBLISH_DROP_KEYS = OLD_SOURCE_KEYS | {
    "article_role",
    "compatibility_role",
    "cut_metrics_digest",
    "legacy_groups",
    "legacy_source_bundle",
    "legacy_source_group",
    "measurement_id",
    "machine",
    "previous_bundle_record_id",
    "profile_digest",
    "profile_run_metadata",
    "profile_series_digest",
    "provenance",
    "reviewer_followup_type",
    "run_key",
    "run_key_with_machine",
    "series_digest",
    "stream_id",
    "cut_metrics_run_metadata",
}
PRESERVED_GROUPS = {
    "dyn_pubmed_leiden_repeats",
    "gamma_sweep_measurements",
    "single_container_real_graph_measurements",
    "single_container_workload_profiles",
}
PRESERVED_EXISTING_FAMILIES = {
    "dyn_pubmed_leiden_repeats",
    "gamma_sweep",
}
COMMITTED_PRESERVED_FILES = [
    ("dyn_pubmed_leiden_repeats", Path("results/icdm-2026-1/measurements/dyn_pubmed_leiden_repeats.json")),
    ("gamma_sweep_measurements", Path("results/icdm-2026-1/measurements/gamma_sweep_measurements.json")),
]
TYPE_OUTPUTS = {
    "experiment": {
        "filename": "experiment_measurements.json",
        "description": (
            "Full experiment time-series records used by article tables, "
            "ablation analyses, long-horizon runs, DSBM stress tests, and "
            "directed-control checks."
        ),
    },
    "workload_profile": {
        "filename": "workload_profiles.json",
        "description": (
            "Full workload-profile payloads for Leiden, DF-Leiden, and S2CAG "
            "mechanism analysis."
        ),
    },
    "neighborhood": {
        "filename": "neighborhood_measurements.json",
        "description": (
            "Neighborhood growth records used by the empirical neighborhood "
            "table."
        ),
    },
    "cut_metrics": {
        "filename": "cut_metrics.json",
        "description": (
            "Targeted final-partition conductance and normalized-cut metrics "
            "for key Leiden 999:10 runs."
        ),
    },
}
SERVICE_PATH_KEYS = {
    "cache_dir",
    "log_dir",
    "log_root",
    "output",
    "output_dir",
    "result_path",
    "results_root",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def current_commit(repo: Path) -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        return "unknown"
    return proc.stdout.strip()


def finite(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, list):
        return [finite(item) for item in value]
    if isinstance(value, dict):
        return {key: finite(item) for key, item in value.items()}
    return value


def strip_service_paths(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text.lower() in SERVICE_PATH_KEYS:
                continue
            cleaned_item = strip_service_paths(item)
            if cleaned_item is not None:
                cleaned[key] = cleaned_item
        return cleaned
    if isinstance(value, list):
        return [
            cleaned_item
            for item in value
            if (cleaned_item := strip_service_paths(item)) is not None
        ]
    if isinstance(value, str) and "results/paper_icdm" in value:
        return None
    return value


def digest_payload(value: Any) -> str:
    payload = json.dumps(finite(value), sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def task_parts(task: str) -> dict[str, Any]:
    match = re.match(r"^(?P<index>\d+)_?(?P<body>.*?)(?:_r(?P<repeat>\d+))?$", task)
    if not match:
        return {"task": task, "task_index": None, "repeat": None}
    return {
        "task": task,
        "task_index": int(match.group("index")),
        "task_name": match.group("body") or None,
        "repeat": int(match.group("repeat")) if match.group("repeat") else None,
    }


def provenance_from_path(path: Path, input_dir: Path) -> dict[str, Any]:
    rel = path.relative_to(input_dir)
    collection = rel.parts[0]
    series_name = rel.parts[1] if len(rel.parts) > 1 else None
    series_match = re.match(r"series_(\d+)$", series_name or "")
    task = rel.parts[2] if len(rel.parts) > 3 else path.parent.name
    provenance = {
        "collection": collection,
        "series": int(series_match.group(1)) if series_match else None,
        "series_name": series_name,
        "artifact": path.stem,
    }
    provenance.update(task_parts(task))
    return provenance


def provenance_from_legacy_record(record: dict[str, Any]) -> dict[str, Any]:
    source = record.get("source_artifact") or record.get("source_file") or ""
    if source:
        path = Path(str(source))
        parts = path.parts
        try:
            start = parts.index("results") + 1
            collection = parts[start]
            series_name = parts[start + 1]
            task = parts[start + 2] if len(parts) > start + 3 else path.parent.name
        except (ValueError, IndexError):
            collection = str(record.get("source_owner") or "previous_icdm_2026_1")
            series_name = None
            task = path.parent.name
        series_match = re.match(r"series_(\d+)$", series_name or "")
        provenance = {
            "collection": collection,
            "series": int(series_match.group(1)) if series_match else None,
            "series_name": series_name,
            "artifact": path.stem,
        }
        provenance.update(task_parts(task))
        return provenance

    source_series = str(record.get("source_series") or "")
    parts = source_series.split("/")
    collection = parts[0] if parts and parts[0] else "previous_icdm_2026_1"
    series_name = parts[1] if len(parts) > 1 else None
    series_match = re.match(r"series_(\d+)$", series_name or "")
    return {
        "collection": collection,
        "series": int(series_match.group(1)) if series_match else None,
        "series_name": series_name,
        "task": None,
        "task_index": None,
        "task_name": None,
        "repeat": None,
        "artifact": None,
    }


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


def logical_stream_key(record: dict[str, Any]) -> str:
    provenance = record.get("provenance", {})
    fields = [
        provenance.get("collection"),
        provenance.get("series"),
        provenance.get("task"),
        provenance.get("artifact"),
        record.get("machine"),
        record.get("base_dataset"),
        record.get("force_undirected"),
        record.get("batch_strategy"),
        record.get("updates"),
    ]
    return "|".join(str(field) for field in fields)


def measurement_family(
    provenance: dict[str, Any],
    measurement_type: str,
    record: dict[str, Any] | None = None,
) -> str:
    if record and record.get("reviewer_followup_type") in {
        "dyn_pubmed_leiden_repeats",
        "gamma_sweep",
    }:
        return str(record["reviewer_followup_type"])

    collection = provenance.get("collection")
    series = provenance.get("series")
    task = str(provenance.get("task") or "")
    if measurement_type == "cut_metrics":
        return "leiden_cut_metrics"
    if measurement_type == "neighborhood":
        return "article_neighborhood_growth"
    if measurement_type == "run_manifest":
        return "run_manifest"
    if collection == "paper_icdm_article_completion":
        return "article_completion_workload_profiles" if measurement_type == "workload_profile" else "article_completion_real_graph"
    if collection == "article_compatibility_baselines":
        return "article_compatibility_baselines"
    if collection == "paper_icdm_reviewer_followup" and series == 12:
        return "gamma_sweep"
    if collection == "paper_icdm_reviewer_followup":
        return "reviewer_followup"
    if task.startswith("11_"):
        return "single_container_dsbm"
    if task.startswith("12_"):
        return "single_container_directed_control"
    if measurement_type == "workload_profile":
        return "single_container_workload_profiles"
    return "single_container_real_graph"


def flatten_launch_result(path: Path, payload: dict[str, Any], input_dir: Path) -> list[dict[str, Any]]:
    provenance = provenance_from_path(path, input_dir)
    family = measurement_family(provenance, "experiment")
    records: list[dict[str, Any]] = []

    for algorithm, datasets in payload.items():
        if not isinstance(datasets, dict):
            continue
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
                for batch_strategy, series in batches.items():
                    if not isinstance(series, list) or not series:
                        continue
                    if not all(isinstance(item, dict) for item in series):
                        continue

                    final = copy.deepcopy(series[-1])
                    times = [numeric(item.get("time")) for item in series]
                    times = [item for item in times if item is not None]
                    record: dict[str, Any] = {
                        "measurement_type": "experiment",
                        "measurement_family": family,
                        "provenance": provenance,
                        "algorithm": algorithm,
                        "machine": machine,
                        "batch_strategy": str(batch_strategy),
                        "updates": len(series),
                        "total_time": sum(times) if times else None,
                        "final_entry": final,
                        "final_modularity": numeric(final.get("modularity")),
                        "series_digest": series_digest(series),
                        "series": copy.deepcopy(series),
                    }
                    record.update(alg)
                    record.update(ds)

                    for key, value in final.items():
                        if key in {"modularity", "time"}:
                            continue
                        metric_key = f"final_{key.lower()}"
                        number = numeric(value)
                        record[metric_key] = number if number is not None else value

                    record["stream_key"] = logical_stream_key(record)
                    record["stream_id"] = stable_digest(record["stream_key"].split("|"))
                    record["run_key"] = run_key(record, include_machine=False)
                    record["run_key_with_machine"] = run_key(record, include_machine=True)
                    record["measurement_id"] = stable_digest(
                        [
                            record["stream_key"],
                            record["algorithm"],
                            record["series_digest"],
                        ]
                    )
                    records.append(record)

    return records


def flatten_profile_result(path: Path, payload: dict[str, Any], input_dir: Path) -> list[dict[str, Any]]:
    provenance = provenance_from_path(path, input_dir)
    family = measurement_family(provenance, "workload_profile")
    run_metadata = {
        key: strip_service_paths(copy.deepcopy(value))
        for key, value in payload.items()
        if key != "profiles"
    }
    records: list[dict[str, Any]] = []

    for profile in payload.get("profiles", []):
        if not isinstance(profile, dict):
            continue
        rows = profile.get("rows", [])
        if not isinstance(rows, list):
            rows = []
        metrics = profile.get("metrics", {})
        record: dict[str, Any] = {
            "measurement_type": "workload_profile",
            "measurement_family": family,
            "provenance": provenance,
            "profile_run_metadata": copy.deepcopy(run_metadata),
            "profile": copy.deepcopy(profile),
            "profile_digest": digest_payload(profile),
            "profile_series_digest": digest_payload(rows),
            "profile_series": copy.deepcopy(rows),
            "algorithm": profile.get("algorithm"),
            "variant": profile.get("variant", "full"),
            "method": profile.get("method"),
            "mode": profile.get("mode"),
            "dataset": profile.get("dataset"),
            "base_dataset": profile.get("base_dataset") or profile.get("dataset"),
            "batch_strategy": str(profile.get("batch_strategy")),
            "feature_mode": profile.get("feature_mode"),
            "iterations": profile.get("baseline_iter"),
            "subcoms_depth": profile.get("smart_depth"),
            "radius": profile.get("smart_radius"),
            "aggregation": profile.get("aggregation_mode"),
            "device": profile.get("device"),
            "updates_profiled": profile.get("updates_profiled"),
            "total_profiled_time": profile.get("total_profiled_time"),
            "peak_cuda_allocated_mb": profile.get("peak_cuda_allocated_mb"),
            "peak_rss_mb": profile.get("peak_rss_mb"),
            "final_modularity": numeric(metrics.get("Final modularity")),
            "final_nmi": numeric(metrics.get("NMI")),
        }
        record["run_key"] = run_key(record, include_machine=False)
        record["measurement_id"] = stable_digest(
            [
                provenance.get("collection"),
                provenance.get("series"),
                provenance.get("task"),
                provenance.get("artifact"),
                record["algorithm"],
                record["dataset"],
                record["batch_strategy"],
                record["profile_digest"],
            ]
        )
        records.append(record)

    return records


def flatten_cut_metrics(path: Path, payload: dict[str, Any], input_dir: Path) -> list[dict[str, Any]]:
    provenance = provenance_from_path(path, input_dir)
    run_metadata = {
        key: strip_service_paths(copy.deepcopy(value))
        for key, value in payload.items()
        if key != "records"
    }
    records: list[dict[str, Any]] = []
    for index, row in enumerate(payload.get("records", [])):
        if not isinstance(row, dict):
            continue
        record = copy.deepcopy(row)
        record.update(
            {
                "measurement_type": "cut_metrics",
                "measurement_family": measurement_family(provenance, "cut_metrics"),
                "provenance": provenance,
                "cut_metrics_run_metadata": run_metadata,
                "cut_metrics": copy.deepcopy(row.get("cut_metrics", {})),
                "cut_metrics_digest": digest_payload(row),
            }
        )
        record["algorithm"] = "leidenalg"
        record["batch_strategy"] = str(record.get("batch"))
        record["base_dataset"] = record.get("dataset")
        record["updates"] = int(str(record.get("batch_strategy")).split(":")[-1])
        record["measurement_id"] = stable_digest(
            [
                provenance.get("collection"),
                provenance.get("series"),
                provenance.get("task"),
                index,
                record["dataset"],
                record["mode"],
                record["cut_metrics_digest"],
            ]
        )
        records.append(record)
    return records


def is_manifest_payload(path: Path, payload: Any) -> bool:
    if path.name.startswith("manifest"):
        return True
    return isinstance(payload, dict) and any(
        key in payload for key in ["selected_streams", "attempted_runs", "successful_runs", "outputs"]
    )


def is_cut_metrics_payload(payload: Any) -> bool:
    return (
        isinstance(payload, dict)
        and str(payload.get("group", "")).startswith("leiden_cut_metrics")
        and isinstance(payload.get("records"), list)
    )


def scan_current_sources(input_dir: Path, output_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    source_artifacts: list[dict[str, Any]] = []
    for path in sorted(input_dir.glob("paper_icdm_*/*/**/*.json")):
        if output_dir in path.parents:
            continue
        payload = load_json(path)
        provenance = provenance_from_path(path, input_dir)
        artifact_meta = {
            "collection": provenance["collection"],
            "series": provenance["series"],
            "task": provenance["task"],
            "artifact": provenance["artifact"],
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }
        if is_profile_file(payload):
            artifact_meta["artifact_type"] = "workload_profile"
            new_records = flatten_profile_result(path, payload, input_dir)
        elif is_cut_metrics_payload(payload):
            artifact_meta["artifact_type"] = "cut_metrics"
            new_records = flatten_cut_metrics(path, payload, input_dir)
        elif is_manifest_payload(path, payload):
            continue
        elif isinstance(payload, dict):
            artifact_meta["artifact_type"] = "launch_result"
            new_records = flatten_launch_result(path, payload, input_dir)
        else:
            artifact_meta["artifact_type"] = type(payload).__name__
            new_records = []
        artifact_meta["record_count"] = len(new_records)
        source_artifacts.append(artifact_meta)
        records.extend(new_records)

    return records, source_artifacts


def normalize_existing_record(record: dict[str, Any], group: str | None = None) -> dict[str, Any]:
    record = copy.deepcopy(record)
    preserved_family = record.get("measurement_family")
    provenance = copy.deepcopy(record.get("provenance")) if isinstance(record.get("provenance"), dict) else None
    if provenance is None:
        provenance = provenance_from_legacy_record(record)

    previous_id = record.get("bundle_record_id")
    for key in OLD_SOURCE_KEYS | {"bundle_record_id"}:
        record.pop(key, None)
    if previous_id is not None:
        record["previous_bundle_record_id"] = previous_id
    if group is not None:
        record["legacy_groups"] = sorted(set(record.get("legacy_groups", []) + [group]))
    record["provenance"] = provenance
    measurement_type = str(record.get("measurement_type", "experiment"))
    if preserved_family in PRESERVED_EXISTING_FAMILIES:
        record["measurement_family"] = preserved_family
    else:
        record["measurement_family"] = measurement_family(provenance, measurement_type, record)

    if measurement_type == "experiment":
        record["series_digest"] = record.get("series_digest") or digest_payload(record.get("series", []))
        record["stream_key"] = logical_stream_key(record)
        record["stream_id"] = stable_digest(record["stream_key"].split("|"))
        record["run_key"] = run_key(record, include_machine=False)
        record["run_key_with_machine"] = run_key(record, include_machine=True)
        record["measurement_id"] = stable_digest(
            [
                record["stream_key"],
                record.get("algorithm"),
                record["series_digest"],
            ]
        )
    elif measurement_type == "workload_profile":
        record["profile_series_digest"] = record.get("profile_series_digest") or digest_payload(
            record.get("profile_series", [])
        )
        record["profile_digest"] = record.get("profile_digest") or digest_payload(
            record.get("profile") or record.get("profile_series", [])
        )
        record["run_key"] = run_key(record, include_machine=False)
        record["measurement_id"] = stable_digest(
            [
                provenance.get("collection"),
                provenance.get("series"),
                provenance.get("task"),
                provenance.get("artifact"),
                record.get("algorithm"),
                record.get("dataset"),
                record.get("batch_strategy"),
                record["profile_series_digest"],
            ]
        )
    return record


def load_committed_preserved_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for group, path in COMMITTED_PRESERVED_FILES:
        proc = subprocess.run(
            ["git", "show", f"HEAD:{path.as_posix()}"],
            cwd=REPO_ROOT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if proc.returncode != 0:
            continue
        payload = json.loads(proc.stdout)
        for row in payload.get("records", []):
            if isinstance(row, dict):
                records.append(normalize_existing_record(row, group))
    return records


def load_existing_records(output_dir: Path) -> list[dict[str, Any]]:
    measurement_dir = output_dir / "measurements"
    if not measurement_dir.exists():
        return load_committed_preserved_records()

    existing_type_files = [
        measurement_dir / meta["filename"]
        for meta in TYPE_OUTPUTS.values()
    ]
    if any(path.exists() for path in existing_type_files):
        records: list[dict[str, Any]] = []
        for path in existing_type_files:
            if not path.exists():
                continue
            payload = load_json(path)
            batch = payload.get("records", [])
            if isinstance(batch, list):
                for record in batch:
                    if not isinstance(record, dict):
                        continue
                    provenance = record.get("provenance", {})
                    family = record.get("measurement_family")
                    if (
                        family in PRESERVED_EXISTING_FAMILIES
                        or (
                        isinstance(provenance, dict)
                        and provenance.get("collection") == "paper_icdm_reviewer_followup"
                        and provenance.get("series") == 12
                        )
                    ):
                        records.append(normalize_existing_record(record))
        seen_families = {record.get("measurement_family") for record in records}
        if not PRESERVED_EXISTING_FAMILIES.issubset(seen_families):
            records.extend(load_committed_preserved_records())
        return records

    records: list[dict[str, Any]] = []
    for path in sorted(measurement_dir.glob("*.json")):
        if path.name in {"summary_by_run.json"} | {
            str(meta["filename"]) for meta in TYPE_OUTPUTS.values()
        }:
            continue
        payload = load_json(path)
        group = payload.get("group") if isinstance(payload, dict) else None
        if group not in PRESERVED_GROUPS:
            continue
        for row in payload.get("records", []):
            if not isinstance(row, dict):
                continue
            records.append(normalize_existing_record(row, group))
    seen_families = {record.get("measurement_family") for record in records}
    if not PRESERVED_EXISTING_FAMILIES.issubset(seen_families):
        records.extend(load_committed_preserved_records())
    return records


def is_legacy_compatibility_record(record: dict[str, Any]) -> bool:
    if record.get("measurement_type") != "experiment":
        return False
    method = record.get("method")
    if method not in LEGACY_COMPATIBILITY_ALGORITHMS:
        return False
    if record.get("base_dataset") not in ARTICLE_DATASETS:
        return False
    if str(record.get("batch_strategy")) != "999:10":
        return False
    if record.get("final_nmi") is None:
        return False

    algorithms = LEGACY_COMPATIBILITY_ALGORITHMS[str(method)]
    return record.get("algorithm") in {algorithms["baseline"], algorithms["local"]}


def normalize_legacy_compatibility_record(
    record: dict[str, Any],
    source_group: str,
) -> dict[str, Any]:
    record = copy.deepcopy(record)
    for key in OLD_SOURCE_KEYS | {"bundle_record_id"}:
        record.pop(key, None)

    record["provenance"] = {
        "collection": "article_compatibility_baselines",
        "series": None,
        "series_name": None,
        "task": "compatibility_baselines",
        "task_index": None,
        "task_name": "compatibility_baselines",
        "repeat": None,
        "artifact": source_group,
    }
    record["measurement_family"] = "article_compatibility_baselines"
    record["series_digest"] = record.get("series_digest") or digest_payload(record.get("series", []))
    record["stream_key"] = logical_stream_key(record)
    record["stream_id"] = stable_digest(record["stream_key"].split("|"))
    record["run_key"] = run_key(record, include_machine=False)
    record["run_key_with_machine"] = run_key(record, include_machine=True)
    record["measurement_id"] = stable_digest(
        [
            record["stream_key"],
            record.get("algorithm"),
            record["series_digest"],
        ]
    )
    return finite(strip_service_paths(record))


def load_legacy_compatibility_records(input_dir: Path) -> list[dict[str, Any]]:
    measurement_dir = input_dir / "icdm-2026-0" / "measurements"
    records: list[dict[str, Any]] = []
    for source_group in ["real_graph_measurements", "legacy_measurements"]:
        path = measurement_dir / f"{source_group}.json"
        if not path.exists():
            continue
        payload = load_json(path)
        for row in payload.get("records", []):
            if not isinstance(row, dict):
                continue
            if not is_legacy_compatibility_record(row):
                continue
            records.append(normalize_legacy_compatibility_record(row, source_group))
    return records


def normalize_legacy_neighborhood_record(record: dict[str, Any]) -> dict[str, Any]:
    record = copy.deepcopy(record)
    for key in OLD_SOURCE_KEYS | {"bundle_record_id"}:
        record.pop(key, None)

    record["measurement_family"] = "article_neighborhood_growth"
    record["provenance"] = {
        "collection": "article_neighborhood_growth",
        "series": None,
        "series_name": None,
        "task": "article_neighborhood_growth",
        "task_index": None,
        "task_name": "article_neighborhood_growth",
        "repeat": None,
        "artifact": "neighborhood_measurements",
    }
    record["measurement_id"] = stable_digest(
        [
            "article_neighborhood_growth",
            record.get("base_dataset"),
            record.get("batch_strategy"),
            digest_payload(record.get("neighborhood_series", [])),
        ]
    )
    return finite(strip_service_paths(record))


def load_legacy_neighborhood_records(input_dir: Path) -> list[dict[str, Any]]:
    path = input_dir / "icdm-2026-0" / "measurements" / "neighborhood_measurements.json"
    if not path.exists():
        return []
    payload = load_json(path)
    records: list[dict[str, Any]] = []
    for row in payload.get("records", []):
        if not isinstance(row, dict):
            continue
        if row.get("measurement_type") != "neighborhood":
            continue
        if row.get("base_dataset") not in ARTICLE_DATASETS:
            continue
        if str(row.get("batch_strategy")) != "999:10":
            continue
        records.append(normalize_legacy_neighborhood_record(row))
    return records


def dedupe_key(record: dict[str, Any]) -> tuple[Any, ...]:
    provenance = record.get("provenance", {})
    if record.get("measurement_type") == "experiment":
        return (
            "experiment",
            provenance.get("collection"),
            provenance.get("series"),
            provenance.get("task"),
            provenance.get("artifact"),
            record.get("algorithm"),
            record.get("dataset"),
            record.get("machine"),
            record.get("batch_strategy"),
            record.get("series_digest") or digest_payload(record.get("series", [])),
        )
    if record.get("measurement_type") == "workload_profile":
        return (
            "workload_profile",
            provenance.get("collection"),
            provenance.get("series"),
            provenance.get("task"),
            provenance.get("artifact"),
            record.get("algorithm"),
            record.get("dataset"),
            record.get("batch_strategy"),
            record.get("profile_series_digest") or digest_payload(record.get("profile_series", [])),
        )
    if record.get("measurement_type") == "neighborhood":
        return (
            "neighborhood",
            provenance.get("collection"),
            provenance.get("task"),
            provenance.get("artifact"),
            record.get("base_dataset"),
            record.get("batch_strategy"),
            digest_payload(record.get("neighborhood_series", [])),
        )
    return (
        record.get("measurement_type"),
        provenance.get("collection"),
        provenance.get("series"),
        provenance.get("task"),
        provenance.get("artifact"),
        record.get("measurement_id"),
    )


def merge_records(current: list[dict[str, Any]], existing: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    merged: list[dict[str, Any]] = []
    seen: dict[tuple[Any, ...], dict[str, Any]] = {}
    duplicates: list[dict[str, Any]] = []

    for origin, batch in (("current_source", current), ("previous_bundle", existing)):
        for record in batch:
            key = dedupe_key(record)
            if key in seen:
                kept = seen[key]
                legacy_groups = set(kept.get("legacy_groups", []))
                legacy_groups.update(record.get("legacy_groups", []))
                if legacy_groups:
                    kept["legacy_groups"] = sorted(legacy_groups)
                duplicates.append(
                    {
                        "dropped_origin": origin,
                        "kept_measurement_id": kept.get("measurement_id"),
                        "dropped_measurement_id": record.get("measurement_id"),
                        "measurement_type": record.get("measurement_type"),
                    }
                )
                continue
            record = finite(record)
            seen[key] = record
            merged.append(record)

    for index, record in enumerate(merged):
        record["bundle_record_id"] = f"measurement-{index:06d}"
    return merged, duplicates


def numeric_values(records: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for record in records:
        value = numeric(record.get(key))
        if value is not None:
            values.append(value)
    return values


def summarize_by_run(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("measurement_type") == "experiment" and record.get("run_key"):
            groups[str(record["run_key"])].append(record)

    summary: list[dict[str, Any]] = []
    for key, items in sorted(groups.items()):
        first = items[0]
        times = numeric_values(items, "total_time")
        modularities = numeric_values(items, "final_modularity")
        nmis = numeric_values(items, "final_nmi")
        summary.append(
            {
                "run_key": key,
                "measurement_type": "experiment",
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
                "records": len(items),
                "families": sorted({str(item.get("measurement_family")) for item in items}),
                "collections": sorted({str(item.get("provenance", {}).get("collection")) for item in items}),
                "bundle_record_ids": [item["bundle_record_id"] for item in items],
                "time_mean": mean(times) if times else None,
                "time_std": stdev(times) if len(times) > 1 else 0.0 if times else None,
                "modularity_mean": mean(modularities) if modularities else None,
                "modularity_std": stdev(modularities) if len(modularities) > 1 else 0.0 if modularities else None,
                "nmi_mean": mean(nmis) if nmis else None,
                "nmi_std": stdev(nmis) if len(nmis) > 1 else 0.0 if nmis else None,
            }
        )
    return summary


def collection_summary(source_artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for artifact in source_artifacts:
        grouped[(artifact["collection"], artifact["series"])].append(artifact)
    return [
        {
            "collection": collection,
            "series": series,
            "json_artifacts": len(items),
            "records": sum(int(item.get("record_count", 0)) for item in items),
            "artifact_types": dict(sorted(Counter(str(item.get("artifact_type")) for item in items).items())),
        }
        for (collection, series), items in sorted(grouped.items())
    ]


def publish_record(record: dict[str, Any]) -> dict[str, Any]:
    """Remove packaging-only keys while preserving measured payloads."""
    cleaned = copy.deepcopy(record)
    for key in PUBLISH_DROP_KEYS:
        cleaned.pop(key, None)
    if isinstance(cleaned.get("profile"), dict):
        cleaned["profile"].pop("updated_at", None)
    return cleaned


def has_record(
    records: list[dict[str, Any]],
    *,
    method: str,
    algorithm: str,
    dataset: str,
    batch_strategy: str,
) -> bool:
    return any(
        record.get("measurement_type") == "experiment"
        and record.get("method") == method
        and record.get("algorithm") == algorithm
        and record.get("base_dataset") == dataset
        and str(record.get("batch_strategy")) == batch_strategy
        and record.get("final_modularity") is not None
        and record.get("total_time") is not None
        for record in records
    )


def has_neighborhood_record(records: list[dict[str, Any]], dataset: str) -> bool:
    return any(
        record.get("measurement_type") == "neighborhood"
        and record.get("base_dataset") == dataset
        and str(record.get("batch_strategy")) == "999:10"
        for record in records
    )


def coverage_audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    core_requirements: list[dict[str, str]] = []
    for dataset in sorted(ARTICLE_DATASETS):
        core_requirements.extend(
            [
                {
                    "method": "leidenalg",
                    "algorithm": "leidenalg-naive",
                    "dataset": dataset,
                    "batch_strategy": "999:10",
                },
                {
                    "method": "leidenalg",
                    "algorithm": "leidenalg-L:3-r:1-gpu",
                    "dataset": dataset,
                    "batch_strategy": "999:10",
                },
                {
                    "method": "dfleiden",
                    "algorithm": "dfleiden-dynamic",
                    "dataset": dataset,
                    "batch_strategy": "999:10",
                },
                {
                    "method": "dfleiden",
                    "algorithm": "dfleiden-L:3-r:1-gpu",
                    "dataset": dataset,
                    "batch_strategy": "999:10",
                },
                {
                    "method": "s2cag",
                    "algorithm": "s2cag-i:10-naive-feat:dataset",
                    "dataset": dataset,
                    "batch_strategy": "999:10",
                },
                {
                    "method": "s2cag",
                    "algorithm": "s2cag-i:10-L:3-r:1-agg:norm-gpu-feat:dataset",
                    "dataset": dataset,
                    "batch_strategy": "999:10",
                },
            ]
        )
    for dataset in ["dyn_pubmed", "arxivmath"]:
        for method, baseline, local in [
            ("leidenalg", "leidenalg-naive", "leidenalg-L:3-r:1-gpu"),
            ("dfleiden", "dfleiden-dynamic", "dfleiden-L:3-r:1-gpu"),
        ]:
            core_requirements.extend(
                [
                    {
                        "method": method,
                        "algorithm": baseline,
                        "dataset": dataset,
                        "batch_strategy": "9:500",
                    },
                    {
                        "method": method,
                        "algorithm": local,
                        "dataset": dataset,
                        "batch_strategy": "9:500",
                    },
                ]
            )
    for dataset in ["dyn_cora", "dyn_pubmed"]:
        for feature_mode in ["dataset", "random", "onehot"]:
            core_requirements.extend(
                [
                    {
                        "method": "s2cag",
                        "algorithm": f"s2cag-i:10-naive-feat:{feature_mode}",
                        "dataset": dataset,
                        "batch_strategy": "999:100",
                    },
                    {
                        "method": "s2cag",
                        "algorithm": f"s2cag-i:10-L:3-r:1-agg:norm-gpu-feat:{feature_mode}",
                        "dataset": dataset,
                        "batch_strategy": "999:100",
                    },
                ]
            )

    missing_core = [
        item
        for item in core_requirements
        if not has_record(records, **item)
    ]

    legacy_missing: list[dict[str, str]] = []
    expected_oom_absences: list[dict[str, str]] = []
    for method, algorithms in LEGACY_COMPATIBILITY_ALGORITHMS.items():
        for dataset in sorted(ARTICLE_DATASETS):
            for role in ["baseline", "local"]:
                item = {
                    "method": method,
                    "algorithm": algorithms[role],
                    "dataset": dataset,
                    "batch_strategy": "999:10",
                }
                if has_record(records, **item):
                    continue
                if role == "baseline" and (method, dataset) in LEGACY_EXPECTED_OOM_BASELINES:
                    expected_oom_absences.append(item)
                else:
                    legacy_missing.append(item)

    missing_neighborhood = [
        {"dataset": dataset, "batch_strategy": "999:10"}
        for dataset in sorted(ARTICLE_DATASETS)
        if not has_neighborhood_record(records, dataset)
    ]

    return {
        "core_methods": sorted(CORE_METHODS),
        "missing_core_article_records": missing_core,
        "legacy_compatibility_methods": sorted(LEGACY_COMPATIBILITY_ALGORITHMS),
        "missing_legacy_compatibility_records": legacy_missing,
        "expected_oom_baseline_absences": expected_oom_absences,
        "missing_article_neighborhood_records": missing_neighborhood,
    }


def legacy_import_audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    imported_experiments = [
        record
        for record in records
        if record.get("measurement_type") == "experiment"
        and record.get("measurement_family") == "article_compatibility_baselines"
    ]
    imported_neighborhood = [
        record
        for record in records
        if record.get("measurement_type") == "neighborhood"
        and record.get("measurement_family") == "article_neighborhood_growth"
    ]
    by_method = Counter(str(record.get("method")) for record in imported_experiments)
    core_imports = [
        {
            "method": record.get("method"),
            "algorithm": record.get("algorithm"),
            "dataset": record.get("base_dataset"),
            "batch_strategy": record.get("batch_strategy"),
        }
        for record in imported_experiments
        if record.get("method") in CORE_METHODS
    ]
    return {
        "imported_experiment_records": len(imported_experiments),
        "imported_experiment_records_by_method": dict(sorted(by_method.items())),
        "core_methods_not_imported_from_icdm_2026_0": not core_imports,
        "core_method_import_violations": core_imports,
        "imported_method_neutral_neighborhood_records": len(imported_neighborhood),
    }


def readme_text(manifest: dict[str, Any]) -> str:
    lines = [
        "# icdm-2026-1 Measurement Bundle",
        "",
        "This directory stores the canonical measurement data used by the ICDM 2026 article.",
        "",
        "Measurements are split by data type. Experiment time series, workload",
        "profiles, neighborhood-growth rows, and cut metrics each have a",
        "separate JSON file with full payloads. Original result file paths are",
        "not embedded in records.",
        "",
        "## Article Navigation",
        "",
    ]
    for item in manifest["article_navigation"]:
        line = f"- {item['article_item']}: `{item['data']}`"
        if "families" in item:
            families = ", ".join(f"`{family}`" for family in item["families"])
            line += f" (`measurement_family`: {families})"
        if "note" in item:
            line += f". {item['note']}"
        lines.append(line)

    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `manifest.json`: navigation from article items to data files.",
            "- `measurements/experiment_measurements.json`: experiment records with complete update series.",
            "- `measurements/workload_profiles.json`: workload-profile records with complete profile payloads.",
            "- `measurements/neighborhood_measurements.json`: neighborhood growth rows used by the empirical neighborhood table.",
            "- `measurements/cut_metrics.json`: targeted structural-quality metrics.",
            "",
            "## Record Counts",
            "",
            f"- Total unified records: `{manifest['counts']['total_records']}`",
        "",
        "## Records By Type",
        "",
        ]
    )
    for name, count in manifest["counts"]["records_by_type"].items():
        lines.append(f"- `{name}`: `{count}` records.")

    lines.extend(["", "## Records By Family", ""])
    for name, count in manifest["counts"]["records_by_family"].items():
        lines.append(f"- `{name}`: `{count}` records.")

    lines.append("")
    return "\n".join(lines)


def build_bundle(input_dir: Path, output_dir: Path, tag: str, tag_commit: str) -> dict[str, Any]:
    existing_records = load_existing_records(output_dir)
    current_records, _source_artifacts = scan_current_sources(input_dir, output_dir)
    legacy_compatibility_records = load_legacy_compatibility_records(input_dir)
    legacy_neighborhood_records = load_legacy_neighborhood_records(input_dir)
    imported_records = legacy_compatibility_records + legacy_neighborhood_records
    records, _duplicates = merge_records(current_records + imported_records, existing_records)

    measurement_dir = output_dir / "measurements"
    measurement_dir.mkdir(parents=True, exist_ok=True)
    for path in measurement_dir.glob("*.json"):
        path.unlink()

    counts_by_type = Counter(str(record.get("measurement_type")) for record in records)
    counts_by_family = Counter(str(record.get("measurement_family")) for record in records)
    coverage = coverage_audit(records)
    legacy_audit = legacy_import_audit(records)
    if coverage["missing_core_article_records"]:
        raise RuntimeError("Missing core article records in icdm-2026-1 bundle.")
    if coverage["missing_legacy_compatibility_records"]:
        raise RuntimeError("Missing non-core compatibility records in icdm-2026-1 bundle.")
    if coverage["missing_article_neighborhood_records"]:
        raise RuntimeError("Missing article neighborhood records in icdm-2026-1 bundle.")
    if legacy_audit["core_method_import_violations"]:
        raise RuntimeError("Core-method compatibility records leaked into imported non-core set.")

    measurement_meta: dict[str, dict[str, Any]] = {}
    for measurement_type, meta in TYPE_OUTPUTS.items():
        type_records = [
            record
            for record in records
            if record.get("measurement_type") == measurement_type
        ]
        if not type_records:
            continue
        output = f"measurements/{meta['filename']}"
        measurement_meta[measurement_type] = {
            "description": meta["description"],
            "measurement_entry_count": len(type_records),
            "output": output,
        }
        published_records = [publish_record(record) for record in type_records]
        write_json(
            output_dir / output,
            {
                "bundle": tag,
                "type": measurement_type,
                "description": meta["description"],
                "record_count": len(published_records),
                "records": published_records,
            },
        )
    manifest = {
        "bundle": tag,
        "description": "Canonical measurement data used by the ICDM 2026 article.",
        "measurement_files": measurement_meta,
        "article_navigation": ARTICLE_NAVIGATION,
        "counts": {
            "total_records": len(records),
            "records_by_type": dict(sorted(counts_by_type.items())),
            "records_by_family": dict(sorted(counts_by_family.items())),
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    (output_dir / "README.md").write_text(readme_text(manifest), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--tag-commit", default=None)
    args = parser.parse_args()

    tag_commit = args.tag_commit or current_commit(REPO_ROOT)
    manifest = build_bundle(args.input_dir, args.output_dir, args.tag, tag_commit)
    print(json.dumps(manifest["counts"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
