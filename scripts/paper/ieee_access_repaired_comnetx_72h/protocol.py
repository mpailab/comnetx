"""Shared definitions and validators for repaired-ComNetX reruns."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Iterable

import numpy as np


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[2]
PROTOCOL_PATH = PACKAGE_DIR / "protocol.json"
RESULTS_ROOT = (
    PROJECT_ROOT
    / "results"
    / "ieee-access-2026-1"
    / "raw"
    / "repaired-comnetx"
)

ALL_DATASETS = (
    "dyn_cora",
    "dyn_acm",
    "dyn_citeseer",
    "patent",
    "dyn_pubmed",
    "arxivmath",
)
CORE_DATASETS = ("dyn_pubmed", "arxivmath")
EXPECTED_UPDATES = {
    "999:10": {
        "dyn_cora": 10,
        "dyn_acm": 10,
        "dyn_citeseer": 8,
        "patent": 10,
        "dyn_pubmed": 10,
        "arxivmath": 10,
    },
    "999:50": {
        "dyn_cora": 10,
        "dyn_pubmed": 44,
        "arxivmath": 50,
    },
    "999:100": {"dyn_cora": 10, "dyn_pubmed": 44},
    "9:500": {"dyn_pubmed": 500, "arxivmath": 500},
}

SOURCE_FILES = tuple(sorted({
    *(PROJECT_ROOT / "src").rglob("*.py"),
    PROJECT_ROOT / "src" / "optimizer.py",
    PROJECT_ROOT / "src" / "launcher.py",
    PROJECT_ROOT / "src" / "sparse.py",
    PROJECT_ROOT / "src" / "datasets.py",
    PROJECT_ROOT / "src" / "metrics.py",
    PROJECT_ROOT / "src" / "baselines" / "dgc.py",
    PROJECT_ROOT / "src" / "baselines" / "leiden.py",
    PROJECT_ROOT / "scripts" / "launch.py",
    PROJECT_ROOT / "scripts" / "paper" / "collect_hardware_info.py",
    PROJECT_ROOT / "scripts" / "paper" / "audit_adapter_invariants.py",
    PROJECT_ROOT / "scripts" / "paper" / "audit_stream_invariants.py",
    PROJECT_ROOT / "scripts" / "paper" / "profile_smart_workload.py",
    PROJECT_ROOT / "scripts" / "paper" / "run_dsbm_stress.py",
    PROJECT_ROOT / "scripts" / "paper" / "summarize_dsbm_streams.py",
    PROJECT_ROOT / "scripts" / "paper" / "compute_leiden_cut_metrics.py",
    PACKAGE_DIR / "check_measurement_environment.py",
    PACKAGE_DIR / "input_manifest.py",
    PACKAGE_DIR / "protocol.py",
    PACKAGE_DIR / "run_queue.py",
    PACKAGE_DIR / "validate_campaign.py",
}))


class ValidationError(RuntimeError):
    """Raised when an artifact cannot support the registered evidence."""


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_path(relative: str) -> Path:
    return PACKAGE_DIR / relative


def load_protocol() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    validate_protocol_definition(protocol)
    return protocol


def stage_map(protocol: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {stage["id"]: stage for stage in protocol["stages"]}


def validate_protocol_definition(protocol: dict[str, Any]) -> None:
    if protocol.get("schema") != "comnetx-ieee-access-repaired-72h-v1":
        raise ValidationError("unexpected repaired-rerun protocol schema")
    stages = protocol.get("stages")
    if not isinstance(stages, list) or len(stages) != 9:
        raise ValidationError("the repaired queue must contain nine ordered queue entries")
    ids = [stage.get("id") for stage in stages]
    if len(ids) != len(set(ids)):
        raise ValidationError("stage ids must be unique")
    if [stage.get("priority") for stage in stages] != list(range(1, 10)):
        raise ValidationError("stage priorities must be the contiguous order 1..9")
    by_id = stage_map(protocol)
    for stage in stages:
        for dependency in stage.get("dependencies", []):
            if dependency not in by_id:
                raise ValidationError(f"unknown dependency {dependency!r}")
            if by_id[dependency]["priority"] >= stage["priority"]:
                raise ValidationError(f"dependency order is invalid for {stage['id']}")
        paths = []
        if stage.get("config"):
            paths.append(stage["config"])
        paths.extend(stage.get("configs", []))
        for relative in paths:
            path = config_path(relative)
            if not path.is_file():
                raise ValidationError(f"missing registered config: {path}")
            validate_launcher_config(path, read_json(path))

    if protocol.get("common", {}).get("smart_depth") != 3:
        raise ValidationError("the core repaired protocol requires L=3")
    if (
        protocol.get("common", {}).get("initial_hierarchy_cache_schema")
        != "parent_quotient_v1"
    ):
        raise ValidationError("the repaired hierarchy cache schema changed")
    if protocol.get("common", {}).get("production_api_must_be_acknowledged") is not True:
        raise ValidationError("production API acknowledgment gate was removed")
    for lock in (
        "source_hash_locked_per_campaign",
        "data_hash_locked_per_campaign",
        "hardware_and_environment_locked_per_campaign",
    ):
        if protocol.get("common", {}).get(lock) is not True:
            raise ValidationError(f"campaign identity lock was removed: {lock}")
    if set(protocol.get("claim_mapping", {})) != {
        "RQ1",
        "RQ2",
        "RQ3",
        "RQ4",
        "RQ5",
    }:
        raise ValidationError("the five-RQ evidence mapping is incomplete")
    dsbm = by_id["stage6_dsbm"]
    if dsbm.get("estimated_gpu_hours") != 26.95:
        raise ValidationError("the registered DSBM cost estimate changed")
    if dsbm.get("minimum_hours_remaining", 0) < 30:
        raise ValidationError("DSBM queue must retain at least a 30-hour start gate")


def validate_launcher_config(path: Path, config: dict[str, Any]) -> None:
    required = {
        "CATCH_ERRORS": False,
        "USE_TIMESTAMP_SUFFIX": False,
        "GROUND_TRUTH_METRICS": True,
        "USE_GPU": True,
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValidationError(f"{path.name}: expected {key}={expected!r}")
    if not config.get("DATASETS") or not config.get("BATCHES"):
        raise ValidationError(f"{path.name}: datasets and batches must be explicit")
    if not config.get("BASELINES") or not config.get("MODES"):
        raise ValidationError(f"{path.name}: methods and modes must be explicit")
    if "smart" in config["MODES"]:
        grid = config.get("SMART_PARAMS_GRID", {})
        if not grid.get("smart_subcoms_depth") or not grid.get(
            "smart_neighborhood_step"
        ):
            raise ValidationError(f"{path.name}: smart L/r grid is missing")


def protocol_fingerprint(protocol: dict[str, Any] | None = None) -> dict[str, Any]:
    protocol = load_protocol() if protocol is None else protocol
    configs = {}
    for stage in protocol["stages"]:
        relatives = []
        if stage.get("config"):
            relatives.append(stage["config"])
        relatives.extend(stage.get("configs", []))
        for relative in relatives:
            configs[relative] = sha256_file(config_path(relative))
    sources = {
        str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
        for path in SOURCE_FILES
    }
    return {
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "config_sha256": configs,
        "source_sha256": sources,
    }


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{label} is not numeric: {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValidationError(f"{label} is not finite: {value!r}")
    return result


def normalize_dataset(name: str, expected: Iterable[str]) -> str:
    expected = set(expected)
    if name in expected:
        return name
    if name.endswith("-sym") and name[:-4] in expected:
        return name[:-4]
    raise ValidationError(f"unexpected dataset key: {name!r}")


def flatten_launcher_payload(
    payload: dict[str, Any],
    *,
    expected_datasets: Iterable[str],
    expected_batch: str,
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    expected_datasets = tuple(expected_datasets)
    flat: dict[tuple[str, str], list[dict[str, Any]]] = {}
    if not isinstance(payload, dict) or not payload:
        raise ValidationError("launcher payload is empty or not an object")
    observed_datasets: set[str] = set()
    datasets_by_algorithm: dict[str, set[str]] = {}
    for algorithm, dataset_map in payload.items():
        if not isinstance(dataset_map, dict):
            raise ValidationError(f"{algorithm}: dataset payload is invalid")
        for stored_dataset, machine_map in dataset_map.items():
            dataset = normalize_dataset(stored_dataset, expected_datasets)
            observed_datasets.add(dataset)
            datasets_by_algorithm.setdefault(algorithm, set()).add(dataset)
            if not isinstance(machine_map, dict) or len(machine_map) != 1:
                raise ValidationError(
                    f"{algorithm}/{stored_dataset}: expected one machine entry"
                )
            machine_payload = next(iter(machine_map.values()))
            if not isinstance(machine_payload, dict) or set(machine_payload) != {
                expected_batch
            }:
                raise ValidationError(
                    f"{algorithm}/{stored_dataset}: batch key mismatch"
                )
            series = machine_payload[expected_batch]
            if not isinstance(series, list) or not series:
                raise ValidationError(
                    f"{algorithm}/{stored_dataset}: empty result series"
                )
            key = (algorithm, dataset)
            if key in flat:
                raise ValidationError(f"duplicate launcher result: {key}")
            flat[key] = series
    if observed_datasets != set(expected_datasets):
        raise ValidationError(
            f"expected datasets {sorted(expected_datasets)}, got {sorted(observed_datasets)}"
        )
    incomplete = {
        algorithm: sorted(set(expected_datasets) - datasets)
        for algorithm, datasets in datasets_by_algorithm.items()
        if datasets != set(expected_datasets)
    }
    if incomplete:
        raise ValidationError(
            f"launcher output is not an algorithm-by-dataset Cartesian grid: {incomplete}"
        )
    return flat


def validate_series(
    series: list[dict[str, Any]],
    *,
    dataset: str,
    batch: str,
) -> dict[str, float | int]:
    expected = EXPECTED_UPDATES[batch][dataset]
    if len(series) != expected:
        raise ValidationError(
            f"{dataset}/{batch}: expected {expected} updates, got {len(series)}"
        )
    total_time = 0.0
    for index, row in enumerate(series, start=1):
        time_value = _finite(row.get("time"), f"{dataset}/update-{index}/time")
        modularity = _finite(
            row.get("modularity"), f"{dataset}/update-{index}/modularity"
        )
        if time_value < 0 or not -1 <= modularity <= 1:
            raise ValidationError(f"{dataset}/update-{index}: invalid time or modularity")
        total_time += time_value
    final = series[-1]
    final_q = _finite(final.get("Final modularity"), f"{dataset}/Final modularity")
    final_nmi = _finite(final.get("NMI"), f"{dataset}/NMI")
    if not -1 <= final_q <= 1 or not 0 <= final_nmi <= 1:
        raise ValidationError(f"{dataset}: invalid final quality metrics")
    return {
        "updates": expected,
        "total_time": total_time,
        "final_modularity": final_q,
        "final_nmi": final_nmi,
    }


def aggregate(rows: Iterable[dict[str, float | int]]) -> dict[str, Any]:
    rows = list(rows)
    if not rows:
        raise ValidationError("cannot aggregate empty repetitions")
    result: dict[str, Any] = {"repetitions": len(rows)}
    for metric in ("total_time", "final_modularity", "final_nmi"):
        values = [float(row[metric]) for row in rows]
        result[metric] = {
            "mean": fmean(values),
            "sample_sd": stdev(values) if len(values) > 1 else None,
            "values": values,
        }
    return result


def bootstrap_cache_hashes(cache_dir: Path) -> dict[str, str]:
    return {
        path.name: sha256_file(path)
        for path in sorted(cache_dir.glob("*.npz"))
        if path.is_file()
    }


def validate_bootstrap_hash_history(campaign_dir: Path) -> dict[str, str]:
    """Require every reused bootstrap file to keep one immutable digest.

    Command metadata records a snapshot of the campaign-local bootstrap cache.
    The cache may grow as later stages introduce new depths or resolutions, but
    a filename that has already appeared must never change content.
    """
    observed: dict[str, str] = {}
    metadata_paths = sorted(
        (campaign_dir / "stages").glob("**/attempt-*/metadata.json")
    )
    for metadata_path in metadata_paths:
        metadata = read_json(metadata_path)
        if metadata.get("status") != "completed":
            continue
        hashes = metadata.get("bootstrap_cache_sha256")
        if not isinstance(hashes, dict):
            raise ValidationError(
                f"{metadata_path}: completed command lacks bootstrap hashes"
            )
        for filename, digest in hashes.items():
            if not isinstance(filename, str) or not isinstance(digest, str):
                raise ValidationError(
                    f"{metadata_path}: malformed bootstrap-cache hash entry"
                )
            previous = observed.setdefault(filename, digest)
            if previous != digest:
                raise ValidationError(
                    f"bootstrap cache changed in place: {filename}"
                )
    cache_dir = campaign_dir / "bootstrap-cache"
    for filename, digest in observed.items():
        path = cache_dir / filename
        if not path.is_file() or sha256_file(path) != digest:
            raise ValidationError(
                f"bootstrap cache differs from completed-command metadata: {filename}"
            )
    return observed


def require_bootstrap_cache(
    attempt_dir: Path,
    *,
    dataset: str,
    initial_batch: int,
    depth: int,
    resolution: float = 1.0,
) -> dict[str, str]:
    """Return the unique production-bootstrap digest recorded by a command."""
    metadata_path = attempt_dir / "metadata.json"
    metadata = read_json(metadata_path)
    hashes = metadata.get("bootstrap_cache_sha256")
    if not isinstance(hashes, dict):
        raise ValidationError(f"{metadata_path}: bootstrap hashes are missing")

    depth_suffix = (
        "" if depth == 1 else f"_d:{depth}_parent_quotient_v1"
    )
    resolution_suffix = "" if float(resolution) == 1.0 else f"_res:{resolution:g}"
    suffix = (
        f"_b:{initial_batch}_by_leidenalg{depth_suffix}"
        f"{resolution_suffix}.npz"
    )
    candidates = {
        filename: digest
        for filename, digest in hashes.items()
        if filename in {f"{dataset}{suffix}", f"{dataset}-sym{suffix}"}
    }
    if len(candidates) != 1:
        raise ValidationError(
            f"{metadata_path}: expected one {dataset} bootstrap at depth {depth} "
            f"and resolution {resolution:g}, got {sorted(candidates)}"
        )
    return candidates


def validate_bootstrap_cache_entry(
    campaign_dir: Path,
    attempt_dir: Path,
    *,
    dataset: str,
    initial_batch: int,
    depth: int,
    resolution: float = 1.0,
    include_partition: bool = False,
) -> dict[str, Any]:
    """Validate the recorded digest and semantic shape of one cache entry."""
    recorded = require_bootstrap_cache(
        attempt_dir,
        dataset=dataset,
        initial_batch=initial_batch,
        depth=depth,
        resolution=resolution,
    )
    filename, digest = next(iter(recorded.items()))
    cache_path = campaign_dir / "bootstrap-cache" / filename
    if not cache_path.is_file() or sha256_file(cache_path) != digest:
        raise ValidationError(f"bootstrap cache no longer matches metadata: {filename}")
    with np.load(cache_path, allow_pickle=False) as payload:
        if set(payload.files) not in ({"partition", "mod"}, {"partition", "mod", "schema"}):
            raise ValidationError(f"{filename}: unexpected cache payload fields")
        partition = np.asarray(payload["partition"])
        modularity = float(np.asarray(payload["mod"]).item())
        schema = (
            str(np.asarray(payload["schema"]).item())
            if "schema" in payload
            else None
        )
    if not math.isfinite(modularity) or not -1 <= modularity <= 1:
        raise ValidationError(f"{filename}: invalid bootstrap modularity")
    expected_shape = 1 if depth == 1 else 2
    if partition.ndim != expected_shape:
        raise ValidationError(f"{filename}: bootstrap partition rank mismatch")
    if depth > 1 and (
        partition.shape[0] != depth or schema != "parent_quotient_v1"
    ):
        raise ValidationError(f"{filename}: hierarchy depth/schema mismatch")
    if depth == 1 and schema is not None:
        raise ValidationError(f"{filename}: flat cache unexpectedly has a schema")
    if not np.issubdtype(partition.dtype, np.integer):
        raise ValidationError(f"{filename}: partition labels are not integers")
    result = {
        "filename": filename,
        "sha256": digest,
        "modularity": modularity,
    }
    if include_partition:
        result["partition"] = partition
    return result


def validate_paired_bootstrap(
    campaign_dir: Path,
    attempt_dir: Path,
    *,
    dataset: str,
    initial_batch: int,
    smart_depth: int = 3,
) -> dict[str, Any]:
    """Require full and smart cache entries to share one level-zero state."""
    full = validate_bootstrap_cache_entry(
        campaign_dir,
        attempt_dir,
        dataset=dataset,
        initial_batch=initial_batch,
        depth=1,
        include_partition=True,
    )
    smart = validate_bootstrap_cache_entry(
        campaign_dir,
        attempt_dir,
        dataset=dataset,
        initial_batch=initial_batch,
        depth=smart_depth,
        include_partition=True,
    )
    if not np.array_equal(full["partition"], smart["partition"][0]):
        raise ValidationError(
            f"{dataset}: full and smart runs used different level-zero bootstraps"
        )
    if not math.isclose(
        float(full["modularity"]),
        float(smart["modularity"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValidationError(
            f"{dataset}: full and smart bootstrap modularity differs"
        )
    for entry in (full, smart):
        entry.pop("partition")
    return {"full_depth_1": full, f"smart_depth_{smart_depth}": smart}


def json_file_hashes(directory: Path) -> dict[str, str]:
    return {
        str(path.relative_to(directory)): sha256_file(path)
        for path in sorted(directory.rglob("*.json"))
        if path.is_file() and path.name != "metadata.json"
    }
