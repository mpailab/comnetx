"""Shared protocol loading, fingerprinting, and raw-result validation."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Iterable

import numpy as np


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[2]
PROTOCOL_PATH = PACKAGE_DIR / "protocol.json"
RESULTS_ROOT = PROJECT_ROOT / "results" / "ieee-access-2026-1" / "raw" / "ldleiden"
ALGORITHM_KEY = "ldleiden-dynamic"

def measurement_source_files() -> tuple[Path, ...]:
    """Discover the complete runtime source set on every identity check."""

    return tuple(sorted({
        *(PROJECT_ROOT / "src").rglob("*.py"),
        PROJECT_ROOT / "scripts" / "launch.py",
        PROJECT_ROOT / "scripts" / "paper" / "collect_hardware_info.py",
        *(PACKAGE_DIR.glob("*.py")),
    }))


SOURCE_FILES = measurement_source_files()


class ValidationError(RuntimeError):
    """Raised when a campaign cannot support the pre-registered comparison."""


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


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


def sha256_json(payload: Any) -> str:
    """Hash a JSON-compatible identity payload in one canonical encoding."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonicalize_partition(partition: np.ndarray) -> np.ndarray:
    """Return min-original-vertex labels for one flat partition."""

    labels = np.asarray(partition)
    if labels.ndim != 1 or labels.size == 0:
        raise ValidationError("partition must be a non-empty rank-one array")
    if labels.dtype.kind not in {"i", "u"}:
        raise ValidationError("partition labels must be integral")
    canonical = np.empty(labels.shape, dtype=np.int64)
    groups: dict[int, list[int]] = {}
    for vertex, raw_label in enumerate(labels.astype(np.int64, copy=False)):
        groups.setdefault(int(raw_label), []).append(vertex)
    for vertices in groups.values():
        canonical[vertices] = min(vertices)
    return canonical


def partition_sha256(partition: np.ndarray) -> str:
    """Hash a partition relation after deterministic canonicalization."""

    canonical = canonicalize_partition(partition).astype("<i8", copy=False)
    digest = hashlib.sha256()
    digest.update(f"shape={canonical.shape};dtype=int64;".encode("ascii"))
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def hierarchy_is_nested(hierarchy: np.ndarray) -> bool:
    """Check fine-to-coarse refinement, allowing equal adjacent levels."""

    labels = np.asarray(hierarchy)
    if labels.ndim != 2 or labels.shape[0] < 1 or labels.shape[1] < 1:
        return False
    for level in range(labels.shape[0] - 1):
        fine = labels[level]
        coarse = labels[level + 1]
        parent_by_child: dict[int, int] = {}
        for child, parent in zip(fine.tolist(), coarse.tolist()):
            child = int(child)
            parent = int(parent)
            existing = parent_by_child.setdefault(child, parent)
            if existing != parent:
                return False
    return True


def validate_partition_semantics(partition: np.ndarray) -> dict[str, Any]:
    """Validate canonical labels and optional fine-to-coarse nesting."""

    labels = np.asarray(partition)
    if labels.ndim == 1:
        canonical = canonicalize_partition(labels)
        if not np.array_equal(labels.astype(np.int64, copy=False), canonical):
            raise ValidationError("bootstrap partition labels are not canonical")
        return {
            "rank": 1,
            "levels": 1,
            "vertices": int(labels.size),
            "nested": True,
            "level_zero_sha256": partition_sha256(labels),
        }
    if labels.ndim != 2 or labels.shape[0] < 1 or labels.shape[1] < 1:
        raise ValidationError("bootstrap hierarchy must be a non-empty rank-two array")
    if labels.dtype.kind not in {"i", "u"}:
        raise ValidationError("bootstrap hierarchy labels must be integral")
    for level, row in enumerate(labels):
        canonical = canonicalize_partition(row)
        if not np.array_equal(row.astype(np.int64, copy=False), canonical):
            raise ValidationError(
                f"bootstrap hierarchy level {level} is not canonical"
            )
    if not hierarchy_is_nested(labels):
        raise ValidationError("bootstrap hierarchy is not nested")
    return {
        "rank": 2,
        "levels": int(labels.shape[0]),
        "vertices": int(labels.shape[1]),
        "nested": True,
        "level_zero_sha256": partition_sha256(labels[0]),
    }


def validate_bootstrap_cache_file(path: Path) -> dict[str, Any]:
    """Validate one immutable Leiden bootstrap cache and its flat level zero."""

    if not path.is_file():
        raise ValidationError(f"bootstrap cache is missing: {path}")
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != {"partition", "mod"}:
            raise ValidationError(f"{path.name}: unexpected bootstrap payload fields")
        partition = np.asarray(payload["partition"])
        modularity = float(payload["mod"])
    if not math.isfinite(modularity) or not -1.0 <= modularity <= 1.0:
        raise ValidationError(f"{path.name}: invalid bootstrap modularity")
    semantics = validate_partition_semantics(partition)
    if semantics["rank"] != 1 or semantics["levels"] != 1:
        raise ValidationError(
            f"{path.name}: native LD-Leiden requires one flat level-zero bootstrap"
        )
    return {
        **semantics,
        "modularity": modularity,
        "file_sha256": sha256_file(path),
    }


def bootstrap_cache_reference(
    cache_dir: Path,
    cache_hashes: dict[str, str],
    dataset: str,
    batch_strategy: str,
    observed_reference: str | None = None,
) -> dict[str, Any]:
    """Resolve and semantically attest one dataset/initial-snapshot cache."""

    initial_batch = batch_strategy.split(":", 1)[0]
    suffix = f"_b:{initial_batch}_by_leidenalg.npz"
    candidates = [
        (filename, digest)
        for filename, digest in cache_hashes.items()
        if filename in {f"{dataset}{suffix}", f"{dataset}-sym{suffix}"}
    ]
    if len(candidates) != 1:
        raise ValidationError(
            f"expected one cached Leiden bootstrap for {dataset}/{batch_strategy}, "
            f"got {[name for name, _ in candidates]}"
        )
    filename, recorded_digest = candidates[0]
    path = cache_dir / filename
    semantics = validate_bootstrap_cache_file(path)
    if semantics["file_sha256"] != recorded_digest:
        raise ValidationError(f"recorded bootstrap digest differs: {filename}")
    if (
        observed_reference is not None
        and semantics["level_zero_sha256"] != observed_reference
    ):
        raise ValidationError(
            f"{dataset}: launcher bootstrap reference differs from cached level zero"
        )
    return {"filename": filename, **semantics}


def load_protocol() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    validate_protocol_definition(protocol)
    return protocol


def phase_map(protocol: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {phase["id"]: phase for phase in protocol["phases"]}


def cli_phase_map(protocol: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {phase["cli_name"]: phase for phase in protocol["phases"]}


def config_path(phase: dict[str, Any]) -> Path:
    return PACKAGE_DIR / phase["config"]


def validate_protocol_definition(protocol: dict[str, Any]) -> None:
    if protocol.get("schema") != "comnetx-ieee-access-ldleiden-protocol-v1":
        raise ValidationError("unexpected protocol schema")
    phases = protocol.get("phases")
    if not isinstance(phases, list) or len(phases) != 3:
        raise ValidationError("the protocol must contain exactly three phases")

    by_cli = {phase.get("cli_name"): phase for phase in phases}
    if set(by_cli) != {"smoke", "short", "long"}:
        raise ValidationError("expected smoke, short, and long phases")
    if len({phase.get("id") for phase in phases}) != len(phases):
        raise ValidationError("phase ids must be unique")

    smoke = by_cli["smoke"]
    short = by_cli["short"]
    long = by_cli["long"]
    if smoke.get("included_in_analysis") is not False or smoke.get("repetitions") != 1:
        raise ValidationError("the one smoke run must be excluded from analysis")
    if short.get("repetitions") != 5:
        raise ValidationError("short evidence requires five separate measured repetitions")
    if long.get("repetitions") != 3:
        raise ValidationError("long evidence requires three separate measured repetitions")
    if set(short.get("datasets", [])) != {"dyn_pubmed", "arxivmath"}:
        raise ValidationError("short measured datasets changed")
    if set(long.get("datasets", [])) != {"dyn_pubmed", "arxivmath"}:
        raise ValidationError("long measured datasets changed")
    if protocol.get("common", {}).get("num_jobs") != 1:
        raise ValidationError("LD-Leiden worker count must remain j=1")
    if protocol.get("common", {}).get("num_jobs_source") != (
        "verified AlgorithmOptions default"
    ) or protocol.get("common", {}).get(
        "num_jobs_explicitly_passed_by_launcher"
    ) is not False:
        raise ValidationError(
            "j=1 must be recorded as a verified AlgorithmOptions default, "
            "not as an explicitly passed launcher option"
        )
    if protocol.get("clock_policy", {}).get("principal_field") != "optimization_time":
        raise ValidationError("principal LD-Leiden clock must be optimization_time")
    for lock in (
        "priming_partition_audit_required",
        "source_hash_locked_before_and_after_every_repetition",
        "input_content_hash_locked_before_and_after_every_repetition",
        "hardware_container_and_wheel_locked_before_and_after_every_repetition",
    ):
        if protocol.get("common", {}).get(lock) is not True:
            raise ValidationError(f"registered LD-Leiden identity gate was removed: {lock}")

    for phase in phases:
        datasets = phase.get("datasets", [])
        expected_updates = phase.get("expected_updates", {})
        if not datasets or set(datasets) != set(expected_updates):
            raise ValidationError(f"invalid dataset/update definition for {phase.get('id')}")
        path = config_path(phase)
        if not path.is_file():
            raise ValidationError(f"missing config: {path}")
        validate_config(phase, read_json(path))


def validate_config(phase: dict[str, Any], config: dict[str, Any]) -> None:
    expected_scalar = {
        "CATCH_ERRORS": False,
        "USE_TIMESTAMP_SUFFIX": False,
        "GROUND_TRUTH_METRICS": True,
        "FORCE_UNDIRECTED": True,
        "USE_GPU": False,
        "RESOLUTION": 1.0,
        "FEATURE_MODE": "dataset",
    }
    for key, expected in expected_scalar.items():
        if config.get(key) != expected:
            raise ValidationError(
                f"{phase['id']} config requires {key}={expected!r}, "
                f"got {config.get(key)!r}"
            )
    expected_lists = {
        "DATASETS": phase["datasets"],
        "BATCHES": [phase["batch_strategy"]],
        "BASELINES": ["ldleiden"],
        "MODES": ["dynamic"],
    }
    for key, expected in expected_lists.items():
        if config.get(key) != expected:
            raise ValidationError(
                f"{phase['id']} config requires {key}={expected!r}, "
                f"got {config.get(key)!r}"
            )


def protocol_fingerprint(protocol: dict[str, Any] | None = None) -> dict[str, Any]:
    protocol = load_protocol() if protocol is None else protocol
    configs = {
        phase["id"]: sha256_file(config_path(phase))
        for phase in protocol["phases"]
    }
    sources = {
        str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
        for path in measurement_source_files()
    }
    return {
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "config_sha256": configs,
        "source_sha256": sources,
    }


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{label} is not numeric: {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise ValidationError(f"{label} is not finite: {value!r}")
    return value


def _sha256_value(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValidationError(f"{label} is not a SHA-256 digest")
    return value


def _base_dataset(dataset_key: str, expected: set[str]) -> str:
    if dataset_key in expected:
        return dataset_key
    if dataset_key.endswith("-sym") and dataset_key[:-4] in expected:
        return dataset_key[:-4]
    raise ValidationError(f"unexpected dataset key: {dataset_key!r}")


def _single_machine_series(
    payload: dict[str, Any],
    dataset_key: str,
    batch_strategy: str,
) -> list[dict[str, Any]]:
    machine_map = payload[ALGORITHM_KEY][dataset_key]
    if not isinstance(machine_map, dict) or len(machine_map) != 1:
        raise ValidationError(f"{dataset_key}: expected exactly one machine entry")
    machine_payload = next(iter(machine_map.values()))
    if not isinstance(machine_payload, dict) or set(machine_payload) != {batch_strategy}:
        raise ValidationError(
            f"{dataset_key}: expected only batch strategy {batch_strategy!r}"
        )
    series = machine_payload[batch_strategy]
    if not isinstance(series, list):
        raise ValidationError(f"{dataset_key}: result series is not a list")
    return series


def validate_launcher_payload(
    payload: dict[str, Any],
    phase: dict[str, Any],
) -> dict[str, Any]:
    """Validate one launcher output and return analysis-ready run summaries."""

    if not isinstance(payload, dict) or set(payload) != {ALGORITHM_KEY}:
        raise ValidationError(
            f"expected the single algorithm key {ALGORITHM_KEY!r}, "
            f"got {sorted(payload) if isinstance(payload, dict) else type(payload)}"
        )
    dataset_map = payload[ALGORITHM_KEY]
    if not isinstance(dataset_map, dict):
        raise ValidationError("algorithm payload is not an object")

    expected = set(phase["datasets"])
    normalized = {
        _base_dataset(dataset_key, expected): dataset_key
        for dataset_key in dataset_map
    }
    if set(normalized) != expected or len(normalized) != len(dataset_map):
        raise ValidationError(
            f"{phase['id']}: expected datasets {sorted(expected)}, "
            f"got {sorted(normalized)}"
        )

    summaries: dict[str, Any] = {}
    for dataset in phase["datasets"]:
        series = _single_machine_series(
            payload,
            normalized[dataset],
            phase["batch_strategy"],
        )
        expected_updates = int(phase["expected_updates"][dataset])
        if len(series) != expected_updates:
            raise ValidationError(
                f"{dataset}: expected {expected_updates} updates, got {len(series)}"
            )

        optimization_total = 0.0
        update_total = 0.0
        end_to_end_total = 0.0
        bootstrap_references: set[str] = set()
        for index, row in enumerate(series, start=1):
            label = f"{dataset}/update-{index}"
            if row.get("timing_split_supported") is not True:
                raise ValidationError(
                    f"{label}: split timing is unavailable; install the "
                    "paper measurement LD-Leiden wheel"
                )
            if row.get("priming_audit_supported") is not True:
                raise ValidationError(
                    f"{label}: post-priming partition audit is unavailable"
                )
            if row.get("priming_partition_relation_preserved") is not True:
                raise ValidationError(
                    f"{label}: LD-Leiden priming changed the bootstrap relation"
                )
            bootstrap_digest = _sha256_value(
                row.get("bootstrap_reference_sha256"),
                f"{label}/bootstrap_reference_sha256",
            )
            post_priming_digest = _sha256_value(
                row.get("post_priming_partition_sha256"),
                f"{label}/post_priming_partition_sha256",
            )
            if bootstrap_digest != post_priming_digest:
                raise ValidationError(
                    f"{label}: post-priming partition differs from the bootstrap"
                )
            bootstrap_references.add(bootstrap_digest)
            optimization = _finite_number(
                row.get("optimization_time"), f"{label}/optimization_time"
            )
            compatibility = _finite_number(row.get("time"), f"{label}/time")
            update = _finite_number(row.get("update_time"), f"{label}/update_time")
            end_to_end = _finite_number(
                row.get("end_to_end_time"), f"{label}/end_to_end_time"
            )
            modularity = _finite_number(row.get("modularity"), f"{label}/modularity")
            if optimization < 0 or update < 0 or end_to_end < 0:
                raise ValidationError(f"{label}: timing values must be non-negative")
            if not math.isclose(compatibility, optimization, rel_tol=1e-12, abs_tol=1e-12):
                raise ValidationError(
                    f"{label}: time must remain an alias of optimization_time"
                )
            internal_total = optimization + update
            tolerance = max(1e-3, 0.05 * internal_total)
            if end_to_end + tolerance < internal_total:
                raise ValidationError(
                    f"{label}: end_to_end_time is inconsistent with split clocks"
                )
            if not -1.0 <= modularity <= 1.0:
                raise ValidationError(f"{label}: modularity is outside [-1, 1]")
            optimization_total += optimization
            update_total += update
            end_to_end_total += end_to_end

        if len(bootstrap_references) != 1:
            raise ValidationError(
                f"{dataset}: bootstrap reference changed within one run"
            )

        final = series[-1]
        final_modularity = _finite_number(
            final.get("Final modularity"), f"{dataset}/Final modularity"
        )
        final_nmi = _finite_number(final.get("NMI"), f"{dataset}/NMI")
        if not -1.0 <= final_modularity <= 1.0:
            raise ValidationError(f"{dataset}: Final modularity is outside [-1, 1]")
        if not 0.0 <= final_nmi <= 1.0:
            raise ValidationError(f"{dataset}: NMI is outside [0, 1]")

        summaries[dataset] = {
            "updates": expected_updates,
            "optimization_seconds": optimization_total,
            "update_seconds": update_total,
            "end_to_end_seconds": end_to_end_total,
            "framework_overhead_seconds": (
                end_to_end_total - optimization_total - update_total
            ),
            "final_modularity": final_modularity,
            "final_nmi": final_nmi,
            "bootstrap_reference_sha256": next(iter(bootstrap_references)),
        }
    return summaries


def validate_launcher_file(path: Path, phase: dict[str, Any]) -> dict[str, Any]:
    return validate_launcher_payload(read_json(path), phase)


def launcher_result_file(attempt_dir: Path, phase: dict[str, Any]) -> Path:
    config_name = config_path(phase).stem
    return attempt_dir / f"{config_name}_now.json"


def aggregate_run_summaries(
    rows: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    rows = list(rows)
    if not rows:
        raise ValidationError("cannot aggregate an empty result group")
    metrics = (
        "optimization_seconds",
        "update_seconds",
        "end_to_end_seconds",
        "framework_overhead_seconds",
        "final_modularity",
        "final_nmi",
    )
    aggregate: dict[str, Any] = {
        "repetitions": len(rows),
        "updates": rows[0]["updates"],
    }
    if any(row["updates"] != aggregate["updates"] for row in rows):
        raise ValidationError("update count changed across repetitions")
    for metric in metrics:
        values = [float(row[metric]) for row in rows]
        aggregate[metric] = {
            "mean": fmean(values),
            "sample_sd": stdev(values) if len(values) > 1 else None,
            "values": values,
        }
    return aggregate
