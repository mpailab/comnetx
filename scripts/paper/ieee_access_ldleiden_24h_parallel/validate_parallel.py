#!/usr/bin/env python3
"""Validate the eight-shard IEEE Access LD-Leiden measurement campaign.

The parallel campaign deliberately uses one sealed LD-Leiden campaign per
measured repetition.  This validator compares the sealed identities without
requiring the container namespace to be shared, validates the campaign-local
bootstrap authority, checks that the configured CPU affinities are disjoint,
validates the single smoke gate, and aggregates the five short and three long
repetitions.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import datetime
import math
from pathlib import Path
import sys
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.paper.ieee_access_ldleiden_72h.protocol import (  # noqa: E402
    ValidationError,
    aggregate_run_summaries,
    bootstrap_cache_reference,
    launcher_result_file,
    load_protocol,
    protocol_fingerprint,
    read_json,
    sha256_file,
    sha256_json,
    validate_bootstrap_cache_file,
    validate_launcher_file,
    write_json,
)
from scripts.paper.ieee_access_ldleiden_72h.validate_results import (  # noqa: E402
    validate_window_attestation,
)
PARALLEL_SCHEMA = "comnetx-ieee-access-ldleiden-parallel-v1"
PARALLEL_REGISTRATION_SCHEMA = (
    "comnetx-ieee-access-ldleiden-parallel-manifest-registration-v1"
)
PARALLEL_REPORT_SCHEMA = "comnetx-ieee-access-ldleiden-parallel-validation-v1"
CAMPAIGN_SCHEMA = "comnetx-ieee-access-ldleiden-campaign-v1"
INPUT_SCHEMA = "comnetx-ieee-access-ldleiden-real-inputs-v1"
BOOTSTRAP_SOURCE_SCHEMA = "comnetx-ieee-access-ldleiden-bootstrap-source-v1"
SHARED_BOOTSTRAP_SCHEMA = "comnetx-ieee-access-ldleiden-shared-bootstrap-v1"
SHARED_BOOTSTRAP_REGISTRATION_SCHEMA = (
    "comnetx-ieee-access-ldleiden-shared-bootstrap-registration-v1"
)
SHARED_BOOTSTRAP_POLICY = {
    "schema": "comnetx-ieee-access-ldleiden-shared-bootstrap-policy-v1",
    "kind": "campaign-local-generated",
    "producer_shard": "01",
    "relative_path": "shared-bootstrap",
}
EVIDENCE_SCOPE = {
    "mode": "standalone-native-ldleiden",
    "paired_comnetx_validated": False,
    "matched_speedup_claim_allowed": False,
    "allowed_claims": [
        "native LD-Leiden timing and quality under the sealed protocol",
        "within-LD-Leiden repeatability across the registered shards",
    ],
}
LONG_TIMEOUT_SECONDS = 3600.0
BOOTSTRAP_REQUIREMENTS = (
    ("dyn_cora", "999:10"),
    ("dyn_acm", "999:10"),
    ("dyn_citeseer", "999:10"),
    ("patent", "999:10"),
    ("dyn_pubmed", "999:10"),
    ("arxivmath", "999:10"),
    ("dyn_pubmed", "9:500"),
    ("arxivmath", "9:500"),
)


@dataclass(frozen=True)
class ShardSpec:
    directory: str
    ordinal: int
    phase_cli: str
    phase_id: str
    repeat: int


EXPECTED_SHARDS = tuple(
    [
        ShardSpec(
            directory=f"{repeat:02d}-short-r{repeat:02d}",
            ordinal=repeat,
            phase_cli="short",
            phase_id="measured_999_10",
            repeat=repeat,
        )
        for repeat in range(1, 6)
    ]
    + [
        ShardSpec(
            directory=f"{repeat + 5:02d}-long-r{repeat:02d}",
            ordinal=repeat + 5,
            phase_cli="long",
            phase_id="measured_9_500",
            repeat=repeat,
        )
        for repeat in range(1, 4)
    ]
)


@dataclass
class ShardRecord:
    spec: ShardSpec
    directory: Path
    manifest: dict[str, Any]
    hardware: dict[str, Any]
    runtime_identity: dict[str, Any]
    input_manifest: dict[str, Any]
    cpu_affinity: tuple[int, ...]
    cpu_topology: dict[str, Any]


@dataclass
class BootstrapAuthorityRecord:
    directory: Path
    identity: dict[str, Any]
    manifest: dict[str, Any]
    plan: dict[str, Any]
    cache: Path
    entries: dict[str, dict[str, Any]]


def _require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValidationError(f"{label} must be a JSON object")
    return value


def _registered_artifact(
    campaign_dir: Path,
    manifest: dict[str, Any],
    key: str,
) -> tuple[Path, dict[str, Any]]:
    registration = manifest.get(key)
    if not isinstance(registration, dict):
        raise ValidationError(f"{campaign_dir}: missing {key} registration")
    filename = registration.get("filename")
    digest = registration.get("sha256")
    if not isinstance(filename, str) or not filename:
        raise ValidationError(f"{campaign_dir}: malformed {key} filename")
    if not isinstance(digest, str):
        raise ValidationError(f"{campaign_dir}: malformed {key} digest")
    if Path(filename).name != filename:
        raise ValidationError(f"{campaign_dir}: {key} must be a direct artifact")
    path = campaign_dir / filename
    if not path.is_file() or sha256_file(path) != digest:
        raise ValidationError(f"{campaign_dir}: registered {key} artifact changed")
    return path, _require_object(read_json(path), f"{campaign_dir}/{filename}")


def _validate_parallel_registration(root: Path, manifest_path: Path) -> str:
    registration_path = root / "parallel_manifest_registration.json"
    if not registration_path.is_file():
        raise ValidationError(
            f"missing parallel manifest registration: {registration_path}"
        )
    registration = _require_object(
        read_json(registration_path), str(registration_path)
    )
    digest = sha256_file(manifest_path)
    expected = {
        "schema": PARALLEL_REGISTRATION_SCHEMA,
        "filename": manifest_path.name,
        "sha256": digest,
    }
    if registration != expected:
        raise ValidationError(
            "parallel_manifest_registration.json does not seal the live "
            "parallel manifest"
        )
    return digest


def _validate_parallel_window(value: Any) -> dict[str, float]:
    window = _require_object(value, "parallel_manifest.window")
    required = ("budget_hours", "started_epoch", "deadline_epoch")
    normalized: dict[str, float] = {}
    for key in required:
        item = window.get(key)
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValidationError(f"parallel_manifest.window.{key} must be numeric")
        number = float(item)
        if not math.isfinite(number):
            raise ValidationError(
                f"parallel_manifest.window.{key} must be finite"
            )
        normalized[key] = number
    budget = normalized["budget_hours"]
    started = normalized["started_epoch"]
    deadline = normalized["deadline_epoch"]
    if not 0.0 < budget <= 24.0:
        raise ValidationError("parallel measurement budget must be in (0, 24] hours")
    if started <= 0.0 or deadline <= started:
        raise ValidationError("parallel measurement window has invalid epoch bounds")
    if not math.isclose(
        deadline - started,
        budget * 3600.0,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValidationError(
            "parallel measurement deadline is inconsistent with its numeric budget"
        )
    return normalized


def _parse_affinity(value: Any, label: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ValidationError(f"{label} must be a list containing one logical CPU")
    values = value
    if len(values) != 1:
        raise ValidationError(f"{label} must contain exactly one logical CPU")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in values):
        raise ValidationError(f"{label} contains a non-integral CPU id")
    if any(item < 0 for item in values):
        raise ValidationError(f"{label} contains a negative CPU id")
    return tuple(values)


def _validate_cpu_topology(
    value: Any,
    affinity: tuple[int, ...],
    label: str,
) -> dict[str, Any]:
    topology = _require_object(value, label)
    expected_keys = {
        "logical_cpu",
        "physical_package_id",
        "core_id",
        "thread_siblings_list",
    }
    if set(topology) != expected_keys:
        raise ValidationError(
            f"{label} must contain exactly {sorted(expected_keys)}, "
            f"got {sorted(topology)}"
        )
    for key in ("logical_cpu", "physical_package_id", "core_id"):
        item = topology.get(key)
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValidationError(f"{label}.{key} must be a non-negative integer")
    siblings = topology.get("thread_siblings_list")
    if not isinstance(siblings, str) or not siblings.strip():
        raise ValidationError(f"{label}.thread_siblings_list must be non-empty")
    if topology["logical_cpu"] != affinity[0]:
        raise ValidationError(f"{label}.logical_cpu differs from cpu_affinity")
    return topology


def _validate_assignment(
    campaign_dir: Path,
    manifest: dict[str, Any],
    spec: ShardSpec,
    *,
    parallel_manifest_sha256: str,
    bootstrap_authority_identity: dict[str, Any],
) -> tuple[dict[str, Any], tuple[int, ...], dict[str, Any]]:
    assignment = _require_object(
        manifest.get("parallel_shard"),
        f"{campaign_dir}/manifest.json parallel_shard",
    )
    raw_id = assignment.get("id")
    if raw_id != f"{spec.ordinal:02d}":
        raise ValidationError(
            f"{campaign_dir}: parallel shard id {raw_id!r} does not identify "
            f"{spec.directory}"
        )
    expected = {
        "phase_cli": spec.phase_cli,
        "phase_id": spec.phase_id,
        "repeat": spec.repeat,
    }
    for key, value in expected.items():
        if assignment.get(key) != value:
            raise ValidationError(
                f"{campaign_dir}: expected parallel_shard.{key}={value!r}, "
                f"got {assignment.get(key)!r}"
            )
    bootstrap_source = assignment.get("bootstrap_source")
    if bootstrap_source in (None, "", {}, []):
        raise ValidationError(f"{campaign_dir}: bootstrap_source is not recorded")
    bootstrap_source_path = campaign_dir / "bootstrap_source.json"
    if (
        not bootstrap_source_path.is_file()
        or read_json(bootstrap_source_path) != bootstrap_source
    ):
        raise ValidationError(
            f"{campaign_dir}: bootstrap_source.json differs from its sealed assignment"
        )
    affinity = _parse_affinity(
        assignment.get("cpu_affinity"),
        f"{campaign_dir}/parallel_shard.cpu_affinity",
    )
    topology = _validate_cpu_topology(
        assignment.get("cpu_topology"),
        affinity,
        f"{campaign_dir}/parallel_shard.cpu_topology",
    )
    if assignment.get("parallel_manifest_sha256") != parallel_manifest_sha256:
        raise ValidationError(
            f"{campaign_dir}: parallel manifest hash differs from registration"
        )
    if assignment.get("bootstrap_authority") != bootstrap_authority_identity:
        raise ValidationError(
            f"{campaign_dir}: shared-bootstrap authority identity differs from "
            "the campaign authority"
        )
    return assignment, affinity, topology


def _validate_backend_and_workers(
    campaign_dir: Path,
    manifest: dict[str, Any],
) -> None:
    backend = _require_object(manifest.get("backend"), f"{campaign_dir}/backend")
    expected_backend = {
        "num_jobs_verified": True,
        "num_jobs_observed": 1,
        "num_jobs_source": "AlgorithmOptions default",
        "num_jobs_explicitly_passed_by_launcher": False,
    }
    for key, expected in expected_backend.items():
        if backend.get(key) != expected:
            raise ValidationError(
                f"{campaign_dir}: backend {key} must be {expected!r}, "
                f"got {backend.get(key)!r}"
            )
    worker = _require_object(
        manifest.get("worker_policy"), f"{campaign_dir}/worker_policy"
    )
    if worker.get("num_jobs") != 1:
        raise ValidationError(f"{campaign_dir}: worker policy is not j=1")
    if worker.get("num_jobs_source") != "verified AlgorithmOptions default":
        raise ValidationError(f"{campaign_dir}: invalid worker-count provenance")
    if worker.get("num_jobs_explicitly_passed_by_launcher") is not False:
        raise ValidationError(f"{campaign_dir}: j=1 was unexpectedly passed explicitly")
    thread_environment = worker.get("thread_environment")
    if (
        not isinstance(thread_environment, dict)
        or not thread_environment
        or set(thread_environment.values()) != {"1"}
    ):
        raise ValidationError(f"{campaign_dir}: thread environment is not fixed at one")


def _runtime_without_container(runtime_identity: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(runtime_identity)
    normalized.pop("container", None)
    # The full hardware digest remains sealed inside each shard and is checked
    # by every attempt.  It is intentionally not compared across shards because
    # CPU-only LD-Leiden containers may expose different GPU subsets.
    normalized.pop("hardware_sha256", None)
    return normalized


def _comparable_hardware(hardware: dict[str, Any]) -> dict[str, Any]:
    """Return the CPU-measurement hardware fields shared by all shards."""

    required = ("platform", "cpu", "memory", "python", "packages")
    missing = [key for key in required if key not in hardware]
    if missing:
        raise ValidationError(
            f"hardware record lacks comparable fields: {', '.join(missing)}"
        )
    return {key: hardware[key] for key in required}


def _cross_shard_value(
    shards: Iterable[ShardRecord],
    label: str,
    getter: Any,
) -> tuple[Any, str]:
    values: list[tuple[str, Any, str]] = []
    for shard in shards:
        value = getter(shard)
        values.append((shard.spec.directory, value, sha256_json(value)))
    digests = {digest for _, _, digest in values}
    if len(digests) != 1:
        details = ", ".join(f"{name}={digest}" for name, _, digest in values)
        raise ValidationError(f"{label} differs across shards: {details}")
    return values[0][1], values[0][2]


def _validate_disjoint_affinity(shards: Iterable[ShardRecord]) -> None:
    owners: dict[int, str] = {}
    physical_owners: dict[tuple[int, int], str] = {}
    for shard in shards:
        for cpu in shard.cpu_affinity:
            previous = owners.setdefault(cpu, shard.spec.directory)
            if previous != shard.spec.directory:
                raise ValidationError(
                    f"CPU {cpu} is assigned to both {previous} and "
                    f"{shard.spec.directory}"
                )
        physical_core = (
            shard.cpu_topology["physical_package_id"],
            shard.cpu_topology["core_id"],
        )
        previous = physical_owners.setdefault(physical_core, shard.spec.directory)
        if previous != shard.spec.directory:
            raise ValidationError(
                f"physical CPU core {physical_core} is assigned to both {previous} "
                f"and {shard.spec.directory}"
            )
    if len(owners) != len(EXPECTED_SHARDS):
        raise ValidationError("the eight shards do not use eight logical CPUs")
    if len(physical_owners) != len(EXPECTED_SHARDS):
        raise ValidationError("the eight shards do not use eight physical CPU cores")


def _parallel_manifest_shard_names(payload: dict[str, Any]) -> set[str] | None:
    raw = payload.get("shards")
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise ValidationError("parallel_manifest.shards must be a list")
    names: set[str] = set()
    for entry in raw:
        if isinstance(entry, str):
            name = entry
        elif isinstance(entry, dict):
            name = entry.get("directory") or entry.get("name")
        else:
            name = None
        if not isinstance(name, str) or not name:
            raise ValidationError("parallel_manifest.shards contains a malformed entry")
        names.add(Path(name).name)
    return names


def _validate_parallel_assignments(payload: dict[str, Any]) -> None:
    raw = payload.get("assignments")
    if not isinstance(raw, dict):
        raise ValidationError("parallel_manifest.assignments must be an object")
    expected = {
        f"{spec.ordinal:02d}": {
            "phase_cli": spec.phase_cli,
            "phase_id": spec.phase_id,
            "repeat": spec.repeat,
            "campaign": spec.directory,
        }
        for spec in EXPECTED_SHARDS
    }
    if raw != expected:
        raise ValidationError("parallel_manifest.assignments differs from the fixed 5+3 design")


def _bootstrap_requirements() -> dict[str, str]:
    return {
        f"{dataset}/{batch_strategy.split(':', 1)[0]}": batch_strategy
        for dataset, batch_strategy in BOOTSTRAP_REQUIREMENTS
    }


def _safe_direct_filename(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or Path(value).name != value
    ):
        raise ValidationError(f"{label} must be a non-empty direct filename")
    return value


def _load_bootstrap_authority(
    root: Path,
    parallel_manifest: dict[str, Any],
    parallel_manifest_sha256: str,
) -> BootstrapAuthorityRecord:
    policy = _require_object(
        parallel_manifest.get("bootstrap_authority"),
        "parallel_manifest.bootstrap_authority",
    )
    if policy != SHARED_BOOTSTRAP_POLICY:
        raise ValidationError("parallel bootstrap-authority policy changed")
    evidence_scope = _require_object(
        parallel_manifest.get("evidence_scope"),
        "parallel_manifest.evidence_scope",
    )
    if evidence_scope != EVIDENCE_SCOPE:
        raise ValidationError("parallel evidence scope changed")

    relative_path = _safe_direct_filename(
        policy.get("relative_path"), "parallel bootstrap authority relative_path"
    )
    directory = (root / relative_path).resolve()
    if directory.parent != root or not directory.is_dir():
        raise ValidationError(
            f"shared-bootstrap authority is missing or outside the campaign: {directory}"
        )
    manifest_path = directory / "manifest.json"
    registration_path = directory / "manifest_registration.json"
    if not manifest_path.is_file() or not registration_path.is_file():
        raise ValidationError(f"shared-bootstrap authority is incomplete: {directory}")
    manifest = _require_object(read_json(manifest_path), str(manifest_path))
    registration = _require_object(
        read_json(registration_path), str(registration_path)
    )
    manifest_sha256 = sha256_file(manifest_path)
    expected_registration = {
        "schema": SHARED_BOOTSTRAP_REGISTRATION_SCHEMA,
        "filename": manifest_path.name,
        "sha256": manifest_sha256,
    }
    if registration != expected_registration:
        raise ValidationError(
            "shared-bootstrap registration does not seal its live manifest"
        )

    expected_manifest_fields = {
        "schema": SHARED_BOOTSTRAP_SCHEMA,
        "run_id": root.name,
        "parallel_manifest_sha256": parallel_manifest_sha256,
        "bootstrap_policy": SHARED_BOOTSTRAP_POLICY,
        "evidence_scope": EVIDENCE_SCOPE,
    }
    for key, expected in expected_manifest_fields.items():
        if manifest.get(key) != expected:
            raise ValidationError(
                f"shared-bootstrap manifest {key} differs from the campaign"
            )
    created_at = manifest.get("created_at_utc")
    if not isinstance(created_at, str) or not created_at:
        raise ValidationError("shared-bootstrap creation timestamp is missing")

    generation = _require_object(
        manifest.get("generation"), "shared-bootstrap manifest generation"
    )
    protocol = load_protocol()
    expected_generation = {
        "producer_shard": "01",
        "producer_campaign": EXPECTED_SHARDS[0].directory,
        "method": "leidenalg",
        "api": "launcher._compute_launch_initial_partition",
        "resolution": float(protocol["common"]["resolution"]),
        "force_undirected": bool(protocol["common"]["force_undirected"]),
        "thread_environment": {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        },
        "rng_control": "backend-default; no explicit seed is exposed",
        "reproducibility_policy": (
            "archive and reuse the sealed NPZ bytes; regeneration is not "
            "claimed to reproduce the same partition"
        ),
    }
    for key, expected in expected_generation.items():
        if generation.get(key) != expected:
            raise ValidationError(
                f"shared-bootstrap generation {key} differs: "
                f"{generation.get(key)!r} != {expected!r}"
            )
    plan_name = _safe_direct_filename(
        generation.get("plan_filename"),
        "shared-bootstrap generation plan_filename",
    )
    plan_path = directory / plan_name
    plan_sha256 = generation.get("plan_sha256")
    if (
        not isinstance(plan_sha256, str)
        or len(plan_sha256) != 64
        or not plan_path.is_file()
        or sha256_file(plan_path) != plan_sha256
    ):
        raise ValidationError("shared-bootstrap generation plan changed")
    plan = _require_object(read_json(plan_path), str(plan_path))
    requirements = _bootstrap_requirements()
    expected_plan_fields = {
        "schema": "comnetx-ieee-access-ldleiden-shared-bootstrap-plan-v1",
        "run_id": root.name,
        "parallel_manifest_sha256": parallel_manifest_sha256,
        "producer_shard": "01",
        "producer_campaign": EXPECTED_SHARDS[0].directory,
        "expected_git_sha": parallel_manifest.get("expected_git_sha"),
        "protocol_id": parallel_manifest.get("protocol_id"),
        "paths_config": parallel_manifest.get("paths_config"),
        "requirements": requirements,
    }
    for key, expected in expected_plan_fields.items():
        if plan.get(key) != expected:
            raise ValidationError(
                f"shared-bootstrap initialization plan {key} differs from "
                "the parallel campaign"
            )

    raw_entries = _require_object(
        manifest.get("entries"), "shared-bootstrap manifest entries"
    )
    if set(raw_entries) != set(requirements):
        raise ValidationError(
            "shared-bootstrap authority must contain exactly the eight fixed keys"
        )
    cache = (directory / "cache").resolve()
    if cache.parent != directory or not cache.is_dir():
        raise ValidationError("shared-bootstrap cache directory is missing")
    entries: dict[str, dict[str, Any]] = {}
    filenames: set[str] = set()
    for key, batch_strategy in requirements.items():
        record = _require_object(
            raw_entries.get(key), f"shared-bootstrap entry {key}"
        )
        expected_dataset = key.rsplit("/", 1)[0]
        if record.get("dataset") != expected_dataset:
            raise ValidationError(
                f"shared-bootstrap dataset identity differs for {key}"
            )
        if record.get("batch_strategy") != batch_strategy:
            raise ValidationError(
                f"shared-bootstrap batch strategy differs for {key}"
            )
        filename = _safe_direct_filename(
            record.get("filename"), f"shared-bootstrap entry {key} filename"
        )
        if filename in filenames:
            raise ValidationError(
                f"shared-bootstrap filename is reused by multiple keys: {filename}"
            )
        filenames.add(filename)
        semantics = validate_bootstrap_cache_file(cache / filename)
        expected_semantics = {
            "file_sha256": semantics["file_sha256"],
            "level_zero_sha256": semantics["level_zero_sha256"],
            "vertices": semantics["vertices"],
        }
        for field, expected in expected_semantics.items():
            if record.get(field) != expected:
                raise ValidationError(
                    f"shared-bootstrap {field} changed for {key}"
                )
        modularity = record.get("modularity")
        if (
            isinstance(modularity, bool)
            or not isinstance(modularity, (int, float))
            or not math.isclose(
                float(modularity),
                float(semantics["modularity"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValidationError(
                f"shared-bootstrap modularity changed for {key}"
            )
        entries[key] = record
    live_filenames = {
        path.name for path in cache.iterdir() if path.is_file()
    }
    if live_filenames != filenames:
        raise ValidationError(
            "shared-bootstrap cache contains missing or unregistered files"
        )

    identity = {
        "schema": SHARED_BOOTSTRAP_SCHEMA,
        "relative_path": relative_path,
        "manifest": {
            "filename": manifest_path.name,
            "sha256": manifest_sha256,
        },
        "registration": {
            "filename": registration_path.name,
            "sha256": sha256_file(registration_path),
        },
        "entries_sha256": sha256_json(raw_entries),
        "parallel_manifest_sha256": parallel_manifest_sha256,
    }
    return BootstrapAuthorityRecord(
        directory=directory,
        identity=identity,
        manifest=manifest,
        plan=plan,
        cache=cache,
        entries=entries,
    )


def _expected_shard_bootstrap_keys(spec: ShardSpec) -> set[str]:
    initial_batch = "999" if spec.phase_cli == "short" else "9"
    required = {f"dyn_pubmed/{initial_batch}", f"arxivmath/{initial_batch}"}
    if spec.ordinal == 1:
        required |= {
            "dyn_cora/999",
            "dyn_acm/999",
            "dyn_citeseer/999",
            "patent/999",
        }
    return required


def _bootstrap_authority_source_report(
    shards: Iterable[ShardRecord],
    authority: BootstrapAuthorityRecord,
) -> dict[str, Any]:
    observations: dict[str, list[dict[str, Any]]] = {}
    for shard in shards:
        source = _require_object(
            shard.manifest["parallel_shard"].get("bootstrap_source"),
            f"{shard.directory}/parallel_shard.bootstrap_source",
        )
        if source.get("schema") != BOOTSTRAP_SOURCE_SCHEMA:
            raise ValidationError(
                f"{shard.directory}: invalid bootstrap-source schema"
            )
        if source.get("authority") != authority.identity:
            raise ValidationError(
                f"{shard.directory}: bootstrap source belongs to another "
                "shared authority"
            )
        if source.get("source_directory") != str(authority.cache):
            raise ValidationError(
                f"{shard.directory}: bootstrap source directory is not the "
                "sealed campaign authority"
            )
        entries = _require_object(
            source.get("entries"),
            f"{shard.directory}/bootstrap_source.entries",
        )
        required = _expected_shard_bootstrap_keys(shard.spec)
        if set(entries) != required:
            raise ValidationError(
                f"{shard.directory}: bootstrap-source entries differ; "
                f"expected {sorted(required)}, got {sorted(entries)}"
            )
        for key, raw_entry in entries.items():
            entry = _require_object(
                raw_entry, f"{shard.directory}/bootstrap_source/{key}"
            )
            authority_entry = authority.entries[key]
            filename = authority_entry["filename"]
            expected_entry = {
                "source": filename,
                "source_sha256": authority_entry["file_sha256"],
                "target": filename,
                "target_sha256": authority_entry["file_sha256"],
                "level_zero_sha256": authority_entry["level_zero_sha256"],
            }
            for field, expected in expected_entry.items():
                if entry.get(field) != expected:
                    raise ValidationError(
                        f"{shard.directory}: bootstrap {field} differs for {key}"
                    )
            modularity = entry.get("modularity")
            if (
                isinstance(modularity, bool)
                or not isinstance(modularity, (int, float))
                or not math.isclose(
                    float(modularity),
                    float(authority_entry["modularity"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise ValidationError(
                    f"{shard.directory}: bootstrap modularity differs for {key}"
                )
            target_path = shard.directory / "bootstrap-cache" / filename
            target_semantics = validate_bootstrap_cache_file(target_path)
            if target_semantics["file_sha256"] != authority_entry["file_sha256"]:
                raise ValidationError(
                    f"{shard.directory}: copied bootstrap bytes differ for {key}"
                )
            if (
                target_semantics["level_zero_sha256"]
                != authority_entry["level_zero_sha256"]
            ):
                raise ValidationError(
                    f"{shard.directory}: copied bootstrap partition differs for {key}"
                )
            if not math.isclose(
                float(target_semantics["modularity"]),
                float(authority_entry["modularity"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValidationError(
                    f"{shard.directory}: copied bootstrap modularity differs for {key}"
                )
            observations.setdefault(key, []).append(
                {
                    "source_sha256": entry["source_sha256"],
                    "target_sha256": entry["target_sha256"],
                    "level_zero_sha256": entry["level_zero_sha256"],
                    "modularity": float(entry["modularity"]),
                    "vertices": int(authority_entry["vertices"]),
                    "filename": filename,
                }
            )

    if set(observations) != set(authority.entries):
        raise ValidationError(
            "the shard bootstrap copies do not cover all eight authority keys"
        )
    result: dict[str, Any] = {}
    for key, values in sorted(observations.items()):
        digests = {sha256_json(value) for value in values}
        if len(digests) != 1:
            raise ValidationError(f"bootstrap source differs across shards: {key}")
        result[key] = {"observations": len(values), **values[0]}
    return result


def _validate_bootstrap_producer(
    authority: BootstrapAuthorityRecord,
    producer: ShardRecord,
) -> None:
    if producer.spec.ordinal != 1:
        raise ValidationError("shared-bootstrap producer must be shard 01")
    plan = authority.plan
    expected = {
        "producer_cpu_affinity": list(producer.cpu_affinity),
        "producer_cpu_topology": producer.cpu_topology,
        "protocol_fingerprint_sha256": sha256_json(
            producer.manifest["fingerprint"]
        ),
        "backend_sha256": sha256_json(producer.manifest["backend"]),
        "hardware": producer.manifest["hardware"],
        "runtime_identity": producer.manifest["runtime_identity"],
        "real_input_manifest": producer.manifest["real_input_manifest"],
    }
    for key, value in expected.items():
        if plan.get(key) != value:
            raise ValidationError(
                f"shared-bootstrap producer identity differs for {key}"
            )


def _load_preflight(root: Path) -> tuple[dict[str, Any], list[ShardRecord], dict[str, Any]]:
    parallel_manifest_path = root / "parallel_manifest.json"
    if not parallel_manifest_path.is_file():
        raise ValidationError(f"missing parallel manifest: {parallel_manifest_path}")
    parallel_manifest = _require_object(
        read_json(parallel_manifest_path), str(parallel_manifest_path)
    )
    if parallel_manifest.get("schema") != PARALLEL_SCHEMA:
        raise ValidationError(
            f"unexpected parallel campaign schema: {parallel_manifest.get('schema')!r}"
        )
    parallel_manifest_sha256 = _validate_parallel_registration(
        root, parallel_manifest_path
    )
    window = _validate_parallel_window(parallel_manifest.get("window"))
    run_id = parallel_manifest.get("run_id")
    if not isinstance(run_id, str) or run_id != root.name:
        raise ValidationError("parallel manifest run_id differs from its root directory")
    authority = _load_bootstrap_authority(
        root, parallel_manifest, parallel_manifest_sha256
    )
    recorded_names = _parallel_manifest_shard_names(parallel_manifest)
    _validate_parallel_assignments(parallel_manifest)
    expected_names = {spec.directory for spec in EXPECTED_SHARDS}
    if recorded_names is not None and recorded_names != expected_names:
        raise ValidationError(
            "parallel_manifest.shards does not contain exactly the registered "
            f"assignments: expected {sorted(expected_names)}, got {sorted(recorded_names)}"
        )

    protocol = load_protocol()
    if parallel_manifest.get("protocol_id") != protocol["protocol_id"]:
        raise ValidationError("parallel manifest and registered protocol differ")
    phases = {phase["id"]: phase for phase in protocol["phases"]}
    for spec in EXPECTED_SHARDS:
        phase = phases.get(spec.phase_id)
        if phase is None or phase.get("cli_name") != spec.phase_cli:
            raise ValidationError(f"registered protocol no longer contains {spec.phase_id}")
    current_fingerprint = protocol_fingerprint(protocol)

    shards: list[ShardRecord] = []
    for spec in EXPECTED_SHARDS:
        campaign_dir = root / "shards" / spec.directory
        manifest_path = campaign_dir / "manifest.json"
        if not manifest_path.is_file():
            raise ValidationError(f"missing shard manifest: {manifest_path}")
        manifest = _require_object(read_json(manifest_path), str(manifest_path))
        if manifest.get("schema") != CAMPAIGN_SCHEMA:
            raise ValidationError(f"{campaign_dir}: unexpected campaign schema")
        if manifest.get("campaign_id") != spec.directory:
            raise ValidationError(f"{campaign_dir}: campaign id differs from assignment")
        if manifest.get("protocol_id") != protocol["protocol_id"]:
            raise ValidationError(f"{campaign_dir}: protocol id differs")
        git = _require_object(manifest.get("git"), f"{campaign_dir}/git")
        commit = git.get("commit")
        if (
            not isinstance(commit, str)
            or len(commit) != 40
            or any(character not in "0123456789abcdef" for character in commit)
        ):
            raise ValidationError(f"{campaign_dir}: missing Git commit")
        if git.get("dirty") is not False:
            raise ValidationError(f"{campaign_dir}: measurement checkout was dirty")
        fingerprint = _require_object(
            manifest.get("fingerprint"), f"{campaign_dir}/fingerprint"
        )
        if fingerprint != current_fingerprint:
            raise ValidationError(
                f"{campaign_dir}: measurement source, protocol, or launcher configs differ "
                "from the validator checkout"
            )
        _validate_backend_and_workers(campaign_dir, manifest)
        _, affinity, topology = _validate_assignment(
            campaign_dir,
            manifest,
            spec,
            parallel_manifest_sha256=parallel_manifest_sha256,
            bootstrap_authority_identity=authority.identity,
        )
        _, hardware = _registered_artifact(campaign_dir, manifest, "hardware")
        _, runtime_identity = _registered_artifact(
            campaign_dir, manifest, "runtime_identity"
        )
        _, input_manifest = _registered_artifact(
            campaign_dir, manifest, "real_input_manifest"
        )
        if runtime_identity.get("backend") != manifest.get("backend"):
            raise ValidationError(f"{campaign_dir}: runtime and manifest backend differ")
        if runtime_identity.get("hardware_sha256") != sha256_json(hardware):
            raise ValidationError(f"{campaign_dir}: runtime and hardware identity differ")
        container_identity = runtime_identity.get("container")
        if not isinstance(container_identity, dict) or not container_identity:
            raise ValidationError(f"{campaign_dir}: container identity is missing")
        if input_manifest.get("schema") != INPUT_SCHEMA:
            raise ValidationError(f"{campaign_dir}: invalid real-input manifest schema")
        if not isinstance(input_manifest.get("files"), list) or not input_manifest["files"]:
            raise ValidationError(f"{campaign_dir}: real-input manifest has no files")
        paths_digest = manifest.get("paths_config", {}).get("sha256")
        if input_manifest.get("paths_config_sha256") != paths_digest:
            raise ValidationError(f"{campaign_dir}: paths-config identity differs")
        shards.append(
            ShardRecord(
                spec=spec,
                directory=campaign_dir,
                manifest=manifest,
                hardware=hardware,
                runtime_identity=runtime_identity,
                input_manifest=input_manifest,
                cpu_affinity=affinity,
                cpu_topology=topology,
            )
        )

    _validate_disjoint_affinity(shards)
    _validate_bootstrap_producer(authority, shards[0])
    bootstrap_sources = _bootstrap_authority_source_report(shards, authority)
    git_commit, git_digest = _cross_shard_value(
        shards, "Git commit", lambda shard: {"commit": shard.manifest["git"]["commit"]}
    )
    fingerprint, fingerprint_digest = _cross_shard_value(
        shards, "source/config fingerprint", lambda shard: shard.manifest["fingerprint"]
    )
    backend, backend_digest = _cross_shard_value(
        shards, "LD-Leiden backend", lambda shard: shard.manifest["backend"]
    )
    hardware, hardware_digest = _cross_shard_value(
        shards, "CPU measurement hardware", lambda shard: _comparable_hardware(
            shard.hardware
        )
    )
    input_manifest, input_digest = _cross_shard_value(
        shards, "real inputs", lambda shard: shard.input_manifest
    )
    runtime_common, runtime_common_digest = _cross_shard_value(
        shards,
        "runtime excluding container identity",
        lambda shard: _runtime_without_container(shard.runtime_identity),
    )
    del backend, hardware, input_manifest, runtime_common

    expected_commit = parallel_manifest.get("expected_git_sha")
    if (
        not isinstance(expected_commit, str)
        or len(expected_commit) != 40
        or any(
            character not in "0123456789abcdef"
            for character in expected_commit
        )
        or expected_commit != git_commit["commit"]
    ):
        raise ValidationError("parallel manifest and shard Git commits differ")
    parallel_paths_digest = parallel_manifest.get("paths_config", {}).get("sha256")
    shard_paths_digest = shards[0].manifest.get("paths_config", {}).get("sha256")
    if parallel_paths_digest != shard_paths_digest:
        raise ValidationError("parallel manifest and shard paths-config hashes differ")

    container_digests = {
        shard.spec.directory: sha256_json(shard.runtime_identity["container"])
        for shard in shards
    }
    if len(set(container_digests.values())) != len(EXPECTED_SHARDS):
        raise ValidationError(
            "the parallel campaign must contain eight unique container identities"
        )
    preflight = {
        "parallel_manifest_sha256": parallel_manifest_sha256,
        "parallel_manifest_registration_sha256": sha256_file(
            root / "parallel_manifest_registration.json"
        ),
        "window": window,
        "protocol_id": protocol["protocol_id"],
        "git_commit": git_commit["commit"],
        "identity": {
            "git_sha256": git_digest,
            "fingerprint_sha256": fingerprint_digest,
            "backend_sha256": backend_digest,
            "hardware_sha256": hardware_digest,
            "real_input_manifest_sha256": input_digest,
            "runtime_without_container_sha256": runtime_common_digest,
            "source_matches_validator_checkout": (
                fingerprint.get("source_sha256")
                == current_fingerprint.get("source_sha256")
            ),
        },
        "container_identity_policy": "eight distinct container identities required",
        "unique_container_identities": len(set(container_digests.values())),
        "evidence_scope": EVIDENCE_SCOPE,
        "bootstrap_authority": {
            "directory": str(authority.directory),
            "identity": authority.identity,
            "generation": authority.manifest["generation"],
            "entries": authority.entries,
        },
        "bootstrap_source": bootstrap_sources,
        "shards": {
            shard.spec.directory: {
                "campaign_id": shard.manifest.get("campaign_id"),
                "phase_cli": shard.spec.phase_cli,
                "phase_id": shard.spec.phase_id,
                "repeat": shard.spec.repeat,
                "cpu_affinity": list(shard.cpu_affinity),
                "cpu_topology": shard.cpu_topology,
                "physical_core": [
                    shard.cpu_topology["physical_package_id"],
                    shard.cpu_topology["core_id"],
                ],
                "container_identity_sha256": container_digests[shard.spec.directory],
                "bootstrap_authority": shard.manifest["parallel_shard"][
                    "bootstrap_authority"
                ],
                "bootstrap_source": shard.manifest["parallel_shard"][
                    "bootstrap_source"
                ],
            }
            for shard in shards
        },
    }
    return parallel_manifest, shards, preflight


def _attempt_directories(shard: ShardRecord) -> list[Path]:
    result: list[Path] = []
    for metadata_path in sorted(shard.directory.glob("*/repeat-*/attempt-*/metadata.json")):
        metadata = _require_object(read_json(metadata_path), str(metadata_path))
        if metadata.get("status") == "completed":
            result.append(metadata_path.parent)
    return result


def _expected_attempt_identity(shard: ShardRecord) -> dict[str, str]:
    manifest = shard.manifest
    return {
        "fingerprint_sha256": sha256_json(manifest["fingerprint"]),
        "hardware_sha256": manifest["hardware"]["sha256"],
        "runtime_identity_sha256": manifest["runtime_identity"]["sha256"],
        "real_input_manifest_sha256": manifest["real_input_manifest"]["sha256"],
    }


def _timestamp_epoch(value: Any, label: str) -> float:
    if not isinstance(value, str):
        raise ValidationError(f"{label} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("timestamp has no UTC offset")
        epoch = parsed.timestamp()
    except (ValueError, OverflowError) as exc:
        raise ValidationError(f"{label} is not a valid timestamp") from exc
    if not math.isfinite(epoch):
        raise ValidationError(f"{label} is not finite")
    return epoch


def _validate_attempt_window(
    metadata: dict[str, Any],
    metadata_path: Path,
    phase: dict[str, Any],
    parallel_window: dict[str, float],
) -> None:
    window_id = metadata.get("measurement_window_id")
    raw_deadline = metadata.get("window_deadline_epoch")
    if (
        isinstance(raw_deadline, bool)
        or not isinstance(raw_deadline, (int, float))
        or not math.isfinite(float(raw_deadline))
    ):
        raise ValidationError(f"{metadata_path}: measurement deadline is required")
    deadline = float(raw_deadline)
    started = _timestamp_epoch(
        metadata.get("started_at_utc"), f"{metadata_path}/started_at_utc"
    )
    finished = _timestamp_epoch(
        metadata.get("finished_at_utc"), f"{metadata_path}/finished_at_utc"
    )
    global_start = parallel_window["started_epoch"]
    global_deadline = parallel_window["deadline_epoch"]
    if (
        started < global_start - 2.0
        or started >= global_deadline
        or finished > global_deadline + 2.0
    ):
        raise ValidationError(
            f"{metadata_path}: attempt lies outside the registered global window"
        )
    if phase["id"] == "smoke_999_10":
        expected_window_id = "parallel-smoke-gate"
        if window_id != expected_window_id or raw_deadline != global_deadline:
            raise ValidationError(
                f"{metadata_path}: smoke must use the exact global deadline and "
                f"{expected_window_id!r}"
            )
    elif phase["id"] == "measured_999_10":
        expected_window_id = "parallel-24h-window"
        if window_id != expected_window_id or raw_deadline != global_deadline:
            raise ValidationError(
                f"{metadata_path}: short measurement must use the exact global "
                f"deadline and {expected_window_id!r}"
            )
    elif phase["id"] == "measured_9_500":
        if window_id != "parallel-long-one-hour-limit":
            raise ValidationError(
                f"{metadata_path}: long measurement has an invalid window id"
            )
        if deadline > global_deadline or deadline > started + LONG_TIMEOUT_SECONDS:
            raise ValidationError(
                f"{metadata_path}: long measurement exceeds its global or one-hour "
                "deadline"
            )
        if deadline <= started:
            raise ValidationError(
                f"{metadata_path}: long deadline does not follow its start"
            )
    else:
        raise ValidationError(f"{metadata_path}: unregistered phase window")


def _validate_attempt(
    shard: ShardRecord,
    attempt_dir: Path,
    phase: dict[str, Any],
    repeat: int,
    parallel_window: dict[str, float],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    metadata_path = attempt_dir / "metadata.json"
    metadata = _require_object(read_json(metadata_path), str(metadata_path))
    validate_window_attestation(metadata, metadata_path)
    _validate_attempt_window(metadata, metadata_path, phase, parallel_window)
    expected_metadata = {
        "status": "completed",
        "phase_id": phase["id"],
        "role": phase["role"],
        "included_in_analysis": phase["included_in_analysis"],
        "repeat": repeat,
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise ValidationError(
                f"{metadata_path}: expected {key}={expected!r}, "
                f"got {metadata.get(key)!r}"
            )
    expected_identity = _expected_attempt_identity(shard)
    if metadata.get("identity_before") != expected_identity:
        raise ValidationError(f"{metadata_path}: pre-run identity mismatch")
    if metadata.get("identity_after") != expected_identity:
        raise ValidationError(f"{metadata_path}: post-run identity mismatch")
    thread_environment = metadata.get("thread_environment")
    if (
        not isinstance(thread_environment, dict)
        or not thread_environment
        or set(thread_environment.values()) != {"1"}
    ):
        raise ValidationError(f"{metadata_path}: thread environment is not fixed")
    recorded_affinity = metadata.get("cpu_affinity")
    if _parse_affinity(
        recorded_affinity, f"{metadata_path}/cpu_affinity"
    ) != shard.cpu_affinity:
        raise ValidationError(f"{metadata_path}: CPU affinity differs from preflight")
    if metadata.get("cpu_topology") != shard.cpu_topology:
        raise ValidationError(f"{metadata_path}: CPU topology differs from preflight")
    expected_parallel_sha = shard.manifest["parallel_shard"][
        "parallel_manifest_sha256"
    ]
    if metadata.get("parallel_manifest_sha256") != expected_parallel_sha:
        raise ValidationError(
            f"{metadata_path}: parallel manifest hash is missing or differs"
        )

    expected_result_path = launcher_result_file(attempt_dir, phase)
    result_filename = metadata.get("result_file")
    if result_filename is not None and result_filename != expected_result_path.name:
        raise ValidationError(f"{metadata_path}: unexpected result filename")
    if not expected_result_path.is_file():
        raise ValidationError(f"missing launcher result: {expected_result_path}")
    if metadata.get("result_sha256") != sha256_file(expected_result_path):
        raise ValidationError(f"{metadata_path}: launcher result digest changed")
    summaries = validate_launcher_file(expected_result_path, phase)

    cache_hashes = metadata.get("bootstrap_cache_sha256")
    if not isinstance(cache_hashes, dict) or not cache_hashes:
        raise ValidationError(f"{metadata_path}: bootstrap-cache hashes are missing")
    cache_dir = shard.directory / "bootstrap-cache"
    semantics = {
        dataset: bootstrap_cache_reference(
            cache_dir,
            cache_hashes,
            dataset,
            phase["batch_strategy"],
            summaries[dataset]["bootstrap_reference_sha256"],
        )
        for dataset in phase["datasets"]
    }
    if metadata.get("bootstrap_semantics") != semantics:
        raise ValidationError(f"{metadata_path}: bootstrap semantics changed")
    if metadata.get("validated_summary") != summaries:
        raise ValidationError(f"{metadata_path}: validated summary changed")
    return metadata, {dataset: semantics[dataset] for dataset in phase["datasets"]}


def _record_bootstraps(
    observations: dict[str, list[dict[str, Any]]],
    phase: dict[str, Any],
    semantics: dict[str, dict[str, Any]],
) -> None:
    initial_batch = phase["batch_strategy"].split(":", 1)[0]
    for dataset, record in semantics.items():
        observations.setdefault(f"{dataset}/{initial_batch}", []).append(record)


def _finalize_bootstraps(
    observations: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for key, records in sorted(observations.items()):
        file_digests = {record.get("file_sha256") for record in records}
        partition_digests = {record.get("level_zero_sha256") for record in records}
        vertices = {record.get("vertices") for record in records}
        modularities = {record.get("modularity") for record in records}
        if (
            None in file_digests
            or len(file_digests) != 1
            or len(partition_digests) != 1
            or None in partition_digests
            or len(vertices) != 1
            or len(modularities) != 1
        ):
            raise ValidationError(f"bootstrap identity differs across attempts: {key}")
        report[key] = {
            "observations": len(records),
            "file_sha256": sorted(file_digests),
            "level_zero_sha256": next(iter(partition_digests)),
            "vertices": next(iter(vertices)),
            "modularity": next(iter(modularities)),
            "filenames": sorted({str(record.get("filename")) for record in records}),
        }
    return report


def _compare_bootstrap_source_and_attempts(
    source: dict[str, Any],
    observed: dict[str, Any],
) -> None:
    if set(source) != set(observed):
        raise ValidationError(
            "validated bootstrap keys differ from the sealed bootstrap source"
        )
    for key in source:
        expected = source[key]
        actual = observed[key]
        if actual["file_sha256"] != [expected["target_sha256"]]:
            raise ValidationError(f"attempt bootstrap bytes differ from source: {key}")
        if expected["level_zero_sha256"] != actual["level_zero_sha256"]:
            raise ValidationError(f"attempt bootstrap differs from source: {key}")
        if float(expected["modularity"]) != float(actual["modularity"]):
            raise ValidationError(f"attempt bootstrap modularity differs from source: {key}")


def _find_smoke(
    shards: list[ShardRecord],
    phases: dict[str, dict[str, Any]],
) -> tuple[ShardRecord, Path]:
    smoke_phase = phases["smoke_999_10"]
    matches: list[tuple[ShardRecord, Path]] = []
    for shard in shards:
        for attempt_dir in _attempt_directories(shard):
            metadata = _require_object(
                read_json(attempt_dir / "metadata.json"),
                str(attempt_dir / "metadata.json"),
            )
            if metadata.get("phase_id") == smoke_phase["id"]:
                matches.append((shard, attempt_dir))
    if len(matches) != 1:
        raise ValidationError(
            f"expected exactly one completed smoke attempt, found {len(matches)}"
        )
    shard, attempt_dir = matches[0]
    if shard.spec.directory != EXPECTED_SHARDS[0].directory:
        raise ValidationError("the smoke attempt must belong to shard 01-short-r01")
    expected = shard.directory / smoke_phase["id"] / "repeat-01"
    if attempt_dir.parent != expected:
        raise ValidationError(f"smoke attempt is stored in an unexpected directory: {attempt_dir}")
    return shard, attempt_dir


def _validate_smoke(
    shards: list[ShardRecord],
    phases: dict[str, dict[str, Any]],
    bootstrap_observations: dict[str, list[dict[str, Any]]],
    parallel_window: dict[str, float],
) -> dict[str, Any]:
    phase = phases["smoke_999_10"]
    shard, attempt_dir = _find_smoke(shards, phases)
    metadata, semantics = _validate_attempt(
        shard, attempt_dir, phase, repeat=1, parallel_window=parallel_window
    )
    _record_bootstraps(bootstrap_observations, phase, semantics)
    return {
        "shard": shard.spec.directory,
        "attempt": str(attempt_dir.relative_to(shard.directory)),
        "included_in_analysis": False,
        "finished_at_utc": metadata.get("finished_at_utc"),
        "datasets": metadata["validated_summary"],
    }


def _validate_measured(
    shards: list[ShardRecord],
    phases: dict[str, dict[str, Any]],
    bootstrap_observations: dict[str, list[dict[str, Any]]],
    parallel_window: dict[str, float],
) -> tuple[dict[str, Any], list[str]]:
    rows: dict[str, dict[str, list[dict[str, Any]]]] = {
        "measured_999_10": {"dyn_pubmed": [], "arxivmath": []},
        "measured_9_500": {"dyn_pubmed": [], "arxivmath": []},
    }
    attempts: dict[str, list[dict[str, Any]]] = {
        "measured_999_10": [],
        "measured_9_500": [],
    }
    failed_attempts: list[str] = []
    for shard in shards:
        phase = phases[shard.spec.phase_id]
        completed = _attempt_directories(shard)
        allowed_phase_ids = {phase["id"]}
        if shard.spec.directory == EXPECTED_SHARDS[0].directory:
            allowed_phase_ids.add("smoke_999_10")
        assigned_matches: list[Path] = []
        for attempt_dir in completed:
            metadata = _require_object(
                read_json(attempt_dir / "metadata.json"),
                str(attempt_dir / "metadata.json"),
            )
            phase_id = metadata.get("phase_id")
            if phase_id not in allowed_phase_ids:
                raise ValidationError(
                    f"{attempt_dir}: completed unassigned phase {phase_id!r}"
                )
            if phase_id == phase["id"]:
                assigned_matches.append(attempt_dir)
        expected_parent = shard.directory / phase["id"] / f"repeat-{shard.spec.repeat:02d}"
        if len(assigned_matches) != 1 or assigned_matches[0].parent != expected_parent:
            raise ValidationError(
                f"{shard.directory}: expected exactly one assigned completed attempt "
                f"below {expected_parent}"
            )
        for metadata_path in sorted(
            shard.directory.glob("*/repeat-*/attempt-*/metadata.json")
        ):
            if read_json(metadata_path).get("status") != "completed":
                failed_attempts.append(
                    f"{shard.spec.directory}/{metadata_path.relative_to(shard.directory)}"
                )

        attempt_dir = assigned_matches[0]
        metadata, semantics = _validate_attempt(
            shard,
            attempt_dir,
            phase,
            repeat=shard.spec.repeat,
            parallel_window=parallel_window,
        )
        _record_bootstraps(bootstrap_observations, phase, semantics)
        summaries = metadata["validated_summary"]
        for dataset in phase["datasets"]:
            rows[phase["id"]][dataset].append(summaries[dataset])
        attempts[phase["id"]].append(
            {
                "shard": shard.spec.directory,
                "repeat": shard.spec.repeat,
                "attempt": str(attempt_dir.relative_to(shard.directory)),
                "datasets": summaries,
            }
        )

    result: dict[str, Any] = {}
    expected_repetitions = {"measured_999_10": 5, "measured_9_500": 3}
    for phase_id, dataset_rows in rows.items():
        repetitions = expected_repetitions[phase_id]
        if len(attempts[phase_id]) != repetitions:
            raise ValidationError(
                f"{phase_id}: expected {repetitions} attempts, "
                f"got {len(attempts[phase_id])}"
            )
        aggregates = {
            dataset: aggregate_run_summaries(values)
            for dataset, values in dataset_rows.items()
        }
        for dataset, aggregate in aggregates.items():
            if aggregate["repetitions"] != repetitions:
                raise ValidationError(
                    f"{phase_id}/{dataset}: incomplete repetition set"
                )
        result[phase_id] = {
            "role": phases[phase_id]["role"],
            "included_in_analysis": True,
            "expected_repetitions": repetitions,
            "completed_repetitions": len(attempts[phase_id]),
            "attempts": attempts[phase_id],
            "datasets": aggregates,
        }
    return result, failed_attempts


def _precision_report(phases: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    long_phase = phases["measured_9_500"]
    all_within = True
    for dataset, aggregate in long_phase["datasets"].items():
        timing = aggregate["optimization_seconds"]
        mean = float(timing["mean"])
        sample_sd = timing["sample_sd"]
        cv = None if sample_sd is None or mean <= 0 else float(sample_sd) / mean
        within = cv is not None and cv <= 0.05
        all_within = all_within and within
        checks[dataset] = {
            "metric": "cumulative optimization_seconds",
            "coefficient_of_variation": cv,
            "target": 0.05,
            "within_target": within,
        }
    return {
        "all_long_optimization_cvs_within_5_percent": all_within,
        "datasets": checks,
    }


def validate_parallel(root: Path, stage: str) -> dict[str, Any]:
    root = root.expanduser().resolve()
    if stage not in {"preflight", "smoke", "final"}:
        raise ValueError(f"unsupported validation stage: {stage}")
    parallel_manifest, shards, preflight = _load_preflight(root)
    protocol = load_protocol()
    phases = {phase["id"]: phase for phase in protocol["phases"]}
    report: dict[str, Any] = {
        "schema": PARALLEL_REPORT_SCHEMA,
        "campaign_id": parallel_manifest["run_id"],
        "stage": stage,
        "status": "valid_preflight",
        "evidence_scope": EVIDENCE_SCOPE,
        "preflight": preflight,
    }
    if stage == "preflight":
        return report

    bootstrap_observations: dict[str, list[dict[str, Any]]] = {}
    report["smoke"] = _validate_smoke(
        shards,
        phases,
        bootstrap_observations,
        preflight["window"],
    )
    report["status"] = "valid_smoke"
    if stage == "smoke":
        report["bootstrap_cache"] = _finalize_bootstraps(bootstrap_observations)
        smoke_source = {
            key: value
            for key, value in preflight["bootstrap_source"].items()
            if key.endswith("/999")
        }
        _compare_bootstrap_source_and_attempts(
            smoke_source, report["bootstrap_cache"]
        )
        return report

    measured, failed_attempts = _validate_measured(
        shards,
        phases,
        bootstrap_observations,
        preflight["window"],
    )
    report["phases"] = measured
    report["bootstrap_cache"] = _finalize_bootstraps(bootstrap_observations)
    _compare_bootstrap_source_and_attempts(
        preflight["bootstrap_source"], report["bootstrap_cache"]
    )
    report["failed_or_invalid_attempts"] = failed_attempts
    report["precision"] = _precision_report(measured)
    report["status"] = "valid"
    return report


def _default_report_path(root: Path, stage: str) -> Path:
    names = {
        "preflight": "preflight_report.json",
        "smoke": "smoke_gate.json",
        "final": "validation_report.json",
    }
    return root / names[stage]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        type=Path,
        help="Parallel campaign root containing parallel_manifest.json and shards/.",
    )
    parser.add_argument(
        "--stage",
        choices=("preflight", "smoke", "final"),
        required=True,
        help="Validation gate to execute.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help=(
            "Output JSON path. Defaults to ROOT/preflight_report.json, "
            "ROOT/smoke_gate.json, or ROOT/validation_report.json."
        ),
    )
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    try:
        report = validate_parallel(root, args.stage)
    except (OSError, ValueError, ValidationError) as exc:
        raise SystemExit(f"Parallel LD-Leiden validation failed: {exc}") from exc
    report_path = (
        args.report.expanduser().resolve()
        if args.report is not None
        else _default_report_path(root, args.stage)
    )
    write_json(report_path, report)
    print(f"Validation status: {report['status']}")
    print(f"Report: {report_path}")
    if args.stage == "final":
        for phase_id, phase in report["phases"].items():
            print(
                f"{phase_id}: {phase['completed_repetitions']}/"
                f"{phase['expected_repetitions']} repetitions"
            )


if __name__ == "__main__":
    main()
