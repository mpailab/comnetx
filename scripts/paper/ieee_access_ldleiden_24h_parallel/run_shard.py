#!/usr/bin/env python3
"""Run one sealed repeat of the parallel IEEE Access LD-Leiden campaign."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import math
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.paper.ieee_access_72h_launch.sync_bootstrap import (  # noqa: E402
    _canonical,
    _load_source,
    _partition_sha256,
    _sha256,
    _source_candidate,
)
from scripts.paper.ieee_access_ldleiden_72h.protocol import (  # noqa: E402
    cli_phase_map,
    load_protocol,
    read_json,
    sha256_file,
    sha256_json,
    validate_bootstrap_cache_file,
    write_json,
)
from scripts.paper.ieee_access_ldleiden_72h.process_control import (  # noqa: E402
    install_termination_signal_handlers,
)
RESULTS_ROOT = (
    PROJECT_ROOT / "results" / "ieee-access-2026-1" / "raw" / "ldleiden"
)
REPAIRED_ROOT = (
    PROJECT_ROOT
    / "results"
    / "ieee-access-2026-1"
    / "raw"
    / "repaired-comnetx"
)
BASE_RUNNER = (
    PROJECT_ROOT
    / "scripts"
    / "paper"
    / "ieee_access_ldleiden_72h"
    / "run_protocol.py"
)
SCHEMA = "comnetx-ieee-access-ldleiden-parallel-v1"
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
LONG_TIMEOUT_SECONDS = 3600.0
PARALLEL_REGISTRATION_SCHEMA = (
    "comnetx-ieee-access-ldleiden-parallel-manifest-registration-v1"
)
PARALLEL_PENDING_MANIFEST = "parallel_manifest.initializing.json"


SHARDS: dict[str, dict[str, Any]] = {
    "01": {"phase_cli": "short", "phase_id": "measured_999_10", "repeat": 1},
    "02": {"phase_cli": "short", "phase_id": "measured_999_10", "repeat": 2},
    "03": {"phase_cli": "short", "phase_id": "measured_999_10", "repeat": 3},
    "04": {"phase_cli": "short", "phase_id": "measured_999_10", "repeat": 4},
    "05": {"phase_cli": "short", "phase_id": "measured_999_10", "repeat": 5},
    "06": {"phase_cli": "long", "phase_id": "measured_9_500", "repeat": 1},
    "07": {"phase_cli": "long", "phase_id": "measured_9_500", "repeat": 2},
    "08": {"phase_cli": "long", "phase_id": "measured_9_500", "repeat": 3},
}


class CampaignTimeBoundary(RuntimeError):
    """Raised when the shared one-day or one-hour shard deadline expires."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def resolve_project_path(value: Path) -> Path:
    value = value.expanduser()
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def resolve_repaired_campaign(value: Path) -> Path:
    value = value.expanduser()
    if value.is_absolute():
        return value.resolve()
    if len(value.parts) == 1:
        return (REPAIRED_ROOT / value).resolve()
    return (PROJECT_ROOT / value).resolve()


def run_root(run_id: str) -> Path:
    if RUN_ID_RE.fullmatch(run_id) is None:
        raise ValueError("run id may contain only letters, digits, dot, underscore, and dash")
    return RESULTS_ROOT / run_id


def campaign_name(shard_id: str) -> str:
    assignment = SHARDS[shard_id]
    return f"{shard_id}-{assignment['phase_cli']}-r{assignment['repeat']:02d}"


def campaign_dir(root: Path, shard_id: str) -> Path:
    return root / "shards" / campaign_name(shard_id)


def parse_cpu_set(value: str) -> list[int]:
    cpus: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            raise ValueError("CPU set contains an empty component")
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start < 0 or end < start:
                raise ValueError(f"invalid CPU range: {part}")
            cpus.update(range(start, end + 1))
        else:
            cpu = int(part)
            if cpu < 0:
                raise ValueError(f"invalid CPU: {cpu}")
            cpus.add(cpu)
    if not cpus:
        raise ValueError("CPU set must not be empty")
    return sorted(cpus)


def require_affinity(expected: list[int]) -> None:
    if not hasattr(os, "sched_getaffinity"):
        raise RuntimeError("Linux CPU-affinity inspection is required")
    observed = sorted(os.sched_getaffinity(0))
    if observed != expected:
        raise RuntimeError(
            f"expected CPU affinity {expected}, observed {observed}; invoke through taskset"
        )


def git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def require_clean_commit(expected_sha: str) -> None:
    if SHA_RE.fullmatch(expected_sha) is None:
        raise RuntimeError("EXPECTED_GIT_SHA must be one full lowercase commit SHA")
    actual = git_output("rev-parse", "HEAD")
    if actual != expected_sha:
        raise RuntimeError(f"expected git SHA {expected_sha}, found {actual}")
    status = git_output("status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise RuntimeError("parallel measurements require a clean checkout:\n" + status)


def registered_artifact(
    campaign: Path,
    manifest: dict[str, Any],
    key: str,
) -> tuple[dict[str, str], dict[str, Any]]:
    registration = manifest.get(key)
    if not isinstance(registration, dict):
        raise RuntimeError(f"repaired campaign lacks {key} registration")
    filename = registration.get("filename")
    digest = registration.get("sha256")
    if not isinstance(filename, str) or not isinstance(digest, str):
        raise RuntimeError(f"repaired campaign has malformed {key} registration")
    path = campaign / filename
    if not path.is_file() or sha256_file(path) != digest:
        raise RuntimeError(f"repaired campaign {key} artifact changed: {path}")
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise RuntimeError(f"repaired campaign {key} artifact is not an object")
    return {"filename": filename, "sha256": digest}, payload


def repaired_source_equivalence(manifest: dict[str, Any]) -> dict[str, Any]:
    fingerprint = manifest.get("fingerprint")
    if not isinstance(fingerprint, dict):
        raise RuntimeError("repaired campaign lacks a source fingerprint")
    sources = fingerprint.get("source_sha256")
    if not isinstance(sources, dict) or not sources:
        raise RuntimeError("repaired campaign source fingerprint is empty")
    required = {
        "src/optimizer.py",
        "scripts/launch.py",
        "scripts/paper/collect_hardware_info.py",
    }
    if not required <= set(sources):
        raise RuntimeError("repaired campaign source fingerprint is incomplete")
    for relative, expected_digest in sources.items():
        if (
            not isinstance(relative, str)
            or not isinstance(expected_digest, str)
            or len(expected_digest) != 64
        ):
            raise RuntimeError("repaired campaign contains a malformed source identity")
        relative_path = Path(relative)
        path = (PROJECT_ROOT / relative_path).resolve()
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative_path.as_posix() != relative
            or PROJECT_ROOT not in path.parents
            or not path.is_file()
            or sha256_file(path) != expected_digest
        ):
            raise RuntimeError(
                "current checkout is not source-equivalent to repaired ComNetX: "
                + relative
            )
    git = manifest.get("git")
    repaired_commit = git.get("commit") if isinstance(git, dict) else None
    if not isinstance(repaired_commit, str) or SHA_RE.fullmatch(repaired_commit) is None:
        raise RuntimeError("repaired campaign commit is malformed")
    metadata_prefix = "datasets-info/json"
    repaired_names = {
        line
        for line in git_output(
            "ls-tree", "-r", "--name-only", repaired_commit, "--", metadata_prefix
        ).splitlines()
        if line
    }
    current_names = {
        line
        for line in git_output(
            "ls-tree", "-r", "--name-only", "HEAD", "--", metadata_prefix
        ).splitlines()
        if line
    }
    if not repaired_names or repaired_names != current_names:
        raise RuntimeError(
            "current dataset metadata tree differs from repaired ComNetX"
        )
    metadata_hashes: dict[str, str] = {}
    for relative in sorted(repaired_names):
        old = subprocess.run(
            ["git", "show", f"{repaired_commit}:{relative}"],
            cwd=PROJECT_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
        old_digest = hashlib.sha256(old).hexdigest()
        current_path = (PROJECT_ROOT / relative).resolve()
        if (
            PROJECT_ROOT not in current_path.parents
            or not current_path.is_file()
            or sha256_file(current_path) != old_digest
        ):
            raise RuntimeError(
                "current dataset metadata is not source-equivalent to repaired "
                "ComNetX: " + relative
            )
        metadata_hashes[relative] = old_digest
    return {
        "policy": (
            "all repaired measurement-source and dataset-metadata files "
            "byte-identical"
        ),
        "files": len(sources),
        "source_map_sha256": sha256_json(sources),
        "runtime_metadata_files": len(metadata_hashes),
        "runtime_metadata_map_sha256": sha256_json(metadata_hashes),
    }


def repaired_campaign_identity(
    campaign: Path,
    *,
    expected_git_sha: str,
    paths_config: Path,
) -> dict[str, Any]:
    manifest_path = campaign / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"repaired ComNetX manifest is missing: {manifest_path}")
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise RuntimeError("repaired ComNetX manifest is not a JSON object")
    expected_manifest = {
        "schema": "comnetx-ieee-access-repaired-campaign-v1",
        "campaign_id": campaign.name,
        "protocol_id": "ieee-access-repaired-comnetx-72h-v1",
        "production_api_acknowledged": True,
    }
    for key, expected in expected_manifest.items():
        if manifest.get(key) != expected:
            raise RuntimeError(
                f"repaired ComNetX campaign {key} differs: "
                f"{manifest.get(key)!r} != {expected!r}"
            )
    git = manifest.get("git")
    if not isinstance(git, dict) or git.get("dirty") is not False:
        raise RuntimeError("repaired ComNetX campaign did not seal a clean checkout")
    source_equivalence = repaired_source_equivalence(manifest)
    paths_sha256 = sha256_file(paths_config)
    if manifest.get("paths_config", {}).get("sha256") != paths_sha256:
        raise RuntimeError("repaired ComNetX and LD-Leiden paths maps differ")
    required_stages = (
        "stage1_correctness_smoke",
        "stage2_core_short",
        "stage4_long_core",
    )
    status = manifest.get("stage_status")
    if not isinstance(status, dict):
        raise RuntimeError("repaired ComNetX campaign lacks stage status")
    incomplete = [stage for stage in required_stages if status.get(stage) != "validated"]
    if incomplete:
        raise RuntimeError(
            "paired repaired-ComNetX evidence is not validated for: "
            + ", ".join(incomplete)
        )
    preflight = campaign / "preflight" / "validation.json"
    if not preflight.is_file() or read_json(preflight).get("status") != "validated":
        raise RuntimeError("repaired ComNetX preflight is not validated")
    hardware_registration, _ = registered_artifact(campaign, manifest, "hardware")
    input_registration, _ = registered_artifact(
        campaign, manifest, "real_input_manifest"
    )
    environment_registration, _ = registered_artifact(
        campaign, manifest, "environment"
    )
    cache = campaign / str(manifest.get("bootstrap_cache", "bootstrap-cache"))
    if not cache.is_dir():
        raise RuntimeError(f"repaired ComNetX bootstrap cache is missing: {cache}")
    return {
        "path": str(campaign),
        "campaign_id": campaign.name,
        "schema": manifest["schema"],
        "protocol_id": manifest["protocol_id"],
        "git_commit": git["commit"],
        "ld_git_commit": expected_git_sha,
        "source_equivalence": source_equivalence,
        "paths_config_sha256": paths_sha256,
        "hardware": hardware_registration,
        "real_input_manifest": input_registration,
        "environment": environment_registration,
        "required_stage_status": {stage: status[stage] for stage in required_stages},
        "bootstrap_cache": str(cache),
    }


def cpu_topology(cpu_affinity: list[int]) -> dict[str, Any]:
    if len(cpu_affinity) != 1:
        raise RuntimeError("each LD-Leiden shard must be pinned to exactly one logical CPU")
    cpu = cpu_affinity[0]
    topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
    fields = {
        "physical_package_id": topology / "physical_package_id",
        "core_id": topology / "core_id",
        "thread_siblings_list": topology / "thread_siblings_list",
    }
    values: dict[str, Any] = {"logical_cpu": cpu}
    for key, path in fields.items():
        if not path.is_file():
            raise RuntimeError(f"CPU topology field is unavailable: {path}")
        text = path.read_text(encoding="utf-8").strip()
        if key == "thread_siblings_list":
            values[key] = text
        else:
            values[key] = int(text)
    return values


def validate_window(window: Any, *, expected_budget_hours: float | None = None) -> None:
    if not isinstance(window, dict):
        raise RuntimeError("parallel measurement window is malformed")
    budget = window.get("budget_hours")
    started = window.get("started_epoch")
    deadline = window.get("deadline_epoch")
    values = (budget, started, deadline)
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        raise RuntimeError("parallel measurement window contains a non-numeric value")
    budget = float(budget)
    started = float(started)
    deadline = float(deadline)
    if not all(math.isfinite(value) for value in (budget, started, deadline)):
        raise RuntimeError("parallel measurement window contains a non-finite value")
    if not 0.0 < budget <= 24.0:
        raise RuntimeError("parallel measurement budget must be in (0, 24] hours")
    if expected_budget_hours is not None and not math.isclose(
        budget, expected_budget_hours, rel_tol=0.0, abs_tol=1e-12
    ):
        raise RuntimeError("parallel measurement budget changed on resume")
    if not math.isclose(
        deadline - started,
        budget * 3600.0,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise RuntimeError("parallel deadline is inconsistent with its start and budget")


def validate_parallel_registration(root: Path, manifest_path: Path) -> str:
    registration_path = root / "parallel_manifest_registration.json"
    if not registration_path.is_file():
        raise RuntimeError("parallel manifest registration is missing")
    registration = read_json(registration_path)
    digest = sha256_file(manifest_path)
    if (
        not isinstance(registration, dict)
        or registration.get("schema") != PARALLEL_REGISTRATION_SCHEMA
        or registration.get("filename") != manifest_path.name
        or registration.get("sha256") != digest
    ):
        raise RuntimeError("parallel manifest changed after its initial registration")
    return digest


def initialize_parallel_manifest(
    root: Path,
    *,
    run_id: str,
    expected_git_sha: str,
    paths_config: Path,
    repaired_campaign: dict[str, Any],
    budget_hours: float,
) -> dict[str, Any]:
    if not math.isfinite(budget_hours) or budget_hours <= 0 or budget_hours > 24:
        raise ValueError("the initial parallel window must be in (0, 24] hours")
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".parallel-init.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        path = root / "parallel_manifest.json"
        registration_path = root / "parallel_manifest_registration.json"
        pending_path = root / PARALLEL_PENDING_MANIFEST
        if path.is_file():
            if pending_path.exists():
                raise RuntimeError(
                    "parallel manifest and its initialization artifact both exist"
                )
            manifest = read_json(path)
            validate_parallel_registration(root, path)
        else:
            if pending_path.is_file():
                manifest = read_json(pending_path)
            else:
                if registration_path.exists():
                    raise RuntimeError(
                        "parallel manifest registration exists without its manifest"
                    )
                now = time.time()
                protocol = load_protocol()
                manifest = {
                    "schema": SCHEMA,
                    "run_id": run_id,
                    "created_at_utc": utc_now(),
                    "expected_git_sha": expected_git_sha,
                    "protocol_id": protocol["protocol_id"],
                    "paths_config": {
                        "path": str(paths_config),
                        "sha256": sha256_file(paths_config),
                    },
                    "repaired_campaign": repaired_campaign,
                    "window": {
                        "budget_hours": budget_hours,
                        "started_epoch": now,
                        "deadline_epoch": now + budget_hours * 3600.0,
                    },
                    "assignments": {
                        shard_id: {
                            **assignment,
                            "campaign": campaign_name(shard_id),
                        }
                        for shard_id, assignment in SHARDS.items()
                    },
                }
                write_json(pending_path, manifest)
            pending_digest = sha256_file(pending_path)
            registration = {
                "schema": PARALLEL_REGISTRATION_SCHEMA,
                "filename": path.name,
                "sha256": pending_digest,
            }
            if registration_path.is_file():
                if read_json(registration_path) != registration:
                    raise RuntimeError(
                        "parallel manifest initialization registration changed"
                    )
            else:
                write_json(registration_path, registration)
            pending_path.replace(path)
            validate_parallel_registration(root, path)
        validate_window(
            manifest.get("window"), expected_budget_hours=budget_hours
        )
        expected = {
            "schema": SCHEMA,
            "run_id": run_id,
            "expected_git_sha": expected_git_sha,
            "protocol_id": load_protocol()["protocol_id"],
            "repaired_campaign": repaired_campaign,
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise RuntimeError(
                    f"parallel campaign {key} differs: {manifest.get(key)!r} != {value!r}"
                )
        if manifest.get("paths_config", {}).get("sha256") != sha256_file(paths_config):
            raise RuntimeError("parallel campaign uses another paths-config file")
        if manifest.get("paths_config", {}).get("path") != str(paths_config):
            raise RuntimeError("parallel campaign uses another paths-config path")
        if manifest.get("assignments") != {
            shard_id: {**assignment, "campaign": campaign_name(shard_id)}
            for shard_id, assignment in SHARDS.items()
        }:
            raise RuntimeError("parallel shard assignment map changed")
        return manifest


def source_bootstrap_entries(
    source_dir: Path,
    shard_id: str,
) -> list[tuple[str, int]]:
    assignment = SHARDS[shard_id]
    initial_batch = 999 if assignment["phase_cli"] == "short" else 9
    entries = [("dyn_pubmed", initial_batch), ("arxivmath", initial_batch)]
    if shard_id == "01":
        entries = [
            (dataset, 999)
            for dataset in (
                "dyn_cora",
                "dyn_acm",
                "dyn_citeseer",
                "patent",
                "dyn_pubmed",
                "arxivmath",
            )
        ]
    if not source_dir.is_dir():
        raise FileNotFoundError(f"bootstrap source directory not found: {source_dir}")
    return entries


def seed_bootstrap_cache(
    repaired_campaign: dict[str, Any],
    target_campaign: Path,
    shard_id: str,
) -> dict[str, Any]:
    source_dir = Path(repaired_campaign["bootstrap_cache"])
    target_dir = target_campaign / "bootstrap-cache"
    target_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schema": "comnetx-ieee-access-ldleiden-bootstrap-source-v1",
        "repaired_campaign": repaired_campaign,
        "source_directory": str(source_dir),
        "entries": {},
    }
    for dataset, initial_batch in source_bootstrap_entries(source_dir, shard_id):
        source = _source_candidate(source_dir, dataset, initial_batch)
        labels, modularity = _load_source(source)
        labels = _canonical(labels)
        prefix = source.name.split(f"_b:{initial_batch}_by_leidenalg", 1)[0]
        target = target_dir / f"{prefix}_b:{initial_batch}_by_leidenalg.npz"
        if target.is_file():
            semantics = validate_bootstrap_cache_file(target)
            if (
                semantics["level_zero_sha256"] != _partition_sha256(labels)
                or not math.isclose(
                    float(semantics["modularity"]),
                    modularity,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise RuntimeError(f"existing shard bootstrap differs from source: {target}")
        else:
            temporary = target.with_suffix(target.suffix + ".tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    partition=labels.astype(np.int64, copy=False),
                    mod=np.asarray(modularity),
                )
            shutil.copymode(source, temporary)
            temporary.replace(target)
            semantics = validate_bootstrap_cache_file(target)
        key = f"{dataset}/{initial_batch}"
        report["entries"][key] = {
            "source": source.name,
            "source_sha256": _sha256(source),
            "target": target.name,
            "target_sha256": semantics["file_sha256"],
            "level_zero_sha256": semantics["level_zero_sha256"],
            "modularity": modularity,
        }
    write_json(target_campaign / "bootstrap_source.json", report)
    return report


def invoke_base_preflight(
    root: Path,
    shard_id: str,
    paths_config: Path,
) -> Path:
    shards_root = root / "shards"
    name = campaign_name(shard_id)
    target = shards_root / name
    command = [
        sys.executable,
        str(BASE_RUNNER),
        "--paths-config",
        str(paths_config),
        "--campaign-id",
        name,
        "--output-root",
        str(shards_root),
        "--preflight-only",
    ]
    if (target / "manifest.json").is_file():
        command.append("--resume")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    return target


def prepare_shard(args: argparse.Namespace) -> None:
    root = run_root(args.run_id)
    paths_config = resolve_project_path(args.paths_config)
    repaired_campaign = resolve_repaired_campaign(args.repaired_campaign)
    cpu_affinity = parse_cpu_set(args.cpu_set)
    require_affinity(cpu_affinity)
    require_clean_commit(args.expected_git_sha)
    source_identity = repaired_campaign_identity(
        repaired_campaign,
        expected_git_sha=args.expected_git_sha,
        paths_config=paths_config,
    )
    parallel = initialize_parallel_manifest(
        root,
        run_id=args.run_id,
        expected_git_sha=args.expected_git_sha,
        paths_config=paths_config,
        repaired_campaign=source_identity,
        budget_hours=args.budget_hours,
    )
    if time.time() >= float(parallel["window"]["deadline_epoch"]):
        raise CampaignTimeBoundary("parallel measurement window expired before preflight")
    target = invoke_base_preflight(root, args.shard, paths_config)
    bootstrap_report = seed_bootstrap_cache(source_identity, target, args.shard)
    manifest_path = target / "manifest.json"
    manifest = read_json(manifest_path)
    topology = cpu_topology(cpu_affinity)
    assignment = {
        "id": args.shard,
        **SHARDS[args.shard],
        "cpu_affinity": cpu_affinity,
        "cpu_topology": topology,
        "parallel_manifest_sha256": sha256_file(root / "parallel_manifest.json"),
        "repaired_campaign": source_identity,
        "bootstrap_source": bootstrap_report,
    }
    recorded = manifest.get("parallel_shard")
    if recorded is not None and recorded != assignment:
        raise RuntimeError("sealed parallel shard assignment changed")
    manifest["parallel_shard"] = assignment
    manifest["status"] = "parallel_preflight_complete"
    manifest["updated_at_utc"] = utc_now()
    write_json(manifest_path, manifest)
    marker = {
        "schema": "comnetx-ieee-access-ldleiden-shard-marker-v1",
        "stage": "preflight",
        "run_id": args.run_id,
        "shard": args.shard,
        "campaign": target.name,
        "parallel_manifest_sha256": sha256_file(root / "parallel_manifest.json"),
        "manifest_sha256": sha256_file(manifest_path),
        "recorded_at_utc": utc_now(),
    }
    marker_dir = root / "markers"
    marker_dir.mkdir(parents=True, exist_ok=True)
    write_json(marker_dir / f"preflight-{args.shard}.json", marker)
    (marker_dir / f"failure-{args.shard}.json").unlink(missing_ok=True)
    print(f"Shard {args.shard} preflight complete: {target}")


def load_parallel(root: Path) -> dict[str, Any]:
    path = root / "parallel_manifest.json"
    if not path.is_file():
        raise RuntimeError(f"parallel manifest is missing: {path}")
    manifest = read_json(path)
    if manifest.get("schema") != SCHEMA:
        raise RuntimeError("unexpected parallel manifest schema")
    validate_window(manifest.get("window"))
    validate_parallel_registration(root, path)
    return manifest


def run_registered_attempt(
    root: Path,
    shard_id: str,
    paths_config: Path,
    phase_cli: str,
    repeat_index: int,
    *,
    deadline_epoch: float,
    window_id: str,
) -> Path:
    from scripts.paper.ieee_access_ldleiden_72h.run_protocol import (
        completed_attempt,
        recover_interrupted_phase,
        run_attempt,
    )

    protocol = load_protocol()
    phase = cli_phase_map(protocol)[phase_cli]
    target = campaign_dir(root, shard_id)
    manifest_path = target / "manifest.json"
    manifest = read_json(manifest_path)
    recovered = recover_interrupted_phase(target, manifest, phase["id"])
    if recovered:
        print(f"Recovered {recovered} interrupted attempt(s) in {phase['id']}")
    repeat_dir = target / phase["id"] / f"repeat-{repeat_index:02d}"
    completed = completed_attempt(repeat_dir)
    if completed is not None:
        metadata = read_json(completed / "metadata.json")
        expected_affinity = sorted(os.sched_getaffinity(0))
        if metadata.get("cpu_affinity") != expected_affinity:
            raise RuntimeError("completed attempt lacks the sealed CPU affinity")
        if metadata.get("cpu_topology") != cpu_topology(expected_affinity):
            raise RuntimeError("completed attempt lacks the sealed physical-core identity")
        if metadata.get("parallel_manifest_sha256") != sha256_file(
            root / "parallel_manifest.json"
        ):
            raise RuntimeError("completed attempt belongs to another parallel manifest")
        print(f"Skipping completed {phase['id']} repeat {repeat_index}: {completed}")
        return completed
    actual_affinity = sorted(os.sched_getaffinity(0))
    attestation = {
        "cpu_affinity": actual_affinity,
        "cpu_topology": cpu_topology(actual_affinity),
        "parallel_manifest_sha256": sha256_file(
            root / "parallel_manifest.json"
        ),
    }
    sealed_assignment = manifest.get("parallel_shard", {})
    if (
        sealed_assignment.get("cpu_affinity") != attestation["cpu_affinity"]
        or sealed_assignment.get("cpu_topology") != attestation["cpu_topology"]
        or sealed_assignment.get("parallel_manifest_sha256")
        != attestation["parallel_manifest_sha256"]
    ):
        raise RuntimeError("live parallel attestation differs from shard preflight")
    manifest["phase_status"][phase["id"]] = "running"
    manifest["status"] = "parallel_running"
    manifest["updated_at_utc"] = utc_now()
    write_json(manifest_path, manifest)
    try:
        run_attempt(
            target,
            paths_config,
            protocol,
            phase,
            repeat_index,
            deadline_epoch,
            window_id,
            additional_metadata=attestation,
        )
    except BaseException as exc:
        manifest = read_json(manifest_path)
        manifest["phase_status"][phase["id"]] = "failed"
        manifest["status"] = "failed"
        manifest.setdefault("parallel_failures", {})[phase["id"]] = {
            "recorded_at_utc": utc_now(),
            "reason": f"{type(exc).__name__}: {exc}",
        }
        manifest["updated_at_utc"] = utc_now()
        write_json(manifest_path, manifest)
        raise
    manifest = read_json(manifest_path)
    manifest["phase_status"][phase["id"]] = "shard_completed"
    manifest["status"] = "parallel_attempt_complete"
    manifest["updated_at_utc"] = utc_now()
    write_json(manifest_path, manifest)
    completed = completed_attempt(repeat_dir)
    if completed is None:
        raise RuntimeError(
            f"attempt completed without a recoverable result directory: {repeat_dir}"
        )
    return completed


def run_smoke(args: argparse.Namespace) -> None:
    if args.shard != "01":
        raise ValueError("the all-six smoke gate belongs only to shard 01")
    root = run_root(args.run_id)
    parallel = load_parallel(root)
    deadline = float(parallel["window"]["deadline_epoch"])
    if time.time() >= deadline:
        raise CampaignTimeBoundary("parallel window expired before smoke")
    affinity = parse_cpu_set(args.cpu_set)
    require_affinity(affinity)
    target_manifest = read_json(campaign_dir(root, args.shard) / "manifest.json")
    if target_manifest.get("parallel_shard", {}).get("cpu_affinity") != affinity:
        raise RuntimeError("CPU affinity differs from shard preflight")
    run_registered_attempt(
        root,
        args.shard,
        resolve_project_path(args.paths_config),
        "smoke",
        1,
        deadline_epoch=deadline,
        window_id="parallel-smoke-gate",
    )
    print("All-six LD-Leiden smoke attempt completed")


def run_measurement(args: argparse.Namespace) -> None:
    root = run_root(args.run_id)
    parallel = load_parallel(root)
    affinity = parse_cpu_set(args.cpu_set)
    require_affinity(affinity)
    target = campaign_dir(root, args.shard)
    manifest = read_json(target / "manifest.json")
    assignment = SHARDS[args.shard]
    if manifest.get("parallel_shard", {}).get("cpu_affinity") != affinity:
        raise RuntimeError("CPU affinity differs from shard preflight")
    smoke_gate_path = root / "smoke_gate.json"
    if not smoke_gate_path.is_file():
        raise RuntimeError("validated smoke gate is missing")
    smoke_gate = read_json(smoke_gate_path)
    if (
        smoke_gate.get("schema")
        != "comnetx-ieee-access-ldleiden-parallel-validation-v1"
        or smoke_gate.get("campaign_id") != args.run_id
        or smoke_gate.get("stage") != "smoke"
        or smoke_gate.get("status") != "valid_smoke"
        or smoke_gate.get("preflight", {}).get("parallel_manifest_sha256")
        != sha256_file(root / "parallel_manifest.json")
    ):
        raise RuntimeError("smoke gate is malformed or was not validated")
    global_deadline = float(parallel["window"]["deadline_epoch"])
    deadline = global_deadline
    window_id = "parallel-24h-window"
    if assignment["phase_cli"] == "long":
        deadline = min(global_deadline, time.time() + LONG_TIMEOUT_SECONDS)
        window_id = "parallel-long-one-hour-limit"
    if time.time() >= deadline:
        raise CampaignTimeBoundary("measurement deadline reached before shard start")
    attempt_dir = run_registered_attempt(
        root,
        args.shard,
        resolve_project_path(args.paths_config),
        assignment["phase_cli"],
        int(assignment["repeat"]),
        deadline_epoch=deadline,
        window_id=window_id,
    )
    manifest = read_json(target / "manifest.json")
    manifest["status"] = "parallel_shard_complete"
    manifest["updated_at_utc"] = utc_now()
    write_json(target / "manifest.json", manifest)
    marker = {
        "schema": "comnetx-ieee-access-ldleiden-shard-marker-v1",
        "stage": "done",
        "run_id": args.run_id,
        "shard": args.shard,
        "campaign": target.name,
        "parallel_manifest_sha256": sha256_file(root / "parallel_manifest.json"),
        "attempt": str(attempt_dir.relative_to(root)),
        "recorded_at_utc": utc_now(),
    }
    write_json(root / "markers" / f"done-{args.shard}.json", marker)
    (root / "markers" / f"failure-{args.shard}.json").unlink(missing_ok=True)
    print(f"Shard {args.shard} measurement complete: {attempt_dir}")


def wait_for_stage(args: argparse.Namespace) -> None:
    root = run_root(args.run_id)
    parallel = load_parallel(root)
    deadline = float(parallel["window"]["deadline_epoch"])
    if args.stage == "preflight":
        expected = [root / "markers" / f"preflight-{shard}.json" for shard in SHARDS]
    elif args.stage == "done":
        expected = [root / "markers" / f"done-{shard}.json" for shard in SHARDS]
    else:
        expected = [root / "smoke_gate.json"]
    while True:
        failures = sorted((root / "markers").glob("failure-*.json"))
        failures.extend(sorted((root / "markers").glob("failure-*.shell")))
        if failures:
            names = ", ".join(path.name for path in failures)
            raise RuntimeError(f"one or more parallel shards failed: {names}")
        missing = [path for path in expected if not path.is_file()]
        if not missing:
            parallel_sha256 = sha256_file(root / "parallel_manifest.json")
            if args.stage == "smoke":
                gate = read_json(expected[0])
                if (
                    not isinstance(gate, dict)
                    or gate.get("schema")
                    != "comnetx-ieee-access-ldleiden-parallel-validation-v1"
                    or gate.get("campaign_id") != args.run_id
                    or gate.get("stage") != "smoke"
                    or gate.get("status") != "valid_smoke"
                    or gate.get("preflight", {}).get(
                        "parallel_manifest_sha256"
                    )
                    != parallel_sha256
                ):
                    raise RuntimeError("stale or malformed smoke gate")
            else:
                for path in expected:
                    marker = read_json(path)
                    shard_id = path.stem.rsplit("-", 1)[-1]
                    if (
                        not isinstance(marker, dict)
                        or marker.get("schema")
                        != "comnetx-ieee-access-ldleiden-shard-marker-v1"
                        or marker.get("stage") != args.stage
                        or marker.get("run_id") != args.run_id
                        or marker.get("shard") != shard_id
                        or marker.get("parallel_manifest_sha256")
                        != parallel_sha256
                    ):
                        raise RuntimeError(f"stale or malformed marker: {path}")
            print(f"Parallel stage ready: {args.stage}")
            return
        if time.time() >= deadline:
            names = ", ".join(path.name for path in missing)
            raise CampaignTimeBoundary(
                f"parallel window expired while waiting for {args.stage}: {names}"
            )
        time.sleep(min(5.0, max(0.1, deadline - time.time())))


def list_shards() -> None:
    for shard_id, assignment in SHARDS.items():
        print(
            f"{shard_id} phase={assignment['phase_cli']} "
            f"repeat={assignment['repeat']}"
        )


def common_parser(subparsers: Any, name: str) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(name)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--shard", required=True, choices=tuple(SHARDS))
    parser.add_argument(
        "--paths-config",
        type=Path,
        default=Path("datasets-info/paths/cn69.json"),
    )
    parser.add_argument("--cpu-set", required=True)
    return parser


def main() -> None:
    install_termination_signal_handlers()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List the eight fixed shards.")
    subparsers = parser.add_subparsers(dest="command")
    prepare = common_parser(subparsers, "prepare")
    prepare.add_argument("--repaired-campaign", type=Path, required=True)
    prepare.add_argument("--expected-git-sha", required=True)
    prepare.add_argument("--budget-hours", type=float, default=24.0)
    common_parser(subparsers, "smoke")
    common_parser(subparsers, "measure")
    wait = subparsers.add_parser("wait")
    wait.add_argument("--run-id", required=True)
    wait.add_argument("--stage", choices=("preflight", "smoke", "done"), required=True)
    args = parser.parse_args()
    if args.list:
        list_shards()
        return
    try:
        if args.command == "prepare":
            prepare_shard(args)
        elif args.command == "smoke":
            run_smoke(args)
        elif args.command == "measure":
            run_measurement(args)
        elif args.command == "wait":
            wait_for_stage(args)
        else:
            parser.error("select a command or pass --list")
    except BaseException as exc:
        if args.command in {"prepare", "smoke", "measure"}:
            root = run_root(args.run_id)
            marker_dir = root / "markers"
            marker_dir.mkdir(parents=True, exist_ok=True)
            write_json(
                marker_dir / f"failure-{args.shard}.json",
                {
                    "schema": "comnetx-ieee-access-ldleiden-shard-marker-v1",
                    "stage": "failure",
                    "run_id": args.run_id,
                    "shard": args.shard,
                    "command": args.command,
                    "parallel_manifest_sha256": (
                        sha256_file(root / "parallel_manifest.json")
                        if (root / "parallel_manifest.json").is_file()
                        else None
                    ),
                    "reason": f"{type(exc).__name__}: {exc}",
                    "recorded_at_utc": utc_now(),
                },
            )
        raise


if __name__ == "__main__":
    main()
