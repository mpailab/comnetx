#!/usr/bin/env python3
"""Run one sealed repeat of the parallel IEEE Access LD-Leiden campaign."""

from __future__ import annotations

import argparse
import fcntl
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


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BOOTSTRAP_THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
os.environ.update(BOOTSTRAP_THREAD_ENVIRONMENT)

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
BOOTSTRAP_HEARTBEAT_STALE_SECONDS = 1800.0
PARALLEL_REGISTRATION_SCHEMA = (
    "comnetx-ieee-access-ldleiden-parallel-manifest-registration-v1"
)
PARALLEL_PENDING_MANIFEST = "parallel_manifest.initializing.json"
SHARED_BOOTSTRAP_DIRNAME = "shared-bootstrap"
SHARED_BOOTSTRAP_PENDING_DIRNAME = "shared-bootstrap.initializing"
SHARED_BOOTSTRAP_SCHEMA = "comnetx-ieee-access-ldleiden-shared-bootstrap-v1"
SHARED_BOOTSTRAP_REGISTRATION_SCHEMA = (
    "comnetx-ieee-access-ldleiden-shared-bootstrap-registration-v1"
)
SHARED_BOOTSTRAP_POLICY_SCHEMA = (
    "comnetx-ieee-access-ldleiden-shared-bootstrap-policy-v1"
)
SHARED_BOOTSTRAP_MARKER_SCHEMA = (
    "comnetx-ieee-access-ldleiden-shared-bootstrap-marker-v1"
)
BOOTSTRAP_SOURCE_SCHEMA = "comnetx-ieee-access-ldleiden-bootstrap-source-v1"

SHARED_BOOTSTRAP_POLICY: dict[str, Any] = {
    "schema": SHARED_BOOTSTRAP_POLICY_SCHEMA,
    "kind": "campaign-local-generated",
    "producer_shard": "01",
    "relative_path": SHARED_BOOTSTRAP_DIRNAME,
}

EVIDENCE_SCOPE: dict[str, Any] = {
    "mode": "standalone-native-ldleiden",
    "paired_comnetx_validated": False,
    "matched_speedup_claim_allowed": False,
    "allowed_claims": [
        "native LD-Leiden timing and quality under the sealed protocol",
        "within-LD-Leiden repeatability across the registered shards",
    ],
}

BOOTSTRAP_REQUIREMENTS: tuple[tuple[str, str], ...] = (
    ("dyn_cora", "999:10"),
    ("dyn_acm", "999:10"),
    ("dyn_citeseer", "999:10"),
    ("patent", "999:10"),
    ("dyn_pubmed", "999:10"),
    ("arxivmath", "999:10"),
    ("dyn_pubmed", "9:500"),
    ("arxivmath", "9:500"),
)


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


def _failure_markers(root: Path) -> list[Path]:
    marker_dir = root / "markers"
    return sorted(
        {
            *marker_dir.glob("failure-*.json"),
            *marker_dir.glob("failure-*.shell"),
        }
    )


def _write_bootstrap_heartbeat(
    root: Path,
    parallel_manifest_sha256: str,
    *,
    state: str,
    key: str | None = None,
) -> None:
    write_json(
        root / "markers" / "bootstrap-producer-heartbeat.json",
        {
            "schema": SHARED_BOOTSTRAP_MARKER_SCHEMA,
            "stage": "bootstrap-heartbeat",
            "producer_shard": "01",
            "parallel_manifest_sha256": parallel_manifest_sha256,
            "state": state,
            "key": key,
            "recorded_epoch": time.time(),
            "recorded_at_utc": utc_now(),
        },
    )


def _validate_bootstrap_heartbeat(
    root: Path,
    parallel_manifest_sha256: str,
    *,
    campaign_started_epoch: float,
) -> None:
    path = root / "markers" / "bootstrap-producer-heartbeat.json"
    if not path.is_file():
        age = time.time() - campaign_started_epoch
        if age > BOOTSTRAP_HEARTBEAT_STALE_SECONDS:
            raise RuntimeError(
                "shared bootstrap producer did not publish a heartbeat within "
                f"{BOOTSTRAP_HEARTBEAT_STALE_SECONDS:.0f} seconds"
            )
        return
    heartbeat = read_json(path)
    recorded_epoch = heartbeat.get("recorded_epoch") if isinstance(heartbeat, dict) else None
    if (
        not isinstance(heartbeat, dict)
        or heartbeat.get("schema") != SHARED_BOOTSTRAP_MARKER_SCHEMA
        or heartbeat.get("stage") != "bootstrap-heartbeat"
        or heartbeat.get("producer_shard") != "01"
        or heartbeat.get("parallel_manifest_sha256") != parallel_manifest_sha256
        or isinstance(recorded_epoch, bool)
        or not isinstance(recorded_epoch, (int, float))
        or not math.isfinite(float(recorded_epoch))
    ):
        raise RuntimeError("shared bootstrap producer heartbeat is malformed")
    age = time.time() - float(recorded_epoch)
    if age > BOOTSTRAP_HEARTBEAT_STALE_SECONDS:
        raise RuntimeError(
            "shared bootstrap producer heartbeat is stale by "
            f"{age:.0f} seconds"
        )


def resolve_project_path(value: Path) -> Path:
    value = value.expanduser()
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


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
                    "bootstrap_authority": SHARED_BOOTSTRAP_POLICY,
                    "evidence_scope": EVIDENCE_SCOPE,
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
            "bootstrap_authority": SHARED_BOOTSTRAP_POLICY,
            "evidence_scope": EVIDENCE_SCOPE,
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


def shard_bootstrap_entries(shard_id: str) -> list[tuple[str, int]]:
    """Return the exact shared-cache entries needed by one fixed shard."""

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
    return entries


def _bootstrap_requirements() -> dict[str, str]:
    """Bind the authority to the eight inputs registered by the LD protocol."""

    expected = {
        f"{dataset}/{batch_strategy.split(':', 1)[0]}": batch_strategy
        for dataset, batch_strategy in BOOTSTRAP_REQUIREMENTS
    }
    observed: dict[str, str] = {}
    for phase in load_protocol()["phases"]:
        batch_strategy = str(phase["batch_strategy"])
        initial_batch = batch_strategy.split(":", 1)[0]
        for dataset in phase["datasets"]:
            key = f"{dataset}/{initial_batch}"
            previous = observed.setdefault(key, batch_strategy)
            if previous != batch_strategy:
                raise RuntimeError(
                    f"registered phases disagree on bootstrap strategy for {key}"
                )
    if observed != expected:
        raise RuntimeError(
            "registered LD-Leiden protocol no longer has the fixed eight "
            "bootstrap inputs"
        )
    return expected


def _shared_bootstrap_identity(
    package: Path,
    *,
    run_id: str,
    parallel_manifest_sha256: str,
) -> dict[str, Any]:
    """Validate one immutable shared authority and return its compact identity."""

    manifest_path = package / "manifest.json"
    registration_path = package / "manifest_registration.json"
    if not manifest_path.is_file() or not registration_path.is_file():
        raise RuntimeError(f"shared bootstrap authority is incomplete: {package}")
    manifest = read_json(manifest_path)
    registration = read_json(registration_path)
    manifest_sha256 = sha256_file(manifest_path)
    expected_registration = {
        "schema": SHARED_BOOTSTRAP_REGISTRATION_SCHEMA,
        "filename": manifest_path.name,
        "sha256": manifest_sha256,
    }
    if registration != expected_registration:
        raise RuntimeError("shared bootstrap manifest registration changed")
    expected_manifest = {
        "schema": SHARED_BOOTSTRAP_SCHEMA,
        "run_id": run_id,
        "parallel_manifest_sha256": parallel_manifest_sha256,
        "bootstrap_policy": SHARED_BOOTSTRAP_POLICY,
        "evidence_scope": EVIDENCE_SCOPE,
    }
    for key, expected in expected_manifest.items():
        if not isinstance(manifest, dict) or manifest.get(key) != expected:
            raise RuntimeError(f"shared bootstrap {key} differs from the campaign")
    generation = manifest.get("generation")
    if not isinstance(generation, dict):
        raise RuntimeError("shared bootstrap generation provenance is missing")
    plan_filename = generation.get("plan_filename")
    plan_sha256 = generation.get("plan_sha256")
    if (
        not isinstance(plan_filename, str)
        or Path(plan_filename).name != plan_filename
        or not isinstance(plan_sha256, str)
    ):
        raise RuntimeError("shared bootstrap generation plan registration is malformed")
    plan_path = package / plan_filename
    if not plan_path.is_file() or sha256_file(plan_path) != plan_sha256:
        raise RuntimeError("shared bootstrap generation plan changed")
    if {path.name for path in package.iterdir()} != {
        "cache",
        plan_filename,
        manifest_path.name,
        registration_path.name,
    }:
        raise RuntimeError("shared bootstrap authority contains unregistered artifacts")
    entries = manifest.get("entries")
    requirements = _bootstrap_requirements()
    if not isinstance(entries, dict) or set(entries) != set(requirements):
        raise RuntimeError("shared bootstrap authority does not contain eight fixed keys")
    cache_dir = package / "cache"
    if not cache_dir.is_dir():
        raise RuntimeError("shared bootstrap cache directory is missing")
    expected_filenames: set[str] = set()
    for key, batch_strategy in requirements.items():
        record = entries.get(key)
        if not isinstance(record, dict):
            raise RuntimeError(f"shared bootstrap entry is malformed: {key}")
        filename = record.get("filename")
        if (
            not isinstance(filename, str)
            or not filename
            or Path(filename).name != filename
        ):
            raise RuntimeError(f"shared bootstrap filename is unsafe: {key}")
        expected_filenames.add(filename)
        dataset, initial_batch = key.rsplit("/", 1)
        if record.get("dataset") != dataset:
            raise RuntimeError(f"shared bootstrap dataset identity changed: {key}")
        if record.get("batch_strategy") != batch_strategy:
            raise RuntimeError(f"shared bootstrap strategy changed: {key}")
        allowed_filenames = {
            f"{dataset}_b:{initial_batch}_by_leidenalg.npz",
            f"{dataset}-sym_b:{initial_batch}_by_leidenalg.npz",
        }
        if filename not in allowed_filenames:
            raise RuntimeError(f"shared bootstrap graph identity changed: {key}")
        semantics = validate_bootstrap_cache_file(cache_dir / filename)
        expected_semantics = {
            "file_sha256": semantics["file_sha256"],
            "level_zero_sha256": semantics["level_zero_sha256"],
            "modularity": semantics["modularity"],
            "vertices": semantics["vertices"],
        }
        for field, expected in expected_semantics.items():
            recorded = record.get(field)
            if field == "modularity":
                if (
                    isinstance(recorded, bool)
                    or not isinstance(recorded, (int, float))
                    or not math.isclose(
                        float(recorded), float(expected), rel_tol=0.0, abs_tol=1e-12
                    )
                ):
                    raise RuntimeError(f"shared bootstrap modularity changed: {key}")
            elif recorded != expected:
                raise RuntimeError(
                    f"shared bootstrap {field} changed after registration: {key}"
                )
    live_filenames = {
        path.name for path in cache_dir.iterdir() if path.is_file()
    }
    if live_filenames != expected_filenames:
        raise RuntimeError("shared bootstrap cache contains unregistered files")
    return {
        "schema": SHARED_BOOTSTRAP_SCHEMA,
        "relative_path": SHARED_BOOTSTRAP_DIRNAME,
        "manifest": {"filename": manifest_path.name, "sha256": manifest_sha256},
        "registration": {
            "filename": registration_path.name,
            "sha256": sha256_file(registration_path),
        },
        "entries_sha256": sha256_json(entries),
        "parallel_manifest_sha256": parallel_manifest_sha256,
    }


def _generate_shared_bootstrap_entry(
    cache_dir: Path,
    paths_config: Path,
    dataset: str,
    batch_strategy: str,
) -> dict[str, Any]:
    """Create or resume one canonical flat bootstrap through the launch API."""

    src_root = PROJECT_ROOT / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from datasets import Dataset  # type: ignore  # noqa: PLC0415
    from launcher import (  # type: ignore  # noqa: PLC0415
        _compute_launch_initial_partition,
        _first_snapshot,
    )

    protocol = load_protocol()
    common = protocol["common"]
    graph = Dataset(dataset, str(paths_config))
    graph.load(
        batches_strategy=batch_strategy,
        feature_mode=str(common["feature_mode"]),
    )
    if common["force_undirected"] and graph.is_directed:
        graph._force_undirected()
        graph.name = f"{dataset}-sym"
    initial_batch = batch_strategy.split(":", 1)[0]
    filename = f"{graph.name}_b:{initial_batch}_by_leidenalg.npz"
    path = cache_dir / filename
    if path.is_file():
        try:
            semantics = validate_bootstrap_cache_file(path)
        except Exception:
            # The authority is not published yet; an interrupted cache write is
            # safe to discard and regenerate under the same sealed plan.
            path.unlink()
        else:
            return {
                "dataset": dataset,
                "batch_strategy": batch_strategy,
                "filename": filename,
                "file_sha256": semantics["file_sha256"],
                "level_zero_sha256": semantics["level_zero_sha256"],
                "modularity": semantics["modularity"],
                "vertices": semantics["vertices"],
            }
    _compute_launch_initial_partition(
        adj_matrix=_first_snapshot(graph.adj),
        dataset_name=graph.name,
        init_batch_number=initial_batch,
        cache_dir=cache_dir,
        subcoms_depth=1,
        device="cpu",
        verbose=1,
        resolution=float(common["resolution"]),
    )
    semantics = validate_bootstrap_cache_file(path)
    return {
        "dataset": dataset,
        "batch_strategy": batch_strategy,
        "filename": filename,
        "file_sha256": semantics["file_sha256"],
        "level_zero_sha256": semantics["level_zero_sha256"],
        "modularity": semantics["modularity"],
        "vertices": semantics["vertices"],
    }


def seal_shared_bootstrap(
    root: Path,
    target_campaign: Path,
    paths_config: Path,
    parallel: dict[str, Any],
) -> dict[str, Any]:
    """Generate and atomically publish shard 01's campaign-local authority."""

    parallel_manifest_sha256 = sha256_file(root / "parallel_manifest.json")
    package = root / SHARED_BOOTSTRAP_DIRNAME
    pending = root / SHARED_BOOTSTRAP_PENDING_DIRNAME
    lock_path = root / ".shared-bootstrap.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if package.is_dir():
            if pending.exists():
                raise RuntimeError(
                    "shared bootstrap authority and its initialization directory "
                    "both exist"
                )
            return _shared_bootstrap_identity(
                package,
                run_id=str(parallel["run_id"]),
                parallel_manifest_sha256=parallel_manifest_sha256,
            )
        if package.exists():
            raise RuntimeError("shared bootstrap authority path is not a directory")
        pending.mkdir(parents=False, exist_ok=True)
        for temporary in pending.glob("*.tmp"):
            if temporary.is_file():
                temporary.unlink()
        campaign_manifest = read_json(target_campaign / "manifest.json")
        producer_affinity = sorted(os.sched_getaffinity(0))
        plan = {
            "schema": "comnetx-ieee-access-ldleiden-shared-bootstrap-plan-v1",
            "run_id": parallel["run_id"],
            "parallel_manifest_sha256": parallel_manifest_sha256,
            "producer_shard": "01",
            "producer_campaign": target_campaign.name,
            "producer_cpu_affinity": producer_affinity,
            "producer_cpu_topology": cpu_topology(producer_affinity),
            "expected_git_sha": parallel["expected_git_sha"],
            "protocol_id": parallel["protocol_id"],
            "protocol_fingerprint_sha256": sha256_json(
                campaign_manifest["fingerprint"]
            ),
            "paths_config": parallel["paths_config"],
            "backend_sha256": sha256_json(campaign_manifest["backend"]),
            "hardware": campaign_manifest["hardware"],
            "runtime_identity": campaign_manifest["runtime_identity"],
            "real_input_manifest": campaign_manifest["real_input_manifest"],
            "requirements": _bootstrap_requirements(),
        }
        plan_path = pending / "initialization.json"
        if plan_path.is_file():
            if read_json(plan_path) != plan:
                raise RuntimeError("shared bootstrap initialization identity changed")
        else:
            write_json(plan_path, plan)
        cache_dir = pending / "cache"
        cache_dir.mkdir(exist_ok=True)
        deadline = float(parallel["window"]["deadline_epoch"])
        entries: dict[str, dict[str, Any]] = {}
        _write_bootstrap_heartbeat(
            root, parallel_manifest_sha256, state="starting"
        )
        for key, batch_strategy in plan["requirements"].items():
            failures = _failure_markers(root)
            if failures:
                raise RuntimeError(
                    "parallel shard failed during shared bootstrap generation: "
                    + ", ".join(path.name for path in failures)
                )
            if time.time() >= deadline:
                raise CampaignTimeBoundary(
                    "parallel window expired while generating shared bootstrap"
                )
            _write_bootstrap_heartbeat(
                root,
                parallel_manifest_sha256,
                state="generating",
                key=key,
            )
            print(f"Generating or validating shared bootstrap {key}")
            entries[key] = _generate_shared_bootstrap_entry(
                cache_dir,
                paths_config,
                key.rsplit("/", 1)[0],
                batch_strategy,
            )
            _write_bootstrap_heartbeat(
                root,
                parallel_manifest_sha256,
                state="entry-complete",
                key=key,
            )
        expected_filenames = {record["filename"] for record in entries.values()}
        live_filenames = {
            path.name for path in cache_dir.iterdir() if path.is_file()
        }
        if live_filenames != expected_filenames:
            raise RuntimeError(
                "shared bootstrap initialization contains unexpected cache files"
            )
        manifest = {
            "schema": SHARED_BOOTSTRAP_SCHEMA,
            "run_id": parallel["run_id"],
            "created_at_utc": utc_now(),
            "parallel_manifest_sha256": parallel_manifest_sha256,
            "bootstrap_policy": SHARED_BOOTSTRAP_POLICY,
            "evidence_scope": EVIDENCE_SCOPE,
            "generation": {
                "producer_shard": "01",
                "producer_campaign": target_campaign.name,
                "method": "leidenalg",
                "api": "launcher._compute_launch_initial_partition",
                "resolution": float(load_protocol()["common"]["resolution"]),
                "force_undirected": bool(
                    load_protocol()["common"]["force_undirected"]
                ),
                "thread_environment": BOOTSTRAP_THREAD_ENVIRONMENT,
                "rng_control": "backend-default; no explicit seed is exposed",
                "reproducibility_policy": (
                    "archive and reuse the sealed NPZ bytes; regeneration is not "
                    "claimed to reproduce the same partition"
                ),
                "plan_filename": plan_path.name,
                "plan_sha256": sha256_file(plan_path),
            },
            "entries": entries,
        }
        manifest_path = pending / "manifest.json"
        write_json(manifest_path, manifest)
        write_json(
            pending / "manifest_registration.json",
            {
                "schema": SHARED_BOOTSTRAP_REGISTRATION_SCHEMA,
                "filename": manifest_path.name,
                "sha256": sha256_file(manifest_path),
            },
        )
        _shared_bootstrap_identity(
            pending,
            run_id=str(parallel["run_id"]),
            parallel_manifest_sha256=parallel_manifest_sha256,
        )
        pending.replace(package)
        _write_bootstrap_heartbeat(
            root, parallel_manifest_sha256, state="sealed"
        )
        return _shared_bootstrap_identity(
            package,
            run_id=str(parallel["run_id"]),
            parallel_manifest_sha256=parallel_manifest_sha256,
        )


def wait_for_shared_bootstrap(
    root: Path,
    parallel: dict[str, Any],
) -> dict[str, Any]:
    """Wait for shard 01's atomic authority while honoring failures/deadline."""

    deadline = float(parallel["window"]["deadline_epoch"])
    campaign_started = float(parallel["window"]["started_epoch"])
    parallel_manifest_sha256 = sha256_file(root / "parallel_manifest.json")
    package = root / SHARED_BOOTSTRAP_DIRNAME
    while True:
        if package.is_dir():
            return _shared_bootstrap_identity(
                package,
                run_id=str(parallel["run_id"]),
                parallel_manifest_sha256=parallel_manifest_sha256,
            )
        failures = _failure_markers(root)
        if failures:
            raise RuntimeError(
                "parallel shard failed while waiting for shared bootstrap: "
                + ", ".join(path.name for path in failures)
            )
        _validate_bootstrap_heartbeat(
            root,
            parallel_manifest_sha256,
            campaign_started_epoch=campaign_started,
        )
        if time.time() >= deadline:
            raise CampaignTimeBoundary(
                "parallel window expired while waiting for shared bootstrap authority"
            )
        time.sleep(min(5.0, max(0.1, deadline - time.time())))


def seed_bootstrap_cache(
    root: Path,
    authority: dict[str, Any],
    target_campaign: Path,
    shard_id: str,
) -> dict[str, Any]:
    """Copy exact authority bytes into one private, resumable shard cache."""

    source_dir = root / SHARED_BOOTSTRAP_DIRNAME / "cache"
    package_manifest = read_json(
        root / SHARED_BOOTSTRAP_DIRNAME / authority["manifest"]["filename"]
    )
    target_dir = target_campaign / "bootstrap-cache"
    target_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schema": BOOTSTRAP_SOURCE_SCHEMA,
        "authority": authority,
        "source_directory": str(source_dir),
        "entries": {},
    }
    for dataset, initial_batch in shard_bootstrap_entries(shard_id):
        key = f"{dataset}/{initial_batch}"
        source_record = package_manifest["entries"].get(key)
        if not isinstance(source_record, dict):
            raise RuntimeError(f"shared bootstrap authority lacks {key}")
        source = source_dir / str(source_record["filename"])
        if not source.is_file() or sha256_file(source) != source_record["file_sha256"]:
            raise RuntimeError(f"shared bootstrap authority changed: {source}")
        target = target_dir / source.name
        if target.is_file():
            semantics = validate_bootstrap_cache_file(target)
            if semantics["file_sha256"] != source_record["file_sha256"]:
                raise RuntimeError(f"existing shard bootstrap differs from source: {target}")
        else:
            temporary = target.with_suffix(target.suffix + ".tmp")
            shutil.copy2(source, temporary)
            if sha256_file(temporary) != source_record["file_sha256"]:
                temporary.unlink(missing_ok=True)
                raise RuntimeError(f"shared bootstrap copy changed bytes: {source}")
            temporary.replace(target)
            semantics = validate_bootstrap_cache_file(target)
        report["entries"][key] = {
            "source": source.name,
            "source_sha256": source_record["file_sha256"],
            "target": target.name,
            "target_sha256": semantics["file_sha256"],
            "level_zero_sha256": semantics["level_zero_sha256"],
            "modularity": semantics["modularity"],
        }
    report_path = target_campaign / "bootstrap_source.json"
    if report_path.is_file() and read_json(report_path) != report:
        raise RuntimeError("sealed shard bootstrap-source identity changed")
    write_json(report_path, report)
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
    cpu_affinity = parse_cpu_set(args.cpu_set)
    require_affinity(cpu_affinity)
    require_clean_commit(args.expected_git_sha)
    parallel = initialize_parallel_manifest(
        root,
        run_id=args.run_id,
        expected_git_sha=args.expected_git_sha,
        paths_config=paths_config,
        budget_hours=args.budget_hours,
    )
    if time.time() >= float(parallel["window"]["deadline_epoch"]):
        raise CampaignTimeBoundary("parallel measurement window expired before preflight")
    target = invoke_base_preflight(root, args.shard, paths_config)
    if args.shard == "01":
        authority = seal_shared_bootstrap(root, target, paths_config, parallel)
        write_json(
            root / "markers" / "bootstrap-ready.json",
            {
                "schema": SHARED_BOOTSTRAP_MARKER_SCHEMA,
                "stage": "bootstrap",
                "run_id": args.run_id,
                "parallel_manifest_sha256": sha256_file(
                    root / "parallel_manifest.json"
                ),
                "authority": authority,
                "recorded_at_utc": utc_now(),
            },
        )
    else:
        authority = wait_for_shared_bootstrap(root, parallel)
    bootstrap_report = seed_bootstrap_cache(
        root, authority, target, args.shard
    )
    manifest_path = target / "manifest.json"
    manifest = read_json(manifest_path)
    topology = cpu_topology(cpu_affinity)
    assignment = {
        "id": args.shard,
        **SHARDS[args.shard],
        "cpu_affinity": cpu_affinity,
        "cpu_topology": topology,
        "parallel_manifest_sha256": sha256_file(root / "parallel_manifest.json"),
        "bootstrap_authority": authority,
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
