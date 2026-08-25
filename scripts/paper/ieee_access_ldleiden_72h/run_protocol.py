#!/usr/bin/env python3
"""Run the pre-registered IEEE Access LD-Leiden measurement campaign."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import math
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .input_manifest import build_input_manifest, validate_input_manifest
    from .process_control import (
        LauncherSignalInterrupt,
        install_termination_signal_handlers,
        run_streaming_process,
    )
    from .protocol import (
        PROJECT_ROOT,
        RESULTS_ROOT,
        ValidationError,
        bootstrap_cache_reference,
        config_path,
        launcher_result_file,
        load_protocol,
        protocol_fingerprint,
        read_json,
        sha256_file,
        sha256_json,
        validate_launcher_file,
        write_json,
    )
except ImportError:  # Direct ``python path/to/run_protocol.py`` execution.
    from input_manifest import build_input_manifest, validate_input_manifest
    from process_control import (
        LauncherSignalInterrupt,
        install_termination_signal_handlers,
        run_streaming_process,
    )
    from protocol import (
        PROJECT_ROOT,
        RESULTS_ROOT,
        ValidationError,
        bootstrap_cache_reference,
        config_path,
        launcher_result_file,
        load_protocol,
        protocol_fingerprint,
        read_json,
        sha256_file,
        sha256_json,
        validate_launcher_file,
        write_json,
    )


THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
CAMPAIGN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
DEADLINE_KILL_GRACE_SECONDS = 60.0


class CampaignTimeBoundary(RuntimeError):
    """Raised when the shared campaign deadline stops an LD-Leiden attempt."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def run_text(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def git_metadata() -> dict[str, Any]:
    status = run_text(["git", "status", "--porcelain=v1", "--untracked-files=all"])
    status_text = status or ""
    return {
        "commit": run_text(["git", "rev-parse", "HEAD"]),
        "branch": run_text(["git", "branch", "--show-current"]),
        "dirty": bool(status_text),
        "status_sha256": hashlib.sha256(status_text.encode("utf-8")).hexdigest(),
    }


def collect_hardware() -> dict[str, Any]:
    module_path = PROJECT_ROOT / "scripts" / "paper" / "collect_hardware_info.py"
    spec = importlib.util.spec_from_file_location("collect_hardware_info", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load hardware collector: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.collect()


def backend_metadata() -> dict[str, Any]:
    try:
        module = importlib.import_module("dynamic_graphs_communities")
    except ImportError as exc:
        raise RuntimeError(
            "dynamic_graphs_communities is unavailable; run this protocol in "
            "the measurement container with the LD-Leiden wheel installed"
        ) from exc

    options_type = getattr(module, "AlgorithmOptions", None)
    if options_type is None:
        raise RuntimeError("LD-Leiden wheel does not expose AlgorithmOptions")
    options = options_type()
    observed_attribute = None
    observed_value = None
    for candidate in ("num_jobs", "n_jobs", "numJobs"):
        if not hasattr(options, candidate):
            continue
        observed_attribute = candidate
        observed_value = getattr(options, candidate)
        if callable(observed_value):
            observed_value = observed_value()
        break
    if observed_attribute is None:
        raise RuntimeError(
            "cannot verify the LD-Leiden worker count from AlgorithmOptions; "
            "do not collect paper timings with an unverified wheel"
        )
    if int(observed_value) != 1:
        raise RuntimeError(
            f"LD-Leiden default {observed_attribute}={observed_value!r}; "
            "the pre-registered protocol requires j=1"
        )

    version = getattr(module, "__version__", None)
    for distribution in (
        "dynamic-graphs-communities",
        "dynamic_graphs_communities",
    ):
        if version:
            break
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    module_file = Path(module.__file__).resolve() if module.__file__ else None
    artifact_hashes: dict[str, str] = {}
    if module_file is not None:
        if module_file.name == "__init__.py":
            package_root = module_file.parent
            for path in sorted(package_root.rglob("*")):
                if path.is_file() and path.suffix in {".py", ".pyi", ".so"}:
                    artifact_hashes[str(path.relative_to(package_root))] = sha256_file(path)
        else:
            artifact_hashes[module_file.name] = sha256_file(module_file)
    return {
        "module": "dynamic_graphs_communities",
        "version": version,
        "artifact_sha256": artifact_hashes,
        "algorithm_options_attribute": observed_attribute,
        "num_jobs_observed": int(observed_value),
        "num_jobs_required": 1,
        "num_jobs_verified": True,
        "num_jobs_source": "AlgorithmOptions default",
        "num_jobs_explicitly_passed_by_launcher": False,
    }


def _namespace_inode(name: str) -> int | None:
    try:
        return int((Path("/proc/self/ns") / name).stat().st_ino)
    except OSError:
        return None


def container_identity() -> dict[str, Any]:
    """Return a stable, non-plain-text identity for the current container."""

    try:
        cgroup = Path("/proc/self/cgroup").read_bytes()
    except OSError:
        cgroup = b""
    return {
        "schema": "comnetx-ieee-access-container-identity-v1",
        "node_sha256": hashlib.sha256(platform.node().encode("utf-8")).hexdigest(),
        "cgroup_sha256": hashlib.sha256(cgroup).hexdigest(),
        "pid_namespace_inode": _namespace_inode("pid"),
        "mount_namespace_inode": _namespace_inode("mnt"),
        "python_executable": str(Path(sys.executable).resolve()),
    }


def collect_runtime_identity(
    *,
    hardware: dict[str, Any] | None = None,
    backend: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind hardware, container, interpreter, and wheel to one campaign."""

    hardware = collect_hardware() if hardware is None else hardware
    backend = backend_metadata() if backend is None else backend
    return {
        "schema": "comnetx-ieee-access-ldleiden-runtime-v1",
        "hardware_sha256": sha256_json(hardware),
        "container": container_identity(),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "backend": backend,
    }


def _write_registered_json(
    campaign_dir: Path,
    filename: str,
    payload: dict[str, Any],
) -> dict[str, str]:
    path = campaign_dir / filename
    write_json(path, payload)
    return {"filename": filename, "sha256": sha256_file(path)}


def _read_registered_json(
    campaign_dir: Path,
    registration: Any,
    label: str,
) -> dict[str, Any]:
    if not isinstance(registration, dict):
        raise RuntimeError(f"campaign lacks registered {label}")
    filename = registration.get("filename")
    digest = registration.get("sha256")
    if not isinstance(filename, str) or not isinstance(digest, str):
        raise RuntimeError(f"campaign {label} registration is malformed")
    path = campaign_dir / filename
    if not path.is_file() or sha256_file(path) != digest:
        raise RuntimeError(f"campaign {label} artifact changed")
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise RuntimeError(f"campaign {label} artifact is not an object")
    return payload


def resolve_project_path(value: Path) -> Path:
    value = value.expanduser()
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def create_manifest(
    campaign_id: str,
    paths_config: Path,
    protocol: dict[str, Any],
    *,
    backend: dict[str, Any] | None = None,
) -> dict[str, Any]:
    backend = backend_metadata() if backend is None else backend
    return {
        "schema": "comnetx-ieee-access-ldleiden-campaign-v1",
        "campaign_id": campaign_id,
        "protocol_id": protocol["protocol_id"],
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "status": "created",
        "fingerprint": protocol_fingerprint(protocol),
        "git": git_metadata(),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "backend": backend,
        "worker_policy": {
            "num_jobs": 1,
            "num_jobs_source": "verified AlgorithmOptions default",
            "num_jobs_explicitly_passed_by_launcher": False,
            "thread_environment": THREAD_ENVIRONMENT,
        },
        "paths_config": {
            "filename": paths_config.name,
            "sha256": sha256_file(paths_config),
        },
        "bootstrap_cache": "bootstrap-cache",
        "phase_status": {
            phase["id"]: "pending" for phase in protocol["phases"]
        },
    }


def assert_campaign_identity(
    manifest: dict[str, Any],
    campaign_dir: Path,
    paths_config: Path,
    protocol: dict[str, Any],
    *,
    verify_input_content: bool,
) -> dict[str, str]:
    """Require the sealed source, data, hardware, container, and wheel."""

    if manifest.get("protocol_id") != protocol["protocol_id"]:
        raise RuntimeError("campaign protocol id differs from the current protocol")
    current_fingerprint = protocol_fingerprint(protocol)
    if manifest.get("fingerprint") != current_fingerprint:
        raise RuntimeError(
            "protocol, config, or measurement source changed; start a new campaign "
            "instead of mixing results"
        )
    if manifest.get("paths_config", {}).get("sha256") != sha256_file(paths_config):
        raise RuntimeError("paths config changed; start a new campaign")
    if manifest.get("backend", {}).get("num_jobs_verified") is not True:
        raise RuntimeError("campaign does not contain a verified j=1 backend")
    current_backend = backend_metadata()
    if current_backend != manifest.get("backend"):
        raise RuntimeError("LD-Leiden wheel identity changed; start a new campaign")

    sealed_hardware = _read_registered_json(
        campaign_dir,
        manifest.get("hardware"),
        "hardware",
    )
    current_hardware = collect_hardware()
    if current_hardware != sealed_hardware:
        raise RuntimeError("live hardware differs from the sealed campaign")

    sealed_runtime = _read_registered_json(
        campaign_dir,
        manifest.get("runtime_identity"),
        "runtime identity",
    )
    current_runtime = collect_runtime_identity(
        hardware=current_hardware,
        backend=current_backend,
    )
    if current_runtime != sealed_runtime:
        raise RuntimeError(
            "live container/interpreter/backend identity differs from the campaign"
        )

    sealed_inputs = _read_registered_json(
        campaign_dir,
        manifest.get("real_input_manifest"),
        "real-input manifest",
    )
    if sealed_inputs.get("paths_config_sha256") != sha256_file(paths_config):
        raise RuntimeError("input manifest was built from a different paths config")
    validate_input_manifest(
        sealed_inputs,
        verify_content=verify_input_content,
    )
    return {
        "fingerprint_sha256": sha256_json(current_fingerprint),
        "hardware_sha256": manifest["hardware"]["sha256"],
        "runtime_identity_sha256": manifest["runtime_identity"]["sha256"],
        "real_input_manifest_sha256": manifest["real_input_manifest"]["sha256"],
    }


def assert_resume_compatible(
    manifest: dict[str, Any],
    campaign_dir: Path,
    paths_config: Path,
    protocol: dict[str, Any],
) -> None:
    assert_campaign_identity(
        manifest,
        campaign_dir,
        paths_config,
        protocol,
        verify_input_content=True,
    )


def completed_attempt(repeat_dir: Path) -> Path | None:
    completed = []
    for metadata_path in sorted(repeat_dir.glob("attempt-*/metadata.json")):
        metadata = read_json(metadata_path)
        if metadata.get("status") == "completed":
            completed.append(metadata_path.parent)
    if len(completed) > 1:
        raise RuntimeError(f"multiple completed attempts in {repeat_dir}")
    return completed[0] if completed else None


def next_attempt_dir(repeat_dir: Path) -> Path:
    indices = []
    for path in repeat_dir.glob("attempt-*"):
        try:
            indices.append(int(path.name.split("-", 1)[1]))
        except (IndexError, ValueError):
            continue
    return repeat_dir / f"attempt-{max(indices, default=0) + 1:02d}"


def recover_interrupted_phase(
    campaign_dir: Path,
    manifest: dict[str, Any],
    phase_id: str,
) -> int:
    """Recover stale LD attempt markers only after their process groups are gone."""

    if manifest.get("phase_status", {}).get(phase_id) != "running":
        return 0
    recovered = 0
    phase_dir = campaign_dir / phase_id
    for repeat_dir in sorted(path for path in phase_dir.glob("repeat-*") if path.is_dir()):
        records = []
        for metadata_path in sorted(repeat_dir.glob("attempt-*/metadata.json")):
            metadata = read_json(metadata_path)
            if metadata.get("status") == "running":
                process_group = metadata.get("process_group_id")
                if not isinstance(process_group, int) or process_group <= 0:
                    raise RuntimeError(
                        f"cannot prove interrupted process cleanup for {metadata_path}"
                    )
                try:
                    os.killpg(process_group, 0)
                except ProcessLookupError:
                    pass
                except PermissionError as exc:
                    raise RuntimeError(
                        f"recorded process group is still present: {process_group}"
                    ) from exc
                else:
                    raise RuntimeError(
                        f"recorded process group is still present: {process_group}"
                    )
                metadata.update(
                    {
                        "status": "failed",
                        "failure_kind": "launcher_interrupted",
                        "recovered_at_utc": utc_now(),
                    }
                )
                write_json(metadata_path, metadata)
                recovered += 1
            records.append((metadata_path, metadata))
        has_completed = any(
            metadata.get("status") == "completed" for _, metadata in records
        )
        for metadata_path, metadata in records:
            status = metadata.get("status")
            if status == "completed":
                continue
            if status == "failed" and (
                has_completed
                or metadata.get("failure_kind")
                in {"campaign_time_boundary", "launcher_interrupted"}
            ):
                continue
            raise RuntimeError(
                f"stale phase contains an unaudited failure: {metadata_path}"
            )
    manifest["phase_status"][phase_id] = "failed"
    manifest.setdefault("interrupted_phases", {})[phase_id] = {
        "recorded_at_utc": utc_now(),
        "reason": "recovered stale running state after process-group absence check",
        "recovered_attempts": recovered,
    }
    manifest["updated_at_utc"] = utc_now()
    write_json(campaign_dir / "manifest.json", manifest)
    return recovered


def bootstrap_cache_fingerprint(cache_dir: Path) -> dict[str, str]:
    return {
        path.name: sha256_file(path)
        for path in sorted(cache_dir.glob("*.npz"))
        if path.is_file()
    }


def stream_process(
    command: list[str],
    env: dict[str, str],
    log_path: Path,
    *,
    deadline: float | None = None,
    on_start: Any = None,
) -> tuple[int, bool]:
    return run_streaming_process(
        command,
        env,
        log_path,
        cwd=PROJECT_ROOT,
        deadline=deadline,
        deadline_kill_grace_seconds=DEADLINE_KILL_GRACE_SECONDS,
        on_start=on_start,
    )


def run_attempt(
    campaign_dir: Path,
    paths_config: Path,
    protocol: dict[str, Any],
    phase: dict[str, Any],
    repeat_index: int,
    deadline_epoch: float | None,
) -> dict[str, Any]:
    if deadline_epoch is not None and time.time() >= deadline_epoch:
        raise CampaignTimeBoundary(
            f"shared campaign deadline reached before {phase['id']} repeat {repeat_index}"
        )
    campaign_manifest = read_json(campaign_dir / "manifest.json")
    identity_before = assert_campaign_identity(
        campaign_manifest,
        campaign_dir,
        paths_config,
        protocol,
        verify_input_content=True,
    )
    repeat_dir = campaign_dir / phase["id"] / f"repeat-{repeat_index:02d}"
    attempt_dir = next_attempt_dir(repeat_dir)
    attempt_dir.mkdir(parents=True, exist_ok=False)
    result_path = launcher_result_file(attempt_dir, phase)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "launch.py"),
        str(config_path(phase)),
        "--paths-config",
        str(paths_config),
    ]
    env = os.environ.copy()
    env.update(THREAD_ENVIRONMENT)
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "RESULTS_DIR": str(attempt_dir),
            "COMNETX_CACHE_DIR": str(campaign_dir / "bootstrap-cache"),
        }
    )
    metadata = {
        "schema": "comnetx-ieee-access-ldleiden-attempt-v1",
        "phase_id": phase["id"],
        "role": phase["role"],
        "included_in_analysis": phase["included_in_analysis"],
        "repeat": repeat_index,
        "attempt": int(attempt_dir.name.split("-")[1]),
        "status": "running",
        "started_at_utc": utc_now(),
        "command": [
            sys.executable,
            "scripts/launch.py",
            str(config_path(phase).relative_to(PROJECT_ROOT)),
            "--paths-config",
            paths_config.name,
        ],
        "thread_environment": THREAD_ENVIRONMENT,
        "result_file": result_path.name,
        "stdout_log": "stdout.log",
        "identity_before": identity_before,
    }
    metadata_path = attempt_dir / "metadata.json"
    write_json(metadata_path, metadata)

    start = time.perf_counter()
    return_code = None
    deadline_reached = False
    process_error: BaseException | None = None
    def record_process(process_id: int) -> None:
        metadata["process_id"] = process_id
        metadata["process_group_id"] = process_id
        write_json(metadata_path, metadata)

    command_deadline = (
        None
        if deadline_epoch is None
        else time.monotonic() + deadline_epoch - time.time()
    )
    try:
        return_code, deadline_reached = stream_process(
            command,
            env,
            attempt_dir / "stdout.log",
            deadline=command_deadline,
            on_start=record_process,
        )
    except BaseException as exc:  # Preserve interrupts while sealing post-state.
        process_error = exc
    metadata["finished_at_utc"] = utc_now()
    metadata["wall_seconds"] = time.perf_counter() - start
    metadata["return_code"] = return_code
    metadata["deadline_reached"] = deadline_reached
    try:
        identity_after = assert_campaign_identity(
            campaign_manifest,
            campaign_dir,
            paths_config,
            protocol,
            verify_input_content=True,
        )
    except Exception as exc:
        metadata["status"] = "failed_identity"
        metadata["identity_error"] = f"{type(exc).__name__}: {exc}"
        write_json(metadata_path, metadata)
        raise
    metadata["identity_after"] = identity_after
    if identity_after != identity_before:
        metadata["status"] = "failed_identity"
        write_json(metadata_path, metadata)
        raise RuntimeError(
            f"{phase['id']} repeat {repeat_index}: sealed identity changed during run"
        )
    if process_error is not None:
        metadata["status"] = "failed"
        metadata["failure_kind"] = "launcher_interrupted"
        metadata["process_error"] = (
            f"{type(process_error).__name__}: {process_error}"
        )
        write_json(metadata_path, metadata)
        raise process_error
    if deadline_reached:
        metadata["status"] = "failed"
        metadata["failure_kind"] = "campaign_time_boundary"
        write_json(metadata_path, metadata)
        raise CampaignTimeBoundary(
            f"{phase['id']} repeat {repeat_index} stopped at the shared deadline"
        )
    if return_code != 0:
        metadata["status"] = "failed"
        write_json(metadata_path, metadata)
        raise RuntimeError(
            f"{phase['id']} repeat {repeat_index} failed with exit code {return_code}; "
            f"see {attempt_dir / 'stdout.log'}"
        )
    if not result_path.is_file():
        metadata["status"] = "failed_validation"
        metadata["validation_error"] = f"missing result file: {result_path.name}"
        write_json(metadata_path, metadata)
        raise RuntimeError(metadata["validation_error"])

    try:
        summary = validate_launcher_file(result_path, phase)
    except ValidationError as exc:
        metadata["status"] = "failed_validation"
        metadata["validation_error"] = str(exc)
        write_json(metadata_path, metadata)
        raise
    metadata["result_sha256"] = sha256_file(result_path)
    cache_dir = campaign_dir / "bootstrap-cache"
    cache_hashes = bootstrap_cache_fingerprint(cache_dir)
    metadata["bootstrap_cache_sha256"] = cache_hashes
    metadata["bootstrap_semantics"] = {
        dataset: bootstrap_cache_reference(
            cache_dir,
            cache_hashes,
            dataset,
            phase["batch_strategy"],
            summary[dataset]["bootstrap_reference_sha256"],
        )
        for dataset in phase["datasets"]
    }
    metadata["validated_summary"] = summary
    metadata["status"] = "completed"
    write_json(metadata_path, metadata)
    return metadata


def selected_phases(
    protocol: dict[str, Any],
    names: list[str],
) -> list[dict[str, Any]]:
    if not names or "all" in names:
        return list(protocol["phases"])
    selected = set(names)
    return [
        phase for phase in protocol["phases"]
        if phase["cli_name"] in selected
    ]


def main() -> None:
    install_termination_signal_handlers()
    protocol = load_protocol()
    parser = argparse.ArgumentParser(
        description="Run the pre-registered 72-hour LD-Leiden measurement plan."
    )
    parser.add_argument(
        "--paths-config",
        type=Path,
        required=True,
        help="Dataset path map on the measurement server.",
    )
    parser.add_argument(
        "--campaign-id",
        default=datetime.now(timezone.utc).strftime("ldleiden-%Y%m%dT%H%M%SZ"),
        help="Stable directory name for this measurement campaign.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=RESULTS_ROOT,
        help="Root below which the campaign directory is created.",
    )
    parser.add_argument(
        "--phase",
        action="append",
        choices=("all", "smoke", "short", "long"),
        default=[],
        help="Run one phase; repeat to select several. Default: all in order.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the same campaign, preserving failed attempts and skipping completed ones.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Create/seal the campaign by hashing source and inputs and locking "
            "hardware, container, wheel, and j=1; no graph run is executed."
        ),
    )
    parser.add_argument(
        "--deadline-epoch",
        type=float,
        help="Absolute deadline of the shared repaired/LD measurement campaign.",
    )
    args = parser.parse_args()

    if args.deadline_epoch is not None and not math.isfinite(args.deadline_epoch):
        parser.error("--deadline-epoch must be finite")

    if not CAMPAIGN_RE.fullmatch(args.campaign_id):
        parser.error("--campaign-id may contain only letters, digits, dot, underscore, and dash")
    paths_config = resolve_project_path(args.paths_config)
    if not paths_config.is_file():
        parser.error(f"paths config does not exist: {paths_config}")
    output_root = resolve_project_path(args.output_root)
    campaign_dir = output_root / args.campaign_id
    manifest_path = campaign_dir / "manifest.json"

    if manifest_path.exists():
        if not args.resume:
            parser.error(
                f"campaign already exists: {campaign_dir}; pass --resume or use a new id"
            )
        manifest = read_json(manifest_path)
        assert_resume_compatible(
            manifest,
            campaign_dir,
            paths_config,
            protocol,
        )
    else:
        if args.resume:
            parser.error(f"cannot resume missing campaign: {campaign_dir}")
        if campaign_dir.exists():
            parser.error(
                f"campaign directory exists without a manifest: {campaign_dir}; "
                "use a new campaign id and preserve the incomplete directory for audit"
            )
        if not args.preflight_only:
            parser.error(
                "create the sealed campaign with --preflight-only before any phase"
            )
        backend = backend_metadata()
        hardware = collect_hardware()
        runtime_identity = collect_runtime_identity(
            hardware=hardware,
            backend=backend,
        )
        input_manifest = build_input_manifest(paths_config)
        validate_input_manifest(input_manifest, verify_content=True)
        manifest = create_manifest(
            args.campaign_id,
            paths_config,
            protocol,
            backend=backend,
        )
        campaign_dir.mkdir(parents=True, exist_ok=False)
        manifest["hardware"] = _write_registered_json(
            campaign_dir,
            "hardware.json",
            hardware,
        )
        manifest["runtime_identity"] = _write_registered_json(
            campaign_dir,
            "runtime_identity.json",
            runtime_identity,
        )
        manifest["real_input_manifest"] = _write_registered_json(
            campaign_dir,
            "real_input_manifest.json",
            input_manifest,
        )
        manifest["status"] = "preflight_complete"
        manifest["updated_at_utc"] = utc_now()
        write_json(manifest_path, manifest)

    print(f"Protocol: {protocol['protocol_id']}")
    print(f"Campaign: {campaign_dir}")
    print("Verified LD-Leiden workers: j=1")
    print(f"Protocol SHA-256: {manifest['fingerprint']['protocol_sha256']}")
    if manifest["git"].get("dirty"):
        print(
            "WARNING: the worktree was dirty at campaign creation; exact measurement "
            "source hashes are recorded in manifest.json."
        )
    if args.preflight_only:
        print(
            "Preflight complete; source, input content, hardware, container, "
            "wheel, and j=1 were sealed; no measurements were run."
        )
        return

    phases = selected_phases(protocol, args.phase)
    runs_measured_phase = any(phase["cli_name"] != "smoke" for phase in phases)
    smoke_is_selected = any(phase["cli_name"] == "smoke" for phase in phases)
    smoke_is_complete = manifest["phase_status"]["smoke_999_10"] == "completed"
    if runs_measured_phase and not (smoke_is_selected or smoke_is_complete):
        parser.error(
            "the diagnostic smoke phase must complete before measured phases; "
            "run --phase smoke first or select it in the same command"
        )
    manifest["status"] = "running"
    manifest["updated_at_utc"] = utc_now()
    write_json(manifest_path, manifest)
    active_phase: dict[str, Any] | None = None
    active_phase_started = False
    try:
        for phase in phases:
            active_phase = phase
            active_phase_started = False
            recovered = recover_interrupted_phase(
                campaign_dir,
                manifest,
                phase["id"],
            )
            if recovered:
                print(f"Recovered {recovered} interrupted {phase['id']} attempt(s)")
            print(
                f"Starting {phase['id']}: {phase['repetitions']} repetition(s), "
                f"included_in_analysis={phase['included_in_analysis']}"
            )
            manifest["phase_status"][phase["id"]] = "running"
            active_phase_started = True
            manifest["updated_at_utc"] = utc_now()
            write_json(manifest_path, manifest)
            for repeat_index in range(1, int(phase["repetitions"]) + 1):
                repeat_dir = campaign_dir / phase["id"] / f"repeat-{repeat_index:02d}"
                done = completed_attempt(repeat_dir)
                if done is not None:
                    print(f"Skipping completed {phase['id']} repeat {repeat_index}: {done}")
                    continue
                print(f"Running {phase['id']} repeat {repeat_index}")
                run_attempt(
                    campaign_dir,
                    paths_config,
                    protocol,
                    phase,
                    repeat_index,
                    args.deadline_epoch,
                )
            manifest["phase_status"][phase["id"]] = "completed"
            manifest["updated_at_utc"] = utc_now()
            write_json(manifest_path, manifest)
    except BaseException as exc:
        manifest["status"] = "failed"
        if active_phase is not None and active_phase_started:
            manifest["phase_status"][active_phase["id"]] = "failed"
            if isinstance(exc, CampaignTimeBoundary):
                manifest.setdefault("time_boundary_stops", {})[
                    active_phase["id"]
                ] = {
                    "recorded_at_utc": utc_now(),
                    "deadline_epoch": args.deadline_epoch,
                    "reason": str(exc),
                }
            elif isinstance(exc, (KeyboardInterrupt, LauncherSignalInterrupt)):
                manifest.setdefault("interrupted_phases", {})[
                    active_phase["id"]
                ] = {
                    "recorded_at_utc": utc_now(),
                    "reason": f"{type(exc).__name__}: {exc}",
                }
        manifest["updated_at_utc"] = utc_now()
        write_json(manifest_path, manifest)
        raise

    all_complete = all(
        status == "completed" for status in manifest["phase_status"].values()
    )
    manifest["status"] = "completed" if all_complete else "partial"
    manifest["updated_at_utc"] = utc_now()
    write_json(manifest_path, manifest)
    print(f"Selected phases complete; campaign status: {manifest['status']}")


if __name__ == "__main__":
    main()
