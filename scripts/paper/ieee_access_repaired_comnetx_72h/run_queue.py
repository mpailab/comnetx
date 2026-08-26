#!/usr/bin/env python3
"""Run one gated stage of the repaired-ComNetX 72-hour evidence queue."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .input_manifest import (
        build_dsbm_input_manifest,
        validate_input_manifest,
    )
    from .process_control import (
        LauncherSignalInterrupt,
        install_termination_signal_handlers,
        run_streaming_process,
    )
    from .protocol import (
        ALL_DATASETS,
        CORE_DATASETS,
        PACKAGE_DIR,
        PROJECT_ROOT,
        RESULTS_ROOT,
        ValidationError,
        bootstrap_cache_hashes,
        config_path,
        json_file_hashes,
        load_protocol,
        protocol_fingerprint,
        read_json,
        sha256_file,
        stage_map,
        write_json,
    )
    from .validate_campaign import validate_stage
except ImportError:  # Direct script execution.
    from input_manifest import build_dsbm_input_manifest, validate_input_manifest
    from process_control import (
        LauncherSignalInterrupt,
        install_termination_signal_handlers,
        run_streaming_process,
    )
    from protocol import (
        ALL_DATASETS,
        CORE_DATASETS,
        PACKAGE_DIR,
        PROJECT_ROOT,
        RESULTS_ROOT,
        ValidationError,
        bootstrap_cache_hashes,
        config_path,
        json_file_hashes,
        load_protocol,
        protocol_fingerprint,
        read_json,
        sha256_file,
        stage_map,
        write_json,
    )
    from validate_campaign import validate_stage


CAMPAIGN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
DEADLINE_KILL_GRACE_SECONDS = 60.0


class CampaignTimeBoundary(RuntimeError):
    """Raised when a registered campaign-time boundary stops a command."""


@dataclass(frozen=True)
class CommandSpec:
    name: str
    command: list[str]
    environment: dict[str, str] = field(default_factory=dict)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def resolve_project_path(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def run_text(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
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
    return completed.stdout.strip() if completed.returncode == 0 else None


def git_metadata() -> dict[str, Any]:
    status = run_text(["git", "status", "--porcelain=v1", "--untracked-files=all"])
    status = status or ""
    return {
        "commit": run_text(["git", "rev-parse", "HEAD"]),
        "branch": run_text(["git", "branch", "--show-current"]),
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }


def collect_hardware() -> dict[str, Any]:
    path = PROJECT_ROOT / "scripts" / "paper" / "collect_hardware_info.py"
    spec = importlib.util.spec_from_file_location("collect_hardware_info", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load hardware collector: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.collect()


def collect_measurement_environment() -> dict[str, Any]:
    path = PACKAGE_DIR / "check_measurement_environment.py"
    spec = importlib.util.spec_from_file_location(
        "check_measurement_environment", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load environment collector: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.inspect_environment()
    module.validate_environment(report)
    # Match the JSON representation stored by the preflight command (notably,
    # importlib returns distribution pairs as tuples, while JSON reads lists).
    return json.loads(json.dumps(report, sort_keys=True))


def assert_runtime_identity(
    manifest: dict[str, Any],
    campaign_dir: Path,
) -> dict[str, str | None]:
    """Require the live hardware/software identity sealed by the campaign."""
    hardware_registration = manifest.get("hardware", {})
    hardware_path = campaign_dir / hardware_registration.get("filename", "")
    if (
        not hardware_path.is_file()
        or hardware_registration.get("sha256") != sha256_file(hardware_path)
    ):
        raise RuntimeError("campaign hardware metadata changed")
    if collect_hardware() != read_json(hardware_path):
        raise RuntimeError("live hardware differs from the sealed campaign")

    environment_digest = None
    environment_registration = manifest.get("environment")
    if environment_registration is not None:
        environment_path = campaign_dir / environment_registration.get(
            "filename", ""
        )
        if (
            not environment_path.is_file()
            or environment_registration.get("sha256")
            != sha256_file(environment_path)
        ):
            raise RuntimeError("campaign software-environment metadata changed")
        if collect_measurement_environment() != read_json(environment_path):
            raise RuntimeError(
                "live CUDA/package environment differs from the sealed campaign"
            )
        environment_digest = environment_registration["sha256"]

    return {
        "hardware_sha256": hardware_registration["sha256"],
        "environment_sha256": environment_digest,
    }


def make_manifest(
    protocol: dict[str, Any],
    campaign_id: str,
    paths_config: Path,
) -> dict[str, Any]:
    return {
        "schema": "comnetx-ieee-access-repaired-campaign-v1",
        "campaign_id": campaign_id,
        "protocol_id": protocol["protocol_id"],
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "status": "created",
        "production_api_acknowledged": True,
        "historical_smart_l_ge_2_status": "provisional",
        "fingerprint": protocol_fingerprint(protocol),
        "git": git_metadata(),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "paths_config": {
            "filename": paths_config.name,
            "sha256": sha256_file(paths_config),
        },
        "bootstrap_cache": "bootstrap-cache",
        "stage_status": {stage["id"]: "pending" for stage in protocol["stages"]},
    }


def assert_campaign_compatible(
    manifest: dict[str, Any],
    protocol: dict[str, Any],
    paths_config: Path,
    campaign_dir: Path,
) -> None:
    if manifest.get("production_api_acknowledged") is not True:
        raise RuntimeError("campaign lacks the production-API acknowledgment")
    if manifest.get("protocol_id") != protocol["protocol_id"]:
        raise RuntimeError("campaign protocol id changed")
    if manifest.get("fingerprint") != protocol_fingerprint(protocol):
        raise RuntimeError(
            "repaired source, profiler, audit, validator, or config changed; "
            "start a new campaign instead of mixing measurements"
        )
    if manifest.get("paths_config", {}).get("sha256") != sha256_file(paths_config):
        raise RuntimeError("dataset paths config changed; start a new campaign")
    assert_runtime_identity(manifest, campaign_dir)
    assert_sealed_inputs(manifest, campaign_dir=campaign_dir)


def assert_sealed_inputs(
    manifest: dict[str, Any],
    *,
    campaign_dir: Path,
    verify_content: bool = False,
) -> None:
    """Verify sealed real/DSBM inputs; a pre-preflight campaign has none yet."""
    for key in ("real_input_manifest", "dsbm_input_manifest"):
        registered = manifest.get(key)
        if registered is None:
            continue
        path = campaign_dir / registered["filename"]
        if not path.is_file() or sha256_file(path) != registered["sha256"]:
            raise RuntimeError(f"sealed input manifest changed: {path}")
        validate_input_manifest(read_json(path), verify_content=verify_content)


def launcher_spec(name: str, relative_config: str, paths_config: Path) -> CommandSpec:
    return CommandSpec(
        name=name,
        command=[
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "launch.py"),
            str(config_path(relative_config)),
            "--paths-config",
            str(paths_config),
        ],
        environment={"launcher_config": relative_config},
    )


def build_stage_commands(
    stage_id: str,
    *,
    campaign_dir: Path,
    paths_config: Path,
    dsbm_root: Path | None,
    repetitions: int | None,
    dsbm_seed: int | None = None,
) -> list[CommandSpec]:
    cache_dir = campaign_dir / "bootstrap-cache"
    output_root = campaign_dir / "stages" / stage_id
    if stage_id == "stage1_correctness_smoke":
        commands = [
            launcher_spec(
                "all_six_smart_smoke",
                "configs/stage1_smart_smoke_999_10.json",
                paths_config,
            )
        ]
        for dataset in ALL_DATASETS:
            commands.append(
                CommandSpec(
                    name=f"invariant_audit_{dataset}",
                    command=[
                        sys.executable,
                        str(PROJECT_ROOT / "scripts" / "paper" / "audit_stream_invariants.py"),
                        "--dataset",
                        dataset,
                        "--batch-strategy",
                        "999:10",
                        "--paths-config",
                        str(paths_config),
                        "--depth",
                        "3",
                        "--radius",
                        "1",
                        "--aggregation-mode",
                        "sum",
                        "--cache-dir",
                        str(cache_dir),
                        "--force-undirected",
                        "--use-gpu",
                        "--output",
                        "{attempt_dir}/audit.json",
                    ],
                )
            )
        return commands
    if stage_id == "stage2_core_short":
        return [
            launcher_spec(
                f"paired_repeat_{repeat:02d}",
                "configs/stage2_paired_short_999_10.json",
                paths_config,
            )
            for repeat in range(1, 6)
        ]
    if stage_id == "stage3_mechanism":
        common = [
            "--datasets",
            "dyn_cora",
            "dyn_pubmed",
            "arxivmath",
            "--methods",
            "leidenalg",
            "--paths-config",
            str(paths_config),
            "--cache-dir",
            str(cache_dir),
            "--smart-depth",
            "3",
            "--smart-radius",
            "1",
            "--aggregation-mode",
            "sum",
            "--max-updates",
            "50",
            "--use-gpu",
            "--force-undirected",
            "--ground-truth-metrics",
        ]
        script = str(PROJECT_ROOT / "scripts" / "paper" / "profile_smart_workload.py")
        return [
            CommandSpec(
                name="production_profile_parity",
                command=[
                    sys.executable,
                    script,
                    *common,
                    "--batches",
                    "999:10",
                    "--variants",
                    "full",
                    "--output-dir",
                    "{attempt_dir}",
                    "--name",
                    "parity",
                ],
            ),
            CommandSpec(
                name="closure_contraction_profiles",
                command=[
                    sys.executable,
                    script,
                    *common,
                    "--batches",
                    "999:50",
                    "--variants",
                    "full",
                    "no_closure",
                    "no_contraction",
                    "--output-dir",
                    "{attempt_dir}",
                    "--name",
                    "mechanism",
                ],
            ),
        ]
    if stage_id == "stage4_long_core":
        return [
            launcher_spec(
                "fresh_paired_long",
                "configs/stage4_paired_long_9_500.json",
                paths_config,
            )
        ]
    if stage_id == "stage4_long_repeatability":
        count = 2 if repetitions is None else repetitions
        if count not in (1, 2):
            raise ValueError("long repeatability accepts one or two repetitions")
        return [
            launcher_spec(
                f"smart_long_repeat_{repeat:02d}",
                "configs/stage4_smart_long_9_500.json",
                paths_config,
            )
            for repeat in range(1, count + 1)
        ]
    if stage_id == "stage5_topology_controls":
        commands = [
            launcher_spec(
                "topology_grid",
                "configs/stage5_topology_grid_999_10.json",
                paths_config,
            ),
            launcher_spec(
                "directed_control",
                "configs/stage5_directed_control_999_10.json",
                paths_config,
            ),
            CommandSpec(
                name="cut_metrics",
                command=[
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "paper" / "compute_leiden_cut_metrics.py"),
                    "--datasets",
                    *CORE_DATASETS,
                    "--modes",
                    "naive",
                    "smart",
                    "--batch",
                    "999:10",
                    "--smart-depth",
                    "3",
                    "--smart-radius",
                    "1",
                    "--cache-dir",
                    str(cache_dir),
                    "--paths-config",
                    str(paths_config),
                    "--out",
                    "{attempt_dir}/cut_metrics.json",
                    "--use-gpu",
                ],
            ),
            launcher_spec(
                "resolution_control",
                "configs/stage5_resolution_control_999_50.json",
                paths_config,
            ),
        ]
        return commands
    if stage_id == "stage6_dsbm":
        if dsbm_root is None:
            raise ValueError("--dsbm-root is required for stage6_dsbm")
        if dsbm_seed is not None and dsbm_seed not in (42, 43, 44, 45, 46):
            raise ValueError("DSBM seed must be one of 42, 43, 44, 45, or 46")
        seeds = (dsbm_seed,) if dsbm_seed is not None else (42, 43, 44, 45, 46)
        return [
            CommandSpec(
                name=(
                    f"dsbm_seed_{seed}_{regime}_mc{max_changes}_paired"
                ),
                command=[
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "paper" / "run_dsbm_stress.py"),
                    "--root",
                    str(dsbm_root),
                    "--batch-suffix",
                    "100_batches",
                    "--regimes",
                    regime,
                    "--max-changes",
                    str(max_changes),
                    "--seeds",
                    str(seed),
                    "--methods",
                    "leidenalg",
                    "--modes",
                    "naive",
                    "smart",
                    "--smart-depth",
                    "3",
                    "--smart-radius",
                    "1",
                    "--cache-dir",
                    str(cache_dir),
                    "--use-gpu",
                    "--output-dir",
                    "{attempt_dir}",
                    "--name",
                    (
                        f"repaired_dsbm_seed_{seed}_{regime}_"
                        f"mc{max_changes}_paired"
                    ),
                ],
            )
            for seed in seeds
            for max_changes in (290, 1450)
            for regime in ("random", "hubs", "community")
        ]
    if stage_id == "stage7_dfleiden_interface":
        commands = [
            launcher_spec(
                f"dfleiden_smart_repeat_{repeat:02d}",
                "configs/stage7_dfleiden_smart_999_10.json",
                paths_config,
            )
            for repeat in range(1, 6)
        ]
        commands.append(
            launcher_spec(
                "dfleiden_smart_long_coverage",
                "configs/stage7_dfleiden_smart_9_500.json",
                paths_config,
            )
        )
        return commands
    if stage_id == "stage7_s2cag_interface":
        count = 1 if repetitions is None else repetitions
        if not 1 <= count <= 5:
            raise ValueError("S2CAG interface accepts one to five repetitions")
        commands = [
            launcher_spec(
                f"s2cag_smart_repeat_{repeat:02d}",
                "configs/stage7_s2cag_smart_999_10.json",
                paths_config,
            )
            for repeat in range(1, count + 1)
        ]
        commands.append(
            launcher_spec(
                "s2cag_feature_modes_coverage",
                "configs/stage7_s2cag_feature_modes_999_100.json",
                paths_config,
            )
        )
        return commands
    raise ValueError(f"unknown stage: {stage_id}")


def completed_attempt(command_dir: Path) -> Path | None:
    completed = []
    for metadata_path in sorted(command_dir.glob("attempt-*/metadata.json")):
        if read_json(metadata_path).get("status") == "completed":
            completed.append(metadata_path.parent)
    if len(completed) > 1:
        raise RuntimeError(f"multiple completed attempts in {command_dir}")
    return completed[0] if completed else None


def dsbm_seed_progress(
    campaign_dir: Path,
) -> tuple[list[int], int | None, int]:
    """Return complete seeds, first incomplete seed, and its completed cells."""
    conditions = tuple(
        (regime, max_changes)
        for max_changes in (290, 1450)
        for regime in ("random", "hubs", "community")
    )
    completed_seeds: list[int] = []
    saw_incomplete = False
    next_seed: int | None = None
    next_seed_completed_conditions = 0
    for seed in (42, 43, 44, 45, 46):
        completed_conditions = 0
        for regime, max_changes in conditions:
            command_dir = (
                campaign_dir
                / "stages"
                / "stage6_dsbm"
                / f"dsbm_seed_{seed}_{regime}_mc{max_changes}_paired"
            )
            completed_conditions += completed_attempt(command_dir) is not None
        if completed_conditions == len(conditions):
            if saw_incomplete:
                raise RuntimeError(
                    "completed DSBM seed appears after an incomplete fixed-prefix seed"
                )
            completed_seeds.append(seed)
            continue
        if next_seed is None:
            next_seed = seed
            next_seed_completed_conditions = completed_conditions
        saw_incomplete = True
    return completed_seeds, next_seed, next_seed_completed_conditions


def next_attempt_dir(command_dir: Path) -> Path:
    indices = []
    for path in command_dir.glob("attempt-*"):
        try:
            indices.append(int(path.name.split("-", 1)[1]))
        except (IndexError, ValueError):
            continue
    return command_dir / f"attempt-{max(indices, default=0) + 1:02d}"


def recover_interrupted_stage(
    campaign_dir: Path,
    manifest: dict[str, Any],
    stage_id: str,
) -> int:
    """Recover a stale running marker only after every recorded group is gone."""

    if manifest.get("stage_status", {}).get(stage_id) != "running":
        return 0
    recovered = 0
    stage_dir = campaign_dir / "stages" / stage_id
    for command_dir in sorted(path for path in stage_dir.glob("*") if path.is_dir()):
        records = []
        for metadata_path in sorted(command_dir.glob("attempt-*/metadata.json")):
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
                f"stale stage contains an unaudited failure: {metadata_path}"
            )
    manifest["stage_status"][stage_id] = "failed"
    manifest.setdefault("interrupted_stages", {})[stage_id] = {
        "recorded_at_utc": utc_now(),
        "reason": "recovered stale running state after process-group absence check",
        "recovered_attempts": recovered,
    }
    manifest["updated_at_utc"] = utc_now()
    write_json(campaign_dir / "manifest.json", manifest)
    return recovered


def materialize_command(spec: CommandSpec, attempt_dir: Path) -> list[str]:
    return [part.replace("{attempt_dir}", str(attempt_dir)) for part in spec.command]


def run_process(
    command: list[str],
    environment: dict[str, str],
    log_path: Path,
    *,
    deadline: float | None = None,
    on_start: Any = None,
) -> tuple[int, bool]:
    return run_streaming_process(
        command,
        environment,
        log_path,
        cwd=PROJECT_ROOT,
        deadline=deadline,
        deadline_kill_grace_seconds=DEADLINE_KILL_GRACE_SECONDS,
        on_start=on_start,
    )


def run_command(
    campaign_dir: Path,
    stage_id: str,
    spec: CommandSpec,
    *,
    deadline: float | None = None,
    deadline_epoch: float | None = None,
    measurement_window_id: str | None = None,
    stop_boundary_hours_left: float | None = None,
) -> None:
    command_dir = campaign_dir / "stages" / stage_id / spec.name
    if completed_attempt(command_dir) is not None:
        print(f"Skipping completed command: {stage_id}/{spec.name}")
        return
    if deadline is not None and time.monotonic() >= deadline:
        raise CampaignTimeBoundary(
            f"{stage_id}: registered reserve reached before {spec.name}"
        )
    campaign_manifest = read_json(campaign_dir / "manifest.json")
    assert_sealed_inputs(campaign_manifest, campaign_dir=campaign_dir)
    runtime_identity_before = assert_runtime_identity(
        campaign_manifest,
        campaign_dir,
    )
    sealed_fingerprint = campaign_manifest.get("fingerprint")
    fingerprint_before = protocol_fingerprint()
    if fingerprint_before != sealed_fingerprint:
        raise RuntimeError(
            "source/config fingerprint changed before command start; use a new campaign"
        )
    attempt_dir = next_attempt_dir(command_dir)
    attempt_dir.mkdir(parents=True, exist_ok=False)
    command = materialize_command(spec, attempt_dir)
    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "RESULTS_DIR": str(attempt_dir),
            "COMNETX_CACHE_DIR": str(campaign_dir / "bootstrap-cache"),
        }
    )
    metadata = {
        "schema": "comnetx-ieee-access-repaired-command-v1",
        "stage_id": stage_id,
        "command_name": spec.name,
        "status": "running",
        "started_at_utc": utc_now(),
        "command": command,
        "environment": {
            "RESULTS_DIR": ".",
            "COMNETX_CACHE_DIR": "../../../bootstrap-cache",
            **spec.environment,
        },
        "fingerprint_before": fingerprint_before,
        "runtime_identity_before": runtime_identity_before,
    }
    if deadline_epoch is not None:
        metadata["measurement_window_id"] = measurement_window_id
        metadata["window_deadline_epoch"] = deadline_epoch
    if stop_boundary_hours_left is not None:
        metadata["stop_boundary_hours_left"] = stop_boundary_hours_left
    metadata_path = attempt_dir / "metadata.json"
    write_json(metadata_path, metadata)
    started = time.perf_counter()
    def record_process(process_id: int) -> None:
        metadata["process_id"] = process_id
        metadata["process_group_id"] = process_id
        write_json(metadata_path, metadata)

    try:
        return_code, deadline_reached = run_process(
            command,
            env,
            attempt_dir / "stdout.log",
            deadline=deadline,
            on_start=record_process,
        )
    except BaseException as exc:
        metadata.update(
            {
                "status": "failed",
                "failure_kind": "launcher_interrupted",
                "finished_at_utc": utc_now(),
                "wall_seconds": time.perf_counter() - started,
                "process_error": f"{type(exc).__name__}: {exc}",
            }
        )
        log_path = attempt_dir / "stdout.log"
        if log_path.is_file():
            metadata["stdout_sha256"] = sha256_file(log_path)
        write_json(metadata_path, metadata)
        raise
    stdout_hash = sha256_file(attempt_dir / "stdout.log")
    metadata.update(
        {
            "finished_at_utc": utc_now(),
            "wall_seconds": time.perf_counter() - started,
            "return_code": return_code,
            "deadline_reached": deadline_reached,
            "stdout_sha256": stdout_hash,
        }
    )
    fingerprint_after = protocol_fingerprint()
    metadata["fingerprint_after"] = fingerprint_after
    if fingerprint_after != sealed_fingerprint:
        metadata["status"] = "failed"
        write_json(metadata_path, metadata)
        raise RuntimeError(
            f"{stage_id}/{spec.name}: source/config fingerprint changed during execution"
        )
    try:
        assert_sealed_inputs(campaign_manifest, campaign_dir=campaign_dir)
        runtime_identity_after = assert_runtime_identity(
            campaign_manifest,
            campaign_dir,
        )
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["postcondition_error"] = f"{type(exc).__name__}: {exc}"
        write_json(metadata_path, metadata)
        raise
    metadata["runtime_identity_after"] = runtime_identity_after
    if runtime_identity_after != runtime_identity_before:
        metadata["status"] = "failed"
        write_json(metadata_path, metadata)
        raise RuntimeError(
            f"{stage_id}/{spec.name}: hardware/software identity changed "
            "during execution"
        )
    if deadline_reached:
        metadata["status"] = "failed"
        metadata["failure_kind"] = "campaign_time_boundary"
        write_json(metadata_path, metadata)
        raise CampaignTimeBoundary(
            f"{stage_id}/{spec.name} stopped at the registered campaign-time "
            f"boundary; see {attempt_dir / 'stdout.log'}"
        )
    if return_code != 0:
        metadata["status"] = "failed"
        write_json(metadata_path, metadata)
        raise RuntimeError(
            f"{stage_id}/{spec.name} failed; see {attempt_dir / 'stdout.log'}"
        )
    metadata["status"] = "completed"
    metadata["json_sha256"] = json_file_hashes(attempt_dir)
    metadata["bootstrap_cache_sha256"] = bootstrap_cache_hashes(
        campaign_dir / "bootstrap-cache"
    )
    write_json(metadata_path, metadata)


def run_preflight(
    campaign_dir: Path,
    paths_config: Path,
) -> dict[str, dict[str, str]]:
    preflight_dir = campaign_dir / "preflight"
    for name, command in (
        ("lint", ["make", "lint"]),
        ("unit", ["make", "unit"]),
        (
            "gpu_environment",
            [
                sys.executable,
                str(PACKAGE_DIR / "check_measurement_environment.py"),
                "--output",
                "{attempt_dir}/environment.json",
            ],
        ),
        (
            "input_manifest",
            [
                sys.executable,
                str(PACKAGE_DIR / "input_manifest.py"),
                "--paths-config",
                str(paths_config),
                "--output",
                "{attempt_dir}/input_manifest.json",
            ],
        ),
    ):
        spec = CommandSpec(name=name, command=command)
        run_command(campaign_dir, "preflight", spec)
    input_attempt = completed_attempt(
        campaign_dir / "stages" / "preflight" / "input_manifest"
    )
    if input_attempt is None:
        raise RuntimeError("input-manifest preflight did not complete")
    input_payload = read_json(input_attempt / "input_manifest.json")
    validate_input_manifest(input_payload, verify_content=True)
    sealed_path = campaign_dir / "real_input_manifest.json"
    write_json(sealed_path, input_payload)
    environment_attempt = completed_attempt(
        campaign_dir / "stages" / "preflight" / "gpu_environment"
    )
    if environment_attempt is None:
        raise RuntimeError("environment preflight did not complete")
    environment_payload = read_json(environment_attempt / "environment.json")
    if collect_measurement_environment() != environment_payload:
        raise RuntimeError(
            "live CUDA/package environment changed during campaign preflight"
        )
    sealed_environment_path = campaign_dir / "environment.json"
    write_json(sealed_environment_path, environment_payload)
    write_json(
        preflight_dir / "validation.json",
        {
            "status": "validated",
            "checks": [
                "make lint",
                "make unit",
                "CUDA and production-backend availability",
                "content hashes for every registered real-stream input",
            ],
            "completed_at_utc": utc_now(),
        },
    )
    return {
        "real_input_manifest": {
            "filename": sealed_path.name,
            "sha256": sha256_file(sealed_path),
        },
        "environment": {
            "filename": sealed_environment_path.name,
            "sha256": sha256_file(sealed_environment_path),
        },
    }


def seal_dsbm_inputs(
    campaign_dir: Path,
    manifest: dict[str, Any],
    dsbm_root: Path,
) -> None:
    registered = manifest.get("dsbm_input_manifest")
    if registered is not None:
        assert_sealed_inputs(
            manifest,
            campaign_dir=campaign_dir,
            verify_content=True,
        )
        registered_path = campaign_dir / registered["filename"]
        registered_payload = read_json(registered_path)
        current_payload = build_dsbm_input_manifest(dsbm_root)
        if current_payload != registered_payload:
            raise RuntimeError(
                "DSBM root or sealed input content changed; start a new campaign"
            )
        return
    payload = build_dsbm_input_manifest(dsbm_root)
    path = campaign_dir / "dsbm_input_manifest.json"
    write_json(path, payload)
    manifest["dsbm_input_manifest"] = {
        "filename": path.name,
        "sha256": sha256_file(path),
    }
    manifest["updated_at_utc"] = utc_now()
    write_json(campaign_dir / "manifest.json", manifest)


def print_queue(protocol: dict[str, Any]) -> None:
    for stage in protocol["stages"]:
        requirement = "REQUIRED" if stage["required"] else "optional"
        print(
            f"Priority {stage['priority']} | Stage {stage['number']} | "
            f"{stage['id']} [{requirement}] "
            f"min_remaining={stage['minimum_hours_remaining']}h: {stage['purpose']}"
        )


def dependency_is_resolved(
    dependency: dict[str, Any],
    manifest: dict[str, Any],
) -> bool:
    status = manifest["stage_status"].get(dependency["id"])
    if dependency["required"]:
        return status == "validated"
    return status in {
        "validated",
        "skipped_for_budget",
        "skipped_by_stage2_no_go",
    }


def main() -> None:
    install_termination_signal_handlers()
    protocol = load_protocol()
    stages = stage_map(protocol)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths-config", type=Path)
    parser.add_argument("--campaign-id")
    parser.add_argument("--output-root", type=Path, default=RESULTS_ROOT)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--stage", choices=tuple(stages))
    action.add_argument("--recover-interrupted-stage", choices=tuple(stages))
    parser.add_argument("--hours-left", type=float)
    parser.add_argument(
        "--deadline-epoch",
        type=float,
        help=(
            "Absolute shared campaign deadline. When present it supersedes the "
            "rounded --hours-left snapshot for all gates and hard stops."
        ),
    )
    parser.add_argument(
        "--measurement-window-id",
        help=(
            "Identifier of the append-only measurement window that owns "
            "--deadline-epoch. Required whenever a deadline is supplied."
        ),
    )
    parser.add_argument(
        "--stop-with-hours-left",
        type=float,
        default=0.0,
        help=(
            "Terminate the current measurement command at this campaign-time "
            "reserve boundary."
        ),
    )
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--dsbm-root", type=Path)
    parser.add_argument(
        "--dsbm-seed",
        type=int,
        choices=(42, 43, 44, 45, 46),
        help=(
            "Run one fixed 3x2 paired DSBM seed block. Seeds 42..44 form the "
            "primary design; seeds 45..46 are a budget-contingent precision extension."
        ),
    )
    action.add_argument("--preflight", action="store_true")
    parser.add_argument("--resume", action="store_true")
    action.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ack-production-api-settled", action="store_true")
    action.add_argument(
        "--skip-for-budget",
        choices=tuple(stage["id"] for stage in protocol["stages"] if not stage["required"]),
    )
    action.add_argument(
        "--skip-for-stage2-no-go",
        choices=(
            "stage5_topology_controls",
            "stage7_dfleiden_interface",
            "stage7_s2cag_interface",
        ),
    )
    args = parser.parse_args()

    if args.list:
        print_queue(protocol)
        return
    if not args.paths_config or not args.campaign_id:
        parser.error("--paths-config and --campaign-id are required")
    if not CAMPAIGN_RE.fullmatch(args.campaign_id):
        parser.error("invalid campaign id")
    paths_config = resolve_project_path(args.paths_config)
    if not paths_config.is_file():
        parser.error(f"paths config does not exist: {paths_config}")
    output_root = resolve_project_path(args.output_root)
    campaign_dir = output_root / args.campaign_id
    manifest_path = campaign_dir / "manifest.json"

    if manifest_path.exists():
        if not args.resume:
            parser.error("campaign exists; pass --resume")
        manifest = read_json(manifest_path)
        assert_campaign_compatible(manifest, protocol, paths_config, campaign_dir)
    else:
        if args.resume:
            parser.error("cannot resume a missing campaign")
        if not args.ack_production_api_settled:
            parser.error(
                "new campaigns require --ack-production-api-settled after the repair API is final"
            )
        if campaign_dir.exists():
            parser.error("campaign directory exists without a manifest; preserve it and use a new id")
        manifest = make_manifest(protocol, args.campaign_id, paths_config)
        hardware = collect_hardware()
        campaign_dir.mkdir(parents=True, exist_ok=False)
        hardware_path = campaign_dir / "hardware.json"
        write_json(hardware_path, hardware)
        manifest["hardware"] = {
            "filename": hardware_path.name,
            "sha256": sha256_file(hardware_path),
        }
        write_json(manifest_path, manifest)

    if args.preflight:
        preflight_registration = run_preflight(campaign_dir, paths_config)
        manifest.update(preflight_registration)
        manifest["status"] = "preflight_validated"
        manifest["updated_at_utc"] = utc_now()
        write_json(manifest_path, manifest)
        print("Preflight validated; no graph measurements were run.")
        return

    if (
        args.skip_for_budget
        or args.skip_for_stage2_no_go
        or args.recover_interrupted_stage
    ) and not (
        campaign_dir / "preflight" / "validation.json"
    ).is_file():
        parser.error("run --preflight successfully before recording a stage skip")

    target_stage = (
        args.stage
        or args.skip_for_budget
        or args.skip_for_stage2_no_go
        or args.recover_interrupted_stage
    )
    recovered = 0
    if (
        target_stage is not None
        and manifest.get("stage_status", {}).get(target_stage) == "running"
    ):
        recovered = recover_interrupted_stage(campaign_dir, manifest, target_stage)
        print(f"Recovered stale running stage {target_stage}: {recovered} attempt(s)")

    if args.deadline_epoch is not None:
        if not math.isfinite(args.deadline_epoch):
            parser.error("--deadline-epoch must be finite")
        if not args.measurement_window_id:
            parser.error(
                "--measurement-window-id is required with --deadline-epoch"
            )
        args.hours_left = max(0.0, (args.deadline_epoch - time.time()) / 3600.0)
    elif args.measurement_window_id:
        parser.error("--measurement-window-id requires --deadline-epoch")

    if args.recover_interrupted_stage:
        if manifest.get("stage_status", {}).get(args.recover_interrupted_stage) == "running":
            parser.error("running stage could not be recovered")
        print(
            f"Interrupted-stage recovery complete: {args.recover_interrupted_stage} "
            f"({recovered} attempt(s))"
        )
        return

    if args.skip_for_budget:
        stage = stages[args.skip_for_budget]
        if stage["required"]:
            parser.error("required core stages cannot be marked skipped for budget")
        if args.hours_left is None or not math.isfinite(args.hours_left):
            parser.error("--hours-left is required when recording a budget skip")
        if args.hours_left >= float(stage["minimum_hours_remaining"]):
            parser.error(
                "budget skip is allowed only below the registered start-time gate"
            )
        if manifest["stage_status"].get(stage["id"]) not in {"pending", "failed"}:
            parser.error(f"stage cannot be budget-skipped: {stage['id']}")
        for dependency in stage["dependencies"]:
            if not dependency_is_resolved(stages[dependency], manifest):
                parser.error(f"dependency is not validated: {dependency}")
        manifest["stage_status"][stage["id"]] = "skipped_for_budget"
        manifest.setdefault("budget_skips", {})[stage["id"]] = {
            "recorded_at_utc": utc_now(),
            "hours_left": args.hours_left,
            "purpose_preserved": stage["purpose"],
        }
        write_json(manifest_path, manifest)
        print(f"Recorded budget skip without deleting the planned stage: {stage['id']}")
        return

    if args.skip_for_stage2_no_go:
        stage = stages[args.skip_for_stage2_no_go]
        if manifest["stage_status"].get(stage["id"]) != "pending":
            parser.error(f"stage is not pending: {stage['id']}")
        for dependency in stage["dependencies"]:
            if not dependency_is_resolved(stages[dependency], manifest):
                parser.error(f"dependency is not resolved: {dependency}")
        stage2_validation = campaign_dir / "stages" / "stage2_core_short" / "validation.json"
        if not stage2_validation.is_file() or read_json(stage2_validation).get(
            "optional_breadth_go"
        ) is not False:
            parser.error("Stage 2 did not trigger the registered breadth no-go rule")
        manifest["stage_status"][stage["id"]] = "skipped_by_stage2_no_go"
        manifest.setdefault("scientific_no_go_skips", {})[stage["id"]] = {
            "recorded_at_utc": utc_now(),
            "rule": "neither core graph met speedup>1 with Delta Q >= -0.05",
            "purpose_preserved": stage["purpose"],
        }
        write_json(manifest_path, manifest)
        print(f"Recorded Stage-2 scientific no-go without deleting: {stage['id']}")
        return

    if not args.stage:
        parser.error(
            "select --preflight, --stage, a registered skip action, or --list"
        )
    if not (campaign_dir / "preflight" / "validation.json").is_file():
        parser.error("run --preflight successfully before any measurement stage")
    stage = stages[args.stage]
    for dependency in stage["dependencies"]:
        if not dependency_is_resolved(stages[dependency], manifest):
            parser.error(f"dependency is not resolved: {dependency}")
    if stage["id"] in {
        "stage5_topology_controls",
        "stage7_dfleiden_interface",
        "stage7_s2cag_interface",
    }:
        stage2_decision = read_json(
            campaign_dir / "stages" / "stage2_core_short" / "validation.json"
        ).get("optional_breadth_go")
        if stage2_decision is not True:
            parser.error(
                "Stage 2 triggered the breadth no-go rule; record "
                "--skip-for-stage2-no-go instead"
            )
    if args.hours_left is None or not math.isfinite(args.hours_left):
        parser.error("--hours-left is required to enforce the 72-hour stop rules")
    if not math.isfinite(args.stop_with_hours_left) or args.stop_with_hours_left < 0:
        parser.error("--stop-with-hours-left must be a finite non-negative number")
    stop_boundary = float(args.stop_with_hours_left)
    dsbm_root = resolve_project_path(args.dsbm_root) if args.dsbm_root else None
    if dsbm_root is not None and not dsbm_root.exists():
        parser.error(f"DSBM root does not exist: {dsbm_root}")
    if stage["id"] == "stage6_dsbm" and dsbm_root is None:
        parser.error("--dsbm-root is required for stage6_dsbm")
    if stage["id"] == "stage6_dsbm" and args.dsbm_seed is None:
        parser.error(
            "--dsbm-seed is required so Stage 6 advances one fixed-prefix seed at a time"
        )
    if args.dsbm_seed is not None and stage["id"] != "stage6_dsbm":
        parser.error("--dsbm-seed is valid only for stage6_dsbm")
    commands = build_stage_commands(
        stage["id"],
        campaign_dir=campaign_dir,
        paths_config=paths_config,
        dsbm_root=dsbm_root,
        repetitions=args.repetitions,
        dsbm_seed=args.dsbm_seed,
    )
    all_requested_commands_completed = all(
        completed_attempt(
            campaign_dir / "stages" / stage["id"] / spec.name
        )
        is not None
        for spec in commands
    )
    stage_minimum_hours = float(stage["minimum_hours_remaining"])
    if stage["id"] == "stage6_dsbm":
        completed_seeds, next_seed, completed_conditions = dsbm_seed_progress(
            campaign_dir
        )
        stage_minimum_hours = float(
            stage[
                "partial_seed_minimum_hours_remaining"
                if completed_conditions
                else "new_seed_minimum_hours_remaining"
            ]
        )
        validation_path = campaign_dir / "stages" / "stage6_dsbm" / "validation.json"
        validated_seeds = (
            read_json(validation_path).get("completed_seeds", [])
            if validation_path.is_file()
            else []
        )
        if validated_seeds != completed_seeds[: len(validated_seeds)]:
            parser.error("saved DSBM validation is not the completed fixed seed prefix")
        if len(validated_seeds) < len(completed_seeds):
            finalize_seed = completed_seeds[-1]
            if args.dsbm_seed != finalize_seed or not all_requested_commands_completed:
                parser.error(
                    f"finalize already completed DSBM seed {finalize_seed} before "
                    "starting another seed"
                )
        elif args.dsbm_seed != next_seed and not all_requested_commands_completed:
            parser.error(
                f"the next registered DSBM seed is {next_seed}, not {args.dsbm_seed}"
            )
    if (
        args.hours_left < stage_minimum_hours
        and not all_requested_commands_completed
    ):
        parser.error(
            f"{stage['id']} requires at least {stage_minimum_hours:g}h remaining; "
            "do not consume the reserve"
        )
    if stop_boundary >= args.hours_left and not all_requested_commands_completed:
        parser.error(
            "the command stop boundary must be below the current hours remaining"
        )
    if args.dry_run:
        print(f"Stage: {stage['id']}")
        print(f"Stop boundary: {stop_boundary:g} hours left")
        for spec in commands:
            print(spec.name + ":")
            print("  " + " ".join(materialize_command(spec, Path("ATTEMPT_DIR"))))
        return

    if stage["id"] == "stage6_dsbm":
        assert dsbm_root is not None
        seal_dsbm_inputs(campaign_dir, manifest, dsbm_root)

    if args.deadline_epoch is not None:
        args.hours_left = max(0.0, (args.deadline_epoch - time.time()) / 3600.0)
        if (
            args.hours_left < stage_minimum_hours
            and not all_requested_commands_completed
        ):
            parser.error(
                f"{stage['id']} fell below its "
                f"{stage_minimum_hours:g}h start gate during preconditions; "
                "re-run the launcher to record the registered budget outcome"
            )
        if stop_boundary >= args.hours_left and not all_requested_commands_completed:
            parser.error(
                "the command stop boundary was reached during preconditions; "
                "re-run the launcher"
            )

    manifest["stage_status"][stage["id"]] = "running"
    manifest.setdefault("time_boundary_stops", {}).pop(stage["id"], None)
    manifest.setdefault("interrupted_stages", {}).pop(stage["id"], None)
    manifest["updated_at_utc"] = utc_now()
    write_json(manifest_path, manifest)
    if args.deadline_epoch is None:
        command_deadline = time.monotonic() + (
            args.hours_left - stop_boundary
        ) * 3600.0
    else:
        command_deadline = time.monotonic() + (
            args.deadline_epoch - time.time() - stop_boundary * 3600.0
        )
    try:
        for spec in commands:
            run_command(
                campaign_dir,
                stage["id"],
                spec,
                deadline=command_deadline,
                deadline_epoch=args.deadline_epoch,
                measurement_window_id=args.measurement_window_id,
                stop_boundary_hours_left=stop_boundary,
            )
        validation = validate_stage(campaign_dir, stage["id"])
    except CampaignTimeBoundary as exc:
        manifest["stage_status"][stage["id"]] = "failed"
        manifest.setdefault("time_boundary_stops", {})[stage["id"]] = {
            "recorded_at_utc": utc_now(),
            "hours_left_at_launch": args.hours_left,
            "stop_boundary_hours_left": stop_boundary,
            "measurement_window_id": args.measurement_window_id,
            "window_deadline_epoch": args.deadline_epoch,
            "reason": str(exc),
        }
        manifest["updated_at_utc"] = utc_now()
        write_json(manifest_path, manifest)
        raise
    except (KeyboardInterrupt, LauncherSignalInterrupt) as exc:
        manifest["stage_status"][stage["id"]] = "failed"
        manifest.setdefault("interrupted_stages", {})[stage["id"]] = {
            "recorded_at_utc": utc_now(),
            "reason": f"{type(exc).__name__}: {exc}",
        }
        manifest["updated_at_utc"] = utc_now()
        write_json(manifest_path, manifest)
        raise
    except Exception:
        manifest["stage_status"][stage["id"]] = "failed"
        manifest["updated_at_utc"] = utc_now()
        write_json(manifest_path, manifest)
        raise
    write_json(campaign_dir / "stages" / stage["id"] / "validation.json", validation)
    manifest["stage_status"][stage["id"]] = (
        "partial"
        if stage["id"] == "stage6_dsbm"
        and validation.get("status") in {"valid_partial", "primary_validated"}
        else "validated"
    )
    manifest["updated_at_utc"] = utc_now()
    required = [item["id"] for item in protocol["stages"] if item["required"]]
    manifest["status"] = (
        "core_validated"
        if all(manifest["stage_status"][item] == "validated" for item in required)
        else "running"
    )
    write_json(manifest_path, manifest)
    print(f"Validated stage: {stage['id']}")


if __name__ == "__main__":
    main()
