#!/usr/bin/env python3
"""Validate that repaired-ComNetX and LD-Leiden campaigns are comparable."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_ROOT = PROJECT_ROOT / "results" / "ieee-access-2026-1" / "raw"
REPAIRED_ROOT = RAW_ROOT / "repaired-comnetx"
LD_ROOT = RAW_ROOT / "ldleiden"


def _campaign(value: Path, root: Path) -> Path:
    value = value.expanduser()
    if value.is_absolute():
        return value.resolve()
    if len(value.parts) == 1:
        return (root / value).resolve()
    return (PROJECT_ROOT / value).resolve()


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _registered(campaign: Path, manifest: dict[str, Any], key: str) -> dict[str, Any]:
    registration = manifest.get(key)
    if not isinstance(registration, dict):
        raise ValueError(f"missing {key} registration in {campaign}")
    path = campaign / str(registration.get("filename", ""))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != registration.get("sha256"):
        raise ValueError(f"changed {key} artifact: {path}")
    return _read(path)


def _file_identity(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record.get("size"),
        record.get("mtime_ns"),
        record.get("sha256"),
    )


def _validate_input_manifest_structure(
    payload: dict[str, Any], *, expected_schema: str
) -> list[dict[str, Any]]:
    if payload.get("schema") != expected_schema:
        raise ValueError(f"unexpected input-manifest schema: {payload.get('schema')}")
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("input manifest has no files")
    paths = [record.get("path") for record in files if isinstance(record, dict)]
    if len(paths) != len(files) or any(not isinstance(path, str) for path in paths):
        raise ValueError("input manifest contains a malformed path")
    if len(paths) != len(set(paths)):
        raise ValueError("input manifest contains duplicate paths")
    return files


def _canonical(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.size == 0 or labels.dtype.kind not in {"i", "u"}:
        raise ValueError("bootstrap partition must be a non-empty integral vector")
    result = np.empty(labels.shape, dtype=np.int64)
    groups: dict[int, list[int]] = {}
    for vertex, label in enumerate(labels.astype(np.int64, copy=False)):
        groups.setdefault(int(label), []).append(vertex)
    for vertices in groups.values():
        result[vertices] = min(vertices)
    return result


def _bootstrap(path: Path, *, allow_hierarchy: bool) -> tuple[np.ndarray, float]:
    with np.load(path, allow_pickle=False) as payload:
        fields = set(payload.files)
        if "partition" not in fields or "mod" not in fields:
            raise ValueError(f"malformed bootstrap payload: {path}")
        partition = np.asarray(payload["partition"])
        modularity = float(np.asarray(payload["mod"]).item())
        schema = str(np.asarray(payload["schema"]).item()) if "schema" in payload else None
    if partition.ndim == 2:
        if (
            not allow_hierarchy
            or fields != {"partition", "mod", "schema"}
            or partition.shape[0] != 3
            or schema != "parent_quotient_v1"
        ):
            raise ValueError(f"malformed hierarchy bootstrap: {path}")
        partition = partition[0]
    elif partition.ndim == 1:
        if fields != {"partition", "mod"}:
            raise ValueError(f"malformed flat bootstrap: {path}")
    else:
        raise ValueError(f"unexpected bootstrap rank: {path}")
    canonical = _canonical(partition)
    if not np.array_equal(partition.astype(np.int64, copy=False), canonical):
        raise ValueError(f"bootstrap is not canonical on disk: {path}")
    if not math.isfinite(modularity) or not -1 <= modularity <= 1:
        raise ValueError(f"non-finite bootstrap modularity: {path}")
    return canonical, modularity


def _bootstrap_path(
    campaign: Path, dataset: str, initial_batch: int, *, allow_hierarchy: bool
) -> Path:
    cache = campaign / "bootstrap-cache"
    prefixes = (dataset, f"{dataset}-sym")
    candidates = [
        cache / f"{prefix}_b:{initial_batch}_by_leidenalg.npz"
        for prefix in prefixes
    ]
    if allow_hierarchy:
        candidates.extend(
            cache
            / (
                f"{prefix}_b:{initial_batch}_by_leidenalg_"
                "d:3_parent_quotient_v1.npz"
            )
            for prefix in prefixes
        )
    existing = [path for path in candidates if path.is_file()]
    observed_prefixes = {
        path.name.split(f"_b:{initial_batch}_by_leidenalg", 1)[0]
        for path in existing
    }
    if len(observed_prefixes) > 1:
        raise ValueError(
            f"ambiguous bootstrap graph representations: {sorted(observed_prefixes)}"
        )
    flat = [path for path in existing if "_d:3_" not in path.name]
    selected = flat or existing
    if len(selected) != 1:
        raise ValueError(
            f"expected one bootstrap for {dataset}/{initial_batch} in {campaign}"
        )
    return selected[0]


def assert_clock_creation_safe(repaired: Path, ld: Path) -> None:
    """Refuse to replace a lost clock after either campaign started measuring."""

    specifications = (
        (repaired, "stage_status", ("stages", "preflight")),
        (ld, "phase_status", None),
    )
    for campaign, status_key, allowed_attempt_parent in specifications:
        manifest_path = campaign / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = _read(manifest_path)
        statuses = manifest.get(status_key)
        if not isinstance(statuses, dict) or any(
            status != "pending" for status in statuses.values()
        ):
            raise ValueError(
                f"cannot create a new shared clock after campaign progress: {campaign}"
            )
        for metadata_path in campaign.rglob("attempt-*/metadata.json"):
            relative_parts = metadata_path.relative_to(campaign).parts
            if (
                allowed_attempt_parent is not None
                and len(relative_parts) >= 2
                and relative_parts[:2] == allowed_attempt_parent
            ):
                continue
            raise ValueError(
                "cannot create a new shared clock after a measurement attempt: "
                f"{metadata_path}"
            )


def _numeric(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def _validate_launch_budget(
    budget: dict[str, Any],
    *,
    repaired: Path,
    ld: Path,
    git_sha: str,
    paths_config_sha256: str,
) -> dict[str, dict[str, Any]]:
    expected = {
        "schema": "comnetx-ieee-access-launch-budget-v2",
        "git_sha": git_sha,
        "repaired_campaign_id": repaired.name,
        "ld_campaign_id": ld.name,
        "paths_config_sha256": paths_config_sha256,
        "initial_window_hours": 24,
        "max_total_hours": 72,
    }
    for key, value in expected.items():
        if budget.get(key) != value:
            raise ValueError(f"cross-pack launch budget differs for {key}")
    windows = budget.get("windows")
    if not isinstance(windows, list) or not windows:
        raise ValueError("cross-pack launch budget has no measurement windows")
    by_id: dict[str, dict[str, Any]] = {}
    total = 0.0
    previous_deadline: int | None = None
    for index, window in enumerate(windows, start=1):
        if not isinstance(window, dict):
            raise ValueError("cross-pack launch budget contains a malformed window")
        window_id = window.get("id")
        if window_id != f"window-{index:03d}" or window_id in by_id:
            raise ValueError("cross-pack launch budget has an invalid window id")
        granted = _numeric(window.get("granted_hours"), f"{window_id} grant")
        started = window.get("started_at_epoch")
        deadline = window.get("deadline_epoch")
        if (
            granted <= 0
            or not isinstance(started, int)
            or not isinstance(deadline, int)
            or deadline != started + int(round(granted * 3600))
        ):
            raise ValueError(f"cross-pack launch budget window is malformed: {window_id}")
        if previous_deadline is not None and started < previous_deadline:
            raise ValueError("cross-pack launch budget windows overlap")
        if index == 1 and granted != 24:
            raise ValueError("the first measurement window must be exactly 24 hours")
        total += granted
        previous_deadline = deadline
        by_id[window_id] = window
    if total > 72 + 1e-12:
        raise ValueError("cross-pack launch budget exceeds its 72-hour cap")
    return by_id


def _validate_attempt_windows(
    campaign: Path,
    windows: dict[str, dict[str, Any]],
) -> int:
    checked = 0
    for metadata_path in sorted(campaign.rglob("attempt-*/metadata.json")):
        relative = metadata_path.relative_to(campaign)
        if len(relative.parts) >= 2 and relative.parts[:2] == ("stages", "preflight"):
            continue
        metadata = _read(metadata_path)
        window_id = metadata.get("measurement_window_id")
        window = windows.get(window_id) if isinstance(window_id, str) else None
        if window is None:
            raise ValueError(f"attempt lacks a registered measurement window: {metadata_path}")
        if metadata.get("window_deadline_epoch") != window["deadline_epoch"]:
            raise ValueError(f"attempt has a mismatched window deadline: {metadata_path}")
        started_at = metadata.get("started_at_utc")
        if not isinstance(started_at, str):
            raise ValueError(f"attempt lacks a UTC start timestamp: {metadata_path}")
        try:
            started_epoch = datetime.fromisoformat(
                started_at.replace("Z", "+00:00")
            ).timestamp()
        except ValueError as exc:
            raise ValueError(f"attempt has an invalid UTC start: {metadata_path}") from exc
        if not window["started_at_epoch"] - 2 <= started_epoch <= window["deadline_epoch"]:
            raise ValueError(f"attempt started outside its registered window: {metadata_path}")
        if metadata.get("status") == "completed":
            finished_at = metadata.get("finished_at_utc")
            if not isinstance(finished_at, str):
                raise ValueError(f"completed attempt lacks a UTC finish: {metadata_path}")
            try:
                finished_epoch = datetime.fromisoformat(
                    finished_at.replace("Z", "+00:00")
                ).timestamp()
            except ValueError as exc:
                raise ValueError(
                    f"completed attempt has an invalid UTC finish: {metadata_path}"
                ) from exc
            effective_deadline = float(window["deadline_epoch"])
            stop_reserve = metadata.get("stop_boundary_hours_left")
            if stop_reserve is not None:
                stop_reserve = _numeric(
                    stop_reserve, f"{metadata_path} stop-boundary reserve"
                )
                if stop_reserve < 0:
                    raise ValueError(
                        f"attempt has a negative stop-boundary reserve: {metadata_path}"
                    )
                effective_deadline -= stop_reserve * 3600.0
            if not started_epoch - 2 <= finished_epoch <= effective_deadline + 2:
                raise ValueError(
                    f"completed attempt finished outside its registered window: {metadata_path}"
                )
        checked += 1
    return checked


def _assert_ld_resumable_failure(
    campaign: Path,
    manifest: dict[str, Any],
) -> None:
    """Admit only a currently audited boundary/interruption failure."""

    phase_status = manifest.get("phase_status")
    if not isinstance(phase_status, dict):
        raise ValueError("failed LD campaign lacks phase status")
    allowed_statuses = {"pending", "completed", "failed"}
    if any(status not in allowed_statuses for status in phase_status.values()):
        raise ValueError("failed LD campaign has an unresolved phase status")
    failed_phases = {
        phase_id for phase_id, status in phase_status.items() if status == "failed"
    }
    current_failures = manifest.get("phase_failures")
    if (
        not failed_phases
        or not isinstance(current_failures, dict)
        or set(current_failures) != failed_phases
    ):
        raise ValueError("failed LD campaign lacks an exact current-failure audit")

    allowed_failure_kinds = {
        "campaign_time_boundary",
        "launcher_interrupted",
    }
    for phase_id in sorted(failed_phases):
        failure = current_failures.get(phase_id)
        if not isinstance(failure, dict):
            raise ValueError(f"failed LD phase lacks an audit: {phase_id}")
        failure_kind = failure.get("failure_kind")
        if failure_kind not in allowed_failure_kinds:
            raise ValueError(
                f"LD phase failed outside a resumable boundary: {phase_id}"
            )

        if failure_kind == "campaign_time_boundary":
            registrations = manifest.get("time_boundary_stops")
            registration = (
                registrations.get(phase_id)
                if isinstance(registrations, dict)
                else None
            )
            if not isinstance(registration, dict):
                raise ValueError(f"LD phase lacks its time-boundary audit: {phase_id}")
            window_id = registration.get("measurement_window_id")
            deadline = registration.get("deadline_epoch")
            if (
                not isinstance(window_id, str)
                or not window_id
                or isinstance(deadline, bool)
                or not isinstance(deadline, (int, float))
                or not math.isfinite(float(deadline))
            ):
                raise ValueError(f"LD phase has a malformed time-boundary audit: {phase_id}")
        else:
            registrations = manifest.get("interrupted_phases")
            registration = (
                registrations.get(phase_id)
                if isinstance(registrations, dict)
                else None
            )
            if not isinstance(registration, dict) or not isinstance(
                registration.get("reason"), str
            ):
                raise ValueError(f"LD phase lacks its interruption audit: {phase_id}")

        for metadata_path in sorted(
            (campaign / phase_id).glob("repeat-*/attempt-*/metadata.json")
        ):
            metadata = _read(metadata_path)
            status = metadata.get("status")
            if status == "completed":
                continue
            if (
                status != "failed"
                or metadata.get("failure_kind") not in allowed_failure_kinds
            ):
                raise ValueError(
                    "LD resumable phase contains an unaudited attempt failure: "
                    f"{metadata_path}"
                )


def _assert_repaired_resumable_failures(
    campaign: Path,
    manifest: dict[str, Any],
) -> None:
    """Reject failed repaired stages without an exact resumable audit."""

    stage_status = manifest.get("stage_status")
    if not isinstance(stage_status, dict):
        raise ValueError("repaired campaign lacks stage status")
    failed_stages = {
        stage_id for stage_id, status in stage_status.items() if status == "failed"
    }
    if not failed_stages:
        return

    boundary_stops = manifest.get("time_boundary_stops", {})
    interruptions = manifest.get("interrupted_stages", {})
    if not isinstance(boundary_stops, dict) or not isinstance(interruptions, dict):
        raise ValueError("failed repaired campaign has malformed failure audits")

    allowed_failure_kinds = {
        "campaign_time_boundary",
        "launcher_interrupted",
    }
    for stage_id in sorted(failed_stages):
        has_boundary = stage_id in boundary_stops
        has_interruption = stage_id in interruptions
        if has_boundary == has_interruption:
            raise ValueError(
                "failed repaired stage lacks an exact resumable audit: "
                f"{stage_id}"
            )

        if has_boundary:
            boundary = boundary_stops[stage_id]
            if not isinstance(boundary, dict):
                raise ValueError(
                    f"repaired stage has a malformed time-boundary audit: {stage_id}"
                )
            window_id = boundary.get("measurement_window_id")
            deadline = boundary.get("window_deadline_epoch")
            reserve = boundary.get("stop_boundary_hours_left")
            if (
                not isinstance(window_id, str)
                or not window_id
                or isinstance(deadline, bool)
                or not isinstance(deadline, (int, float))
                or not math.isfinite(float(deadline))
                or isinstance(reserve, bool)
                or not isinstance(reserve, (int, float))
                or not math.isfinite(float(reserve))
                or float(reserve) < 0
            ):
                raise ValueError(
                    f"repaired stage has a malformed time-boundary audit: {stage_id}"
                )
        else:
            interruption = interruptions[stage_id]
            if not isinstance(interruption, dict) or not isinstance(
                interruption.get("reason"), str
            ):
                raise ValueError(
                    f"repaired stage has a malformed interruption audit: {stage_id}"
                )

        for metadata_path in sorted(
            (campaign / "stages" / stage_id).glob("*/attempt-*/metadata.json")
        ):
            metadata = _read(metadata_path)
            status = metadata.get("status")
            if status == "completed":
                continue
            if (
                status != "failed"
                or metadata.get("failure_kind") not in allowed_failure_kinds
            ):
                raise ValueError(
                    "repaired resumable stage contains an unaudited attempt failure: "
                    f"{metadata_path}"
                )


def validate_pair(
    repaired: Path, ld: Path, *, preflight_only: bool
) -> dict[str, Any]:
    repaired_manifest = _read(repaired / "manifest.json")
    ld_manifest = _read(ld / "manifest.json")
    expected_manifest = (
        (
            repaired_manifest,
            repaired,
            "comnetx-ieee-access-repaired-campaign-v1",
            "ieee-access-repaired-comnetx-72h-v1",
        ),
        (
            ld_manifest,
            ld,
            "comnetx-ieee-access-ldleiden-campaign-v1",
            "ieee-access-ldleiden-72h-v1",
        ),
    )
    for manifest, campaign, schema, protocol_id in expected_manifest:
        if (
            manifest.get("schema") != schema
            or manifest.get("protocol_id") != protocol_id
            or manifest.get("campaign_id") != campaign.name
        ):
            raise ValueError(f"campaign manifest identity differs: {campaign}")
    if preflight_only:
        if repaired_manifest.get("status") not in {
            "preflight_validated",
            "running",
            "core_validated",
        }:
            raise ValueError("campaign preflight status is invalid")
        _assert_repaired_resumable_failures(repaired, repaired_manifest)
        ld_status = ld_manifest.get("status")
        if ld_status == "failed":
            _assert_ld_resumable_failure(ld, ld_manifest)
        elif ld_status not in {
            "preflight_complete",
            "partial",
            "completed",
        }:
            raise ValueError("campaign preflight status is invalid")
    elif (
        repaired_manifest.get("status") != "core_validated"
        or ld_manifest.get("status") != "completed"
    ):
        raise ValueError("campaign pair is not complete")
    repaired_git = repaired_manifest.get("git", {})
    ld_git = ld_manifest.get("git", {})
    if repaired_git.get("dirty") is not False or ld_git.get("dirty") is not False:
        raise ValueError("both campaigns must originate from clean checkouts")
    if repaired_git.get("commit") != ld_git.get("commit"):
        raise ValueError("campaigns use different git commits")
    if repaired_manifest.get("paths_config", {}).get("sha256") != ld_manifest.get(
        "paths_config", {}
    ).get("sha256"):
        raise ValueError("campaigns use different paths maps")

    repaired_hardware = _registered(repaired, repaired_manifest, "hardware")
    ld_hardware = _registered(ld, ld_manifest, "hardware")
    if repaired_hardware != ld_hardware:
        raise ValueError("campaigns were sealed on different hardware")

    repaired_inputs = _registered(
        repaired, repaired_manifest, "real_input_manifest"
    )
    ld_inputs = _registered(ld, ld_manifest, "real_input_manifest")
    repaired_files = _validate_input_manifest_structure(
        repaired_inputs,
        expected_schema="comnetx-ieee-access-real-inputs-v1",
    )
    ld_files = _validate_input_manifest_structure(
        ld_inputs,
        expected_schema="comnetx-ieee-access-ldleiden-real-inputs-v1",
    )
    repaired_by_path = {
        str(record["path"]): _file_identity(record)
        for record in repaired_files
    }
    for record in ld_files:
        path = str(record["path"])
        if repaired_by_path.get(path) != _file_identity(record):
            raise ValueError(f"cross-pack input identity differs: {path}")

    bootstrap_report: dict[str, Any] = {}
    attempt_window_report: dict[str, int] = {}
    if not preflight_only:
        budget = _read(repaired / "launch_budget.json")
        windows = _validate_launch_budget(
            budget,
            repaired=repaired,
            ld=ld,
            git_sha=str(repaired_git.get("commit")),
            paths_config_sha256=str(
                repaired_manifest.get("paths_config", {}).get("sha256")
            ),
        )
        attempt_window_report = {
            "repaired": _validate_attempt_windows(repaired, windows),
            "ldleiden": _validate_attempt_windows(ld, windows),
        }
        for dataset in ("dyn_pubmed", "arxivmath"):
            for initial_batch in (999, 9):
                repaired_path = _bootstrap_path(
                    repaired, dataset, initial_batch, allow_hierarchy=True
                )
                ld_path = _bootstrap_path(
                    ld, dataset, initial_batch, allow_hierarchy=False
                )
                repaired_partition, repaired_modularity = _bootstrap(
                    repaired_path, allow_hierarchy=True
                )
                ld_partition, ld_modularity = _bootstrap(
                    ld_path, allow_hierarchy=False
                )
                if not np.array_equal(repaired_partition, ld_partition):
                    raise ValueError(
                        f"cross-pack bootstrap partition differs: "
                        f"{dataset}/{initial_batch}"
                    )
                if not math.isclose(
                    repaired_modularity,
                    ld_modularity,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        f"cross-pack bootstrap modularity differs: "
                        f"{dataset}/{initial_batch}"
                    )
                bootstrap_report[f"{dataset}/{initial_batch}"] = {
                    "repaired_file": repaired_path.name,
                    "ld_file": ld_path.name,
                    "vertices": int(repaired_partition.size),
                    "modularity": repaired_modularity,
                }

    return {
        "schema": "comnetx-ieee-access-campaign-pair-validation-v1",
        "status": "valid_preflight_pair" if preflight_only else "valid_pair",
        "git_commit": repaired_git.get("commit"),
        "repaired_campaign": repaired.name,
        "ld_campaign": ld.name,
        "shared_input_files": len(ld_files),
        "measurement_window_attempts": attempt_window_report,
        "bootstrap": bootstrap_report,
    }


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repaired-campaign", type=Path, required=True)
    parser.add_argument("--ld-campaign", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--assert-clock-creation-safe", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    repaired = _campaign(args.repaired_campaign, REPAIRED_ROOT)
    ld = _campaign(args.ld_campaign, LD_ROOT)
    if args.assert_clock_creation_safe:
        assert_clock_creation_safe(repaired, ld)
        print("Shared clock may be created: neither campaign has measurement progress")
        return
    report = validate_pair(repaired, ld, preflight_only=args.preflight_only)
    report_path = args.report or repaired / "cross_pack_validation.json"
    _write(report_path.expanduser().resolve(), report)
    print(f"Cross-pack validation status: {report['status']}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
