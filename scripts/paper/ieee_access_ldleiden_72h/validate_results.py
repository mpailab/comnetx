#!/usr/bin/env python3
"""Validate and summarize one IEEE Access LD-Leiden measurement campaign."""

from __future__ import annotations

import argparse
from datetime import datetime
import math
from pathlib import Path
from typing import Any

try:
    from .protocol import (
        RESULTS_ROOT,
        ValidationError,
        aggregate_run_summaries,
        bootstrap_cache_reference,
        launcher_result_file,
        load_protocol,
        protocol_fingerprint,
        read_json,
        sha256_file,
        sha256_json,
        validate_launcher_file,
        write_json,
    )
except ImportError:  # Direct ``python path/to/validate_results.py`` execution.
    from protocol import (
        RESULTS_ROOT,
        ValidationError,
        aggregate_run_summaries,
        bootstrap_cache_reference,
        launcher_result_file,
        load_protocol,
        protocol_fingerprint,
        read_json,
        sha256_file,
        sha256_json,
        validate_launcher_file,
        write_json,
    )


def successful_attempts(repeat_dir: Path) -> tuple[list[Path], list[Path]]:
    completed: list[Path] = []
    unsuccessful: list[Path] = []
    for metadata_path in sorted(repeat_dir.glob("attempt-*/metadata.json")):
        metadata = read_json(metadata_path)
        if metadata.get("status") == "completed":
            completed.append(metadata_path.parent)
        else:
            unsuccessful.append(metadata_path.parent)
    return completed, unsuccessful


def validate_window_attestation(metadata: dict[str, Any], metadata_path: Path) -> None:
    window_id = metadata.get("measurement_window_id")
    deadline = metadata.get("window_deadline_epoch")
    if window_id is None and deadline is None:
        return
    if not isinstance(window_id, str) or not window_id:
        raise ValidationError(f"invalid measurement window id: {metadata_path}")
    if (
        isinstance(deadline, bool)
        or not isinstance(deadline, (int, float))
        or not math.isfinite(float(deadline))
    ):
        raise ValidationError(f"invalid measurement window deadline: {metadata_path}")
    timestamps = {}
    for field in ("started_at_utc", "finished_at_utc"):
        value = metadata.get(field)
        if not isinstance(value, str):
            raise ValidationError(
                f"window-attested attempt lacks {field}: {metadata_path}"
            )
        try:
            timestamps[field] = datetime.fromisoformat(
                value.replace("Z", "+00:00")
            ).timestamp()
        except ValueError as exc:
            raise ValidationError(
                f"window-attested attempt has invalid {field}: {metadata_path}"
            ) from exc
    if not (
        timestamps["started_at_utc"] - 2
        <= timestamps["finished_at_utc"]
        <= float(deadline) + 2
    ):
        raise ValidationError(
            f"completed attempt finished outside its measurement window: {metadata_path}"
        )


def registered_artifact(
    campaign_dir: Path,
    manifest: dict[str, Any],
    key: str,
) -> tuple[Path, dict[str, Any]]:
    registration = manifest.get(key)
    if not isinstance(registration, dict):
        raise ValidationError(f"campaign is missing {key} registration")
    filename = registration.get("filename")
    digest = registration.get("sha256")
    if not isinstance(filename, str) or not isinstance(digest, str):
        raise ValidationError(f"campaign has malformed {key} registration")
    path = campaign_dir / filename
    if not path.is_file() or sha256_file(path) != digest:
        raise ValidationError(f"campaign {key} artifact changed")
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValidationError(f"campaign {key} artifact is not an object")
    return path, payload


def bootstrap_hash_for_dataset(
    cache_hashes: dict[str, str],
    dataset: str,
    batch_strategy: str,
) -> tuple[str, str]:
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
    return candidates[0]


def validate_campaign(campaign_dir: Path, allow_partial: bool = False) -> dict[str, Any]:
    protocol = load_protocol()
    manifest_path = campaign_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValidationError(f"missing campaign manifest: {manifest_path}")
    manifest = read_json(manifest_path)
    if manifest.get("schema") != "comnetx-ieee-access-ldleiden-campaign-v1":
        raise ValidationError("unexpected campaign manifest schema")
    if manifest.get("protocol_id") != protocol["protocol_id"]:
        raise ValidationError("campaign uses a different protocol id")
    current_fingerprint = protocol_fingerprint(protocol)
    recorded_fingerprint = manifest.get("fingerprint", {})
    if recorded_fingerprint.get("protocol_sha256") != current_fingerprint.get(
        "protocol_sha256"
    ) or recorded_fingerprint.get("config_sha256") != current_fingerprint.get(
        "config_sha256"
    ):
        raise ValidationError(
            "protocol or launcher configs differ from campaign start"
        )
    if manifest.get("backend", {}).get("num_jobs_verified") is not True:
        raise ValidationError("campaign does not verify LD-Leiden j=1")
    if manifest.get("backend", {}).get("num_jobs_observed") != 1:
        raise ValidationError("campaign used an unexpected LD-Leiden worker count")
    if manifest.get("backend", {}).get("num_jobs_source") != (
        "AlgorithmOptions default"
    ) or manifest.get("backend", {}).get(
        "num_jobs_explicitly_passed_by_launcher"
    ) is not False:
        raise ValidationError(
            "campaign does not establish j=1 from the readable AlgorithmOptions default"
        )
    worker_policy = manifest.get("worker_policy", {})
    if worker_policy.get("num_jobs") != 1 or set(
        worker_policy.get("thread_environment", {}).values()
    ) != {"1"}:
        raise ValidationError("campaign worker/thread policy is not the registered j=1 policy")
    if worker_policy.get("num_jobs_source") != (
        "verified AlgorithmOptions default"
    ) or worker_policy.get("num_jobs_explicitly_passed_by_launcher") is not False:
        raise ValidationError("campaign misstates how the j=1 policy was established")
    hardware_path, hardware = registered_artifact(campaign_dir, manifest, "hardware")
    runtime_path, runtime_identity = registered_artifact(
        campaign_dir,
        manifest,
        "runtime_identity",
    )
    input_manifest_path, input_manifest = registered_artifact(
        campaign_dir,
        manifest,
        "real_input_manifest",
    )
    if runtime_identity.get("backend") != manifest.get("backend"):
        raise ValidationError("runtime identity and manifest backend differ")
    if runtime_identity.get("hardware_sha256") != sha256_json(hardware):
        raise ValidationError("runtime identity and sealed hardware differ")
    if input_manifest.get("paths_config_sha256") != manifest.get(
        "paths_config", {}
    ).get("sha256"):
        raise ValidationError("input manifest and paths-config identity differ")
    if (
        input_manifest.get("schema")
        != "comnetx-ieee-access-ldleiden-real-inputs-v1"
        or not isinstance(input_manifest.get("files"), list)
        or not input_manifest["files"]
    ):
        raise ValidationError("sealed real-input manifest is structurally invalid")

    expected_attempt_identity = {
        "fingerprint_sha256": sha256_json(recorded_fingerprint),
        "hardware_sha256": manifest["hardware"]["sha256"],
        "runtime_identity_sha256": manifest["runtime_identity"]["sha256"],
        "real_input_manifest_sha256": manifest["real_input_manifest"]["sha256"],
    }

    report: dict[str, Any] = {
        "schema": "comnetx-ieee-access-ldleiden-validation-v1",
        "campaign_id": manifest["campaign_id"],
        "protocol_id": protocol["protocol_id"],
        "status": "valid",
        "allow_partial": allow_partial,
        "clock_policy": protocol["clock_policy"],
        "measurement_source_sha256": recorded_fingerprint.get("source_sha256"),
        "measurement_source_matches_current": (
            recorded_fingerprint.get("source_sha256")
            == current_fingerprint.get("source_sha256")
        ),
        "sealed_artifacts": {
            "hardware": {
                "filename": hardware_path.name,
                "sha256": manifest["hardware"]["sha256"],
            },
            "runtime_identity": {
                "filename": runtime_path.name,
                "sha256": manifest["runtime_identity"]["sha256"],
            },
            "real_input_manifest": {
                "filename": input_manifest_path.name,
                "sha256": manifest["real_input_manifest"]["sha256"],
                "files": len(input_manifest.get("files", [])),
            },
        },
        "smoke_excluded_from_analysis": True,
        "failed_or_invalid_attempts": [],
        "phases": {},
    }
    missing: list[str] = []
    bootstrap_observations: dict[str, dict[str, set[str]]] = {}
    for phase in protocol["phases"]:
        per_dataset: dict[str, list[dict[str, Any]]] = {
            dataset: [] for dataset in phase["datasets"]
        }
        phase_report: dict[str, Any] = {
            "role": phase["role"],
            "included_in_analysis": phase["included_in_analysis"],
            "expected_repetitions": phase["repetitions"],
            "completed_repetitions": 0,
            "datasets": {},
        }
        for repeat_index in range(1, int(phase["repetitions"]) + 1):
            repeat_dir = campaign_dir / phase["id"] / f"repeat-{repeat_index:02d}"
            completed, unsuccessful = successful_attempts(repeat_dir)
            report["failed_or_invalid_attempts"].extend(
                str(path.relative_to(campaign_dir)) for path in unsuccessful
            )
            if len(completed) > 1:
                raise ValidationError(
                    f"{phase['id']} repeat {repeat_index} has multiple completed attempts"
                )
            if not completed:
                missing.append(f"{phase['id']}/repeat-{repeat_index:02d}")
                continue
            attempt_dir = completed[0]
            metadata = read_json(attempt_dir / "metadata.json")
            validate_window_attestation(metadata, attempt_dir / "metadata.json")
            expected_metadata = {
                "phase_id": phase["id"],
                "role": phase["role"],
                "included_in_analysis": phase["included_in_analysis"],
                "repeat": repeat_index,
            }
            for key, expected in expected_metadata.items():
                if metadata.get(key) != expected:
                    raise ValidationError(
                        f"{attempt_dir}: expected metadata {key}={expected!r}, "
                        f"got {metadata.get(key)!r}"
                    )
            if metadata.get("identity_before") != expected_attempt_identity:
                raise ValidationError(
                    f"{attempt_dir}: pre-run source/input/runtime identity mismatch"
                )
            if metadata.get("identity_after") != expected_attempt_identity:
                raise ValidationError(
                    f"{attempt_dir}: post-run source/input/runtime identity mismatch"
                )
            if set(metadata.get("thread_environment", {}).values()) != {"1"}:
                raise ValidationError(f"{attempt_dir}: thread policy is not fixed at one")
            result_path = launcher_result_file(attempt_dir, phase)
            if not result_path.is_file():
                raise ValidationError(f"missing result file: {result_path}")
            if metadata.get("result_sha256") != sha256_file(result_path):
                raise ValidationError(f"result hash changed: {result_path}")
            cache_hashes = metadata.get("bootstrap_cache_sha256")
            if not isinstance(cache_hashes, dict):
                raise ValidationError(
                    f"missing bootstrap cache hashes: {attempt_dir / 'metadata.json'}"
                )
            for dataset in phase["datasets"]:
                filename, digest = bootstrap_hash_for_dataset(
                    cache_hashes,
                    dataset,
                    phase["batch_strategy"],
                )
                key = f"{dataset}/{phase['batch_strategy'].split(':', 1)[0]}"
                observation = bootstrap_observations.setdefault(
                    key, {"filenames": set(), "sha256": set()}
                )
                observation["filenames"].add(filename)
                observation["sha256"].add(digest)
            summaries = validate_launcher_file(result_path, phase)
            semantic_records = {
                dataset: bootstrap_cache_reference(
                    campaign_dir / "bootstrap-cache",
                    cache_hashes,
                    dataset,
                    phase["batch_strategy"],
                    summaries[dataset]["bootstrap_reference_sha256"],
                )
                for dataset in phase["datasets"]
            }
            if metadata.get("bootstrap_semantics") != semantic_records:
                raise ValidationError(
                    f"{attempt_dir}: bootstrap semantic attestation changed"
                )
            if metadata.get("validated_summary") != summaries:
                raise ValidationError(
                    f"{attempt_dir}: stored launcher summary differs from validation"
                )
            phase_report["completed_repetitions"] += 1
            for dataset, summary in summaries.items():
                per_dataset[dataset].append(summary)

        if phase_report["completed_repetitions"]:
            phase_report["datasets"] = {
                dataset: aggregate_run_summaries(rows)
                for dataset, rows in per_dataset.items()
                if rows
            }
        report["phases"][phase["id"]] = phase_report

    inconsistent_bootstraps = {
        key: values
        for key, values in bootstrap_observations.items()
        if len(values["filenames"]) != 1 or len(values["sha256"]) != 1
    }
    if inconsistent_bootstraps:
        raise ValidationError(
            "cached Leiden bootstrap changed across repetitions: "
            + ", ".join(sorted(inconsistent_bootstraps))
        )
    report["bootstrap_cache"] = {
        key: {
            "filename": next(iter(values["filenames"])),
            "sha256": next(iter(values["sha256"])),
        }
        for key, values in sorted(bootstrap_observations.items())
    }

    if missing:
        report["missing_repetitions"] = missing
        if not allow_partial:
            raise ValidationError(
                "campaign is incomplete: " + ", ".join(missing)
            )
        if report["phases"]["smoke_999_10"]["completed_repetitions"] != 1:
            raise ValidationError(
                "partial validation requires a completed smoke timing gate"
            )
        report["status"] = "valid_partial"
    else:
        short = report["phases"]["measured_999_10"]
        long = report["phases"]["measured_9_500"]
        if short["completed_repetitions"] != 5:
            raise ValidationError("measured short phase must contain exactly five repeats")
        if long["completed_repetitions"] != 3:
            raise ValidationError("measured long phase must contain exactly three repeats")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate split clocks, repetition counts, and quality fields."
    )
    parser.add_argument(
        "campaign",
        type=Path,
        help="Campaign directory or campaign id below the default result root.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Permit incomplete phases for an interim smoke/queue check.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Output path; defaults to CAMPAIGN/validation_report.json.",
    )
    args = parser.parse_args()

    campaign_dir = args.campaign.expanduser()
    if not campaign_dir.is_absolute() and not campaign_dir.exists():
        campaign_dir = RESULTS_ROOT / campaign_dir
    campaign_dir = campaign_dir.resolve()
    report = validate_campaign(campaign_dir, allow_partial=args.allow_partial)
    report_path = (
        args.report.expanduser().resolve()
        if args.report
        else campaign_dir / "validation_report.json"
    )
    write_json(report_path, report)
    print(f"Validation status: {report['status']}")
    print(f"Report: {report_path}")
    for phase_id, phase in report["phases"].items():
        print(
            f"{phase_id}: {phase['completed_repetitions']}/"
            f"{phase['expected_repetitions']} repetitions"
        )


if __name__ == "__main__":
    main()
