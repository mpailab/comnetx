#!/usr/bin/env python3
"""Validate stages and summarize a repaired-ComNetX measurement campaign."""

from __future__ import annotations

import argparse
from datetime import datetime
import math
import re
from pathlib import Path
from typing import Any

try:
    from .input_manifest import validate_input_manifest
    from .protocol import (
        ALL_DATASETS,
        CORE_DATASETS,
        EXPECTED_UPDATES,
        RESULTS_ROOT,
        ValidationError,
        aggregate,
        flatten_launcher_payload,
        json_file_hashes,
        load_protocol,
        protocol_fingerprint,
        read_json,
        sha256_file,
        stage_map,
        validate_bootstrap_cache_entry,
        validate_bootstrap_hash_history,
        validate_paired_bootstrap,
        validate_series,
        write_json,
    )
except ImportError:  # Direct script execution.
    from input_manifest import validate_input_manifest
    from protocol import (
        ALL_DATASETS,
        CORE_DATASETS,
        EXPECTED_UPDATES,
        RESULTS_ROOT,
        ValidationError,
        aggregate,
        flatten_launcher_payload,
        json_file_hashes,
        load_protocol,
        protocol_fingerprint,
        read_json,
        sha256_file,
        stage_map,
        validate_bootstrap_cache_entry,
        validate_bootstrap_hash_history,
        validate_paired_bootstrap,
        validate_series,
        write_json,
    )


SMART_LEIDEN = "leidenalg-L:3-r:1-gpu"
FULL_LEIDEN = "leidenalg-naive"
DSBM_RE = re.compile(
    r"^dsbm-(random|hubs|community)-.*-mc(290|1450)-(42|43|44|45|46)$"
)


def _numeric(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValidationError(f"{label} is not finite")
    return result


def validate_launch_budget(
    campaign_dir: Path,
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    budget_path = campaign_dir / "launch_budget.json"
    if not budget_path.is_file():
        raise ValidationError("campaign launch budget is missing")
    budget = read_json(budget_path)
    expected = {
        "schema": "comnetx-ieee-access-launch-budget-v2",
        "git_sha": manifest.get("git", {}).get("commit"),
        "repaired_campaign_id": manifest.get("campaign_id"),
        "paths_config_sha256": manifest.get("paths_config", {}).get("sha256"),
        "initial_window_hours": 24,
        "max_total_hours": 72,
    }
    for key, value in expected.items():
        if budget.get(key) != value:
            raise ValidationError(f"campaign launch budget differs for {key}")
    if not isinstance(budget.get("ld_campaign_id"), str):
        raise ValidationError("campaign launch budget lacks the LD campaign id")
    windows = budget.get("windows")
    if not isinstance(windows, list) or not windows:
        raise ValidationError("campaign launch budget has no measurement windows")
    by_id: dict[str, dict[str, Any]] = {}
    total = 0.0
    previous_deadline: int | None = None
    for index, window in enumerate(windows, start=1):
        if not isinstance(window, dict):
            raise ValidationError("campaign launch budget contains a malformed window")
        window_id = window.get("id")
        if window_id != f"window-{index:03d}" or window_id in by_id:
            raise ValidationError("campaign launch budget has an invalid window id")
        granted = _numeric(window.get("granted_hours"), f"{window_id} grant")
        started = window.get("started_at_epoch")
        deadline = window.get("deadline_epoch")
        if (
            granted <= 0
            or not isinstance(started, int)
            or not isinstance(deadline, int)
            or deadline != started + int(round(granted * 3600))
        ):
            raise ValidationError(f"malformed measurement window: {window_id}")
        if previous_deadline is not None and started < previous_deadline:
            raise ValidationError("campaign measurement windows overlap")
        if index == 1 and granted != 24:
            raise ValidationError("the first measurement window must be exactly 24 hours")
        total += granted
        previous_deadline = deadline
        by_id[window_id] = window
    if total > 72 + 1e-12:
        raise ValidationError("campaign launch budget exceeds its 72-hour cap")
    return budget, by_id


def validate_attempt_window(
    metadata: dict[str, Any],
    metadata_path: Path,
    windows: dict[str, dict[str, Any]],
) -> None:
    window_id = metadata.get("measurement_window_id")
    window = windows.get(window_id) if isinstance(window_id, str) else None
    if window is None:
        raise ValidationError(f"attempt lacks a registered window: {metadata_path}")
    if metadata.get("window_deadline_epoch") != window["deadline_epoch"]:
        raise ValidationError(f"attempt window deadline changed: {metadata_path}")
    started_at = metadata.get("started_at_utc")
    if not isinstance(started_at, str):
        raise ValidationError(f"attempt lacks a UTC start timestamp: {metadata_path}")
    try:
        started_epoch = datetime.fromisoformat(
            started_at.replace("Z", "+00:00")
        ).timestamp()
    except ValueError as exc:
        raise ValidationError(f"attempt has an invalid UTC start: {metadata_path}") from exc
    if not window["started_at_epoch"] - 2 <= started_epoch <= window["deadline_epoch"]:
        raise ValidationError(f"attempt started outside its window: {metadata_path}")
    if metadata.get("status") == "completed":
        finished_at = metadata.get("finished_at_utc")
        if not isinstance(finished_at, str):
            raise ValidationError(
                f"completed attempt lacks a UTC finish: {metadata_path}"
            )
        try:
            finished_epoch = datetime.fromisoformat(
                finished_at.replace("Z", "+00:00")
            ).timestamp()
        except ValueError as exc:
            raise ValidationError(
                f"completed attempt has an invalid UTC finish: {metadata_path}"
            ) from exc
        effective_deadline = float(window["deadline_epoch"])
        stop_reserve = metadata.get("stop_boundary_hours_left")
        if stop_reserve is not None:
            stop_reserve = _numeric(
                stop_reserve, f"{metadata_path} stop-boundary reserve"
            )
            if stop_reserve < 0:
                raise ValidationError(
                    f"attempt has a negative stop-boundary reserve: {metadata_path}"
                )
            effective_deadline -= stop_reserve * 3600.0
        if not started_epoch - 2 <= finished_epoch <= effective_deadline + 2:
            raise ValidationError(
                f"completed attempt finished outside its window: {metadata_path}"
            )


def completed_attempt(campaign_dir: Path, stage_id: str, command_name: str) -> Path:
    command_dir = campaign_dir / "stages" / stage_id / command_name
    attempts = []
    for metadata_path in sorted(command_dir.glob("attempt-*/metadata.json")):
        metadata = read_json(metadata_path)
        if metadata.get("status") == "completed":
            attempts.append(metadata_path.parent)
    if len(attempts) != 1:
        raise ValidationError(
            f"{stage_id}/{command_name}: expected one completed attempt, got {len(attempts)}"
        )
    attempt = attempts[0]
    metadata = read_json(attempt / "metadata.json")
    if metadata.get("stage_id") != stage_id or metadata.get("command_name") != command_name:
        raise ValidationError(f"{attempt}: command metadata identity mismatch")
    campaign_manifest = read_json(campaign_dir / "manifest.json")
    if stage_id != "preflight" and (campaign_dir / "launch_budget.json").is_file():
        _, windows = validate_launch_budget(campaign_dir, campaign_manifest)
        validate_attempt_window(metadata, attempt / "metadata.json", windows)
    sealed = campaign_manifest.get("fingerprint")
    if (
        metadata.get("fingerprint_before") != sealed
        or metadata.get("fingerprint_after") != sealed
    ):
        raise ValidationError(f"{attempt}: command fingerprint does not match campaign")
    runtime_before = metadata.get("runtime_identity_before")
    runtime_after = metadata.get("runtime_identity_after")
    if (
        not isinstance(runtime_before, dict)
        or runtime_before != runtime_after
        or not isinstance(runtime_before.get("hardware_sha256"), str)
    ):
        raise ValidationError(f"{attempt}: runtime identity was not held fixed")
    expected_runtime = {
        "hardware_sha256": campaign_manifest.get("hardware", {}).get("sha256"),
        "environment_sha256": (
            None
            if stage_id == "preflight"
            else campaign_manifest.get("environment", {}).get("sha256")
        ),
    }
    if runtime_before != expected_runtime:
        raise ValidationError(f"{attempt}: runtime identity differs from campaign")
    current_hashes = json_file_hashes(attempt)
    if metadata.get("json_sha256") != current_hashes:
        raise ValidationError(f"{attempt}: JSON output changed after command completion")
    stdout_path = attempt / "stdout.log"
    if not stdout_path.is_file() or metadata.get("stdout_sha256") != sha256_file(
        stdout_path
    ):
        raise ValidationError(f"{attempt}: stdout changed after command completion")
    if any(path.startswith("errors_") for path in current_hashes):
        raise ValidationError(f"{attempt}: error output is present")
    return attempt


def launcher_output(attempt: Path, config_stem: str) -> Path:
    path = attempt / f"{config_stem}_now.json"
    if not path.is_file():
        raise ValidationError(f"missing launcher output: {path}")
    return path


def validate_preflight(campaign_dir: Path) -> dict[str, Any]:
    marker = read_json(campaign_dir / "preflight" / "validation.json")
    if marker.get("status") != "validated":
        raise ValidationError("campaign preflight marker is missing or invalid")
    completed_attempt(campaign_dir, "preflight", "lint")
    completed_attempt(campaign_dir, "preflight", "unit")
    environment_attempt = completed_attempt(
        campaign_dir, "preflight", "gpu_environment"
    )
    environment = read_json(environment_attempt / "environment.json")
    if (
        environment.get("schema") != "comnetx-ieee-access-gpu-environment-v1"
        or environment.get("cuda_available") is not True
        or int(environment.get("cuda_device_count", 0)) < 1
        or environment.get("dynamic_graphs_communities_import")
        != "dynamic_graphs_communities"
    ):
        raise ValidationError("preflight did not verify the requested GPU/backend stack")
    environment_registration = read_json(campaign_dir / "manifest.json").get(
        "environment", {}
    )
    sealed_environment = campaign_dir / environment_registration.get(
        "filename", ""
    )
    if (
        not sealed_environment.is_file()
        or environment_registration.get("sha256")
        != sha256_file(sealed_environment)
        or read_json(sealed_environment) != environment
    ):
        raise ValidationError("sealed environment differs from preflight")
    input_attempt = completed_attempt(campaign_dir, "preflight", "input_manifest")
    measured_inputs = read_json(input_attempt / "input_manifest.json")
    sealed_inputs_path = campaign_dir / "real_input_manifest.json"
    if not sealed_inputs_path.is_file() or read_json(sealed_inputs_path) != measured_inputs:
        raise ValidationError("sealed real-input manifest differs from preflight")
    validate_input_manifest(measured_inputs, verify_content=True)
    return {
        "environment": environment,
        "real_input_files": len(measured_inputs["files"]),
        "real_input_manifest_sha256": sha256_file(sealed_inputs_path),
    }


def validate_launcher(
    path: Path,
    *,
    datasets: tuple[str, ...],
    batch: str,
    required_algorithms: set[str] | None = None,
) -> dict[tuple[str, str], dict[str, float | int]]:
    flat = flatten_launcher_payload(
        read_json(path),
        expected_datasets=datasets,
        expected_batch=batch,
    )
    algorithms = {algorithm for algorithm, _ in flat}
    if required_algorithms is not None and algorithms != required_algorithms:
        raise ValidationError(
            f"{path}: expected algorithms {sorted(required_algorithms)}, "
            f"got {sorted(algorithms)}"
        )
    return {
        key: validate_series(series, dataset=key[1], batch=batch)
        for key, series in flat.items()
    }


def audit_summary(path: Path, dataset: str) -> dict[str, Any]:
    report = read_json(path)
    if report.get("schema") != "comnetx_stream_invariant_audit_v1":
        raise ValidationError(f"{dataset}: unexpected audit schema")
    if (
        report.get("dataset") not in {dataset, f"{dataset}-sym"}
        or report.get("batch_strategy") != "999:10"
        or report.get("method") != "leidenalg"
        or report.get("mode") != "smart"
        or report.get("depth") != 3
        or report.get("radius") != 1
        or float(report.get("resolution", math.nan)) != 1.0
        or report.get("aggregation_mode") != "sum"
        or report.get("force_undirected") is not True
        or not str(report.get("device", "")).startswith("cuda")
    ):
        raise ValidationError(f"{dataset}: audit protocol identity mismatch")
    bootstrap = report.get("bootstrap", {})
    if (
        bootstrap.get("strategy_has_bootstrap_batch") is not True
        or bootstrap.get("implementation")
        != "launcher._compute_launch_initial_partition"
        or bootstrap.get("initial_batch_number") != "999"
        or bootstrap.get("hierarchy_cache_schema") != "parent_quotient_v1"
        or bootstrap.get("campaign_cache_enabled") is not True
    ):
        raise ValidationError(f"{dataset}: audit bootstrap identity mismatch")
    if report.get("initial_refinement", {}).get("all_adjacent_refine") is not True:
        raise ValidationError(f"{dataset}: initial hierarchy is not nested")
    summary = report.get("summary", {})
    expected = EXPECTED_UPDATES["999:10"][dataset]
    if summary.get("updates_audited") != expected:
        raise ValidationError(f"{dataset}: incomplete stream audit")
    empty_gates = (
        "pre_update_refinement_failure_updates",
        "post_update_refinement_failure_updates",
        "scope_collision_before_updates",
        "scope_collision_after_updates",
        "outside_entry_write_updates",
    )
    failures = {name: summary.get(name) for name in empty_gates if summary.get(name) != []}
    if failures:
        raise ValidationError(f"{dataset}: invariant gate failed: {failures}")
    final_q = summary.get("final_reported_modularity")
    if not isinstance(final_q, (int, float)) or not math.isfinite(float(final_q)):
        raise ValidationError(f"{dataset}: audit final modularity is invalid")
    return summary


def validate_stage1(campaign_dir: Path) -> dict[str, Any]:
    attempt = completed_attempt(
        campaign_dir, "stage1_correctness_smoke", "all_six_smart_smoke"
    )
    launcher = validate_launcher(
        launcher_output(attempt, "stage1_smart_smoke_999_10"),
        datasets=ALL_DATASETS,
        batch="999:10",
        required_algorithms={SMART_LEIDEN},
    )
    bootstrap = {
        dataset: validate_bootstrap_cache_entry(
            campaign_dir,
            attempt,
            dataset=dataset,
            initial_batch=999,
            depth=3,
        )
        for dataset in ALL_DATASETS
    }
    audits = {}
    for dataset in ALL_DATASETS:
        audit_attempt = completed_attempt(
            campaign_dir,
            "stage1_correctness_smoke",
            f"invariant_audit_{dataset}",
        )
        audits[dataset] = audit_summary(audit_attempt / "audit.json", dataset)
    return {
        "status": "validated",
        "hard_gate": "hierarchy_namespace_and_scope_locality_passed",
        "launcher": {
            dataset: launcher[(SMART_LEIDEN, dataset)] for dataset in ALL_DATASETS
        },
        "bootstrap_sha256": bootstrap,
        "audits": audits,
    }


def validate_stage2(campaign_dir: Path) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, float | int]]] = {
        (algorithm, dataset): []
        for algorithm in (FULL_LEIDEN, SMART_LEIDEN)
        for dataset in CORE_DATASETS
    }
    bootstraps: dict[str, list[dict[str, Any]]] = {
        dataset: [] for dataset in CORE_DATASETS
    }
    for repeat in range(1, 6):
        attempt = completed_attempt(
            campaign_dir, "stage2_core_short", f"paired_repeat_{repeat:02d}"
        )
        rows = validate_launcher(
            launcher_output(attempt, "stage2_paired_short_999_10"),
            datasets=CORE_DATASETS,
            batch="999:10",
            required_algorithms={FULL_LEIDEN, SMART_LEIDEN},
        )
        for dataset in CORE_DATASETS:
            bootstraps[dataset].append(
                validate_paired_bootstrap(
                    campaign_dir,
                    attempt,
                    dataset=dataset,
                    initial_batch=999,
                )
            )
        for key, row in rows.items():
            grouped[key].append(row)
    aggregates = {
        f"{algorithm}/{dataset}": aggregate(rows)
        for (algorithm, dataset), rows in grouped.items()
    }
    paired = {}
    for dataset in CORE_DATASETS:
        full = aggregates[f"{FULL_LEIDEN}/{dataset}"]
        smart = aggregates[f"{SMART_LEIDEN}/{dataset}"]
        paired[dataset] = {
            "speedup": full["total_time"]["mean"] / smart["total_time"]["mean"],
            "delta_q": (
                smart["final_modularity"]["mean"]
                - full["final_modularity"]["mean"]
            ),
            "delta_nmi": smart["final_nmi"]["mean"] - full["final_nmi"]["mean"],
            "pareto_relevant_rule": (
                smart["total_time"]["mean"] < full["total_time"]["mean"]
                and smart["final_modularity"]["mean"]
                - full["final_modularity"]["mean"]
                >= -0.05
            ),
        }
    return {
        "status": "validated",
        "repetitions": 5,
        "aggregates": aggregates,
        "bootstrap_sha256_by_repeat": bootstraps,
        "paired": paired,
        "optional_breadth_go": any(
            row["pareto_relevant_rule"] for row in paired.values()
        ),
        "decision_note": (
            "A false optional_breadth_go retains these results and triggers "
            "claim reframing; it does not invalidate or delete the measurement."
        ),
    }


def completed_profiles(path: Path, expected_count: int) -> list[dict[str, Any]]:
    payload = read_json(path)
    if payload.get("profile_schema") != "comnetx_smart_workload_v2":
        raise ValidationError(f"profile output does not use the production-run schema: {path}")
    if payload.get("status") != "completed" or payload.get("errors") != []:
        raise ValidationError(f"profile output is incomplete or has errors: {path}")
    profiles = payload.get("profiles")
    if not isinstance(profiles, list) or len(profiles) != expected_count:
        raise ValidationError(
            f"{path}: expected {expected_count} profiles, got "
            f"{len(profiles) if isinstance(profiles, list) else type(profiles)}"
        )
    if any(profile.get("incomplete") is not False for profile in profiles):
        raise ValidationError(f"{path}: at least one profile is incomplete")
    return profiles


BOUNDARY_CERTIFICATE_BASE_FIELDS = (
    "gamma",
    "W",
    "W_U",
    "beta_out",
    "beta_in",
)
BOUNDARY_CERTIFICATE_DERIVED_FIELDS = (
    "B_plus",
    "B_minus",
    "certificate_width",
    "D",
    "Q_U",
)


def validate_boundary_certificates(
    row: dict[str, Any],
    *,
    context: str,
    expected_levels: int,
) -> list[dict[str, Any]]:
    """Validate every per-level boundary certificate in one profile row."""

    levels = row.get("levels")
    mirrored = row.get("boundary_certificates_by_level")
    if not isinstance(levels, list) or len(levels) != expected_levels:
        raise ValidationError(f"{context}: incomplete certificate level profile")
    if not isinstance(mirrored, list) or len(mirrored) != expected_levels:
        raise ValidationError(f"{context}: incomplete mirrored certificates")

    validated: list[dict[str, Any]] = []
    level_certificate_times = []
    reference_w = None
    for index, level in enumerate(levels):
        if not isinstance(level, dict) or level.get("level") != index:
            raise ValidationError(f"{context}: certificate level order mismatch")
        certificate_time = level.get("certificate_time")
        if (
            not isinstance(certificate_time, (int, float))
            or not math.isfinite(float(certificate_time))
            or float(certificate_time) < 0.0
        ):
            raise ValidationError(f"{context}/level-{index}: invalid certificate time")
        level_certificate_times.append(float(certificate_time))
        certificate = level.get("boundary_certificate")
        if not isinstance(certificate, dict) or mirrored[index] != certificate:
            raise ValidationError(
                f"{context}/level-{index}: missing or inconsistent certificate"
            )
        finite = certificate.get("finite")
        if not isinstance(finite, bool):
            raise ValidationError(
                f"{context}/level-{index}: missing finite certificate flag"
            )
        if any(
            not isinstance(certificate.get(field), (int, float))
            or isinstance(certificate.get(field), bool)
            or not math.isfinite(float(certificate[field]))
            for field in BOUNDARY_CERTIFICATE_BASE_FIELDS
        ):
            raise ValidationError(
                f"{context}/level-{index}: invalid certificate mass"
            )

        values = {
            field: float(certificate[field])
            for field in BOUNDARY_CERTIFICATE_BASE_FIELDS
        }
        gamma = values["gamma"]
        w = values["W"]
        w_u = values["W_U"]
        beta_out = values["beta_out"]
        beta_in = values["beta_in"]
        if (
            gamma < 0.0
            or w < 0.0
            or w_u < 0.0
            or beta_out < 0.0
            or beta_in < 0.0
        ):
            raise ValidationError(
                f"{context}/level-{index}: certificate violates sign assumptions"
            )

        weight_tolerance = 1e-10 * max(1.0, abs(w))
        if w_u > w + weight_tolerance or (
            w_u + beta_out + beta_in > w + weight_tolerance
        ):
            raise ValidationError(
                f"{context}/level-{index}: inconsistent graph-weight decomposition"
            )
        if reference_w is None:
            reference_w = w
        elif not math.isclose(
            w, reference_w, rel_tol=1e-12, abs_tol=weight_tolerance
        ):
            raise ValidationError(
                f"{context}/level-{index}: full weight changed across levels"
            )

        if not finite:
            if w > 0.0 and w_u > 0.0:
                raise ValidationError(
                    f"{context}/level-{index}: non-finite flag is inconsistent"
                )
            if any(
                certificate.get(field) is not None
                for field in BOUNDARY_CERTIFICATE_DERIVED_FIELDS
            ):
                raise ValidationError(
                    f"{context}/level-{index}: undefined certificate has scalars"
                )
            validated.append(
                {
                    **values,
                    **{
                        field: None
                        for field in BOUNDARY_CERTIFICATE_DERIVED_FIELDS
                    },
                    "finite": False,
                }
            )
            continue

        if w <= 0.0 or w_u <= 0.0 or any(
            not isinstance(certificate.get(field), (int, float))
            or isinstance(certificate.get(field), bool)
            or not math.isfinite(float(certificate[field]))
            for field in BOUNDARY_CERTIFICATE_DERIVED_FIELDS
        ):
            raise ValidationError(
                f"{context}/level-{index}: invalid finite certificate scalar"
            )
        values.update(
            {
                field: float(certificate[field])
                for field in BOUNDARY_CERTIFICATE_DERIVED_FIELDS
            }
        )
        b_plus = values["B_plus"]
        b_minus = values["B_minus"]
        width = values["certificate_width"]
        mismatch = values["D"]
        if b_plus < 0.0 or b_minus < 0.0 or width < 0.0:
            raise ValidationError(
                f"{context}/level-{index}: certificate violates sign assumptions"
            )

        scale = gamma / (w * w)
        expected_b_plus = scale * w_u * (w - w_u)
        expected_b_minus = scale * (
            w_u * (beta_out + beta_in) + beta_out * beta_in
        )
        if not math.isclose(
            b_plus, expected_b_plus, rel_tol=1e-10, abs_tol=1e-12
        ) or not math.isclose(
            b_minus, expected_b_minus, rel_tol=1e-10, abs_tol=1e-12
        ):
            raise ValidationError(
                f"{context}/level-{index}: certificate bounds were miscomputed"
            )
        if not math.isclose(
            width, b_plus + b_minus, rel_tol=1e-10, abs_tol=1e-12
        ):
            raise ValidationError(
                f"{context}/level-{index}: certificate width is inconsistent"
            )
        bound_tolerance = 1e-10 * max(
            1.0, b_plus, b_minus, abs(mismatch)
        )
        if (
            mismatch < -b_minus - bound_tolerance
            or mismatch > b_plus + bound_tolerance
        ):
            raise ValidationError(
                f"{context}/level-{index}: realized mismatch exceeds its bounds"
            )
        validated.append({**values, "finite": True})
    total_certificate_time = row.get("certificate_time")
    if (
        not isinstance(total_certificate_time, (int, float))
        or not math.isfinite(float(total_certificate_time))
        or float(total_certificate_time) < 0.0
        or not math.isclose(
            float(total_certificate_time),
            sum(level_certificate_times),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise ValidationError(f"{context}: inconsistent total certificate time")
    return validated


def summarize_boundary_certificates(
    certificates: list[dict[str, Any]],
) -> dict[str, float | int | None]:
    """Return article-facing summaries after certificate validation."""

    if not certificates:
        raise ValidationError("cannot summarize an empty certificate collection")
    finite = [item for item in certificates if item["finite"] is True]
    widths = [float(item["certificate_width"]) for item in finite]
    mismatches = [abs(float(item["D"])) for item in finite]
    internal_fractions = [
        float(item["W_U"]) / float(item["W"]) for item in certificates
        if float(item["W"]) > 0.0
    ]
    return {
        "total_levels": len(certificates),
        "finite_levels": len(finite),
        "undefined_zero_weight_levels": len(certificates) - len(finite),
        "finite_fraction": len(finite) / len(certificates),
        "mean_certificate_width": (
            sum(widths) / len(widths) if widths else None
        ),
        "max_certificate_width": max(widths) if widths else None,
        "mean_absolute_realized_mismatch": (
            sum(mismatches) / len(mismatches) if mismatches else None
        ),
        "max_absolute_realized_mismatch": (
            max(mismatches) if mismatches else None
        ),
        "mean_internal_weight_fraction": (
            sum(internal_fractions) / len(internal_fractions)
            if internal_fractions
            else None
        ),
    }


RANKING_CERTIFICATE_NUMERIC_FIELDS = (
    "scaled_local_gap",
    "restricted_full_objective_gap",
    "certificate_width",
    "output_D",
    "reference_D",
    "output_Q_U",
    "reference_Q_U",
    "W_U_over_W",
    "comparison_tolerance",
)


def validate_identity_ranking_certificates(
    row: dict[str, Any],
    boundary_certificates: list[dict[str, Any]],
    *,
    context: str,
    expected_levels: int,
) -> list[dict[str, Any]]:
    """Independently check identity-reference ranking signs and thresholds."""

    levels = row.get("levels")
    mirrored = row.get("identity_ranking_certificates_by_level")
    if (
        not isinstance(levels, list)
        or len(levels) != expected_levels
        or len(boundary_certificates) != expected_levels
        or not isinstance(mirrored, list)
        or len(mirrored) != expected_levels
    ):
        raise ValidationError(f"{context}: incomplete ranking certificate profile")

    validated = []
    for index, (level, boundary) in enumerate(zip(levels, boundary_certificates)):
        ranking = level.get("ranking_certificate")
        if not isinstance(ranking, dict) or mirrored[index] != ranking:
            raise ValidationError(
                f"{context}/level-{index}: missing or inconsistent ranking comparison"
            )
        if ranking.get("reference") != "identity atom partition":
            raise ValidationError(
                f"{context}/level-{index}: unregistered ranking reference"
            )
        if ranking.get("candidate_family") != "same scope and quotient atoms":
            raise ValidationError(
                f"{context}/level-{index}: ranking candidate family changed"
            )
        finite = ranking.get("finite")
        if not isinstance(finite, bool):
            raise ValidationError(
                f"{context}/level-{index}: missing ranking finite flag"
            )
        if not finite:
            if boundary.get("finite") is not False or any(
                ranking.get(field) is not None
                for field in RANKING_CERTIFICATE_NUMERIC_FIELDS
            ):
                raise ValidationError(
                    f"{context}/level-{index}: invalid undefined ranking comparison"
                )
            if (
                ranking.get("local_gap_sign") != 0
                or ranking.get("full_gap_sign") != 0
                or ranking.get("strict_threshold_passed") is not False
                or ranking.get("ranking_sign_certified") is not False
                or ranking.get("status") != "undefined_zero_weight_scope"
            ):
                raise ValidationError(
                    f"{context}/level-{index}: inconsistent undefined ranking flags"
                )
            validated.append(dict(ranking))
            continue

        if boundary.get("finite") is not True or any(
            not isinstance(ranking.get(field), (int, float))
            or isinstance(ranking.get(field), bool)
            or not math.isfinite(float(ranking[field]))
            for field in RANKING_CERTIFICATE_NUMERIC_FIELDS
        ):
            raise ValidationError(
                f"{context}/level-{index}: invalid finite ranking scalar"
            )
        values = {
            field: float(ranking[field])
            for field in RANKING_CERTIFICATE_NUMERIC_FIELDS
        }
        local_gap = values["scaled_local_gap"]
        full_gap = values["restricted_full_objective_gap"]
        width = values["certificate_width"]
        output_d = values["output_D"]
        reference_d = values["reference_D"]
        output_q_u = values["output_Q_U"]
        reference_q_u = values["reference_Q_U"]
        local_scale = values["W_U_over_W"]
        tolerance = values["comparison_tolerance"]
        if width < 0.0 or tolerance <= 0.0 or not 0.0 < local_scale <= 1.0:
            raise ValidationError(
                f"{context}/level-{index}: invalid ranking width or tolerance"
            )
        expected_tolerance = 1e-10 * max(
            1.0, abs(local_gap), abs(full_gap), abs(width)
        )
        if not math.isclose(
            tolerance, expected_tolerance, rel_tol=1e-12, abs_tol=1e-15
        ):
            raise ValidationError(
                f"{context}/level-{index}: ranking tolerance is inconsistent"
            )
        if not math.isclose(
            width,
            float(boundary["certificate_width"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ) or not math.isclose(
            output_d,
            float(boundary["D"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValidationError(
                f"{context}/level-{index}: ranking/output certificate mismatch"
            )
        if not math.isclose(
            output_q_u,
            float(boundary["Q_U"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ) or not math.isclose(
            local_scale,
            float(boundary["W_U"]) / float(boundary["W"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ) or not math.isclose(
            local_gap,
            local_scale * (output_q_u - reference_q_u),
            rel_tol=1e-11,
            abs_tol=1e-12,
        ):
            raise ValidationError(
                f"{context}/level-{index}: scaled local gap is inconsistent"
            )
        if not math.isclose(
            full_gap,
            local_gap + output_d - reference_d,
            rel_tol=1e-11,
            abs_tol=1e-12,
        ):
            raise ValidationError(
                f"{context}/level-{index}: restricted objective gap is inconsistent"
            )
        b_plus = float(boundary["B_plus"])
        b_minus = float(boundary["B_minus"])
        bound_tolerance = 1e-10 * max(1.0, b_plus, b_minus, abs(reference_d))
        if (
            reference_d < -b_minus - bound_tolerance
            or reference_d > b_plus + bound_tolerance
            or abs(output_d - reference_d) > width + bound_tolerance
        ):
            raise ValidationError(
                f"{context}/level-{index}: reference mismatch exceeds its bounds"
            )

        def signed(value: float) -> int:
            if abs(value) <= tolerance:
                return 0
            return 1 if value > 0.0 else -1

        local_sign = signed(local_gap)
        full_sign = signed(full_gap)
        strict_threshold = abs(local_gap) > width
        informative = (
            abs(local_gap) - width > tolerance and local_sign != 0
        )
        same_sign = local_sign != 0 and local_sign == full_sign
        certified = informative and same_sign
        expected_status = (
            "local_tie"
            if local_sign == 0
            else "within_certificate_width"
            if not informative
            else "certified_same_sign"
            if certified
            else "sign_mismatch"
        )
        if (
            ranking.get("local_gap_sign") != local_sign
            or ranking.get("full_gap_sign") != full_sign
            or ranking.get("strict_threshold_passed") is not strict_threshold
            or ranking.get("ranking_sign_certified") is not certified
            or ranking.get("status") != expected_status
        ):
            raise ValidationError(
                f"{context}/level-{index}: ranking sign or threshold flag is wrong"
            )
        if informative and not same_sign:
            raise ValidationError(
                f"{context}/level-{index}: ranking theorem sign check failed"
            )
        validated.append(dict(ranking))
    return validated


def summarize_identity_ranking_certificates(
    rankings: list[dict[str, Any]],
) -> dict[str, float | int]:
    """Aggregate descriptive ranking coverage without inferential intervals."""

    if not rankings:
        raise ValidationError("cannot summarize an empty ranking collection")
    finite = [item for item in rankings if item["finite"] is True]
    informative = [
        item for item in finite if item["ranking_sign_certified"] is True
    ]
    return {
        "total_comparisons": len(rankings),
        "finite_comparisons": len(finite),
        "comparison_coverage": len(finite) / len(rankings),
        "identity_tie_count": sum(
            item["status"] == "local_tie" for item in finite
        ),
        "within_width_count": sum(
            item["status"] == "within_certificate_width" for item in finite
        ),
        "informative_count": len(informative),
        "informative_rate_denominator": len(finite),
        "informative_rate": len(informative) / len(finite) if finite else 0.0,
        "certified_positive_count": sum(
            item["local_gap_sign"] == 1 for item in informative
        ),
        "certified_negative_count": sum(
            item["local_gap_sign"] == -1 for item in informative
        ),
    }


def validate_stage3(campaign_dir: Path) -> dict[str, Any]:
    parity_attempt = completed_attempt(
        campaign_dir, "stage3_mechanism", "production_profile_parity"
    )
    parity_profiles = completed_profiles(parity_attempt / "parity.json", 3)
    expected_parity = {
        (dataset, "full")
        for dataset in ("dyn_cora", "dyn_pubmed", "arxivmath")
    }
    observed_parity = {
        (profile.get("base_dataset"), profile.get("variant"))
        for profile in parity_profiles
    }
    if observed_parity != expected_parity or len(observed_parity) != len(
        parity_profiles
    ):
        raise ValidationError("production/profile parity grid is incomplete")
    parity_bootstrap = {
        dataset: validate_bootstrap_cache_entry(
            campaign_dir,
            parity_attempt,
            dataset=dataset,
            initial_batch=999,
            depth=3,
        )
        for dataset in ("dyn_cora", "dyn_pubmed", "arxivmath")
    }
    stage1 = read_json(
        campaign_dir / "stages" / "stage1_correctness_smoke" / "validation.json"
    )
    parity = {}
    for profile in parity_profiles:
        dataset = profile["base_dataset"]
        if (
            profile.get("batch_strategy") != "999:10"
            or profile.get("smart_depth") != 3
            or profile.get("smart_radius") != 1
            or profile.get("aggregation_mode") != "sum"
            or profile.get("use_gpu") is not True
            or not str(profile.get("device", "")).startswith("cuda")
        ):
            raise ValidationError(f"{dataset}: parity profile protocol mismatch")
        production = stage1["launcher"][dataset]
        profile_q = float(profile["metrics"]["Final modularity"])
        profile_nmi = float(profile["metrics"]["NMI"])
        if not math.isfinite(profile_q) or not math.isfinite(profile_nmi):
            raise ValidationError(f"{dataset}: non-finite parity metric")
        delta_q = profile_q - float(production["final_modularity"])
        delta_nmi = profile_nmi - float(production["final_nmi"])
        if abs(delta_q) > 1e-6 or abs(delta_nmi) > 1e-6:
            raise ValidationError(
                f"profile/production parity failed for {dataset}: "
                f"delta_q={delta_q}, delta_nmi={delta_nmi}"
            )
        parity_certificates = []
        parity_rankings = []
        for row_index, row in enumerate(profile.get("rows", [])):
            row_context = f"{dataset}/parity/update-{row_index}"
            row_certificates = validate_boundary_certificates(
                row,
                context=row_context,
                expected_levels=3,
            )
            parity_certificates.extend(row_certificates)
            parity_rankings.extend(
                validate_identity_ranking_certificates(
                    row,
                    row_certificates,
                    context=row_context,
                    expected_levels=3,
                )
            )
        parity[dataset] = {
            "delta_q": delta_q,
            "delta_nmi": delta_nmi,
            "boundary_certificate": summarize_boundary_certificates(
                parity_certificates
            ),
            "identity_ranking_certificate": (
                summarize_identity_ranking_certificates(parity_rankings)
            ),
        }

    mechanism_attempt = completed_attempt(
        campaign_dir, "stage3_mechanism", "closure_contraction_profiles"
    )
    profiles = completed_profiles(mechanism_attempt / "mechanism.json", 9)
    observed = {(row["base_dataset"], row["variant"]) for row in profiles}
    expected = {
        (dataset, variant)
        for dataset in ("dyn_cora", "dyn_pubmed", "arxivmath")
        for variant in ("full", "no_closure", "no_contraction")
    }
    if observed != expected:
        raise ValidationError("mechanism profile dataset/variant grid is incomplete")
    mechanism = {}
    variant_options = {
        "full": (True, "hierarchical", "repaired full method"),
        "no_closure": (False, "hierarchical", "radius-only scope"),
        "no_contraction": (True, "singleton", "vertex-level base atoms"),
    }
    for profile in profiles:
        dataset = profile["base_dataset"]
        variant = profile["variant"]
        closure_enabled, base_atom_policy, mechanism_label = variant_options[variant]
        expected_state_policy = (
            "isolated_one_step_from_production_pre_update_state"
            if variant == "no_closure"
            else "persistent_variant_trajectory"
        )
        if (
            profile.get("batch_strategy") != "999:50"
            or profile.get("smart_depth") != 3
            or profile.get("smart_radius") != 1
            or profile.get("aggregation_mode") != "sum"
            or profile.get("use_gpu") is not True
            or not str(profile.get("device", "")).startswith("cuda")
            or profile.get("state_policy") != expected_state_policy
        ):
            raise ValidationError(f"{dataset}/{variant}: mechanism protocol mismatch")
        expected_updates = EXPECTED_UPDATES["999:50"][dataset]
        if profile.get("updates_profiled") != expected_updates:
            raise ValidationError(f"{dataset}/{profile['variant']}: update count mismatch")
        if variant == "no_closure" and profile.get("metrics", {}).get(
            "interpretation"
        ) != (
            "final one-step radius-only counterfactual from the production "
            "pre-update state; not a persistent multibatch trajectory"
        ):
            raise ValidationError(
                f"{dataset}: radius-only final metric was presented as a trajectory"
            )
        contraction_ratios = []
        closure_ratios = []
        contraction_ratios_by_level = [[] for _ in range(3)]
        closure_ratios_by_level = [[] for _ in range(3)]
        boundary_certificates = []
        identity_rankings = []
        isolated_numeric_outside_relabel_updates = 0
        for row_index, row in enumerate(profile["rows"]):
            if (
                row.get("closure_enabled") is not closure_enabled
                or row.get("base_atom_policy") != base_atom_policy
                or row.get("mechanism_label") != mechanism_label
                or float(row.get("cut_time", math.nan)) != 0.0
            ):
                raise ValidationError(
                    f"{dataset}/{variant}: production ablation metadata mismatch"
                )
            if row.get("state_policy") != expected_state_policy:
                raise ValidationError(f"{dataset}/{variant}: state policy mismatch")
            radius = max(1, int(row["radius_vertices"]))
            closure = row.get("closure_vertices_by_level", [])
            contracted = row.get("contracted_nodes_by_level", [])
            if len(closure) != 3 or len(contracted) != 3:
                raise ValidationError(f"{dataset}/{variant}: incomplete level profile")
            row_context = f"{dataset}/{variant}/update-{row_index}"
            row_certificates = validate_boundary_certificates(
                row,
                context=row_context,
                expected_levels=3,
            )
            boundary_certificates.extend(row_certificates)
            identity_rankings.extend(
                validate_identity_ranking_certificates(
                    row,
                    row_certificates,
                    context=row_context,
                    expected_levels=3,
                )
            )
            if variant == "no_closure" and any(
                int(value) != int(row["radius_vertices"]) for value in closure
            ):
                raise ValidationError(f"{dataset}: radius-only scope was not respected")
            if variant == "no_closure":
                isolated = row.get("isolated_invariants", {})
                collisions = isolated.get("scope_label_collisions_by_level", [])
                outside_preserved = isolated.get(
                    "outside_partition_preserved_by_level", []
                )
                outside_numeric_writes = isolated.get(
                    "outside_numeric_writes_by_level", []
                )
                if (
                    row.get("persistent_state_unchanged_by_control") is not True
                    or isolated.get("nested_before") is not True
                    or isolated.get("nested_after") is not True
                    or row.get("production_post_advance_nested") is not True
                    or len(collisions) != 3
                    or any(int(value) != 0 for value in collisions)
                    or len(outside_preserved) != 3
                    or not all(value is True for value in outside_preserved)
                    or len(outside_numeric_writes) != 3
                    or any(
                        not isinstance(value, int) or value < 0
                        for value in outside_numeric_writes
                    )
                ):
                    raise ValidationError(
                        f"{dataset}: radius-only counterfactual violated its "
                        "partition/nesting isolation gate"
                    )
                if any(outside_numeric_writes):
                    isolated_numeric_outside_relabel_updates += 1
            if variant == "no_contraction" and int(contracted[0]) != int(closure[0]):
                raise ValidationError(
                    f"{dataset}: vertex-level base atoms were not respected"
                )
            numeric_times = (
                "radius_time",
                "closure_time",
                "prepare_time",
                "reset_time",
                "aggregation_time",
                "backend_time",
                "projection_time",
                "certificate_time",
                "instrumented_wall_time",
                "optimizer_time",
                "total_profiled_time",
                "principal_profiled_time",
            )
            if any(
                not isinstance(row.get(field), (int, float))
                or not math.isfinite(float(row[field]))
                or float(row[field]) < 0
                for field in numeric_times
            ):
                raise ValidationError(f"{dataset}/{variant}: invalid profile timing")
            backend_time = float(row["backend_time"])
            conversion_time = float(row.get("backend_conversion_time", math.nan))
            certificate_time = float(row["certificate_time"])
            optimizer_time = float(row["optimizer_time"])
            instrumented_wall_time = float(row["instrumented_wall_time"])
            if (
                not math.isfinite(conversion_time)
                or conversion_time < 0
                or conversion_time > backend_time + 1e-6
                or row.get("timing_accounting")
                != (
                    "certificate_time is excluded once from total_profiled_time; "
                    "backend_conversion_time is a diagnostic subcomponent of "
                    "backend_time, included once in total_profiled_time, and "
                    "subtracted once only for principal_profiled_time"
                )
                or not math.isclose(
                    instrumented_wall_time,
                    optimizer_time + certificate_time,
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                )
                or not math.isclose(
                    float(row["total_profiled_time"]),
                    float(row["radius_time"]) + optimizer_time,
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                )
                or not math.isclose(
                    float(row["principal_profiled_time"]),
                    max(0.0, float(row["total_profiled_time"]) - conversion_time),
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            ):
                raise ValidationError(
                    f"{dataset}/{variant}: backend conversion was double-counted"
                )
            closure_ratios.extend(float(value) / radius for value in closure)
            contraction_ratios.extend(
                float(contracted_value) / max(1, int(closure_value))
                for contracted_value, closure_value in zip(contracted, closure)
            )
            for level, (contracted_value, closure_value) in enumerate(
                zip(contracted, closure)
            ):
                closure_ratios_by_level[level].append(
                    float(closure_value) / radius
                )
                contraction_ratios_by_level[level].append(
                    float(contracted_value) / max(1, int(closure_value))
                )
        mechanism[f"{dataset}/{variant}"] = {
            "updates": expected_updates,
            "mechanism_label": mechanism_label,
            "state_policy": expected_state_policy,
            "mean_closure_to_radius": (
                sum(closure_ratios) / len(closure_ratios) if closure_ratios else None
            ),
            "mean_contracted_to_closure": (
                sum(contraction_ratios) / len(contraction_ratios)
                if contraction_ratios
                else None
            ),
            "mean_closure_to_radius_by_level": [
                sum(values) / len(values) for values in closure_ratios_by_level
            ],
            "mean_backend_atoms_to_scope_by_level": [
                sum(values) / len(values)
                for values in contraction_ratios_by_level
            ],
            "total_profiled_time": profile["total_profiled_time"],
            "final_modularity": profile["metrics"]["Final modularity"],
            "boundary_certificate": summarize_boundary_certificates(
                boundary_certificates
            ),
            "identity_ranking_certificate": (
                summarize_identity_ranking_certificates(identity_rankings)
            ),
            "isolated_numeric_outside_relabel_updates": (
                isolated_numeric_outside_relabel_updates
                if variant == "no_closure"
                else None
            ),
            "isolated_partition_and_nesting_gate": "passed"
            if variant == "no_closure"
            else None,
        }
    return {
        "status": "validated",
        "profile_production_parity": parity,
        "bootstrap_sha256": parity_bootstrap,
        "mechanism": mechanism,
    }


def validate_stage4_core(campaign_dir: Path) -> dict[str, Any]:
    attempt = completed_attempt(
        campaign_dir, "stage4_long_core", "fresh_paired_long"
    )
    rows = validate_launcher(
        launcher_output(attempt, "stage4_paired_long_9_500"),
        datasets=CORE_DATASETS,
        batch="9:500",
        required_algorithms={FULL_LEIDEN, SMART_LEIDEN},
    )
    bootstrap = {
        dataset: validate_paired_bootstrap(
            campaign_dir,
            attempt,
            dataset=dataset,
            initial_batch=9,
        )
        for dataset in CORE_DATASETS
    }
    paired = {}
    for dataset in CORE_DATASETS:
        full = rows[(FULL_LEIDEN, dataset)]
        smart = rows[(SMART_LEIDEN, dataset)]
        paired[dataset] = {
            "full": full,
            "smart": smart,
            "speedup": float(full["total_time"]) / float(smart["total_time"]),
            "delta_q": float(smart["final_modularity"])
            - float(full["final_modularity"]),
            "delta_nmi": float(smart["final_nmi"]) - float(full["final_nmi"]),
        }
    return {
        "status": "validated",
        "fresh_paired_run": True,
        "bootstrap_sha256": bootstrap,
        "paired": paired,
    }


def command_names(campaign_dir: Path, stage_id: str, prefix: str) -> list[str]:
    root = campaign_dir / "stages" / stage_id
    return sorted(path.name for path in root.glob(f"{prefix}*") if path.is_dir())


def completed_repeatability_names(campaign_dir: Path) -> list[str]:
    """Return completed repeats while admitting only audited boundary remnants."""

    stage_id = "stage4_long_repeatability"
    names = command_names(campaign_dir, stage_id, "smart_long_repeat_")
    if not 1 <= len(names) <= 2:
        raise ValidationError("long repeatability must contain one or two run slots")
    completed = []
    for name in names:
        metadata_paths = sorted(
            (campaign_dir / "stages" / stage_id / name).glob(
                "attempt-*/metadata.json"
            )
        )
        if not metadata_paths:
            raise ValidationError(f"{stage_id}/{name}: attempt metadata is missing")
        payloads = [read_json(path) for path in metadata_paths]
        completed_count = sum(
            payload.get("status") == "completed" for payload in payloads
        )
        if completed_count > 1:
            raise ValidationError(f"{stage_id}/{name}: multiple completed attempts")
        if completed_count == 1:
            completed.append(name)
            continue
        if any(
            payload.get("status") != "failed"
            or payload.get("failure_kind")
            not in {"campaign_time_boundary", "launcher_interrupted"}
            for payload in payloads
        ):
            raise ValidationError(
                f"{stage_id}/{name}: incomplete run was not an audited stop"
            )
    if not completed:
        raise ValidationError("long repeatability has no completed run")
    return completed


def validate_stage4_repeats(campaign_dir: Path) -> dict[str, Any]:
    names = completed_repeatability_names(campaign_dir)
    rows = {dataset: [] for dataset in CORE_DATASETS}
    for name in names:
        attempt = completed_attempt(campaign_dir, "stage4_long_repeatability", name)
        current = validate_launcher(
            launcher_output(attempt, "stage4_smart_long_9_500"),
            datasets=CORE_DATASETS,
            batch="9:500",
            required_algorithms={SMART_LEIDEN},
        )
        for dataset in CORE_DATASETS:
            rows[dataset].append(current[(SMART_LEIDEN, dataset)])
    return {
        "status": "validated",
        "additional_repetitions": len(names),
        "smart": {dataset: aggregate(values) for dataset, values in rows.items()},
    }


def validate_stage5(campaign_dir: Path) -> dict[str, Any]:
    grid_attempt = completed_attempt(
        campaign_dir, "stage5_topology_controls", "topology_grid"
    )
    grid_algorithms = {FULL_LEIDEN} | {
        f"leidenalg-L:{depth}-r:{radius}-gpu"
        for depth in (1, 2, 3, 4)
        for radius in (0, 1, 2)
    }
    grid = validate_launcher(
        launcher_output(grid_attempt, "stage5_topology_grid_999_10"),
        datasets=("dyn_cora", "dyn_pubmed", "arxivmath"),
        batch="999:10",
        required_algorithms=grid_algorithms,
    )
    directed_attempt = completed_attempt(
        campaign_dir, "stage5_topology_controls", "directed_control"
    )
    directed = validate_launcher(
        launcher_output(directed_attempt, "stage5_directed_control_999_10"),
        datasets=CORE_DATASETS,
        batch="999:10",
        required_algorithms={FULL_LEIDEN, SMART_LEIDEN},
    )
    cut_attempt = completed_attempt(
        campaign_dir, "stage5_topology_controls", "cut_metrics"
    )
    cut_payload = read_json(cut_attempt / "cut_metrics.json")
    cut_records = cut_payload.get("records")
    if (
        cut_payload.get("record_count") != 4
        or not isinstance(cut_records, list)
        or len(cut_records) != 4
    ):
        raise ValidationError("cut-metric control must contain four paired records")
    expected_cut_cells = {
        (dataset, mode) for dataset in CORE_DATASETS for mode in ("naive", "smart")
    }
    observed_cut_cells = {
        (row.get("dataset"), row.get("mode")) for row in cut_records
    }
    if observed_cut_cells != expected_cut_cells:
        raise ValidationError("cut-metric control is not the registered paired grid")
    for row in cut_records:
        dataset = row["dataset"]
        mode = row["mode"]
        if (
            row.get("batch") != "999:10"
            or row.get("method") != "leidenalg"
            or row.get("force_undirected") is not True
            or float(row.get("resolution", math.nan)) != 1.0
            or row.get("updates") != EXPECTED_UPDATES["999:10"][dataset]
            or (mode == "smart" and (row.get("smart_depth"), row.get("smart_radius")) != (3, 1))
            or (mode == "naive" and (row.get("smart_depth") is not None or row.get("smart_radius") is not None))
        ):
            raise ValidationError(f"cut metric protocol mismatch: {dataset}/{mode}")
        for field in ("total_time", "wall_time"):
            value = row.get(field)
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0
            ):
                raise ValidationError(f"{dataset}/{mode}: invalid {field}")
        modularity = row.get("final_modularity")
        if (
            not isinstance(modularity, (int, float))
            or not math.isfinite(float(modularity))
            or not -1 <= float(modularity) <= 1
        ):
            raise ValidationError(f"{dataset}/{mode}: invalid cut modularity")
        metrics = row.get("cut_metrics", {})
        for field in (
            "ncut",
            "conductance_mean",
            "conductance_median",
            "conductance_max",
            "conductance_volume_weighted",
        ):
            value = metrics.get(field)
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0
            ):
                raise ValidationError(f"{dataset}/{mode}: invalid cut metric {field}")
    resolution_attempt = completed_attempt(
        campaign_dir, "stage5_topology_controls", "resolution_control"
    )
    resolution_algorithms = {
        "leidenalg-res:0.5-naive",
        "leidenalg-res:0.5-L:3-r:1-gpu",
        FULL_LEIDEN,
        SMART_LEIDEN,
        "leidenalg-res:2-naive",
        "leidenalg-res:2-L:3-r:1-gpu",
    }
    resolution = validate_launcher(
        launcher_output(resolution_attempt, "stage5_resolution_control_999_50"),
        datasets=CORE_DATASETS,
        batch="999:50",
        required_algorithms=resolution_algorithms,
    )
    if len(grid) != 39 or len(directed) != 4 or len(resolution) != 12:
        raise ValidationError("topology-control Cartesian grid cardinality mismatch")
    return {
        "status": "validated",
        "grid_rows": len(grid),
        "directed_rows": len(directed),
        "cut_metric_rows": 4,
        "resolution_rows": len(resolution),
    }


def validate_stage6(campaign_dir: Path) -> dict[str, Any]:
    campaign_manifest = read_json(campaign_dir / "manifest.json")
    registration = campaign_manifest.get("dsbm_input_manifest", {})
    input_path = campaign_dir / registration.get("filename", "")
    if (
        not input_path.is_file()
        or registration.get("sha256") != sha256_file(input_path)
    ):
        raise ValidationError("Stage 6 lacks a sealed DSBM input manifest")
    input_manifest = read_json(input_path)
    validate_input_manifest(input_manifest, verify_content=True)
    sealed_datasets = {
        use
        for record in input_manifest["files"]
        if Path(record["path"]).name.startswith("out.")
        for use in record["uses"]
    }
    seed_order = (42, 43, 44, 45, 46)
    conditions = tuple(
        (regime, max_changes)
        for max_changes in (290, 1450)
        for regime in ("random", "hubs", "community")
    )
    completed_seeds: list[int] = []
    paired_cells: list[dict[str, Any]] = []
    partial_seed: int | None = None
    saw_seed_gap = False
    for seed in seed_order:
        completed_conditions = 0
        for regime, max_changes in conditions:
            command_name = (
                f"dsbm_seed_{seed}_{regime}_mc{max_changes}_paired"
            )
            command_dir = campaign_dir / "stages" / "stage6_dsbm" / command_name
            metadata_paths = sorted(command_dir.glob("attempt-*/metadata.json"))
            completed_paths = [
                path
                for path in metadata_paths
                if read_json(path).get("status") == "completed"
            ]
            if len(completed_paths) > 1:
                raise ValidationError(f"{command_name}: multiple completed attempts")
            if not completed_paths:
                continue
            if saw_seed_gap:
                raise ValidationError(
                    "DSBM conditions were completed after a gap in the fixed seed order"
                )
            completed_conditions += 1
            attempt = completed_attempt(campaign_dir, "stage6_dsbm", command_name)
            stem = (
                f"repaired_dsbm_seed_{seed}_{regime}_"
                f"mc{max_changes}_paired"
            )
            run_manifest = read_json(attempt / f"manifest_{stem}.json")
            if (
                run_manifest.get("status") != "completed"
                or run_manifest.get("selected_streams") != 1
                or run_manifest.get("attempted_runs") != 2
                or run_manifest.get("successful_runs") != 2
                or run_manifest.get("errors") != 0
                or run_manifest.get("seeds") != [seed]
                or run_manifest.get("regimes") != [regime]
                or run_manifest.get("max_changes") != [max_changes]
                or run_manifest.get("modes") != ["naive", "smart"]
            ):
                raise ValidationError(
                    f"DSBM condition {seed}/{regime}/{max_changes} is incomplete"
                )
            payload = read_json(attempt / f"{stem}.json")
            if set(payload) != {FULL_LEIDEN, SMART_LEIDEN}:
                raise ValidationError(
                    f"DSBM condition {seed}/{regime}/{max_changes} is not paired"
                )
            algorithm_datasets = {
                algorithm: set(payload[algorithm])
                for algorithm in (FULL_LEIDEN, SMART_LEIDEN)
            }
            if algorithm_datasets[FULL_LEIDEN] != algorithm_datasets[SMART_LEIDEN]:
                raise ValidationError(f"{command_name}: full/smart datasets differ")
            if len(algorithm_datasets[FULL_LEIDEN]) != 1:
                raise ValidationError(f"{command_name}: expected exactly one dataset")
            dataset = next(iter(algorithm_datasets[FULL_LEIDEN]))
            match = DSBM_RE.match(dataset)
            if match is None or (
                match.group(1), int(match.group(2)), int(match.group(3))
            ) != (regime, max_changes, seed):
                raise ValidationError(f"{command_name}: dataset identity mismatch")
            if dataset not in sealed_datasets:
                raise ValidationError(f"{dataset}: dataset is absent from the input seal")

            series_by_algorithm: dict[str, list[dict[str, Any]]] = {}
            for algorithm in (FULL_LEIDEN, SMART_LEIDEN):
                machine_map = payload[algorithm][dataset]
                if len(machine_map) != 1:
                    raise ValidationError(f"{dataset}/{algorithm}: expected one machine")
                batch_map = next(iter(machine_map.values()))
                if set(batch_map) != {"0:99"}:
                    raise ValidationError(
                        f"{dataset}/{algorithm}: unexpected DSBM batch keys"
                    )
                series = batch_map["0:99"]
                if not isinstance(series, list) or len(series) != 99:
                    raise ValidationError(
                        f"{dataset}/{algorithm}: expected 99 measured updates"
                    )
                for index, row in enumerate(series, start=1):
                    for field in ("time", "modularity"):
                        value = row.get(field)
                        if not isinstance(value, (int, float)) or not math.isfinite(
                            float(value)
                        ):
                            raise ValidationError(
                                f"{dataset}/{algorithm}/update-{index}: invalid {field}"
                            )
                    if float(row["time"]) < 0 or not -1 <= float(
                        row["modularity"]
                    ) <= 1:
                        raise ValidationError(
                            f"{dataset}/{algorithm}/update-{index}: metric out of range"
                        )
                final = series[-1]
                for field in ("Final modularity", "NMI"):
                    value = final.get(field)
                    if not isinstance(value, (int, float)) or not math.isfinite(
                        float(value)
                    ):
                        raise ValidationError(
                            f"{dataset}/{algorithm}: invalid {field}"
                        )
                if not -1 <= float(final["Final modularity"]) <= 1 or not 0 <= float(
                    final["NMI"]
                ) <= 1:
                    raise ValidationError(
                        f"{dataset}/{algorithm}: final metric out of range"
                    )
                series_by_algorithm[algorithm] = series

            bootstrap = validate_paired_bootstrap(
                campaign_dir,
                attempt,
                dataset=dataset,
                initial_batch=0,
            )
            full_series = series_by_algorithm[FULL_LEIDEN]
            smart_series = series_by_algorithm[SMART_LEIDEN]
            full_time = sum(float(row["time"]) for row in full_series)
            smart_time = sum(float(row["time"]) for row in smart_series)
            if full_time <= 0 or smart_time <= 0:
                raise ValidationError(f"{dataset}: paired cumulative time must be positive")
            paired_cells.append(
                {
                    "seed": seed,
                    "update_type": regime,
                    "changed_edges_per_update": max_changes,
                    "dataset": dataset,
                    "full_total_time": full_time,
                    "smart_total_time": smart_time,
                    "speedup": full_time / smart_time,
                    "delta_q": float(smart_series[-1]["Final modularity"])
                    - float(full_series[-1]["Final modularity"]),
                    "delta_nmi": float(smart_series[-1]["NMI"])
                    - float(full_series[-1]["NMI"]),
                    "bootstrap_sha256": bootstrap,
                }
            )
        if completed_conditions == len(conditions):
            completed_seeds.append(seed)
        elif completed_conditions:
            if partial_seed is not None:
                raise ValidationError("more than one DSBM seed is partially complete")
            partial_seed = seed
            saw_seed_gap = True
        else:
            saw_seed_gap = True
    if not completed_seeds:
        raise ValidationError("DSBM stage has no completed seed block")
    expected_prefix = list(seed_order[: len(completed_seeds)])
    if completed_seeds != expected_prefix:
        raise ValidationError(
            "DSBM seeds must be accumulated without outcome-based selection in "
            f"the fixed order {list(seed_order)}"
        )
    publishable_cells = [
        cell for cell in paired_cells if cell["seed"] in completed_seeds
    ]
    if len(publishable_cells) != 6 * len(completed_seeds):
        raise ValidationError("complete DSBM seeds do not contribute exactly six pairs")
    partial_cell_count = len(paired_cells) - len(publishable_cells)
    return {
        "status": (
            "validated"
            if len(completed_seeds) == 5
            else "primary_validated"
            if len(completed_seeds) >= 3
            else "valid_partial"
        ),
        "completed_seeds": completed_seeds,
        "completed_seed_count": len(completed_seeds),
        "primary_analysis_seeds": [42, 43, 44],
        "primary_design_complete": len(completed_seeds) >= 3,
        "minimum_publishable_seed_count": 3,
        "minimum_publishable_seed_count_met": len(completed_seeds) >= 3,
        "precision_extension_seeds": [45, 46],
        "precision_target_seed_count": 5,
        "precision_extension_complete": len(completed_seeds) == 5,
        "paired_condition_count": len(publishable_cells),
        "fresh_full_runs": len(publishable_cells),
        "fresh_smart_runs": len(publishable_cells),
        "paired_cells": publishable_cells,
        "partial_seed_progress": (
            {
                "seed": partial_seed,
                "completed_condition_pairs": partial_cell_count,
                "eligible_for_analysis": False,
            }
            if partial_seed is not None
            else None
        ),
        "historical_total_algorithm_hours_registered": 63.73,
        "historical_total_wall_hours_registered": 76.06,
    }


def validate_stage7(
    campaign_dir: Path,
    *,
    stage_id: str,
    prefix: str,
    config_stem: str,
    expected_algorithm: str,
    minimum: int,
    maximum: int,
) -> dict[str, Any]:
    names = command_names(campaign_dir, stage_id, prefix)
    if not minimum <= len(names) <= maximum:
        raise ValidationError(
            f"{stage_id}: expected {minimum}..{maximum} repetitions, got {len(names)}"
        )
    grouped = {dataset: [] for dataset in CORE_DATASETS}
    for name in names:
        attempt = completed_attempt(campaign_dir, stage_id, name)
        rows = validate_launcher(
            launcher_output(attempt, config_stem),
            datasets=CORE_DATASETS,
            batch="999:10",
            required_algorithms={expected_algorithm},
        )
        for dataset in CORE_DATASETS:
            grouped[dataset].append(rows[(expected_algorithm, dataset)])
    return {
        "status": "validated",
        "repetitions": len(names),
        "rows": {dataset: aggregate(rows) for dataset, rows in grouped.items()},
    }


def validate_stage7_dfleiden(campaign_dir: Path) -> dict[str, Any]:
    short = validate_stage7(
        campaign_dir,
        stage_id="stage7_dfleiden_interface",
        prefix="dfleiden_smart_repeat_",
        config_stem="stage7_dfleiden_smart_999_10",
        expected_algorithm="dfleiden-L:3-r:1-gpu",
        minimum=5,
        maximum=5,
    )
    attempt = completed_attempt(
        campaign_dir,
        "stage7_dfleiden_interface",
        "dfleiden_smart_long_coverage",
    )
    long_rows = validate_launcher(
        launcher_output(attempt, "stage7_dfleiden_smart_9_500"),
        datasets=CORE_DATASETS,
        batch="9:500",
        required_algorithms={"dfleiden-L:3-r:1-gpu"},
    )
    short["long_horizon_smart_coverage"] = {
        dataset: long_rows[("dfleiden-L:3-r:1-gpu", dataset)]
        for dataset in CORE_DATASETS
    }
    short["native_full_row_reuse"] = (
        "admissible only under the registered exact protocol/hash policy"
    )
    return short


def validate_stage7_s2cag(campaign_dir: Path) -> dict[str, Any]:
    short = validate_stage7(
        campaign_dir,
        stage_id="stage7_s2cag_interface",
        prefix="s2cag_smart_repeat_",
        config_stem="stage7_s2cag_smart_999_10",
        expected_algorithm="s2cag-i:10-L:3-r:1-agg:norm-gpu-feat:dataset",
        minimum=1,
        maximum=5,
    )
    attempt = completed_attempt(
        campaign_dir,
        "stage7_s2cag_interface",
        "s2cag_feature_modes_coverage",
    )
    feature_algorithms = {
        f"s2cag-i:10-L:3-r:1-agg:norm-gpu-feat:{mode}"
        for mode in ("dataset", "random", "onehot")
    }
    feature_rows = validate_launcher(
        launcher_output(attempt, "stage7_s2cag_feature_modes_999_100"),
        datasets=("dyn_cora", "dyn_pubmed"),
        batch="999:100",
        required_algorithms=feature_algorithms,
    )
    short["feature_mode_smart_coverage"] = {
        f"{algorithm}/{dataset}": row
        for (algorithm, dataset), row in feature_rows.items()
    }
    short["historical_full_row_reuse"] = (
        "admissible only under the registered exact protocol/hash policy"
    )
    return short


def validate_stage(campaign_dir: Path, stage_id: str) -> dict[str, Any]:
    validate_bootstrap_hash_history(campaign_dir)
    if stage_id == "stage1_correctness_smoke":
        return validate_stage1(campaign_dir)
    if stage_id == "stage2_core_short":
        return validate_stage2(campaign_dir)
    if stage_id == "stage3_mechanism":
        return validate_stage3(campaign_dir)
    if stage_id == "stage4_long_core":
        return validate_stage4_core(campaign_dir)
    if stage_id == "stage4_long_repeatability":
        return validate_stage4_repeats(campaign_dir)
    if stage_id == "stage5_topology_controls":
        return validate_stage5(campaign_dir)
    if stage_id == "stage6_dsbm":
        return validate_stage6(campaign_dir)
    if stage_id == "stage7_dfleiden_interface":
        return validate_stage7_dfleiden(campaign_dir)
    if stage_id == "stage7_s2cag_interface":
        return validate_stage7_s2cag(campaign_dir)
    raise ValidationError(f"unknown stage: {stage_id}")


def validate_campaign(campaign_dir: Path) -> dict[str, Any]:
    protocol = load_protocol()
    manifest = read_json(campaign_dir / "manifest.json")
    if manifest.get("schema") != "comnetx-ieee-access-repaired-campaign-v1":
        raise ValidationError("unexpected campaign manifest schema")
    if manifest.get("protocol_id") != protocol["protocol_id"]:
        raise ValidationError("campaign protocol id mismatch")
    current = protocol_fingerprint(protocol)
    recorded = manifest.get("fingerprint", {})
    if recorded.get("protocol_sha256") != current["protocol_sha256"] or recorded.get(
        "config_sha256"
    ) != current["config_sha256"]:
        raise ValidationError("registered protocol or configs changed")
    if recorded.get("source_sha256") != current.get("source_sha256"):
        raise ValidationError(
            "measurement source changed; this campaign cannot be validated"
        )
    hardware_registration = manifest.get("hardware", {})
    hardware_path = campaign_dir / hardware_registration.get("filename", "")
    if (
        not hardware_path.is_file()
        or hardware_registration.get("sha256") != sha256_file(hardware_path)
    ):
        raise ValidationError("hardware metadata is missing or changed")
    environment_registration = manifest.get("environment", {})
    environment_path = campaign_dir / environment_registration.get("filename", "")
    if (
        not environment_path.is_file()
        or environment_registration.get("sha256") != sha256_file(environment_path)
    ):
        raise ValidationError("software-environment metadata is missing or changed")
    real_registration = manifest.get("real_input_manifest", {})
    real_path = campaign_dir / real_registration.get("filename", "")
    if (
        not real_path.is_file()
        or real_registration.get("sha256") != sha256_file(real_path)
    ):
        raise ValidationError("real-input manifest is missing or changed")
    validate_input_manifest(read_json(real_path), verify_content=True)
    launch_budget, windows = validate_launch_budget(campaign_dir, manifest)
    for metadata_path in sorted(campaign_dir.glob("stages/*/*/attempt-*/metadata.json")):
        if metadata_path.parts[-4] == "preflight":
            continue
        validate_attempt_window(read_json(metadata_path), metadata_path, windows)
    dsbm_registration = manifest.get("dsbm_input_manifest")
    if dsbm_registration is not None:
        dsbm_path = campaign_dir / dsbm_registration.get("filename", "")
        if (
            not dsbm_path.is_file()
            or dsbm_registration.get("sha256") != sha256_file(dsbm_path)
        ):
            raise ValidationError("DSBM input manifest is missing or changed")
        validate_input_manifest(read_json(dsbm_path), verify_content=True)
    report = {
        "schema": "comnetx-ieee-access-repaired-validation-v1",
        "campaign_id": manifest["campaign_id"],
        "core_status": "validated",
        "historical_smart_l_ge_2_status": "provisional",
        "measurement_source_sha256": recorded.get("source_sha256"),
        "measurement_source_matches_current": True,
        "launch_budget": launch_budget,
        "preflight_environment": validate_preflight(campaign_dir),
        "stages": {},
    }
    for stage in protocol["stages"]:
        status = manifest.get("stage_status", {}).get(stage["id"])
        if stage["required"] and status != "validated":
            raise ValidationError(f"required stage is not validated: {stage['id']}")
        report["stages"][stage["id"]] = {
            "status": status,
            "required": stage["required"],
            "purpose": stage["purpose"],
        }
        validation_path = campaign_dir / "stages" / stage["id"] / "validation.json"
        if status in {"validated", "partial"}:
            validation = validate_stage(campaign_dir, stage["id"])
            if status == "partial" and validation.get("status") not in {
                "valid_partial",
                "primary_validated",
            }:
                raise ValidationError(
                    f"partial stage does not have partial evidence: {stage['id']}"
                )
            if validation_path.is_file() and read_json(validation_path) != validation:
                raise ValidationError(f"saved validation report changed: {stage['id']}")
            report["stages"][stage["id"]]["validation"] = validation
        elif status == "skipped_for_budget":
            if stage["required"]:
                raise ValidationError("required stage was marked skipped")
            report["stages"][stage["id"]]["budget_skip"] = manifest.get(
                "budget_skips", {}
            ).get(stage["id"])
            boundary_stop = manifest.get("time_boundary_stops", {}).get(stage["id"])
            if boundary_stop is not None:
                report["stages"][stage["id"]][
                    "time_boundary_stop"
                ] = boundary_stop
        elif status == "skipped_by_stage2_no_go":
            if stage["required"]:
                raise ValidationError("required stage was marked skipped")
            report["stages"][stage["id"]]["scientific_no_go_skip"] = manifest.get(
                "scientific_no_go_skips", {}
            ).get(stage["id"])
        elif status == "failed":
            boundary_stop = manifest.get("time_boundary_stops", {}).get(stage["id"])
            interruption = manifest.get("interrupted_stages", {}).get(stage["id"])
            if not isinstance(boundary_stop, dict) and not isinstance(
                interruption, dict
            ):
                raise ValidationError(
                    f"optional stage failed outside a registered resumable boundary: {stage['id']}"
                )
            if isinstance(boundary_stop, dict):
                report["stages"][stage["id"]]["time_boundary_stop"] = boundary_stop
            if isinstance(interruption, dict):
                report["stages"][stage["id"]]["interruption"] = interruption
        elif status == "running":
            raise ValidationError(
                f"stage still has an unresolved running marker: {stage['id']}"
            )
        elif status != "pending":
            raise ValidationError(f"unexpected stage status: {stage['id']}={status!r}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--stage", choices=tuple(stage_map(load_protocol())))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    campaign_dir = args.campaign.expanduser()
    if not campaign_dir.is_absolute() and not campaign_dir.exists():
        campaign_dir = RESULTS_ROOT / campaign_dir
    campaign_dir = campaign_dir.resolve()
    report = (
        validate_stage(campaign_dir, args.stage)
        if args.stage
        else validate_campaign(campaign_dir)
    )
    report_path = args.report or (
        campaign_dir / (f"validation_{args.stage}.json" if args.stage else "validation_report.json")
    )
    report_path = report_path.expanduser().resolve()
    write_json(report_path, report)
    print(f"Validation status: {report.get('status', report.get('core_status'))}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
