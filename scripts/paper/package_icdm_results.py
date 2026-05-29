"""Build a self-contained supplemental measurement bundle for ICDM."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.paper.collect_results_registry import (
    flatten_errors,
    flatten_launch_result,
    flatten_neighborhood,
    flatten_profiles,
    is_error_file,
    is_neighborhood_file,
    is_profile_file,
    load_json,
)


DEFAULT_TAG = "icdm-2026-0"
DEFAULT_TAG_COMMIT = "ca2ea6fd698f78241047143ad6a033dd149dcb03"
DEFAULT_INPUT_DIR = Path("results")


GROUPS: dict[str, str] = {
    "real_graph_measurements": (
        "ICDM paper measurements on real graph streams: topology, GNN, "
        "batch-sweep, robustness, and bad-locality runs."
    ),
    "synthetic_dsbm_measurements": (
        "Synthetic DSBM stress measurements with controlled update streams."
    ),
    "workload_profiles": "Mechanism and workload profiling records.",
    "neighborhood_measurements": (
        "Neighborhood-size measurements used for locality and workload analysis."
    ),
    "legacy_measurements": (
        "Earlier team, archive, imported, and traceability measurements."
    ),
    "unclassified_measurements": "Measurement records outside the named groups.",
}


DROP_MEASUREMENT_KEYS = {"source_file", "stream_key", "duplicate_sources"}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def is_output(path: Path, output_dir: Path) -> bool:
    try:
        path.relative_to(output_dir)
    except ValueError:
        return False
    return True


def is_registry_artifact(path: Path, input_dir: Path) -> bool:
    rel = path.relative_to(input_dir)
    return bool(rel.parts) and rel.parts[0] == "registry"


def is_manifest_payload(path: Path, payload: Any) -> bool:
    if path.name.startswith("manifest"):
        return True
    return isinstance(payload, dict) and any(
        key in payload for key in ["selected_streams", "attempted_runs", "successful_runs", "outputs"]
    )


def dataset_names(record: dict[str, Any]) -> set[str]:
    return {
        str(record.get("dataset", "")),
        str(record.get("base_dataset", "")),
    }


def is_dsbm_record(record: dict[str, Any]) -> bool:
    if "dsbm" in str(record.get("algorithm", "")).lower():
        return True
    return any(name.startswith("dsbm-") or name.startswith("dsbm_") for name in dataset_names(record))


def measurement_group(record: dict[str, Any]) -> str:
    if record.get("measurement_type") == "workload_profile":
        return "workload_profiles"
    if record.get("measurement_type") == "neighborhood":
        return "neighborhood_measurements"
    if is_dsbm_record(record):
        return "synthetic_dsbm_measurements"
    owner = str(record.get("source_owner", ""))
    if owner == "paper_icdm":
        return "real_graph_measurements"
    if owner in {"archive", "drobyshev", "egorov", "konovalov", "imported_logs"}:
        return "legacy_measurements"
    return "unclassified_measurements"


def sanitize_string(value: str) -> str:
    replacements = [
        ("results/", "bundle://source-root/"),
        ("results", "source_root"),
        ("all_results_with_series", "registry_full_series"),
        ("all_results", "registry_compact"),
        ("collect_results_registry", "collect_measurement_registry"),
        ("Results Registry", "Registry Export"),
        ("source_file", "origin_artifact"),
    ]
    out = value
    for old, new in replacements:
        out = out.replace(old, new)
    return out


def sanitize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {sanitize_string(str(key)): sanitize_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_value(item) for item in value]
    if isinstance(value, str):
        return sanitize_string(value)
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return None
    return value


def clean_measurement(record: dict[str, Any], index: int) -> dict[str, Any]:
    cleaned = {
        key: value
        for key, value in record.items()
        if key not in DROP_MEASUREMENT_KEYS
    }
    cleaned["bundle_record_id"] = f"measurement-{index:06d}"
    return sanitize_value(cleaned)


def clean_aux_record(record: Any, index: int, kind: str) -> dict[str, Any]:
    return {
        "bundle_record_id": f"{kind}-{index:05d}",
        "payload": sanitize_value(record),
    }


def scan_input(input_dir: Path, output_dir: Path) -> tuple[list[dict[str, Any]], list[Any], list[Any]]:
    measurements: list[dict[str, Any]] = []
    manifests: list[Any] = []
    raw_errors: list[Any] = []

    for path in sorted(input_dir.rglob("*.json")):
        if is_output(path, output_dir) or is_registry_artifact(path, input_dir):
            continue

        payload = load_json(path)
        if is_error_file(path, payload):
            raw_errors.append(flatten_errors(path, payload, input_dir.parent))
        elif is_profile_file(payload):
            measurements.extend(flatten_profiles(path, payload, input_dir.parent))
        elif is_neighborhood_file(payload):
            measurements.extend(flatten_neighborhood(path, payload, input_dir.parent))
        elif is_manifest_payload(path, payload):
            manifests.append(payload)
        elif isinstance(payload, dict):
            measurements.extend(flatten_launch_result(path, payload, input_dir.parent))

    return measurements, manifests, raw_errors


def load_registry_snapshot(input_dir: Path) -> dict[str, Any]:
    registry_dir = input_dir / "registry"
    names = {
        "compact_records": "all_results.json",
        "full_series_records": "all_results_with_series.json",
        "summary_by_run_key": "summary_by_run_key.json",
        "registry_manifest": "manifest.json",
        "error_records": "errors.json",
        "deduplicated_records": "deduplicated_sources.json",
        "unreadable_records": "unreadable.json",
        "dsbm_stream_summary_10_batches": "dsbm_stream_summary_10_batches.json",
        "dsbm_stream_summary_all_batches": "dsbm_stream_summary_all_batches.json",
    }
    snapshot = {}
    for key, filename in names.items():
        path = registry_dir / filename
        if path.exists():
            snapshot[key] = sanitize_value(load_json(path))
    return snapshot


def build_self_checks(
    grouped: dict[str, list[dict[str, Any]]],
    raw_measurements: list[dict[str, Any]],
) -> dict[str, Any]:
    grouped_records = [record for records in grouped.values() for record in records]
    grouped_total = len(grouped_records)
    type_counts = Counter(str(record.get("measurement_type", "unknown")) for record in raw_measurements)
    grouped_type_counts = Counter(str(record.get("measurement_type", "unknown")) for record in grouped_records)
    record_ids = [record.get("bundle_record_id") for record in grouped_records]
    expected_ids = {f"measurement-{index:06d}" for index in range(len(raw_measurements))}

    checks = {
        "all_scanned_measurements_grouped": grouped_total == len(raw_measurements),
        "bundle_record_ids_are_unique": len(record_ids) == len(set(record_ids)),
        "bundle_record_id_sequence_complete": set(record_ids) == expected_ids,
        "grouped_type_counts_match_raw_scan": grouped_type_counts == type_counts,
        "category_counts_sum_to_total": sum(len(records) for records in grouped.values()) == len(raw_measurements),
    }
    return {
        "checks": checks,
        "raw_measurement_entries_by_type": dict(sorted(type_counts.items())),
        "grouped_measurement_entries_by_type": dict(sorted(grouped_type_counts.items())),
        "measurement_entries_by_category": {
            group: len(records) for group, records in grouped.items() if records
        },
    }


def readme_text(tag: str, tag_commit: str, manifest: dict[str, Any]) -> str:
    lines = [
        f"# {tag} Measurement Bundle",
        "",
        "This directory is a self-contained supplemental measurement snapshot for",
        "the ICDM 2026 paper iteration. It stores flattened measurement records,",
        "including repeated runs, in semantic JSON bundles.",
        "",
        "## Provenance",
        "",
        f"- Tag: `{tag}`",
        f"- Tagged commit: `{tag_commit}`",
        f"- Total measurement entries: `{manifest['counts']['total_measurement_entries']}`",
        "",
        "## Measurement Entries By Type",
        "",
    ]
    for name, count in manifest["self_check"]["raw_measurement_entries_by_type"].items():
        lines.append(f"- `{name}`: `{count}` entries.")

    lines.extend(["", "## Measurement Entries By Category", ""])
    for group, description in GROUPS.items():
        count = manifest["measurement_groups"].get(group, {}).get("measurement_entry_count", 0)
        if count:
            lines.append(f"- `{group}`: {description} Entries: `{count}`.")

    lines.extend(["", "## Auxiliary Bundles", ""])
    for name, meta in manifest["auxiliary_bundles"].items():
        if name == "registry_snapshot":
            lines.append(f"- `{meta['output']}`: derived summary tables and registry snapshots.")
        elif name == "launch_manifests":
            lines.append(f"- `{meta['output']}`: launch manifests for reproducibility.")
        elif name == "launcher_errors":
            lines.append(f"- `{meta['output']}`: recorded launcher failures.")
        else:
            lines.append(f"- `{meta['output']}`.")

    lines.extend(
        [
            "",
            "All measurement data needed to inspect the reported runs is embedded",
            "in this directory.",
            "",
        ]
    )
    return "\n".join(lines)


def build_bundle(input_dir: Path, output_dir: Path, tag: str, tag_commit: str, force: bool) -> None:
    if output_dir.exists():
        if not force:
            raise SystemExit(f"{output_dir} already exists; rerun with --force to replace it.")
        shutil.rmtree(output_dir)

    raw_measurements, raw_manifests, raw_errors = scan_input(input_dir, output_dir)
    cleaned_measurements = [
        clean_measurement(record, index)
        for index, record in enumerate(raw_measurements)
    ]

    grouped: dict[str, list[dict[str, Any]]] = {group: [] for group in GROUPS}
    for record in cleaned_measurements:
        grouped[measurement_group(record)].append(record)

    generated_at = datetime.now(timezone.utc).isoformat()
    measurement_meta: dict[str, dict[str, Any]] = {}
    for group, records in grouped.items():
        if not records:
            continue
        output = f"measurements/{group}.json"
        measurement_meta[group] = {
            "description": GROUPS[group],
            "measurement_entry_count": len(records),
            "output": output,
        }
        write_json(
            output_dir / output,
            {
                "group": group,
                "description": GROUPS[group],
                "tag": tag,
                "tag_commit": tag_commit,
                "generated_at": generated_at,
                "measurement_entry_count": len(records),
                "records": records,
            },
        )

    registry_snapshot = load_registry_snapshot(input_dir)
    registry_snapshot = sanitize_value(registry_snapshot)
    flattened_errors = [item for batch in raw_errors for item in batch]
    auxiliary = {
        "launch_manifests": [clean_aux_record(item, i, "launch-manifest") for i, item in enumerate(raw_manifests)],
        "launcher_errors": [clean_aux_record(item, i, "launcher-error") for i, item in enumerate(flattened_errors)],
        "registry_snapshot": registry_snapshot,
    }

    auxiliary_meta = {
        "launch_manifests": {
            "record_count": len(auxiliary["launch_manifests"]),
            "output": "auxiliary/launch_manifests.json",
        },
        "launcher_errors": {
            "record_count": len(auxiliary["launcher_errors"]),
            "output": "auxiliary/launcher_errors.json",
        },
        "registry_snapshot": {
            "record_count": sum(
                len(value) if isinstance(value, list) else 1
                for value in registry_snapshot.values()
            ),
            "output": "auxiliary/registry_snapshot.json",
        },
    }
    for name, payload in auxiliary.items():
        output = auxiliary_meta[name]["output"]
        write_json(
            output_dir / output,
            {
                "group": name,
                "tag": tag,
                "tag_commit": tag_commit,
                "generated_at": generated_at,
                **(
                    {"records": payload}
                    if isinstance(payload, list)
                    else {"payload": payload}
                ),
            },
        )

    self_check = build_self_checks(grouped, raw_measurements)
    manifest = {
        "tag": tag,
        "tag_commit": tag_commit,
        "generated_at": generated_at,
        "counts": {
            "total_measurement_entries": len(cleaned_measurements),
            "measurement_entries_by_category": {
                group: meta["measurement_entry_count"]
                for group, meta in measurement_meta.items()
            },
        },
        "measurement_groups": measurement_meta,
        "auxiliary_bundles": auxiliary_meta,
        "self_check": self_check,
    }
    write_json(output_dir / "manifest.json", manifest)
    (output_dir / "README.md").write_text(readme_text(tag, tag_commit, manifest), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--tag-commit", default=DEFAULT_TAG_COMMIT)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir or args.input_dir / args.tag
    build_bundle(args.input_dir, output_dir, args.tag, args.tag_commit, args.force)
    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()
