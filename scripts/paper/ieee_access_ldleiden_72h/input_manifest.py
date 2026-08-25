#!/usr/bin/env python3
"""Build and verify content-addressed real-stream inputs for LD-Leiden."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterable


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from datasets import Dataset  # noqa: E402

try:
    from .protocol import (
        load_protocol,
        sha256_file,
        write_json,
    )
except ImportError:  # Direct script execution.
    from protocol import load_protocol, sha256_file, write_json


def _batch_source_suffix(batch: str) -> str:
    initial, _ = batch.split(":", 1)
    return str(int(initial) + 1)


def _dataset_files(
    paths_config: Path,
    dataset: str,
    batch: str,
) -> tuple[str, list[Path]]:
    descriptor = Dataset(dataset, str(paths_config))
    dataset_format = descriptor.dataset_format
    root = descriptor.dataset_root
    source_suffix = _batch_source_suffix(batch)
    name = descriptor.name
    lower_name = name[0].lower() + name[1:]

    if dataset_format == "konect":
        files = [root / name / f"out.{name}.{source_suffix}_batches"]
    elif dataset_format == "dyn_attr":
        pure_name = name.split("dyn_")[-1]
        suffix = f"{pure_name}-{source_suffix}_batches.npz"
        files = [
            root / lower_name / f"dynamic_{suffix}",
            root / lower_name / f"feat_{suffix}",
        ]
    elif dataset_format == "tgc":
        base = root / lower_name
        files = [
            base / f"{lower_name}_coo_adj-{source_suffix}_batches.joblib",
            base / f"{lower_name}_label.npy",
        ]
        feature_path = base / f"{lower_name}_feat.npy"
        if feature_path.is_file():
            files.append(feature_path)
    else:
        raise ValueError(
            f"unsupported LD-Leiden stream format {dataset_format!r} for {dataset}"
        )
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing registered LD-Leiden inputs: {missing}")
    return dataset_format, files


def _file_record(path: Path, uses: Iterable[str]) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": sha256_file(path),
        "uses": sorted(set(uses)),
    }


def build_input_manifest(paths_config: Path) -> dict[str, Any]:
    """Hash every stream file selected by the three registered phases."""

    paths_config = paths_config.resolve()
    protocol = load_protocol()
    combinations = {
        (str(dataset), str(phase["batch_strategy"]))
        for phase in protocol["phases"]
        for dataset in phase["datasets"]
    }
    paths_to_uses: dict[Path, list[str]] = {}
    formats: dict[str, str] = {}
    for dataset, batch in sorted(combinations):
        dataset_format, files = _dataset_files(paths_config, dataset, batch)
        formats[dataset] = dataset_format
        for path in files:
            paths_to_uses.setdefault(path.resolve(), []).append(f"{dataset}/{batch}")

    return {
        "schema": "comnetx-ieee-access-ldleiden-real-inputs-v1",
        "paths_config": str(paths_config),
        "paths_config_sha256": sha256_file(paths_config),
        "dataset_formats": formats,
        "dataset_batch_combinations": [
            {"dataset": dataset, "batch": batch}
            for dataset, batch in sorted(combinations)
        ],
        "files": [
            _file_record(path, uses)
            for path, uses in sorted(paths_to_uses.items(), key=lambda item: str(item[0]))
        ],
    }


def validate_input_manifest(
    manifest: dict[str, Any],
    *,
    verify_content: bool,
) -> None:
    """Reject missing, replaced, resized, retimestamped, or rehashed inputs."""

    if manifest.get("schema") != "comnetx-ieee-access-ldleiden-real-inputs-v1":
        raise ValueError("unexpected LD-Leiden input-manifest schema")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("LD-Leiden input manifest has no files")
    for record in files:
        path = Path(record["path"])
        if not path.is_file():
            raise FileNotFoundError(f"sealed LD-Leiden input disappeared: {path}")
        stat = path.stat()
        if (
            int(record.get("size", -1)) != stat.st_size
            or int(record.get("mtime_ns", -1)) != stat.st_mtime_ns
        ):
            raise ValueError(f"sealed LD-Leiden input metadata changed: {path}")
        if verify_content and record.get("sha256") != sha256_file(path):
            raise ValueError(f"sealed LD-Leiden input content changed: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_input_manifest(args.paths_config)
    validate_input_manifest(payload, verify_content=True)
    write_json(args.output, payload)
    print(f"Sealed {len(payload['files'])} LD-Leiden input files")


if __name__ == "__main__":
    main()
