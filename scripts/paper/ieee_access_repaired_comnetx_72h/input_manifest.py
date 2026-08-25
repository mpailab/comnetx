#!/usr/bin/env python3
"""Build and verify content-addressed manifests for measurement inputs."""

from __future__ import annotations

import argparse
import json
import re
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
    from .protocol import config_path, load_protocol, sha256_file, write_json
except ImportError:  # Direct script execution.
    from protocol import config_path, load_protocol, sha256_file, write_json


DSBM_RE = re.compile(
    r"^dsbm-(random|hubs|community)-.*-mc(290|1450)-(42|43|44|45|46)$"
)


def _batch_source_suffix(batch: str) -> str:
    if ":" in batch:
        initial, _ = batch.split(":", 1)
        return str(int(initial) + 1)
    return str(int(batch))


def _real_dataset_files(
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
            f"unsupported registered real-stream format {dataset_format!r} "
            f"for {dataset}"
        )
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing registered input files: {missing}")
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


def build_registered_real_input_manifest(paths_config: Path) -> dict[str, Any]:
    paths_config = paths_config.resolve()
    protocol = load_protocol()
    combinations: set[tuple[str, str]] = set()
    for stage in protocol["stages"]:
        relatives = ([stage["config"]] if stage.get("config") else []) + list(
            stage.get("configs", [])
        )
        for relative in relatives:
            config = json.loads(config_path(relative).read_text(encoding="utf-8"))
            combinations.update(
                (str(dataset), str(batch))
                for dataset in config["DATASETS"]
                for batch in config["BATCHES"]
            )

    paths_to_uses: dict[Path, list[str]] = {}
    formats: dict[str, str] = {}
    for dataset, batch in sorted(combinations):
        dataset_format, paths = _real_dataset_files(paths_config, dataset, batch)
        formats[dataset] = dataset_format
        for path in paths:
            paths_to_uses.setdefault(path.resolve(), []).append(f"{dataset}/{batch}")

    return {
        "schema": "comnetx-ieee-access-real-inputs-v1",
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


def build_dsbm_input_manifest(root: Path) -> dict[str, Any]:
    root = root.resolve()
    selected: list[tuple[Path, re.Match[str]]] = []
    for path in sorted(root.glob("**/out.*.100_batches")):
        match = DSBM_RE.match(path.parent.name)
        if path.is_file() and match is not None:
            selected.append((path, match))
    observed = {
        (match.group(1), int(match.group(2)), int(match.group(3)))
        for _, match in selected
    }
    expected = {
        (regime, changes, seed)
        for regime in ("random", "hubs", "community")
        for changes in (290, 1450)
        for seed in (42, 43, 44, 45, 46)
    }
    if observed != expected or len(selected) != 30:
        raise ValueError("DSBM inputs are not the registered 3x2x5 design")

    records = []
    for stream, _ in selected:
        communities = stream.parent / f"coms.{stream.parent.name}.100_batches.npz"
        if not communities.is_file():
            raise FileNotFoundError(f"missing DSBM communities: {communities}")
        records.extend(
            [
                _file_record(stream, [stream.parent.name]),
                _file_record(communities, [stream.parent.name]),
            ]
        )
    return {
        "schema": "comnetx-ieee-access-dsbm-inputs-v1",
        "root": str(root),
        "selected_streams": 30,
        "files": records,
    }


def validate_input_manifest(
    manifest: dict[str, Any],
    *,
    verify_content: bool,
) -> None:
    if manifest.get("schema") not in {
        "comnetx-ieee-access-real-inputs-v1",
        "comnetx-ieee-access-dsbm-inputs-v1",
    }:
        raise ValueError("unexpected input-manifest schema")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("input manifest has no files")
    for record in files:
        path = Path(record["path"])
        if not path.is_file():
            raise FileNotFoundError(f"sealed input disappeared: {path}")
        stat = path.stat()
        if (
            int(record.get("size", -1)) != stat.st_size
            or int(record.get("mtime_ns", -1)) != stat.st_mtime_ns
        ):
            raise ValueError(f"sealed input metadata changed: {path}")
        if verify_content and record.get("sha256") != sha256_file(path):
            raise ValueError(f"sealed input content changed: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_registered_real_input_manifest(args.paths_config)
    validate_input_manifest(manifest, verify_content=True)
    write_json(args.output, manifest)
    print(f"Sealed {len(manifest['files'])} real-stream input files")


if __name__ == "__main__":
    main()
