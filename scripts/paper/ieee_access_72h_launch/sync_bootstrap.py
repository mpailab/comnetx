#!/usr/bin/env python3
"""Seed LD-Leiden with the exact repaired-ComNetX level-zero bootstrap."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPAIRED_ROOT = (
    PROJECT_ROOT / "results" / "ieee-access-2026-1" / "raw" / "repaired-comnetx"
)
LD_ROOT = PROJECT_ROOT / "results" / "ieee-access-2026-1" / "raw" / "ldleiden"


def _campaign(value: Path, root: Path) -> Path:
    value = value.expanduser()
    if value.is_absolute():
        return value.resolve()
    if len(value.parts) == 1:
        return (root / value).resolve()
    return (PROJECT_ROOT / value).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.size == 0 or labels.dtype.kind not in {"i", "u"}:
        raise ValueError("bootstrap level zero must be a non-empty integral vector")
    canonical = np.empty(labels.shape, dtype=np.int64)
    groups: dict[int, list[int]] = {}
    for vertex, label in enumerate(labels.astype(np.int64, copy=False)):
        groups.setdefault(int(label), []).append(vertex)
    for vertices in groups.values():
        canonical[vertices] = min(vertices)
    return canonical


def _partition_sha256(labels: np.ndarray) -> str:
    labels = _canonical(labels).astype("<i8", copy=False)
    digest = hashlib.sha256()
    digest.update(f"shape={labels.shape};dtype=int64;".encode("ascii"))
    digest.update(labels.tobytes(order="C"))
    return digest.hexdigest()


def _load_source(path: Path) -> tuple[np.ndarray, float]:
    with np.load(path, allow_pickle=False) as payload:
        if "partition" not in payload or "mod" not in payload:
            raise ValueError(f"malformed repaired bootstrap: {path}")
        partition = np.asarray(payload["partition"])
        modularity = float(np.asarray(payload["mod"]).item())
        schema = str(np.asarray(payload["schema"]).item()) if "schema" in payload else None
    if partition.ndim == 2:
        if partition.shape[0] != 3 or schema != "parent_quotient_v1":
            raise ValueError(f"malformed repaired hierarchy bootstrap: {path}")
        partition = partition[0]
    elif partition.ndim == 1 and schema is not None:
        raise ValueError(f"flat repaired bootstrap unexpectedly has a schema: {path}")
    if partition.ndim != 1:
        raise ValueError(f"unexpected repaired bootstrap rank: {path}")
    if not math.isfinite(modularity) or not -1 <= modularity <= 1:
        raise ValueError(f"invalid repaired bootstrap modularity: {path}")
    return _canonical(partition), modularity


def _load_target(path: Path) -> tuple[np.ndarray, float]:
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != {"partition", "mod"}:
            raise ValueError(f"malformed LD-Leiden bootstrap: {path}")
        raw_partition = np.asarray(payload["partition"])
        modularity = float(np.asarray(payload["mod"]).item())
    partition = _canonical(raw_partition)
    if not np.array_equal(raw_partition.astype(np.int64, copy=False), partition):
        raise ValueError(f"LD-Leiden bootstrap is not canonical on disk: {path}")
    if not math.isfinite(modularity) or not -1 <= modularity <= 1:
        raise ValueError(f"invalid LD-Leiden bootstrap modularity: {path}")
    return partition, modularity


def _source_candidate(
    cache_dir: Path, dataset: str, initial_batch: int
) -> Path:
    prefixes = (dataset, f"{dataset}-sym")
    flat = [
        cache_dir / f"{prefix}_b:{initial_batch}_by_leidenalg.npz"
        for prefix in prefixes
    ]
    hierarchical = [
        cache_dir
        / (
            f"{prefix}_b:{initial_batch}_by_leidenalg_"
            "d:3_parent_quotient_v1.npz"
        )
        for prefix in prefixes
    ]
    candidates = [path for path in (*flat, *hierarchical) if path.is_file()]
    if not candidates:
        raise FileNotFoundError(
            f"no repaired bootstrap for {dataset}/initial={initial_batch}"
        )
    observed_prefixes = {
        path.name.split(f"_b:{initial_batch}_by_leidenalg", 1)[0]
        for path in candidates
    }
    if len(observed_prefixes) != 1:
        raise ValueError(
            f"ambiguous repaired graph representations: {sorted(observed_prefixes)}"
        )
    flat_candidates = [path for path in candidates if "_d:3_" not in path.name]
    selected = flat_candidates or candidates
    if len(selected) != 1:
        raise ValueError(f"ambiguous repaired bootstrap candidates: {selected}")
    return selected[0]


def _target_candidate(cache_dir: Path, source: Path, initial_batch: int) -> Path:
    prefix = source.name.split(f"_b:{initial_batch}_by_leidenalg", 1)[0]
    return cache_dir / f"{prefix}_b:{initial_batch}_by_leidenalg.npz"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def synchronize(
    repaired_campaign: Path,
    ld_campaign: Path,
    *,
    datasets: list[str],
    initial_batch: int,
) -> dict[str, Any]:
    repaired_cache = repaired_campaign / "bootstrap-cache"
    ld_cache = ld_campaign / "bootstrap-cache"
    ld_cache.mkdir(parents=True, exist_ok=True)
    report_path = ld_campaign / "bootstrap_sync.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        report = {
            "schema": "comnetx-ieee-access-bootstrap-sync-v1",
            "repaired_campaign": repaired_campaign.name,
            "ld_campaign": ld_campaign.name,
            "entries": {},
        }

    for dataset in datasets:
        source = _source_candidate(repaired_cache, dataset, initial_batch)
        source_partition, source_modularity = _load_source(source)
        target = _target_candidate(ld_cache, source, initial_batch)
        if target.exists():
            target_partition, target_modularity = _load_target(target)
            if not np.array_equal(target_partition, source_partition) or not math.isclose(
                target_modularity, source_modularity, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(
                    f"existing LD-Leiden bootstrap differs from repaired source: {target}"
                )
        else:
            temporary = target.with_suffix(target.suffix + ".tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    partition=source_partition,
                    mod=np.asarray(source_modularity),
                )
            shutil.copymode(source, temporary)
            temporary.replace(target)

        key = f"{dataset}/{initial_batch}"
        report["entries"][key] = {
            "source": source.name,
            "source_sha256": _sha256(source),
            "target": target.name,
            "target_sha256": _sha256(target),
            "level_zero_sha256": _partition_sha256(source_partition),
            "modularity": source_modularity,
        }

    _write_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repaired-campaign", type=Path, required=True)
    parser.add_argument("--ld-campaign", type=Path, required=True)
    parser.add_argument("--initial-batch", type=int, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    args = parser.parse_args()
    repaired = _campaign(args.repaired_campaign, REPAIRED_ROOT)
    ld = _campaign(args.ld_campaign, LD_ROOT)
    report = synchronize(
        repaired,
        ld,
        datasets=args.datasets,
        initial_batch=args.initial_batch,
    )
    print(
        f"Synchronized {len(args.datasets)} bootstraps; "
        f"registered entries: {len(report['entries'])}"
    )


if __name__ == "__main__":
    main()
