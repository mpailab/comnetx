"""Run ComNetX stress tests on DSBM synthetic edge streams.

Unlike the main launcher configs, DSBM streams already contain one large
initial layer at time 0 followed by controlled update layers. This runner loads
one stream at a time and calls ``dynamic_launch`` with a synthetic
``0:<updates>`` strategy so the first layer is used only for initialization and
the remaining layers are measured as updates.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any


PROJECT_PATH = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_PATH / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from launcher import dynamic_launch  # noqa: E402

from summarize_dsbm_streams import (  # noqa: E402
    find_streams,
    matching_coms_path,
    read_npz_metadata,
    stream_batch_suffix,
)


METHOD_SUPPORTS_DYNAMIC = {"dfleiden", "ldleiden", "mfc"}
METHOD_ITER_DEFAULTS = {"flmig": 1, "s2cag": 10, "dmon": 10, "magi": 10, "mfc": 100}
SMART_ABBR = {"smart_subcoms_depth": "L", "smart_neighborhood_step": "r"}
DATASET_RE = re.compile(r"^dsbm-(?P<regime>random|hubs|community)-.*-mc(?P<max_changes>\d+)-(?P<seed>\d+)$")
BATCH_SUFFIX_RE = re.compile(r"^(?P<updates>\d+)_batches$")


def load_existing_registry(path: str | None) -> set[tuple[str, str, str, str]]:
    if not path:
        return set()
    registry_path = Path(path)
    if not registry_path.exists():
        raise SystemExit(f"Skip registry does not exist: {registry_path}")

    with registry_path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)

    existing: set[tuple[str, str, str, str]] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("measurement_type") != "experiment":
            continue
        dataset = str(row.get("base_dataset") or row.get("dataset") or "")
        if not DATASET_RE.match(dataset):
            continue
        method = row.get("method")
        mode = row.get("mode")
        batch_strategy = row.get("batch_strategy")
        if method and mode and batch_strategy:
            existing.add((dataset, str(batch_strategy), str(method), str(mode)))
    return existing


def algorithm_name(
    method: str,
    mode: str,
    baseline_iter: int | None,
    smart_depth: int,
    smart_radius: int,
    use_gpu: bool,
) -> str:
    prefix = f"{method}-i:{baseline_iter}" if baseline_iter is not None else method
    if mode == "smart":
        device = "gpu" if use_gpu else "cpu"
        return f"{prefix}-L:{smart_depth}-r:{smart_radius}-{device}"
    return f"{prefix}-{mode}"


def load_stream_as_dataset(out_path: Path):
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("run_dsbm_stress.py requires torch") from exc

    coms = read_npz_metadata(matching_coms_path(out_path))
    labels = torch.tensor(coms["communities"], dtype=torch.long)
    with out_path.open("r", encoding="utf-8") as handle:
        header = handle.readline().split()
        num_nodes = int(header[0])
        rows_by_t: dict[int, list[int]] = {}
        cols_by_t: dict[int, list[int]] = {}
        vals_by_t: dict[int, list[float]] = {}
        min_node = None
        for line in handle:
            if not line.strip():
                continue
            source_raw, target_raw, weight_raw, time_raw = line.split()[:4]
            source = int(source_raw)
            target = int(target_raw)
            weight = float(weight_raw)
            time_idx = int(time_raw)
            min_node = source if min_node is None else min(min_node, source, target)
            rows_by_t.setdefault(time_idx, []).append(source)
            cols_by_t.setdefault(time_idx, []).append(target)
            vals_by_t.setdefault(time_idx, []).append(weight)

    offset = 1 if min_node == 1 else 0
    adjs = []
    for time_idx in sorted(rows_by_t):
        rows = torch.tensor([item - offset for item in rows_by_t[time_idx]], dtype=torch.long)
        cols = torch.tensor([item - offset for item in cols_by_t[time_idx]], dtype=torch.long)
        indices = torch.stack([rows, cols], dim=0)
        values = torch.tensor(vals_by_t[time_idx], dtype=torch.float32)
        adj = torch.sparse_coo_tensor(indices, values, size=(num_nodes, num_nodes)).coalesce()
        adj = (adj + adj.transpose(0, 1)).coalesce()
        adjs.append(adj)

    ds = SimpleNamespace()
    ds.name = out_path.parent.name
    ds.adj = torch.stack(adjs)
    ds.features = None
    ds.label = labels
    ds.is_directed = False
    return ds


def selected_streams(
    root: Path,
    batch_suffix: str | None,
    regimes: set[str],
    max_changes: set[int],
    seeds: set[int] | None = None,
) -> list[Path]:
    streams = []
    for path in find_streams(root, batch_suffix):
        dataset = path.parent.name
        match = DATASET_RE.match(dataset)
        if not match:
            continue
        if regimes and match.group("regime") not in regimes:
            continue
        if max_changes and int(match.group("max_changes")) not in max_changes:
            continue
        if seeds and int(match.group("seed")) not in seeds:
            continue
        streams.append(path)
    return sorted(streams, key=stream_sort_key)


def stream_sort_key(path: Path) -> tuple[Any, ...]:
    dataset = path.parent.name
    dataset_match = DATASET_RE.match(dataset)
    suffix_match = BATCH_SUFFIX_RE.match(stream_batch_suffix(path))
    max_changes = int(dataset_match.group("max_changes")) if dataset_match else sys.maxsize
    updates = int(suffix_match.group("updates")) if suffix_match else sys.maxsize
    regime = dataset_match.group("regime") if dataset_match else ""
    seed = int(dataset_match.group("seed")) if dataset_match else sys.maxsize
    return (max_changes, updates, regime, seed, str(path))


def write_json_atomic(path: Path, payload: Any, ensure_ascii: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=ensure_ascii),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def save_results(output_dir: Path, name: str, db: dict[str, Any], errors: list[Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_dir / f"{name}.json", db)
    if errors:
        write_json_atomic(output_dir / f"errors_{name}.json", errors)


def save_manifest(output_dir: Path, name: str, manifest: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_dir / f"manifest_{name}.json", manifest, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="datasets-sbm")
    batch_group = parser.add_mutually_exclusive_group(required=True)
    batch_group.add_argument(
        "--batch-suffix",
        help="Run one DSBM stream granularity, for example 5_batches, 10_batches, or 100_batches.",
    )
    batch_group.add_argument(
        "--all-batches",
        action="store_true",
        help="Run every available DSBM stream granularity.",
    )
    parser.add_argument("--regimes", nargs="*", default=["random", "hubs", "community"])
    parser.add_argument("--max-changes", nargs="*", type=int, default=[290, 1450, 2900, 14500, 29000])
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=[],
        help="Run only these predeclared DSBM seeds; an empty list selects all seeds.",
    )
    parser.add_argument("--methods", nargs="*", default=["leidenalg", "dfleiden"])
    parser.add_argument("--modes", nargs="*", default=["naive", "smart", "dynamic"])
    parser.add_argument("--smart-depth", type=int, default=3)
    parser.add_argument("--smart-radius", type=int, default=1)
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Shared directory for paired full/smart bootstrap partitions.",
    )
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--output-dir", default="results/paper_icdm")
    parser.add_argument("--name", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--skip-registry",
        default=None,
        help=(
            "Path to results/registry/all_results.json. Existing DSBM "
            "dataset/batch/method/mode rows are skipped instead of rerun."
        ),
    )
    parser.add_argument("--catch-errors", action="store_true")
    parser.add_argument(
        "--list-streams",
        action="store_true",
        help="Print selected DSBM streams and exit before running algorithms.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    output_name = args.name or f"dsbm_stress_{datetime.now().strftime('%Y%m%d_%H%M')}"
    batch_suffix = None if args.all_batches else args.batch_suffix
    existing_registry = load_existing_registry(args.skip_registry)

    if not root.exists():
        raise SystemExit(
            f"DSBM root does not exist: {root}. "
            "Set DSBM_ROOT or pass --root to the DSBM runner."
        )

    streams = selected_streams(
        root,
        batch_suffix,
        set(args.regimes),
        set(args.max_changes),
        set(args.seeds),
    )
    if args.limit is not None:
        streams = streams[: args.limit]

    manifest: dict[str, Any] = {
        "root": str(root),
        "batch_suffix": batch_suffix,
        "all_batches": bool(args.all_batches),
        "regimes": args.regimes,
        "max_changes": args.max_changes,
        "seeds": args.seeds,
        "methods": args.methods,
        "modes": args.modes,
        "selected_streams": len(streams),
        "selected_stream_paths": [str(path) for path in streams],
        "attempted_runs": 0,
        "successful_runs": 0,
        "skipped_existing": 0,
        "errors": 0,
        "status": "selected",
        "started_at": datetime.now().isoformat(),
        "updated_at": None,
        "current_attempt": None,
        "output": str(Path(args.output_dir) / f"{output_name}.json"),
    }

    if args.list_streams:
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        if not streams:
            raise SystemExit("No DSBM streams selected.")
        return

    if not streams:
        save_results(Path(args.output_dir), output_name, {}, [])
        save_manifest(Path(args.output_dir), output_name, manifest)
        raise SystemExit(
            "No DSBM streams selected. Check --root, --batch-suffix, --regimes, "
            "and --max-changes. The empty result file was written only as a "
            "failure marker."
        )

    print(
        json.dumps(
            {
                "root": str(root),
                "selected_streams": len(streams),
                "methods": args.methods,
                "modes": args.modes,
                "output": manifest["output"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    machine = os.getenv("PARENT_HOSTNAME") or os.getenv("HOSTNAME") or "unknown"
    db: dict[str, Any] = {}
    errors: list[Any] = []
    output_dir = Path(args.output_dir)

    def checkpoint(status: str | None = None) -> None:
        if status is not None:
            manifest["status"] = status
        manifest["errors"] = len(errors)
        manifest["updated_at"] = datetime.now().isoformat()
        save_results(output_dir, output_name, db, errors)
        save_manifest(output_dir, output_name, manifest)

    def handle_signal(signum: int, _frame: Any) -> None:
        signal_name = signal.Signals(signum).name
        manifest["interrupted_by_signal"] = signal_name
        checkpoint("interrupted")
        raise SystemExit(f"Interrupted by {signal_name}; checkpoint was written.")

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    checkpoint("started")

    for out_path in streams:
        ds = load_stream_as_dataset(out_path)
        updates = int(ds.adj.shape[0] - 1)
        batch_strategy = f"0:{updates}"
        for method in args.methods:
            for mode in args.modes:
                if mode == "dynamic" and method not in METHOD_SUPPORTS_DYNAMIC:
                    continue
                registry_key = (ds.name, batch_strategy, method, mode)
                if registry_key in existing_registry:
                    manifest["skipped_existing"] += 1
                    print(f"Skip existing DSBM run: {registry_key}")
                    checkpoint("running")
                    continue
                baseline_iter = METHOD_ITER_DEFAULTS.get(method)
                alg = algorithm_name(
                    method,
                    mode,
                    baseline_iter,
                    args.smart_depth,
                    args.smart_radius,
                    args.use_gpu,
                )
                manifest["attempted_runs"] += 1
                manifest["current_attempt"] = {
                    "algorithm": alg,
                    "dataset": ds.name,
                    "stream": str(out_path),
                    "batch_strategy": batch_strategy,
                    "attempt_index": manifest["attempted_runs"],
                }
                checkpoint("running")
                try:
                    results = dynamic_launch(
                        ds,
                        batch_strategy,
                        method,
                        baseline_iter=baseline_iter,
                        mode=mode,
                        smart_subcoms_depth=args.smart_depth,
                        smart_neighborhood_step=args.smart_radius,
                        verbose=1,
                        use_gpu=args.use_gpu,
                        aggregation_mode="sum",
                        cache_dir=args.cache_dir,
                        ground_truth_metrics=True,
                    )
                except Exception as exc:
                    error_record = [alg, ds.name, batch_strategy, str(exc)]
                    errors.append(error_record)
                    manifest["last_failed_attempt"] = manifest["current_attempt"]
                    manifest["current_attempt"] = None
                    print(f"Error on {alg}, {ds.name}, {batch_strategy}: {exc}")
                    checkpoint("running")
                    if not args.catch_errors:
                        raise
                    continue

                db.setdefault(alg, {}).setdefault(ds.name, {}).setdefault(machine, {})[
                    batch_strategy
                ] = results
                manifest["successful_runs"] += 1
                manifest["current_attempt"] = None
                checkpoint("running")

    manifest["current_attempt"] = None
    checkpoint("completed")

    if manifest["attempted_runs"] == 0:
        if manifest["skipped_existing"] > 0:
            checkpoint("completed")
            print(json.dumps(manifest, indent=2, ensure_ascii=False))
            return
        checkpoint("failed")
        raise SystemExit(
            "No runnable method/mode combinations were attempted. "
            "Check --methods and --modes."
        )
    if manifest["successful_runs"] == 0:
        checkpoint("failed")
        raise SystemExit(
            f"All {manifest['attempted_runs']} DSBM runs failed. "
            f"See {Path(args.output_dir) / f'errors_{output_name}.json'} and "
            f"{Path(args.output_dir) / f'manifest_{output_name}.json'}."
        )

    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
