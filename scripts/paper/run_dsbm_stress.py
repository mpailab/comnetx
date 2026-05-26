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


METHOD_SUPPORTS_DYNAMIC = {"dfleiden", "ldleiden", "mfc", "lago"}
METHOD_ITER_DEFAULTS = {"flmig": 1, "s2cag": 10, "dmon": 10, "magi": 10, "mfc": 100, "lago": 1}
SMART_ABBR = {"smart_subcoms_depth": "L", "smart_neighborhood_step": "r"}
DATASET_RE = re.compile(r"^dsbm-(?P<regime>random|hubs|community)-.*-mc(?P<max_changes>\d+)-(?P<seed>\d+)$")


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


def selected_streams(root: Path, batch_suffix: str | None, regimes: set[str], max_changes: set[int]) -> list[Path]:
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
        streams.append(path)
    return streams


def save_results(output_dir: Path, name: str, db: dict[str, Any], errors: list[Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{name}.json").write_text(json.dumps(db, indent=2), encoding="utf-8")
    if errors:
        (output_dir / f"errors_{name}.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")


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
    parser.add_argument("--methods", nargs="*", default=["leidenalg", "dfleiden"])
    parser.add_argument("--modes", nargs="*", default=["naive", "smart", "dynamic"])
    parser.add_argument("--smart-depth", type=int, default=3)
    parser.add_argument("--smart-radius", type=int, default=1)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--output-dir", default="results/paper_icdm")
    parser.add_argument("--name", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--catch-errors", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    output_name = args.name or f"dsbm_stress_{datetime.now().strftime('%Y%m%d_%H%M')}"
    batch_suffix = None if args.all_batches else args.batch_suffix
    streams = selected_streams(root, batch_suffix, set(args.regimes), set(args.max_changes))
    if args.limit is not None:
        streams = streams[: args.limit]

    machine = os.getenv("PARENT_HOSTNAME") or os.getenv("HOSTNAME") or "unknown"
    db: dict[str, Any] = {}
    errors: list[Any] = []

    for out_path in streams:
        ds = load_stream_as_dataset(out_path)
        updates = int(ds.adj.shape[0] - 1)
        batch_strategy = f"0:{updates}"
        for method in args.methods:
            for mode in args.modes:
                if mode == "dynamic" and method not in METHOD_SUPPORTS_DYNAMIC:
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
                        ground_truth_metrics=True,
                    )
                except Exception as exc:
                    if not args.catch_errors:
                        raise
                    errors.append([alg, ds.name, batch_strategy, str(exc)])
                    continue

                db.setdefault(alg, {}).setdefault(ds.name, {}).setdefault(machine, {})[
                    batch_strategy
                ] = results
                save_results(Path(args.output_dir), output_name, db, errors)

    save_results(Path(args.output_dir), output_name, db, errors)
    print(json.dumps({"streams": len(streams), "output": str(Path(args.output_dir) / f"{output_name}.json"), "errors": len(errors)}, indent=2))


if __name__ == "__main__":
    main()
