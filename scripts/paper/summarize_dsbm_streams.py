"""Summarize DSBM synthetic dynamic edge streams for the ICDM paper.

The script scans ``datasets-sbm`` without requiring NumPy. It reads the
KONECT-like ``out.*`` edge streams and the small ``coms.*.npz`` metadata files,
then writes compact CSV/JSON summaries that describe the stress-test regimes:
random updates, hub-centered updates, and community-internal updates.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
import struct
import zipfile
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


DATASET_RE = re.compile(
    r"^dsbm-(?P<regime>random|hubs|community)-(?P<tag>\d+-\d+)-mc(?P<max_changes>\d+)-(?P<seed>\d+)$"
)


def _npy_header(payload: bytes) -> tuple[dict[str, Any], int]:
    if payload[:6] != b"\x93NUMPY":
        raise ValueError("Not an NPY payload")
    version = (payload[6], payload[7])
    if version == (1, 0):
        header_len = struct.unpack("<H", payload[8:10])[0]
        data_start = 10 + header_len
        header_start = 10
    elif version in {(2, 0), (3, 0)}:
        header_len = struct.unpack("<I", payload[8:12])[0]
        data_start = 12 + header_len
        header_start = 12
    else:
        raise ValueError(f"Unsupported NPY version: {version}")
    header_text = payload[header_start:data_start].decode("latin1").strip()
    return ast.literal_eval(header_text), data_start


def _shape_count(shape: Any) -> int:
    if shape == ():
        return 1
    count = 1
    for dim in shape:
        count *= int(dim)
    return count


def read_npy(payload: bytes) -> Any:
    header, data_start = _npy_header(payload)
    descr = header["descr"]
    shape = header["shape"]
    count = _shape_count(shape)
    data = payload[data_start:]

    if descr == "<i4":
        values = list(struct.unpack("<" + "i" * count, data[: count * 4]))
    elif descr == "<i8":
        values = list(struct.unpack("<" + "q" * count, data[: count * 8]))
    elif descr == "<f8":
        values = list(struct.unpack("<" + "d" * count, data[: count * 8]))
    elif isinstance(descr, str) and descr.startswith("<U"):
        chars = int(descr[2:])
        raw = data[: count * chars * 4]
        values = [raw.decode("utf-32le").rstrip("\x00")]
    else:
        raise ValueError(f"Unsupported NPY dtype: {descr}")

    return values[0] if shape == () else values


def read_npz_metadata(path: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            key = name.removesuffix(".npy")
            value = read_npy(archive.read(name))
            if key.startswith("coms_t"):
                metadata["communities"] = value
                metadata["communities_key"] = key
            else:
                metadata[key] = value
    return metadata


def parse_dataset_name(name: str) -> dict[str, Any]:
    match = DATASET_RE.match(name)
    if not match:
        raise ValueError(f"Unsupported DSBM dataset name: {name}")
    out = match.groupdict()
    out["max_changes"] = int(out["max_changes"])
    out["seed"] = int(out["seed"])
    return out


def find_streams(root: Path, batch_suffix: str | None) -> list[Path]:
    streams = []
    for path in root.glob("**/out.*"):
        if not path.is_file():
            continue
        if batch_suffix is not None and not path.name.endswith(f".{batch_suffix}"):
            continue
        streams.append(path)
    return sorted(streams)


def stream_batch_suffix(path: Path) -> str:
    return path.name.rsplit(".", 1)[-1]


def matching_coms_path(out_path: Path) -> Path:
    dataset = out_path.parent.name
    suffix = stream_batch_suffix(out_path)
    return out_path.parent / f"coms.{dataset}.{suffix}.npz"


def pct(part: float, whole: float) -> float | None:
    if whole == 0:
        return None
    return part / whole * 100.0


def summarize_stream(out_path: Path) -> dict[str, Any]:
    dataset = out_path.parent.name
    name_parts = parse_dataset_name(dataset)
    coms_path = matching_coms_path(out_path)
    metadata = read_npz_metadata(coms_path)
    communities = metadata.get("communities")
    planted_hubs = set(metadata.get("planted_hubs", []))

    counts: dict[int, int] = defaultdict(int)
    affected: dict[int, set[int]] = defaultdict(set)
    affected_hubs: dict[int, set[int]] = defaultdict(set)
    hub_incident_edges: dict[int, int] = defaultdict(int)
    intra_edges: dict[int, int] = defaultdict(int)
    min_node = math.inf
    max_node = -math.inf

    with out_path.open("r", encoding="utf-8") as handle:
        header = handle.readline().split()
        if len(header) < 2:
            raise ValueError(f"Bad stream header in {out_path}")
        header_nodes = int(header[0])
        header_edges = int(header[1])

        for line in handle:
            if not line.strip():
                continue
            source_raw, target_raw, _weight, time_raw = line.split()[:4]
            source = int(source_raw)
            target = int(target_raw)
            time_idx = int(time_raw)
            min_node = min(min_node, source, target)
            max_node = max(max_node, source, target)
            counts[time_idx] += 1

            if time_idx == 0:
                continue

            # DSBM edge streams are one-based; keep a fallback for zero-based
            # files so the script remains useful for regenerated variants.
            source0 = source - 1 if source >= 1 else source
            target0 = target - 1 if target >= 1 else target
            affected[time_idx].update((source0, target0))

            if source0 in planted_hubs:
                affected_hubs[time_idx].add(source0)
            if target0 in planted_hubs:
                affected_hubs[time_idx].add(target0)
            if source0 in planted_hubs or target0 in planted_hubs:
                hub_incident_edges[time_idx] += 1

            if communities and communities[source0] == communities[target0]:
                intra_edges[time_idx] += 1

    update_times = sorted(time for time in counts if time != 0)
    update_edges = [counts[time] for time in update_times]
    affected_counts = [len(affected[time]) for time in update_times]
    hub_endpoint_pcts = [
        pct(len(affected_hubs[time]), len(affected[time]))
        for time in update_times
        if len(affected[time]) > 0
    ]
    hub_edge_pcts = [
        pct(hub_incident_edges[time], counts[time])
        for time in update_times
        if counts[time] > 0
    ]
    intra_edge_pcts = [
        pct(intra_edges[time], counts[time])
        for time in update_times
        if counts[time] > 0 and communities
    ]

    regime = name_parts["regime"]
    display_regime = "community-internal" if regime == "community" else regime

    return {
        "dataset": dataset,
        "family": out_path.parent.parent.name,
        "regime": regime,
        "display_regime": display_regime,
        "batch_suffix": stream_batch_suffix(out_path),
        "seed": name_parts["seed"],
        "header_nodes": header_nodes,
        "header_edges": header_edges,
        "min_node_id": None if min_node is math.inf else int(min_node),
        "max_node_id": None if max_node == -math.inf else int(max_node),
        "metadata_n": metadata.get("n"),
        "metadata_k": metadata.get("k"),
        "metadata_avr_deg": metadata.get("avr_deg"),
        "metadata_max_deg": metadata.get("max_deg"),
        "metadata_dynamic_mode": metadata.get("dynamic_mode"),
        "metadata_hub_fraction": metadata.get("hub_fraction"),
        "metadata_hub_avr_deg": metadata.get("hub_avr_deg"),
        "metadata_nonhub_avr_deg": metadata.get("nonhub_avr_deg"),
        "max_changes": name_parts["max_changes"],
        "change_pct_of_total_edges": pct(name_parts["max_changes"], header_edges),
        "batches": len(counts),
        "updates": len(update_times),
        "initial_edges": counts.get(0, 0),
        "update_edges_mean": mean(update_edges) if update_edges else None,
        "update_edges_min": min(update_edges) if update_edges else None,
        "update_edges_max": max(update_edges) if update_edges else None,
        "affected_vertices_mean": mean(affected_counts) if affected_counts else None,
        "affected_vertices_max": max(affected_counts) if affected_counts else None,
        "affected_vertices_pct_mean": pct(mean(affected_counts), header_nodes)
        if affected_counts
        else None,
        "affected_vertices_pct_max": pct(max(affected_counts), header_nodes)
        if affected_counts
        else None,
        "planted_hub_endpoint_pct_mean": mean(hub_endpoint_pcts)
        if hub_endpoint_pcts
        else None,
        "hub_incident_edge_pct_mean": mean(hub_edge_pcts) if hub_edge_pcts else None,
        "intra_community_edge_pct_mean": mean(intra_edge_pcts)
        if intra_edge_pcts
        else None,
        "out_file": str(out_path),
        "coms_file": str(coms_path),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def latex_rows(rows: list[dict[str, Any]]) -> str:
    ordered = sorted(rows, key=lambda row: (row["max_changes"], row["regime"]))
    lines = []
    for row in ordered:
        lines.append(
            (
                f"{row['display_regime']} & {row['max_changes']} & "
                f"{row['change_pct_of_total_edges']:.3f} & {row['updates']} & "
                f"{row['update_edges_mean']:.0f} & {row['affected_vertices_mean']:.1f} & "
                f"{row['affected_vertices_pct_mean']:.3f} & "
                f"{row['planted_hub_endpoint_pct_mean']:.1f} & "
                f"{row['intra_community_edge_pct_mean']:.1f} \\\\"
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="datasets-sbm")
    parser.add_argument(
        "--batch-suffix",
        default=None,
        help="Restrict to one stream suffix, for example 10_batches. By default all suffixes are scanned.",
    )
    parser.add_argument("--all-batches", action="store_true")
    parser.add_argument("--output-dir", default="results/registry")
    args = parser.parse_args()

    root = Path(args.root)
    output_dir = Path(args.output_dir)
    batch_suffix = None if args.all_batches or args.batch_suffix is None else args.batch_suffix
    rows = [summarize_stream(path) for path in find_streams(root, batch_suffix)]

    suffix = "all_batches" if batch_suffix is None else batch_suffix
    base = output_dir / f"dsbm_stream_summary_{suffix}"
    write_json(base.with_suffix(".json"), rows)
    write_csv(base.with_suffix(".csv"), rows)
    (base.with_suffix(".tex")).write_text(latex_rows(rows), encoding="utf-8")

    manifest = {
        "root": str(root),
        "batch_suffix": batch_suffix,
        "streams": len(rows),
        "outputs": {
            "json": str(base.with_suffix(".json")),
            "csv": str(base.with_suffix(".csv")),
            "latex_rows": str(base.with_suffix(".tex")),
        },
    }
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
