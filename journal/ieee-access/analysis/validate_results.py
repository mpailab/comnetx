#!/usr/bin/env python3
"""Validate the IEEE Access evidence set and regenerate paper figures/rows.

The script deliberately uses a narrow whitelist of measurement families.  It
does not reproduce the ICDM compatibility aggregation, which mixed legacy and
single-container records with different graph directions and feature modes.

Chart map
---------
``long_horizon.pdf``
    Question: how do the modularity gap and cumulative runtime advantage evolve
    over 500 updates?  Two-panel line chart, 500 observations per series.
``dsbm_speedup.pdf``
    Question: where does local processing cross the 1x break-even boundary?
    Horizontal mean-and-range dot plot, five paired seeds per condition.
``topology_ablation.pdf``
    Question: how do hierarchy depth and radius trade quality for speed?
    Three faceted scatter plots with all 12 measured configurations.
``method_pipeline.pdf``
    Non-quantitative reader-oriented overview of the update workflow.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable

import reportlab
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[3]
JOURNAL = ROOT / "journal" / "ieee-access"
ANALYSIS = JOURNAL / "analysis"
FIGURES = JOURNAL / "figures"
GENERATED = JOURNAL / "generated"
REPORTLAB_FONTS = Path(reportlab.__file__).resolve().parent / "fonts"
FIGURE_FONT = "ComNetXFigureSans"
FIGURE_FONT_BOLD = "ComNetXFigureSans-Bold"
pdfmetrics.registerFont(TTFont(FIGURE_FONT, REPORTLAB_FONTS / "Vera.ttf"))
pdfmetrics.registerFont(TTFont(FIGURE_FONT_BOLD, REPORTLAB_FONTS / "VeraBd.ttf"))

EXPERIMENTS = (
    ROOT
    / "results"
    / "icdm-2026-1"
    / "measurements"
    / "experiment_measurements.json"
)
WORKLOADS = (
    ROOT
    / "results"
    / "icdm-2026-1"
    / "measurements"
    / "workload_profiles.json"
)
NEIGHBORHOODS = (
    ROOT
    / "results"
    / "icdm-2026-1"
    / "measurements"
    / "neighborhood_measurements.json"
)
CUT_METRICS = (
    ROOT
    / "results"
    / "icdm-2026-1"
    / "measurements"
    / "cut_metrics.json"
)
DATA_INFO = {
    "dyn_cora": ROOT / "datasets-info" / "json" / "dyn_attr.json",
    "dyn_acm": ROOT / "datasets-info" / "json" / "dyn_attr.json",
    "dyn_citeseer": ROOT / "datasets-info" / "json" / "dyn_attr.json",
    "dyn_pubmed": ROOT / "datasets-info" / "json" / "dyn_attr.json",
    "patent": ROOT / "datasets-info" / "json" / "tgc.json",
    "arxivmath": ROOT / "datasets-info" / "json" / "tgc.json",
}

DATASETS = (
    "dyn_cora",
    "dyn_acm",
    "dyn_citeseer",
    "patent",
    "dyn_pubmed",
    "arxivmath",
)
DISPLAY = {
    "dyn_cora": "dyn\\_cora",
    "dyn_acm": "dyn\\_acm",
    "dyn_citeseer": "dyn\\_citeseer",
    "patent": "patent",
    "dyn_pubmed": "dyn\\_pubmed",
    "arxivmath": "arxivmath",
}
STREAM_KIND = {
    "dyn_cora": "stored artificial order",
    "dyn_acm": "stored artificial order",
    "dyn_citeseer": "stored artificial order",
    "dyn_pubmed": "stored artificial order",
    "patent": "timestamp order",
    "arxivmath": "timestamp order",
}
EXPECTED_SHORT_COUNTS = {
    "dyn_cora": 1,
    "dyn_acm": 1,
    "dyn_citeseer": 1,
    "patent": 1,
    "dyn_pubmed": 5,
    "arxivmath": 5,
}
EXPECTED_SHORT_UPDATES = {
    "dyn_cora": 10,
    "dyn_acm": 10,
    "dyn_citeseer": 8,
    "patent": 10,
    "dyn_pubmed": 10,
    "arxivmath": 10,
}

BLUE = "#2F5D8A"
ORANGE = "#C5672A"
INK = "#252A2E"
MID = "#727A80"
LIGHT = "#D9DEE2"

def load_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload["records"]
    assert payload["record_count"] == len(records), path
    return records


def select(
    records: Iterable[dict[str, Any]],
    **criteria: Any,
) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if all(record.get(key) == value for key, value in criteria.items())
    ]


def only(records: list[dict[str, Any]], context: str) -> dict[str, Any]:
    assert len(records) == 1, f"{context}: expected one record, got {len(records)}"
    return records[0]


def avg(records: list[dict[str, Any]], key: str) -> float:
    assert records
    return mean(float(record[key]) for record in records)


def sd(records: list[dict[str, Any]], key: str) -> float:
    values = [float(record[key]) for record in records]
    return stdev(values) if len(values) > 1 else 0.0


def record_ids(records: Iterable[dict[str, Any]]) -> list[str]:
    return [record["bundle_record_id"] for record in records]


def fmt_time(value: float) -> str:
    if value >= 100:
        return f"{value:.1f}"
    if value >= 10:
        return f"{value:.2f}"
    return f"{value:.3f}"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def data_info() -> dict[str, dict[str, Any]]:
    cache: dict[Path, dict[str, Any]] = {}
    out: dict[str, dict[str, Any]] = {}
    for dataset, path in DATA_INFO.items():
        if path not in cache:
            cache[path] = json.loads(path.read_text(encoding="utf-8"))
        out[dataset] = cache[path][dataset]
    return out


def validate_short_horizon(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for dataset in DATASETS:
        common = {
            "measurement_family": "single_container_real_graph",
            "base_dataset": dataset,
            "batch_strategy": "999:10",
            "method": "leidenalg",
        }
        full = select(records, **common, mode="naive")
        local = select(
            records,
            **common,
            mode="smart",
            subcoms_depth="3",
            radius="1",
        )
        expected = EXPECTED_SHORT_COUNTS[dataset]
        assert len(full) == expected, (dataset, "full", len(full))
        assert len(local) == expected, (dataset, "local", len(local))
        expected_updates = EXPECTED_SHORT_UPDATES[dataset]
        assert {record["updates"] for record in full + local} == {
            expected_updates
        }
        assert all(len(record["series"]) == record["updates"] for record in full + local)

        full_time = avg(full, "total_time")
        local_time = avg(local, "total_time")
        out[dataset] = {
            "updates": expected_updates,
            "repetitions": expected,
            "full": {
                "q": avg(full, "final_modularity"),
                "nmi": avg(full, "final_nmi"),
                "time_mean": full_time,
                "time_sd": sd(full, "total_time"),
                "record_ids": record_ids(full),
            },
            "local": {
                "q": avg(local, "final_modularity"),
                "nmi": avg(local, "final_nmi"),
                "time_mean": local_time,
                "time_sd": sd(local, "total_time"),
                "record_ids": record_ids(local),
            },
            "delta_q": avg(local, "final_modularity")
            - avg(full, "final_modularity"),
            "delta_nmi": avg(local, "final_nmi") - avg(full, "final_nmi"),
            "speedup": full_time / local_time,
        }
    return out


def validate_backend_repeatability(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for dataset in ("dyn_pubmed", "arxivmath"):
        out[dataset] = {}
        for method in ("leidenalg", "dfleiden", "s2cag"):
            base_mode = "dynamic" if method == "dfleiden" else "naive"
            common = {
                "measurement_family": "single_container_real_graph",
                "base_dataset": dataset,
                "batch_strategy": "999:10",
                "method": method,
            }
            base = select(records, **common, mode=base_mode)
            local = select(
                records,
                **common,
                mode="smart",
                subcoms_depth="3",
                radius="1",
            )
            assert len(base) == len(local) == 5, (dataset, method)
            assert all(record["updates"] == 10 for record in base + local)
            if method == "s2cag":
                assert {record["feature_mode"] for record in base + local} == {
                    "dataset"
                }
                assert {str(record["iterations"]) for record in base + local} == {"10"}
            base_time = avg(base, "total_time")
            local_time = avg(local, "total_time")
            out[dataset][method] = {
                "base_mode": base_mode,
                "base": {
                    "q": avg(base, "final_modularity"),
                    "nmi": avg(base, "final_nmi"),
                    "time_mean": base_time,
                    "time_sd": sd(base, "total_time"),
                    "record_ids": record_ids(base),
                },
                "local": {
                    "q": avg(local, "final_modularity"),
                    "nmi": avg(local, "final_nmi"),
                    "time_mean": local_time,
                    "time_sd": sd(local, "total_time"),
                    "record_ids": record_ids(local),
                },
                "speedup": base_time / local_time,
            }
    return out


def validate_long_horizon(records: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for dataset in ("dyn_pubmed", "arxivmath"):
        out[dataset] = {}
        for method in ("leidenalg", "dfleiden"):
            base_mode = "dynamic" if method == "dfleiden" else "naive"
            common = {
                "measurement_family": "single_container_real_graph",
                "base_dataset": dataset,
                "batch_strategy": "9:500",
                "method": method,
                "force_undirected": True,
                "updates": 500,
            }
            base = only(select(records, **common, mode=base_mode), f"{dataset}/{method}/base")
            local = only(
                select(
                    records,
                    **common,
                    mode="smart",
                    subcoms_depth="3",
                    radius="1",
                ),
                f"{dataset}/{method}/local",
            )
            assert len(base["series"]) == len(local["series"]) == 500
            q_deltas = [
                float(local_row["modularity"]) - float(base_row["modularity"])
                for base_row, local_row in zip(base["series"], local["series"])
            ]
            worst_index = min(range(len(q_deltas)), key=q_deltas.__getitem__)
            out[dataset][method] = {
                "base_mode": base_mode,
                "base": {
                    "q": float(base["final_modularity"]),
                    "nmi": float(base["final_nmi"]),
                    "time": float(base["total_time"]),
                    "record_id": base["bundle_record_id"],
                },
                "local": {
                    "q": float(local["final_modularity"]),
                    "nmi": float(local["final_nmi"]),
                    "time": float(local["total_time"]),
                    "record_id": local["bundle_record_id"],
                },
                "delta_q": float(local["final_modularity"])
                - float(base["final_modularity"]),
                "delta_nmi": float(local["final_nmi"])
                - float(base["final_nmi"]),
                "speedup": float(base["total_time"]) / float(local["total_time"]),
                "trajectory_delta_q_mean": mean(q_deltas),
                "trajectory_delta_q_worst": q_deltas[worst_index],
                "trajectory_delta_q_worst_update": worst_index + 1,
                "series": {
                    "base_q": [float(row["modularity"]) for row in base["series"]],
                    "local_q": [float(row["modularity"]) for row in local["series"]],
                    "base_time": [float(row["time"]) for row in base["series"]],
                    "local_time": [float(row["time"]) for row in local["series"]],
                },
            }
    return out


DSBM_RE = re.compile(
    r"^dsbm-(?P<regime>random|hubs|community)-1024-58-mc"
    r"(?P<changes>290|1450)-(?P<seed>4[2-6])$"
)


def validate_dsbm(records: list[dict[str, Any]]) -> dict[str, Any]:
    dsbm = select(records, measurement_family="single_container_dsbm")
    assert len(dsbm) == 60
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for record in dsbm:
        match = DSBM_RE.match(record["base_dataset"])
        assert match, record["base_dataset"]
        assert record["batch_strategy"] == "0:99"
        assert record["updates"] == len(record["series"]) == 99
        grouped[(int(match["changes"]), match["regime"])].append(record)

    out: dict[str, Any] = {}
    for changes in (290, 1450):
        rate = "0.01" if changes == 290 else "0.05"
        out[rate] = {}
        for regime in ("random", "hubs", "community"):
            rows = grouped[(changes, regime)]
            assert len(rows) == 10
            by_seed: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
            for record in rows:
                match = DSBM_RE.match(record["base_dataset"])
                assert match
                by_seed[int(match["seed"])][record["mode"]] = record
            assert sorted(by_seed) == list(range(42, 47))

            paired = []
            for seed, modes in sorted(by_seed.items()):
                assert set(modes) == {"naive", "smart"}
                base, local = modes["naive"], modes["smart"]
                paired.append(
                    {
                        "seed": seed,
                        "speedup": float(base["total_time"])
                        / float(local["total_time"]),
                        "delta_q": float(local["final_modularity"])
                        - float(base["final_modularity"]),
                        "delta_nmi": float(local["final_nmi"])
                        - float(base["final_nmi"]),
                        "base_id": base["bundle_record_id"],
                        "local_id": local["bundle_record_id"],
                    }
                )
            speedups = [row["speedup"] for row in paired]
            out[rate][regime] = {
                "changed_edges_per_update": changes,
                "seeds": 5,
                "speedup_mean": mean(speedups),
                "speedup_sd": stdev(speedups),
                "speedup_min": min(speedups),
                "speedup_max": max(speedups),
                "worst_delta_q": min(row["delta_q"] for row in paired),
                "worst_delta_nmi": min(row["delta_nmi"] for row in paired),
                "paired": paired,
            }
    return out


def validate_workloads(
    records: list[dict[str, Any]],
    nodes: dict[str, int],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for dataset in ("dyn_cora", "dyn_pubmed", "arxivmath"):
        out[dataset] = {}
        for variant in ("full", "no_closure", "no_contraction"):
            record = only(
                select(
                    records,
                    measurement_family="single_container_workload_profiles",
                    base_dataset=dataset,
                    method="leidenalg",
                    mode="smart",
                    variant=variant,
                    subcoms_depth=3,
                    radius=1,
                ),
                f"workload/{dataset}/{variant}",
            )
            profile = record["profile"]
            assert record["aggregation"] == "sum"
            assert record["device"] == "cuda"
            rows = profile["rows"]
            assert len(rows) == record["updates_profiled"]
            contracted = [
                row["contracted_nodes_by_level"][-1] / nodes[dataset] * 100.0
                for row in rows
            ]
            closure = [
                row["closure_vertices_by_level"][-1] / nodes[dataset] * 100.0
                for row in rows
            ]
            out[dataset][variant] = {
                "updates": record["updates_profiled"],
                "q": float(record["final_modularity"]),
                "time": float(record["total_profiled_time"]),
                "contracted_nodes_pct_mean": mean(contracted),
                "closure_nodes_pct_mean": mean(closure),
                "record_id": record["bundle_record_id"],
            }
    return out


def validate_neighborhoods(records: list[dict[str, Any]]) -> dict[str, Any]:
    assert len(records) == 6
    out: dict[str, Any] = {}
    for dataset in DATASETS:
        record = only(
            select(
                records,
                measurement_family="article_neighborhood_growth",
                base_dataset=dataset,
            ),
            f"neighborhood/{dataset}",
        )
        assert len(record["neighborhood_series"]) == record["updates"]
        out[dataset] = {
            "updates": record["updates"],
            "affected_mean": float(record["B0_mean"]),
            "b1_pct_mean": float(record["B1_pct_mean"]),
            "b2_pct_mean": float(record["B2_pct_mean"]),
            "b3_pct_mean": float(record["B3_pct_mean"]),
            "b4_pct_mean": float(record["B4_pct_mean"]),
            "record_id": record["bundle_record_id"],
        }
    return out


def validate_topology_ablation(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for dataset in ("dyn_cora", "dyn_pubmed", "arxivmath"):
        full_records = select(
            records,
            measurement_family="single_container_real_graph",
            base_dataset=dataset,
            batch_strategy="999:10",
            method="leidenalg",
            mode="naive",
        )
        assert len(full_records) == EXPECTED_SHORT_COUNTS[dataset]
        full_time = avg(full_records, "total_time")
        out[dataset] = {
            "full": {
                "q": avg(full_records, "final_modularity"),
                "time": full_time,
                "record_ids": record_ids(full_records),
            },
            "points": [],
        }
        for level in range(1, 5):
            for radius in range(3):
                if (level, radius) == (3, 1):
                    family = "single_container_real_graph"
                elif dataset == "dyn_cora":
                    family = "article_completion_real_graph"
                else:
                    family = "single_container_real_graph"
                matches = select(
                    records,
                    measurement_family=family,
                    base_dataset=dataset,
                    batch_strategy="999:10",
                    method="leidenalg",
                    mode="smart",
                    subcoms_depth=str(level),
                    radius=str(radius),
                )
                expected = (
                    EXPECTED_SHORT_COUNTS[dataset]
                    if (level, radius) == (3, 1)
                    else 1
                )
                assert len(matches) == expected, (
                    dataset,
                    level,
                    radius,
                    family,
                    len(matches),
                )
                local_time = avg(matches, "total_time")
                out[dataset]["points"].append(
                    {
                        "level": level,
                        "radius": radius,
                        "q": avg(matches, "final_modularity"),
                        "time": local_time,
                        "speedup": full_time / local_time,
                        "repetitions": len(matches),
                        "record_ids": record_ids(matches),
                    }
                )
        assert len(out[dataset]["points"]) == 12
    return out


def validate_directed_control(records: list[dict[str, Any]]) -> dict[str, Any]:
    directed = select(
        records,
        measurement_family="single_container_directed_control",
        batch_strategy="999:10",
        method="leidenalg",
    )
    assert len(directed) == 4
    out: dict[str, Any] = {}
    for dataset in ("dyn_pubmed", "arxivmath"):
        full = only(
            select(directed, base_dataset=dataset, mode="naive"),
            f"directed/{dataset}/full",
        )
        local = only(
            select(
                directed,
                base_dataset=dataset,
                mode="smart",
                subcoms_depth="3",
                radius="1",
            ),
            f"directed/{dataset}/local",
        )
        assert full["force_undirected"] is local["force_undirected"] is False
        assert full["updates"] == local["updates"] == 10
        assert len(full["series"]) == len(local["series"]) == 10
        out[dataset] = {
            "updates": 10,
            "full": {
                "q": float(full["final_modularity"]),
                "nmi": float(full["final_nmi"]),
                "time": float(full["total_time"]),
                "record_id": full["bundle_record_id"],
            },
            "local": {
                "q": float(local["final_modularity"]),
                "nmi": float(local["final_nmi"]),
                "time": float(local["total_time"]),
                "record_id": local["bundle_record_id"],
            },
            "speedup": float(full["total_time"]) / float(local["total_time"]),
        }
    return out


def validate_gamma_sweep(records: list[dict[str, Any]]) -> dict[str, Any]:
    sweep = select(
        records,
        measurement_family="gamma_sweep",
        batch_strategy="999:50",
        method="leidenalg",
    )
    assert len(sweep) == 12
    expected_updates = {"dyn_pubmed": 44, "arxivmath": 50}
    out: dict[str, Any] = {}
    for dataset in ("dyn_pubmed", "arxivmath"):
        out[dataset] = {}
        for gamma in (0.5, 1.0, 2.0):
            stored_gamma: str | None = None if gamma == 1.0 else f"{gamma:g}"
            full = only(
                select(
                    sweep,
                    base_dataset=dataset,
                    mode="naive",
                    resolution=stored_gamma,
                ),
                f"gamma/{dataset}/{gamma}/full",
            )
            local = only(
                select(
                    sweep,
                    base_dataset=dataset,
                    mode="smart",
                    resolution=stored_gamma,
                    subcoms_depth="3",
                    radius="1",
                ),
                f"gamma/{dataset}/{gamma}/local",
            )
            updates = expected_updates[dataset]
            assert full["force_undirected"] is local["force_undirected"] is True
            assert full["updates"] == local["updates"] == updates
            assert len(full["series"]) == len(local["series"]) == updates
            full_q = float(full["final_modularity"])
            local_q = float(local["final_modularity"])
            full_time = float(full["total_time"])
            local_time = float(local["total_time"])
            out[dataset][str(gamma)] = {
                "updates": updates,
                "gamma": gamma,
                "full_q": full_q,
                "local_q": local_q,
                "delta_q": local_q - full_q,
                "full_nmi": float(full["final_nmi"]),
                "local_nmi": float(local["final_nmi"]),
                "full_time": full_time,
                "local_time": local_time,
                "speedup": full_time / local_time,
                "record_ids": [full["bundle_record_id"], local["bundle_record_id"]],
            }
    return out


def validate_feature_ablation(records: list[dict[str, Any]]) -> dict[str, Any]:
    rows = select(
        records,
        measurement_family="single_container_real_graph",
        batch_strategy="999:100",
        method="s2cag",
    )
    rows = [row for row in rows if row["base_dataset"] in {"dyn_cora", "dyn_pubmed"}]
    assert len(rows) == 12
    expected_updates = {"dyn_cora": 10, "dyn_pubmed": 44}
    out: dict[str, Any] = {}
    for dataset in ("dyn_cora", "dyn_pubmed"):
        out[dataset] = {}
        for feature_mode in ("dataset", "random", "onehot"):
            out[dataset][feature_mode] = {}
            for mode in ("naive", "smart"):
                criteria: dict[str, Any] = {
                    "base_dataset": dataset,
                    "feature_mode": feature_mode,
                    "mode": mode,
                }
                if mode == "smart":
                    criteria.update(subcoms_depth="3", radius="1")
                record = only(
                    select(rows, **criteria),
                    f"features/{dataset}/{feature_mode}/{mode}",
                )
                updates = expected_updates[dataset]
                assert record["updates"] == len(record["series"]) == updates
                assert str(record["iterations"]) == "10"
                assert record["force_undirected"] is (dataset == "dyn_pubmed")
                if mode == "smart":
                    assert record["aggregation"] == "norm"
                out[dataset][feature_mode][mode] = {
                    "updates": updates,
                    "q": float(record["final_modularity"]),
                    "nmi": float(record["final_nmi"]),
                    "time": float(record["total_time"]),
                    "record_id": record["bundle_record_id"],
                }
    return out


def validate_cut_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    assert len(records) == 4
    definitions = {record["cut_metrics"]["definition"] for record in records}
    assert len(definitions) == 1
    out: dict[str, Any] = {"definition": definitions.pop(), "datasets": {}}
    for dataset in ("dyn_pubmed", "arxivmath"):
        out["datasets"][dataset] = {}
        for mode in ("naive", "smart"):
            record = only(
                select(
                    records,
                    measurement_family="leiden_cut_metrics",
                    base_dataset=dataset,
                    batch_strategy="999:10",
                    method="leidenalg",
                    mode=mode,
                ),
                f"cut/{dataset}/{mode}",
            )
            assert record["force_undirected"] is True
            assert record["resolution"] == 1.0
            assert record["updates"] == 10
            if mode == "smart":
                assert record["smart_depth"] == 3 and record["smart_radius"] == 1
            metrics = record["cut_metrics"]
            assert metrics["num_nodes"] > 0
            assert metrics["num_clusters"] == metrics["nonzero_volume_clusters"]
            out["datasets"][dataset][mode] = {
                "q": float(record["final_modularity"]),
                "conductance_mean": float(metrics["conductance_mean"]),
                "ncut": float(metrics["ncut"]),
                "clusters": int(metrics["num_clusters"]),
                "record_id": record["bundle_record_id"],
            }
    return out


def emit_rows(
    info: dict[str, dict[str, Any]],
    short: dict[str, Any],
    repeatability: dict[str, Any],
    long_horizon: dict[str, Any],
    workloads: dict[str, Any],
    neighborhoods: dict[str, Any],
    dsbm: dict[str, Any],
    topology: dict[str, Any],
    directed: dict[str, Any],
    gamma_sweep: dict[str, Any],
    feature_ablation: dict[str, Any],
    cut_metrics: dict[str, Any],
) -> None:
    dataset_rows = []
    short_rows = []
    for dataset in DATASETS:
        meta = info[dataset]
        dataset_rows.append(
            f"{DISPLAY[dataset]} & {meta['n']:,} & {meta['m']:,} & "
            f"{STREAM_KIND[dataset]} & {short[dataset]['updates']} \\\\"
        )
        row = short[dataset]
        short_rows.append(
            f"{DISPLAY[dataset]} & {row['full']['q']:.4f} & "
            f"{row['local']['q']:.4f} & {row['full']['nmi']:.3f} & "
            f"{row['local']['nmi']:.3f} & {fmt_time(row['full']['time_mean'])} & "
            f"{fmt_time(row['local']['time_mean'])} & {row['speedup']:.1f}$\\times$ \\\\"
        )

    backend_rows = []
    backend_names = {
        "leidenalg": "Leiden",
        "dfleiden": "DF-Leiden",
        "s2cag": "S$^2$CAG",
    }
    for dataset in ("dyn_pubmed", "arxivmath"):
        for method in ("leidenalg", "dfleiden", "s2cag"):
            row = repeatability[dataset][method]
            base, local = row["base"], row["local"]
            backend_rows.append(
                f"{DISPLAY[dataset]} & {backend_names[method]} & "
                f"{base['q']:.3f}/{local['q']:.3f} & "
                f"{base['nmi']:.3f}/{local['nmi']:.3f} & "
                f"{fmt_time(base['time_mean'])} ({base['time_sd']:.3f})/"
                f"{fmt_time(local['time_mean'])} ({local['time_sd']:.3f}) & "
                f"{row['speedup']:.1f}$\\times$ \\\\"
            )

    long_rows = []
    for dataset in ("dyn_pubmed", "arxivmath"):
        for method in ("leidenalg", "dfleiden"):
            row = long_horizon[dataset][method]
            base, local = row["base"], row["local"]
            long_rows.append(
                f"{DISPLAY[dataset]} & {backend_names[method]} & "
                f"{base['q']:.3f}/{local['q']:.3f} & "
                f"{base['nmi']:.3f}/{local['nmi']:.3f} & "
                f"{fmt_time(base['time'])}/{fmt_time(local['time'])} & "
                f"{row['speedup']:.2f}$\\times$ \\\\"
            )

    workload_rows = []
    variant_names = {
        "full": "Full ComNetX",
        "no_closure": "Radius-only scope",
        "no_contraction": "Vertex-level graphs",
    }
    for dataset in ("dyn_cora", "dyn_pubmed", "arxivmath"):
        for variant in ("full", "no_closure", "no_contraction"):
            row = workloads[dataset][variant]
            workload_rows.append(
                f"{DISPLAY[dataset]} & {variant_names[variant]} & {row['updates']} & "
                f"{row['q']:.3f} & {fmt_time(row['time'])} & "
                f"{row['closure_nodes_pct_mean']:.2f} & "
                f"{row['contracted_nodes_pct_mean']:.2f} \\\\"
            )

    neighborhood_rows = []
    for dataset in DATASETS:
        row = neighborhoods[dataset]
        neighborhood_rows.append(
            f"{DISPLAY[dataset]} & {row['affected_mean']:.1f} & "
            f"{row['b1_pct_mean']:.2f} & {row['b2_pct_mean']:.2f} & "
            f"{row['b3_pct_mean']:.2f} & {row['b4_pct_mean']:.2f} \\\\"
        )

    dsbm_rows = []
    regime_names = {
        "random": "random",
        "hubs": "hub-centered",
        "community": "community-internal",
    }
    for rate in ("0.01", "0.05"):
        for regime in ("random", "hubs", "community"):
            row = dsbm[rate][regime]
            dsbm_rows.append(
                f"{row['changed_edges_per_update']} & {regime_names[regime]} & "
                f"{row['speedup_mean']:.2f} $\\pm$ {row['speedup_sd']:.3f} & "
                f"${row['worst_delta_q']:.4f}$ & "
                f"${row['worst_delta_nmi']:.4f}$ \\\\"
            )

    gamma_rows = []
    for dataset in ("dyn_pubmed", "arxivmath"):
        for gamma in (0.5, 1.0, 2.0):
            row = gamma_sweep[dataset][str(gamma)]
            gamma_rows.append(
                f"{DISPLAY[dataset]} & {gamma:g} & {row['updates']} & "
                f"{row['full_q']:.4f}/{row['local_q']:.4f} & "
                f"${row['delta_q']:.4f}$ & {row['speedup']:.1f}$\\times$ \\\\"
            )

    feature_rows = []
    feature_names = {
        "dataset": "dataset features",
        "random": "random features",
        "onehot": "one-hot features",
    }
    mode_names = {"naive": "Full", "smart": "ComNetX"}
    for feature_mode in ("dataset", "random", "onehot"):
        for mode in ("naive", "smart"):
            cora = feature_ablation["dyn_cora"][feature_mode][mode]
            pubmed = feature_ablation["dyn_pubmed"][feature_mode][mode]
            feature_rows.append(
                f"{mode_names[mode]}, {feature_names[feature_mode]} & "
                f"${cora['q']:.3f}$/{fmt_time(cora['time'])}/{cora['nmi']:.3f} & "
                f"${pubmed['q']:.3f}$/{fmt_time(pubmed['time'])}/{pubmed['nmi']:.3f} \\\\"
            )

    directed_rows = []
    for dataset in ("dyn_pubmed", "arxivmath"):
        row = directed[dataset]
        full, local = row["full"], row["local"]
        directed_rows.append(
            f"{DISPLAY[dataset]} & {full['q']:.3f}/{local['q']:.3f} & "
            f"{full['nmi']:.3f}/{local['nmi']:.3f} & "
            f"{fmt_time(full['time'])}/{fmt_time(local['time'])} & "
            f"{row['speedup']:.1f}$\\times$ \\\\"
        )

    cut_rows = []
    cut_mode_names = {"naive": "Full", "smart": "ComNetX"}
    for dataset in ("dyn_pubmed", "arxivmath"):
        for mode in ("naive", "smart"):
            row = cut_metrics["datasets"][dataset][mode]
            cut_rows.append(
                f"{DISPLAY[dataset]} & {cut_mode_names[mode]} & {row['q']:.3f} & "
                f"{row['clusters']:,} & {row['conductance_mean']:.6f} & "
                f"{row['ncut']:.3f} \\\\"
            )

    # Keep the closing rule inside each input file.  A file boundary between a
    # row terminator and booktabs' \bottomrule can make TeX reject \noalign.
    write_text(GENERATED / "dataset_rows.tex", "\n".join(dataset_rows) + "\n\\bottomrule")
    write_text(GENERATED / "short_results_rows.tex", "\n".join(short_rows) + "\n\\bottomrule")
    write_text(GENERATED / "backend_rows.tex", "\n".join(backend_rows) + "\n\\bottomrule")
    write_text(GENERATED / "long_horizon_rows.tex", "\n".join(long_rows) + "\n\\bottomrule")
    write_text(GENERATED / "workload_rows.tex", "\n".join(workload_rows) + "\n\\bottomrule")
    write_text(GENERATED / "neighborhood_rows.tex", "\n".join(neighborhood_rows) + "\n\\bottomrule")
    write_text(GENERATED / "dsbm_rows.tex", "\n".join(dsbm_rows) + "\n\\bottomrule")
    write_text(GENERATED / "gamma_rows.tex", "\n".join(gamma_rows) + "\n\\bottomrule")
    write_text(
        GENERATED / "feature_ablation_rows.tex",
        "\n".join(feature_rows) + "\n\\bottomrule",
    )
    write_text(GENERATED / "directed_rows.tex", "\n".join(directed_rows) + "\n\\bottomrule")
    write_text(GENERATED / "cut_metric_rows.tex", "\n".join(cut_rows) + "\n\\bottomrule")

    arxiv_short = short["arxivmath"]
    arxiv_long = long_horizon["arxivmath"]["leidenalg"]
    pubmed_topology = next(
        point
        for point in topology["dyn_pubmed"]["points"]
        if point["level"] == 2 and point["radius"] == 0
    )
    arxiv_topology = next(
        point
        for point in topology["arxivmath"]["points"]
        if point["level"] == 1 and point["radius"] == 0
    )
    macros = f"""
\\newcommand{{\\ArxivShortSpeedup}}{{{arxiv_short['speedup']:.1f}}}
\\newcommand{{\\ArxivShortFullQ}}{{{arxiv_short['full']['q']:.3f}}}
\\newcommand{{\\ArxivShortLocalQ}}{{{arxiv_short['local']['q']:.3f}}}
\\newcommand{{\\ArxivShortQGap}}{{{abs(arxiv_short['delta_q']):.3f}}}
\\newcommand{{\\ArxivLongSpeedup}}{{{arxiv_long['speedup']:.1f}}}
\\newcommand{{\\ArxivLongFullQ}}{{{arxiv_long['base']['q']:.3f}}}
\\newcommand{{\\ArxivLongLocalQ}}{{{arxiv_long['local']['q']:.3f}}}
\\newcommand{{\\ArxivLongQGap}}{{{abs(arxiv_long['delta_q']):.3f}}}
\\newcommand{{\\ArxivLongFullNMI}}{{{arxiv_long['base']['nmi']:.3f}}}
\\newcommand{{\\ArxivLongLocalNMI}}{{{arxiv_long['local']['nmi']:.3f}}}
\\newcommand{{\\DSBMHighRandomSpeedup}}{{{dsbm['0.05']['random']['speedup_mean']:.2f}}}
\\newcommand{{\\DSBMHighHubSpeedup}}{{{dsbm['0.05']['hubs']['speedup_mean']:.2f}}}
\\newcommand{{\\PubMedTopologyQ}}{{{pubmed_topology['q']:.4f}}}
\\newcommand{{\\PubMedTopologySpeedup}}{{{pubmed_topology['speedup']:.1f}}}
\\newcommand{{\\ArxivTopologyQ}}{{{arxiv_topology['q']:.4f}}}
\\newcommand{{\\ArxivTopologySpeedup}}{{{arxiv_topology['speedup']:.1f}}}
"""
    write_text(GENERATED / "metrics_macros.tex", macros)


def new_canvas(
    name: str,
    width_in: float,
    height_in: float,
    *,
    output_path: Path | None = None,
) -> tuple[canvas.Canvas, float, float]:
    """Create a deterministic vector PDF canvas in the article figure folder."""
    path = output_path if output_path is not None else FIGURES / name
    path.parent.mkdir(parents=True, exist_ok=True)
    width = width_in * 72.0
    height = height_in * 72.0
    pdf = canvas.Canvas(
        str(path),
        pagesize=(width, height),
        pageCompression=1,
        invariant=1,
        initialFontName=FIGURE_FONT,
    )
    pdf.setTitle("ComNetX validated evidence figure")
    pdf.setAuthor("ComNetX authors")
    pdf.setCreator("journal/ieee-access/analysis/validate_results.py")
    return pdf, width, height


def draw_text(
    pdf: canvas.Canvas,
    x: float,
    y: float,
    value: str,
    size: float = 7.0,
    font: str = FIGURE_FONT,
    align: str = "left",
    color: str = INK,
) -> None:
    pdf.setFillColor(HexColor(color))
    pdf.setFont(font, size)
    if align == "center":
        pdf.drawCentredString(x, y, value)
    elif align == "right":
        pdf.drawRightString(x, y, value)
    else:
        pdf.drawString(x, y, value)


def draw_backed_text(
    pdf: canvas.Canvas,
    x: float,
    y: float,
    value: str,
    size: float,
    *,
    font: str = FIGURE_FONT,
    align: str = "left",
    color: str = INK,
    pad_x: float = 1.2,
    pad_y: float = 0.7,
) -> None:
    """Draw a small opaque text halo so annotations stay clear of marks."""
    width = stringWidth(value, font, size)
    if align == "center":
        left = x - width / 2
    elif align == "right":
        left = x - width
    else:
        left = x
    pdf.saveState()
    pdf.setFillColor(HexColor("#FFFFFF"))
    pdf.setStrokeColor(HexColor("#FFFFFF"))
    pdf.roundRect(
        left - pad_x,
        y - size * 0.24 - pad_y,
        width + 2 * pad_x,
        size * 1.02 + 2 * pad_y,
        1.1,
        stroke=0,
        fill=1,
    )
    pdf.restoreState()
    draw_text(pdf, x, y, value, size, font, align, color)


def draw_vertical_text(
    pdf: canvas.Canvas,
    x: float,
    y: float,
    value: str,
    size: float = 6.3,
    font: str = FIGURE_FONT,
) -> None:
    """Draw a centered vertical axis label without overlapping tick labels."""
    pdf.saveState()
    pdf.translate(x, y)
    pdf.rotate(90)
    draw_text(pdf, 0, -size / 3, value, size=size, font=font, align="center")
    pdf.restoreState()


def wrapped_lines(value: str, font: str, size: float, max_width: float) -> list[str]:
    words = value.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if stringWidth(candidate, font, size) <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_wrapped(
    pdf: canvas.Canvas,
    x: float,
    y: float,
    value: str,
    max_width: float,
    size: float = 6.5,
    leading: float = 8.0,
    font: str = FIGURE_FONT,
    color: str = INK,
) -> None:
    for index, line in enumerate(wrapped_lines(value, font, size, max_width)):
        draw_text(pdf, x, y - index * leading, line, size=size, font=font, color=color)


def draw_marker(
    pdf: canvas.Canvas,
    x: float,
    y: float,
    shape: str,
    radius: float,
    stroke: str,
    fill: str | None,
) -> None:
    pdf.setStrokeColor(HexColor(stroke))
    pdf.setFillColor(HexColor(fill) if fill else HexColor("#FFFFFF"))
    pdf.setLineWidth(0.8)
    if shape == "circle":
        pdf.circle(x, y, radius, stroke=1, fill=1)
    elif shape == "square":
        pdf.rect(x - radius, y - radius, 2 * radius, 2 * radius, stroke=1, fill=1)
    elif shape == "triangle":
        path = pdf.beginPath()
        path.moveTo(x, y + radius)
        path.lineTo(x - radius, y - radius)
        path.lineTo(x + radius, y - radius)
        path.close()
        pdf.drawPath(path, stroke=1, fill=1)
    elif shape == "star":
        path = pdf.beginPath()
        for point in range(10):
            angle = math.pi / 2 + point * math.pi / 5
            r = radius if point % 2 == 0 else radius * 0.42
            px = x + math.cos(angle) * r
            py = y + math.sin(angle) * r
            if point == 0:
                path.moveTo(px, py)
            else:
                path.lineTo(px, py)
        path.close()
        pdf.drawPath(path, stroke=1, fill=1)
    else:
        raise ValueError(shape)


def draw_panel_axes(
    pdf: canvas.Canvas,
    x0: float,
    y0: float,
    width: float,
    height: float,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    xticks: list[float],
    yticks: list[float],
    xlog: bool = False,
    xformat: str = "{:.0f}",
    yformat: str = "{:.2f}",
    tick_size: float = 5.7,
) -> tuple[Any, Any]:
    def tx(value: float) -> float:
        if xlog:
            return x0 + (math.log10(value) - math.log10(xmin)) / (
                math.log10(xmax) - math.log10(xmin)
            ) * width
        return x0 + (value - xmin) / (xmax - xmin) * width

    def ty(value: float) -> float:
        return y0 + (value - ymin) / (ymax - ymin) * height

    pdf.setStrokeColor(HexColor(INK))
    pdf.setLineWidth(0.6)
    pdf.line(x0, y0, x0, y0 + height)
    pdf.line(x0, y0, x0 + width, y0)
    for value in yticks:
        y = ty(value)
        pdf.setStrokeColor(HexColor(LIGHT))
        pdf.setLineWidth(0.45)
        pdf.line(x0, y, x0 + width, y)
        draw_text(
            pdf,
            x0 - 3,
            y - tick_size * 0.38,
            yformat.format(value),
            tick_size,
            align="right",
        )
    for value in xticks:
        if value < xmin or value > xmax:
            continue
        x = tx(value)
        pdf.setStrokeColor(HexColor(INK))
        pdf.setLineWidth(0.45)
        pdf.line(x, y0, x, y0 - 2.5)
        draw_text(pdf, x, y0 - 10.5, xformat.format(value), tick_size, align="center")
    return tx, ty


def polyline(
    pdf: canvas.Canvas,
    points: list[tuple[float, float]],
    color: str,
    dashed: bool = False,
    width: float = 1.0,
) -> None:
    pdf.setStrokeColor(HexColor(color))
    pdf.setLineWidth(width)
    pdf.setDash(3, 2) if dashed else pdf.setDash()
    path = pdf.beginPath()
    for index, (x, y) in enumerate(points):
        if index == 0:
            path.moveTo(x, y)
        else:
            path.lineTo(x, y)
    pdf.drawPath(path, stroke=1, fill=0)
    pdf.setDash()


def plot_long_horizon(long_horizon: dict[str, Any]) -> None:
    pdf, page_width, page_height = new_canvas("long_horizon.pdf", 7.16, 2.35)
    left = 38.0
    bottom = 31.0
    gap = 36.0
    panel_width = (page_width - left - 16.0 - gap) / 2
    panel_height = page_height - bottom - 24.0
    prepared: dict[str, dict[str, list[float]]] = {}
    for dataset in ("dyn_pubmed", "arxivmath"):
        series = long_horizon[dataset]["leidenalg"]["series"]
        gaps = [local - base for local, base in zip(series["local_q"], series["base_q"])]
        cumulative_full: list[float] = []
        cumulative_local: list[float] = []
        full_sum = 0.0
        local_sum = 0.0
        for full_time, local_time in zip(series["base_time"], series["local_time"]):
            full_sum += full_time
            local_sum += local_time
            cumulative_full.append(full_sum)
            cumulative_local.append(local_sum)
        speed = [full / local for full, local in zip(cumulative_full, cumulative_local)]
        prepared[dataset] = {"gap": gaps, "speed": speed}

    gap_min = min(min(row["gap"]) for row in prepared.values())
    gap_max = max(max(row["gap"]) for row in prepared.values())
    gap_pad = max((gap_max - gap_min) * 0.08, 0.002)
    gap_ymin = gap_min - gap_pad
    gap_ymax = max(0.0, gap_max) + gap_pad
    gap_ticks = [gap_ymin + i * (gap_ymax - gap_ymin) / 4 for i in range(5)]
    x0 = left
    draw_text(pdf, x0 + panel_width / 2, page_height - 12, "Per-update modularity deviation", 7.5, FIGURE_FONT_BOLD, "center")
    tx_gap, ty_gap = draw_panel_axes(
        pdf,
        x0,
        bottom,
        panel_width,
        panel_height,
        1,
        500,
        gap_ymin,
        gap_ymax,
        [1, 100, 200, 300, 400, 500],
        gap_ticks,
        yformat="{:.3f}",
    )
    if gap_ymin <= 0 <= gap_ymax:
        pdf.setStrokeColor(HexColor(INK))
        pdf.setLineWidth(0.55)
        pdf.line(x0, ty_gap(0), x0 + panel_width, ty_gap(0))
    draw_text(pdf, x0 + panel_width / 2, 8, "Processed update batch", 6.3, align="center")
    draw_vertical_text(pdf, 7, bottom + panel_height / 2, "Local minus full modularity")

    speed_min = min(min(row["speed"]) for row in prepared.values())
    speed_max = max(max(row["speed"]) for row in prepared.values())
    speed_ymin = min(1.0, speed_min) * 0.95
    speed_ymax = speed_max * 1.06
    speed_ticks = [speed_ymin + i * (speed_ymax - speed_ymin) / 4 for i in range(5)]
    x1 = left + panel_width + gap
    draw_text(pdf, x1 + panel_width / 2, page_height - 12, "Cumulative processing-time ratio", 7.5, FIGURE_FONT_BOLD, "center")
    tx_speed, ty_speed = draw_panel_axes(
        pdf,
        x1,
        bottom,
        panel_width,
        panel_height,
        1,
        500,
        speed_ymin,
        speed_ymax,
        [1, 100, 200, 300, 400, 500],
        speed_ticks,
        yformat="{:.1f}",
    )
    if speed_ymin <= 1 <= speed_ymax:
        pdf.setStrokeColor(HexColor(INK))
        pdf.setLineWidth(0.55)
        pdf.setDash(2, 2)
        pdf.line(x1, ty_speed(1), x1 + panel_width, ty_speed(1))
        pdf.setDash()
    draw_text(pdf, x1 + panel_width / 2, 8, "Processed update batch", 6.3, align="center")
    draw_vertical_text(pdf, x1 - 25, bottom + panel_height / 2, "Full/local time")

    styles = {
        "dyn_pubmed": (BLUE, False, "dyn_pubmed"),
        "arxivmath": (ORANGE, True, "arxivmath"),
    }
    for dataset, (color, dashed, label) in styles.items():
        polyline(
            pdf,
            [(tx_gap(index), ty_gap(value)) for index, value in enumerate(prepared[dataset]["gap"], 1)],
            color,
            dashed,
        )
        polyline(
            pdf,
            [(tx_speed(index), ty_speed(value)) for index, value in enumerate(prepared[dataset]["speed"], 1)],
            color,
            dashed,
        )
    legend_y = page_height - 23
    legend_x = x1 + panel_width - 105
    for offset, dataset in enumerate(("dyn_pubmed", "arxivmath")):
        color, dashed, label = styles[dataset]
        y = legend_y - offset * 10
        polyline(pdf, [(legend_x, y), (legend_x + 15, y)], color, dashed, 1.0)
        draw_text(pdf, legend_x + 19, y - 2.2, label, 6.0)
    pdf.showPage()
    pdf.save()

def plot_dsbm(dsbm: dict[str, Any]) -> None:
    pdf, page_width, page_height = new_canvas("dsbm_speedup.pdf", 3.5, 2.65)
    categories = [
        (rate, regime)
        for rate in ("0.01", "0.05")
        for regime in ("random", "hubs", "community")
    ]
    left = 91.0
    right = 23.0
    bottom = 26.0
    top = 19.0
    plot_width = page_width - left - right
    plot_height = page_height - bottom - top
    xmin, xmax = 0.7, 4.65

    def tx(value: float) -> float:
        return left + (value - xmin) / (xmax - xmin) * plot_width

    pdf.setStrokeColor(HexColor(INK))
    pdf.setLineWidth(0.6)
    pdf.line(left, bottom, left + plot_width, bottom)
    for tick in (1, 2, 3, 4):
        x = tx(tick)
        pdf.setStrokeColor(HexColor(LIGHT))
        pdf.setLineWidth(0.45)
        pdf.line(x, bottom, x, bottom + plot_height)
        draw_text(pdf, x, bottom - 10, str(tick), 5.8, align="center")
    break_even = tx(1.0)
    pdf.setStrokeColor(HexColor(INK))
    pdf.setLineWidth(0.7)
    pdf.setDash(2, 2)
    pdf.line(break_even, bottom, break_even, bottom + plot_height)
    pdf.setDash()
    draw_text(pdf, break_even + 3, bottom + plot_height - 7, "break-even", 5.7)

    step = plot_height / 6
    regime_names = {
        "random": "random",
        "hubs": "hub-centered",
        "community": "community-internal",
    }
    for index, (rate, regime) in enumerate(categories):
        y = bottom + plot_height - step * (index + 0.5)
        row = dsbm[rate][regime]
        color = BLUE if rate == "0.01" else ORANGE
        shape = "circle" if rate == "0.01" else "square"
        label = f"{row['changed_edges_per_update']}  {regime_names[regime]}"
        draw_text(pdf, left - 5, y - 2.2, label, 6.0, align="right")
        pdf.setStrokeColor(HexColor(color))
        pdf.setLineWidth(1.2)
        pdf.line(tx(row["speedup_min"]), y, tx(row["speedup_max"]), y)
        draw_marker(
            pdf,
            tx(row["speedup_mean"]),
            y,
            shape,
            3.0,
            color,
            color if rate == "0.01" else None,
        )
        draw_text(pdf, tx(row["speedup_max"]) + 4, y - 2.2, f"{row['speedup_mean']:.2f}x", 5.8)
    draw_text(pdf, left + plot_width / 2, 7, "Full/Local cumulative processing time", 6.3, align="center")
    pdf.showPage()
    pdf.save()


def plot_topology(topology: dict[str, Any]) -> None:
    """Plot the measured radius/depth surface and its Pareto frontier."""
    pdf, page_width, page_height = new_canvas("topology_ablation.pdf", 7.16, 2.62)
    datasets = ("dyn_cora", "dyn_pubmed", "arxivmath")
    radius_style = {
        0: ("circle", "#1F77B4"),
        1: ("square", "#D95F02"),
        2: ("triangle", "#2CA02C"),
    }
    x_axes = {
        "dyn_cora": (0.0, 26.0, [1, 5, 10, 15, 20, 25], False),
        "dyn_pubmed": (0.85, 220.0, [1, 10, 100], True),
        "arxivmath": (0.85, 900.0, [1, 10, 100], True),
    }
    badge_offsets = {
        "dyn_cora": {(0, 3): (0.0, -10.0), (2, 4): (0.0, -10.0)},
        "dyn_pubmed": {
            (2, 1): (0.0, -10.0),
            (0, 3): (-10.0, 0.0),
            (2, 3): (0.0, 10.0),
        },
        "arxivmath": {
            (1, 2): (0.0, 10.0),
            (2, 3): (0.0, -10.0),
            (0, 4): (0.0, 10.0),
            (1, 4): (0.0, -10.0),
            (2, 4): (0.0, 10.0),
        },
    }

    def pareto_front(data: dict[str, Any]) -> list[tuple[float, float]]:
        candidates = [(1.0, float(data["full"]["q"]))]
        candidates.extend(
            (float(point["speedup"]), float(point["q"]))
            for point in data["points"]
        )
        front = []
        for speedup, quality in candidates:
            dominated = any(
                other_speedup >= speedup
                and other_quality >= quality
                and (other_speedup > speedup or other_quality > quality)
                for other_speedup, other_quality in candidates
            )
            if not dominated:
                front.append((speedup, quality))
        return sorted(front)

    left = 36.0
    right = 8.0
    gap = 30.0
    bottom = 36.0
    top = 27.0
    panel_width = (page_width - left - right - 2 * gap) / 3
    panel_height = page_height - bottom - top

    for panel, dataset in enumerate(datasets):
        data = topology[dataset]
        x0 = left + panel * (panel_width + gap)
        all_q = [data["full"]["q"]] + [point["q"] for point in data["points"]]
        qmin, qmax = min(all_q), max(all_q)
        qpad = max((qmax - qmin) * 0.11, 0.0007)
        ymin, ymax = qmin - qpad, qmax + qpad
        xmin, xmax, xticks, xlog = x_axes[dataset]
        yticks = [ymin + index * (ymax - ymin) / 4 for index in range(5)]
        tx, ty = draw_panel_axes(
            pdf,
            x0,
            bottom,
            panel_width,
            panel_height,
            xmin,
            xmax,
            ymin,
            ymax,
            xticks,
            yticks,
            xlog=xlog,
            xformat="{:.0f}x",
            yformat="{:.3f}",
            tick_size=6.3,
        )
        draw_text(
            pdf,
            x0 + panel_width / 2,
            page_height - 9.0,
            dataset,
            8.0,
            FIGURE_FONT_BOLD,
            "center",
        )

        frontier = pareto_front(data)
        polyline(
            pdf,
            [(tx(speedup), ty(quality)) for speedup, quality in frontier],
            INK,
            width=1.0,
        )

        full_x = tx(1.0)
        full_y = ty(data["full"]["q"])
        draw_marker(pdf, full_x, full_y, "star", 5.2, "#FFFFFF", INK)

        badges: list[tuple[float, float, str, str, float, str]] = []
        for point in data["points"]:
            shape, color = radius_style[point["radius"]]
            is_default = point["level"] == 3 and point["radius"] == 1
            px = tx(point["speedup"])
            py = ty(point["q"])
            dx, dy = badge_offsets[dataset].get(
                (int(point["radius"]), int(point["level"])),
                (0.0, 0.0),
            )
            bx, by = px + dx, py + dy
            if dx or dy:
                pdf.setStrokeColor(HexColor(color))
                pdf.setLineWidth(0.55)
                pdf.line(px, py, bx, by)
                draw_marker(pdf, px, py, "circle", 1.25, "#FFFFFF", color)
            radius = 5.4 if is_default else 5.5 if shape == "triangle" else 4.8
            badges.append((bx, by, shape, color, radius, str(point["level"])))

        # Draw all badges after anchors and leaders so no line can erase a digit.
        for bx, by, shape, color, radius, _ in badges:
            draw_marker(
                pdf,
                bx,
                by,
                shape,
                radius,
                INK if shape == "square" and radius > 5.0 else "#FFFFFF",
                color,
            )
        for bx, by, _, _, _, label in badges:
            draw_text(
                pdf,
                bx,
                by - 2.8,
                label,
                7.6,
                FIGURE_FONT_BOLD,
                "center",
                "#FFFFFF",
            )
        draw_backed_text(
            pdf,
            full_x + 7.0,
            full_y + 3.5,
            "Full",
            6.4,
            color=INK,
        )
        draw_text(pdf, x0 + panel_width / 2, 10, "Full/ComNetX time", 6.8, align="center")

        if dataset == "arxivmath":
            legend_w = 40.0
            legend_h = 47.0
            legend_x = x0 + panel_width - legend_w - 2.0
            legend_y = bottom + panel_height - legend_h - 2.0
            pdf.setFillColor(HexColor("#FFFFFF"))
            pdf.setStrokeColor(HexColor("#C7CDD2"))
            pdf.setLineWidth(0.55)
            pdf.roundRect(legend_x, legend_y, legend_w, legend_h, 2.5, stroke=1, fill=1)
            legend_entries = [
                ("circle", "#1F77B4", "r=0"),
                ("square", "#D95F02", "r=1"),
                ("triangle", "#2CA02C", "r=2"),
            ]
            for index, (shape, color, label) in enumerate(legend_entries):
                row_y = legend_y + legend_h - 10.0 - index * 10.0
                draw_marker(pdf, legend_x + 8.0, row_y + 1.0, shape, 2.8, "#FFFFFF", color)
                draw_text(pdf, legend_x + 15.0, row_y - 1.2, label, 6.2)
            pareto_y = legend_y + 6.0
            pdf.setStrokeColor(HexColor(INK))
            pdf.setLineWidth(1.0)
            pdf.line(legend_x + 4.5, pareto_y + 1.5, legend_x + 11.5, pareto_y + 1.5)
            draw_text(pdf, legend_x + 15.0, pareto_y - 0.8, "Pareto", 6.2)

    draw_vertical_text(pdf, 7, bottom + panel_height / 2, "Final modularity", 6.8)
    pdf.showPage()
    pdf.save()


def plot_method_pipeline() -> None:
    """Draw a source-faithful, fully vector overview of one ComNetX update."""
    temporary = tempfile.TemporaryDirectory(prefix="comnetx-method-pipeline-")
    temporary_path = Path(temporary.name)
    base_path = temporary_path / "method_pipeline_base.pdf"
    pdf, page_width, page_height = new_canvas(
        "method_pipeline_base.pdf",
        7.16,
        4.56,
        output_path=base_path,
    )
    formula_nodes: list[dict[str, Any]] = []
    pdf.setLineCap(1)
    pdf.setLineJoin(1)

    indigo = "#2B2085"
    indigo_light = "#51479B"
    ink = "#1E2028"
    muted = "#5B5D69"
    hairline = "#B9B8CC"
    faint = "#F8F7FC"
    coral = "#F36F60"
    coral_light = "#FFF0EC"
    coral_neighbor = "#F8B3A9"
    cut_red = "#DF4F49"
    purple = "#8975BA"
    purple_light = "#F1EDF8"
    blue = "#63A3D4"
    blue_light = "#EDF5FA"
    teal = "#4DA6A0"
    teal_light = "#ECF7F5"
    olive = "#A7BA70"
    olive_light = "#F3F6EA"
    amber = "#DEB45E"
    amber_light = "#FBF5E9"
    slate = "#8191AA"
    slate_light = "#F0F2F6"
    mauve = "#A96F99"
    mauve_light = "#F7EDF4"
    community_colors = (purple, blue, teal, olive, amber, slate)
    community_styles = {
        "c1": (purple, purple_light),
        "c2": (blue, blue_light),
        "c3": (olive, olive_light),
        "c4": (teal, teal_light),
        "c5": (amber, amber_light),
        "c6": (slate, slate_light),
        "c7": (mauve, mauve_light),
    }
    # Three illustrative levels are enough to show the recursive structure.
    # Each level-1 child owns its own palette at level 2: purple descendants
    # remain purple, blue descendants remain blue, and so on.  Keeping this
    # relationship explicit prevents the visually plausible but semantically
    # wrong palette cycling that obscured ancestry in the previous drawing.
    hierarchy_families = (
        {
            "name": "cool",
            "span": (0.05, 0.31),
            "level_colors": (
                ("#7064A8",),
                ("#7653A7", "#4F86BE"),
                ("#5D3E91", "#9A72BF", "#2F6EA7", "#73A9D1"),
            ),
        },
        {
            "name": "warm",
            "span": (0.37, 0.64),
            "level_colors": (
                ("#D48E3F",),
                ("#DDA842", "#C86E38"),
                ("#F0C55B", "#CC902B", "#E0843F", "#AC5434"),
            ),
            # The local recomputation splits the warm community already at
            # level 0.  Every later row then refines each of those two roots
            # within its own inherited colour family (2 -> 4 -> 8).
            "updated_level_colors": (
                ("#DDA842", "#B85C49"),
                ("#F0C55B", "#CC902B", "#E0843F", "#AC5434"),
                (
                    "#F7D97B",
                    "#E8B944",
                    "#D8A13A",
                    "#B97A28",
                    "#F0A05A",
                    "#D97838",
                    "#C56857",
                    "#91403F",
                ),
            ),
        },
        {
            "name": "green",
            "span": (0.70, 0.96),
            "level_colors": (
                ("#52957F",),
                ("#369A96", "#72A457"),
                ("#287E8E", "#63B7AC", "#568B43", "#9AB968"),
            ),
        },
    )

    def frame(
        x: float,
        y: float,
        width: float,
        height: float,
        *,
        fill: str = "#FFFFFF",
        stroke: str = indigo,
        radius: float = 2.4,
        line_width: float = 0.65,
    ) -> None:
        pdf.setFillColor(HexColor(fill))
        pdf.setStrokeColor(HexColor(stroke))
        pdf.setLineWidth(line_width)
        pdf.roundRect(x, y, width, height, radius, stroke=1, fill=1)

    def rule(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        color: str = hairline,
        width: float = 0.45,
        dash: tuple[float, float] | None = None,
    ) -> None:
        pdf.setStrokeColor(HexColor(color))
        pdf.setLineWidth(width)
        if dash:
            pdf.setDash(list(dash), 0)
        else:
            pdf.setDash()
        pdf.line(x1, y1, x2, y2)
        pdf.setDash()

    def arrow(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        color: str = indigo,
        width: float = 0.9,
        head: float = 3.7,
    ) -> None:
        pdf.setStrokeColor(HexColor(color))
        pdf.setFillColor(HexColor(color))
        pdf.setLineWidth(width)
        pdf.setDash()
        pdf.line(x1, y1, x2, y2)
        angle = math.atan2(y2 - y1, x2 - x1)
        path = pdf.beginPath()
        path.moveTo(x2, y2)
        path.lineTo(
            x2 - head * math.cos(angle - 0.5),
            y2 - head * math.sin(angle - 0.5),
        )
        path.lineTo(
            x2 - head * math.cos(angle + 0.5),
            y2 - head * math.sin(angle + 0.5),
        )
        path.close()
        pdf.drawPath(path, stroke=0, fill=1)

    def smooth_closed_path(
        points: tuple[tuple[float, float], ...],
        tension: float = 0.75,
    ) -> Any:
        """Convert a closed Catmull-Rom contour to deterministic cubic Beziers."""
        path = pdf.beginPath()
        path.moveTo(*points[0])
        count = len(points)
        for index in range(count):
            p0 = points[(index - 1) % count]
            p1 = points[index]
            p2 = points[(index + 1) % count]
            p3 = points[(index + 2) % count]
            c1 = (
                p1[0] + tension * (p2[0] - p0[0]) / 6.0,
                p1[1] + tension * (p2[1] - p0[1]) / 6.0,
            )
            c2 = (
                p2[0] - tension * (p3[0] - p1[0]) / 6.0,
                p2[1] - tension * (p3[1] - p1[1]) / 6.0,
            )
            path.curveTo(c1[0], c1[1], c2[0], c2[1], p2[0], p2[1])
        path.close()
        return path

    def blob_anchors(
        cx: float,
        cy: float,
        rx: float,
        ry: float,
        *,
        angle: float = 0.0,
        variant: int = 0,
    ) -> tuple[tuple[float, float], ...]:
        patterns = (
            ((1.00, 0.02), (0.72, 0.78), (0.02, 1.00), (-0.73, 0.76),
             (-1.00, -0.03), (-0.70, -0.78), (-0.02, -0.98), (0.72, -0.75)),
            ((1.00, -0.02), (0.68, 0.82), (-0.03, 0.96), (-0.76, 0.73),
             (-0.98, 0.04), (-0.66, -0.82), (0.04, -1.00), (0.76, -0.70)),
            ((0.98, 0.04), (0.75, 0.73), (0.00, 1.00), (-0.69, 0.81),
             (-1.00, -0.02), (-0.74, -0.74), (0.02, -0.96), (0.68, -0.82)),
        )
        local = patterns[variant % len(patterns)]
        cosine, sine = math.cos(angle), math.sin(angle)
        result = []
        for px, py in local:
            dx, dy = px * rx, py * ry
            result.append((cx + dx * cosine - dy * sine, cy + dx * sine + dy * cosine))
        return tuple(result)

    def community_contour(
        anchors: tuple[tuple[float, float], ...],
        *,
        stroke: str,
        fill: str,
        fill_alpha: float = 0.74,
        stroke_alpha: float = 0.82,
        width: float = 0.38,
        dash: tuple[float, float] | None = (1.25, 0.95),
    ) -> None:
        pdf.saveState()
        pdf.setFillColor(HexColor(fill))
        pdf.setStrokeColor(HexColor(stroke))
        pdf.setFillAlpha(fill_alpha)
        pdf.setStrokeAlpha(stroke_alpha)
        pdf.setLineWidth(width)
        if dash:
            pdf.setDash(list(dash), 0)
        pdf.drawPath(smooth_closed_path(anchors, tension=0.84), stroke=1, fill=1)
        pdf.restoreState()

    def rounded_hull_contour(
        points: tuple[tuple[float, float], ...],
        *,
        radius: float,
        stroke: str,
        fill: str,
        fill_alpha: float,
    ) -> None:
        """Draw the exact circular offset of a convex hull of node centers."""

        def cross(
            origin: tuple[float, float],
            first: tuple[float, float],
            second: tuple[float, float],
        ) -> float:
            return (
                (first[0] - origin[0]) * (second[1] - origin[1])
                - (first[1] - origin[1]) * (second[0] - origin[0])
            )

        unique = sorted(set(points))
        if len(unique) < 3:
            contour_around(
                tuple(unique),
                stroke=stroke,
                fill=fill,
                padding_x=radius,
                padding_y=radius,
            )
            return
        lower: list[tuple[float, float]] = []
        for point in unique:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
                lower.pop()
            lower.append(point)
        upper: list[tuple[float, float]] = []
        for point in reversed(unique):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
                upper.pop()
            upper.append(point)
        hull = lower[:-1] + upper[:-1]

        corners: list[
            tuple[
                tuple[float, float],
                tuple[float, float],
                float,
                float,
            ]
        ] = []
        for index, point in enumerate(hull):
            previous = hull[(index - 1) % len(hull)]
            following = hull[(index + 1) % len(hull)]
            incoming = (point[0] - previous[0], point[1] - previous[1])
            outgoing = (following[0] - point[0], following[1] - point[1])
            incoming_length = math.hypot(*incoming)
            outgoing_length = math.hypot(*outgoing)
            incoming_normal = (
                incoming[1] / incoming_length,
                -incoming[0] / incoming_length,
            )
            outgoing_normal = (
                outgoing[1] / outgoing_length,
                -outgoing[0] / outgoing_length,
            )
            start_angle = math.atan2(incoming_normal[1], incoming_normal[0])
            end_angle = math.atan2(outgoing_normal[1], outgoing_normal[0])
            while end_angle <= start_angle:
                end_angle += 2.0 * math.pi
            start = (
                point[0] + incoming_normal[0] * radius,
                point[1] + incoming_normal[1] * radius,
            )
            end = (
                point[0] + outgoing_normal[0] * radius,
                point[1] + outgoing_normal[1] * radius,
            )
            corners.append((start, end, start_angle, end_angle))

        path = pdf.beginPath()
        path.moveTo(*corners[0][0])
        for index, (start, end, start_angle, end_angle) in enumerate(corners):
            if index:
                path.lineTo(*start)
            sweep = end_angle - start_angle
            segments = max(1, math.ceil(sweep / (math.pi / 2.0)))
            segment_angle = sweep / segments
            angle = start_angle
            current = start
            center = hull[index]
            for _ in range(segments):
                next_angle = angle + segment_angle
                target = (
                    center[0] + math.cos(next_angle) * radius,
                    center[1] + math.sin(next_angle) * radius,
                )
                factor = 4.0 / 3.0 * math.tan(segment_angle / 4.0)
                control_1 = (
                    current[0] - math.sin(angle) * radius * factor,
                    current[1] + math.cos(angle) * radius * factor,
                )
                control_2 = (
                    target[0] + math.sin(next_angle) * radius * factor,
                    target[1] - math.cos(next_angle) * radius * factor,
                )
                path.curveTo(*control_1, *control_2, *target)
                current = target
                angle = next_angle
            next_start = corners[(index + 1) % len(corners)][0]
            path.lineTo(*next_start)
        path.close()

        pdf.saveState()
        pdf.setFillColor(HexColor(fill))
        pdf.setStrokeColor(HexColor(stroke))
        pdf.setFillAlpha(fill_alpha)
        pdf.setStrokeAlpha(0.80)
        pdf.setLineWidth(0.29)
        pdf.setDash([1.15, 1.35], 0)
        pdf.drawPath(path, stroke=1, fill=1)
        pdf.restoreState()

    def concave_neighborhood_contour(cx: float, cy: float, scale: float) -> None:
        """Draw the source-style lobed envelope around the expanded region."""
        relative = (
            (-1.20, 1.08), (-0.84, 1.15), (-0.42, 1.01),
            (0.18, 1.03), (0.62, 1.07), (0.86, 1.15),
            (1.04, 0.88), (0.98, 0.52), (1.05, 0.15),
            (0.92, -0.10), (0.66, -0.13), (0.50, -0.36),
            (0.35, -0.60), (0.15, -0.90), (-0.18, -1.00),
            (-0.52, -0.88), (-0.90, -1.00), (-1.16, -0.85),
            (-1.22, -0.50), (-1.45, -0.25), (-1.45, 0.12),
            (-1.28, 0.40), (-1.22, 0.75),
        )
        anchors = tuple(
            (cx + x_offset * scale, cy + y_offset * scale)
            for x_offset, y_offset in relative
        )
        pdf.saveState()
        pdf.setFillColor(HexColor(coral_light))
        pdf.setStrokeColor(HexColor("#EE9B91"))
        pdf.setFillAlpha(0.21)
        pdf.setStrokeAlpha(0.80)
        pdf.setLineWidth(0.29)
        pdf.setDash([1.15, 1.35], 0)
        pdf.drawPath(
            smooth_closed_path(anchors, tension=0.86),
            stroke=1,
            fill=1,
        )
        pdf.restoreState()

    def contour_around(
        members: tuple[tuple[float, float], ...],
        *,
        stroke: str,
        fill: str,
        padding_x: float = 5.0,
        padding_y: float = 4.2,
        variant: int = 0,
    ) -> None:
        """Draw a smooth peanut envelope that enters the inter-node gap."""
        del variant
        pdf.saveState()
        pdf.setFillColor(HexColor(fill))
        pdf.setStrokeColor(HexColor(stroke))
        pdf.setFillAlpha(0.14)
        pdf.setStrokeAlpha(0.93)
        pdf.setLineWidth(0.31)
        pdf.setDash([1.15, 1.35], 0)
        if len(members) == 1:
            cx, cy = members[0]
            pdf.ellipse(
                cx - padding_x,
                cy - padding_y,
                cx + padding_x,
                cy + padding_y,
                stroke=1,
                fill=1,
            )
        else:
            first, second = members[0], members[-1]
            cx = (first[0] + second[0]) / 2.0
            cy = (first[1] + second[1]) / 2.0
            angle = math.atan2(second[1] - first[1], second[0] - first[0])
            length = math.hypot(second[0] - first[0], second[1] - first[1])
            half_span = length / 2.0
            waist = 0.68 * padding_y
            local = (
                (-half_span - padding_x, 0.0),
                (-half_span - 0.75 * padding_x, 0.68 * padding_y),
                (-half_span, padding_y),
                (-0.36 * half_span, 0.84 * padding_y),
                (0.0, waist),
                (0.36 * half_span, 0.84 * padding_y),
                (half_span, padding_y),
                (half_span + 0.75 * padding_x, 0.68 * padding_y),
                (half_span + padding_x, 0.0),
                (half_span + 0.75 * padding_x, -0.68 * padding_y),
                (half_span, -padding_y),
                (0.36 * half_span, -0.84 * padding_y),
                (0.0, -waist),
                (-0.36 * half_span, -0.84 * padding_y),
                (-half_span, -padding_y),
                (-half_span - 0.75 * padding_x, -0.68 * padding_y),
            )
            cosine, sine = math.cos(angle), math.sin(angle)
            anchors = tuple(
                (
                    cx + local_x * cosine - local_y * sine,
                    cy + local_x * sine + local_y * cosine,
                )
                for local_x, local_y in local
            )
            pdf.drawPath(
                smooth_closed_path(anchors, tension=0.86),
                stroke=1,
                fill=1,
            )
        pdf.restoreState()

    def soft_circle(
        x: float,
        y: float,
        radius: float,
        *,
        fill: str,
        stroke: str,
        line_width: float = 0.48,
    ) -> None:
        pdf.saveState()
        pdf.setFillColor(HexColor("#514B70"))
        pdf.setFillAlpha(0.10)
        pdf.circle(x + 0.55, y - 0.55, radius + 0.25, stroke=0, fill=1)
        pdf.restoreState()
        pdf.setFillColor(HexColor(fill))
        pdf.setStrokeColor(HexColor(stroke))
        pdf.setLineWidth(line_width)
        pdf.circle(x, y, radius, stroke=1, fill=1)
        pdf.saveState()
        pdf.setFillColor(HexColor("#FFFFFF"))
        pdf.setFillAlpha(0.24)
        pdf.ellipse(
            x - radius * 0.52,
            y + radius * 0.10,
            x + radius * 0.18,
            y + radius * 0.62,
            stroke=0,
            fill=1,
        )
        pdf.restoreState()

    def formula(
        x: float,
        y: float,
        tex: str,
        size: float = 5.1,
        *,
        align: str = "left",
        color: str = ink,
    ) -> None:
        """Queue native TeX math for composition after the vector base is drawn."""
        if size <= 0:
            raise ValueError("Formula size must be positive")
        formula_nodes.append(
            {
                "x": x,
                "y": y,
                "tex": tex,
                "scale": size / 10.0,
                "align": align,
                "color": color,
            }
        )

    def step_header(number: int, title: str, x: float, y: float) -> None:
        pdf.setFillColor(HexColor(indigo))
        pdf.setStrokeColor(HexColor(indigo))
        pdf.circle(x + 6.3, y + 0.8, 5.8, stroke=0, fill=1)
        draw_text(
            pdf,
            x + 6.3,
            y - 1.6,
            str(number),
            6.7,
            FIGURE_FONT_BOLD,
            "center",
            "#FFFFFF",
        )
        draw_text(pdf, x + 16.0, y - 1.1, title, 7.4, FIGURE_FONT_BOLD, color=indigo)

    def sparse_matrix(x: float, y: float, size: float) -> None:
        cells = 8
        cell = size / cells
        entries = {
            (0, 2), (0, 6),
            (1, 0), (1, 3),
            (2, 4), (2, 7),
            (3, 2), (3, 5),
            (4, 1), (4, 6),
            (5, 0), (5, 4), (5, 7),
            (6, 2), (6, 5),
            (7, 1), (7, 3),
        }
        pdf.setStrokeColor(HexColor("#D4D2E4"))
        pdf.setLineWidth(0.28)
        for row in range(cells):
            for column in range(cells):
                pdf.setFillColor(HexColor(indigo if (row, column) in entries else "#FFFFFF"))
                pdf.rect(
                    x + column * cell,
                    y + (cells - row - 1) * cell,
                    cell,
                    cell,
                    stroke=1,
                    fill=1,
                )

    def feature_matrix(x: float, y: float, width: float, height: float) -> None:
        rows, columns = 6, 5
        cell_w, cell_h = width / columns, height / rows
        values = (
            (0, 2, 1, 3, 1),
            (2, 1, 3, 0, 2),
            (1, 3, 0, 2, 3),
            (3, 0, 2, 1, 0),
            (0, 1, 3, 2, 1),
            (2, 3, 1, 0, 3),
        )
        shades = ("#F4F1FA", "#DCD5ED", "#BBAEDB", indigo)
        pdf.setStrokeColor(HexColor("#D4D2E4"))
        pdf.setLineWidth(0.28)
        for row in range(rows):
            for column in range(columns):
                pdf.setFillColor(HexColor(shades[values[row][column]]))
                pdf.rect(
                    x + column * cell_w,
                    y + (rows - row - 1) * cell_h,
                    cell_w,
                    cell_h,
                    stroke=1,
                    fill=1,
                )

    HierarchyGroup = tuple[str, str, float, float]

    def family_level_groups(
        family: dict[str, Any],
        level: int,
        *,
        updated: bool,
    ) -> list[HierarchyGroup]:
        """Return one level of an explicit recursive refinement tree."""
        start, end = family["span"]
        color_levels = (
            family.get("updated_level_colors", family["level_colors"])
            if updated
            else family["level_colors"]
        )
        colors = color_levels[level]
        parts = len(colors)
        local_boundaries = tuple(index / parts for index in range(parts + 1))
        boundaries = tuple(start + value * (end - start) for value in local_boundaries)

        groups: list[HierarchyGroup] = []
        for index, (left, right, color) in enumerate(
            zip(boundaries[:-1], boundaries[1:], colors)
        ):
            inset = 0.0025 if level else 0.0
            identity = f"{family['name']}:{level}:{index}"
            groups.append((identity, color, left + inset, right - inset))
        return groups

    def build_hierarchy_levels(*, updated: bool = False) -> list[list[HierarchyGroup]]:
        return [
            [
                group
                for family in hierarchy_families
                for group in family_level_groups(family, level, updated=updated)
            ]
            for level in range(3)
        ]

    current_hierarchy = build_hierarchy_levels()
    updated_hierarchy = build_hierarchy_levels(updated=True)

    def stored_hierarchy_tensor(x: float, y: float, width: float) -> None:
        """Draw one contiguous level-by-vertex matrix with stored IDs."""
        label_width = 20.5
        data_width = width - label_width
        columns = 12
        cell_width = data_width / columns
        level_height = 9.0
        omitted_height = 5.0
        header_height = 8.0
        total_height = 3 * level_height + 2 * omitted_height + header_height
        grid_x = x + label_width
        level_text_size = 4.7
        cell_text_size = 3.55
        tick_text_size = 3.8

        rows = (
            (
                0.0,
                r"0",
                ["#7064A8"] * 4 + ["#D48E3F"] * 4 + ["#52957F"] * 4,
                ["1"] * 4 + ["2"] * 4 + ["3"] * 4,
            ),
            (
                level_height + omitted_height,
                r"i",
                ["#7653A7"] * 2
                + ["#4F86BE"] * 2
                + ["#DDA842"] * 2
                + ["#C86E38"] * 2
                + ["#369A96"] * 2
                + ["#72A457"] * 2,
                [str(value) for value in range(1, 7) for _ in range(2)],
            ),
            (
                2 * (level_height + omitted_height),
                r"L\!-\!1",
                [
                    "#5D3E91",
                    "#9A72BF",
                    "#2F6EA7",
                    "#73A9D1",
                    "#F0C55B",
                    "#CC902B",
                    "#E0843F",
                    "#AC5434",
                    "#287E8E",
                    "#63B7AC",
                    "#568B43",
                    "#9AB968",
                ],
                [str(value) for value in range(1, 13)],
            ),
        )
        light_cells = {"#F0C55B", "#73A9D1", "#63B7AC", "#9AB968"}

        # Header and omitted rows are part of the same rectangular grid.
        header_y = y + 3 * level_height + 2 * omitted_height
        for band_y, band_height in (
            (y + level_height, omitted_height),
            (y + 2 * level_height + omitted_height, omitted_height),
            (header_y, header_height),
        ):
            pdf.setFillColor(HexColor(faint))
            pdf.rect(x, band_y, width, band_height, stroke=0, fill=1)

        for row_offset, label, colors, values in rows:
            row_y = y + row_offset
            for column, (fill_color, value) in enumerate(zip(colors, values)):
                cell_x = grid_x + column * cell_width
                pdf.setFillColor(HexColor(fill_color))
                pdf.setStrokeColor(HexColor("#FFFFFF"))
                pdf.setLineWidth(0.42)
                pdf.rect(cell_x, row_y, cell_width, level_height, stroke=1, fill=1)
                value_size = (
                    cell_text_size
                    if len(value) == 1
                    else 1.95
                )
                formula(
                    cell_x + cell_width / 2.0,
                    row_y + (2.55 if len(value) == 1 else 2.85),
                    value,
                    value_size,
                    align="center",
                    color=ink if fill_color in light_cells else "#FFFFFF",
                )
            formula(
                x + label_width / 2.0,
                row_y + 2.25,
                label,
                level_text_size,
                align="center",
                color=ink,
            )

        # Continuous column grid in the non-coloured header and omitted rows.
        for column in range(columns + 1):
            column_x = grid_x + column * cell_width
            for band_y, band_height in (
                (y + level_height, omitted_height),
                (y + 2 * level_height + omitted_height, omitted_height),
                (header_y, header_height),
            ):
                rule(
                    column_x,
                    band_y,
                    column_x,
                    band_y + band_height,
                    color="#D4D2E4",
                    width=0.28,
                )

        # Row separators, label/data divider, and one shared outer contour.
        row_boundaries = (
            level_height,
            level_height + omitted_height,
            2 * level_height + omitted_height,
            2 * (level_height + omitted_height),
            3 * level_height + 2 * omitted_height,
        )
        for boundary in row_boundaries:
            rule(x, y + boundary, x + width, y + boundary, color="#928DAA", width=0.38)
        rule(grid_x, y, grid_x, y + total_height, color="#928DAA", width=0.55)
        pdf.setFillColor(HexColor("#FFFFFF"))
        pdf.setStrokeColor(HexColor("#928DAA"))
        pdf.setLineWidth(0.55)
        pdf.rect(x, y, width, total_height, stroke=1, fill=0)

        # Header and omitted-level notation are also native TeX objects.
        formula(
            x + label_width / 2.0,
            header_y + 2.0,
            r"\ell\backslash v",
            tick_text_size,
            align="center",
            color=muted,
        )
        for label, column in ((r"1", 0), (r"2", 1), (r"\cdots", 5), (r"n", 11)):
            formula(
                grid_x + (column + 0.5) * cell_width,
                header_y + 2.0,
                label,
                tick_text_size,
                align="center",
                color=muted,
            )
        for band_y in (y + level_height, y + 2 * level_height + omitted_height):
            for center_x in (
                x + label_width / 2.0,
                grid_x + 0.5 * cell_width,
                grid_x + 5.5 * cell_width,
                grid_x + 11.5 * cell_width,
            ):
                pdf.setFillColor(HexColor(muted))
                for offset in (0.8, 2.5, 4.2):
                    pdf.circle(center_x, band_y + offset, 0.24, stroke=0, fill=1)

    def labeled_node(
        x: float,
        y: float,
        label: str,
        *,
        fill: str = "#FFFFFF",
        radius: float = 3.4,
        stroke: str = ink,
        accent: bool = False,
    ) -> None:
        if fill != "#FFFFFF":
            pdf.saveState()
            pdf.setFillColor(HexColor("#514B70"))
            pdf.setFillAlpha(0.08)
            pdf.circle(x + 0.35, y - 0.35, radius + 0.18, stroke=0, fill=1)
            pdf.restoreState()
        pdf.setFillColor(HexColor(fill))
        pdf.setStrokeColor(HexColor(stroke))
        pdf.setLineWidth(0.55)
        pdf.circle(x, y, radius, stroke=1, fill=1)
        if accent:
            pdf.setFillColor(HexColor("#FFFFFF"))
            pdf.setStrokeColor(HexColor(coral))
            pdf.setLineWidth(0.54)
            pdf.circle(x, y, radius + 0.75, stroke=1, fill=0)
        formula(
            x,
            y - 1.35,
            label,
            4.1,
            align="center",
            color=ink,
        )

    graph_coords = (
        (-0.90, 0.78),
        (0.82, 0.78),
        (-1.18, 0.04),
        (-0.18, 0.08),
        (0.82, 0.04),
        (-0.86, -0.70),
        (-0.08, -0.72),
        (0.78, -0.68),
        (1.30, -0.27),
    )
    graph_labels = ("1", "6", "4", "3", "7", "2", "9", "8", "n")
    graph_edges = (
        (0, 2), (0, 3), (2, 3), (2, 5), (3, 5), (3, 6),
        (3, 4), (4, 1), (4, 6), (4, 7), (4, 8), (6, 7),
        (7, 8), (5, 6),
    )
    update_edges = ((0, 1), (1, 5))
    endpoint_nodes = {0, 1, 5}
    neighborhood_nodes = {0, 1, 2, 3, 4, 5, 6}
    edge_bends = {
        (0, 2): 0.55,
        (0, 3): -0.70,
        (2, 3): 0.45,
        (2, 5): -0.50,
        (3, 5): 0.75,
        (3, 6): -0.65,
        (4, 1): 0.55,
        (4, 6): -0.65,
        (4, 7): 0.70,
        (4, 8): -0.75,
        (6, 7): 0.45,
        (7, 8): -0.55,
    }
    changed_bends = {(0, 1): 0.45, (1, 5): -0.70}

    def curved_graph_edge(
        first: tuple[float, float],
        second: tuple[float, float],
        bend: float,
        *,
        color: str,
        width: float,
        dash: tuple[float, float] | None = None,
    ) -> None:
        dx, dy = second[0] - first[0], second[1] - first[1]
        length = math.hypot(dx, dy)
        normal_x, normal_y = (-dy / length, dx / length) if length else (0.0, 0.0)
        control_1 = (
            first[0] + dx / 3.0 + normal_x * bend,
            first[1] + dy / 3.0 + normal_y * bend,
        )
        control_2 = (
            first[0] + 2.0 * dx / 3.0 + normal_x * bend,
            first[1] + 2.0 * dy / 3.0 + normal_y * bend,
        )
        path = pdf.beginPath()
        path.moveTo(*first)
        path.curveTo(*control_1, *control_2, *second)
        pdf.setStrokeColor(HexColor(color))
        pdf.setLineWidth(width)
        if dash:
            pdf.setDash(list(dash), 0)
        else:
            pdf.setDash()
        pdf.drawPath(path, stroke=1, fill=0)
        pdf.setDash()

    def update_graph(
        cx: float,
        cy: float,
        scale: float,
        *,
        affected: set[int] | None = None,
        neighborhood: bool = False,
    ) -> None:
        affected = affected or set()
        points = [(cx + px * scale, cy + py * scale) for px, py in graph_coords]
        if neighborhood:
            concave_neighborhood_contour(cx, cy, scale)
        for first, second in graph_edges:
            if (first, second) in update_edges:
                continue
            curved_graph_edge(
                points[first],
                points[second],
                edge_bends.get((first, second), 0.0),
                color=ink,
                width=0.47,
            )
        for first, second in update_edges:
            curved_graph_edge(
                points[first],
                points[second],
                changed_bends[(first, second)],
                color=cut_red,
                width=0.43,
                dash=(1.8, 1.45),
            )
        for index, (px, py) in enumerate(points):
            if index in affected:
                fill = coral
            elif neighborhood and index in neighborhood_nodes:
                fill = coral_neighbor
            else:
                fill = "#FFFFFF"
            labeled_node(px, py, graph_labels[index], fill=fill, radius=3.35)

    def update_batch(cx: float, cy: float) -> None:
        frame(cx - 18.0, cy - 23.0, 36.0, 46.0, fill="#FFFFFF", stroke=indigo_light, radius=2.2)
        for row, (left_label, right_label) in enumerate((("1", "6"), ("6", "2"))):
            row_y = cy + 7.0 - row * 14.0
            labeled_node(cx - 10.5, row_y, left_label, radius=2.8, stroke=indigo)
            labeled_node(cx + 10.5, row_y, right_label, radius=2.8, stroke=indigo)
            rule(
                cx - 7.4,
                row_y,
                cx + 7.4,
                row_y,
                color=cut_red,
                width=0.43,
                dash=(1.8, 1.45),
            )

    def nested_group_contour(
        members: tuple[tuple[float, float], ...],
        *,
        stroke: str,
        fill: str,
    ) -> None:
        """Draw a solid inner contour for one stored subcommunity."""
        if len(members) == 1:
            center_x, center_y = members[0]
            radius_x, radius_y, angle = 4.25, 4.25, 0.0
        else:
            first, last = members[0], members[-1]
            center_x = (first[0] + last[0]) / 2.0
            center_y = (first[1] + last[1]) / 2.0
            distance = math.hypot(last[0] - first[0], last[1] - first[1])
            radius_x = distance / 2.0 + 3.8
            radius_y = 4.15
            angle = math.atan2(last[1] - first[1], last[0] - first[0])
        community_contour(
            blob_anchors(center_x, center_y, radius_x, radius_y, angle=angle),
            stroke=stroke,
            fill=fill,
            fill_alpha=0.28,
            stroke_alpha=0.72,
            width=0.28,
            dash=None,
        )

    def parent_community_contour(
        members: tuple[tuple[float, float], ...],
        *,
        stroke: str,
        fill: str,
        radius: float = 5.0,
    ) -> None:
        rounded_hull_contour(
            members,
            radius=radius,
            stroke=stroke,
            fill=fill,
            fill_alpha=0.08,
        )

    def stage_edges(
        points: dict[str, tuple[float, float]],
        edges: tuple[tuple[str, str], ...],
        *,
        width: float = 0.48,
    ) -> None:
        for first, second in edges:
            rule(*points[first], *points[second], color=ink, width=width)

    def contracted_pill(
        x: float,
        y: float,
        label: str,
        *,
        fill: str,
    ) -> None:
        pdf.saveState()
        pdf.setFillColor(HexColor(fill))
        pdf.setStrokeColor(HexColor(ink))
        pdf.setLineWidth(0.48)
        pdf.roundRect(x - 5.5, y - 3.65, 11.0, 7.3, 3.1, stroke=1, fill=1)
        pdf.restoreState()
        formula(x, y - 1.25, label, 3.35, align="center", color="#FFFFFF")

    def affected_community_halo(
        members: tuple[tuple[float, float], ...],
    ) -> None:
        if len(members) == 1:
            center_x, center_y = members[0]
            radius_x, radius_y, angle = 6.1, 6.0, 0.0
        else:
            first, last = members[0], members[-1]
            center_x = (first[0] + last[0]) / 2.0
            center_y = (first[1] + last[1]) / 2.0
            distance = math.hypot(last[0] - first[0], last[1] - first[1])
            radius_x = distance / 2.0 + 5.7
            radius_y = 5.9
            angle = math.atan2(last[1] - first[1], last[0] - first[0])
        community_contour(
            blob_anchors(center_x, center_y, radius_x, radius_y, angle=angle),
            stroke=coral,
            fill=coral_light,
            fill_alpha=0.035,
            stroke_alpha=0.96,
            width=0.52,
            dash=(1.45, 1.0),
        )

    # Reader-oriented schematic of the grouping rule: updated-edge endpoints
    # are kept as singleton inputs, while each unaffected stored subcommunity
    # is represented by one contracted vertex.  The drawing intentionally
    # omits redundant induced edges so the grouping transformation stays clear.
    def closure_stage(cx: float, cy: float) -> None:
        points = {
            "1": (cx - 22.0, cy + 13.0),
            "3": (cx - 13.0, cy + 14.0),
            "4": (cx - 13.0, cy + 5.0),
            "6": (cx + 2.0, cy + 12.0),
            "2": (cx + 12.0, cy + 11.0),
            "7": (cx + 3.0, cy - 1.0),
            "9": (cx + 13.0, cy - 3.0),
            "8": (cx - 14.0, cy - 15.0),
            "n": (cx - 4.0, cy - 16.0),
        }
        purple_parent = (points["1"], points["3"], points["4"])
        teal_parent = (points["6"], points["2"], points["7"], points["9"])
        untouched_parent = (points["8"], points["n"])
        parent_community_contour(
            purple_parent,
            stroke=purple,
            fill=purple_light,
            radius=5.0,
        )
        parent_community_contour(
            teal_parent,
            stroke=teal,
            fill=teal_light,
            radius=5.0,
        )
        parent_community_contour(
            untouched_parent,
            stroke=slate,
            fill=slate_light,
            radius=4.7,
        )
        nested_group_contour((points["1"],), stroke=purple, fill=purple_light)
        nested_group_contour((points["3"], points["4"]), stroke=blue, fill=blue_light)
        nested_group_contour((points["6"],), stroke=teal, fill=teal_light)
        nested_group_contour((points["2"],), stroke=olive, fill=olive_light)
        nested_group_contour((points["7"], points["9"]), stroke="#72A457", fill=olive_light)
        node_styles = {
            "1": purple,
            "3": blue,
            "4": blue,
            "6": teal,
            "2": olive,
            "7": "#72A457",
            "9": "#72A457",
            "8": slate,
            "n": "#FFFFFF",
        }
        for label, (px, py) in points.items():
            labeled_node(
                px,
                py,
                label,
                fill=node_styles[label],
                radius=2.35,
                stroke=slate if label == "n" else ink,
                accent=label in {"1", "2", "6"},
            )

    def working_graph(cx: float, cy: float) -> None:
        points = {
            "1": (cx - 18.0, cy + 11.0),
            "3": (cx - 8.0, cy + 13.0),
            "4": (cx - 7.0, cy + 3.0),
            "6": (cx + 7.0, cy + 11.0),
            "2": (cx + 18.0, cy + 10.0),
            "7": (cx + 8.0, cy - 3.0),
            "9": (cx + 18.0, cy - 5.0),
        }
        parent_community_contour(
            (points["1"], points["3"], points["4"]),
            stroke=purple,
            fill=purple_light,
            radius=5.0,
        )
        parent_community_contour(
            (points["6"], points["2"], points["7"], points["9"]),
            stroke=teal,
            fill=teal_light,
            radius=5.0,
        )
        nested_group_contour((points["1"],), stroke=purple, fill=purple_light)
        nested_group_contour((points["3"], points["4"]), stroke=blue, fill=blue_light)
        nested_group_contour((points["6"],), stroke=teal, fill=teal_light)
        nested_group_contour((points["2"],), stroke=olive, fill=olive_light)
        nested_group_contour((points["7"], points["9"]), stroke="#72A457", fill=olive_light)
        stage_edges(
            points,
            (
                ("1", "3"),
                ("1", "4"),
                ("3", "4"),
                ("1", "6"),
                ("6", "2"),
                ("6", "7"),
                ("2", "9"),
                ("7", "9"),
                ("3", "7"),
            ),
        )
        node_styles = {
            "1": purple,
            "3": blue,
            "4": blue,
            "6": teal,
            "2": olive,
            "7": "#72A457",
            "9": "#72A457",
        }
        for label, (px, py) in points.items():
            labeled_node(
                px,
                py,
                label,
                fill=node_styles[label],
                radius=2.55,
                accent=label in {"1", "2", "6"},
            )

    def contracted_graph(cx: float, cy: float) -> None:
        points = {
            "1": (cx - 17.0, cy + 10.0),
            "34": (cx - 5.0, cy + 8.0),
            "6": (cx + 7.0, cy + 11.0),
            "2": (cx + 18.0, cy + 8.0),
            "79": (cx + 12.0, cy - 7.0),
        }
        parent_community_contour(
            (points["1"], points["34"]),
            stroke=purple,
            fill=purple_light,
            radius=5.0,
        )
        parent_community_contour(
            (points["6"], points["2"], points["79"]),
            stroke=teal,
            fill=teal_light,
            radius=5.0,
        )
        stage_edges(
            points,
            (
                ("1", "34"),
                ("1", "6"),
                ("34", "79"),
                ("6", "2"),
                ("6", "79"),
                ("2", "79"),
            ),
            width=0.55,
        )
        labeled_node(*points["1"], "1", fill=purple, radius=3.0, accent=True)
        contracted_pill(*points["34"], r"3,4", fill=blue)
        labeled_node(*points["6"], "6", fill=teal, radius=3.0, accent=True)
        labeled_node(*points["2"], "2", fill=olive, radius=3.0, accent=True)
        contracted_pill(*points["79"], r"7,9", fill="#72A457")

    def gear(cx: float, cy: float, radius: float) -> None:
        pdf.setFillColor(HexColor("#FFFFFF"))
        pdf.setStrokeColor(HexColor(indigo))
        pdf.setLineWidth(0.55)
        tooth_width = max(0.75, radius * 0.22)
        for angle in range(0, 360, 45):
            pdf.saveState()
            pdf.translate(cx, cy)
            pdf.rotate(angle)
            pdf.roundRect(
                -tooth_width / 2,
                radius * 0.72,
                tooth_width,
                radius * 0.56,
                tooth_width * 0.25,
                stroke=1,
                fill=1,
            )
            pdf.restoreState()
        pdf.circle(cx, cy, radius * 0.82, stroke=1, fill=1)
        pdf.circle(cx, cy, radius * 0.30, stroke=1, fill=0)

    def backend_stage(cx: float, cy: float) -> None:
        gear(cx - 4.0, cy + 18.0, 3.6)
        gear(cx + 5.0, cy + 14.5, 2.7)
        points = {
            "1": (cx - 18.0, cy + 1.0),
            "6": (cx - 9.0, cy - 8.0),
            "34": (cx + 1.0, cy + 3.0),
            "79": (cx + 10.0, cy - 7.0),
            "2": (cx + 18.0, cy + 5.0),
        }
        parent_community_contour(
            (points["1"], points["6"]),
            stroke=mauve,
            fill=mauve_light,
            radius=5.1,
        )
        parent_community_contour(
            (points["34"], points["79"]),
            stroke=amber,
            fill=amber_light,
            radius=5.1,
        )
        parent_community_contour(
            (points["2"],),
            stroke=teal,
            fill=teal_light,
            radius=5.0,
        )
        stage_edges(
            points,
            (
                ("1", "34"),
                ("1", "6"),
                ("34", "79"),
                ("6", "2"),
                ("6", "79"),
                ("2", "79"),
            ),
            width=0.52,
        )
        labeled_node(*points["1"], "1", fill=purple, radius=2.8, accent=True)
        labeled_node(*points["6"], "6", fill=teal, radius=2.8, accent=True)
        contracted_pill(*points["34"], r"3,4", fill=blue)
        contracted_pill(*points["79"], r"7,9", fill="#72A457")
        labeled_node(*points["2"], "2", fill=olive, radius=2.8, accent=True)

    def updated_communities_stage(cx: float, cy: float) -> None:
        points = {
            "1": (cx - 19.0, cy + 11.0),
            "6": (cx - 8.0, cy + 10.0),
            "3": (cx - 16.0, cy - 6.0),
            "4": (cx - 7.0, cy - 10.0),
            "7": (cx + 4.0, cy - 9.0),
            "9": (cx + 12.0, cy - 4.0),
            "2": (cx + 18.0, cy + 10.0),
        }
        purple_output = (points["1"], points["6"])
        amber_output = (points["3"], points["4"], points["7"], points["9"])
        teal_output = (points["2"],)
        parent_community_contour(
            purple_output,
            stroke=mauve,
            fill=mauve_light,
            radius=4.8,
        )
        parent_community_contour(
            amber_output,
            stroke=amber,
            fill=amber_light,
            radius=4.6,
        )
        nested_group_contour(
            (points["3"], points["4"]),
            stroke="#DDA842",
            fill=amber_light,
        )
        nested_group_contour(
            (points["7"], points["9"]),
            stroke="#C86E38",
            fill=amber_light,
        )
        parent_community_contour(
            teal_output,
            stroke=teal,
            fill=teal_light,
            radius=4.8,
        )
        affected_community_halo(purple_output)
        affected_community_halo(teal_output)
        # Restore the complete induced working graph after expanding the
        # backend labels.  Earlier drafts retained only intra-community edges,
        # which made this frame look like a different source graph.  Because
        # this is the completed state, every edge uses the same solid style.
        project_edges = (
            ("1", "4"),
            ("1", "3"),
            ("3", "4"),
            ("4", "2"),
            ("3", "2"),
            ("3", "9"),
            ("3", "7"),
            ("6", "7"),
            ("7", "9"),
            ("2", "9"),
            ("1", "6"),
            ("6", "2"),
        )
        project_edge_bends = {
            ("1", "4"): 0.9,
            ("1", "3"): -0.8,
            ("4", "2"): -2.0,
            ("3", "2"): 2.4,
            ("6", "7"): 0.9,
        }
        for first, second in project_edges:
            curved_graph_edge(
                points[first],
                points[second],
                project_edge_bends.get((first, second), 0.0),
                color=ink,
                width=0.38,
            )
        node_colors = {
            "1": mauve,
            "6": mauve,
            "3": "#DDA842",
            "4": "#DDA842",
            "7": "#C86E38",
            "9": "#C86E38",
            "2": teal,
        }
        for label, (px, py) in points.items():
            labeled_node(px, py, label, fill=node_colors[label], radius=2.45, accent=False)

    def hierarchy_stack(
        x: float,
        y: float,
        width: float,
        *,
        levels: list[list[HierarchyGroup]],
        highlight: bool = False,
    ) -> None:
        """Draw a restrained three-level stack, following the source figure."""
        layer_height = 11.8
        vertical_step = 13.2
        slant = 6.5
        level_offset = 0.8
        depth = 1.1
        blob_shift = slant * 0.42

        def quad_path(points: tuple[tuple[float, float], ...]) -> Any:
            path = pdf.beginPath()
            path.moveTo(*points[0])
            for point in points[1:]:
                path.lineTo(*point)
            path.close()
            return path

        def layer_capsule(
            center_x: float,
            center_y: float,
            half_width: float,
            color: str,
            refinement_level: int,
        ) -> None:
            if refinement_level == 2:
                half_height = min(2.62, 1.75 + 0.35 * half_width)
            else:
                half_height = 2.45
            pdf.saveState()
            pdf.setFillColor(HexColor("#514B70"))
            pdf.setFillAlpha(0.07)
            pdf.ellipse(
                center_x - half_width + 0.45,
                center_y - half_height - 0.45,
                center_x + half_width + 0.45,
                center_y + half_height - 0.45,
                stroke=0,
                fill=1,
            )
            pdf.restoreState()
            pdf.setFillColor(HexColor(color))
            pdf.setStrokeColor(HexColor(color))
            pdf.setLineWidth(0.32)
            pdf.ellipse(
                center_x - half_width,
                center_y - half_height,
                center_x + half_width,
                center_y + half_height,
                stroke=1,
                fill=1,
            )
            pdf.saveState()
            pdf.setFillColor(HexColor("#FFFFFF"))
            pdf.setFillAlpha(0.16)
            pdf.ellipse(
                center_x - half_width * 0.66,
                center_y + 0.35,
                center_x + half_width * 0.30,
                center_y + 1.25,
                stroke=0,
                fill=1,
            )
            pdf.restoreState()

        affected_box: tuple[float, float, float, float] | None = None
        if highlight:
            warm_start, warm_end = hierarchy_families[1]["span"]
            box_left = x + blob_shift + warm_start * width - 2.2
            box_right = (
                x
                + (len(levels) - 1) * level_offset
                + blob_shift
                + warm_end * width
                + 2.2
            )
            box_bottom = y - 1.8
            box_top = y + (len(levels) - 1) * vertical_step + layer_height + 1.8
            affected_box = (box_left, box_bottom, box_right, box_top)
            pdf.saveState()
            pdf.setFillColor(HexColor(coral))
            pdf.setFillAlpha(0.045)
            pdf.rect(
                box_left,
                box_bottom,
                box_right - box_left,
                box_top - box_bottom,
                stroke=0,
                fill=1,
            )
            pdf.restoreState()

        for level, groups in enumerate(levels):
            layer_y = y + level * vertical_step
            offset = level * level_offset
            top_face = (
                (x + offset, layer_y),
                (x + width + offset, layer_y),
                (x + width + slant + offset, layer_y + layer_height),
                (x + slant + offset, layer_y + layer_height),
            )
            shadow_face = tuple((px + 0.9, py - 1.0) for px, py in top_face)
            pdf.saveState()
            pdf.setFillColor(HexColor("#514B70"))
            pdf.setFillAlpha(0.09)
            pdf.drawPath(quad_path(shadow_face), stroke=0, fill=1)
            pdf.restoreState()
            front_face = (
                (x + offset, layer_y),
                (x + width + offset, layer_y),
                (x + width + offset, layer_y - depth),
                (x + offset, layer_y - depth),
            )
            pdf.setFillColor(HexColor("#E8E8F0"))
            pdf.setStrokeColor(HexColor("#A3A2B5"))
            pdf.setLineWidth(0.30)
            pdf.drawPath(quad_path(front_face), stroke=1, fill=1)
            pdf.setFillColor(HexColor("#FDFDFE"))
            pdf.setStrokeColor(HexColor("#A3A2B5"))
            pdf.setLineWidth(0.42)
            pdf.drawPath(quad_path(top_face), stroke=1, fill=1)
            pdf.setStrokeColor(HexColor("#D9D8E2"))
            pdf.setLineWidth(0.28)
            pdf.line(x + slant + offset, layer_y + layer_height,
                     x + width + slant + offset, layer_y + layer_height)

            blob_y = layer_y + 4.8
            for _, color, left, right in groups:
                center = (left + right) / 2.0
                center_x = x + offset + blob_shift + center * width
                minimum_width = 1.70 if level == 2 else 2.35
                half_width = max(
                    minimum_width,
                    (right - left) * width / 2.0 - 0.55,
                )
                layer_capsule(center_x, blob_y, half_width, color, level)

        if affected_box is not None:
            box_left, box_bottom, box_right, box_top = affected_box
            pdf.saveState()
            pdf.setStrokeColor(HexColor(cut_red))
            pdf.setStrokeAlpha(0.86)
            pdf.setLineWidth(0.48)
            pdf.setDash([1.35, 1.0], 0)
            pdf.rect(
                box_left,
                box_bottom,
                box_right - box_left,
                box_top - box_bottom,
                stroke=1,
                fill=0,
            )
            pdf.restoreState()

    # Source-faithful outer structure: a state rail and one continuous workflow.
    rail_x, rail_y, rail_w, rail_h = 4.0, 18.0, 112.0, 307.0
    flow_x, flow_y, flow_w, flow_h = 120.0, 18.0, 391.5, 307.0
    frame(rail_x, rail_y, rail_w, rail_h, fill="#FFFFFF", radius=2.2)
    frame(flow_x, flow_y, flow_w, flow_h, fill="#FFFFFF", radius=2.2)
    for separator_y in (304.0, 232.0, 169.0):
        rule(rail_x, separator_y, rail_x + rail_w, separator_y, color=indigo_light, width=0.5)
    for separator_y in (225.0, 104.0):
        rule(flow_x, separator_y, flow_x + flow_w, separator_y, color=indigo_light, width=0.62)

    draw_text(
        pdf,
        rail_x + rail_w / 2,
        311.5,
        "Data stored",
        6.3,
        FIGURE_FONT_BOLD,
        "center",
        indigo,
    )

    # State rail: one accumulated adjacency, optional features, the hierarchy,
    # and the level loop used by Steps 2--3.  All content follows a common
    # six-point inset so the text and graphics share balanced outer margins.
    rail_content_x = 10.0
    rail_content_w = 100.0
    state_math_size = 5.6
    entry_math_size = 5.0
    type_text_size = 3.75
    meaning_text_size = 3.65

    draw_text(
        pdf,
        rail_content_x,
        290.5,
        "1) Accumulated adjacency",
        5.8,
        FIGURE_FONT_BOLD,
        color=indigo,
    )
    formula(
        rail_content_x,
        274.0,
        r"\bm{A}_{t}\in\mathbb{R}^{n\times n}",
        state_math_size,
    )
    draw_text(
        pdf,
        rail_content_x,
        263.0,
        "sparse floating-point tensor",
        type_text_size,
        color=muted,
    )
    formula(
        rail_content_x,
        249.0,
        r"[\bm{A}_{t}]_{ij}",
        entry_math_size,
    )
    draw_text(
        pdf,
        rail_content_x,
        238.0,
        "edge weight for the indexed vertex pair",
        meaning_text_size,
        color=muted,
    )
    matrix_x, matrix_y, matrix_size = 76.0, 248.0, 28.0
    sparse_matrix(matrix_x, matrix_y, matrix_size)
    matrix_cell = matrix_size / 8.0
    matrix_index_size = 3.8
    for label, column in ((r"1", 0.5), (r"2", 1.5), (r"\cdots", 4.5), (r"n", 7.5)):
        formula(
            matrix_x + column * matrix_cell,
            matrix_y + matrix_size + 1.3,
            label,
            matrix_index_size,
            align="center",
            color=muted,
        )
    for label, row in ((r"1", 0.5), (r"2", 1.5), (r"\vdots", 4.5), (r"n", 7.5)):
        formula(
            matrix_x - 2.4,
            matrix_y + matrix_size - row * matrix_cell - 1.4,
            label,
            matrix_index_size,
            align="right",
            color=muted,
        )

    draw_text(
        pdf,
        rail_content_x,
        218.5,
        "2) Optional vertex features",
        5.8,
        FIGURE_FONT_BOLD,
        color=indigo,
    )
    formula(
        rail_content_x,
        203.0,
        r"\bm{X}\in\mathbb{R}^{n\times d}",
        state_math_size,
    )
    draw_text(
        pdf,
        rail_content_x,
        192.0,
        "dense/sparse floating-point tensor",
        type_text_size,
        color=muted,
    )
    formula(
        rail_content_x,
        180.5,
        r"[\bm{X}]_{ik}",
        entry_math_size,
    )
    draw_text(
        pdf,
        25.0,
        181.2,
        "indexed feature value",
        meaning_text_size,
        color=muted,
    )
    feature_matrix(80.0, 182.5, 30.0, 30.0)

    draw_text(
        pdf,
        rail_content_x,
        155.5,
        "3) Stored label hierarchy",
        5.8,
        FIGURE_FONT_BOLD,
        color=indigo,
    )
    formula(
        rail_content_x,
        139.5,
        r"\bm{C}_{t}\in\mathbb{Z}^{L\times n}",
        state_math_size,
    )
    draw_text(
        pdf,
        rail_content_x,
        128.5,
        "dense integer label tensor",
        type_text_size,
        color=muted,
    )
    stored_hierarchy_tensor(rail_content_x, 74.0, rail_content_w)

    # Backend choice is configuration rather than stored state.  A balanced
    # separator gives it a distinct visual band without breaking the rail.
    rule(rail_x, 67.0, rail_x + rail_w, 67.0, color=indigo_light, width=0.5)

    frame(
        rail_content_x,
        24.0,
        rail_content_w,
        36.0,
        fill=faint,
        stroke="#C4C2D2",
        radius=1.5,
        line_width=0.42,
    )
    draw_text(
        pdf,
        60.0,
        52.5,
        "Backend configuration",
        4.8,
        FIGURE_FONT_BOLD,
        "center",
        indigo,
    )
    rule(60.0, 28.0, 60.0, 47.5, color="#D8D6E4", width=0.4)
    draw_text(
        pdf,
        35.0,
        41.5,
        "feature-aware",
        3.45,
        FIGURE_FONT_BOLD,
        "center",
        ink,
    )
    draw_text(pdf, 35.0, 35.5, "group sum or mean", 3.15, align="center", color=muted)
    formula(
        35.0,
        28.0,
        r"(\bar{\bm{A}}_\ell,\bar{\bm{X}}_\ell)",
        4.2,
        align="center",
        color=indigo,
    )
    draw_text(
        pdf,
        85.0,
        41.5,
        "topology-only",
        3.45,
        FIGURE_FONT_BOLD,
        "center",
        ink,
    )
    draw_text(pdf, 85.0, 35.5, "adjacency only", 3.15, align="center", color=muted)
    formula(
        85.0,
        28.0,
        r"\bar{\bm{A}}_\ell",
        4.45,
        align="center",
        color=indigo,
    )

    # Step 1 follows the original four-frame narrative.
    step_header(1, "Update and affected-region detection", 126.0, 310.0)
    top_centers = (148.0, 239.0, 343.0, 458.0)
    top_titles = ("New batch", "Graph after update", "Affected endpoints", "Expanded neighborhood")
    for center, title in zip(top_centers, top_titles):
        draw_text(pdf, center, 291.0, title, 5.5, FIGURE_FONT_BOLD, "center", indigo)
    update_batch(top_centers[0], 259.0)
    update_graph(top_centers[1], 259.0, 16.5)
    update_graph(top_centers[2], 259.0, 16.5, affected=endpoint_nodes)
    update_graph(top_centers[3], 259.0, 16.5, affected=endpoint_nodes, neighborhood=True)
    arrow(168.0, 260.0, 205.0, 260.0)
    arrow(273.0, 260.0, 305.0, 260.0)
    arrow(378.0, 260.0, 420.0, 260.0)
    draw_text(pdf, 186.5, 270.0, "insert into", 4.4, FIGURE_FONT_BOLD, "center", ink)
    draw_text(pdf, 186.5, 264.0, "adjacency", 4.4, FIGURE_FONT_BOLD, "center", ink)
    draw_text(pdf, 289.0, 270.0, "mark endpoints", 4.4, FIGURE_FONT_BOLD, "center", ink)
    draw_text(pdf, 289.0, 264.0, "as affected", 4.4, FIGURE_FONT_BOLD, "center", ink)
    draw_text(pdf, 399.0, 270.0, "expand", 4.4, FIGURE_FONT_BOLD, "center", ink)
    draw_text(pdf, 399.0, 264.0, "neighborhood", 4.4, FIGURE_FONT_BOLD, "center", ink)
    diagram_formula_size = 5.55
    formula(
        top_centers[0],
        230.0,
        r"\bm{\Delta}_t",
        diagram_formula_size,
        align="center",
    )
    formula(
        top_centers[1],
        230.0,
        r"\bm{A}_t\gets\bm{A}_{t-1}+\bm{\Delta}_t",
        diagram_formula_size,
        align="center",
    )
    formula(
        top_centers[2],
        230.0,
        r"S_t",
        diagram_formula_size,
        align="center",
    )
    formula(
        top_centers[3],
        230.0,
        r"B_t=N_r(S_t)",
        diagram_formula_size,
        align="center",
    )

    # Step 2 uses the full workflow width to show filtering, contraction,
    # repartitioning, and projection without collapsing distinct semantics.
    step_header(2, "Hierarchical local recomputation", 126.0, 210.0)
    pipeline_centers = (158.0, 237.0, 316.0, 395.0, 474.0)
    pipeline_titles = (
        "Community closure",
        "Working graph",
        "Contract groups",
        "Run backend",
        "Updated communities",
    )
    for center, title in zip(pipeline_centers, pipeline_titles):
        draw_text(pdf, center, 190.0, title, 5.2, FIGURE_FONT_BOLD, "center", indigo)
    for divider_x in (197.5, 276.5, 355.5, 434.5):
        rule(divider_x, 128.0, divider_x, 151.0, color=hairline, width=0.38, dash=(1.6, 1.6))
        rule(divider_x, 163.0, divider_x, 194.0, color=hairline, width=0.38, dash=(1.6, 1.6))
    closure_stage(pipeline_centers[0], 158.0)
    working_graph(pipeline_centers[1], 158.0)
    contracted_graph(pipeline_centers[2], 158.0)
    backend_stage(pipeline_centers[3], 158.0)
    updated_communities_stage(pipeline_centers[4], 158.0)
    for left, right in zip(pipeline_centers[:-1], pipeline_centers[1:]):
        arrow(left + 26.0, 157.0, right - 26.0, 157.0, head=3.2)
    draw_backed_text(
        pdf,
        197.5,
        162.0,
        "retain touched",
        3.2,
        font=FIGURE_FONT_BOLD,
        align="center",
        color=muted,
        pad_x=0.8,
        pad_y=0.35,
    )
    formula(
        pipeline_centers[0],
        126.0,
        r"S_t=\{1,2,6\},\ B_t\subseteq U_t^\ell",
        4.45,
        align="center",
    )
    formula(
        pipeline_centers[1],
        126.0,
        r"\bm{A}^{\mathrm{work}}_\ell",
        diagram_formula_size,
        align="center",
    )
    formula(
        pipeline_centers[2],
        126.0,
        r"\bar{\bm{A}}_\ell=\bm{P}_\ell\bm{A}^{\mathrm{work}}_\ell\bm{P}_\ell^{\mathsf T}",
        diagram_formula_size,
        align="center",
    )
    formula(
        pipeline_centers[3],
        126.0,
        r"\bar{\bm{y}}_\ell=\mathcal{B}(\cdot)",
        diagram_formula_size,
        align="center",
    )
    formula(
        pipeline_centers[4],
        126.0,
        r"C_t^\ell",
        diagram_formula_size,
        align="center",
    )

    frame(126.0, 109.5, 379.0, 12.0, fill="#FFFFFF", stroke=hairline, radius=1.6, line_width=0.45)
    legend_y = 115.5
    pdf.setFillColor(HexColor(purple))
    pdf.setStrokeColor(HexColor(ink))
    pdf.setLineWidth(0.42)
    pdf.circle(139.0, legend_y, 2.25, stroke=1, fill=1)
    pdf.setStrokeColor(HexColor(coral))
    pdf.setLineWidth(0.54)
    pdf.circle(139.0, legend_y, 3.0, stroke=1, fill=0)
    draw_text(pdf, 145.0, legend_y - 1.8, "affected input vertex", 4.15)
    pdf.setFillColor(HexColor("#FFFFFF"))
    pdf.setStrokeColor(HexColor(ink))
    pdf.setLineWidth(0.42)
    pdf.circle(215.0, legend_y, 2.5, stroke=1, fill=1)
    draw_text(pdf, 221.0, legend_y - 1.8, "other vertex", 4.15)
    for index, color in enumerate((purple, blue, olive)):
        anchors = blob_anchors(279.0 + index * 3.3, legend_y, 2.7, 2.2, variant=index)
        pdf.setFillColor(HexColor(color))
        pdf.setStrokeColor(HexColor(color))
        pdf.setLineWidth(0.28)
        pdf.drawPath(smooth_closed_path(anchors), stroke=1, fill=1)
    draw_text(pdf, 292.0, legend_y - 1.8, "contracted group", 4.15)
    community_contour(
        blob_anchors(404.0, legend_y, 6.2, 2.8),
        stroke=coral,
        fill=coral_light,
        fill_alpha=0.035,
        stroke_alpha=0.96,
        width=0.50,
        dash=(1.35, 0.95),
    )
    draw_text(pdf, 413.0, legend_y - 1.8, "affected output community", 4.05)

    # Give the hierarchy comparison a taller band and a more generous lower
    # margin so neither stack appears pinned to the workflow boundary.
    step_header(3, "Repeat backend calls across stored levels", 126.0, 88.4)
    for label, baseline in ((r"0", 31.9), (r"i", 45.1), (r"L\!-\!1", 58.3)):
        formula(144.0, baseline, label, 5.1, align="right", color=muted)

    hierarchy_stack(149.0, 28.9, 120.0, levels=current_hierarchy, highlight=True)
    draw_text(
        pdf,
        213.1,
        75.4,
        "local recomputation scope",
        4.0,
        FIGURE_FONT_BOLD,
        "center",
        cut_red,
    )

    arrow(286.0, 50.9, 363.0, 50.9, head=3.4)
    draw_text(pdf, 324.5, 59.9, "level-wise update", 4.7, FIGURE_FONT_BOLD, "center", indigo)

    hierarchy_stack(372.0, 28.9, 124.0, levels=updated_hierarchy)
    draw_text(pdf, 436.0, 77.4, "Recomputed hierarchy", 5.2, FIGURE_FONT_BOLD, "center", indigo)

    # Thin source-style edge legend below the workflow.
    frame(120.0, 3.0, 391.5, 12.0, fill="#FFFFFF", stroke=indigo_light, radius=1.8, line_width=0.5)
    draw_text(pdf, 129.0, 6.2, "Legend:", 4.7, FIGURE_FONT_BOLD, color=indigo)
    rule(202.0, 9.0, 219.0, 9.0, color=ink, width=0.55)
    draw_text(pdf, 224.0, 6.2, "edge", 4.4)
    rule(324.0, 9.0, 341.0, 9.0, color=cut_red, width=0.54, dash=(1.55, 1.10))
    draw_text(pdf, 346.0, 6.2, "new / changed edge", 4.4)

    pdf.showPage()
    pdf.save()

    color_names: dict[str, str] = {}
    for node in formula_nodes:
        color_names.setdefault(node["color"], f"FormulaColor{len(color_names)}")

    tex_lines = [
        r"\let\TeXprimitiveyear\year",
        r"\documentclass[class=ieeeaccess,crop,border=0pt]{standalone}",
        r"\let\year\TeXprimitiveyear",
        r"\ifdefined\pdfinfoomitdate\pdfinfoomitdate=1\fi",
        r"\ifdefined\pdfsuppressptexinfo\pdfsuppressptexinfo=-1\fi",
        r"\ifdefined\pdftrailerid\pdftrailerid{}\fi",
        r"\usepackage{amsmath,amssymb,amsfonts,bm,tikz,graphicx}",
        r"\makeatletter",
        r"\AtBeginDocument{\DeclareMathVersion{bold}",
        r"\SetSymbolFont{operators}{bold}{T1}{times}{b}{n}",
        r"\SetSymbolFont{NewLetters}{bold}{T1}{times}{b}{it}",
        r"\SetMathAlphabet{\mathrm}{bold}{T1}{times}{b}{n}",
        r"\SetMathAlphabet{\mathit}{bold}{T1}{times}{b}{it}",
        r"\SetMathAlphabet{\mathbf}{bold}{T1}{times}{b}{n}",
        r"\SetMathAlphabet{\mathtt}{bold}{OT1}{pcr}{b}{n}",
        r"\SetSymbolFont{symbols}{bold}{OMS}{cmsy}{b}{n}",
        r"\renewcommand\boldmath{\@nomath\boldmath\mathversion{bold}}}",
        r"\makeatother",
    ]
    for color, name in color_names.items():
        tex_lines.append(f"\\definecolor{{{name}}}{{HTML}}{{{color.lstrip('#')}}}")
    tex_lines.extend(
        [
            r"\begin{document}",
            r"\global\eoddefinedtrue",
            r"\begin{tikzpicture}[x=1bp,y=1bp]",
            f"\\path[use as bounding box] (0,0) rectangle ({page_width:.2f},{page_height:.2f});",
            (
                r"\node[anchor=south west,inner sep=0bp,outer sep=0bp] at (0,0) "
                + "{\\includegraphics[width="
                + f"{page_width:.2f}bp,height={page_height:.2f}bp]"
                + "{"
                + base_path.as_posix()
                + "}};"
            ),
        ]
    )
    anchors = {"left": "base west", "center": "base", "right": "base east"}
    for node in formula_nodes:
        content = "{$" + node["tex"] + "$}"
        tex_lines.append(
            "\\node[anchor="
            + anchors[node["align"]]
            + ",inner sep=0bp,outer sep=0bp,transform shape,scale="
            + f"{node['scale']:.3f},text="
            + color_names[node["color"]]
            + f"] at ({node['x']:.2f},{node['y']:.2f}) "
            + content
            + ";"
        )
    tex_lines.extend([r"\end{tikzpicture}", r"\end{document}"])

    overlay_source = temporary_path / "method_pipeline_overlay.tex"
    write_text(overlay_source, "\n".join(tex_lines))
    compiler = shutil.which("pdflatex")
    if compiler:
        command = [
            compiler,
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-output-directory={temporary_path}",
            str(overlay_source),
        ]
    else:
        compiler = shutil.which("tectonic")
        if compiler is None:
            raise RuntimeError("pdflatex or tectonic is required to compose method_pipeline.pdf")
        command = [
            compiler,
            "--only-cached",
            "--outdir",
            str(temporary_path),
            str(overlay_source),
        ]
    environment = os.environ.copy()
    environment.setdefault("SOURCE_DATE_EPOCH", "946684800")
    result = subprocess.run(
        command,
        cwd=JOURNAL,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        diagnostic = (result.stdout + "\n" + result.stderr)[-5000:]
        raise RuntimeError(f"TeX composition of method_pipeline.pdf failed:\n{diagnostic}")
    composed_path = temporary_path / "method_pipeline_overlay.pdf"
    if not composed_path.exists():
        raise RuntimeError("TeX composition did not create method_pipeline_overlay.pdf")
    shutil.copyfile(composed_path, FIGURES / "method_pipeline.pdf")
    temporary.cleanup()


def main() -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    GENERATED.mkdir(parents=True, exist_ok=True)

    experiment_records = load_records(EXPERIMENTS)
    workload_records = load_records(WORKLOADS)
    neighborhood_records = load_records(NEIGHBORHOODS)
    cut_metric_records = load_records(CUT_METRICS)
    assert len(experiment_records) == 446
    assert len(workload_records) == 24
    assert len(neighborhood_records) == 6
    assert len(cut_metric_records) == 4

    info = data_info()
    nodes = {dataset: int(meta["n"]) for dataset, meta in info.items()}
    short = validate_short_horizon(experiment_records)
    repeatability = validate_backend_repeatability(experiment_records)
    long_horizon = validate_long_horizon(experiment_records)
    dsbm = validate_dsbm(experiment_records)
    workloads = validate_workloads(workload_records, nodes)
    neighborhoods = validate_neighborhoods(neighborhood_records)
    topology = validate_topology_ablation(experiment_records)
    directed = validate_directed_control(experiment_records)
    gamma_sweep = validate_gamma_sweep(experiment_records)
    feature_ablation = validate_feature_ablation(experiment_records)
    cut_metrics = validate_cut_metrics(cut_metric_records)

    evidence = {
        "sources": {
            "experiments": str(EXPERIMENTS.relative_to(ROOT)),
            "workloads": str(WORKLOADS.relative_to(ROOT)),
            "neighborhoods": str(NEIGHBORHOODS.relative_to(ROOT)),
            "cut_metrics": str(CUT_METRICS.relative_to(ROOT)),
        },
        "source_record_counts": {
            "experiments": len(experiment_records),
            "workloads": len(workload_records),
            "neighborhoods": len(neighborhood_records),
            "cut_metrics": len(cut_metric_records),
        },
        "short_horizon": short,
        "backend_repeatability": repeatability,
        "long_horizon": long_horizon,
        "dsbm": dsbm,
        "workloads": workloads,
        "neighborhoods": neighborhoods,
        "topology_ablation": topology,
        "directed_control": directed,
        "gamma_sweep": gamma_sweep,
        "feature_ablation": feature_ablation,
        "cut_metrics": cut_metrics,
    }
    write_text(
        ANALYSIS / "validated_results.json",
        json.dumps(evidence, indent=2, ensure_ascii=False, sort_keys=True),
    )
    emit_rows(
        info,
        short,
        repeatability,
        long_horizon,
        workloads,
        neighborhoods,
        dsbm,
        topology,
        directed,
        gamma_sweep,
        feature_ablation,
        cut_metrics,
    )

    plot_long_horizon(long_horizon)
    plot_dsbm(dsbm)
    plot_topology(topology)
    plot_method_pipeline()

    # Explicitly verify the headline values used by the manuscript macros.
    assert math.isclose(short["arxivmath"]["speedup"], 41.8805194204, rel_tol=1e-9)
    assert math.isclose(long_horizon["arxivmath"]["leidenalg"]["speedup"], 30.5365, rel_tol=1e-4)
    assert math.isclose(dsbm["0.05"]["random"]["speedup_mean"], 0.8893095922, rel_tol=1e-9)
    assert math.isclose(
        gamma_sweep["dyn_pubmed"]["2.0"]["delta_q"],
        -0.0529655814,
        rel_tol=1e-8,
    )
    assert directed["dyn_pubmed"]["updates"] == 10
    assert feature_ablation["dyn_pubmed"]["onehot"]["naive"]["updates"] == 44
    assert cut_metrics["datasets"]["arxivmath"]["smart"]["clusters"] == 8094
    print("Validated evidence and regenerated IEEE Access figures/tables.")


if __name__ == "__main__":
    main()
