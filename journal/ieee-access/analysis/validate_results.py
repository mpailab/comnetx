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
    Non-quantitative implementation-traceable flow diagram.
"""

from __future__ import annotations

import json
import math
import re
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
        "no_closure": "No closure",
        "no_contraction": "No contraction",
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


def new_canvas(name: str, width_in: float, height_in: float) -> tuple[canvas.Canvas, float, float]:
    """Create a deterministic vector PDF canvas in the article figure folder."""
    path = FIGURES / name
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
        draw_text(pdf, x0 - 3, y - 2.2, yformat.format(value), 5.7, align="right")
    for value in xticks:
        if value < xmin or value > xmax:
            continue
        x = tx(value)
        pdf.setStrokeColor(HexColor(INK))
        pdf.setLineWidth(0.45)
        pdf.line(x, y0, x, y0 - 2.5)
        draw_text(pdf, x, y0 - 10, xformat.format(value), 5.7, align="center")
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
    pdf, page_width, page_height = new_canvas("topology_ablation.pdf", 7.16, 2.45)
    datasets = ("dyn_cora", "dyn_pubmed", "arxivmath")
    left = 32.0
    right = 10.0
    gap = 22.0
    bottom = 35.0
    top = 18.0
    panel_width = (page_width - left - right - 2 * gap) / 3
    panel_height = page_height - bottom - top
    marker_names = {0: "circle", 1: "square", 2: "triangle"}
    for panel, dataset in enumerate(datasets):
        data = topology[dataset]
        x0 = left + panel * (panel_width + gap)
        all_q = [data["full"]["q"]] + [point["q"] for point in data["points"]]
        qmin, qmax = min(all_q), max(all_q)
        qpad = max((qmax - qmin) * 0.10, 0.001)
        ymin, ymax = qmin - qpad, qmax + qpad
        max_speed = max(point["speedup"] for point in data["points"])
        xmax = 10 ** math.ceil(math.log10(max_speed * 1.05))
        yticks = [ymin + i * (ymax - ymin) / 4 for i in range(5)]
        tx, ty = draw_panel_axes(
            pdf,
            x0,
            bottom,
            panel_width,
            panel_height,
            1,
            xmax,
            ymin,
            ymax,
            [1, 10, 100, 1000],
            yticks,
            xlog=True,
            xformat="{:.0f}",
            yformat="{:.3f}",
        )
        draw_text(pdf, x0 + panel_width / 2, page_height - 11, dataset, 7.3, FIGURE_FONT_BOLD, "center")
        draw_marker(pdf, tx(1.0), ty(data["full"]["q"]), "star", 4.0, INK, INK)
        for point in data["points"]:
            default = point["level"] == 3 and point["radius"] == 1
            color = BLUE if default else MID
            draw_marker(
                pdf,
                tx(point["speedup"]),
                ty(point["q"]),
                marker_names[point["radius"]],
                3.1 if default else 2.7,
                color,
                color if default else None,
            )
            draw_text(
                pdf,
                tx(point["speedup"]) + 3.0,
                ty(point["q"]) + 1.8,
                str(point["level"]),
                5.2,
            )
        draw_text(pdf, x0 + panel_width / 2, 12, "Full/Local time", 6.1, align="center")
    draw_vertical_text(pdf, 6, bottom + panel_height / 2, "Final modularity", 6.2)
    legend_y = 5.0
    legend_x = 106.0
    entries = [
        ("star", INK, INK, "Full"),
        ("circle", MID, None, "r=0"),
        ("square", MID, None, "r=1"),
        ("triangle", MID, None, "r=2"),
        ("square", BLUE, BLUE, "default L=3,r=1"),
    ]
    for shape, stroke, fill, label in entries:
        draw_marker(pdf, legend_x, legend_y + 1.5, shape, 2.5, stroke, fill)
        draw_text(pdf, legend_x + 6, legend_y - 1, label, 5.6)
        legend_x += stringWidth(label, FIGURE_FONT, 5.6) + 24
    pdf.showPage()
    pdf.save()


def draw_box(
    pdf: canvas.Canvas,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    body: str,
    edge: str = BLUE,
    fill: str = "#F6F8FA",
) -> None:
    pdf.setStrokeColor(HexColor(edge))
    pdf.setFillColor(HexColor(fill))
    pdf.setLineWidth(0.8)
    pdf.roundRect(x, y, width, height, 5, stroke=1, fill=1)
    draw_text(pdf, x + 6, y + height - 13, title, 7.0, FIGURE_FONT_BOLD)
    draw_wrapped(pdf, x + 6, y + height - 25, body, width - 12, 6.1, 7.2)


def draw_arrow(pdf: canvas.Canvas, x1: float, y1: float, x2: float, y2: float) -> None:
    pdf.setStrokeColor(HexColor(INK))
    pdf.setFillColor(HexColor(INK))
    pdf.setLineWidth(0.7)
    pdf.line(x1, y1, x2, y2)
    angle = math.atan2(y2 - y1, x2 - x1)
    size = 4.0
    path = pdf.beginPath()
    path.moveTo(x2, y2)
    path.lineTo(x2 - size * math.cos(angle - 0.45), y2 - size * math.sin(angle - 0.45))
    path.lineTo(x2 - size * math.cos(angle + 0.45), y2 - size * math.sin(angle + 0.45))
    path.close()
    pdf.drawPath(path, stroke=0, fill=1)


def plot_method_pipeline() -> None:
    pdf, page_width, page_height = new_canvas("method_pipeline.pdf", 7.16, 3.34)

    # This figure deliberately uses only vector primitives.  Its compact visual
    # vocabulary (label strips, membership matrices, and small graphs) mirrors
    # the actual data transformations more faithfully than prose-only boxes.
    blue = BLUE
    blue_fill = "#EDF4F9"
    orange = ORANGE
    orange_fill = "#FCEFE7"
    teal = "#397C72"
    teal_fill = "#EAF5F2"
    purple = "#6B5A91"
    purple_fill = "#F1EEF7"
    green = "#5D7D55"
    green_fill = "#EEF5EB"
    red = "#B94B48"
    border = "#B8C2C9"
    pale = "#F8FAFB"
    node_colors = ("#E98A70", "#76A9D3", "#9EBB86", "#B69AD0")

    def panel(
        x: float,
        y: float,
        width: float,
        height: float,
        step: str,
        title: str,
        accent: str = blue,
        fill: str = "#FFFFFF",
    ) -> None:
        pdf.setStrokeColor(HexColor(border))
        pdf.setFillColor(HexColor(fill))
        pdf.setLineWidth(0.7)
        pdf.roundRect(x, y, width, height, 4.0, stroke=1, fill=1)
        pdf.setStrokeColor(HexColor("#DCE2E6"))
        pdf.line(x, y + height - 15.5, x + width, y + height - 15.5)
        pdf.setFillColor(HexColor(accent))
        pdf.circle(x + 10.0, y + height - 8.0, 6.0, stroke=0, fill=1)
        draw_text(
            pdf,
            x + 10.0,
            y + height - 10.4,
            step,
            7.0,
            FIGURE_FONT_BOLD,
            "center",
            "#FFFFFF",
        )
        draw_text(
            pdf,
            x + 20.0,
            y + height - 10.5,
            title,
            7.4,
            FIGURE_FONT_BOLD,
            color=INK,
        )

    def flow_arrow(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        color: str = blue,
    ) -> None:
        pdf.setStrokeColor(HexColor(color))
        pdf.setFillColor(HexColor(color))
        pdf.setLineWidth(0.9)
        pdf.line(x1, y1, x2, y2)
        angle = math.atan2(y2 - y1, x2 - x1)
        size = 3.3
        path = pdf.beginPath()
        path.moveTo(x2, y2)
        path.lineTo(
            x2 - size * math.cos(angle - 0.48),
            y2 - size * math.sin(angle - 0.48),
        )
        path.lineTo(
            x2 - size * math.cos(angle + 0.48),
            y2 - size * math.sin(angle + 0.48),
        )
        path.close()
        pdf.drawPath(path, stroke=0, fill=1)

    def pill(
        x: float,
        y: float,
        width: float,
        height: float,
        value: str,
        edge: str,
        fill: str,
        font: str = FIGURE_FONT,
    ) -> None:
        pdf.setStrokeColor(HexColor(edge))
        pdf.setFillColor(HexColor(fill))
        pdf.setLineWidth(0.65)
        pdf.roundRect(x, y, width, height, height / 2, stroke=1, fill=1)
        draw_text(
            pdf,
            x + width / 2,
            y + (height - 6.5) / 2 + 0.7,
            value,
            6.5,
            font,
            "center",
        )

    def matrix_icon(
        x: float,
        y: float,
        cols: int,
        rows: int,
        cell: float,
        accent: str,
        diagonal: bool = False,
    ) -> None:
        pdf.setStrokeColor(HexColor("#9DA9B1"))
        pdf.setLineWidth(0.35)
        for col in range(cols + 1):
            pdf.line(x + col * cell, y, x + col * cell, y + rows * cell)
        for row in range(rows + 1):
            pdf.line(x, y + row * cell, x + cols * cell, y + row * cell)
        pdf.setFillColor(HexColor(accent))
        if diagonal:
            for index in range(min(cols, rows)):
                pdf.rect(
                    x + index * cell + 0.5,
                    y + (rows - index - 1) * cell + 0.5,
                    cell - 1.0,
                    cell - 1.0,
                    stroke=0,
                    fill=1,
                )
        else:
            for col, row in ((0, 0), (1, 2), (2, 1), (3, 3)):
                if col < cols and row < rows:
                    pdf.rect(
                        x + col * cell + 0.5,
                        y + row * cell + 0.5,
                        cell - 1.0,
                        cell - 1.0,
                        stroke=0,
                        fill=1,
                    )

    def label_strip(
        x: float,
        y: float,
        widths: tuple[float, ...],
        colors: tuple[str, ...],
        height: float = 9.0,
    ) -> None:
        cursor = x
        for width, color in zip(widths, colors):
            pdf.setStrokeColor(HexColor("#FFFFFF"))
            pdf.setFillColor(HexColor(color))
            pdf.setLineWidth(0.7)
            pdf.roundRect(cursor, y, width, height, 2.2, stroke=1, fill=1)
            cursor += width

    def tiny_graph(
        cx: float,
        cy: float,
        colors: tuple[str, ...],
        cut: bool = False,
    ) -> None:
        points = (
            (cx - 12.0, cy + 6.5),
            (cx + 10.5, cy + 7.0),
            (cx - 8.0, cy - 7.0),
            (cx + 12.5, cy - 6.0),
        )
        edges = ((0, 1), (0, 2), (1, 3), (2, 3), (0, 3))
        pdf.setLineWidth(0.65)
        for index, (left, right) in enumerate(edges):
            if cut and index == 4:
                pdf.setStrokeColor(HexColor(red))
                pdf.setDash(2.2, 1.5)
            else:
                pdf.setStrokeColor(HexColor("#657078"))
                pdf.setDash()
            pdf.line(
                points[left][0],
                points[left][1],
                points[right][0],
                points[right][1],
            )
        pdf.setDash()
        for (px, py), color in zip(points, colors):
            pdf.setStrokeColor(HexColor("#58636A"))
            pdf.setFillColor(HexColor(color))
            pdf.circle(px, py, 3.8, stroke=1, fill=1)
        if cut:
            draw_text(pdf, cx + 1.0, cy - 1.8, "x", 7.0, FIGURE_FONT_BOLD, color=red)

    def closed_level(
        x: float,
        y: float,
        groups: tuple[tuple[int, ...], ...],
        touched: int,
    ) -> None:
        centers = tuple(x + index * 12.0 for index in range(6))
        for group_index, group in enumerate(groups):
            left = centers[min(group)] - 5.0
            right = centers[max(group)] + 5.0
            pdf.setStrokeColor(HexColor(orange if touched in group else "#98A5AD"))
            pdf.setFillColor(HexColor(orange_fill if touched in group else "#F5F7F8"))
            pdf.setLineWidth(0.65)
            if touched in group:
                pdf.setDash(2.0, 1.3)
            pdf.roundRect(left, y - 4.5, right - left, 9.0, 4.5, stroke=1, fill=1)
            pdf.setDash()
            for node in group:
                pdf.setStrokeColor(HexColor("#6E797F"))
                pdf.setFillColor(HexColor(orange if node == touched else "#FFFFFF"))
                pdf.circle(centers[node], y, 2.25, stroke=1, fill=1)

    margin = 5.5
    gap = 4.0
    footer_y, footer_h = 5.5, 19.0
    calls_y, calls_h = footer_y + footer_h + gap, 81.0
    middle_y, middle_h = calls_y + calls_h + gap, 66.0
    bootstrap_y = middle_y + middle_h + gap
    bootstrap_h = page_height - margin - bootstrap_y

    # 0. Fixed Leiden bootstrap: the only point at which nesting is guaranteed.
    panel(
        margin,
        bootstrap_y,
        page_width - 2 * margin,
        bootstrap_h,
        "0",
        "Bootstrap A_0 and optional X with Leiden",
        accent=INK,
        fill="#FCFCFD",
    )
    draw_text(
        pdf,
        page_width - margin - 7.0,
        bootstrap_y + bootstrap_h - 10.5,
        "level 0 -> level L-1 | initially nested; adjacent levels may coincide",
        6.5,
        FIGURE_FONT_BOLD,
        "right",
        blue,
    )
    top_starts = (13.0, 66.0, 123.0, 190.0, 278.0, 342.0, 366.0, 454.0)
    top_widths = (40.0, 43.0, 53.0, 74.0, 50.0, 10.0, 74.0, 49.0)
    top_labels = (
        "A_0 | X",
        "Leiden",
        "C_0^0 (fine)",
        "quotient + Leiden",
        "C_0^1",
        "...",
        "repeat quotient",
        "C_0^(L-1)",
    )
    for start, width, value in zip(top_starts, top_widths, top_labels):
        draw_text(
            pdf,
            start + width / 2,
            bootstrap_y + 22.0,
            value,
            6.5,
            FIGURE_FONT_BOLD if "C_0" in value or value == "Leiden" else FIGURE_FONT,
            "center",
        )
    matrix_icon(top_starts[0] + 3.0, bootstrap_y + 3.5, 4, 4, 3.1, blue, diagonal=True)
    matrix_icon(top_starts[0] + 24.0, bootstrap_y + 3.5, 3, 4, 3.1, teal)
    pill(top_starts[1], bootstrap_y + 4.0, top_widths[1], 13.0, "Leiden", blue, blue_fill, FIGURE_FONT_BOLD)
    label_strip(
        top_starts[2] + 1.0,
        bootstrap_y + 5.0,
        (8.0, 8.0, 8.0, 8.0, 8.0, 8.0),
        (node_colors[0], node_colors[0], node_colors[1], node_colors[1], node_colors[2], node_colors[3]),
    )
    tiny_graph(
        top_starts[3] + 19.0,
        bootstrap_y + 9.5,
        (node_colors[0], node_colors[1], node_colors[2], node_colors[3]),
    )
    pill(top_starts[3] + 40.0, bootstrap_y + 3.0, 32.0, 13.0, "Leiden", blue, blue_fill, FIGURE_FONT_BOLD)
    label_strip(
        top_starts[4] + 2.0,
        bootstrap_y + 5.0,
        (16.0, 16.0, 16.0),
        (node_colors[0], node_colors[1], node_colors[2]),
    )
    draw_text(pdf, top_starts[5] + 5.0, bootstrap_y + 6.0, "...", 7.0, FIGURE_FONT_BOLD, "center", MID)
    tiny_graph(
        top_starts[6] + 19.0,
        bootstrap_y + 9.5,
        (node_colors[0], node_colors[0], node_colors[2], node_colors[2]),
    )
    pill(top_starts[6] + 40.0, bootstrap_y + 3.0, 32.0, 13.0, "Leiden", blue, blue_fill, FIGURE_FONT_BOLD)
    label_strip(
        top_starts[7] + 4.0,
        bootstrap_y + 5.0,
        (20.0, 20.0),
        (node_colors[0], node_colors[2]),
    )
    for index in range(len(top_starts) - 1):
        flow_arrow(
            top_starts[index] + top_widths[index] + 2.0,
            bootstrap_y + 9.5,
            top_starts[index + 1] - 3.0,
            bootstrap_y + 9.5,
        )

    # 1-3. Localize, freeze every mask from the old state, then prepare labels.
    update_x, update_w = margin, 108.0
    freeze_x, freeze_w = update_x + update_w + gap, 204.0
    prepare_x = freeze_x + freeze_w + gap
    prepare_w = page_width - margin - prepare_x
    panel(update_x, middle_y, update_w, middle_h, "1", "Update and localize", orange, "#FFFFFF")
    panel(freeze_x, middle_y, freeze_w, middle_h, "2", "Freeze all masks before writes", blue, "#FFFFFF")
    panel(prepare_x, middle_y, prepare_w, middle_h, "3", "Prepare labels L-1 to 0", purple, "#FFFFFF")

    # Update endpoints S_t and their radius-r neighborhood B_t.
    left_nodes = ((update_x + 17.0, middle_y + 31.0), (update_x + 31.0, middle_y + 42.0), (update_x + 32.0, middle_y + 21.0))
    right_nodes = ((update_x + 70.0, middle_y + 31.0), (update_x + 84.0, middle_y + 42.0), (update_x + 85.0, middle_y + 21.0))
    for nodes, expanded in ((left_nodes, False), (right_nodes, True)):
        if expanded:
            pdf.setStrokeColor(HexColor(orange))
            pdf.setFillColor(HexColor(orange_fill))
            pdf.setDash(2.0, 1.4)
            pdf.ellipse(update_x + 60.0, middle_y + 15.0, update_x + 96.0, middle_y + 48.0, stroke=1, fill=1)
            pdf.setDash()
        pdf.setStrokeColor(HexColor("#6A747B"))
        pdf.setLineWidth(0.65)
        pdf.line(*nodes[0], *nodes[1])
        pdf.line(*nodes[0], *nodes[2])
        pdf.line(*nodes[1], *nodes[2])
        for node_index, (nx, ny) in enumerate(nodes):
            affected = node_index < 2
            pdf.setStrokeColor(HexColor(orange if affected else "#657078"))
            pdf.setFillColor(HexColor(orange if affected or expanded else "#FFFFFF"))
            pdf.circle(nx, ny, 3.5, stroke=1, fill=1)
    pdf.setStrokeColor(HexColor(red))
    pdf.setDash(2.0, 1.4)
    pdf.line(*left_nodes[0], *left_nodes[1])
    pdf.setDash()
    flow_arrow(update_x + 39.0, middle_y + 31.0, update_x + 55.0, middle_y + 31.0, orange)
    draw_text(pdf, update_x + 24.0, middle_y + 7.0, "S_t endpoints", 6.5, FIGURE_FONT_BOLD, "center", orange)
    draw_text(pdf, update_x + 79.0, middle_y + 7.0, "radius-r B_t", 6.5, FIGURE_FONT_BOLD, "center", orange)

    # Frozen closures at several levels; the orange node represents B_t.
    level_x = freeze_x + 39.0
    for label, y, groups in (
        ("l=0", middle_y + 43.0, ((0, 1), (2, 3), (4, 5))),
        ("l=1", middle_y + 31.0, ((0, 1, 2), (3, 4, 5))),
        ("l=L-1", middle_y + 19.0, ((0, 1, 2, 3), (4, 5))),
    ):
        draw_text(pdf, freeze_x + 7.0, y - 2.3, label, 6.5, FIGURE_FONT_BOLD)
        closed_level(level_x, y, groups, touched=1)
    draw_text(pdf, freeze_x + 151.0, middle_y + 43.0, "close each pre-update", 6.5, align="center")
    draw_text(pdf, freeze_x + 151.0, middle_y + 32.0, "community touching B_t", 6.5, FIGURE_FONT_BOLD, "center", blue)
    draw_text(pdf, freeze_x + 153.0, middle_y + 21.0, "from C_{t-1}; masks fixed", 6.5, align="center")
    draw_text(
        pdf,
        freeze_x + freeze_w / 2,
        middle_y + 6.5,
        "U_t^l = {v : C_{t-1}^l(v) is in C_{t-1}^l(B_t)}",
        6.5,
        FIGURE_FONT_BOLD,
        "center",
        blue,
    )

    # Opposite-direction preparation: coarse-to-fine copies inside frozen masks.
    prep_label_x = prepare_x + 8.0
    prep_strip_x = prepare_x + 36.0
    prep_rows = (
        ("L-1", middle_y + 42.5, ("v2", "v7", "v9")),
        ("...", middle_y + 30.0, ("a", "b", "c")),
        ("0", middle_y + 17.5, ("a", "a", "c")),
    )
    for row_index, (level, y, values) in enumerate(prep_rows):
        draw_text(pdf, prep_label_x, y - 2.3, level, 6.5, FIGURE_FONT_BOLD)
        for value_index, value in enumerate(values):
            bx = prep_strip_x + value_index * 20.0
            pdf.setStrokeColor(HexColor(purple))
            pdf.setFillColor(HexColor(purple_fill if row_index else "#E5DDF1"))
            pdf.roundRect(bx, y - 5.0, 17.0, 10.0, 2.0, stroke=1, fill=1)
            draw_text(pdf, bx + 8.5, y - 2.3, value, 6.5, FIGURE_FONT_BOLD, "center", purple)
    flow_arrow(prepare_x + 103.0, middle_y + 43.5, prepare_x + 103.0, middle_y + 17.0, purple)
    draw_text(pdf, prepare_x + 143.0, middle_y + 43.5, "on B_t: provisional IDs", 6.5, FIGURE_FONT_BOLD, "center", purple)
    draw_text(pdf, prepare_x + 143.0, middle_y + 32.0, "not a guaranteed", 6.5, align="center")
    draw_text(pdf, prepare_x + 143.0, middle_y + 23.0, "singleton namespace", 6.5, align="center")
    draw_text(pdf, prepare_x + 140.0, middle_y + 8.0, "copy down in U_t^l", 6.5, FIGURE_FONT_BOLD, "center", purple)

    # 4. One wide, implementation-ordered backend-call trace.
    panel(
        margin,
        calls_y,
        page_width - 2 * margin,
        calls_h,
        "4",
        "Local backend calls run in the opposite direction: l=0 to L-1",
        teal,
        "#FFFFFF",
    )
    pill(
        page_width - margin - 213.0,
        calls_y + calls_h - 13.0,
        207.0,
        10.0,
        "A_work = A_t[U_t^0,U_t^0] once | boundary omitted",
        orange,
        orange_fill,
        FIGURE_FONT_BOLD,
    )
    stage_starts = (12.0, 87.0, 151.0, 243.0, 322.0, 402.0)
    stage_widths = (64.0, 54.0, 81.0, 68.0, 70.0, 101.0)
    stage_titles = (
        "current groups",
        "membership P_l",
        "contract graph",
        "backend B",
        "project labels",
        "cut working edges",
    )
    for index, (x, width, title) in enumerate(zip(stage_starts, stage_widths, stage_titles)):
        pdf.setStrokeColor(HexColor("#DCE3E6"))
        pdf.setFillColor(HexColor(pale if index % 2 == 0 else "#FFFFFF"))
        pdf.setLineWidth(0.55)
        pdf.roundRect(x, calls_y + 5.0, width, 56.0, 3.0, stroke=1, fill=1)
        draw_text(
            pdf,
            x + width / 2,
            calls_y + 50.0,
            title,
            6.5,
            FIGURE_FONT_BOLD,
            "center",
            teal if index < 5 else red,
        )
    for index in range(len(stage_starts) - 1):
        flow_arrow(
            stage_starts[index] + stage_widths[index] + 2.0,
            calls_y + 33.0,
            stage_starts[index + 1] - 3.0,
            calls_y + 33.0,
            teal,
        )

    # Groups within U_t^l.
    group_x = stage_starts[0] + 8.0
    for group_index, (count, color) in enumerate(((2, node_colors[0]), (3, node_colors[1]), (1, node_colors[2]))):
        gx = group_x + group_index * 17.0
        pdf.setStrokeColor(HexColor(color))
        pdf.setFillColor(HexColor("#FFFFFF"))
        pdf.roundRect(gx - 2.0, calls_y + 25.0, 12.0, 16.0, 5.0, stroke=1, fill=1)
        for node_index in range(count):
            pdf.setFillColor(HexColor(color))
            pdf.circle(gx + 4.0, calls_y + 37.5 - node_index * 4.4, 1.55, stroke=0, fill=1)
    draw_text(pdf, stage_starts[0] + 32.0, calls_y + 10.0, "labels in U_t^l", 6.5, align="center")

    # Sparse one-hot membership matrix P_l.
    matrix_icon(stage_starts[1] + 17.0, calls_y + 22.0, 5, 4, 4.0, teal)
    draw_text(pdf, stage_starts[1] + 27.0, calls_y + 10.0, "one 1 / vertex", 6.5, align="center")

    # Contracted graph and optional mean features.
    tiny_graph(
        stage_starts[2] + stage_widths[2] / 2,
        calls_y + 34.0,
        (node_colors[0], node_colors[1], node_colors[2], node_colors[3]),
    )
    draw_text(pdf, stage_starts[2] + stage_widths[2] / 2, calls_y + 15.5, "Abar=P A_work P^T", 6.5, align="center")
    draw_text(pdf, stage_starts[2] + stage_widths[2] / 2, calls_y + 7.0, "Xbar=group mean", 6.5, FIGURE_FONT_BOLD, "center", teal)

    # No local warm start is passed to the backend.
    backend_cx = stage_starts[3] + stage_widths[3] / 2
    pdf.setStrokeColor(HexColor(teal))
    pdf.setFillColor(HexColor(teal_fill))
    pdf.setLineWidth(1.0)
    pdf.circle(backend_cx, calls_y + 34.0, 10.0, stroke=1, fill=1)
    draw_text(pdf, backend_cx, calls_y + 31.5, "B", 7.4, FIGURE_FONT_BOLD, "center", teal)
    draw_text(pdf, backend_cx, calls_y + 10.0, "warm start = none", 6.5, FIGURE_FONT_BOLD, "center", red)

    # Projection expands contracted labels back through P_l.
    project_x = stage_starts[4] + 11.0
    for column, color in enumerate((node_colors[0], node_colors[0], node_colors[1], node_colors[1], node_colors[2])):
        pdf.setStrokeColor(HexColor("#657078"))
        pdf.setFillColor(HexColor(color))
        pdf.circle(project_x + column * 10.0, calls_y + 34.0, 3.4, stroke=1, fill=1)
    draw_text(pdf, stage_starts[4] + stage_widths[4] / 2, calls_y + 15.5, "through P_l", 6.5, align="center")
    draw_text(pdf, stage_starts[4] + stage_widths[4] / 2, calls_y + 7.0, "prior group IDs", 6.5, FIGURE_FONT_BOLD, "center", purple)

    # Cutting removes structural context for the next level call.
    tiny_graph(
        stage_starts[5] + stage_widths[5] / 2,
        calls_y + 34.0,
        (node_colors[0], node_colors[0], node_colors[2], node_colors[2]),
        cut=True,
    )
    draw_text(pdf, stage_starts[5] + stage_widths[5] / 2, calls_y + 15.5, "outside U_t^l or", 6.5, align="center")
    draw_text(pdf, stage_starts[5] + stage_widths[5] / 2, calls_y + 7.0, "cross-label edges", 6.5, FIGURE_FONT_BOLD, "center", red)

    # 5. The maintained output contract and the two invariants/limitations.
    pdf.setStrokeColor(HexColor(border))
    pdf.setFillColor(HexColor(green_fill))
    pdf.setLineWidth(0.7)
    pdf.roundRect(margin, footer_y, page_width - 2 * margin, footer_h, 4.0, stroke=1, fill=1)
    pdf.setFillColor(HexColor(green))
    pdf.circle(margin + 10.0, footer_y + footer_h / 2, 6.0, stroke=0, fill=1)
    draw_text(pdf, margin + 10.0, footer_y + 7.1, "5", 7.0, FIGURE_FONT_BOLD, "center", "#FFFFFF")
    draw_text(pdf, margin + 21.0, footer_y + 7.0, "Return C_t^0", 7.2, FIGURE_FONT_BOLD, color=INK)
    draw_text(pdf, margin + 126.0, footer_y + 7.0, "+", 7.2, FIGURE_FONT_BOLD, "center", green)
    draw_text(pdf, margin + 136.0, footer_y + 7.0, "outside every U_t^l: labels unchanged", 6.5, FIGURE_FONT_BOLD)
    draw_text(pdf, margin + 353.0, footer_y + 7.0, "!", 7.2, FIGURE_FONT_BOLD, "center", orange)
    draw_text(pdf, margin + 364.0, footer_y + 7.0, "post-update nesting not enforced", 6.5, FIGURE_FONT_BOLD)
    pdf.showPage()
    pdf.save()


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
