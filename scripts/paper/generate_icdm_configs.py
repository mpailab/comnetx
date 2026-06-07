"""Generate experiment configs for the ICDM paper iteration.

The script writes JSON configs under ``conf/paper_icdm`` by default. They are
designed for the existing ``scripts/launch.py`` runner and deliberately keep
the primary paper datasets separate from optional larger add-ons.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PRIMARY_DATASETS = [
    "dyn_cora",
    "dyn_acm",
    "dyn_citeseer",
    "patent",
    "dyn_pubmed",
    "arxivmath",
]

ABLATION_DATASETS = [
    "dyn_cora",
    "dyn_pubmed",
    "arxivmath",
]

FEATURE_ABLATION_DATASETS = [
    "dyn_cora",
    "dyn_pubmed",
]

OPTIONAL_SCALE_DATASETS = [
    "arxivcs",
    "dyn_ogbn-arxiv",
    "arxivphy",
]


def base_config() -> dict:
    return {
        "VERBOSE": 1,
        "CATCH_ERRORS": True,
        "USE_TIMESTAMP_SUFFIX": True,
        "GROUND_TRUTH_METRICS": True,
        "FORCE_UNDIRECTED": True,
        "USE_GPU": True,
    }


def write_config(path: Path, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def build_configs(include_scale: bool) -> dict[str, dict]:
    datasets = PRIMARY_DATASETS + (OPTIONAL_SCALE_DATASETS if include_scale else [])

    configs: dict[str, dict] = {}

    cfg = base_config()
    cfg.update(
        {
            "DATASETS": datasets,
            "BATCHES": ["999:10"],
            "BASELINES": ["leidenalg", "dfleiden"],
            "MODES": ["naive", "dynamic", "smart"],
            "SMART_PARAMS_GRID": {
                "smart_subcoms_depth": [3],
                "smart_neighborhood_step": [1],
                "aggregation_mode": ["sum"],
            },
        }
    )
    configs["main_topology.json"] = cfg

    cfg = base_config()
    cfg.update(
        {
            "DATASETS": datasets,
            "BATCHES": ["999:10"],
            "BASELINES": ["s2cag", "dmon"],
            "MODES": ["naive", "smart"],
            "FEATURE_MODES": ["dataset", "random"],
            "RANDOM_FEATURE_DIM": 64,
            "BASELINE_ITERATIONS": [10],
            "SMART_PARAMS_GRID": {
                "smart_subcoms_depth": [3],
                "smart_neighborhood_step": [1],
                "aggregation_mode": ["norm"],
            },
        }
    )
    configs["main_gnn_feature.json"] = cfg

    cfg = base_config()
    cfg.update(
        {
            "DATASETS": datasets,
            "BATCHES": ["999:10"],
            "BASELINES": ["flmig", "prgpt:infomap", "prgpt:locale"],
            "MODES": ["naive", "smart"],
            "BASELINE_ITERATIONS": [10],
            "SMART_PARAMS_GRID": {
                "smart_subcoms_depth": [3],
                "smart_neighborhood_step": [1],
                "aggregation_mode": ["sum"],
            },
        }
    )
    configs["compatibility_classic.json"] = cfg

    cfg = base_config()
    cfg.update(
        {
            "DATASETS": datasets,
            "BATCHES": ["999:10"],
            "BASELINES": ["magi"],
            "MODES": ["naive", "smart"],
            "FEATURE_MODES": ["dataset", "random"],
            "RANDOM_FEATURE_DIM": 64,
            "BASELINE_ITERATIONS": [10],
            "SMART_PARAMS_GRID": {
                "smart_subcoms_depth": [3],
                "smart_neighborhood_step": [1],
                "aggregation_mode": ["norm"],
            },
        }
    )
    configs["compatibility_magi.json"] = cfg

    cfg = base_config()
    cfg.update(
        {
            "DATASETS": datasets,
            "BATCHES": ["999:10"],
            "BASELINES": ["mfc"],
            "MODES": ["dynamic"],
            "FEATURE_MODES": ["dataset"],
            "BASELINE_ITERATIONS": [100],
        }
    )
    configs["compatibility_mfc_dynamic.json"] = cfg

    cfg = base_config()
    cfg.update(
        {
            "DATASETS": datasets,
            "BATCHES": ["999:10"],
            "BASELINES": ["mfc"],
            "MODES": ["smart"],
            "FEATURE_MODES": ["random"],
            "RANDOM_FEATURE_DIM": 64,
            "BASELINE_ITERATIONS": [100],
            "SMART_PARAMS_GRID": {
                "smart_subcoms_depth": [3],
                "smart_neighborhood_step": [1],
                "aggregation_mode": ["norm"],
            },
        }
    )
    configs["compatibility_mfc_local.json"] = cfg

    cfg = base_config()
    cfg.update(
        {
            "DATASETS": ABLATION_DATASETS,
            "BATCHES": ["999:10"],
            "BASELINES": ["leidenalg"],
            "MODES": ["naive", "smart"],
            "SMART_PARAMS_GRID": {
                "smart_subcoms_depth": [1, 2, 3, 4],
                "smart_neighborhood_step": [0, 1, 2],
                "aggregation_mode": ["sum"],
            },
        }
    )
    configs["ablation_topology_radius_depth.json"] = cfg

    cfg = base_config()
    cfg.update(
        {
            "DATASETS": FEATURE_ABLATION_DATASETS,
            "BATCHES": ["999:10"],
            "BASELINES": ["s2cag"],
            "MODES": ["naive", "smart"],
            "FEATURE_MODES": ["dataset", "random", "onehot"],
            "RANDOM_FEATURE_DIM": 64,
            "BASELINE_ITERATIONS": [10],
            "SMART_PARAMS_GRID": {
                "smart_subcoms_depth": [3],
                "smart_neighborhood_step": [1],
                "aggregation_mode": ["norm"],
            },
        }
    )
    configs["ablation_feature_modes.json"] = cfg

    cfg = base_config()
    cfg.update(
        {
            "DATASETS": FEATURE_ABLATION_DATASETS,
            "BATCHES": ["999:10"],
            "BASELINES": ["s2cag"],
            "MODES": ["smart"],
            "FEATURE_MODES": ["random"],
            "RANDOM_FEATURE_DIM": 64,
            "BASELINE_ITERATIONS": [10],
            "SMART_PARAMS_GRID": {
                "smart_subcoms_depth": [3],
                "smart_neighborhood_step": [1],
                "aggregation_mode": ["norm", "sum"],
            },
        }
    )
    configs["ablation_feature_aggregation.json"] = cfg

    cfg = base_config()
    cfg.update(
        {
            "DATASETS": FEATURE_ABLATION_DATASETS,
            "BATCHES": ["999:10"],
            "BASELINES": ["s2cag"],
            "MODES": ["smart"],
            "FEATURE_MODES": ["random"],
            "RANDOM_FEATURE_DIM": 64,
            "BASELINE_ITERATIONS": [10],
            "SMART_PARAMS_GRID": {
                "smart_subcoms_depth": [3],
                "smart_neighborhood_step": [0, 1, 2],
                "aggregation_mode": ["norm"],
            },
        }
    )
    configs["ablation_gnn_radius_light.json"] = cfg

    cfg = base_config()
    cfg.update(
        {
            "DATASETS": ["dyn_pubmed", "arxivmath"],
            "BATCHES": ["999:50", "999:100"],
            "BASELINES": ["leidenalg", "dfleiden"],
            "MODES": ["naive", "dynamic", "smart"],
            "SMART_PARAMS_GRID": {
                "smart_subcoms_depth": [3],
                "smart_neighborhood_step": [1],
                "aggregation_mode": ["sum"],
            },
        }
    )
    configs["long_horizon_topology.json"] = cfg

    for seed in [1, 2, 3, 4, 5]:
        cfg = base_config()
        cfg.update(
            {
                "DATASETS": PRIMARY_DATASETS,
                "BATCHES": ["999:10"],
                "BASELINES": ["s2cag", "dmon"],
                "MODES": ["smart"],
                "FEATURE_MODES": ["random"],
                "RANDOM_FEATURE_DIM": 64,
                "RANDOM_FEATURE_SEED": seed,
                "BASELINE_ITERATIONS": [10],
                "SMART_PARAMS_GRID": {
                    "smart_subcoms_depth": [3],
                    "smart_neighborhood_step": [1],
                    "aggregation_mode": ["norm"],
                },
            }
        )
        configs[f"stability_gnn_seed_{seed}.json"] = cfg

    return configs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="conf/paper_icdm",
        help="Directory where generated JSON configs will be written.",
    )
    parser.add_argument(
        "--include-scale",
        action="store_true",
        help="Include optional larger datasets in the main configs.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    for name, config in build_configs(include_scale=args.include_scale).items():
        write_config(output_dir / name, config)
        print(output_dir / name)


if __name__ == "__main__":
    main()
