"""Generate the final full-picture cn69 measurement queue.

This batch is built from the current ``results/registry/all_results.json``:
do not repeat the already ingested PubMed/Arxiv workload-profile and
closure/contraction cells from ``results/paper_icdm/4``. The core queue covers
only paper-facing gaps: missing long-horizon NMI, the required ``dyn_cora``
small-control profile block, and the random/hub DSBM solver curves.

Bad-locality graphs are kept as optional follow-up probes. They are useful for
the operating-envelope discussion, but they are not part of the current main
table/figure grid.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "conf" / "paper_icdm" / "cn69_final_20260528"
SCRIPT_DIR = PROJECT_ROOT / "scripts" / "paper" / "cn69_final_20260528"


def base_config() -> dict:
    return {
        "VERBOSE": 1,
        "CATCH_ERRORS": True,
        "USE_TIMESTAMP_SUFFIX": True,
        "GROUND_TRUTH_METRICS": True,
        "FORCE_UNDIRECTED": True,
        "USE_GPU": True,
    }


def long_horizon_nmi_config() -> dict:
    cfg = base_config()
    cfg.update(
        {
            "DATASETS": ["dyn_pubmed"],
            "BATCHES": ["999:100"],
            "BASELINES": ["ldleiden"],
            "MODES": ["naive", "smart"],
            "SMART_PARAMS_GRID": {
                "smart_subcoms_depth": [3],
                "smart_neighborhood_step": [1],
                "aggregation_mode": ["sum"],
            },
        }
    )
    return cfg


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def common_header(default_timeout: str) -> str:
    return f"""#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/../../.."

export PARENT_HOSTNAME="${{PARENT_HOSTNAME:-cn69}}"
export PYTHONUNBUFFERED=1

PATHS_CONFIG="${{PATHS_CONFIG:-datasets-info/paths/cn69.json}}"
CACHE_DIR="${{CACHE_DIR:-/home/dev/communities}}"
DSBM_ROOT="${{DSBM_ROOT:-datasets-sbm}}"
SKIP_REGISTRY="${{SKIP_REGISTRY:-results/registry/all_results.json}}"
LOG_DIR="${{LOG_DIR:-output}}"
TIMEOUT="${{TIMEOUT:-{default_timeout}}}"
STAMP="$(date +%Y%m%d_%H%M%S)"
FAILED=0
mkdir -p "$LOG_DIR"
mkdir -p results/paper_icdm
"""


def dsbm_function() -> str:
    return r"""
run_dsbm() {
  local title="$1"
  local name="$2"
  shift 2
  local log="$LOG_DIR/${name}_${STAMP}.log"
  echo "[${title}] $(date -Is) running DSBM stress with timeout=$TIMEOUT on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-container-bound}"
  timeout --kill-after=2m "$TIMEOUT" python scripts/paper/run_dsbm_stress.py \
    "$@" \
    --methods leidenalg dfleiden \
    --modes naive smart dynamic \
    --use-gpu \
    --catch-errors \
    --skip-registry "$SKIP_REGISTRY" \
    --output-dir results/paper_icdm \
    --name "${name}_${STAMP}" \
    2>&1 | tee "$log"
  local status="${PIPESTATUS[0]}"
  if [[ "$status" -ne 0 ]]; then
    echo "[${title}] exited with status $status; see $log"
    FAILED=1
  fi
}
"""


def dsbm_calls(*, prefix: str, runs: list[tuple[str, str, str, list[int]]]) -> str:
    lines = [dsbm_function()]
    for label, regime, suffix, max_changes in runs:
        mc_args = " ".join(str(item) for item in max_changes)
        safe_label = label.lower().replace(" ", "_").replace("/", "_")
        lines.append(
            f"""
run_dsbm "{label}" "{prefix}_{safe_label}" \\
  --root "$DSBM_ROOT" \\
  --batch-suffix {suffix} \\
  --regimes {regime} \\
  --max-changes {mc_args}
"""
        )
    lines.append("\nexit \"$FAILED\"\n")
    return "".join(lines)


def long_horizon_nmi_block() -> str:
    return r"""
need_long_horizon_nmi() {
  python - <<'PY'
import json
from pathlib import Path

registry = Path("results/registry/all_results.json")
targets = {
    ("dyn_pubmed", "999:100", "ldleiden", "naive"),
    ("dyn_pubmed", "999:100", "ldleiden", "smart"),
}
if not registry.exists():
    print("registry missing; running the NMI completion block")
    raise SystemExit(0)
rows = json.loads(registry.read_text(encoding="utf-8"))
seen = set()
for row in rows:
    if row.get("measurement_type") != "experiment":
        continue
    key = (
        row.get("base_dataset"),
        str(row.get("batch_strategy")),
        row.get("method"),
        row.get("mode"),
    )
    if key in targets and row.get("final_nmi") is not None:
        seen.add(key)
missing = sorted(targets - seen)
if missing:
    print("missing long-horizon NMI cells:")
    for item in missing:
        print("  ", item)
    raise SystemExit(0)
print("all target long-horizon NMI cells already exist; skipping")
raise SystemExit(1)
PY
}

if need_long_horizon_nmi; then
  log="$LOG_DIR/gpu0_long_horizon_missing_nmi_${STAMP}.log"
  echo "[long-horizon NMI] $(date -Is) rerunning only the two dyn_pubmed 999:100 LD-Leiden cells that lack final_nmi"
  timeout --kill-after=2m "${NMI_TIMEOUT:-2h}" python scripts/launch.py \
    conf/paper_icdm/cn69_final_20260528/long_horizon_missing_nmi_dyn_pubmed_999100.json \
    --paths-config "$PATHS_CONFIG" \
    2>&1 | tee "$log"
  status="${PIPESTATUS[0]}"
  if [[ "$status" -ne 0 ]]; then
    echo "[long-horizon NMI] exited with status $status; see $log"
    FAILED=1
  fi
fi
"""


def dyn_cora_block() -> str:
    return r"""
need_dyn_cora_profile() {
  python - <<'PY'
import json
from pathlib import Path

registry = Path("results/registry/all_results.json")
targets = {
    ("dyn_cora", "999:50", "leidenalg", None, "full"),
    ("dyn_cora", "999:50", "dfleiden", None, "full"),
    ("dyn_cora", "999:50", "s2cag", "random", "full"),
    ("dyn_cora", "999:50", "leidenalg", None, "no_closure"),
    ("dyn_cora", "999:50", "leidenalg", None, "no_contraction"),
}
if not registry.exists():
    print("registry missing; running dyn_cora small-control block")
    raise SystemExit(0)
rows = json.loads(registry.read_text(encoding="utf-8"))
seen = set()
for row in rows:
    if row.get("measurement_type") != "workload_profile":
        continue
    key = (
        row.get("base_dataset"),
        str(row.get("batch_strategy")),
        row.get("method"),
        row.get("feature_mode"),
        row.get("variant", "full"),
    )
    if key in targets:
        seen.add(key)
missing = sorted(targets - seen, key=str)
if missing:
    print("missing dyn_cora workload-profile cells:")
    for item in missing:
        print("  ", item)
    raise SystemExit(0)
print("all target dyn_cora workload-profile cells already exist; skipping")
raise SystemExit(1)
PY
}

if need_dyn_cora_profile; then
  echo "[dyn_cora small control] $(date -Is) using the already prepared extra27 script"
  TIMEOUT="${DYN_CORA_TIMEOUT:-2h}" scripts/paper/cn69_after3_20260527/extra27_dyn_cora_small_control.sh
  status="$?"
  if [[ "$status" -ne 0 ]]; then
    echo "[dyn_cora small control] exited with status $status"
    FAILED=1
  fi
fi
"""


def profile_script(
    *,
    script_id: str,
    title: str,
    name: str,
    dataset: str,
    method: str,
    variants: list[str],
    smart_radius: int,
    max_updates: int,
    timeout: str,
    feature_modes: list[str] | None = None,
    baseline_iter: int | None = None,
    random_seed: int | None = None,
    aggregation_mode: str = "sum",
) -> str:
    feature_args = ""
    if feature_modes:
        feature_args = f" \\\n  --feature-modes {' '.join(feature_modes)}"
    iter_arg = ""
    if baseline_iter is not None:
        iter_arg = f" \\\n  --baseline-iter {baseline_iter}"
    seed_arg = ""
    if random_seed is not None:
        seed_arg = f" \\\n  --random-feature-seed {random_seed}"

    return f"""{common_header(timeout)}

log="$LOG_DIR/{script_id}_{name}_${{STAMP}}.log"
echo "[{title}] $(date -Is) running optional bad-locality workload profile with timeout=$TIMEOUT on CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-container-bound}}"
timeout --kill-after=2m "$TIMEOUT" python scripts/paper/profile_smart_workload.py \\
  --datasets {dataset} \\
  --batches 999:10 \\
  --methods {method}{feature_args}{iter_arg}{seed_arg} \\
  --variants {' '.join(variants)} \\
  --smart-depth 3 \\
  --smart-radius {smart_radius} \\
  --aggregation-mode {aggregation_mode} \\
  --max-updates {max_updates} \\
  --use-gpu \\
  --force-undirected \\
  --ground-truth-metrics \\
  --catch-errors \\
  --paths-config "$PATHS_CONFIG" \\
  --cache-dir "$CACHE_DIR" \\
  --output-dir results/paper_icdm \\
  --name "{name}_${{STAMP}}" \\
  2>&1 | tee "$log"
status="${{PIPESTATUS[0]}}"
if [[ "$status" -ne 0 ]]; then
  echo "[{title}] exited with status $status; see $log"
fi
exit "$status"
"""


def build_core_scripts() -> dict[str, str]:
    max_changes_all = [290, 1450, 2900, 14500, 29000]
    scripts = {
        "gpu0_long_nmi_and_dsbm_random_small.sh": (
            common_header("8h")
            + long_horizon_nmi_block()
            + dsbm_calls(
                prefix="gpu0",
                runs=[
                    ("DSBM random 5 batches", "random", "5_batches", max_changes_all),
                    ("DSBM random 10 batches", "random", "10_batches", max_changes_all),
                ],
            )
        ),
        "gpu1_dyn_cora_and_dsbm_hubs_small.sh": (
            common_header("8h")
            + dyn_cora_block()
            + dsbm_calls(
                prefix="gpu1",
                runs=[
                    ("DSBM hubs 5 batches", "hubs", "5_batches", max_changes_all),
                    ("DSBM hubs 10 batches", "hubs", "10_batches", max_changes_all),
                ],
            )
        ),
        "gpu2_dsbm_random_100_mc290.sh": (
            common_header("10h")
            + dsbm_calls(
                prefix="gpu2",
                runs=[("DSBM random 100 batches mc290", "random", "100_batches", [290])],
            )
        ),
        "gpu3_dsbm_hubs_100_mc290.sh": (
            common_header("10h")
            + dsbm_calls(
                prefix="gpu3",
                runs=[("DSBM hubs 100 batches mc290", "hubs", "100_batches", [290])],
            )
        ),
        "gpu4_dsbm_random_100_mc1450.sh": (
            common_header("12h")
            + dsbm_calls(
                prefix="gpu4",
                runs=[("DSBM random 100 batches mc1450", "random", "100_batches", [1450])],
            )
        ),
        "gpu5_dsbm_hubs_100_mc1450.sh": (
            common_header("12h")
            + dsbm_calls(
                prefix="gpu5",
                runs=[("DSBM hubs 100 batches mc1450", "hubs", "100_batches", [1450])],
            )
        ),
        "gpu6_dsbm_random_100_mid_high.sh": (
            common_header("12h")
            + dsbm_calls(
                prefix="gpu6",
                runs=[
                    (
                        "DSBM random 100 batches mid/high",
                        "random",
                        "100_batches",
                        [2900, 14500, 29000],
                    )
                ],
            )
        ),
        "gpu7_dsbm_hubs_100_mid_high.sh": (
            common_header("12h")
            + dsbm_calls(
                prefix="gpu7",
                runs=[
                    (
                        "DSBM hubs 100 batches mid/high",
                        "hubs",
                        "100_batches",
                        [2900, 14500, 29000],
                    )
                ],
            )
        ),
    }
    return scripts


def build_extra_scripts() -> dict[str, str]:
    scripts: dict[str, str] = {}
    idx = 1
    for dataset in ["dyn_blogcatalog", "dyn_wikics", "brain"]:
        dense = dataset == "brain"
        topo_updates = 3 if dense else 10
        heavy_updates = 2 if dense else 5
        gnn_updates = 1 if dense else 3
        entries = [
            ("leiden_r1", "Leiden r1 full", "leidenalg", ["full"], 1, topo_updates, None, None, "sum"),
            ("dfleiden_r1", "DF-Leiden r1 full", "dfleiden", ["full"], 1, topo_updates, None, None, "sum"),
            ("leiden_r0", "Leiden r0 full", "leidenalg", ["full"], 0, topo_updates, None, None, "sum"),
            ("dfleiden_r0", "DF-Leiden r0 full", "dfleiden", ["full"], 0, topo_updates, None, None, "sum"),
            ("leiden_r2", "Leiden r2 full", "leidenalg", ["full"], 2, heavy_updates, None, None, "sum"),
            ("dfleiden_r2", "DF-Leiden r2 full", "dfleiden", ["full"], 2, heavy_updates, None, None, "sum"),
            (
                "leiden_closure_variants",
                "Leiden closure variants",
                "leidenalg",
                ["no_closure", "no_contraction"],
                1,
                heavy_updates,
                None,
                None,
                "sum",
            ),
            (
                "s2cag_random",
                "S2CAG random full",
                "s2cag",
                ["full"],
                1,
                gnn_updates,
                ["random"],
                42,
                "norm",
            ),
        ]
        for suffix, title, method, variants, radius, updates, feature_modes, seed, aggregation in entries:
            script_id = f"extra{idx:02d}"
            name = f"final_badloc_{dataset}_{suffix}_{script_id}"
            scripts[f"{script_id}_badloc_{dataset}_{suffix}.sh"] = profile_script(
                script_id=script_id,
                title=f"{title} on {dataset}",
                name=name,
                dataset=dataset,
                method=method,
                variants=variants,
                smart_radius=radius,
                max_updates=updates,
                timeout="6h",
                feature_modes=feature_modes,
                baseline_iter=10 if method == "s2cag" else None,
                random_seed=seed,
                aggregation_mode=aggregation,
            )
            idx += 1
    assert len(scripts) == 24
    return scripts


README = """# cn69 Final Full-Picture Measurement Queue 2026-05-28

This package is generated after checking `results/registry/all_results.json`.
The current registry already contains the PubMed/Arxiv workload-profile,
closure/contraction, and random-seed blocks from `results/paper_icdm/4`, so
this queue does not repeat them.

Core paper-facing gaps:

- `tab:long-horizon` / `fig:long-horizon-curves`: newer cn69 rows already
  supply NMI for the Leiden and DF-Leiden `dyn_pubmed` `999:100` cells, but
  two LD-Leiden rows still lack final NMI. GPU0 reruns only those two LD-Leiden
  cells with `GROUND_TRUTH_METRICS=true`; it skips the block if the registry
  already contains NMI.
- `tab:contracted-workload`, `fig:workload-speedup`, `tab:closure-ablation`,
  and `tab:breakdown`: GPU1 calls the already prepared
  `scripts/paper/cn69_after3_20260527/extra27_dyn_cora_small_control.sh`.
  It skips the block if the registry already contains the required `dyn_cora`
  workload-profile cells.
- `fig:update-size`: random DSBM is missing completely, and hub-centered DSBM
  is missing except for the single existing `hubs/mc1450/100_batches`
  `leidenalg-naive` row. All DSBM core scripts pass
  `--skip-registry results/registry/all_results.json`, so already ingested
  DSBM cells are skipped.

Balanced core scripts:

- `gpu0_long_nmi_and_dsbm_random_small.sh`: missing LD-Leiden long-horizon NMI
  plus random DSBM `5_batches` and `10_batches` over all update sizes.
- `gpu1_dyn_cora_and_dsbm_hubs_small.sh`: required `dyn_cora` small-control
  profile plus hub DSBM `5_batches` and `10_batches`.
- `gpu2_dsbm_random_100_mc290.sh`: random DSBM `100_batches`, `mc=290`.
- `gpu3_dsbm_hubs_100_mc290.sh`: hub DSBM `100_batches`, `mc=290`.
- `gpu4_dsbm_random_100_mc1450.sh`: random DSBM `100_batches`, `mc=1450`.
- `gpu5_dsbm_hubs_100_mc1450.sh`: hub DSBM `100_batches`, `mc=1450`.
- `gpu6_dsbm_random_100_mid_high.sh`: random DSBM `100_batches`,
  `mc=2900,14500,29000`.
- `gpu7_dsbm_hubs_100_mid_high.sh`: hub DSBM `100_batches`,
  `mc=2900,14500,29000`.

The split uses the completed community-internal DSBM timings as a rough guide:
GPU0/GPU1 combine short mandatory completion blocks with the short DSBM
granularities, while GPU2-GPU7 split the longer `100_batches` slices.

Run from cn69 after rebuilding the registry from any newly copied results:

```bash
python3 scripts/paper/collect_results_registry.py
docker exec -d dev_bokov bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_final_20260528/gpu0_long_nmi_and_dsbm_random_small.sh'
docker exec -d dev_uporova bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_final_20260528/gpu1_dyn_cora_and_dsbm_hubs_small.sh'
docker exec -d dev_konovalov bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_final_20260528/gpu2_dsbm_random_100_mc290.sh'
docker exec -d dev_egorov bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_final_20260528/gpu3_dsbm_hubs_100_mc290.sh'
docker exec -d dev_egorov2 bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_final_20260528/gpu4_dsbm_random_100_mc1450.sh'
docker exec -d dev_drobyshev bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_final_20260528/gpu5_dsbm_hubs_100_mc1450.sh'
docker exec -d dev_drobyshev2 bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_final_20260528/gpu6_dsbm_random_100_mid_high.sh'
docker exec -d dev_drobyshev3 bash -lc 'cd /home/dev/users/bokov/comnetx && scripts/paper/cn69_final_20260528/gpu7_dsbm_hubs_100_mid_high.sh'
```

Optional bad-locality follow-ups:

The 24 `extra*.sh` scripts are deliberately not part of the core paper-facing
queue. They probe whether the method remains admissible when locality is poor
on `dyn_blogcatalog`, `dyn_wikics`, and `brain`. Run them only on containers
that finish the core queue early. Each script is a small workload-profile
probe over `999:10` and writes checkpointed JSON under `results/paper_icdm/`.

After jobs finish or time out, rebuild the registry:

```bash
python3 scripts/paper/collect_results_registry.py
```
"""


def write_scripts(scripts: dict[str, str]) -> None:
    SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    for name, content in scripts.items():
        path = SCRIPT_DIR / name
        path.write_text(content, encoding="utf-8")
        path.chmod(0o755)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    configs = {
        "long_horizon_missing_nmi_dyn_pubmed_999100.json": long_horizon_nmi_config(),
    }
    scripts = {}
    scripts.update(build_core_scripts())
    scripts.update(build_extra_scripts())

    if args.dry_run:
        print(json.dumps({"configs": sorted(configs), "scripts": sorted(scripts)}, indent=2))
        return

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    for name, cfg in configs.items():
        write_json(CONFIG_DIR / name, cfg)
    write_scripts(scripts)
    (SCRIPT_DIR / "README.md").write_text(README, encoding="utf-8")


if __name__ == "__main__":
    main()
