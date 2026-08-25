import json

import numpy as np
import pytest

from scripts.paper.ieee_access_repaired_comnetx_72h import run_queue
from scripts.paper.ieee_access_repaired_comnetx_72h.protocol import (
    CORE_DATASETS,
    ValidationError,
    flatten_launcher_payload,
    load_protocol,
    sha256_file,
    stage_map,
    validate_paired_bootstrap,
    validate_series,
)
from scripts.paper.ieee_access_repaired_comnetx_72h.input_manifest import (
    validate_input_manifest,
)
from scripts.paper.ieee_access_repaired_comnetx_72h.run_queue import (
    build_stage_commands,
)
from scripts.paper.ieee_access_repaired_comnetx_72h.validate_campaign import (
    audit_summary,
)


def _series(updates):
    rows = [
        {
            "time": 0.25,
            "modularity": 0.5,
        }
        for _ in range(updates)
    ]
    rows[-1].update({"Final modularity": 0.51, "NMI": 0.62})
    return rows


def test_protocol_prioritizes_core_and_preserves_dsbm_budget():
    protocol = load_protocol()
    stages = stage_map(protocol)

    assert [stage["priority"] for stage in protocol["stages"]] == list(range(1, 10))
    assert all(stage["required"] for stage in protocol["stages"][:4])
    assert protocol["premise"].startswith("Every historical smart-mode result")
    assert stages["stage5_topology_controls"]["minimum_hours_remaining"] == 38
    assert stages["stage6_dsbm"]["minimum_hours_remaining"] == 30
    assert stages["stage6_dsbm"]["estimated_gpu_hours"] == 26.95
    assert "stage6_dsbm" in stages["stage7_dfleiden_interface"]["dependencies"]
    assert "stage7_dfleiden_interface" in stages[
        "stage7_s2cag_interface"
    ]["dependencies"]


def test_queue_builds_registered_repetition_counts_and_smart_only_dsbm(tmp_path):
    common = {
        "campaign_dir": tmp_path / "campaign",
        "paths_config": tmp_path / "paths.json",
        "dsbm_root": tmp_path / "dsbm",
        "repetitions": None,
    }

    assert len(build_stage_commands("stage1_correctness_smoke", **common)) == 7
    assert len(build_stage_commands("stage2_core_short", **common)) == 5
    assert len(build_stage_commands("stage3_mechanism", **common)) == 2
    assert len(build_stage_commands("stage4_long_core", **common)) == 1
    df_commands = build_stage_commands("stage7_dfleiden_interface", **common)
    assert len(df_commands) == 6
    assert df_commands[-1].name == "dfleiden_smart_long_coverage"

    dsbm = build_stage_commands("stage6_dsbm", **common)[0].command
    assert dsbm[dsbm.index("--modes") + 1] == "smart"
    assert dsbm[dsbm.index("--batch-suffix") + 1] == "100_batches"
    assert dsbm[dsbm.index("--max-changes") + 1 : dsbm.index("--methods")] == [
        "290",
        "1450",
    ]


def test_launcher_payload_validation_keeps_full_and_repaired_rows_separate():
    payload = {}
    for algorithm in ("leidenalg-naive", "leidenalg-L:3-r:1-gpu"):
        payload[algorithm] = {
            f"{dataset}-sym": {"machine": {"999:10": _series(10)}}
            for dataset in CORE_DATASETS
        }

    flat = flatten_launcher_payload(
        payload,
        expected_datasets=CORE_DATASETS,
        expected_batch="999:10",
    )

    assert set(flat) == {
        (algorithm, dataset)
        for algorithm in ("leidenalg-naive", "leidenalg-L:3-r:1-gpu")
        for dataset in CORE_DATASETS
    }
    summary = validate_series(
        flat[("leidenalg-L:3-r:1-gpu", "dyn_pubmed")],
        dataset="dyn_pubmed",
        batch="999:10",
    )
    assert summary["updates"] == 10
    assert summary["total_time"] == pytest.approx(2.5)


def test_launcher_payload_rejects_non_cartesian_algorithm_dataset_grid():
    payload = {
        "leidenalg-naive": {
            f"{dataset}-sym": {"machine": {"999:10": _series(10)}}
            for dataset in CORE_DATASETS
        },
        "leidenalg-L:3-r:1-gpu": {
            "dyn_pubmed-sym": {"machine": {"999:10": _series(10)}}
        },
    }
    with pytest.raises(ValidationError, match="Cartesian"):
        flatten_launcher_payload(
            payload,
            expected_datasets=CORE_DATASETS,
            expected_batch="999:10",
        )


def test_stream_audit_gate_rejects_any_namespace_collision(tmp_path):
    report = {
        "schema": "comnetx_stream_invariant_audit_v1",
        "dataset": "dyn_pubmed-sym",
        "batch_strategy": "999:10",
        "method": "leidenalg",
        "mode": "smart",
        "depth": 3,
        "radius": 1,
        "resolution": 1.0,
        "aggregation_mode": "sum",
        "force_undirected": True,
        "device": "cuda:0",
        "bootstrap": {
            "strategy_has_bootstrap_batch": True,
            "implementation": "launcher._compute_launch_initial_partition",
            "initial_batch_number": "999",
            "hierarchy_cache_schema": "parent_quotient_v1",
            "campaign_cache_enabled": True,
        },
        "initial_refinement": {"all_adjacent_refine": True},
        "summary": {
            "updates_audited": 10,
            "pre_update_refinement_failure_updates": [],
            "post_update_refinement_failure_updates": [],
            "scope_collision_before_updates": [],
            "scope_collision_after_updates": [],
            "outside_entry_write_updates": [],
            "final_reported_modularity": 0.7,
        },
    }
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    assert audit_summary(path, "dyn_pubmed")["updates_audited"] == 10

    report["summary"]["scope_collision_after_updates"] = [2]
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValidationError, match="invariant gate failed"):
        audit_summary(path, "dyn_pubmed")


def test_optional_repeat_bounds_are_explicit(tmp_path):
    common = {
        "campaign_dir": tmp_path / "campaign",
        "paths_config": tmp_path / "paths.json",
        "dsbm_root": None,
    }

    assert len(
        build_stage_commands(
            "stage4_long_repeatability", repetitions=1, **common
        )
    ) == 1
    s2_commands = build_stage_commands(
        "stage7_s2cag_interface", repetitions=5, **common
    )
    assert len(s2_commands) == 6
    assert s2_commands[-1].name == "s2cag_feature_modes_coverage"
    with pytest.raises(ValueError, match="one or two"):
        build_stage_commands(
            "stage4_long_repeatability", repetitions=3, **common
        )


def test_dsbm_stage_has_no_unregistered_positional_argument(tmp_path):
    commands = build_stage_commands(
        "stage6_dsbm",
        campaign_dir=tmp_path / "campaign",
        paths_config=tmp_path / "paths.json",
        dsbm_root=tmp_path / "dsbm",
        repetitions=None,
    )

    command = commands[0].command
    output_index = command.index("--output-dir")
    assert command[output_index + 1] == "{attempt_dir}"
    assert command[output_index + 2] == "--name"


def test_paired_bootstrap_requires_identical_level_zero_partition(tmp_path):
    campaign = tmp_path / "campaign"
    cache = campaign / "bootstrap-cache"
    attempt = campaign / "stages" / "stage" / "command" / "attempt-01"
    cache.mkdir(parents=True)
    attempt.mkdir(parents=True)
    full_name = "dyn_pubmed-sym_b:999_by_leidenalg.npz"
    smart_name = (
        "dyn_pubmed-sym_b:999_by_leidenalg_d:3_parent_quotient_v1.npz"
    )
    row0 = np.asarray([0, 0, 2, 2], dtype=np.int64)
    np.savez_compressed(cache / full_name, partition=row0, mod=0.5)
    np.savez_compressed(
        cache / smart_name,
        partition=np.stack([row0, np.zeros(4, dtype=np.int64), np.zeros(4, dtype=np.int64)]),
        mod=0.5,
        schema=np.asarray("parent_quotient_v1"),
    )
    metadata = {
        "bootstrap_cache_sha256": {
            full_name: sha256_file(cache / full_name),
            smart_name: sha256_file(cache / smart_name),
        }
    }
    (attempt / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    result = validate_paired_bootstrap(
        campaign,
        attempt,
        dataset="dyn_pubmed",
        initial_batch=999,
    )
    assert result["full_depth_1"]["modularity"] == 0.5

    bad_row0 = np.asarray([0, 1, 2, 2], dtype=np.int64)
    np.savez_compressed(
        cache / smart_name,
        partition=np.stack(
            [bad_row0, np.zeros(4, dtype=np.int64), np.zeros(4, dtype=np.int64)]
        ),
        mod=0.5,
        schema=np.asarray("parent_quotient_v1"),
    )
    metadata["bootstrap_cache_sha256"][smart_name] = sha256_file(cache / smart_name)
    (attempt / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValidationError, match="different level-zero"):
        validate_paired_bootstrap(
            campaign,
            attempt,
            dataset="dyn_pubmed",
            initial_batch=999,
        )


def test_input_manifest_detects_content_change_even_with_same_path(tmp_path):
    data = tmp_path / "stream.dat"
    data.write_text("first", encoding="utf-8")
    stat = data.stat()
    manifest = {
        "schema": "comnetx-ieee-access-real-inputs-v1",
        "files": [
            {
                "path": str(data),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": sha256_file(data),
                "uses": ["dyn_pubmed/999:10"],
            }
        ],
    }
    validate_input_manifest(manifest, verify_content=True)
    data.write_text("other", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        validate_input_manifest(manifest, verify_content=True)


def test_runtime_identity_requires_live_hardware_and_environment_match(
    tmp_path,
    monkeypatch,
):
    hardware = {"schema": "hardware", "gpu": "test-gpu"}
    environment = {"schema": "environment", "packages": [["torch", "x"]]}
    hardware_path = tmp_path / "hardware.json"
    environment_path = tmp_path / "environment.json"
    hardware_path.write_text(json.dumps(hardware), encoding="utf-8")
    environment_path.write_text(json.dumps(environment), encoding="utf-8")
    manifest = {
        "hardware": {
            "filename": hardware_path.name,
            "sha256": sha256_file(hardware_path),
        },
        "environment": {
            "filename": environment_path.name,
            "sha256": sha256_file(environment_path),
        },
    }
    monkeypatch.setattr(run_queue, "collect_hardware", lambda: hardware)
    monkeypatch.setattr(
        run_queue,
        "collect_measurement_environment",
        lambda: environment,
    )

    identity = run_queue.assert_runtime_identity(manifest, tmp_path)
    assert identity["hardware_sha256"] == sha256_file(hardware_path)
    assert identity["environment_sha256"] == sha256_file(environment_path)

    monkeypatch.setattr(
        run_queue,
        "collect_measurement_environment",
        lambda: {**environment, "packages": [["torch", "changed"]]},
    )
    with pytest.raises(RuntimeError, match="environment differs"):
        run_queue.assert_runtime_identity(manifest, tmp_path)
