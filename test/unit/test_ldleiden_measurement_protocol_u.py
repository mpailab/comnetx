from pathlib import Path

import numpy as np
import pytest

from scripts.paper.ieee_access_ldleiden_72h import input_manifest as ld_input_manifest
from scripts.paper.ieee_access_ldleiden_72h import run_protocol

from scripts.paper.ieee_access_ldleiden_72h.protocol import (
    ValidationError,
    SOURCE_FILES,
    aggregate_run_summaries,
    bootstrap_cache_reference,
    hierarchy_is_nested,
    load_protocol,
    partition_sha256,
    sha256_file,
    protocol_fingerprint,
    sha256_json,
    write_json,
    validate_bootstrap_cache_file,
    validate_launcher_payload,
    validate_partition_semantics,
)
from scripts.paper.ieee_access_ldleiden_72h.input_manifest import (
    validate_input_manifest,
)
from scripts.paper.ieee_access_ldleiden_72h.validate_results import (
    bootstrap_hash_for_dataset,
    validate_campaign,
    validate_window_attestation,
)


def _row(*, split=True, final=False):
    bootstrap_digest = partition_sha256(np.array([0, 0, 2], dtype=np.int64))
    row = {
        "modularity": 0.5,
        "time": 0.4,
        "optimization_time": 0.4,
        "update_time": 0.1,
        "end_to_end_time": 0.55,
        "timing_split_supported": split,
        "priming_audit_supported": True,
        "priming_partition_relation_preserved": True,
        "bootstrap_reference_sha256": bootstrap_digest,
        "post_priming_partition_sha256": bootstrap_digest,
    }
    if final:
        row.update({"Final modularity": 0.52, "NMI": 0.61})
    return row


def _payload(phase):
    datasets = {}
    for dataset in phase["datasets"]:
        updates = phase["expected_updates"][dataset]
        series = [_row() for _ in range(updates - 1)] + [_row(final=True)]
        datasets[f"{dataset}-sym"] = {
            "test-machine": {phase["batch_strategy"]: series}
        }
    return {"ldleiden-dynamic": datasets}


def test_window_attestation_rejects_completed_attempt_after_deadline(tmp_path):
    metadata_path = tmp_path / "metadata.json"
    metadata = {
        "status": "completed",
        "measurement_window_id": "window-001",
        "window_deadline_epoch": 1100,
        "started_at_utc": "1970-01-01T00:16:40Z",
        "finished_at_utc": "1970-01-01T00:18:18Z",
    }
    validate_window_attestation(metadata, metadata_path)

    metadata["finished_at_utc"] = "1970-01-01T00:18:23Z"
    with pytest.raises(ValidationError, match="finished outside"):
        validate_window_attestation(metadata, metadata_path)


def test_protocol_preregisters_disjoint_smoke_and_measured_repetitions():
    protocol = load_protocol()
    phases = {phase["cli_name"]: phase for phase in protocol["phases"]}

    assert phases["smoke"]["included_in_analysis"] is False
    assert phases["smoke"]["repetitions"] == 1
    assert phases["short"]["repetitions"] == 5
    assert phases["long"]["repetitions"] == 3
    assert protocol["common"]["num_jobs"] == 1
    assert protocol["common"]["num_jobs_source"] == "verified AlgorithmOptions default"
    assert protocol["common"]["num_jobs_explicitly_passed_by_launcher"] is False
    assert protocol["clock_policy"]["principal_field"] == "optimization_time"


def test_source_fingerprint_covers_complete_python_tree():
    expected = set((Path(__file__).resolve().parents[2] / "src").rglob("*.py"))
    assert expected <= set(SOURCE_FILES)


def test_launcher_payload_requires_and_summarizes_split_clocks():
    phase = next(
        phase for phase in load_protocol()["phases"] if phase["cli_name"] == "short"
    )

    summary = validate_launcher_payload(_payload(phase), phase)

    assert set(summary) == {"dyn_pubmed", "arxivmath"}
    assert summary["dyn_pubmed"]["updates"] == 10
    assert summary["dyn_pubmed"]["optimization_seconds"] == pytest.approx(4.0)
    assert summary["dyn_pubmed"]["update_seconds"] == pytest.approx(1.0)
    assert summary["dyn_pubmed"]["final_modularity"] == 0.52
    assert summary["dyn_pubmed"]["final_nmi"] == 0.61


def test_launcher_payload_rejects_legacy_combined_clock():
    phase = next(
        phase for phase in load_protocol()["phases"] if phase["cli_name"] == "short"
    )
    payload = _payload(phase)
    payload["ldleiden-dynamic"]["dyn_pubmed-sym"]["test-machine"]["999:10"][0][
        "timing_split_supported"
    ] = False

    with pytest.raises(ValidationError, match="split timing is unavailable"):
        validate_launcher_payload(payload, phase)


def test_aggregate_keeps_values_and_sample_standard_deviation():
    rows = [
        {
            "updates": 10,
            "optimization_seconds": value,
            "update_seconds": 1.0,
            "end_to_end_seconds": value + 1.1,
            "framework_overhead_seconds": 0.1,
            "final_modularity": 0.5,
            "final_nmi": 0.6,
        }
        for value in (2.0, 4.0)
    ]

    summary = aggregate_run_summaries(rows)

    assert summary["repetitions"] == 2
    assert summary["optimization_seconds"]["mean"] == 3.0
    assert summary["optimization_seconds"]["sample_sd"] == pytest.approx(2**0.5)
    assert summary["optimization_seconds"]["values"] == [2.0, 4.0]


def test_bootstrap_hash_selects_the_matched_initial_graph():
    filename, digest = bootstrap_hash_for_dataset(
        {
            "dyn_pubmed-sym_b:999_by_leidenalg.npz": "short-hash",
            "dyn_pubmed-sym_b:9_by_leidenalg.npz": "long-hash",
        },
        "dyn_pubmed",
        "999:10",
    )

    assert filename == "dyn_pubmed-sym_b:999_by_leidenalg.npz"
    assert digest == "short-hash"


def test_bootstrap_semantics_require_canonical_flat_level_zero(tmp_path):
    path = tmp_path / "dyn_pubmed-sym_b:999_by_leidenalg.npz"
    partition = np.array([0, 0, 2, 2], dtype=np.int64)
    np.savez_compressed(path, partition=partition, mod=0.5)

    semantics = validate_bootstrap_cache_file(path)

    assert semantics["rank"] == 1
    assert semantics["nested"] is True
    assert semantics["level_zero_sha256"] == partition_sha256(partition)
    reference = bootstrap_cache_reference(
        tmp_path,
        {path.name: semantics["file_sha256"]},
        "dyn_pubmed",
        "999:10",
        semantics["level_zero_sha256"],
    )
    assert reference["filename"] == path.name

    np.savez_compressed(
        path,
        partition=np.array([7, 7, 9, 9], dtype=np.int64),
        mod=0.5,
    )
    with pytest.raises(ValidationError, match="not canonical"):
        validate_bootstrap_cache_file(path)


def test_hierarchy_semantics_reject_non_nested_or_noncanonical_levels():
    nested = np.array([[0, 0, 2, 2], [0, 0, 0, 0]], dtype=np.int64)
    assert hierarchy_is_nested(nested)
    assert validate_partition_semantics(nested)["levels"] == 2

    non_nested = np.array([[0, 0, 2, 2], [0, 1, 0, 1]], dtype=np.int64)
    with pytest.raises(ValidationError, match="not canonical|not nested"):
        validate_partition_semantics(non_nested)


def test_input_manifest_detects_content_change_even_when_size_is_unchanged(tmp_path):
    stream = tmp_path / "stream.bin"
    stream.write_bytes(b"abc")
    stat = stream.stat()
    manifest = {
        "schema": "comnetx-ieee-access-ldleiden-real-inputs-v1",
        "files": [
            {
                "path": str(stream),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": sha256_file(stream),
                "uses": ["dyn_pubmed/999:10"],
            }
        ],
    }
    validate_input_manifest(manifest, verify_content=True)

    stream.write_bytes(b"xyz")
    # Restore mtime to demonstrate that content hashing, not metadata alone,
    # protects every pre/post-attempt identity check.
    import os

    os.utime(stream, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(ValueError, match="content changed"):
        validate_input_manifest(manifest, verify_content=True)


def test_preflight_manifest_covers_every_registered_dataset_batch(
    tmp_path,
    monkeypatch,
):
    paths_config = tmp_path / "paths.json"
    paths_config.write_text("{}\n", encoding="utf-8")

    def fake_dataset_files(_paths_config, dataset, batch):
        path = tmp_path / f"{dataset}-{batch.replace(':', '_')}.bin"
        path.write_text(f"{dataset}/{batch}", encoding="utf-8")
        return "fake", [path]

    monkeypatch.setattr(ld_input_manifest, "_dataset_files", fake_dataset_files)
    manifest = ld_input_manifest.build_input_manifest(paths_config)
    observed = {
        (entry["dataset"], entry["batch"])
        for entry in manifest["dataset_batch_combinations"]
    }
    assert observed == {
        *( (dataset, "999:10") for dataset in load_protocol()["phases"][0]["datasets"] ),
        ("dyn_pubmed", "9:500"),
        ("arxivmath", "9:500"),
    }
    validate_input_manifest(manifest, verify_content=True)


def test_campaign_identity_locks_hardware_container_source_and_inputs(
    tmp_path,
    monkeypatch,
):
    protocol = load_protocol()
    paths_config = tmp_path / "paths.json"
    paths_config.write_text("{}\n", encoding="utf-8")
    stream = tmp_path / "stream.bin"
    stream.write_bytes(b"stream")
    stream_stat = stream.stat()
    hardware = {"schema": "hardware", "cpu": "fixed"}
    backend = {"num_jobs_verified": True, "wheel": "fixed"}
    runtime = run_protocol.collect_runtime_identity(
        hardware=hardware,
        backend=backend,
    )
    input_manifest = {
        "schema": "comnetx-ieee-access-ldleiden-real-inputs-v1",
        "paths_config_sha256": sha256_file(paths_config),
        "files": [
            {
                "path": str(stream),
                "size": stream_stat.st_size,
                "mtime_ns": stream_stat.st_mtime_ns,
                "sha256": sha256_file(stream),
                "uses": ["fake/999:10"],
            }
        ],
    }
    registrations = {}
    for key, filename, payload in (
        ("hardware", "hardware.json", hardware),
        ("runtime_identity", "runtime_identity.json", runtime),
        ("real_input_manifest", "real_input_manifest.json", input_manifest),
    ):
        path = tmp_path / filename
        write_json(path, payload)
        registrations[key] = {"filename": filename, "sha256": sha256_file(path)}
    manifest = {
        "protocol_id": protocol["protocol_id"],
        "fingerprint": protocol_fingerprint(protocol),
        "paths_config": {"sha256": sha256_file(paths_config)},
        "backend": backend,
        **registrations,
    }
    monkeypatch.setattr(run_protocol, "backend_metadata", lambda: backend)
    monkeypatch.setattr(run_protocol, "collect_hardware", lambda: hardware)

    identity = run_protocol.assert_campaign_identity(
        manifest,
        tmp_path,
        paths_config,
        protocol,
        verify_input_content=True,
    )
    assert identity["hardware_sha256"] == registrations["hardware"]["sha256"]

    monkeypatch.setattr(
        run_protocol,
        "collect_hardware",
        lambda: {"schema": "hardware", "cpu": "changed"},
    )
    with pytest.raises(RuntimeError, match="hardware differs"):
        run_protocol.assert_campaign_identity(
            manifest,
            tmp_path,
            paths_config,
            protocol,
            verify_input_content=True,
        )


def test_campaign_validator_requires_pre_post_identity_and_bootstrap_semantics(
    tmp_path,
):
    protocol = load_protocol()
    phase = next(item for item in protocol["phases"] if item["cli_name"] == "smoke")
    fingerprint = protocol_fingerprint(protocol)
    paths_digest = "a" * 64
    backend = {
        "num_jobs_verified": True,
        "num_jobs_observed": 1,
        "num_jobs_source": "AlgorithmOptions default",
        "num_jobs_explicitly_passed_by_launcher": False,
    }
    manifest = {
        "schema": "comnetx-ieee-access-ldleiden-campaign-v1",
        "campaign_id": "test-campaign",
        "protocol_id": protocol["protocol_id"],
        "fingerprint": fingerprint,
        "backend": backend,
        "worker_policy": {
            "num_jobs": 1,
            "num_jobs_source": "verified AlgorithmOptions default",
            "num_jobs_explicitly_passed_by_launcher": False,
            "thread_environment": {
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            },
        },
        "paths_config": {"sha256": paths_digest},
    }
    artifacts = {
        "hardware": {"schema": "hardware"},
        "runtime_identity": {
            "backend": backend,
            "hardware_sha256": sha256_json({"schema": "hardware"}),
        },
        "real_input_manifest": {
            "schema": "comnetx-ieee-access-ldleiden-real-inputs-v1",
            "paths_config_sha256": paths_digest,
            "files": [{"path": "/archived/input", "sha256": "b" * 64}],
        },
    }
    for key, payload in artifacts.items():
        path = tmp_path / f"{key}.json"
        write_json(path, payload)
        manifest[key] = {"filename": path.name, "sha256": sha256_file(path)}
    write_json(tmp_path / "manifest.json", manifest)

    payload = _payload(phase)
    result_dir = tmp_path / phase["id"] / "repeat-01" / "attempt-01"
    result_dir.mkdir(parents=True)
    result_path = result_dir / "ldleiden_smoke_999_10_now.json"
    write_json(result_path, payload)
    cache_dir = tmp_path / "bootstrap-cache"
    cache_dir.mkdir()
    partition = np.array([0, 0, 2], dtype=np.int64)
    for dataset in phase["datasets"]:
        np.savez_compressed(
            cache_dir / f"{dataset}-sym_b:999_by_leidenalg.npz",
            partition=partition,
            mod=0.5,
        )
    cache_hashes = {
        path.name: sha256_file(path) for path in sorted(cache_dir.glob("*.npz"))
    }
    summaries = validate_launcher_payload(payload, phase)
    bootstrap_semantics = {
        dataset: bootstrap_cache_reference(
            cache_dir,
            cache_hashes,
            dataset,
            phase["batch_strategy"],
            summaries[dataset]["bootstrap_reference_sha256"],
        )
        for dataset in phase["datasets"]
    }
    expected_identity = {
        "fingerprint_sha256": sha256_json(fingerprint),
        "hardware_sha256": manifest["hardware"]["sha256"],
        "runtime_identity_sha256": manifest["runtime_identity"]["sha256"],
        "real_input_manifest_sha256": manifest["real_input_manifest"]["sha256"],
    }
    write_json(
        result_dir / "metadata.json",
        {
            "status": "completed",
            "phase_id": phase["id"],
            "role": phase["role"],
            "included_in_analysis": False,
            "repeat": 1,
            "thread_environment": {
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            },
            "identity_before": expected_identity,
            "identity_after": expected_identity,
            "result_sha256": sha256_file(result_path),
            "bootstrap_cache_sha256": cache_hashes,
            "bootstrap_semantics": bootstrap_semantics,
            "validated_summary": summaries,
        },
    )

    report = validate_campaign(tmp_path, allow_partial=True)
    assert report["status"] == "valid_partial"

    metadata = run_protocol.read_json(result_dir / "metadata.json")
    metadata["identity_after"] = {**expected_identity, "hardware_sha256": "0" * 64}
    write_json(result_dir / "metadata.json", metadata)
    with pytest.raises(ValidationError, match="post-run.*identity mismatch"):
        validate_campaign(tmp_path, allow_partial=True)
