import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.paper.ieee_access_ldleiden_24h_parallel import run_shard
from scripts.paper.ieee_access_ldleiden_72h.protocol import SOURCE_FILES


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAUNCH_DIR = (
    PROJECT_ROOT
    / "scripts"
    / "paper"
    / "ieee_access_ldleiden_24h_parallel"
)
COMMON = LAUNCH_DIR / "_common.bash"
RUNNER = LAUNCH_DIR / "run_shard.py"
EXPECTED_ASSIGNMENTS = (
    ("01_short_repeat_1.sh", "01", "short", 1),
    ("02_short_repeat_2.sh", "02", "short", 2),
    ("03_short_repeat_3.sh", "03", "short", 3),
    ("04_short_repeat_4.sh", "04", "short", 4),
    ("05_short_repeat_5.sh", "05", "short", 5),
    ("06_long_repeat_1.sh", "06", "long", 1),
    ("07_long_repeat_2.sh", "07", "long", 2),
    ("08_long_repeat_3.sh", "08", "long", 3),
)


@pytest.mark.short
def test_exactly_eight_parallel_wrappers_are_executable_and_shell_valid():
    wrappers = tuple(path.name for path in sorted(LAUNCH_DIR.glob("*.sh")))
    assert wrappers == tuple(item[0] for item in EXPECTED_ASSIGNMENTS)

    for filename, _shard_id, phase, repetition in EXPECTED_ASSIGNMENTS:
        path = LAUNCH_DIR / filename
        text = path.read_text(encoding="utf-8")

        assert text.startswith("#!/usr/bin/env bash\n")
        assert path.stat().st_mode & 0o111
        assert "_common.bash" in text
        assert re.findall(
            r"(?m)^\s*run_shard\s+['\"]?(short|long)['\"]?\s+['\"]?([1-5])['\"]?\s*$",
            text,
        ) == [(phase, str(repetition))]

        lowered = text.lower()
        assert "docker exec" not in lowered
        assert "docker compose" not in lowered
        assert "nohup" not in lowered

    completed = subprocess.run(
        [
            "bash",
            "-n",
            str(COMMON),
            *(str(LAUNCH_DIR / item[0]) for item in EXPECTED_ASSIGNMENTS),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def _parse_list_line(line: str) -> tuple[str, str, int]:
    tokens = re.sub(r"[^a-z0-9]+", " ", line.lower()).split()
    shard_ids = [token for token in tokens if re.fullmatch(r"0[1-8]", token)]
    phases = [token for token in tokens if token in {"short", "long"}]
    assert len(shard_ids) == 1, line
    assert len(phases) == 1, line

    assert "repeat" in tokens, line
    repeat_position = tokens.index("repeat") + 1
    assert repeat_position < len(tokens), line
    assert re.fullmatch(r"[1-5]", tokens[repeat_position]), line
    return shard_ids[0], phases[0], int(tokens[repeat_position])


@pytest.mark.short
def test_runner_lists_the_five_short_and_three_long_shards_once():
    listed = subprocess.run(
        [sys.executable, str(RUNNER), "--list"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert listed.returncode == 0, listed.stderr

    lines = [line for line in listed.stdout.splitlines() if line.strip()]
    assert len(lines) == len(EXPECTED_ASSIGNMENTS)
    observed = tuple(_parse_list_line(line) for line in lines)
    expected = tuple(
        (shard_id, phase, repetition)
        for _filename, shard_id, phase, repetition in EXPECTED_ASSIGNMENTS
    )
    assert observed == expected
    assert [phase for _shard_id, phase, _repeat in observed].count("short") == 5
    assert [phase for _shard_id, phase, _repeat in observed].count("long") == 3


@pytest.mark.short
def test_parallel_runtime_files_are_part_of_the_sealed_source_identity():
    expected = {
        COMMON,
        RUNNER,
        LAUNCH_DIR / "validate_parallel.py",
        *(LAUNCH_DIR / item[0] for item in EXPECTED_ASSIGNMENTS),
        *(PROJECT_ROOT / "datasets-info" / "json").glob("*.json"),
        PROJECT_ROOT
        / "scripts"
        / "paper"
        / "ieee_access_72h_launch"
        / "sync_bootstrap.py",
    }
    assert expected <= set(SOURCE_FILES)


@pytest.mark.short
def test_parallel_window_is_registered_once_and_rejects_tampering(tmp_path):
    paths_config = tmp_path / "paths.json"
    paths_config.write_text("{}\n", encoding="utf-8")
    root = tmp_path / "parallel"
    manifest = run_shard.initialize_parallel_manifest(
        root,
        run_id="parallel-test",
        expected_git_sha="a" * 40,
        paths_config=paths_config,
        budget_hours=24.0,
    )
    run_shard.validate_window(manifest["window"], expected_budget_hours=24.0)
    assert (root / "parallel_manifest_registration.json").is_file()
    assert manifest["bootstrap_authority"] == run_shard.SHARED_BOOTSTRAP_POLICY
    assert manifest["evidence_scope"] == run_shard.EVIDENCE_SCOPE
    assert manifest["evidence_scope"]["paired_comnetx_validated"] is False
    assert manifest["evidence_scope"]["matched_speedup_claim_allowed"] is False

    manifest["window"]["deadline_epoch"] += 1.0
    run_shard.write_json(root / "parallel_manifest.json", manifest)
    with pytest.raises(RuntimeError, match="changed after its initial registration"):
        run_shard.initialize_parallel_manifest(
            root,
            run_id="parallel-test",
            expected_git_sha="a" * 40,
            paths_config=paths_config,
            budget_hours=24.0,
        )


@pytest.mark.short
def test_parallel_window_recovers_without_resetting_after_interrupted_registration(
    tmp_path,
):
    paths_config = tmp_path / "paths.json"
    paths_config.write_text("{}\n", encoding="utf-8")
    root = tmp_path / "parallel"
    original = run_shard.initialize_parallel_manifest(
        root,
        run_id="parallel-test",
        expected_git_sha="a" * 40,
        paths_config=paths_config,
        budget_hours=24.0,
    )
    manifest_path = root / "parallel_manifest.json"
    pending_path = root / run_shard.PARALLEL_PENDING_MANIFEST
    manifest_path.replace(pending_path)
    (root / "parallel_manifest_registration.json").unlink()

    recovered = run_shard.initialize_parallel_manifest(
        root,
        run_id="parallel-test",
        expected_git_sha="a" * 40,
        paths_config=paths_config,
        budget_hours=24.0,
    )

    assert recovered == original
    assert manifest_path.is_file()
    assert not pending_path.exists()
    run_shard.validate_parallel_registration(root, manifest_path)


@pytest.mark.short
@pytest.mark.parametrize(
    "window",
    (
        {"budget_hours": 25.0, "started_epoch": 1.0, "deadline_epoch": 90001.0},
        {"budget_hours": 24.0, "started_epoch": 1.0, "deadline_epoch": 2.0},
        {"budget_hours": True, "started_epoch": 1.0, "deadline_epoch": 2.0},
    ),
)
def test_parallel_window_rejects_invalid_budget_or_arithmetic(window):
    with pytest.raises(RuntimeError):
        run_shard.validate_window(window)


@pytest.mark.short
def test_shared_bootstrap_plan_covers_exactly_the_registered_eight_inputs():
    assert run_shard.load_protocol()["protocol_id"] == (
        "ieee-access-ldleiden-24h-standalone-v1"
    )
    requirements = run_shard._bootstrap_requirements()
    assert requirements == {
        "dyn_cora/999": "999:10",
        "dyn_acm/999": "999:10",
        "dyn_citeseer/999": "999:10",
        "patent/999": "999:10",
        "dyn_pubmed/999": "999:10",
        "arxivmath/999": "999:10",
        "dyn_pubmed/9": "9:500",
        "arxivmath/9": "9:500",
    }
    assert run_shard.SHARED_BOOTSTRAP_POLICY["producer_shard"] == "01"
    assert run_shard.SHARED_BOOTSTRAP_POLICY["relative_path"] == "shared-bootstrap"


@pytest.mark.short
def test_shared_bootstrap_authority_is_atomic_reusable_and_byte_copied(
    tmp_path, monkeypatch
):
    paths_config = tmp_path / "paths.json"
    paths_config.write_text("{}\n", encoding="utf-8")
    root = tmp_path / "parallel"
    parallel = run_shard.initialize_parallel_manifest(
        root,
        run_id="parallel-test",
        expected_git_sha="a" * 40,
        paths_config=paths_config,
        budget_hours=24.0,
    )
    producer = root / "shards" / run_shard.campaign_name("01")
    producer.mkdir(parents=True)
    run_shard.write_json(
        producer / "manifest.json",
        {
            "fingerprint": {"test": True},
            "backend": {"name": "test"},
            "hardware": {"filename": "hardware.json", "sha256": "1" * 64},
            "runtime_identity": {
                "filename": "runtime.json",
                "sha256": "2" * 64,
            },
            "real_input_manifest": {
                "filename": "inputs.json",
                "sha256": "3" * 64,
            },
        },
    )
    monkeypatch.setattr(run_shard.os, "sched_getaffinity", lambda _pid: {7})
    monkeypatch.setattr(
        run_shard,
        "cpu_topology",
        lambda affinity: {
            "logical_cpu": affinity[0],
            "physical_package_id": 0,
            "core_id": 7,
            "thread_siblings_list": "7",
        },
    )
    generated = []

    def fake_generate(cache_dir, _paths_config, dataset, batch_strategy):
        initial = batch_strategy.split(":", 1)[0]
        filename = f"{dataset}_b:{initial}_by_leidenalg.npz"
        path = cache_dir / filename
        np.savez_compressed(
            path,
            partition=np.asarray([0, 0, 2], dtype=np.int64),
            mod=np.asarray(0.25),
        )
        semantics = run_shard.validate_bootstrap_cache_file(path)
        generated.append(f"{dataset}/{initial}")
        return {
            "dataset": dataset,
            "batch_strategy": batch_strategy,
            "filename": filename,
            "file_sha256": semantics["file_sha256"],
            "level_zero_sha256": semantics["level_zero_sha256"],
            "modularity": semantics["modularity"],
            "vertices": semantics["vertices"],
        }

    monkeypatch.setattr(
        run_shard, "_generate_shared_bootstrap_entry", fake_generate
    )
    authority = run_shard.seal_shared_bootstrap(
        root, producer, paths_config, parallel
    )
    assert set(generated) == set(run_shard._bootstrap_requirements())
    assert (root / "shared-bootstrap").is_dir()
    assert not (root / "shared-bootstrap.initializing").exists()
    shared_manifest = run_shard.read_json(root / "shared-bootstrap" / "manifest.json")
    assert shared_manifest["generation"]["rng_control"].startswith("backend-default")
    assert "sealed NPZ bytes" in shared_manifest["generation"][
        "reproducibility_policy"
    ]
    heartbeat = run_shard.read_json(
        root / "markers" / "bootstrap-producer-heartbeat.json"
    )
    assert heartbeat["state"] == "sealed"

    monkeypatch.setattr(
        run_shard,
        "_generate_shared_bootstrap_entry",
        lambda *_args, **_kwargs: pytest.fail("sealed authority was regenerated"),
    )
    assert run_shard.seal_shared_bootstrap(
        root, producer, paths_config, parallel
    ) == authority

    consumer = root / "shards" / run_shard.campaign_name("06")
    consumer.mkdir(parents=True)
    report = run_shard.seed_bootstrap_cache(root, authority, consumer, "06")
    assert set(report["entries"]) == {"dyn_pubmed/9", "arxivmath/9"}
    for entry in report["entries"].values():
        assert entry["source_sha256"] == entry["target_sha256"]


@pytest.mark.short
def test_missing_bootstrap_producer_heartbeat_fails_after_bounded_wait(
    tmp_path, monkeypatch
):
    started = 1000.0
    monkeypatch.setattr(
        run_shard.time,
        "time",
        lambda: started + run_shard.BOOTSTRAP_HEARTBEAT_STALE_SECONDS + 1.0,
    )
    with pytest.raises(RuntimeError, match="did not publish a heartbeat"):
        run_shard._validate_bootstrap_heartbeat(
            tmp_path,
            "a" * 64,
            campaign_started_epoch=started,
        )


@pytest.mark.short
def test_wait_barrier_fails_immediately_on_shell_failure_marker(
    tmp_path, monkeypatch
):
    paths_config = tmp_path / "paths.json"
    paths_config.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(run_shard, "RESULTS_ROOT", tmp_path)
    root = tmp_path / "failed-run"
    run_shard.initialize_parallel_manifest(
        root,
        run_id="failed-run",
        expected_git_sha="b" * 40,
        paths_config=paths_config,
        budget_hours=24.0,
    )
    markers = root / "markers"
    markers.mkdir()
    (markers / "failure-03.shell").write_text("exit_code=1\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="parallel shards failed"):
        run_shard.wait_for_stage(
            SimpleNamespace(run_id="failed-run", stage="preflight")
        )


@pytest.mark.short
def test_bundle_documents_exact_container_mapping_and_fail_fast_guards():
    readme = (LAUNCH_DIR / "README.md").read_text(encoding="utf-8")
    common = COMMON.read_text(encoding="utf-8")
    containers = (
        "dev_drobyshev3",
        "dev_egorov2",
        "dev_drobyshev2",
        "dev_bokov",
        "dev_konovalov",
        "dev_uporova",
        "dev_egorov",
        "dev_drobyshev",
    )
    for (filename, _shard_id, _phase, _repeat), container in zip(
        EXPECTED_ASSIGNMENTS, containers
    ):
        assert readme.count(f"docker exec -d {container} ") == 1
        assert filename in readme
    assert "REPAIRED_CAMPAIGN_ID" not in common
    assert "REPAIRED_CAMPAIGN_ID" not in readme
    assert "shared-bootstrap" in readme
    assert "matched_speedup_claim_allowed" in run_shard.EVIDENCE_SCOPE
    assert run_shard.EVIDENCE_SCOPE["matched_speedup_claim_allowed"] is False
    assert "BOOTSTRAP_SOURCE" not in common
    assert "flock -n" in common
    assert "failure-$shard_id.shell" in common
    assert "forward_signal TERM 143" in common
    assert "forward_signal HUP 129" in common
