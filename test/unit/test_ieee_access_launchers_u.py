import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pytest

from scripts.paper.ieee_access_72h_launch import sync_bootstrap
from scripts.paper.ieee_access_72h_launch.sync_bootstrap import synchronize
from scripts.paper.ieee_access_72h_launch.validate_campaign_pair import (
    assert_clock_creation_safe,
    validate_pair,
)
from scripts.paper.ieee_access_ldleiden_72h.protocol import (
    load_protocol as load_ld_protocol,
)
from scripts.paper.ieee_access_repaired_comnetx_72h.protocol import (
    load_protocol as load_repaired_protocol,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAUNCH_DIR = PROJECT_ROOT / "scripts" / "paper" / "ieee_access_72h_launch"
COMMON = LAUNCH_DIR / "_common.bash"
EXPECTED_LAUNCHERS = (
    "01_preflight.sh",
    "02_smoke_gates.sh",
    "03_short_core.sh",
    "04_mechanism.sh",
    "05_long_core.sh",
    "06_repeatability_and_controls.sh",
    "07_dsbm.sh",
    "08_interfaces_and_final_validation.sh",
)


@pytest.mark.short
def test_exactly_eight_ordered_container_launchers_are_shell_valid():
    launchers = tuple(path.name for path in sorted(LAUNCH_DIR.glob("*.sh")))
    assert launchers == EXPECTED_LAUNCHERS

    for filename in launchers:
        path = LAUNCH_DIR / filename
        text = path.read_text(encoding="utf-8")
        assert text.startswith("#!/usr/bin/env bash\n")
        assert path.stat().st_mode & 0o111
        assert "_common.bash" in text
        assert "prepare_launcher" in text
        assert "docker exec" not in text
        assert "docker compose" not in text
        assert "nohup" not in text

    completed = subprocess.run(
        ["bash", "-n", str(COMMON), *(str(LAUNCH_DIR / name) for name in launchers)],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.short
def test_launchers_cover_the_registered_stage_and_phase_maps():
    repaired_ids = {stage["id"] for stage in load_repaired_protocol()["stages"]}
    assert repaired_ids == {
        "stage1_correctness_smoke",
        "stage2_core_short",
        "stage3_mechanism",
        "stage4_long_core",
        "stage4_long_repeatability",
        "stage5_topology_controls",
        "stage6_dsbm",
        "stage7_dfleiden_interface",
        "stage7_s2cag_interface",
    }
    assert {
        phase["cli_name"] for phase in load_ld_protocol()["phases"]
    } == {"smoke", "short", "long"}

    texts = {
        name: (LAUNCH_DIR / name).read_text(encoding="utf-8")
        for name in EXPECTED_LAUNCHERS
    }
    assert "--preflight" in texts["01_preflight.sh"]
    assert "--preflight-only" in texts["01_preflight.sh"]
    assert "run_repaired_stage stage1_correctness_smoke" in texts["02_smoke_gates.sh"]
    assert "run_ld_phase smoke" in texts["02_smoke_gates.sh"]
    assert "run_repaired_stage stage2_core_short" in texts["03_short_core.sh"]
    assert "run_ld_phase short" in texts["03_short_core.sh"]
    assert "run_repaired_stage stage3_mechanism" in texts["04_mechanism.sh"]
    assert "run_repaired_stage stage4_long_core" in texts["05_long_core.sh"]
    assert "run_ld_phase long" in texts["05_long_core.sh"]
    assert 'repeat_stage="stage4_long_repeatability"' in texts[
        "06_repeatability_and_controls.sh"
    ]
    assert 'control_stage="stage5_topology_controls"' in texts[
        "06_repeatability_and_controls.sh"
    ]
    assert 'stage="stage6_dsbm"' in texts["07_dsbm.sh"]
    assert 'df_stage="stage7_dfleiden_interface"' in texts[
        "08_interfaces_and_final_validation.sh"
    ]
    assert 's2cag_stage="stage7_s2cag_interface"' in texts[
        "08_interfaces_and_final_validation.sh"
    ]
    assert "--dsbm-root" in texts["07_dsbm.sh"]
    assert "--repetitions 2 --stop-with-hours-left 32" in texts[
        "06_repeatability_and_controls.sh"
    ]
    assert "--repetitions 1" in texts["08_interfaces_and_final_validation.sh"]
    assert "if [[ -f \"$BUDGET_STATE_FILE\" ]]" in texts["01_preflight.sh"]

    common_text = COMMON.read_text(encoding="utf-8")
    assert "budget_deadline_epoch" in common_text
    assert "--deadline-epoch \"$deadline_epoch\"" in common_text

    listed = subprocess.run(
        [
            sys.executable,
            "scripts/paper/ieee_access_repaired_comnetx_72h/run_queue.py",
            "--list",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert listed.returncode == 0, listed.stderr
    for stage in repaired_ids:
        assert stage in listed.stdout

    repaired_help = subprocess.run(
        [
            sys.executable,
            "scripts/paper/ieee_access_repaired_comnetx_72h/run_queue.py",
            "--help",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert repaired_help.returncode == 0, repaired_help.stderr
    assert "--deadline-epoch" in repaired_help.stdout

    help_result = subprocess.run(
        [
            sys.executable,
            "scripts/paper/ieee_access_ldleiden_72h/run_protocol.py",
            "--help",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert help_result.returncode == 0, help_result.stderr
    assert "{all,smoke,short,long}" in help_result.stdout


def _policy_run(
    tmp_path: Path,
    filename: str,
    *,
    hours: float,
    decision: str = "go",
    resolved: tuple[str, ...] = (),
    completed_repeats: int = 0,
    completed_repeat_error: bool = False,
    finalizable: tuple[str, ...] = (),
    expected_returncode: int = 0,
    expected_stderr: str | None = None,
) -> list[str]:
    text = (LAUNCH_DIR / filename).read_text(encoding="utf-8")
    source_pattern = re.compile(r'^source .*_common\.bash"$', re.MULTILINE)
    text, substitutions = source_pattern.subn(f'source "{COMMON}"', text, count=1)
    assert substitutions == 1
    repeat_helper = (
        "repeatability_completed_count() { return 1; }"
        if completed_repeat_error
        else f"repeatability_completed_count() {{ printf '%s\\n' '{completed_repeats}'; }}"
    )
    injection = f"""
prepare_launcher() {{ :; }}
require_preflight_campaigns() {{ :; }}
require_repaired_stage_validated() {{ :; }}
require_ld_phase_completed() {{ :; }}
require_optional_stage_can_be_deferred() {{ :; }}
require_all_optional_stages_resolved() {{ :; }}
hours_left() {{ printf '%s\\n' '{hours}'; }}
hours_at_least() {{ awk -v value="$1" -v threshold="$2" 'BEGIN {{ exit !(value >= threshold) }}'; }}
stage2_breadth_decision() {{ printf '%s\\n' '{decision}'; }}
stage_is_resolved() {{ case ':{':'.join(resolved)}:' in *:"$1":*) return 0;; *) return 1;; esac; }}
stage_status() {{ printf '%s\\n' validated; }}
{repeat_helper}
repaired_stage_can_finalize() {{ case ':{':'.join(finalizable)}:' in *:"$1":*) return 0;; *) return 1;; esac; }}
stage_completed_command_count() {{ case ':{':'.join(finalizable)}:' in *:"$1":*) printf '1\\n';; *) printf '0\\n';; esac; }}
run_repaired_stage() {{ printf 'run:%s\\n' "$*"; }}
record_budget_skip() {{ printf 'budget:%s\\n' "$1"; }}
record_stage2_no_go() {{ printf 'no-go:%s\\n' "$1"; }}
note() {{ :; }}
"""
    source_line = f'source "{COMMON}"'
    text = text.replace(source_line, source_line + injection, 1)
    script = tmp_path / filename
    script.write_text(text, encoding="utf-8")
    completed = subprocess.run(
        ["bash", str(script)],
        cwd=PROJECT_ROOT,
        env={
            "PATH": "/usr/bin:/bin",
            "PYTHON_BIN": "/bin/true",
            "DSBM_ROOT": str(tmp_path),
        },
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == expected_returncode, completed.stderr
    if expected_stderr is not None:
        assert expected_stderr in completed.stderr
    return [line for line in completed.stdout.splitlines() if line]


@pytest.mark.short
@pytest.mark.parametrize(
    ("hours", "decision", "expected"),
    (
        (40, "go", ("run:stage4_long_repeatability --repetitions 2 --stop-with-hours-left 32", "run:stage5_topology_controls")),
        (39, "go", ("run:stage4_long_repeatability --repetitions 1 --stop-with-hours-left 32", "run:stage5_topology_controls")),
        (37, "go", ("run:stage4_long_repeatability --repetitions 1 --stop-with-hours-left 32", "budget:stage5_topology_controls")),
        (32, "go", ("budget:stage5_topology_controls",)),
        (29, "go", ("run:stage4_long_repeatability --repetitions 1", "budget:stage5_topology_controls")),
        (11, "go", ("budget:stage4_long_repeatability", "budget:stage5_topology_controls")),
        (40, "no-go", ("run:stage4_long_repeatability --repetitions 2 --stop-with-hours-left 32", "no-go:stage5_topology_controls")),
    ),
)
def test_repeatability_and_control_policy_branches(
    tmp_path, hours, decision, expected
):
    actions = _policy_run(
        tmp_path,
        "06_repeatability_and_controls.sh",
        hours=hours,
        decision=decision,
    )
    assert tuple(actions) == expected


@pytest.mark.short
def test_completed_repeat_is_finalized_before_handoff(tmp_path):
    actions = _policy_run(
        tmp_path,
        "06_repeatability_and_controls.sh",
        hours=32,
        completed_repeats=1,
    )
    assert tuple(actions) == (
        "run:stage4_long_repeatability --repetitions 1",
        "budget:stage5_topology_controls",
    )


@pytest.mark.short
def test_dsbm_refuses_to_strand_completed_repeat(tmp_path):
    _policy_run(
        tmp_path,
        "07_dsbm.sh",
        hours=30,
        resolved=("stage5_topology_controls",),
        completed_repeats=1,
        expected_returncode=1,
        expected_stderr="re-run file 06 before DSBM",
    )


@pytest.mark.short
def test_dsbm_fails_closed_when_repeat_audit_cannot_be_read(tmp_path):
    actions = _policy_run(
        tmp_path,
        "07_dsbm.sh",
        hours=30,
        resolved=("stage5_topology_controls",),
        completed_repeat_error=True,
        expected_returncode=1,
    )
    assert not any(action.startswith("run:stage6_dsbm") for action in actions)


@pytest.mark.short
@pytest.mark.parametrize(
    ("filename", "hours", "decision", "resolved", "expected"),
    (
        (
            "07_dsbm.sh",
            30,
            "go",
            ("stage5_topology_controls",),
            ("run:stage6_dsbm --dsbm-root",),
        ),
        (
            "07_dsbm.sh",
            29.999,
            "go",
            ("stage5_topology_controls",),
            ("budget:stage6_dsbm",),
        ),
        (
            "08_interfaces_and_final_validation.sh",
            12,
            "go",
            ("stage5_topology_controls", "stage6_dsbm"),
            (
                "run:stage4_long_repeatability --repetitions 1",
                "run:stage7_dfleiden_interface",
                "run:stage7_s2cag_interface --repetitions 1",
            ),
        ),
        (
            "08_interfaces_and_final_validation.sh",
            4,
            "go",
            ("stage5_topology_controls", "stage6_dsbm"),
            (
                "budget:stage4_long_repeatability",
                "run:stage7_dfleiden_interface",
                "budget:stage7_s2cag_interface",
            ),
        ),
        (
            "08_interfaces_and_final_validation.sh",
            12,
            "no-go",
            ("stage5_topology_controls", "stage6_dsbm"),
            (
                "run:stage4_long_repeatability --repetitions 1",
                "no-go:stage7_dfleiden_interface",
                "no-go:stage7_s2cag_interface",
            ),
        ),
    ),
)
def test_dsbm_and_final_policy_branches(
    tmp_path, filename, hours, decision, resolved, expected
):
    actions = _policy_run(
        tmp_path,
        filename,
        hours=hours,
        decision=decision,
        resolved=resolved,
    )
    normalized = tuple(
        re.sub(r" --dsbm-root .+$", " --dsbm-root", line)
        for line in actions
        if line.startswith(("run:", "budget:", "no-go:"))
    )
    assert normalized == expected


@pytest.mark.short
def test_completed_dsbm_is_finalized_below_its_start_gate(tmp_path):
    actions = _policy_run(
        tmp_path,
        "07_dsbm.sh",
        hours=29,
        resolved=("stage5_topology_controls",),
        finalizable=("stage6_dsbm",),
    )
    normalized = tuple(
        re.sub(r" --dsbm-root .+$", " --dsbm-root", line)
        for line in actions
        if line.startswith(("run:", "budget:"))
    )
    assert normalized == ("run:stage6_dsbm --dsbm-root",)


@pytest.mark.short
def test_all_registered_threshold_boundaries_are_exact_and_finite():
    for threshold in (40, 38, 33, 32, 30, 12, 4):
        for value, expected in ((threshold, True), (threshold - 0.001, False)):
            completed = subprocess.run(
                [
                    "bash",
                    "-c",
                    'source "$1"; hours_at_least "$2" "$3"',
                    "bash",
                    str(COMMON),
                    str(value),
                    str(threshold),
                ],
                cwd=PROJECT_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            assert (completed.returncode == 0) is expected, completed.stderr
    for value in ("nan", "inf", "-inf"):
        completed = subprocess.run(
            [
                "bash",
                "-c",
                'source "$1"; hours_at_least "$2" 1',
                "bash",
                str(COMMON),
                value,
            ],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode != 0


@pytest.mark.short
def test_generated_campaigns_and_lock_are_ignored_but_readmes_are_tracked():
    ignored_paths = (
        "results/ieee-access-2026-1/raw/repaired-comnetx/example/manifest.json",
        "results/ieee-access-2026-1/raw/ldleiden/example/manifest.json",
        "results/ieee-access-2026-1/raw/.ieee-access-launch.lock",
    )
    for path in ignored_paths:
        assert subprocess.run(
            ["git", "check-ignore", "-q", path], cwd=PROJECT_ROOT, check=False
        ).returncode == 0

    for readme in (
        "results/ieee-access-2026-1/raw/repaired-comnetx/README.md",
        "results/ieee-access-2026-1/raw/ldleiden/README.md",
    ):
        assert subprocess.run(
            ["git", "check-ignore", "-q", readme], cwd=PROJECT_ROOT, check=False
        ).returncode == 1
        assert subprocess.run(
            ["git", "ls-files", "--error-unmatch", readme],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
        ).returncode == 0


@pytest.mark.short
def test_readme_lists_each_launcher_once_in_numeric_order():
    text = (LAUNCH_DIR / "README.md").read_text(encoding="utf-8")
    observed = tuple(
        Path(match).name
        for match in re.findall(
            r"^bash (scripts/paper/ieee_access_72h_launch/\d\d_[^ ]+\.sh)$",
            text,
            flags=re.MULTILINE,
        )
    )
    assert observed == EXPECTED_LAUNCHERS


def _write_registered(campaign: Path, filename: str, payload: dict) -> dict:
    path = campaign / filename
    path.write_text(json.dumps(payload), encoding="utf-8")
    return {
        "filename": filename,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _make_pair_manifests(repaired: Path, ld: Path) -> None:
    repaired.mkdir(parents=True)
    ld.mkdir(parents=True)
    hardware = {"cpu": "fixed", "gpu": "fixed"}
    record = {
        "path": "/data/stream",
        "size": 10,
        "mtime_ns": 20,
        "sha256": "a" * 64,
    }
    repaired_manifest = {
        "schema": "comnetx-ieee-access-repaired-campaign-v1",
        "protocol_id": "ieee-access-repaired-comnetx-72h-v1",
        "campaign_id": repaired.name,
        "status": "core_validated",
        "git": {"dirty": False, "commit": "b" * 40},
        "paths_config": {"sha256": "c" * 64},
        "hardware": _write_registered(repaired, "hardware.json", hardware),
        "real_input_manifest": _write_registered(
            repaired,
            "real_input_manifest.json",
            {"schema": "comnetx-ieee-access-real-inputs-v1", "files": [record]},
        ),
    }
    ld_manifest = {
        "schema": "comnetx-ieee-access-ldleiden-campaign-v1",
        "protocol_id": "ieee-access-ldleiden-72h-v1",
        "campaign_id": ld.name,
        "status": "completed",
        "git": {"dirty": False, "commit": "b" * 40},
        "paths_config": {"sha256": "c" * 64},
        "hardware": _write_registered(ld, "hardware.json", hardware),
        "real_input_manifest": _write_registered(
            ld,
            "real_input_manifest.json",
            {
                "schema": "comnetx-ieee-access-ldleiden-real-inputs-v1",
                "files": [record],
            },
        ),
    }
    (repaired / "manifest.json").write_text(
        json.dumps(repaired_manifest), encoding="utf-8"
    )
    (ld / "manifest.json").write_text(json.dumps(ld_manifest), encoding="utf-8")


@pytest.mark.short
def test_bootstrap_sync_and_cross_pack_validation(tmp_path):
    repaired = tmp_path / "repaired"
    ld = tmp_path / "ld"
    _make_pair_manifests(repaired, ld)
    repaired_cache = repaired / "bootstrap-cache"
    repaired_cache.mkdir()
    source = repaired_cache / (
        "dyn_pubmed_b:999_by_leidenalg_d:3_parent_quotient_v1.npz"
    )
    hierarchy = np.asarray([[0, 0, 2], [0, 0, 0], [0, 0, 0]], dtype=np.int64)
    np.savez_compressed(
        source,
        partition=hierarchy,
        mod=np.asarray(0.42),
        schema=np.asarray("parent_quotient_v1"),
    )

    report = synchronize(
        repaired,
        ld,
        datasets=["dyn_pubmed"],
        initial_batch=999,
    )
    assert report["entries"]["dyn_pubmed/999"]["modularity"] == 0.42
    with np.load(
        ld / "bootstrap-cache" / "dyn_pubmed_b:999_by_leidenalg.npz",
        allow_pickle=False,
    ) as payload:
        assert np.array_equal(payload["partition"], hierarchy[0])

    assert validate_pair(repaired, ld, preflight_only=True)["status"] == (
        "valid_preflight_pair"
    )

    target = ld / "bootstrap-cache" / "dyn_pubmed_b:999_by_leidenalg.npz"
    np.savez_compressed(target, partition=np.asarray([7, 7, 9]), mod=np.asarray(0.42))
    with pytest.raises(ValueError, match="not canonical"):
        synchronize(
            repaired,
            ld,
            datasets=["dyn_pubmed"],
            initial_batch=999,
        )


@pytest.mark.short
def test_missing_clock_cannot_reset_campaign_after_measurement_progress(tmp_path):
    repaired = tmp_path / "repaired"
    ld = tmp_path / "ld"
    repaired.mkdir()
    ld.mkdir()
    repaired_manifest = {
        "stage_status": {"stage1_correctness_smoke": "pending"},
    }
    ld_manifest = {"phase_status": {"smoke_999_10": "pending"}}
    (repaired / "manifest.json").write_text(
        json.dumps(repaired_manifest), encoding="utf-8"
    )
    (ld / "manifest.json").write_text(json.dumps(ld_manifest), encoding="utf-8")
    preflight_attempt = (
        repaired / "stages" / "preflight" / "lint" / "attempt-01"
    )
    preflight_attempt.mkdir(parents=True)
    (preflight_attempt / "metadata.json").write_text(
        json.dumps({"status": "completed"}), encoding="utf-8"
    )

    assert_clock_creation_safe(repaired, ld)

    measured_attempt = repaired / "stages" / "stage1" / "run" / "attempt-01"
    measured_attempt.mkdir(parents=True)
    (measured_attempt / "metadata.json").write_text(
        json.dumps({"status": "running"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="measurement attempt"):
        assert_clock_creation_safe(repaired, ld)

    (measured_attempt / "metadata.json").unlink()
    repaired_manifest["stage_status"]["stage1_correctness_smoke"] = "validated"
    (repaired / "manifest.json").write_text(
        json.dumps(repaired_manifest), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="campaign progress"):
        assert_clock_creation_safe(repaired, ld)


@pytest.mark.short
def test_bare_campaign_ids_always_resolve_below_the_result_namespace(tmp_path):
    root = tmp_path / "result-root"
    assert sync_bootstrap._campaign(Path("tmp"), root) == (root / "tmp").resolve()


@pytest.mark.short
def test_cross_pack_validator_rejects_input_identity_mismatch(tmp_path):
    repaired = tmp_path / "repaired"
    ld = tmp_path / "ld"
    _make_pair_manifests(repaired, ld)
    ld_manifest = json.loads((ld / "manifest.json").read_text(encoding="utf-8"))
    changed = {
        "schema": "comnetx-ieee-access-ldleiden-real-inputs-v1",
        "files": [
            {
                "path": "/data/stream",
                "size": 11,
                "mtime_ns": 20,
                "sha256": "a" * 64,
            }
        ]
    }
    ld_manifest["real_input_manifest"] = _write_registered(
        ld, "real_input_manifest.json", changed
    )
    (ld / "manifest.json").write_text(json.dumps(ld_manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="input identity differs"):
        validate_pair(repaired, ld, preflight_only=True)


@pytest.mark.short
def test_cross_pack_validator_accepts_matched_short_and_long_bootstraps(tmp_path):
    repaired = tmp_path / "repaired"
    ld = tmp_path / "ld"
    _make_pair_manifests(repaired, ld)
    budget = {
        "schema": "comnetx-ieee-access-launch-budget-v1",
        "git_sha": "b" * 40,
        "repaired_campaign_id": repaired.name,
        "ld_campaign_id": ld.name,
        "paths_config_sha256": "c" * 64,
        "budget_hours": 72.0,
        "started_at_epoch": 1000,
        "deadline_epoch": 1000 + 72 * 3600,
    }
    (repaired / "launch_budget.json").write_text(
        json.dumps(budget), encoding="utf-8"
    )
    partition = np.asarray([0, 0, 2], dtype=np.int64)
    for campaign in (repaired, ld):
        cache = campaign / "bootstrap-cache"
        cache.mkdir()
        for dataset in ("dyn_pubmed", "arxivmath"):
            for initial_batch in (999, 9):
                np.savez_compressed(
                    cache / f"{dataset}_b:{initial_batch}_by_leidenalg.npz",
                    partition=partition,
                    mod=np.asarray(0.42),
                )

    report = validate_pair(repaired, ld, preflight_only=False)
    assert report["status"] == "valid_pair"
    assert set(report["bootstrap"]) == {
        "dyn_pubmed/999",
        "dyn_pubmed/9",
        "arxivmath/999",
        "arxivmath/9",
    }
