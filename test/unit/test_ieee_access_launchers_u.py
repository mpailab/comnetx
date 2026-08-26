import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pytest

from scripts.paper.ieee_access_72h_launch import launch_budget, sync_bootstrap
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
    repaired_protocol = load_repaired_protocol()
    repaired_ids = {stage["id"] for stage in repaired_protocol["stages"]}
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
    assert [
        stage["id"]
        for stage in sorted(
            repaired_protocol["stages"], key=lambda item: item["priority"]
        )
    ] == [
        "stage1_correctness_smoke",
        "stage2_core_short",
        "stage3_mechanism",
        "stage4_long_core",
        "stage6_dsbm",
        "stage7_dfleiden_interface",
        "stage7_s2cag_interface",
        "stage4_long_repeatability",
        "stage5_topology_controls",
    ]

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
    assert 'df_stage="stage7_dfleiden_interface"' in texts[
        "06_repeatability_and_controls.sh"
    ]
    assert 's2cag_stage="stage7_s2cag_interface"' in texts[
        "06_repeatability_and_controls.sh"
    ]
    assert 'repeat_stage="stage4_long_repeatability"' in texts[
        "06_repeatability_and_controls.sh"
    ]
    assert 'stage="stage6_dsbm"' in texts["07_dsbm.sh"]
    assert 'control_stage="stage5_topology_controls"' in texts["07_dsbm.sh"]
    assert "--dsbm-root" in texts["07_dsbm.sh"]
    assert 'seed_order = (42, 43, 44, 45, 46)' in texts["07_dsbm.sh"]
    assert "dsbm_protected_reserve_hours" in texts[
        "06_repeatability_and_controls.sh"
    ]
    assert '--stop-with-hours-left "$dsbm_reserve"' in texts[
        "06_repeatability_and_controls.sh"
    ]
    assert "end-of-window reproducibility checkpoint" in texts[
        "08_interfaces_and_final_validation.sh"
    ]
    assert "run_repaired_stage" not in texts["08_interfaces_and_final_validation.sh"]
    assert "if [[ -f \"$BUDGET_STATE_FILE\" ]]" in texts["01_preflight.sh"]
    assert "extend_budget_clock" in texts["01_preflight.sh"]

    interface_text = texts["06_repeatability_and_controls.sh"]
    assert interface_text.index('df_stage="stage7_dfleiden_interface"') < (
        interface_text.index('s2cag_stage="stage7_s2cag_interface"')
    ) < interface_text.index('repeat_stage="stage4_long_repeatability"')
    dsbm_text = texts["07_dsbm.sh"]
    assert dsbm_text.index('stage="stage6_dsbm"') < dsbm_text.index(
        'control_stage="stage5_topology_controls"'
    )
    for filename in (
        "06_repeatability_and_controls.sh",
        "07_dsbm.sh",
        "08_interfaces_and_final_validation.sh",
    ):
        assert "record_budget_skip" not in texts[filename]

    common_text = COMMON.read_text(encoding="utf-8")
    assert "budget_deadline_epoch" in common_text
    assert "--deadline-epoch \"$deadline_epoch\"" in common_text
    assert "CAMPAIGN_BUDGET_HOURS=\"${CAMPAIGN_BUDGET_HOURS:-24}\"" in common_text
    assert "--measurement-window-id \"$measurement_window_id\"" in common_text

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
    completed_dsbm_seeds: int = 0,
    validated_dsbm_seeds: int | None = None,
    partial_dsbm_conditions: int = 0,
    finalizable: tuple[str, ...] = (),
    next_dsbm_seed: int | None = None,
) -> list[str]:
    text = (LAUNCH_DIR / filename).read_text(encoding="utf-8")
    source_pattern = re.compile(r'^source .*_common\.bash"$', re.MULTILINE)
    text, substitutions = source_pattern.subn(f'source "{COMMON}"', text, count=1)
    assert substitutions == 1
    validated_dsbm_seeds = (
        completed_dsbm_seeds
        if validated_dsbm_seeds is None
        else validated_dsbm_seeds
    )
    injection = f"""
mock_hours='{hours}'
mock_dsbm_count='{completed_dsbm_seeds}'
mock_dsbm_partial='{partial_dsbm_conditions}'
prepare_launcher() {{ :; }}
require_preflight_campaigns() {{ :; }}
require_repaired_stage_validated() {{ :; }}
require_ld_phase_completed() {{ :; }}
require_optional_stage_can_be_deferred() {{ :; }}
hours_left() {{ printf '%s\\n' "$mock_hours"; }}
hours_at_least() {{ awk -v value="$1" -v threshold="$2" 'BEGIN {{ exit !(value >= threshold) }}'; }}
stage2_breadth_decision() {{ printf '%s\\n' '{decision}'; }}
stage_is_resolved() {{ case ':{':'.join(resolved)}:' in *:"$1":*) return 0;; *) return 1;; esac; }}
stage_status() {{ printf '%s\\n' validated; }}
repeatability_completed_count() {{ printf '%s\\n' '{completed_repeats}'; }}
dsbm_completed_seed_count() {{ printf '%s\\n' "$mock_dsbm_count"; }}
dsbm_validated_seed_count() {{ printf '%s\\n' '{validated_dsbm_seeds}'; }}
dsbm_current_seed_completed_condition_count() {{ printf '%s\\n' "$mock_dsbm_partial"; }}
dsbm_protected_reserve_hours() {{
  awk -v available="$1" -v completed="$mock_dsbm_count" -v partial="$mock_dsbm_partial" '
    BEGIN {{
      missing = 3 - completed
      reserve = 0
      for (i = 0; i < missing; i++) {{
        cost = (i == 0 && partial > 0) ? 4 : 17
        if (reserve + cost > available) break
        reserve += cost
      }}
      print reserve
    }}'
}}
unfinalized_dsbm_seed() {{ :; }}
repaired_stage_can_finalize() {{ case ':{':'.join(finalizable)}:' in *:"$1":*) return 0;; *) return 1;; esac; }}
run_repaired_stage() {{
  printf 'run:%s\\n' "$*"
  if [[ "$1" == stage6_dsbm ]]; then
    if (( mock_dsbm_partial > 0 )); then
      mock_dsbm_cost=4
    else
      mock_dsbm_cost=17
    fi
    mock_dsbm_count=$((mock_dsbm_count + 1))
    mock_dsbm_partial=0
    mock_hours="$(awk -v value="$mock_hours" -v cost="$mock_dsbm_cost" 'BEGIN {{ print value - cost }}')"
  fi
}}
record_budget_skip() {{ printf 'budget:%s\\n' "$1"; }}
record_stage2_no_go() {{ printf 'no-go:%s\\n' "$1"; }}
note() {{ :; }}
"""
    source_line = f'source "{COMMON}"'
    text = text.replace(source_line, source_line + injection, 1)
    if next_dsbm_seed is not None:
        text = text.replace(
            'stage="stage6_dsbm"',
            f"next_dsbm_seed() {{ printf '%s\\n' \"$(( {next_dsbm_seed} + mock_dsbm_count ))\"; }}\n"
            'stage="stage6_dsbm"',
            1,
        )
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
    assert completed.returncode == 0, completed.stderr
    return [line for line in completed.stdout.splitlines() if line]


@pytest.mark.short
def test_file06_runs_interfaces_before_protected_long_repeats(tmp_path):
    actions = _policy_run(
        tmp_path,
        "06_repeatability_and_controls.sh",
        hours=19,
    )
    assert tuple(actions) == (
        "run:stage7_dfleiden_interface --stop-with-hours-left 17",
        "run:stage7_s2cag_interface --repetitions 1 --stop-with-hours-left 17",
        "run:stage4_long_repeatability --repetitions 2 --stop-with-hours-left 17",
    )


@pytest.mark.short
def test_file06_protects_two_primary_dsbm_seed_opportunities(tmp_path):
    actions = _policy_run(
        tmp_path,
        "06_repeatability_and_controls.sh",
        hours=35,
    )
    assert actions == [
        "run:stage7_dfleiden_interface --stop-with-hours-left 34"
    ]


@pytest.mark.short
def test_file06_waits_for_completed_dsbm_seed_validation(tmp_path):
    actions = _policy_run(
        tmp_path,
        "06_repeatability_and_controls.sh",
        hours=20,
        completed_dsbm_seeds=1,
        validated_dsbm_seeds=0,
    )
    assert actions == []


@pytest.mark.short
def test_temporary_window_shortage_leaves_optional_measurements_pending(tmp_path):
    file06_actions = _policy_run(
        tmp_path,
        "06_repeatability_and_controls.sh",
        hours=0,
    )
    file07_actions = _policy_run(
        tmp_path,
        "07_dsbm.sh",
        hours=0,
        next_dsbm_seed=42,
    )
    assert file06_actions == []
    assert file07_actions == []


@pytest.mark.short
def test_completed_repeat_is_finalized_below_new_window_start_gate(tmp_path):
    actions = _policy_run(
        tmp_path,
        "06_repeatability_and_controls.sh",
        hours=0,
        resolved=("stage7_dfleiden_interface", "stage7_s2cag_interface"),
        completed_repeats=1,
    )
    assert actions == ["run:stage4_long_repeatability --repetitions 1"]


@pytest.mark.short
def test_file07_prioritizes_one_complete_paired_dsbm_seed_before_stage5(tmp_path):
    actions = _policy_run(
        tmp_path,
        "07_dsbm.sh",
        hours=17,
        next_dsbm_seed=42,
    )
    normalized = tuple(
        re.sub(r" --dsbm-root .+ --dsbm-seed", " --dsbm-root --dsbm-seed", line)
        for line in actions
    )
    assert normalized == (
        "run:stage6_dsbm --dsbm-root --dsbm-seed 42",
    )


@pytest.mark.short
def test_file07_resumes_partial_paired_seed_with_four_hours(tmp_path):
    actions = _policy_run(
        tmp_path,
        "07_dsbm.sh",
        hours=4,
        partial_dsbm_conditions=5,
        next_dsbm_seed=42,
    )
    normalized = tuple(
        re.sub(r" --dsbm-root .+ --dsbm-seed", " --dsbm-root --dsbm-seed", line)
        for line in actions
    )
    assert normalized == (
        "run:stage6_dsbm --dsbm-root --dsbm-seed 42",
    )


@pytest.mark.short
def test_stage2_no_go_skips_interfaces_but_not_repeatability(tmp_path):
    actions = _policy_run(
        tmp_path,
        "06_repeatability_and_controls.sh",
        hours=19,
        decision="no-go",
    )
    assert actions == [
        "no-go:stage7_dfleiden_interface",
        "no-go:stage7_s2cag_interface",
        "run:stage4_long_repeatability --repetitions 2 --stop-with-hours-left 17",
    ]


@pytest.mark.short
def test_file08_is_a_checkpoint_and_does_not_launch_more_measurements(tmp_path):
    actions = _policy_run(
        tmp_path,
        "08_interfaces_and_final_validation.sh",
        hours=0,
    )
    assert actions == []


@pytest.mark.short
def test_all_registered_threshold_boundaries_are_exact_and_finite():
    for threshold in (19, 17, 4, 2, 1):
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


def _budget_identity(paths_config: Path, repaired: str, ld: str) -> dict:
    return {
        "schema": launch_budget.SCHEMA,
        "git_sha": "b" * 40,
        "repaired_campaign_id": repaired,
        "ld_campaign_id": ld,
        "paths_config_sha256": hashlib.sha256(paths_config.read_bytes()).hexdigest(),
        "max_total_hours": launch_budget.MAX_TOTAL_HOURS,
        "initial_window_hours": launch_budget.INITIAL_WINDOW_HOURS,
    }


@pytest.mark.short
def test_budget_initialization_is_exactly_one_immutable_24_hour_window(tmp_path):
    paths_config = tmp_path / "paths.json"
    paths_config.write_text("{}\n", encoding="utf-8")
    expected = _budget_identity(paths_config, "repaired", "ld")
    state = tmp_path / "launch_budget.json"

    payload = launch_budget.initialize(state, expected, 24, now_epoch=1000)
    assert payload["windows"] == [
        {
            "id": "window-001",
            "granted_hours": 24.0,
            "started_at_epoch": 1000,
            "deadline_epoch": 1000 + 24 * 3600,
        }
    ]
    assert launch_budget.initialize(
        state, expected, 24, now_epoch=999_999
    ) == payload

    with pytest.raises(launch_budget.BudgetError, match="fixed at 24 hours"):
        launch_budget.initialize(
            tmp_path / "wrong_budget.json", expected, 23, now_epoch=1000
        )


@pytest.mark.short
def test_budget_extensions_require_expiry_and_no_running_attempts_and_obey_cap(
    tmp_path,
):
    paths_config = tmp_path / "paths.json"
    paths_config.write_text("{}\n", encoding="utf-8")
    expected = _budget_identity(paths_config, "repaired", "ld")
    state = tmp_path / "launch_budget.json"
    repaired = tmp_path / "repaired"
    ld = tmp_path / "ld"
    repaired.mkdir()
    ld.mkdir()
    payload = launch_budget.initialize(state, expected, 24, now_epoch=1000)
    first_deadline = payload["windows"][0]["deadline_epoch"]

    with pytest.raises(launch_budget.BudgetError, match="current window expires"):
        launch_budget.extend(
            state, expected, 1, (repaired, ld), now_epoch=first_deadline - 1
        )

    metadata = repaired / "stages" / "stage2" / "attempt-01" / "metadata.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(json.dumps({"status": "running"}), encoding="utf-8")
    with pytest.raises(launch_budget.BudgetError, match="status=running"):
        launch_budget.extend(
            state, expected, 1, (repaired, ld), now_epoch=first_deadline
        )

    metadata.write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    payload = launch_budget.extend(
        state, expected, 24, (repaired, ld), now_epoch=first_deadline + 10
    )
    second = payload["windows"][-1]
    assert second["id"] == "window-002"
    assert second["started_at_epoch"] == first_deadline + 10
    payload = launch_budget.extend(
        state,
        expected,
        24,
        (repaired, ld),
        now_epoch=second["deadline_epoch"],
    )
    assert [window["id"] for window in payload["windows"]] == [
        "window-001",
        "window-002",
        "window-003",
    ]
    assert sum(window["granted_hours"] for window in payload["windows"]) == 72
    with pytest.raises(launch_budget.BudgetError, match="72-hour cap"):
        launch_budget.extend(
            state,
            expected,
            1,
            (repaired, ld),
            now_epoch=payload["windows"][-1]["deadline_epoch"],
        )


@pytest.mark.short
def test_budget_cli_queries_report_the_current_appended_window(tmp_path):
    paths_config = tmp_path / "paths.json"
    paths_config.write_text("{}\n", encoding="utf-8")
    expected = _budget_identity(paths_config, "repaired", "ld")
    state = tmp_path / "launch_budget.json"
    repaired = tmp_path / "repaired"
    ld = tmp_path / "ld"
    repaired.mkdir()
    ld.mkdir()
    payload = launch_budget.initialize(state, expected, 24, now_epoch=1000)
    payload = launch_budget.extend(
        state,
        expected,
        6,
        (repaired, ld),
        now_epoch=payload["windows"][-1]["deadline_epoch"],
    )
    current = payload["windows"][-1]
    common_args = [
        "--state-file",
        str(state),
        "--git-sha",
        "b" * 40,
        "--repaired-campaign-id",
        "repaired",
        "--ld-campaign-id",
        "ld",
        "--paths-config",
        str(paths_config),
    ]
    expectations = {
        "window-id": "window-002",
        "granted-hours": "6",
        "deadline": str(current["deadline_epoch"]),
        "hours-left": "0.000000",
    }
    for command, expected_output in expectations.items():
        completed = subprocess.run(
            [
                sys.executable,
                str(LAUNCH_DIR / "launch_budget.py"),
                command,
                *common_args,
            ],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        assert completed.stdout.strip() == expected_output


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
    launch_order = text.split("## Launch order", 1)[1].split(
        "## First-day scientific order", 1
    )[0]
    observed = tuple(
        Path(match).name
        for match in re.findall(
            r"^bash (scripts/paper/ieee_access_72h_launch/\d\d_[^ ]+\.sh)$",
            launch_order,
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
        "stage_status": {},
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
@pytest.mark.parametrize(
    ("failure_kind", "audit_key"),
    (
        ("campaign_time_boundary", "time_boundary_stops"),
        ("launcher_interrupted", "interrupted_phases"),
    ),
)
def test_extension_precheck_accepts_only_audited_resumable_ld_failure(
    tmp_path,
    failure_kind,
    audit_key,
):
    repaired = tmp_path / "repaired"
    ld = tmp_path / "ld"
    _make_pair_manifests(repaired, ld)
    manifest_path = ld / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    phase_id = "measured_9_500"
    manifest.update(
        {
            "status": "failed",
            "phase_status": {
                "smoke_999_10": "completed",
                "measured_999_10": "completed",
                phase_id: "failed",
            },
            "phase_failures": {
                phase_id: {
                    "failure_kind": failure_kind,
                    "recorded_at_utc": "2026-08-26T00:00:00Z",
                }
            },
        }
    )
    if audit_key == "time_boundary_stops":
        manifest[audit_key] = {
            phase_id: {
                "measurement_window_id": "window-001",
                "deadline_epoch": 123456,
                "reason": "registered deadline",
            }
        }
    else:
        manifest[audit_key] = {phase_id: {"reason": "operator interruption"}}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    attempt = ld / phase_id / "repeat-01" / "attempt-01"
    attempt.mkdir(parents=True)
    (attempt / "metadata.json").write_text(
        json.dumps({"status": "failed", "failure_kind": failure_kind}),
        encoding="utf-8",
    )

    assert validate_pair(repaired, ld, preflight_only=True)["status"] == (
        "valid_preflight_pair"
    )


@pytest.mark.short
def test_extension_precheck_rejects_unaudited_ld_failure(tmp_path):
    repaired = tmp_path / "repaired"
    ld = tmp_path / "ld"
    _make_pair_manifests(repaired, ld)
    manifest_path = ld / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    phase_id = "measured_9_500"
    manifest.update(
        {
            "status": "failed",
            "phase_status": {phase_id: "failed"},
            "phase_failures": {
                phase_id: {
                    "failure_kind": "runtime_failure",
                    "recorded_at_utc": "2026-08-26T00:00:00Z",
                }
            },
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="outside a resumable boundary"):
        validate_pair(repaired, ld, preflight_only=True)

    manifest["phase_failures"][phase_id][
        "failure_kind"
    ] = "campaign_time_boundary"
    manifest["time_boundary_stops"] = {
        phase_id: {
            "measurement_window_id": "window-001",
            "deadline_epoch": 123456,
            "reason": "old registered deadline",
        }
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    attempt = ld / phase_id / "repeat-01" / "attempt-01"
    attempt.mkdir(parents=True)
    (attempt / "metadata.json").write_text(
        json.dumps({"status": "failed", "failure_kind": "runtime_failure"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unaudited attempt failure"):
        validate_pair(repaired, ld, preflight_only=True)


@pytest.mark.short
@pytest.mark.parametrize(
    ("failure_kind", "audit_key"),
    (
        ("campaign_time_boundary", "time_boundary_stops"),
        ("launcher_interrupted", "interrupted_stages"),
    ),
)
def test_extension_precheck_accepts_only_audited_repaired_failure(
    tmp_path,
    failure_kind,
    audit_key,
):
    repaired = tmp_path / "repaired"
    ld = tmp_path / "ld"
    _make_pair_manifests(repaired, ld)
    manifest_path = repaired / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stage_id = "stage6_dsbm"
    manifest["stage_status"] = {stage_id: "failed"}
    if audit_key == "time_boundary_stops":
        manifest[audit_key] = {
            stage_id: {
                "measurement_window_id": "window-001",
                "window_deadline_epoch": 123456,
                "stop_boundary_hours_left": 0,
                "reason": "registered deadline",
            }
        }
    else:
        manifest[audit_key] = {stage_id: {"reason": "operator interruption"}}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    attempt = repaired / "stages" / stage_id / "command" / "attempt-01"
    attempt.mkdir(parents=True)
    metadata_path = attempt / "metadata.json"
    metadata_path.write_text(
        json.dumps({"status": "failed", "failure_kind": failure_kind}),
        encoding="utf-8",
    )

    assert validate_pair(repaired, ld, preflight_only=True)["status"] == (
        "valid_preflight_pair"
    )

    metadata_path.write_text(
        json.dumps({"status": "failed"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="unaudited attempt failure"):
        validate_pair(repaired, ld, preflight_only=True)

    metadata_path.write_text(
        json.dumps({"status": "failed", "failure_kind": failure_kind}),
        encoding="utf-8",
    )
    manifest.pop(audit_key)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="exact resumable audit"):
        validate_pair(repaired, ld, preflight_only=True)


@pytest.mark.short
def test_cross_pack_validator_accepts_matched_short_and_long_bootstraps(tmp_path):
    repaired = tmp_path / "repaired"
    ld = tmp_path / "ld"
    _make_pair_manifests(repaired, ld)
    budget = {
        "schema": "comnetx-ieee-access-launch-budget-v2",
        "git_sha": "b" * 40,
        "repaired_campaign_id": repaired.name,
        "ld_campaign_id": ld.name,
        "paths_config_sha256": "c" * 64,
        "initial_window_hours": 24.0,
        "max_total_hours": 72.0,
        "windows": [
            {
                "id": "window-001",
                "granted_hours": 24.0,
                "started_at_epoch": 1000,
                "deadline_epoch": 1000 + 24 * 3600,
            }
        ],
    }
    (repaired / "launch_budget.json").write_text(
        json.dumps(budget), encoding="utf-8"
    )
    partition = np.asarray([0, 0, 2], dtype=np.int64)
    for campaign in (repaired, ld):
        attempt = campaign / "stages" / "measured" / "attempt-01"
        attempt.mkdir(parents=True)
        (attempt / "metadata.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "measurement_window_id": "window-001",
                    "window_deadline_epoch": 1000 + 24 * 3600,
                    "started_at_utc": "1970-01-01T00:16:40Z",
                    "finished_at_utc": "1970-01-01T00:16:41Z",
                }
            ),
            encoding="utf-8",
        )
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
    assert report["measurement_window_attempts"] == {
        "repaired": 1,
        "ldleiden": 1,
    }

    metadata_path = (
        repaired / "stages" / "measured" / "attempt-01" / "metadata.json"
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["finished_at_utc"] = "1970-01-02T00:16:43Z"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="finished outside its registered window"):
        validate_pair(repaired, ld, preflight_only=False)
