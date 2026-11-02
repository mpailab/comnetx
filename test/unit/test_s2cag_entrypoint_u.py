"""Exercise the server shell entrypoint without packages, data or measurements."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'scripts/paper/run_s2cag.sh'


@pytest.fixture
def checkout(tmp_path):
    script = tmp_path / 'scripts/paper/run_s2cag.sh'
    script.parent.mkdir(parents=True)
    shutil.copy2(SCRIPT, script)
    config = tmp_path / 'conf/ieee_access/s2cag_pairs/confirmatory.json'
    config.parent.mkdir(parents=True)
    shutil.copy2(ROOT / config.relative_to(tmp_path), config)
    paths = tmp_path / 'datasets-info/paths/cn69.json'
    paths.parent.mkdir(parents=True)
    paths.write_text('{}')
    req = tmp_path / 'requirements/ieee-access-s2cag.txt'
    req.parent.mkdir()
    req.write_text('# No installation is performed in this entrypoint test.\n')
    venv = tmp_path / '.venv-s2cag'
    (venv / 'bin').mkdir(parents=True)
    (venv / 'bin/python').symlink_to(sys.executable)
    (venv / '.s2cag-requirements.sha256').write_text(hashlib.sha256(req.read_bytes()).hexdigest()+'\n')
    (script.parent / 'run_s2cag_pairs.py').write_text('''import json, os, pathlib, sys
pathlib.Path('invocation.json').write_text(json.dumps(sys.argv[1:]))
raise SystemExit(int(os.environ.get('S2_TEST_EXIT', '0')))
''')
    return tmp_path, script


def invoke(checkout, *args, env=None):
    root, script = checkout
    return subprocess.run(['bash', str(script), *args], cwd=root.parent,
                          text=True, capture_output=True, env=env)


def test_one_command_runs_confirmatory_directly_with_cuda(checkout):
    result = invoke(checkout)
    assert result.returncode == 0, result.stderr
    root, _ = checkout
    args = json.loads((root/'invocation.json').read_text())
    assert args[0] == 'run'
    assert args[1].endswith('/confirmatory.json')
    assert args[-2:] == ['--device', 'cuda']
    assert 'preflight' not in args and 'diagnostic' not in ' '.join(args)
    assert 'Finished.' in result.stdout


def test_paths_with_spaces_and_explicit_cpu_are_forwarded(checkout):
    root, _ = checkout
    paths = root/'server paths.json'; paths.write_text('{}')
    output = root/'new measurements'
    result = invoke(checkout, '--paths-config', str(paths), '--output-dir', str(output), '--device', 'cpu')
    assert result.returncode == 0, result.stderr
    args = json.loads((root/'invocation.json').read_text())
    assert args[args.index('--paths-config')+1] == str(paths)
    assert args[args.index('--output-dir')+1] == str(output)
    assert args[-1] == 'cpu'


def test_existing_output_is_rejected_before_any_run(checkout):
    root, _ = checkout
    (root/'existing').mkdir()
    result = invoke(checkout, '--output-dir', 'existing')
    assert result.returncode != 0
    assert not (root/'invocation.json').exists()


def test_failed_measurement_propagates_exit_without_success_message(checkout):
    env = {**os.environ, 'S2_TEST_EXIT': '7'}
    result = invoke(checkout, env=env)
    assert result.returncode == 7
    assert 'Finished.' not in result.stdout


def test_help_requires_no_environment_or_data(tmp_path):
    result = subprocess.run(['bash', str(SCRIPT), '--help'], cwd=tmp_path,
                            text=True, capture_output=True)
    assert result.returncode == 0
    assert '20 paired' not in result.stderr
    assert '--paths-config' in result.stdout


@pytest.mark.parametrize('args', [['--device', 'auto'], ['--unknown'], ['--paths-config']])
def test_invalid_arguments_fail_before_execution(checkout, args):
    root, _ = checkout
    result = invoke(checkout, *args)
    assert result.returncode != 0
    assert not (root/'invocation.json').exists()
