"""Checks for reproducible pair selection and admission of new measurements.

All fixture scores/timings are artificial validation values. Tiny NPZ files
exercise output integrity; no S2CAG or graph benchmark is executed here.
"""
from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
import zipfile

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    's2cag_pair_runner_under_test', ROOT / 'scripts/paper/run_s2cag_pairs.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
CONFIGS = ROOT / 'conf/ieee_access/s2cag_pairs'


def config(name='confirmatory'):
    return runner.read_json(CONFIGS / (name + '.json'))


@pytest.mark.parametrize(('name', 'pairs', 'profile'), [
    ('confirmatory', 20, False),
])
def test_presets_plan_complete_paired_executions(name, pairs, profile):
    c = config(name)
    assert c['measurement'] == {'comnetx_profile': profile}
    plan = runner.make_plan(c)
    assert plan['pairs'] == pairs
    assert plan['measured_processes'] == 2 * pairs
    assert len({j['pair_id'] for j in plan['jobs']}) == pairs
    assert all(set(j['order']) == {'full', 'local'} for j in plan['jobs'])


def test_fixed_seed_repetitions_alternate_order_and_keep_the_same_cells():
    plan = runner.make_plan(config('confirmatory'))
    jobs = plan['jobs']
    first = [j for j in jobs if j['cell_id'] == jobs[0]['cell_id']]
    assert [j['repetition'] for j in first] == [1, 2, 3, 4, 5]
    assert [j['order'][0] for j in first] == ['full', 'local', 'full', 'local', 'full']
    assert sum(j['order'][0] == 'full' for j in jobs) == 10
    assert {(j['dataset'], j['protocol']) for j in jobs} == {
        ('dyn_pubmed', '999:5'), ('dyn_pubmed', '999:10'),
        ('arxivmath', '999:5'), ('arxivmath', '999:10')}


def test_growth_notation_maps_to_working_legacy_loader_key():
    c = config()
    c.update(protocols=['0:10'], feature_modes=['random', 'dataset'])
    jobs = runner.make_plan(c)['jobs']
    assert all(j['protocol'] == '0:10' and j['loader_key'] == '10' for j in jobs)
    assert {(j['dataset'], j['feature_mode']) for j in jobs} == {
        ('dyn_pubmed', 'random'), ('dyn_pubmed', 'dataset'),
        ('arxivmath', 'random'), ('arxivmath', 'dataset')}


@pytest.mark.parametrize('protocol', ['10', '0:5', '999:0', '100:10', '999:ten'])
def test_ambiguous_or_unsupported_protocol_is_rejected(protocol):
    c = config()
    c['protocols'] = [protocol]
    with pytest.raises(ValueError):
        runner.make_plan(c)


@pytest.mark.parametrize(('path', 'value'), [
    (('schema_version',), 2),
    (('datasets',), []),
    (('datasets',), ['unknown_graph']),
    (('protocols',), ['999:10', '999:10']),
    (('feature_modes',), ['onehot']),
    (('repetitions',), 0),
    (('repetitions',), True),
    (('backend', 'n_runs'), 1),
    (('backend', 'method'), 'mod'),
    (('initializer', 'seed'), 2),
    (('random_features', 'dimension'), 8),
    (('random_features', 'seed'), 42.5),
    (('runtime', 'cpu_threads'), 0),
    (('runtime', 'device'), 'auto'),
    (('measurement', 'comnetx_profile'), 1),
    (('measurement', 'comnetx_profile'), 'true'),
    (('measurement', 'comnetx_profile'), None),
])
def test_config_rejects_silent_changes_to_campaign_definition(path, value):
    c = config()
    target = c
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        runner.validate_config(c)


def test_unknown_config_key_is_not_silently_ignored():
    c = config()
    c['RANDOM_FEATURE_SEED'] = 99
    with pytest.raises(ValueError):
        runner.validate_config(c)


def test_profiling_policy_must_be_explicit():
    c = config()
    del c['measurement']
    with pytest.raises(ValueError):
        runner.validate_config(c)


def test_unknown_measurement_key_is_rejected():
    c = config()
    c['measurement']['include_sync_overhead'] = False
    with pytest.raises(ValueError):
        runner.validate_config(c)


def test_thread_environment_explicitly_pins_all_worker_pools():
    c = config()
    env = runner.child_env(c)
    assert {env[name] for name in runner.THREAD_VARS} == {'8'}
    assert env['PYTHONHASHSEED'] == '0'
    assert env['PYTHONUNBUFFERED'] == '1'


@pytest.fixture
def paired_case(tmp_path):
    c = config()
    c.update(datasets=['dyn_cora'], protocols=['999:2'], repetitions=1)
    c['runtime'].update(device='cpu', cpu_threads=1)
    plan = runner.make_plan(c)
    job = plan['jobs'][0]
    source = b'# artificial validation source\n'
    source_hashes = {'validation_source.py': hashlib.sha256(source).hexdigest()}
    with zipfile.ZipFile(tmp_path / 'execution-sources.zip', 'w') as archive:
        archive.writestr('validation_source.py', source)
    fingerprint = runner.json_hash(source_hashes)
    init = tmp_path / 'initial/shared.npz'
    init.parent.mkdir()
    np.savez_compressed(init, hierarchy=np.array([[0, 0, 2, 2]] * 3))
    inputs = {'features': 'features-sha', 'reference_labels': 'reference-sha',
              'batches': ['bootstrap-sha', 'batch-1-sha', 'batch-2-sha'],
              'feature_shape': [4, 64], 'nodes': 4, 'directed': False}
    initial = {'file_sha256': runner.digest(init), 'hierarchy_sha256': 'hierarchy-sha',
               'fine_partition_sha256': 'initial-fine-sha'}
    common = dict(schema=runner.SCHEMA, status='complete', job=job, config=c,
                  source_fingerprint=fingerprint, input=inputs, initial=initial,
                  expected_updates=2, actual_tensor_device='cpu',
                  environment={'packages': {'numpy': 'test-version'},
                               'thread_environment': {'OMP_NUM_THREADS': '1'},
                               'torch_threads': 1, 'torch_interop_threads': 1,
                               'cuda_version': None})
    records = {}
    for mode in ('full', 'local'):
        result = deepcopy(common)
        result.update(mode=mode, rows=[])
        for number in (1, 2):
            labels = np.array([0, 0, 2, 2] if mode == 'full' else [0, 1, 0, 1])
            label_path = (tmp_path / 'pairs' / job['pair_id'] /
                          (mode + '-labels') / f'update-{number:04}.npz')
            label_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(label_path, labels=labels)
            wall = float(number) if mode == 'full' else 0.5
            result['rows'].append(dict(
                update=number, source_batch_index=number,
                batch_sha256=f'batch-{number}-sha', snapshot_sha256=f'snapshot-{number}-sha',
                processing_wall_seconds=wall,
                processing_excluding_preprocessing_seconds=wall - 0.1,
                modularity=0.6 if mode == 'full' else 0.64,
                nmi=0.2 if mode == 'full' else 0.23, nonempty_groups=2,
                label_file=str(label_path.relative_to(tmp_path)),
                label_file_sha256=runner.digest(label_path)))
        records[mode] = result
        runner.write_json(tmp_path / 'pairs' / job['pair_id'] / (mode + '.json'), result)
    prepared = dict(schema=runner.SCHEMA, input=inputs, initial=initial,
                    initial_file=str(init.relative_to(tmp_path)), expected_updates=2)
    runner.write_json(tmp_path / 'prepared' / (job['cell_id'] + '.json'), prepared)
    manifest = dict(schema=runner.SCHEMA, status='complete', config=c, plan=plan,
                    source_files=source_hashes, source_fingerprint=fingerprint,
                    source_archive_sha256=runner.digest(tmp_path / 'execution-sources.zip'))
    runner.write_json(tmp_path / 'campaign.json', manifest)
    runner.write_json(tmp_path / 'config.json', c)
    return tmp_path, records['full'], records['local'], manifest


def test_valid_pair_uses_whole_stream_time_and_final_quality(paired_case):
    root, full, local, _ = paired_case
    result = runner.validate_pair(full, local, root)
    assert result['updates'] == 2
    assert result['delta_q'] == pytest.approx(0.04)
    assert result['delta_nmi'] == pytest.approx(0.03)
    assert result['speedup']['processing_wall_seconds'] == pytest.approx(3.0)
    assert result['speedup']['processing_excluding_preprocessing_seconds'] == pytest.approx(3.5)


@pytest.mark.parametrize('part', ['input', 'source_fingerprint', 'initial', 'actual_tensor_device'])
def test_different_inputs_or_execution_identity_cannot_form_a_pair(paired_case, part):
    root, full, local, _ = paired_case
    local[part] = {'different': True} if isinstance(local[part], dict) else 'different'
    with pytest.raises(ValueError, match='Pair mismatch'):
        runner.validate_pair(full, local, root)


def test_different_weighted_snapshot_is_rejected(paired_case):
    root, full, local, _ = paired_case
    local['rows'][0]['snapshot_sha256'] = 'other-snapshot'
    with pytest.raises(ValueError, match='Paired update mismatch'):
        runner.validate_pair(full, local, root)


def test_different_thread_or_library_environment_is_rejected(paired_case):
    root, full, local, _ = paired_case
    local['environment']['torch_threads'] = 8
    with pytest.raises(ValueError, match='environment mismatch'):
        runner.validate_pair(full, local, root)


@pytest.mark.parametrize('incomplete', ['status', 'missing_row', 'expected_updates'])
def test_partial_runs_are_never_summarized_as_complete_pairs(paired_case, incomplete):
    root, full, local, _ = paired_case
    if incomplete == 'status':
        local['status'] = 'failed'
    elif incomplete == 'missing_row':
        local['rows'].pop()
    else:
        local['expected_updates'] += 1
    with pytest.raises(ValueError, match='[Ii]ncomplete'):
        runner.validate_pair(full, local, root)


def test_matching_truncated_runs_cannot_override_protocol_update_count(paired_case):
    root, full, local, _ = paired_case
    for result in (full, local):
        result['job']['protocol'] = '999:5'
        result['config']['protocols'] = ['999:5']
    # Both files consistently claim two completed updates, but q requires five.
    with pytest.raises(ValueError, match='incomplete update counts'):
        runner.validate_pair(full, local, root)


@pytest.mark.parametrize('key,value', [
    ('source_batch_index', 0),
    ('update', 2),
    ('batch_sha256', 'other-batch-sha'),
])
def test_matching_update_metadata_still_has_to_match_input_stream(paired_case, key, value):
    root, full, local, _ = paired_case
    for result in (full, local):
        result['rows'][0][key] = value
    with pytest.raises(ValueError, match='order/input batch mismatch'):
        runner.validate_pair(full, local, root)


def test_changed_output_label_file_is_rejected(paired_case):
    root, full, local, _ = paired_case
    path = root / local['rows'][0]['label_file']
    np.savez_compressed(path, labels=np.array([0, 0, 0, 0]))
    with pytest.raises(ValueError, match='label file changed'):
        runner.validate_pair(full, local, root)


@pytest.mark.parametrize('value', [0.0, float('nan'), float('inf')])
def test_invalid_times_cannot_produce_speedup(paired_case, value):
    root, full, local, _ = paired_case
    local['rows'][0]['processing_wall_seconds'] = value
    with pytest.raises(ValueError, match='Invalid processing'):
        runner.validate_pair(full, local, root)


def test_summary_verifies_saved_campaign_and_writes_only_validated_results(paired_case):
    root, full, _, _ = paired_case
    result = runner.summarize(root)
    cell = result['cells'][full['job']['cell_id']]
    assert result['status'] == 'validated'
    assert cell['repetitions'] == 1
    assert cell['mean_delta_q'] == pytest.approx(0.04)
    assert runner.read_json(root / 'paired-summary.json') == result


def test_two_matching_but_wrong_results_do_not_override_campaign_plan(paired_case):
    root, full, local, _ = paired_case
    for result in (full, local):
        result['job']['repetition'] = 99
        runner.write_json(root / 'pairs' / result['job']['pair_id'] /
                          (result['mode'] + '.json'), result)
    with pytest.raises(ValueError, match='saved campaign/preparation: job'):
        runner.summarize(root)
    assert not (root / 'paired-summary.json').exists()


def test_changed_shared_initializer_is_rejected_even_when_pair_json_agrees(paired_case):
    root, _, _, _ = paired_case
    np.savez_compressed(root / 'initial/shared.npz', hierarchy=np.array([[0, 1, 2, 3]] * 3))
    with pytest.raises(ValueError, match='shared initialization changed'):
        runner.summarize(root)


def test_changed_execution_archive_is_rejected(paired_case):
    root, _, _, _ = paired_case
    with (root / 'execution-sources.zip').open('ab') as f:
        f.write(b'changed')
    with pytest.raises(ValueError, match='source archive changed'):
        runner.summarize(root)


def test_runner_cannot_create_new_measurements_inside_historical_bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    protected = tmp_path / 'results/ieee-access-2026-2/new-campaign'
    with pytest.raises(ValueError, match='immutable'):
        runner.run_campaign(config(), tmp_path / 'unused-paths.json', protected)
    assert not protected.exists()


def test_runtime_does_not_import_private_baselines(monkeypatch):
    import sys
    import torch
    monkeypatch.setitem(sys.modules, 'launcher', None)
    monkeypatch.setitem(sys.modules, 'baselines.dgc', None)
    monkeypatch.setattr(torch, 'set_num_interop_threads', lambda _: None)
    c = config()
    c['runtime'].update(device='cpu', cpu_threads=1)
    rt = runner.runtime(c, need_s2=False)
    assert rt['Optimizer'] is not None
    a = torch.sparse_coo_tensor(torch.tensor([[0], [2]]), torch.ones(1), (4, 4))
    assert list(rt['batches'](a)) == [a]
    assert rt['active'](a, 4, torch.device('cpu')).tolist() == [True, False, True, False]
    assert len(list(rt['batches'](torch.stack([a, a])))) == 2
    dense = a.to_dense()
    assert torch.equal(rt['active'](dense, 4, torch.device('cpu')),
                       rt['active'](a, 4, torch.device('cpu')))
