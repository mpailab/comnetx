"""Adapter orchestration checks without loading the optional ML stack.

Execute the actual s2cag function with dependency stubs. The spectral algorithm
is not rerun or reimplemented here; these checks cover observable trace and
return semantics while the optional TensorFlow/SciPy installation is absent.
"""
import ast
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'src/baselines/s2cag.py'


class _Adjacency:
    def __init__(self, n=4, nnz=8):
        self.shape = (n, n)
        self.nnz = nnz

    def size(self, axis):
        return self.shape[axis]


class _Features:
    def __init__(self, values, events):
        self.values = values
        self.events = events

    def toarray(self):
        self.events.append('densification')
        return self.values.copy()


class _Clock:
    def __init__(self):
        self.value = 0.0
        self.calls = 0

    def __call__(self):
        self.calls += 1
        self.value += 0.25
        return self.value


def _adapter(trace_allowed=True):
    tree = ast.parse(SOURCE.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == 's2cag')
    events, calls, metric_calls = [], [], []
    wall_clock, monotonic_clock = _Clock(), _Clock()

    def monotonic():
        if not trace_allowed:
            raise AssertionError('Tracing must remain optional')
        return monotonic_clock()

    def convert(adj, features, labels):
        events.append('conversion')
        return adj, features, labels, len(np.unique(labels))

    def preprocess(adj, features, **kwargs):
        events.append('preprocessing')
        return adj, _Features(features, events)

    def solver(features, k, norm_adj, T, alpha, **kwargs):
        calls.append({'k': k, 'T': T, 'alpha': alpha, **kwargs})
        events.append('solver')
        # Deliberately distinct first/final outputs detect best/first selection.
        labels = np.array([0, 0, 0, 0] if len(calls) == 1 else [1, 1, 2, 2])
        return labels, np.ones((4, 2))

    def metric(name):
        def compute(labels, prediction):
            metric_calls.append((name, np.array(prediction)))
            events.append(name)
            return 0.5
        return compute

    namespace = {
        'np': np,
        'torch': SimpleNamespace(Tensor=np.ndarray, arange=np.arange,
                                 abs=np.abs, tensor=lambda v: np.array(v, copy=True)),
        'sparse': SimpleNamespace(coo_matrix=lambda adj: adj),
        'time': wall_clock, 'perf_counter': monotonic,
        'torch_to_scipy': convert, 'preprocess_dataset': preprocess,
        'run_SSCAG': solver,
        'clustering_accuracy': metric('accuracy'), 'nmi': metric('nmi'),
        'ari': metric('ari'),
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return namespace['s2cag'], calls, metric_calls, events, wall_clock, monotonic_clock


class S2CAGTraceTests(unittest.TestCase):
    def test_trace_distinguishes_requested_embedding_and_nonempty_counts(self):
        adapter, calls, metrics, events, wall, _ = _adapter()
        trace = {'context': 'pair-1'}
        legacy_timing = {}
        with contextlib.redirect_stdout(io.StringIO()):
            result = adapter(_Adjacency(), np.ones((4, 3)),
                             labels=np.array([0, 0, 1, 2]),
                             timing_info=legacy_timing, trace_info=trace)
        np.testing.assert_array_equal(result, [1, 1, 2, 2])
        self.assertEqual(trace['requested_k'], 3)
        self.assertEqual(trace['embedding_width'], 2)
        self.assertEqual(trace['actual_nonempty_groups'], 2)
        self.assertTrue(trace['labels_supplied'])
        self.assertEqual(trace['input_vertices'], 4)
        self.assertEqual(trace['input_feature_columns'], 3)
        self.assertEqual(trace['normalized_adjacency_nnz'], 8)
        self.assertEqual(trace['solver_calls_completed'], 5)
        self.assertEqual(trace['returned_solver_call'], 4)
        self.assertEqual(trace['return_policy'], 'last_call')
        self.assertEqual(trace['context'], 'pair-1')
        self.assertEqual(trace['internal_metric_observations'], 1)
        self.assertEqual(len(metrics), 3)
        for _, prediction in metrics:
            np.testing.assert_array_equal(prediction, result)
        self.assertEqual(legacy_timing['conversion_time'], trace['legacy_conversion_seconds'])
        self.assertEqual(wall.calls, 2 + 5 + 1)  # Existing wall-clock calls unchanged.
        self.assertEqual(events[:3], ['conversion', 'preprocessing', 'densification'])
        self.assertTrue(all(c['T'] == 10 and c['tau'] == 7 and c['method'] == 'sub'
                            and c['alpha'] == 1 and c['gamma'] == 1 for c in calls))
        json.dumps(trace)  # Persistable without a NumPy-aware JSON encoder.

    def test_stage_intervals_cover_adapter_and_all_solver_calls(self):
        adapter, _, _, _, _, _ = _adapter()
        trace = {}
        with contextlib.redirect_stdout(io.StringIO()):
            adapter(_Adjacency(), np.ones((4, 3)), trace_info=trace, n_runs=2)
        stages = ['preprocessing_seconds', 'densification_seconds',
                  'solver_setup_seconds', 'solver_loop_seconds',
                  'internal_metrics_seconds', 'printing_and_output_seconds']
        self.assertAlmostEqual(sum(trace[name] for name in stages),
                               trace['adapter_total_seconds'])
        self.assertTrue(all(trace[name] >= 0 for name in stages))
        self.assertEqual(len(trace['solver_call_seconds']), 2)
        self.assertGreaterEqual(trace['solver_loop_seconds'],
                                sum(trace['solver_call_seconds']))
        self.assertGreaterEqual(trace['trace_overhead_seconds'], 0)
        self.assertFalse(trace['labels_supplied'])
        self.assertEqual(trace['requested_k'], 4)
        self.assertEqual(trace['actual_nonempty_groups'], 2)

    def test_explicit_settings_forwarded_without_changing_legacy_return(self):
        adapter, calls, _, _, _, _ = _adapter()
        trace = {}
        with contextlib.redirect_stdout(io.StringIO()):
            result, metrics = adapter(
                _Adjacency(), np.ones((4, 3)), np.array([0, 0, 1, 1]),
                T=3, n_runs=2, alpha=0.8, method='mod', gamma=0.7, tau=4,
                fdim=0, dataset='test-graph', metrics_mod=True, trace_info=trace)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[-1], {'k': 2, 'T': 3, 'alpha': 0.8, 'method': 'mod',
                                    'dataset': 'test-graph', 'gamma': 0.7, 'tau': 4})
        self.assertTrue(trace['settings']['gamma_used'])
        self.assertIsNone(trace['helper_defaults']['svd_random_state'])
        self.assertEqual(set(metrics), {'acc', 'nmi', 'ari', 'time'})
        self.assertTrue(all(len(v) == 1 for v in metrics.values()))
        np.testing.assert_array_equal(result, [1, 1, 2, 2])

    def test_untraced_call_keeps_return_and_does_not_read_diagnostic_clock(self):
        adapter, calls, _, _, wall, monotonic = _adapter(trace_allowed=False)
        timing = {}
        with contextlib.redirect_stdout(io.StringIO()):
            result = adapter(_Adjacency(), np.ones((4, 3)),
                             n_runs=2, timing_info=timing)
        self.assertIsInstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1, 1, 2, 2])
        self.assertEqual(len(calls), 2)
        self.assertEqual(monotonic.calls, 0)
        self.assertEqual(wall.calls, 2 + 2 + 1)
        self.assertEqual(timing, {'conversion_time': 0.25})


if __name__ == '__main__':
    unittest.main()
