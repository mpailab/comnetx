#!/usr/bin/env python3
"""Prepare, run, and validate new paired S2CAG experiments (never old results).

The plan and summary commands use only the standard library. Actual runs use
isolated sequential subprocesses, one shared saved initializer, and the current
production Optimizer. This is an instrumented campaign, not a reconstruction
of the missing historical executable.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import inspect
import itertools
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "s2cag-pairs-v1"
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
               "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS")
ENVIRONMENT_KEYS = ("python", "platform", "machine", "processor", "cpu_count",
                    "packages", "installed_packages", "thread_environment",
                    "torch_threads", "torch_interop_threads", "cuda_version",
                    "cuda_visible_devices", "gpu", "blas")
FIXED_BACKEND = dict(method="sub", T=10, alpha=1.0, tau=7, n_runs=5,
                     gamma=1.0, fdim=0, svd_seed=0, oversampling=10,
                     rounding_iterations=100)
FIXED_INITIALIZER = dict(backend="public_leidenalg", resolution=1.0,
                         seed=1, n_iterations=2)
# These audited helpers implement random_state="warn" as seed 0. Keep their
# identity explicit instead of treating the signature sentinel as a new seed.
AUDITED_HELPERS = {
    "func.py": "ba5426fb5424aa333108a4698b5b861bf3d6843b2f340fc20e29987faaea31de",
    "utils_s2cag.py": "a30256cf68de40429ad880dfdc892e84a40ba001b330cb1165501052fad1cfaa",
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def protocol_key(protocol):
    if protocol == "0:10":
        return "10"  # The old loader's p=0 path does not implement ten updates.
    parts = protocol.split(":")
    if len(parts) != 2 or not all(p.isdigit() for p in parts):
        raise ValueError(f"Expected p:q protocol, got {protocol!r}")
    p, q = map(int, parts)
    if p not in (9, 99, 999) or q < 1:
        raise ValueError("Supported protocols: 9:q, 99:q, 999:q, and 0:10")
    return protocol


def validate_config(c):
    required = {"schema_version", "name", "datasets", "protocols", "feature_modes",
                "repetitions", "random_features", "backend", "comnetx",
                "initializer", "runtime", "measurement"}
    if set(c) != required or c["schema_version"] != 1:
        raise ValueError(f"Invalid configuration keys/schema: {set(c) ^ required}")
    if not isinstance(c["name"], str) or not c["name"]:
        raise ValueError("name must be a nonempty string")
    for key in ("datasets", "protocols", "feature_modes"):
        if not isinstance(c[key], list) or not c[key] or len(set(c[key])) != len(c[key]):
            raise ValueError(f"{key} must be a nonempty list without duplicates")
    if not set(c["datasets"]) <= {"dyn_cora", "dyn_pubmed", "arxivmath"}:
        raise ValueError("This campaign supports Cora (smoke), PubMed, and arxivmath")
    for p in c["protocols"]:
        protocol_key(p)
    if not set(c["feature_modes"]) <= {"random", "dataset"}:
        raise ValueError("Supported feature modes: dataset, random")
    if type(c["repetitions"]) is not int or not 1 <= c["repetitions"] <= 20:
        raise ValueError("repetitions must be an integer in [1,20]")
    if c["backend"] != FIXED_BACKEND or c["initializer"] != FIXED_INITIALIZER:
        raise ValueError("This campaign fixes the documented S2CAG and public Leiden settings")
    if c["comnetx"] != dict(depth=3, radius=1, aggregation="normalized"):
        raise ValueError("This campaign fixes L=3, r=1, mean aggregation")
    if (set(c["measurement"]) != {"comnetx_profile"}
            or type(c["measurement"]["comnetx_profile"]) is not bool):
        raise ValueError("measurement requires boolean comnetx_profile")
    f, rt = c["random_features"], c["runtime"]
    if set(f) != {"dimension", "seed"} or f["dimension"] != 64 or type(f["seed"]) is not int:
        raise ValueError("Specify random dimension=64 and integer seed")
    if set(rt) != {"device", "cpu_threads", "seed"} or rt["device"] not in ("cpu", "cuda"):
        raise ValueError("runtime requires device=cpu|cuda, cpu_threads, seed")
    if type(rt["cpu_threads"]) is not int or rt["cpu_threads"] < 1 or type(rt["seed"]) is not int:
        raise ValueError("Invalid runtime threads/seed")
    return c


def make_plan(c):
    validate_config(c)
    jobs = []
    for cell, (ds, feature, protocol) in enumerate(itertools.product(
            c["datasets"], c["feature_modes"], c["protocols"])):
        cell_id = f"{ds}__{feature}__{protocol.replace(':', '-')}"
        for rep in range(c["repetitions"]):
            order = ["full", "local"] if (cell + rep) % 2 == 0 else ["local", "full"]
            jobs.append(dict(cell_id=cell_id, pair_id=f"{cell_id}__r{rep+1:02}",
                             dataset=ds, feature_mode=feature, protocol=protocol,
                             loader_key=protocol_key(protocol), repetition=rep+1, order=order))
    return dict(schema=SCHEMA, name=c["name"], pairs=len(jobs),
                measured_processes=2*len(jobs), jobs=jobs,
                policy="sequential fresh processes; alternating full/local order; fixed seeds")


def source_files():
    files = [Path(__file__).resolve()]
    for rel in ("scripts/paper/run_s2cag.sh", "conf/ieee_access/s2cag_pairs/confirmatory.json",
                "requirements/ieee-access-s2cag.txt"):
        files.append(ROOT / rel)
    for base in (ROOT / "src", ROOT / "baselines/S2CAG/src"):
        files += sorted(base.rglob("*.py"))
    files += sorted((ROOT / "datasets-info/json").glob("*.json"))
    if not (ROOT / "baselines/S2CAG/src/func.py").is_file():
        raise RuntimeError("Local S2CAG sources are required; the clean paper archive contains placeholders")
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(set(files))}


def child_env(c):
    env = os.environ.copy()
    env.update({name: str(c["runtime"]["cpu_threads"]) for name in THREAD_VARS})
    env.update(PYTHONHASHSEED=str(c["runtime"]["seed"]), PYTHONUNBUFFERED="1")
    return env


def runtime(c, need_s2=True):
    """Import on worker path only; thread limits must already be set by parent."""
    start = time.perf_counter()
    sys.path.insert(0, str(ROOT / "src"))
    import numpy as np
    import torch
    from datasets import Dataset
    from optimizer import Optimizer
    from metrics import Metrics
    from baselines.leiden import leidenalg_partition
    torch.set_num_threads(c["runtime"]["cpu_threads"])
    torch.set_num_interop_threads(1)
    device = c["runtime"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use an explicit CPU config/--device cpu, never silent fallback")
    seed = c["runtime"]["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    adapter = None
    if need_s2:
        import scipy
        # The preserved SNEM implementation expects sparse/dense division to
        # densify. SciPy 1.11 changed that operation to return sparse arrays.
        scipy_version = tuple(int(v) for v in scipy.__version__.split(".")[:2])
        if scipy_version >= (1, 11):
            raise RuntimeError("Preserved S2CAG rounding requires SciPy <1.11; use the separate Python 3.11 rerun environment (see docs/s2cag-paired-reruns-ru.md)")
        adapter = importlib.import_module("baselines.s2cag")
        helper = importlib.import_module("func")
        if Path(helper.__file__).resolve() != ROOT / "baselines/S2CAG/src/func.py":
            raise RuntimeError("Unexpected S2CAG helper module (import shadowing)")
        for filename, expected in AUDITED_HELPERS.items():
            if digest(ROOT / "baselines/S2CAG/src" / filename) != expected:
                raise RuntimeError(f"S2CAG helper {filename} differs from audited source; review defaults before running")
        for name in ("subspace_svd", "sub_randomized_svd"):
            parameters = inspect.signature(getattr(helper, name)).parameters
            for key, expected in (("random_state", (0, "warn")), ("n_oversamples", (10,))):
                if parameters[key].default not in expected:
                    raise RuntimeError(f"Unexpected {name} default {key}")
        if inspect.signature(helper.SNEM_rounding).parameters["T"].default != 100:
            raise RuntimeError("Unexpected S2CAG rounding budget")
        if "trace_info" not in inspect.signature(adapter.s2cag).parameters:
            raise RuntimeError("The trace-enabled local S2CAG adapter is required")
    return dict(np=np, torch=torch, Dataset=Dataset, Optimizer=Optimizer,
                batches=adjacency_batches, active=active_nodes_mask, Metrics=Metrics,
                leiden=leidenalg_partition, adapter=adapter,
                import_seconds=time.perf_counter()-start)


def adjacency_batches(adj):
    """Read the loader's static or temporal output without importing baselines."""
    if adj.ndim == 2:
        yield adj
    elif adj.ndim == 3:
        for index in range(adj.size(0)):
            yield adj[index]
    else:
        raise ValueError(f"Unsupported adjacency ndim: {adj.ndim}")


def active_nodes_mask(batch, nodes_num, device):
    import torch
    if batch.is_sparse:
        active = batch.coalesce().indices().unique()
        mask = torch.zeros(nodes_num, dtype=torch.bool, device=device)
        mask[active.to(device)] = True
        return mask
    return ((batch != 0).any(dim=0) | (batch != 0).any(dim=1)).to(device)


def tensor_hash(t):
    if t is None:
        return None
    h = hashlib.sha256()
    h.update(str(tuple(t.shape)).encode()); h.update(str(t.dtype).encode())
    if t.is_sparse:
        t = t.detach().coalesce().cpu()
        arrays = [t.indices().numpy(), t.values().numpy()]
    else:
        arrays = [t.detach().cpu().contiguous().numpy()]
    for arr in arrays:
        h.update(memoryview(arr).cast("B"))
    return h.hexdigest()


def load_inputs(c, job, paths, rt):
    ds = rt["Dataset"](job["dataset"], paths)
    ds.load(batches_strategy=job["loader_key"], feature_mode=job["feature_mode"],
            random_feat_dim=c["random_features"]["dimension"],
            random_feat_seed=c["random_features"]["seed"])
    if ds.is_directed:
        ds._force_undirected()
    if ds.features is None or ds.label is None:
        raise ValueError("This paired campaign requires supplied features and reference labels")
    batches = list(rt["batches"](ds.adj))
    if not batches:
        raise ValueError("Empty stream")
    p, q = map(int, job["protocol"].split(":"))
    if len(batches) != q + int(p > 0):
        raise ValueError(f"Protocol {job['protocol']} requires {q} updates; loader returned {len(batches)} batches including bootstrap. Check stream files and tail size.")
    n = batches[0].shape[0]
    if ds.features.ndim != 2 or ds.features.shape[0] != n or ds.label.shape != (n,):
        raise ValueError("Expected features [n,d] and reference labels [n] on this stream")
    feature_values = ds.features.coalesce().values() if ds.features.is_sparse else ds.features
    if not rt["torch"].isfinite(feature_values).all() or not rt["torch"].isfinite(ds.label).all():
        raise ValueError("Nonfinite input features/reference labels")
    fingerprints = dict(features=tensor_hash(ds.features), reference_labels=tensor_hash(ds.label),
                        batches=[tensor_hash(b) for b in batches],
                        feature_shape=list(ds.features.shape), nodes=int(batches[0].shape[0]),
                        directed=bool(ds.is_directed))
    return ds, batches, fingerprints


def make_hierarchy(adj, c, rt, growth=False):
    torch, Opt = rt["torch"], rt["Optimizer"]
    n = adj.shape[0]
    if growth:
        return torch.arange(n).repeat(c["comnetx"]["depth"], 1)
    import sparse
    base = adj.detach().cpu().float().coalesce()
    layers = [Opt.canonicalize_partition(rt["leiden"](base))]
    for _ in range(1, c["comnetx"]["depth"]):
        labels, inverse = torch.unique(layers[-1], sorted=True, return_inverse=True)
        pattern = sparse.tensor(torch.stack((inverse, torch.arange(n))),
                                (len(labels), n), base.dtype)
        quotient = Opt.aggregate(base, pattern)
        layers.append(Opt.canonicalize_partition(rt["leiden"](quotient)[inverse]))
    hierarchy = torch.stack(layers)
    if not Opt.hierarchy_is_nested(hierarchy):
        raise RuntimeError("Non-nested bootstrap")
    return hierarchy


def environment_info(rt):
    packages = {}
    for name in ("numpy", "scipy", "scikit-learn", "joblib", "torch", "tensorflow",
                 "igraph", "leidenalg", "threadpoolctl"):
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name] = None
    from threadpoolctl import threadpool_info
    torch = rt["torch"]
    info = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                machine=platform.machine(), processor=platform.processor(), cpu_count=os.cpu_count(),
                packages=packages, thread_environment={k: os.environ.get(k) for k in THREAD_VARS},
                torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
                blas=threadpool_info(), cuda_version=torch.version.cuda,
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                s2cag_solver_device="CPU (NumPy/SciPy)")
    info["installed_packages"] = dict(sorted(
        (dist.metadata["Name"], dist.version) for dist in importlib.metadata.distributions()
        if dist.metadata["Name"]))
    if torch.cuda.is_available():
        info["gpu"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    for executable, args in (("nvidia-smi", ["--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
                             ("lscpu", [])):
        try:
            p = subprocess.run([executable]+args, capture_output=True, text=True, timeout=15)
            info[executable] = p.stdout if p.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired): info[executable] = None
    return info


def warmup(c, rt):
    """A discarded tiny numerical call, distinct from every real stream update."""
    torch = rt["torch"]
    n = 12
    idx = torch.arange(n)
    edges = torch.stack((torch.cat((idx, idx)), torch.cat(((idx+1)%n, (idx-1)%n))))
    adj = torch.sparse_coo_tensor(edges, torch.ones(2*n), (n,n)).coalesce()
    g = torch.Generator(device="cpu").manual_seed(42)
    features = torch.randn((n,8), generator=g)
    trace = {}
    start = time.perf_counter()
    rt["adapter"].s2cag(adj, features, idx%2, trace_info=trace,
                        **{k:c["backend"][k] for k in ("T","n_runs","alpha","method","gamma","tau","fdim")})
    return dict(wall_seconds=time.perf_counter()-start, trace=trace,
                purpose="discarded numerical warmup; no actual stream update omitted")


def prepare(spec):
    c, job = spec["config"], spec["job"]
    rt = runtime(c, need_s2=False)
    ds, batches, fingerprints = load_inputs(c, job, spec["paths"], rt)
    root = Path(spec["campaign"])
    # Shared across feature modes, repetitions and all q for the same initial p.
    p = job["protocol"].split(":")[0]
    key = f"{job['dataset']}__p{p}"
    init_path = root / "initial" / (key + ".npz")
    meta_path = init_path.with_suffix(".json")
    start = time.perf_counter()
    if init_path.exists():
        meta = read_json(meta_path)
        if meta["initial_graph_sha256"] != fingerprints["batches"][0]:
            raise ValueError("Initial graph differs within a shared p; refusing to reuse hierarchy")
        if digest(init_path) != meta["file_sha256"]:
            raise ValueError("Initial hierarchy file changed")
        with rt["np"].load(init_path, allow_pickle=False) as z:
            hierarchy = rt["torch"].from_numpy(z["hierarchy"].copy())
    else:
        hierarchy = make_hierarchy(batches[0], c, rt, growth=p=="0")
        init_path.parent.mkdir(parents=True, exist_ok=True)
        rt["np"].savez_compressed(init_path, hierarchy=hierarchy.numpy())
        meta = dict(initial_graph_sha256=fingerprints["batches"][0],
                    file_sha256=digest(init_path), hierarchy_sha256=tensor_hash(hierarchy),
                    fine_partition_sha256=tensor_hash(hierarchy[0]),
                    construction_seconds=time.perf_counter()-start,
                    policy="singletons" if p=="0" else "explicit public leidenalg at every hierarchy level",
                    community_counts=[int(rt["torch"].unique(row).numel()) for row in hierarchy],
                    initializer=c["initializer"], environment=environment_info(rt))
        write_json(meta_path, meta)
    write_json(spec["prepared"], dict(schema=SCHEMA, input=fingerprints,
               initial_file=str(init_path.relative_to(root)), initial=meta,
               expected_updates=len(batches) - (p!="0")))


def measure(spec):
    c, job, mode = spec["config"], spec["job"], spec["mode"]
    root = Path(spec["campaign"])
    prepared = read_json(spec["prepared"])
    rt = runtime(c)
    torch, np, Opt = rt["torch"], rt["np"], rt["Optimizer"]
    ds, batches, fingerprints = load_inputs(c, job, spec["paths"], rt)
    if fingerprints != prepared["input"]:
        raise ValueError("Loaded inputs differ from saved paired preparation")
    init_path = root / prepared["initial_file"]
    if digest(init_path) != prepared["initial"]["file_sha256"]:
        raise ValueError("Saved initial hierarchy changed")
    with np.load(init_path, allow_pickle=False) as z:
        hierarchy = torch.from_numpy(z["hierarchy"].copy())
    if tensor_hash(hierarchy) != prepared["initial"]["hierarchy_sha256"]:
        raise ValueError("Initial hierarchy payload mismatch")
    warm = warmup(c, rt)
    random.seed(c["runtime"]["seed"]); np.random.seed(c["runtime"]["seed"])
    torch.manual_seed(c["runtime"]["seed"])
    opt = Opt(batches[0], ds.features,
              communities=hierarchy[:1].clone() if mode=="full" else hierarchy.clone(),
              subcoms_depth=1 if mode=="full" else c["comnetx"]["depth"],
              method="s2cag", baseline_iter=c["backend"]["T"], verbose=0,
              use_gpu=c["runtime"]["device"]=="cuda", aggregation_mode=c["comnetx"]["aggregation"])
    if tensor_hash(opt.coms[0]) != prepared["initial"]["fine_partition_sha256"]:
        raise ValueError("Actual starting fine labels differ")
    device = opt.runtime_device()
    def sync():
        if device.type=="cuda": torch.cuda.synchronize(device)
    adapter = rt["adapter"]
    original = adapter.s2cag
    calls = []
    def traced(adj, features=None, labels=None, **kwargs):
        trace = {}
        kwargs.update({k:c["backend"][k] for k in ("T","n_runs","alpha","method","gamma","tau","fdim")})
        try:
            result = original(adj, features, labels, trace_info=trace, **kwargs)
        except Exception as exc:
            trace.update(status="failed", input_vertices=int(adj.shape[0]),
                         labels_supplied=labels is not None, error=f"{type(exc).__name__}: {exc}")
            calls.append(trace)
            raise
        calls.append(trace)
        return result
    adapter.s2cag = traced
    out_path = Path(spec["result"])
    label_dir = out_path.parent / (mode + "-labels")
    label_dir.mkdir(parents=True)
    result = dict(schema=SCHEMA, status="running", job=job, mode=mode, config=c,
                  source_fingerprint=spec["source_fingerprint"], input=fingerprints,
                  initial=prepared["initial"], environment=environment_info(rt),
                  actual_tensor_device=str(device), import_seconds=rt["import_seconds"],
                  warmup=warm, rows=[], expected_updates=prepared["expected_updates"],
                  timing_policy="synchronized instrumented wall time, including trace/profile overhead; legacy interval separately")
    write_json(out_path, result)
    try:
        for batch_index, batch in enumerate(batches):
            if batch_index==0 and job["protocol"]!="0:10": continue
            calls.clear()
            sync(); start=time.perf_counter()
            if batch_index==0:
                affected=rt["active"](batch, opt.nodes_num, device)
            else:
                affected=opt.update_adj(batch, return_mask=True)
            sync(); update_seconds=time.perf_counter()-start
            before_conversion=opt.conversion_time
            sync(); start=time.perf_counter()
            neighborhood_seconds=None
            if mode=="local":
                ns=time.perf_counter()
                expanded=opt.neighborhood(opt.runtime_adj(), affected, step=c["comnetx"]["radius"])
                if c["measurement"]["comnetx_profile"]:
                    sync(); neighborhood_seconds=time.perf_counter()-ns
                opt.run(expanded, collect_profile=c["measurement"]["comnetx_profile"])
            else:
                labels=opt.local_algorithm(opt.runtime_adj(), opt.runtime_features(), labels=opt.coms[0])
                opt.set_communities(labels.unsqueeze(0), replace_subcoms_depth=True)
            sync(); wall=time.perf_counter()-start
            conversion=opt.conversion_time-before_conversion
            if conversion<0 or conversion>wall+1e-5:
                raise ValueError("Invalid conversion/outer timer accounting")
            trace_copy=list(calls)
            for call in trace_copy:
                if call["solver_calls_completed"]!=c["backend"]["n_runs"]:
                    raise ValueError("Unexpected number of actual solver calls")
                if not 1 <= call["actual_nonempty_groups"] <= call["embedding_width"] <= call["requested_k"]:
                    raise ValueError("Invalid requested/embedding/realized group counts")
                if call["embedding_width"] > min(call["input_vertices"], call["input_feature_columns"])-1:
                    raise ValueError("Embedding exceeds retained nonleading singular vectors")
            metric_start=time.perf_counter()
            labels=opt.coms[0].detach().cpu()
            if labels.shape != (opt.nodes_num,) or not torch.isfinite(labels).all():
                raise ValueError("Invalid reported partition")
            q=rt["Metrics"].modularity(opt.adj, opt.coms[0], directed=False)
            nmi=rt["Metrics"].nmi(ds.label, labels)
            sync(); metric_seconds=time.perf_counter()-metric_start
            row=dict(update=len(result["rows"])+1, source_batch_index=batch_index,
                     batch_sha256=fingerprints["batches"][batch_index],
                     snapshot_sha256=tensor_hash(opt.adj), label_sha256=tensor_hash(labels),
                     modularity=float(q), nmi=float(nmi), nonempty_groups=int(torch.unique(labels).numel()),
                     batch_apply_seconds=update_seconds, processing_wall_seconds=wall,
                     backend_preprocessing_seconds=conversion,
                     processing_excluding_preprocessing_seconds=wall-conversion,
                     neighborhood_seconds=neighborhood_seconds,
                     external_metrics_seconds=metric_seconds,
                     s2cag_calls=trace_copy,
                     comnetx_profile=opt.last_run_profile if mode=="local" else None)
            label_file=label_dir / f"update-{row['update']:04}.npz"
            io_start=time.perf_counter()
            np.savez_compressed(label_file, labels=labels.numpy())
            row.update(label_file=str(label_file.relative_to(root)),
                       label_file_sha256=digest(label_file), label_write_seconds=time.perf_counter()-io_start)
            result["rows"].append(row)
            write_json(out_path, result)
        if len(result["rows"])!=prepared["expected_updates"]:
            raise ValueError("Incomplete update stream")
        result["status"]="complete"
        write_json(out_path,result)
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                      failed_update_s2cag_calls=list(calls))
        write_json(out_path,result)
        raise
    finally:
        adapter.s2cag=original


def validate_pair(full, local, root):
    if full["status"]!="complete" or local["status"]!="complete":
        raise ValueError("Incomplete/failed pair")
    for key in ("job", "config", "source_fingerprint", "input", "initial", "actual_tensor_device"):
        if full[key]!=local[key]: raise ValueError(f"Pair mismatch: {key}")
    for key in ENVIRONMENT_KEYS:
        if full["environment"].get(key)!=local["environment"].get(key):
            raise ValueError(f"Pair environment mismatch: {key}")
    if (full["mode"],local["mode"])!=("full","local"):
        raise ValueError("Expected full/local modes")
    if (not full["rows"] or len(full["rows"])!=len(local["rows"])
            or len(full["rows"])!=full["expected_updates"]
            or len(local["rows"])!=local["expected_updates"]
            or full["expected_updates"]!=int(full["job"]["protocol"].split(":")[1])):
        raise ValueError("Unequal/incomplete update counts")
    for number,(f,l) in enumerate(zip(full["rows"],local["rows"]),1):
        for key in ("update","source_batch_index","batch_sha256","snapshot_sha256"):
            if f[key]!=l[key]: raise ValueError(f"Paired update mismatch: {key}")
        for row in (f,l):
            batch_index = number - int(full["job"]["protocol"]=="0:10")
            if (row["update"]!=number or row["source_batch_index"]!=batch_index
                    or row["batch_sha256"]!=full["input"]["batches"][batch_index]):
                raise ValueError("Paired update order/input batch mismatch")
            for key in ("processing_wall_seconds", "processing_excluding_preprocessing_seconds"):
                if not math.isfinite(row[key]) or row[key]<=0: raise ValueError(f"Invalid {key}")
            if digest(Path(root)/row["label_file"])!=row["label_file_sha256"]:
                raise ValueError("Output label file changed")
            for key in ("modularity", "nmi"):
                if not math.isfinite(row[key]): raise ValueError(f"Nonfinite {key}")
    times={mode:{key:sum(r[key] for r in value["rows"]) for key in
                  ("processing_wall_seconds","processing_excluding_preprocessing_seconds")}
           for mode,value in (("full",full),("local",local))}
    return dict(pair_id=full["job"]["pair_id"], cell_id=full["job"]["cell_id"],
                repetition=full["job"]["repetition"], updates=len(full["rows"]), time=times,
                speedup={key:times["full"][key]/times["local"][key] for key in times["full"]},
                delta_q=local["rows"][-1]["modularity"]-full["rows"][-1]["modularity"],
                delta_nmi=local["rows"][-1]["nmi"]-full["rows"][-1]["nmi"],
                final={m:{k:r["rows"][-1][k] for k in ("modularity","nmi","nonempty_groups")}
                       for m,r in (("full",full),("local",local))})


def summarize(directory):
    root=Path(directory).resolve()
    manifest=read_json(root/"campaign.json")
    if manifest["status"]!="complete": raise ValueError("Campaign is not complete; partial runs are not admitted")
    if manifest["plan"] != make_plan(manifest["config"]):
        raise ValueError("Saved plan differs from campaign configuration")
    if manifest["source_fingerprint"] != json_hash(manifest["source_files"]):
        raise ValueError("Source manifest fingerprint mismatch")
    if digest(root/"execution-sources.zip") != manifest["source_archive_sha256"]:
        raise ValueError("Saved execution source archive changed")
    if read_json(root/"config.json") != manifest["config"]:
        raise ValueError("Saved configuration changed")
    pairs=[]
    reference_environment=None
    final_snapshots={}
    for job in manifest["plan"]["jobs"]:
        base=root/"pairs"/job["pair_id"]
        full, local = read_json(base/"full.json"), read_json(base/"local.json")
        prepared = read_json(root/"prepared"/(job["cell_id"]+".json"))
        for result in (full, local):
            for key, expected in (("job", job), ("config", manifest["config"]),
                                  ("source_fingerprint", manifest["source_fingerprint"]),
                                  ("input", prepared["input"]), ("initial", prepared["initial"]),
                                  ("expected_updates", prepared["expected_updates"])):
                if result[key] != expected:
                    raise ValueError(f"Result differs from saved campaign/preparation: {key}")
        if digest(root/prepared["initial_file"]) != prepared["initial"]["file_sha256"]:
            raise ValueError("Saved shared initialization changed")
        pairs.append(validate_pair(full, local, root))
        identity={k:full["environment"].get(k) for k in ENVIRONMENT_KEYS}
        if reference_environment is None: reference_environment=identity
        elif identity!=reference_environment:
            raise ValueError("Execution environment changed between pairs; do not pool repetitions")
        # Dividing one fixed tail into q batches must leave the same end graph.
        stream_key=(job["dataset"], job["feature_mode"], job["protocol"].split(":")[0])
        final=full["rows"][-1]["snapshot_sha256"]
        if stream_key in final_snapshots and final_snapshots[stream_key]!=final:
            raise ValueError("Final graph differs across subdivisions/repetitions")
        final_snapshots[stream_key]=final
    grouped={}
    for p in pairs: grouped.setdefault(p["cell_id"],[]).append(p)
    cells={}
    for cell,ps in grouped.items():
        cells[cell]={"repetitions":len(ps),"paired_repeats":ps,
                     "mean_delta_q":statistics.mean(p["delta_q"] for p in ps),
                     "mean_delta_nmi":statistics.mean(p["delta_nmi"] for p in ps),
                     "median_speedup":{k:statistics.median(p["speedup"][k] for p in ps)
                                       for k in ps[0]["speedup"]}}
    result=dict(schema=SCHEMA, status="validated", cells=cells,
                note="Fixed-seed repetitions assess runtime repeatability, not variation across independent graphs/features. No historical records pooled.")
    write_json(root/"paired-summary.json", result)
    return result


def run_child(action, spec, path, c):
    write_json(path,spec)
    with path.with_suffix(".log").open("w") as log:
        try:
            subprocess.run([sys.executable,str(Path(__file__).resolve()),action,str(path)],
                           cwd=ROOT,env=child_env(c),stdout=log,stderr=subprocess.STDOUT,check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"{action} failed; see {path.with_suffix('.log')}") from exc


def run_campaign(c, paths, directory):
    root=Path(directory).resolve()
    # Prevent accidental writes into immutable historical bundles, including a new child.
    protected=ROOT/"results/ieee-access-2026-2"
    if root==protected or protected in root.parents:
        raise ValueError("Historical measurements are immutable")
    sources=source_files(); fp=json_hash(sources)
    root.mkdir(parents=True,exist_ok=False)
    write_json(root/"config.json",c)
    write_json(root/"paths.json",read_json(paths))
    with zipfile.ZipFile(root/"execution-sources.zip","w",zipfile.ZIP_DEFLATED) as z:
        for name in sources: z.write(ROOT/name,name)
    plan=make_plan(c)
    manifest=dict(schema=SCHEMA,status="running",created_at=datetime.now(timezone.utc).isoformat(),
                  config=c,paths_config_sha256=digest(paths),source_files=sources,
                  source_fingerprint=fp,source_archive_sha256=digest(root/"execution-sources.zip"),plan=plan)
    write_json(root/"campaign.json",manifest)
    prepared_cells=set()
    try:
        for job in plan["jobs"]:
            prepared=root/"prepared"/(job["cell_id"]+".json")
            common=dict(config=c,job=job,paths=str(root/"paths.json"),campaign=str(root),
                        prepared=str(prepared),source_fingerprint=fp)
            if job["cell_id"] not in prepared_cells:
                run_child("_prepare",common,root/"jobs"/(job["cell_id"]+"-prepare.json"),c)
                prepared_cells.add(job["cell_id"])
            for mode in job["order"]:
                spec=dict(common,mode=mode,result=str(root/"pairs"/job["pair_id"]/(mode+".json")))
                print(f"{job['pair_id']} {mode}",flush=True)
                run_child("_measure",spec,root/"jobs"/(job["pair_id"]+"-"+mode+".json"),c)
        manifest["status"]="complete"
        write_json(root/"campaign.json",manifest)
        summarize(root)
    except Exception as exc:
        manifest.update(status="failed",error=f"{type(exc).__name__}: {exc}")
        write_json(root/"campaign.json",manifest)
        raise
    return root


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    for action in ("plan","preflight","run"):
        q=sub.add_parser(action)
        q.add_argument("config",type=Path)
        q.add_argument("--device",choices=("cpu","cuda"))
        if action in ("preflight","run"):
            q.add_argument("--paths-config",type=Path,required=True)
        if action=="run": q.add_argument("--output-dir",type=Path,required=True)
    q=sub.add_parser("summarize");q.add_argument("directory",type=Path)
    for action in ("_prepare","_measure","_preflight"):
        q=sub.add_parser(action);q.add_argument("spec",type=Path)
    a=p.parse_args()
    if a.command.startswith("_"):
        spec=read_json(a.spec)
        validate_config(spec["config"])
        if spec["source_fingerprint"]!=json_hash(source_files()):
            raise RuntimeError("Source files changed since campaign preparation")
        if a.command=="_prepare": prepare(spec)
        elif a.command=="_measure": measure(spec)
        else:
            rt=runtime(spec["config"])
            warm=warmup(spec["config"],rt)
            for job in spec["jobs"]:
                ds,batches,fp=load_inputs(spec["config"],job,spec["paths"],rt)
                print(json.dumps(dict(dataset=job["dataset"],protocol=job["protocol"],
                                     feature_mode=job["feature_mode"],input=fp)))
            print(json.dumps(dict(status="passed",environment=environment_info(rt),warmup=warm),indent=2))
        return
    if a.command=="summarize":
        print(json.dumps(summarize(a.directory),indent=2));return
    c=read_json(a.config)
    if a.device: c["runtime"]["device"]=a.device
    validate_config(c)
    if a.command=="plan": print(json.dumps(make_plan(c),indent=2));return
    paths=a.paths_config.resolve()
    if not paths.is_file(): raise FileNotFoundError(paths)
    if a.command=="run": print(run_campaign(c,paths,a.output_dir));return
    import tempfile
    with tempfile.TemporaryDirectory(prefix="comnetx-s2-preflight-") as tmp:
        spec=dict(config=c,paths=str(paths),source_fingerprint=json_hash(source_files()),
                  jobs=list({j["cell_id"]:j for j in make_plan(c)["jobs"]}.values()))
        path=Path(tmp)/"preflight.json"
        write_json(path,spec)
        subprocess.run([sys.executable,str(Path(__file__).resolve()),"_preflight",str(path)],
                       cwd=ROOT,env=child_env(c),check=True)


if __name__=="__main__":
    main()
