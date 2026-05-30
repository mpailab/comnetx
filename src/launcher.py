import torch
import os
import time
import numpy as np
from dataclasses import dataclass
from os import PathLike

from optimizer import Optimizer
import sparse

from baselines.dgc import create_leiden
from metrics import Metrics, calculate_ground_truth_metrics
from our_utils import print_zone


def _initial_partition_cache_path(
    cache_dir: str | PathLike,
    dataset_name,
    init_batch_number,
    method_name,
    subcoms_depth: int,
) -> str:
    """
    Build the cache file path for an initial partition.

    Parameters
    ----------
    cache_dir : str | os.PathLike
        Directory that stores cached initial partitions.
    dataset_name : str
        Dataset key embedded into the file name.
    init_batch_number : int | str
        Bootstrap batch marker embedded into the file name.
    method_name : str
        Static method name embedded into the file name.
    subcoms_depth : int
        Number of stored hierarchy layers. Depth 1 keeps the historical cache
        name, while deeper partitions include a depth suffix.

    Returns
    -------
    str
        Full path to the ``.npz`` cache file.
    """
    depth_suffix = "" if subcoms_depth == 1 else f"_d:{subcoms_depth}"
    filename = f"{dataset_name}_b:{init_batch_number}_by_{method_name}"
    return os.path.join(cache_dir, f"{filename}{depth_suffix}.npz")


def _load_cached_initial_partition(cache_file: str | PathLike | None, device):
    """
    Load an initial partition cache when the file already exists.

    Parameters
    ----------
    cache_file : str | os.PathLike | None
        Candidate cache path. ``None`` disables loading.
    device : torch.device
        Runtime device for the returned partition tensor.

    Returns
    -------
    tuple[torch.Tensor, float] | None
        Cached partition and modularity, or ``None`` when no cache file exists.
    """
    if cache_file is None or not os.path.exists(cache_file):
        return None

    with np.load(cache_file, allow_pickle=False) as data:
        partition = torch.as_tensor(
            data["partition"],
            dtype=torch.long,
            device=device,
        )
        modularity = float(data["mod"])
    return partition, modularity


def _save_cached_initial_partition(
    cache_file: str | PathLike | None,
    partition,
    modularity: float,
) -> None:
    """
    Persist an initial partition cache when caching is enabled.

    Parameters
    ----------
    cache_file : str | os.PathLike | None
        Destination cache path. ``None`` disables saving.
    partition : torch.Tensor
        Partition tensor to save on CPU so cache files are device-independent.
    modularity : float
        Modularity associated with the first full-graph partition.
    """
    if cache_file is None:
        return

    np.savez_compressed(cache_file, partition=partition.cpu().numpy(), mod=modularity)


def _build_layered_initial_partition(
    adj_matrix,
    adj_algo,
    init_partition,
    method_name,
    subcoms_depth: int,
    device,
):
    """
    Extend a flat initial partition into a hierarchy of community layers.

    Parameters
    ----------
    adj_matrix : torch.Tensor
        Original graph snapshot; its node count defines restored layer length.
    adj_algo : torch.Tensor
        Same snapshot already moved to the runtime device used by the backend.
    init_partition : torch.Tensor
        First full-graph partition produced by the static backend.
    method_name : str
        Static method name passed to ``create_leiden`` for aggregated graphs.
    subcoms_depth : int
        Desired number of hierarchy layers in the returned tensor.
    device : torch.device
        Runtime device for backend partitions and returned layers.

    Returns
    -------
    torch.Tensor
        Tensor with shape ``(subcoms_depth, nodes_num)``. Each row contains
        labels restored to the original graph nodes.
    """
    layers = [init_partition]
    nodes_num = adj_matrix.size(0)

    # Optimizer.aggregate expects sparse COO input. Dense snapshots are
    # converted once, then the working adjacency stays sparse between layers.
    adj_work = adj_algo.float()
    if not adj_work.is_sparse:
        adj_work = adj_work.to_sparse_coo()
    adj_work = adj_work.coalesce()

    node_mask = torch.ones(nodes_num, dtype=torch.bool, device=adj_work.device)
    node_ids = torch.arange(nodes_num, device=adj_work.device)

    for _ in range(1, subcoms_depth):
        # Reindex current communities to compact ids before building the
        # aggregation pattern P used in P * A * P.T.
        current_partition = layers[-1].to(adj_work.device)
        old_idx, inverse = torch.unique(
            current_partition,
            sorted=True,
            return_inverse=True,
        )
        aggr_idx = torch.stack((inverse, node_ids))
        aggr_adj_ptn = sparse.tensor(
            aggr_idx,
            (old_idx.size(0), nodes_num),
            adj_work.dtype,
        )
        aggr_adj = Optimizer.aggregate(adj_work, aggr_adj_ptn)
        del aggr_adj_ptn

        # Run the same local method on the aggregated graph and restore labels
        # back to original graph nodes.
        temp_algo = create_leiden(method_name, aggr_adj)
        temp_algo.apply()
        aggr_partition = torch.as_tensor(
            temp_algo.partition(),
            dtype=torch.long,
            device=device,
        )
        restored_partition = old_idx[aggr_partition[inverse]]
        layers.append(restored_partition)

        # Keep the working adjacency consistent with Optimizer.run: after each
        # layer, remove edges that cross the new partition.
        adj_work = Optimizer.cut_by_partition(adj_work, node_mask, restored_partition)

    return torch.stack(layers)


def compute_initial_partition(
    adj_matrix,
    dataset_name,
    init_batch_number,
    method_name="leidenalg",
    cache_dir: str | PathLike | None = None,
    subcoms_depth=1,
    device=None,
):
    """
    Build an initial community partition for an adjacency matrix.

    Parameters
    ----------
    adj_matrix : torch.Tensor
        Adjacency matrix of the initial graph snapshot with shape (n, n).
        The argument used to be named ``batch``, but it is the adjacency matrix
        itself, not a batch of arbitrary data.
    dataset_name : str
        Dataset key used only for the cache file name.
    init_batch_number : int | str
        Initial batch identifier used only for the cache file name.
    method_name : str
        Static community detection method passed to ``create_leiden``.
    cache_dir : str | os.PathLike | None
        Directory for storing/loading the computed partition and modularity.
    subcoms_depth : int
        Number of partition layers to build. ``1`` preserves the previous
        behavior and returns a single label vector with shape (n,). Values above
        ``1`` return a layered tensor/array with shape (subcoms_depth, n).
    device : torch.device | str | None
        Device for all tensors created inside this function. If None, use the
        device of ``adj_matrix``.

    Returns
    -------
    tuple[torch.Tensor, float]
        Initial partition labels and modularity of the first full-graph
        partition. For layered output each row contains labels for original
        graph nodes restored from the corresponding aggregated graph.
    """
    if subcoms_depth < 1:
        raise ValueError("subcoms_depth must be at least 1")

    if device is None:
        device = adj_matrix.device
    else:
        device = torch.device(device)

    cache_file = None
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = _initial_partition_cache_path(
            cache_dir,
            dataset_name,
            init_batch_number,
            method_name,
            subcoms_depth,
        )

        cached_partition = _load_cached_initial_partition(cache_file, device)
        if cached_partition is not None:
            return cached_partition

    adj_algo = adj_matrix.to(device)
    temp_algo = create_leiden(method_name, adj_algo)
    temp_algo.apply()
    # Most backends return partition labels on CPU; keep the launcher state on
    # the requested runtime device from the first conversion onward.
    init_partition = torch.as_tensor(
        temp_algo.partition(),
        dtype=torch.long,
        device=device,
    )
    init_mod = temp_algo.modularity()

    if subcoms_depth > 1:
        init_partition = _build_layered_initial_partition(
            adj_matrix,
            adj_algo,
            init_partition,
            method_name,
            subcoms_depth,
            device,
        )

    _save_cached_initial_partition(cache_file, init_partition, init_mod)

    return init_partition, init_mod


_LAUNCH_MODES = {"smart", "naive", "raw", "dynamic"}


@dataclass(frozen=True)
class _LaunchConfig:
    """Small immutable bundle for options shared by launcher helpers."""

    # Dataset key used in cache filenames and reporting.
    dataset_name: str
    # Static baseline or dynamic backend method name.
    method: str
    # Optional iteration/epoch budget forwarded to baselines that use it.
    baseline_iter: int | None
    # Normalized launch mode: "smart", "naive", "raw", or "dynamic".
    mode: str
    # Number of hierarchy levels maintained by smart mode.
    smart_subcoms_depth: int
    # Graph-neighborhood expansion radius for smart updates.
    smart_neighborhood_step: int
    # Output verbosity level used by launcher-local reporting.
    verbose: int
    # Whether Optimizer-backed modes may use CUDA when available.
    use_gpu: bool
    # Feature aggregation mode passed to Optimizer.
    aggregation_mode: str
    # Optional directory for cached initial partitions.
    cache_dir: str | PathLike | None
    # Bootstrap batch marker extracted from p:n strategies.
    init_batch_number: str | None
    ground_truth_metrics: bool


def _print_verbose(verbose: int, level: int, *args, **kwargs) -> None:
    """Print only when the requested verbosity level is enabled."""
    if verbose >= level:
        print(*args, **kwargs)


def _load_dataset_if_needed(ds, batches_strategy):
    """
    Return a loaded dataset object for both supported call styles.

    Parameters
    ----------
    ds : Dataset | str
        Either an already loaded dataset-like object or a legacy dataset name.
        String inputs are still accepted because older tests and helper scripts
        call ``dynamic_launch("dataset_name", ...)`` directly.
    batches_strategy : int | str | None
        Batch split strategy passed to ``Dataset.load`` when ``ds`` is a name.

    Returns
    -------
    Dataset-like object
        The original object for dataset inputs, or a freshly loaded Dataset for
        string inputs.
    """
    if not isinstance(ds, str):
        return ds

    # Import lazily so unit tests can replace the dataset module without paying
    # for the real dataset configuration and path discovery machinery.
    from datasets import Dataset

    dataset = Dataset(ds)
    dataset.load(batches_strategy=batches_strategy)
    return dataset


def _build_launch_config(
    ds,
    batches_strategy,
    underlying_static_method,
    baseline_iter,
    mode,
    smart_subcoms_depth,
    smart_neighborhood_step,
    verbose,
    use_gpu,
    aggregation_mode,
    cache_dir: str | PathLike | None,
    ground_truth_metrics: bool
) -> _LaunchConfig:
    """
    Normalize public launch parameters into the compact internal config object.

    ``dynamic_launch`` keeps its historical public signature, while private
    helpers receive this single object instead of long, order-sensitive argument
    lists.
    """
    normalized_mode = _normalize_launch_mode(mode)
    if smart_subcoms_depth < 1:
        raise ValueError("smart_subcoms_depth must be at least 1")
    if smart_neighborhood_step < 0:
        raise ValueError("smart_neighborhood_step must be non-negative")

    return _LaunchConfig(
        dataset_name=ds.name,
        method=underlying_static_method,
        baseline_iter=baseline_iter,
        mode=normalized_mode,
        smart_subcoms_depth=smart_subcoms_depth,
        smart_neighborhood_step=smart_neighborhood_step,
        verbose=verbose,
        use_gpu=use_gpu,
        aggregation_mode=aggregation_mode,
        cache_dir=cache_dir,
        init_batch_number=_init_batch_number(batches_strategy),
        ground_truth_metrics= ground_truth_metrics
    )


def _normalize_launch_mode(mode: str) -> str:
    """
    Normalize and validate the launcher mode.

    Parameters
    ----------
    mode : str
        User-provided mode name. Surrounding whitespace is ignored and case is
        normalized before validation.

    Returns
    -------
    str
        One of ``"smart"``, ``"naive"``, ``"raw"``, or ``"dynamic"``.

    Raises
    ------
    ValueError
        If ``mode`` is not a string or is not supported by the launcher.
    """
    if not isinstance(mode, str):
        raise ValueError(f"Unsupported launch mode: {mode!r}")

    normalized_mode = mode.lower().strip()
    if normalized_mode not in _LAUNCH_MODES:
        supported = ", ".join(sorted(_LAUNCH_MODES))
        raise ValueError(
            f"Unsupported launch mode: {mode!r}. Expected one of: {supported}."
        )
    return normalized_mode


def _init_batch_number(batches_strategy):
    """
    Extract the bootstrap batch marker from p:n batch strategies.

    Parameters
    ----------
    batches_strategy : int | str | None
        Strategy used by dataset loading. Strategies containing ``":"`` encode
        an initial large batch followed by dynamic update batches, for example
        ``"999:100"``.

    Returns
    -------
    str | None
        The part before ``":"`` for special strategies, otherwise ``None``.
    """
    strategy = str(batches_strategy)
    if ":" not in strategy:
        return None
    return strategy.split(":", 1)[0]


def _iter_adjacency_batches(adj_matrix):
    """
    Lazily iterate over static or temporal adjacency batches.

    Parameters
    ----------
    adj_matrix : torch.Tensor
        Static adjacency with shape ``(n, n)`` or temporal adjacency with shape
        ``(t, n, n)``.

    Returns
    -------
    Iterator[torch.Tensor]
        A one-element iterator for static graphs, or a lazy iterator over the
        first dimension for dynamic graphs.

    Raises
    ------
    ValueError
        If the adjacency tensor has an unsupported number of dimensions.
    """
    if adj_matrix.ndim == 2:
        yield adj_matrix
        return
    if adj_matrix.ndim == 3:
        for batch_idx in range(adj_matrix.size(0)):
            yield adj_matrix[batch_idx]
        return
    raise ValueError(f"Unsupported ds.adj ndim: {adj_matrix.ndim}")


def _first_snapshot(adj_matrix):
    """
    Return the graph snapshot used for bootstrap partitions and MFC metrics.

    Parameters
    ----------
    adj_matrix : torch.Tensor
        Static adjacency with shape ``(n, n)`` or temporal adjacency with shape
        ``(t, n, n)``.

    Returns
    -------
    torch.Tensor
        The static adjacency itself, or the first temporal snapshot.

    Raises
    ------
    ValueError
        If the adjacency tensor has an unsupported number of dimensions.
    """
    if adj_matrix.ndim == 2:
        return adj_matrix
    if adj_matrix.ndim == 3:
        return adj_matrix[0]
    raise ValueError(f"Unsupported ds.adj ndim: {adj_matrix.ndim}")

def final_adj(adj_matrix):
    if adj_matrix.ndim == 2:
        return adj_matrix
    if adj_matrix.ndim == 3:
        pass


def _compute_launch_initial_partition(
    adj_matrix,
    dataset_name,
    init_batch_number,
    cache_dir: str | PathLike | None = None,
    subcoms_depth=1,
    device=None,
    verbose=0,
):
    """
    Compute or load the initial partition used by p:n launch strategies.

    Parameters
    ----------
    adj_matrix : torch.Tensor
        Initial graph snapshot used for bootstrap community detection.
    dataset_name : str
        Dataset key used in the initial-partition cache file name.
    init_batch_number : int | str
        Batch marker extracted from a p:n strategy, also used in the cache file
        name.
    cache_dir : str | os.PathLike | None
        Optional cache directory forwarded to ``compute_initial_partition``.
    subcoms_depth : int
        Number of community layers to build for smart mode.
    device : torch.device | str | None
        Runtime device expected by the caller. ``None`` keeps the batch device.
    verbose : int
        Launcher verbosity. Values above zero print the bootstrap modularity.

    Returns
    -------
    torch.Tensor
        Initial partition tensor. The modularity is reported here when
        verbosity allows it, but is not returned because launch callers do not
        use it for control flow or metrics.

    Notes
    -----
    The launch code always uses leidenalg for this bootstrap partition, even
    when the measured method is different. Keeping that policy in one helper
    avoids scattering cache naming and reporting logic across all execution
    paths.
    """
    init_partition, init_mod = compute_initial_partition(
        adj_matrix,
        dataset_name,
        init_batch_number,
        "leidenalg",
        cache_dir,
        subcoms_depth=subcoms_depth,
        device=device,
    )
    _print_verbose(verbose, 1, f"Initial modularity: {init_mod:.2g}")
    return init_partition


def _active_nodes_mask(batch, nodes_num: int, device: torch.device) -> torch.Tensor:
    """
    Build the first smart-mode affected-node mask directly from a snapshot.

    Parameters
    ----------
    batch : torch.Tensor
        First adjacency snapshot passed to ``Optimizer``.
    nodes_num : int
        Number of nodes in the graph; this defines the output mask length.
    device : torch.device
        Device where the mask must live, usually ``opt.runtime_device()``.

    Returns
    -------
    torch.Tensor
        Boolean tensor of shape ``(nodes_num,)`` with all nodes touched by at
        least one non-zero adjacency entry marked as ``True``.

    Notes
    -----
    Later smart batches can use Optimizer.update_adj(..., return_mask=True),
    because those batches are explicit graph updates. The first batch is
    different: it initializes Optimizer.adj, so there is no previous graph to
    diff against. Every endpoint of every non-zero edge is therefore treated as
    affected.
    """
    if batch.is_sparse:
        # COO indices already contain both edge endpoints; coalesce first so
        # duplicate entries do not produce unnecessary downstream work.
        batch_coo = batch if batch.is_coalesced() else batch.coalesce()
        active_nodes = batch_coo.indices().unique()
        mask = torch.zeros(nodes_num, dtype=torch.bool, device=device)
        if active_nodes.numel() > 0:
            mask[active_nodes.to(device)] = True
        return mask
    else:
        # Dense matrices can avoid materializing every non-zero coordinate.
        # Reducing rows and columns gives the endpoint set with O(n) output
        # memory instead of O(nnz) coordinate memory.
        active_mask = (batch != 0).any(dim=0) | (batch != 0).any(dim=1)
        return active_mask.to(device)


def _run_optimizer_batch(
    opt: Optimizer,
    config: _LaunchConfig,
    affected_nodes_mask,
) -> float:
    """
    Run one Optimizer-backed batch and return measured algorithm time.

    Parameters
    ----------
    opt : Optimizer
        Stateful optimizer holding the current adjacency, features, and
        community layers.
    config : _LaunchConfig
        Normalized launch options. ``config.mode`` is one of ``"smart"``,
        ``"naive"``, or ``"raw"`` in this helper.
    affected_nodes_mask : torch.Tensor | None
        Smart-mode mask returned by ``_active_nodes_mask`` or
        ``Optimizer.update_adj``. It is ignored by naive/raw modes.
    Returns
    -------
    float
        Wall-clock batch time minus local backend conversion time.

    Notes
    -----
    Optimizer.local_algorithm may spend part of its time converting tensors for
    a backend. Existing result databases store the algorithm time without that
    conversion overhead, so this helper preserves the previous accounting by
    subtracting the conversion-time delta from the wall-clock delta.
    """
    time_s = time.perf_counter()
    conversion_time_s = opt.conversion_time

    if config.mode == "smart":
        # Smart mode expands the directly affected nodes, then reruns the local
        # algorithm only inside the touched hierarchy maintained by Optimizer.
        runtime_adj = opt.runtime_adj()
        affected_nodes_mask = opt.neighborhood(
            runtime_adj,
            affected_nodes_mask,
            step=config.smart_neighborhood_step,
        )
        opt.run(affected_nodes_mask)
    else:
        # Naive mode reuses the previous labels as a warm start. Raw mode starts
        # the static baseline from scratch by passing labels=None.
        labels = opt.coms[0] if config.mode == "naive" else None
        coms = opt.local_algorithm(
            opt.runtime_adj(),
            opt.runtime_features(),
            labels=labels,
        )
        opt.set_communities(communities=coms.unsqueeze(0), replace_subcoms_depth=True)

    time_e = time.perf_counter()
    conversion_time_e = opt.conversion_time

    total_batch_time = time_e - time_s
    conversion_time = conversion_time_e - conversion_time_s
    return total_batch_time - conversion_time


def _print_optimizer_batch_result(
    config: _LaunchConfig,
    opt: Optimizer,
    modularity: float,
    measured_time: float,
) -> None:
    """
    Print a per-batch Optimizer result.

    Parameters
    ----------
    config : _LaunchConfig
        Normalized launch options. Verbosity controls printing, and method/mode
        preserve ldleiden timing semantics for raw and naive runs.
    opt : Optimizer
        Optimizer instance containing optional backend timing metadata.
    modularity : float
        Batch modularity to report.
    measured_time : float
        Batch time to report for all methods except the ldleiden special case.
    """
    if config.verbose < 2:
        return

    print(f"Modularity: {modularity:.2g}")
    if config.method == "ldleiden" and config.mode in {"naive", "raw"}:
        algorithm_time = opt.last_timing_info["algorithm_time"]
        print(f"Algorithm time: {algorithm_time:.2f}")
    else:
        print(f"Time: {measured_time:.2f}")


def _run_dynamic_mfc(
    ds,
    config: _LaunchConfig,
):
    """
    Run MFC's own dynamic implementation.

    Parameters
    ----------
    ds : Dataset-like object
        Loaded dataset containing ``adj`` and ``is_directed`` attributes.
    config : _LaunchConfig
        Normalized launch options, including dataset/cache keys, verbosity, and
        MFC iteration budget.

    Returns
    -------
    list[dict[str, float]]
        Single-result list with final modularity and measured runtime.

    Notes
    -----
    MFC consumes the whole temporal adjacency tensor at once, so it cannot share
    the regular per-batch dynamic backend loop. The initial partition is still
    computed from the first snapshot for p:n strategies to keep bootstrap
    behavior aligned with the other modes.
    """
    from baselines.mfc import mfc_adopted

    time_s = time.perf_counter()
    init_partition = None
    if config.init_batch_number is not None:
        # p:n runs start from a static partition of the first snapshot. That
        # partition is passed into MFC so its dynamic run begins from the same
        # state as the other launch modes.
        first_snapshot = _first_snapshot(ds.adj)
        init_partition = _compute_launch_initial_partition(
            adj_matrix=first_snapshot,
            dataset_name=config.dataset_name,
            init_batch_number=config.init_batch_number,
            cache_dir=config.cache_dir,
            verbose=config.verbose,
        )

    # mfc_adopted owns the temporal loop internally and returns one final label
    # vector. Keep this computation outside verbose-only reporting so verbosity
    # only affects reporting, never whether the dynamic run itself happens.
    with print_zone(config.verbose >= 4):
        last_partition = mfc_adopted(
            adj=ds.adj,
            features=getattr(ds, "features", None),
            network_type="MFC",
            num_epoch=config.baseline_iter,
            initial_partition=init_partition,
        )

    measured_time = time.perf_counter() - time_s
    mod = Metrics.modularity(_first_snapshot(ds.adj), last_partition, directed=ds.is_directed)

    _print_verbose(config.verbose, 2, f"Modularity: {mod:.2g}")
    _print_verbose(config.verbose, 2, f"Baseline calls: {1}")
    _print_verbose(config.verbose, 2, f"Time: {measured_time:.2f}")

    return [{"modularity": mod, "time": measured_time}], last_partition


def _run_dynamic_lago(
    ds,
    config: _LaunchConfig,
):
    """
    Run LAGO on the complete temporal adjacency tensor.

    LAGO consumes a link stream rather than a per-batch update API, so dynamic
    mode forwards the whole loaded adjacency sequence and reports one final
    partition, mirroring the shape of other full-temporal baselines.
    """
    from baselines.lago import lago_partition

    timing_info = {"conversion_time": 0.0}
    time_s = time.perf_counter()
    with print_zone(config.verbose >= 4):
        last_partition = lago_partition(
            ds.adj,
            directed=ds.is_directed,
            nb_iter=config.baseline_iter,
            timing_info=timing_info,
        )
    measured_time = max(
        0.0,
        time.perf_counter() - time_s - timing_info["conversion_time"],
    )
    full_adj = _compute_full_adj(ds.adj)
    mod = Metrics.modularity(full_adj, last_partition, directed=ds.is_directed)

    _print_verbose(config.verbose, 2, f"Modularity: {mod:.2g}")
    _print_verbose(config.verbose, 2, f"Baseline calls: {1}")
    _print_verbose(config.verbose, 2, f"Time: {measured_time:.2f}")

    return [{"modularity": mod, "time": measured_time}], last_partition


def _run_dynamic_backend(
    batches_iter,
    config: _LaunchConfig,
):
    """
    Run LDLeiden/DFLeiden/leidenalg/networkit through the streaming API.

    Parameters
    ----------
    batches_iter : Iterable[torch.Tensor]
        Sequence of adjacency snapshots or update batches.
    config : _LaunchConfig
        Normalized launch options, including backend method, cache keys, and
        verbosity.

    Returns
    -------
    list[dict[str, float]]
        Per-processed-batch modularity and runtime entries.

    Raises
    ------
    ValueError
        If the backend receives no adjacency batches at all.
    """
    results = []
    algo = None
    seen_batch = False

    for batch_idx, batch in enumerate(batches_iter):
        seen_batch = True
        _print_verbose(config.verbose, 2, "  Batch", batch_idx)

        if batch_idx == 0:
            # The first batch constructs the dynamic backend object. For p:n
            # strategies, the first batch is only the initial state; metrics are
            # recorded from subsequent update batches to match historical data.
            init_partition = None
            if config.init_batch_number is not None:
                init_partition = _compute_launch_initial_partition(
                    adj_matrix=batch,
                    dataset_name=config.dataset_name,
                    init_batch_number=config.init_batch_number,
                    cache_dir=config.cache_dir,
                    verbose=config.verbose,
                )

            algo = create_leiden(config.method, batch, partition=init_partition)

            if init_partition is not None:
                # FIXME: Some dynamic backends need a priming apply() after receiving
                # an external partition. Without it, update()+apply() can enter
                # an uninitialized C++ state on the next batch.
                algo.apply()
                continue

        # Keep the historical streaming protocol: every emitted batch,
        # including the first non-special snapshot, is passed through update().
        algo.update(batch)
        elapsed_ms = algo.apply()
        measured_time = elapsed_ms / 1000.0
        mod = algo.modularity()

        _print_verbose(config.verbose, 2, f"Modularity: {mod:.2g}")
        _print_verbose(config.verbose, 2, f"Time: {measured_time:.2f}")

        results.append({"modularity": mod, "time": measured_time})

    if not seen_batch:
        raise ValueError("dynamic backend received no adjacency batches")

    last_partition = torch.as_tensor(
        algo.partition(),
        dtype=torch.long,
    )
    return results, last_partition


def _run_optimizer_modes(
    ds,
    batches_iter,
    config: _LaunchConfig,
):
    """
    Run smart, naive, and raw modes through the shared Optimizer pipeline.

    Parameters
    ----------
    ds : Dataset-like object
        Loaded dataset containing ``features`` and ``is_directed`` attributes.
    batches_iter : Iterable[torch.Tensor]
        Sequence of adjacency snapshots or update batches.
    config : _LaunchConfig
        Normalized launch options for Optimizer-backed modes.

    Returns
    -------
    list[dict[str, float]]
        Per-processed-batch modularity and runtime entries.
    """
    results = []
    opt = None
    features = getattr(ds, "features", None)

    for batch_idx, batch in enumerate(batches_iter):
        _print_verbose(config.verbose, 2, "  Batch", batch_idx)

        if batch_idx == 0:
            # The first batch initializes Optimizer's persistent state. Later
            # batches are added to opt.adj through update_adj().
            opt = Optimizer(
                batch,
                features,
                subcoms_depth=(
                    config.smart_subcoms_depth
                    if config.mode == "smart"
                    else 1
                ),
                method=config.method,
                baseline_iter=config.baseline_iter,
                verbose=config.verbose,
                use_gpu=config.use_gpu,
                aggregation_mode=config.aggregation_mode,
            )

            if config.init_batch_number is not None:
                # p:n strategies treat batch zero as the initial graph state.
                # The initial partition is installed and timing starts from the
                # following update batch.
                init_partition = _compute_launch_initial_partition(
                    adj_matrix=batch,
                    dataset_name=config.dataset_name,
                    init_batch_number=config.init_batch_number,
                    cache_dir=config.cache_dir,
                    subcoms_depth=opt.subcoms_depth,
                    device=opt.runtime_device(),
                    verbose=config.verbose,
                )
                if init_partition.dim() == 1:
                    init_partition = init_partition.unsqueeze(0)
                opt.set_communities(communities=init_partition)
                continue

            # Smart mode needs an initial affected-node set even though there
            # was no previous adjacency to diff against.
            affected_nodes_mask = (
                _active_nodes_mask(batch, opt.nodes_num, opt.runtime_device())
                if config.mode == "smart"
                else None
            )
        else:
            # For update batches, Optimizer both mutates the accumulated graph
            # and optionally returns the directly affected nodes for smart mode.
            affected_nodes_mask = opt.update_adj(
                batch,
                return_mask=(config.mode == "smart"),
            )

        measured_time = _run_optimizer_batch(opt, config, affected_nodes_mask)
        mod = opt.modularity(directed=ds.is_directed)
        _print_optimizer_batch_result(config, opt, mod, measured_time)

        results.append({"modularity": mod, "time": measured_time})
    
    last_partition = opt.coms[0]
    return results, last_partition


def _print_launch_summary(results, metrics_list, verbose: int) -> None:
    """
    Print the same final summary format for all launch paths.

    Parameters
    ----------
    results : list[dict[str, float]]
        Result entries produced by one of the execution helpers.
    verbose : int
        Launcher verbosity. Level 1 prints final modularity and total time;
        higher levels print total time after per-batch details.
    """
    total_measured_time = sum(result["time"] for result in results)
    final_result = results[-1]
    if verbose >= 1:
        for met in metrics_list:
            print(f"{met}: {final_result[met]:.2g}")
        print(f"Total time: {total_measured_time:.2f}")
        print("-----------------------------------------------")

def _compute_full_adj(adj):

    if adj.ndim == 2:
        full_adj = adj
    elif adj.ndim == 3:
        first = adj[0]
        device = first.device
        dtype = first.dtype
        full_adj = None
        for batch in adj:
            batch_dev = batch.to(device=device, dtype=dtype)
            if batch_dev.is_sparse and not batch_dev.is_coalesced():
                batch_dev = batch_dev.coalesce()
            if full_adj is None:
                full_adj = batch_dev.clone()
            else:
                full_adj = full_adj + batch_dev
    else:
        raise ValueError(f"Unsupported ds.adj ndim: {adj.ndim}")

    if not full_adj.is_sparse:
        full_adj = full_adj.to_sparse_coo()
    else:
        full_adj = full_adj.coalesce()

    return full_adj

def dynamic_launch(ds, batches_strategy,
                    underlying_static_method: str,
                    baseline_iter: int = None,
                    mode: str = "smart",
                    smart_subcoms_depth: int = 5,
                    smart_neighborhood_step: int = 1,
                    verbose: int = 1,
                    use_gpu: bool = False,
                    aggregation_mode: str = "sum",
                    cache_dir: str | PathLike | None = None,
                    ground_truth_metrics: bool = False):
    """
    Launch community detection experiments for static and dynamic graph batches.

    Parameters
    ----------
    ds : Dataset-like object | str
        Loaded dataset object, or a dataset name accepted by ``Dataset``.
    batches_strategy : int | str | None
        Batch split strategy. Strategies containing ``":"`` enable the
        bootstrap initial-partition path.
    underlying_static_method : str
        Baseline or backend method name, for example ``"leidenalg"``,
        ``"networkit"``, ``"mfc"``, ``"magi"``, or ``"ldleiden"``.
    baseline_iter : int | None
        Optional iteration/epoch budget forwarded to baselines that use it.
    mode : str
        Launch mode: ``"smart"``, ``"naive"``, ``"raw"``, or ``"dynamic"``.
    smart_subcoms_depth : int
        Number of community hierarchy levels maintained in smart mode.
    smart_neighborhood_step : int
        Neighborhood expansion radius for smart mode updates.
    verbose : int
        Launcher verbosity.
    use_gpu : bool
        Whether Optimizer-backed modes may use CUDA when available.
    aggregation_mode : str
        Feature aggregation mode forwarded to Optimizer.
    cache_dir : str | os.PathLike | None
        Optional directory for cached initial partitions.

    Returns
    -------
    list[dict[str, float]]
        Per-processed-batch result entries with ``"modularity"`` and ``"time"``.

    Notes
    -----
    The public function now acts as an orchestration layer: it normalizes inputs,
    chooses the correct execution path, and delegates the details to small
    helpers. This keeps the subtle mode-specific behavior explicit while
    preserving the existing result format.
    """
    # Normalize the external API first. The rest of the function can then work
    # with a loaded dataset object and one compact config value.
    ds = _load_dataset_if_needed(ds, batches_strategy)
    config = _build_launch_config(
        ds=ds,
        batches_strategy=batches_strategy,
        underlying_static_method=underlying_static_method,
        baseline_iter=baseline_iter,
        mode=mode,
        smart_subcoms_depth=smart_subcoms_depth,
        smart_neighborhood_step=smart_neighborhood_step,
        verbose=verbose,
        use_gpu=use_gpu,
        aggregation_mode=aggregation_mode,
        cache_dir=cache_dir,
        ground_truth_metrics = ground_truth_metrics
    )

    # Select the execution engine. MFC dynamic mode owns its full temporal loop,
    # other dynamic methods use the streaming backend API, and all remaining
    # modes use Optimizer.
    if config.mode == "dynamic" and config.method == "mfc":
        results, last_partition = _run_dynamic_mfc(ds, config)
    elif config.mode == "dynamic" and config.method == "lago":
        results, last_partition = _run_dynamic_lago(ds, config)
    elif config.mode == "dynamic":
        results, last_partition = _run_dynamic_backend(_iter_adjacency_batches(ds.adj), config)
    else:
        results, last_partition = _run_optimizer_modes(ds, _iter_adjacency_batches(ds.adj), config)

    if results:
        full_adj = _compute_full_adj(ds.adj)

        metrics = {}
        if config.ground_truth_metrics and ds.label is not None:
            metrics = calculate_ground_truth_metrics(ds.label, last_partition)
            labels_mod = Metrics.modularity(full_adj, ds.label, directed=ds.is_directed)
            metrics["Labels modularity"] = labels_mod
        final_mod = Metrics.modularity(full_adj, last_partition, directed=ds.is_directed)
        metrics["Final modularity"] = final_mod
        results[-1].update(metrics)
        
        metrics_list = list(metrics.keys())
        _print_launch_summary(results, metrics_list, config.verbose)

    return results
