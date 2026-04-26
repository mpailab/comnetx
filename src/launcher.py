import torch
import json
import os
import time
import numpy as np

from optimizer import Optimizer
from our_utils import print_zone
import sparse

from dynamic_graphs_communities import LDLeiden, DFLeiden, Leidenalg, Networkit
from baselines.dgc import create_leiden
from metrics import Metrics


def compute_initial_partition(
    adj_matrix,
    dataset_name,
    init_batch_number,
    method_name="leidenalg",
    cache_dir=None,
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
    cache_dir : str | None
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

    loaded = False
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        depth_suffix = "" if subcoms_depth == 1 else f"_d:{subcoms_depth}"
        filename = os.path.join(
            cache_dir,
            f"{dataset_name}_b:{init_batch_number}_by_{method_name}{depth_suffix}.npz",
        )
        if os.path.exists(filename):
            with np.load(filename, allow_pickle=True) as data:
                init_partition = torch.as_tensor(
                    data["partition"],
                    dtype=torch.long,
                    device=device,
                )
                init_mod = float(data["mod"])
            loaded = True

    if not loaded:
        adj_algo = adj_matrix.to(device)
        temp_algo = create_leiden(method_name, adj_algo)
        temp_algo.apply()
        # Most backends return partition labels on CPU; keep the launcher state
        # on the requested runtime device from the first conversion onward.
        init_partition = torch.as_tensor(
            temp_algo.partition(),
            dtype=torch.long,
            device=device,
        )
        init_mod = temp_algo.modularity()

        if subcoms_depth > 1:
            layers = [init_partition]
            nodes_num = adj_matrix.size(0)
            adj_work = adj_algo.float()
            if not adj_work.is_sparse:
                adj_work = adj_work.to_sparse_coo()
            adj_work = adj_work.coalesce()
            node_mask = torch.ones(
                nodes_num,
                dtype=torch.bool,
                device=adj_work.device,
            )

            for _ in range(1, subcoms_depth):
                # Reindex current communities to compact ids before building
                # the aggregation pattern P used in P * A * P.T.
                current_partition = layers[-1].to(adj_work.device)
                old_idx, inverse = torch.unique(
                    current_partition,
                    sorted=True,
                    return_inverse=True,
                )
                aggr_idx = torch.stack((
                    inverse,
                    torch.arange(nodes_num, device=device),
                ))
                aggr_adj_ptn = sparse.tensor(
                    aggr_idx,
                    (old_idx.size(0), nodes_num),
                    adj_work.dtype,
                )
                aggr_adj = Optimizer.aggregate(adj_work, aggr_adj_ptn)
                del aggr_adj_ptn

                # Run the same local method on the aggregated graph and restore
                # labels back to original graph nodes.
                temp_algo = create_leiden(method_name, aggr_adj)
                temp_algo.apply()
                aggr_partition = torch.as_tensor(
                    temp_algo.partition(),
                    dtype=torch.long,
                    device=device,
                )
                restored_partition = old_idx[aggr_partition[inverse]]
                layers.append(restored_partition)

                # Keep the working adjacency consistent with Optimizer.run:
                # after each layer, remove edges that cross the new partition.
                adj_work = Optimizer.cut_by_partition(
                    adj_work,
                    node_mask,
                    restored_partition,
                )

            init_partition = torch.stack(layers)

        if cache_dir is not None:
            np.savez_compressed(
                filename,
                partition=init_partition.cpu().numpy(),
                mod=init_mod,
            )

    return init_partition, init_mod


_LAUNCH_MODES = {"smart", "naive", "raw", "dynamic"}


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
    Convert a static or temporal adjacency tensor into an iterable of batches.

    Parameters
    ----------
    adj_matrix : torch.Tensor
        Static adjacency with shape ``(n, n)`` or temporal adjacency with shape
        ``(t, n, n)``.

    Returns
    -------
    list[torch.Tensor] | tuple[torch.Tensor, ...]
        A one-element list for static graphs, or the snapshots returned by
        ``torch.unbind`` for dynamic graphs.

    Raises
    ------
    ValueError
        If the adjacency tensor has an unsupported number of dimensions.
    """
    if adj_matrix.ndim == 2:
        return [adj_matrix]
    if adj_matrix.ndim == 3:
        return torch.unbind(adj_matrix)
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


def _compute_launch_initial_partition(
    adj_matrix,
    dataset_name,
    init_batch_number,
    cache_dir=None,
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
    with print_zone(verbose >= 1):
        print(f"Initial modularity: {init_mod:.2g}")
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
    else:
        # Dense non-zero coordinates have shape (nnz, 2). A flat unique() over
        # both columns gives the same endpoint set as sparse COO indices.
        nonzero_coordinates = torch.nonzero(batch, as_tuple=False)
        active_nodes = nonzero_coordinates.unique()

    mask = torch.zeros(nodes_num, dtype=torch.bool, device=device)
    if active_nodes.numel() > 0:
        mask[active_nodes.to(device)] = True
    return mask


def _run_optimizer_batch(
    opt: Optimizer,
    mode: str,
    affected_nodes_mask,
    smart_neighborhood_step: int,
) -> float:
    """
    Run one Optimizer-backed batch and return measured algorithm time.

    Parameters
    ----------
    opt : Optimizer
        Stateful optimizer holding the current adjacency, features, and
        community layers.
    mode : str
        One of ``"smart"``, ``"naive"``, or ``"raw"``. Dynamic mode is handled
        by a separate backend path.
    affected_nodes_mask : torch.Tensor | None
        Smart-mode mask returned by ``_active_nodes_mask`` or
        ``Optimizer.update_adj``. It is ignored by naive/raw modes.
    smart_neighborhood_step : int
        Number of graph-neighborhood expansion steps applied before smart mode
        reruns the local method.

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

    if mode == "smart":
        # Smart mode expands the directly affected nodes, then reruns the local
        # algorithm only inside the touched hierarchy maintained by Optimizer.
        runtime_adj = opt.runtime_adj()
        affected_nodes_mask = opt.neighborhood(
            runtime_adj,
            affected_nodes_mask,
            step=smart_neighborhood_step,
        )
        opt.run(affected_nodes_mask)
    else:
        # Naive mode reuses the previous labels as a warm start. Raw mode starts
        # the static baseline from scratch by passing labels=None.
        labels = opt.coms if mode == "naive" else None
        coms = opt.local_algorithm(
            opt.runtime_adj(),
            opt.runtime_features(),
            labels=labels,
        )
        opt.set_communities(
            communities=coms.unsqueeze(0),
            replace_subcoms_depth=True,
        )

    time_e = time.perf_counter()
    conversion_time_e = opt.conversion_time

    total_batch_time = time_e - time_s
    conversion_time = conversion_time_e - conversion_time_s
    return total_batch_time - conversion_time


def _print_optimizer_batch_result(
    verbose: int,
    method: str,
    mode: str,
    opt: Optimizer,
    modularity: float,
    measured_time: float,
) -> None:
    """
    Print a per-batch Optimizer result.

    Parameters
    ----------
    verbose : int
        Launcher verbosity; per-batch output is enabled at level 2 and above.
    method : str
        Static method name used by Optimizer.
    mode : str
        Launcher mode, used to preserve ldleiden timing semantics for raw and
        naive runs.
    opt : Optimizer
        Optimizer instance containing optional backend timing metadata.
    modularity : float
        Batch modularity to report.
    measured_time : float
        Batch time to report for all methods except the ldleiden special case.
    """
    with print_zone(verbose >= 2):
        print(f"Modularity: {modularity:.2g}")
        if method == "ldleiden" and mode in {"naive", "raw"}:
            algorithm_time = opt.last_timing_info["algorithm_time"]
            print(f"Algorithm time: {algorithm_time:.2f}")
        else:
            print(f"Time: {measured_time:.2f}")


def _run_dynamic_mfc(
    ds,
    dataset_name,
    init_batch_number,
    baseline_iter,
    verbose,
    cache_dir,
):
    """
    Run MFC's own dynamic implementation.

    Parameters
    ----------
    ds : Dataset-like object
        Loaded dataset containing ``adj`` and ``is_directed`` attributes.
    dataset_name : str
        Dataset key used for initial-partition cache naming.
    init_batch_number : str | None
        Bootstrap marker for p:n strategies. ``None`` disables bootstrap
        partition loading/computation.
    baseline_iter : int | None
        Number of MFC epochs/iterations forwarded to ``mfc_adopted``.
    verbose : int
        Launcher verbosity.
    cache_dir : str | os.PathLike | None
        Optional cache directory for the bootstrap partition.

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
    if init_batch_number is not None:
        # p:n runs start from a static partition of the first snapshot. That
        # partition is passed into MFC so its dynamic run begins from the same
        # state as the other launch modes.
        first_snapshot = _first_snapshot(ds.adj)
        init_partition = _compute_launch_initial_partition(
            adj_matrix=first_snapshot,
            dataset_name=dataset_name,
            init_batch_number=init_batch_number,
            cache_dir=cache_dir,
            verbose=verbose,
        )

    # mfc_adopted owns the temporal loop internally and returns one final label
    # vector. Keep this computation outside print_zone so verbosity only affects
    # reporting, never whether the dynamic run itself happens.
    coms = mfc_adopted(
        adj=ds.adj,
        labels=None,
        network_type="MFC",
        return_labels=True,
        num_epoch=baseline_iter,
        pure_mfc=True,
        initial_partition=init_partition,
    )

    measured_time = time.perf_counter() - time_s
    mod = Metrics.modularity(
        _first_snapshot(ds.adj),
        coms,
        directed=ds.is_directed,
    )

    with print_zone(verbose >= 2):
        print(f"Modularity: {mod:.2g}")
        print(f"Baseline calls: {1}")
        print(f"Time: {measured_time:.2f}")

    return [{"modularity": mod, "time": measured_time}]


def _run_dynamic_backend(
    batches_iter,
    dataset_name,
    method,
    init_batch_number,
    verbose,
    cache_dir,
):
    """
    Run LDLeiden/DFLeiden/leidenalg/networkit through the streaming API.

    Parameters
    ----------
    batches_iter : Iterable[torch.Tensor]
        Sequence of adjacency snapshots or update batches.
    dataset_name : str
        Dataset key used for initial-partition cache naming.
    method : str
        Dynamic backend method name accepted by ``create_leiden``.
    init_batch_number : str | None
        Bootstrap marker for p:n strategies. ``None`` runs without an external
        initial partition.
    verbose : int
        Launcher verbosity.
    cache_dir : str | os.PathLike | None
        Optional cache directory for the bootstrap partition.

    Returns
    -------
    list[dict[str, float]]
        Per-processed-batch modularity and runtime entries.
    """
    results = []
    algo = None

    for batch_idx, batch in enumerate(batches_iter):
        with print_zone(verbose >= 2):
            print("  Batch", batch_idx)

        if batch_idx == 0:
            # The first batch constructs the dynamic backend object. For p:n
            # strategies, the first batch is only the initial state; metrics are
            # recorded from subsequent update batches to match historical data.
            init_partition = None
            if init_batch_number is not None:
                init_partition = _compute_launch_initial_partition(
                    adj_matrix=batch,
                    dataset_name=dataset_name,
                    init_batch_number=init_batch_number,
                    cache_dir=cache_dir,
                    verbose=verbose,
                )

            algo = create_leiden(method, batch, partition=init_partition)

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

        with print_zone(verbose >= 2):
            print(f"Modularity: {mod:.2g}")
            print(f"Time: {measured_time:.2f}")

        results.append({"modularity": mod, "time": measured_time})

    return results


def _run_optimizer_modes(
    ds,
    batches_iter,
    dataset_name,
    method,
    baseline_iter,
    mode,
    smart_subcoms_depth,
    smart_neighborhood_step,
    verbose,
    use_gpu,
    aggregation_mode,
    init_batch_number,
    cache_dir,
):
    """
    Run smart, naive, and raw modes through the shared Optimizer pipeline.

    Parameters
    ----------
    ds : Dataset-like object
        Loaded dataset containing ``features`` and ``is_directed`` attributes.
    batches_iter : Iterable[torch.Tensor]
        Sequence of adjacency snapshots or update batches.
    dataset_name : str
        Dataset key used for initial-partition cache naming.
    method : str
        Static community detection method executed through ``Optimizer``.
    baseline_iter : int | None
        Iteration/epoch count forwarded to baseline methods that use it.
    mode : str
        One of ``"smart"``, ``"naive"``, or ``"raw"``.
    smart_subcoms_depth : int
        Number of hierarchy levels used by smart mode.
    smart_neighborhood_step : int
        Neighborhood expansion radius for smart mode.
    verbose : int
        Launcher verbosity.
    use_gpu : bool
        Whether Optimizer should use CUDA when available.
    aggregation_mode : str
        Feature aggregation mode forwarded to Optimizer.
    init_batch_number : str | None
        Bootstrap marker for p:n strategies. ``None`` runs the first batch
        through the selected mode immediately.
    cache_dir : str | os.PathLike | None
        Optional cache directory for the bootstrap partition.

    Returns
    -------
    list[dict[str, float]]
        Per-processed-batch modularity and runtime entries.
    """
    results = []
    opt = None
    features = getattr(ds, "features", None)

    for batch_idx, batch in enumerate(batches_iter):
        with print_zone(verbose >= 2):
            print("  Batch", batch_idx)

        if batch_idx == 0:
            # The first batch initializes Optimizer's persistent state. Later
            # batches are added to opt.adj through update_adj().
            opt = Optimizer(
                batch,
                features,
                subcoms_depth=smart_subcoms_depth if mode == "smart" else 1,
                method=method,
                baseline_iter=baseline_iter,
                verbose=verbose,
                use_gpu=use_gpu,
                aggregation_mode=aggregation_mode,
            )

            if init_batch_number is not None:
                # p:n strategies treat batch zero as the initial graph state.
                # The initial partition is installed and timing starts from the
                # following update batch.
                init_partition = _compute_launch_initial_partition(
                    adj_matrix=batch,
                    dataset_name=dataset_name,
                    init_batch_number=init_batch_number,
                    cache_dir=cache_dir,
                    subcoms_depth=opt.subcoms_depth,
                    device=opt.runtime_device(),
                    verbose=verbose,
                )
                if init_partition.dim() == 1:
                    init_partition = init_partition.unsqueeze(0)
                opt.set_communities(communities=init_partition)
                continue

            # Smart mode needs an initial affected-node set even though there
            # was no previous adjacency to diff against.
            affected_nodes_mask = (
                _active_nodes_mask(batch, opt.nodes_num, opt.runtime_device())
                if mode == "smart"
                else None
            )
        else:
            # For update batches, Optimizer both mutates the accumulated graph
            # and optionally returns the directly affected nodes for smart mode.
            affected_nodes_mask = opt.update_adj(
                batch,
                return_mask=(mode == "smart"),
            )

        measured_time = _run_optimizer_batch(
            opt,
            mode,
            affected_nodes_mask,
            smart_neighborhood_step,
        )
        mod = opt.modularity(directed=ds.is_directed)
        _print_optimizer_batch_result(
            verbose,
            method,
            mode,
            opt,
            mod,
            measured_time,
        )

        results.append({"modularity": mod, "time": measured_time})

    return results


def _print_launch_summary(results, verbose: int) -> None:
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
    with print_zone(verbose == 1):
        final_mod = results[-1]["modularity"] if results else 0
        print(f"Final modularity: {final_mod:.2g}")
    with print_zone(verbose >= 1):
        print(f"Total time: {total_measured_time:.2f}")
        print("-----------------------------------------------")


def dynamic_launch(ds, batches_strategy,
                    underlying_static_method: str,
                    baseline_iter: int = None,
                    mode: str = "smart",
                    smart_subcoms_depth: int = 5,
                    smart_neighborhood_step: int = 1,
                    verbose: int = 1,
                    use_gpu: bool = False,
                    aggregation_mode: str = "sum",
                    cache_dir = None):
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
    # with a loaded dataset object and one validated mode value.
    ds = _load_dataset_if_needed(ds, batches_strategy)
    mode = _normalize_launch_mode(mode)

    dataset_name = ds.name
    init_batch_number = _init_batch_number(batches_strategy)
    batches_iter = _iter_adjacency_batches(ds.adj)

    # Select the execution engine. MFC dynamic mode owns its full temporal loop,
    # other dynamic methods use the streaming backend API, and all remaining
    # modes use Optimizer.
    if mode == "dynamic" and underlying_static_method == "mfc":
        results = _run_dynamic_mfc(
            ds,
            dataset_name,
            init_batch_number,
            baseline_iter,
            verbose,
            cache_dir,
        )
    elif mode == "dynamic":
        results = _run_dynamic_backend(
            batches_iter,
            dataset_name,
            underlying_static_method,
            init_batch_number,
            verbose,
            cache_dir,
        )
    else:
        results = _run_optimizer_modes(
            ds,
            batches_iter,
            dataset_name,
            underlying_static_method,
            baseline_iter,
            mode,
            smart_subcoms_depth,
            smart_neighborhood_step,
            verbose,
            use_gpu,
            aggregation_mode,
            init_batch_number,
            cache_dir,
        )

    _print_launch_summary(results, verbose)
    return results
