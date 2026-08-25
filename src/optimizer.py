# External imports
import math
import time
import torch
from typing import Optional, Callable


# Internal imports
import sparse
from our_utils import print_zone


def restricted_boundary_objective_certificate(
    adjacency: torch.Tensor,
    scope_mask: torch.Tensor,
    assignments: torch.Tensor,
    gamma: float = 1.0,
) -> dict[str, float | bool | None]:
    """Evaluate the restricted modularity boundary certificate.

    The calculation follows the directed-matrix convention used by
    :func:`metrics.Metrics.modularity`: every stored matrix entry contributes
    once to the total weight.  Consequently, a symmetric representation of an
    undirected edge contributes once in each direction, while a diagonal
    self-loop contributes once.  ``assignments`` is the projected candidate on
    the full vertex set; only its restriction to ``scope_mask`` affects the
    realized mismatch ``D``.

    All reductions use ``float64`` even when the resident graph uses
    ``float32``.  The certificate requires nonnegative finite edge weights,
    finite nonnegative ``gamma``, positive full weight ``W``, and positive
    induced-scope weight ``W_U``.  Degenerate zero-weight scopes retain their
    measured masses but return ``None`` for the undefined derived quantities
    and set ``finite`` to ``False``.
    """

    if adjacency.dim() != 2 or adjacency.size(0) != adjacency.size(1):
        raise ValueError("adjacency must be a square two-dimensional tensor")
    nodes = adjacency.size(0)
    if scope_mask.dim() != 1 or scope_mask.numel() != nodes:
        raise ValueError("scope_mask must contain one entry per graph vertex")
    if assignments.dim() != 1 or assignments.numel() != nodes:
        raise ValueError("assignments must contain one label per graph vertex")
    gamma = float(gamma)
    if not math.isfinite(gamma) or gamma < 0.0:
        raise ValueError("gamma must be finite and nonnegative")

    if adjacency.layout == torch.strided:
        row, col = torch.nonzero(adjacency, as_tuple=True)
        weights = adjacency[row, col]
    else:
        coalesced = adjacency.to_sparse_coo().coalesce()
        row, col = coalesced.indices()
        weights = coalesced.values()

    device = adjacency.device
    row = row.to(device=device)
    col = col.to(device=device)
    weights = weights.to(device=device, dtype=torch.float64)
    scope = scope_mask.to(device=device, dtype=torch.bool)
    labels = assignments.to(device=device, dtype=torch.long)

    if weights.numel() and not bool(torch.isfinite(weights).all().item()):
        raise ValueError("certificate requires finite adjacency weights")
    if weights.numel() and bool((weights < 0).any().item()):
        raise ValueError("certificate requires nonnegative adjacency weights")

    source_inside = scope.index_select(0, row)
    target_inside = scope.index_select(0, col)
    internal = source_inside & target_inside
    outgoing_boundary = source_inside & ~target_inside
    incoming_boundary = ~source_inside & target_inside

    total_weight_t = weights.sum()
    scope_weight_t = weights[internal].sum()
    beta_out_t = weights[outgoing_boundary].sum()
    beta_in_t = weights[incoming_boundary].sum()

    total_weight = float(total_weight_t.item())
    scope_weight = float(scope_weight_t.item())
    beta_out = float(beta_out_t.item())
    beta_in = float(beta_in_t.item())
    result: dict[str, float | bool | None] = {
        "gamma": gamma,
        "W": total_weight,
        "W_U": scope_weight,
        "beta_out": beta_out,
        "beta_in": beta_in,
        "B_plus": None,
        "B_minus": None,
        "certificate_width": None,
        "D": None,
        "Q_U": None,
        "finite": False,
    }
    if total_weight <= 0.0 or scope_weight <= 0.0:
        return result

    scope_vertices = torch.nonzero(scope, as_tuple=True)[0]
    if scope_vertices.numel() == 0:
        return result
    outside_vertices = torch.nonzero(~scope, as_tuple=True)[0]
    inside_labels = labels.index_select(0, scope_vertices)
    if outside_vertices.numel() and bool(
        torch.isin(
            torch.unique(inside_labels),
            torch.unique(labels.index_select(0, outside_vertices)),
        ).any().item()
    ):
        raise ValueError(
            "certificate requires scope and outside community labels to be disjoint"
        )
    _, inverse = torch.unique(
        inside_labels,
        sorted=True,
        return_inverse=True,
    )
    group_by_vertex = torch.full(
        (nodes,), -1, dtype=torch.long, device=device
    )
    group_by_vertex[scope_vertices] = inverse
    groups = int(inverse.max().item()) + 1

    x = torch.zeros(groups, dtype=torch.float64, device=device)
    y = torch.zeros_like(x)
    p = torch.zeros_like(x)
    q = torch.zeros_like(x)
    if bool(internal.any().item()):
        x.scatter_add_(0, group_by_vertex[row[internal]], weights[internal])
        y.scatter_add_(0, group_by_vertex[col[internal]], weights[internal])
    if bool(outgoing_boundary.any().item()):
        p.scatter_add_(
            0,
            group_by_vertex[row[outgoing_boundary]],
            weights[outgoing_boundary],
        )
    if bool(incoming_boundary.any().item()):
        q.scatter_add_(
            0,
            group_by_vertex[col[incoming_boundary]],
            weights[incoming_boundary],
        )

    scale = gamma / (total_weight_t * total_weight_t)
    normalization_term = (
        (total_weight_t - scope_weight_t)
        / scope_weight_t
        * torch.sum(x * y)
    )
    boundary_term = torch.sum(x * q + y * p + p * q)
    mismatch = scale * (normalization_term - boundary_term)
    b_plus = scale * scope_weight_t * (total_weight_t - scope_weight_t)
    b_minus = scale * (
        scope_weight_t * (beta_out_t + beta_in_t) + beta_out_t * beta_in_t
    )
    width = b_plus + b_minus
    internal_source_groups = group_by_vertex[row[internal]]
    internal_target_groups = group_by_vertex[col[internal]]
    same_inside_block = internal_source_groups == internal_target_groups
    internal_block_weight = weights[internal][same_inside_block].sum()
    local_modularity = (
        internal_block_weight
        - gamma * torch.sum(x * y) / scope_weight_t
    ) / scope_weight_t

    derived = {
        "B_plus": float(b_plus.item()),
        "B_minus": float(b_minus.item()),
        "certificate_width": float(width.item()),
        "D": float(mismatch.item()),
        "Q_U": float(local_modularity.item()),
    }
    result.update(derived)
    result["finite"] = all(math.isfinite(value) for value in derived.values())
    return result


def restricted_ranking_certificate(
    adjacency: torch.Tensor,
    scope_mask: torch.Tensor,
    candidate_assignments: torch.Tensor,
    reference_assignments: torch.Tensor,
    gamma: float = 1.0,
    *,
    candidate_certificate: Optional[dict[str, float | bool | None]] = None,
    reference: str = "identity atom partition",
) -> dict[str, float | int | bool | str | None]:
    """Compare a scoped candidate with a reference using the ranking bound.

    ``candidate_assignments`` and ``reference_assignments`` must represent
    partitions on the same scope and keep their inside labels disjoint from
    the fixed outside partition. The caller is responsible for constructing
    the declared reference from the same quotient atoms. A strict ranking is
    considered numerically informative only when its margin above the
    certificate width exceeds the recorded tolerance; local or full ties are
    represented by sign zero.
    """

    candidate = candidate_certificate or restricted_boundary_objective_certificate(
        adjacency,
        scope_mask,
        candidate_assignments,
        gamma=gamma,
    )
    identity = restricted_boundary_objective_certificate(
        adjacency,
        scope_mask,
        reference_assignments,
        gamma=gamma,
    )
    base: dict[str, float | int | bool | str | None] = {
        "reference": reference,
        "candidate_family": "same scope and quotient atoms",
        "scaled_local_gap": None,
        "restricted_full_objective_gap": None,
        "certificate_width": candidate.get("certificate_width"),
        "output_D": candidate.get("D"),
        "reference_D": identity.get("D"),
        "output_Q_U": candidate.get("Q_U"),
        "reference_Q_U": identity.get("Q_U"),
        "W_U_over_W": None,
        "comparison_tolerance": None,
        "local_gap_sign": 0,
        "full_gap_sign": 0,
        "strict_threshold_passed": False,
        "ranking_sign_certified": False,
        "status": "undefined_zero_weight_scope",
        "finite": False,
    }
    if candidate.get("finite") is not True or identity.get("finite") is not True:
        return base

    for field in ("gamma", "W", "W_U", "certificate_width"):
        candidate_value = float(candidate[field])
        identity_value = float(identity[field])
        tolerance = 1e-10 * max(1.0, abs(candidate_value), abs(identity_value))
        if not math.isclose(
            candidate_value,
            identity_value,
            rel_tol=1e-12,
            abs_tol=tolerance,
        ):
            raise ValueError(
                f"candidate and reference certificates disagree on {field}"
            )

    w = float(candidate["W"])
    w_u = float(candidate["W_U"])
    width = float(candidate["certificate_width"])
    output_d = float(candidate["D"])
    reference_d = float(identity["D"])
    scaled_local_gap = w_u / w * (
        float(candidate["Q_U"]) - float(identity["Q_U"])
    )
    full_gap = scaled_local_gap + output_d - reference_d
    comparison_tolerance = 1e-10 * max(
        1.0,
        abs(scaled_local_gap),
        abs(full_gap),
        abs(width),
    )

    def signed(value: float) -> int:
        if abs(value) <= comparison_tolerance:
            return 0
        return 1 if value > 0.0 else -1

    local_sign = signed(scaled_local_gap)
    full_sign = signed(full_gap)
    strict_threshold_passed = abs(scaled_local_gap) > width
    informative = (
        abs(scaled_local_gap) - width > comparison_tolerance
        and local_sign != 0
    )
    same_nonzero_sign = local_sign != 0 and local_sign == full_sign
    ranking_sign_certified = informative and same_nonzero_sign
    if local_sign == 0:
        status = "local_tie"
    elif not informative:
        status = "within_certificate_width"
    elif ranking_sign_certified:
        status = "certified_same_sign"
    else:
        status = "sign_mismatch"

    derived = {
        "scaled_local_gap": scaled_local_gap,
        "restricted_full_objective_gap": full_gap,
        "certificate_width": width,
        "output_D": output_d,
        "reference_D": reference_d,
        "output_Q_U": float(candidate["Q_U"]),
        "reference_Q_U": float(identity["Q_U"]),
        "W_U_over_W": w_u / w,
        "comparison_tolerance": comparison_tolerance,
    }
    base.update(derived)
    base.update(
        {
            "local_gap_sign": local_sign,
            "full_gap_sign": full_sign,
            "strict_threshold_passed": strict_threshold_passed,
            "ranking_sign_certified": ranking_sign_certified,
            "status": status,
            "finite": all(math.isfinite(value) for value in derived.values()),
        }
    )
    return base


class Optimizer:

    def __init__(self,
                 adj_matrix: torch.Tensor,
                 features: Optional[torch.Tensor] = None,
                 communities: Optional[torch.Tensor] = None,
                 subcoms_depth: int = 1,
                 method: str = "leidenalg",
                 baseline_iter: int = None,
                 verbose: int = 0,
                 use_gpu: bool = False,
                 aggregation_mode: str = "normalized",
                 resolution: float = 1.0):
        """


        Parameters
        ----------
        communities : torch.Tensor of the shape (l,n)
            Each elements communities[d,i] defines a community at the level d
            that the node i belongs to, l is the number of community levels and
            n is the number of nodes.

        aggregation_mode : str
            Feature aggregation mode. Supported values: "sum", "normalized".
            The aggregated adjacency matrix is always computed by summing edge
            weights between communities. This parameter only controls feature
            aggregation: "sum" keeps per-community feature sums, while
            "normalized" divides them by community size.
        """
       
        self.size = adj_matrix.size()
        self.nodes_num = adj_matrix.size()[0]            
        self.subcoms_depth = subcoms_depth

        # If GPU mode is requested and CUDA is available, keep all
        # optimizer state on CUDA.
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = adj_matrix.device

        self.adj = adj_matrix.float().to(self.device)
        if self.adj.is_sparse:
            # The linear restriction path relies on lexicographically ordered,
            # duplicate-free COO indices.  Keep this as a resident invariant
            # instead of paying an implicit coalesce at every hierarchy level.
            self.adj = self.adj.coalesce()

        if features is None:
            self.features = torch.zeros((self.nodes_num, 1), dtype=self.adj.dtype, device=self.device)
            self.synthetic_features = True
            self.has_real_features = False
        else:
            self.features = features.float().to(self.device)
            self.synthetic_features = self._is_sparse_identity(self.features)
            self.has_real_features = not self.synthetic_features

        self.set_communities(communities)
        self.method = method
        self.baseline_iter = baseline_iter
        self.resolution = float(resolution)
        self.aggregation_mode = self._normalize_aggregation_mode(
            aggregation_mode
        )

        self.verbose = verbose
        self.conversion_time = 0.0
        self.last_timing_info = None
        self.last_run_profile = None
        self.local_algorithm_calls = 0

    @staticmethod
    def _normalize_aggregation_mode(mode: str) -> str:
        mode = mode.lower().strip()
        aliases = {
            "sum": "sum",
            "norm": "normalized",
            "normalize": "normalized",
            "normalized": "normalized",
        }
        if mode not in aliases:
            supported = ", ".join(["sum", "normalized"])
            raise ValueError(
                f"Unsupported aggregation_mode: {mode}. "
                f"Expected one of: {supported}."
            )
        return aliases[mode]

    @staticmethod
    def _is_sparse_identity(features: torch.Tensor) -> bool:
        if not isinstance(features, torch.Tensor):
            return False
        if features.layout != torch.sparse_coo:
            return False

        feat = features.coalesce()
        if feat.dim() != 2:
            return False

        n, m = feat.shape
        if n != m or feat._nnz() != n:
            return False

        row, col = feat.indices()
        vals = feat.values()

        if not torch.equal(row, col):
            return False
        if not torch.all(vals == 1):
            return False
        if not torch.equal(torch.sort(row).values, torch.arange(n, device=row.device)):
            return False

        return True

    def _aggregate_features(
        self,
        pattern: torch.Tensor,
        features: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if features is None:
            return None

        if isinstance(features, torch.Tensor) and features.layout == torch.sparse_coo:
            features = features.coalesce()

            if self._is_sparse_identity(features):
                return pattern.coalesce()

            return torch.sparse.mm(pattern, features)

        return torch.sparse.mm(pattern, features)

    def _local_algorithm_requires_features(self) -> bool:
        if self.method in {"magi", "dmon"}:
            return True
        if self.method in {"s2cag", "mfc"}:
            return self.has_real_features
        return False

    def runtime_device(self) -> torch.device:
        return self.device

    def runtime_adj(self) -> torch.Tensor:
        return self.adj

    def runtime_features(self) -> torch.Tensor:
        return self.features

    @staticmethod
    def canonicalize_partition(
        labels: torch.Tensor,
        original_vertices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Label every block by its minimum original vertex.

        Canonical labels are deterministic, bounded by ``[0, n - 1]``, and
        collision-free for disjoint blocks. ``original_vertices`` identifies
        the graph vertices represented by a scoped label vector; when omitted,
        the vector is assumed to cover all vertices in their natural order.
        """
        if labels.dim() != 1:
            raise ValueError("partition labels must be a one-dimensional tensor")
        labels = labels.to(dtype=torch.long)
        if labels.numel() == 0:
            return labels.clone()
        if original_vertices is None:
            original_vertices = torch.arange(
                labels.numel(), dtype=torch.long, device=labels.device
            )
        else:
            original_vertices = original_vertices.to(
                device=labels.device, dtype=torch.long
            )
        if original_vertices.shape != labels.shape:
            raise ValueError(
                "original_vertices must have the same shape as partition labels"
            )

        _, inverse = torch.unique(labels, sorted=True, return_inverse=True)
        block_count = int(inverse.max().item()) + 1
        sentinel = int(original_vertices.max().item()) + 1
        representatives = torch.full(
            (block_count,),
            sentinel,
            dtype=torch.long,
            device=labels.device,
        )
        representatives.scatter_reduce_(
            0,
            inverse,
            original_vertices,
            reduce="amin",
            include_self=True,
        )
        return representatives[inverse]

    @classmethod
    def canonicalize_hierarchy(cls, communities: torch.Tensor) -> torch.Tensor:
        """Canonicalize every row of a ``[levels, vertices]`` hierarchy."""
        if communities.dim() != 2:
            raise ValueError("communities must have shape [levels, vertices]")
        return torch.stack(
            [cls.canonicalize_partition(row) for row in communities], dim=0
        )

    @staticmethod
    def hierarchy_is_nested(communities: torch.Tensor) -> bool:
        """Return whether every stored row refines the following row.

        Rows are expected in implementation order (reported fine partition
        first) and to use canonical minimum-vertex labels.  Under that
        convention, a fine block is contained in one coarse block exactly
        when every vertex and its fine representative have the same coarse
        label.
        """
        if communities.dim() != 2:
            return False
        if communities.size(0) <= 1:
            return True
        nodes = communities.size(1)
        if communities.numel() and (
            int(communities.min().item()) < 0
            or int(communities.max().item()) >= nodes
        ):
            return False
        for level in range(communities.size(0) - 1):
            fine = communities[level]
            coarse = communities[level + 1]
            if not torch.equal(coarse, coarse.index_select(0, fine)):
                return False
        return True
    
    def set_communities(
        self,
        communities: Optional[torch.Tensor],
        replace_subcoms_depth: bool = False,
    ):
        n = self.nodes_num
        l = self.subcoms_depth
        if communities is None:
            assigned = (
                torch.arange(0, n, dtype=torch.long, device=self.device)
                .repeat(l)
                .reshape((l, n))
            )
        else:
            communities = communities.to(self.device, dtype=torch.long)
            if communities.dim() == 1:
                print(f"Warning: 1D communities converted to 2D with depth 1")
                communities = communities.unsqueeze(0)
            if communities.size(1) != n:
                print(
                    f"Warning: bad communities shape {communities.shape}, "
                    f"required ({l}, {n})"
                )
                print(f"Use default communities with shape ({l}, {n})")
                assigned = (
                    torch.arange(0, n, dtype=torch.long, device=self.device)
                    .repeat(l)
                    .reshape((l, n))
                )
            else:
                current_depth = communities.size(0)
                if replace_subcoms_depth:
                    self.subcoms_depth = current_depth
                    l = current_depth
                if current_depth == l:
                    assigned = communities
                elif current_depth > l:
                    print(
                        f"Warning: communities depth {current_depth} > "
                        f"subcoms_depth {l}."
                    )
                    print(f"Truncating to {l} levels.")
                    assigned = communities[:l, :]
                else:
                    print(
                        f"Warning: communities depth {current_depth} < "
                        f"subcoms_depth {l}."
                    )
                    print(f"Extending with zeros to {l} levels.")
                    zeros_to_add = torch.zeros(
                        (l - current_depth, n),
                        dtype=communities.dtype,
                        device=self.device,
                    )
                    assigned = torch.cat([communities, zeros_to_add], dim=0)

        # Canonical block representatives make all subsequent scope-local
        # writes collision-free without an unbounded global label allocator.
        self.coms = self.canonicalize_hierarchy(assigned)
        if not self.hierarchy_is_nested(self.coms):
            raise ValueError(
                "community rows must form a nested fine-to-coarse hierarchy"
            )
   
    def modularity_slow(self,
            gamma: float = 1, L: int = 0, directed: bool = False) -> float:
        """
        Args:
            gamma: float, optional (default=1)
            L: int, optional (default=0)
        Returns:
            modularity: float
        """
        from metrics import Metrics
        return Metrics.modularity_slow(
            self.adj,
            self.coms[L],
            gamma,
            directed=directed,
        )
   
    def modularity(self,
            gamma: float = 1, L: int = 0, directed: bool = False) -> float:
        """
        Args:
            gamma: float, optional (default=1)
            L: int, optional (default=0)
        Returns:
            modularity: float
        """
        from metrics import Metrics
        return Metrics.modularity(self.adj, self.coms[L], gamma, directed = directed) 
        
    def update_adj(
        self,
        batch: torch.Tensor,
        return_mask: bool = True,
    ) -> Optional[torch.Tensor]:
        """
        Change the graph based on the current batch of updates.


        Args:
            batch : torch.Tensor of the shape (n, n)
        Returns:


        """

        if self.size != batch.size():
            raise ValueError(
                f"Unsuitable batch size: {batch.size()}. "
                f"{self.size} is required."
            )

        batch_dev = batch if batch.device == self.device else batch.to(self.device)
        if batch_dev.is_sparse and not batch_dev.is_coalesced():
            batch_dev = batch_dev.coalesce()

        updated = self.adj + batch_dev.type(self.adj.dtype)
        self.adj = updated.coalesce() if updated.is_sparse else updated

        if not return_mask:
            return None

        if batch_dev.is_sparse:
            affected_nodes = batch_dev.indices().unique()
        else:
            # For dense updates, track both row and column endpoints as affected nodes.
            nz_rows, nz_cols = torch.nonzero(batch_dev, as_tuple=True)
            affected_nodes = torch.cat((nz_rows, nz_cols), dim=0).unique()
        affected_nodes_mask = torch.zeros(
            self.nodes_num,
            dtype=torch.bool,
            device=self.device,
        )
        affected_nodes_mask[affected_nodes] = True

        return affected_nodes_mask

    @staticmethod
    def neighborhood(adj: torch.Tensor,
                    nodes_mask: torch.Tensor,
                    step: int = 1,
                    is_symmetric=False) -> torch.Tensor:

        visited = nodes_mask.clone()
        if (step <= 0) or visited.all() or not visited.any():
            return visited

        A = adj.coalesce()
        if not is_symmetric:
            idx = A.indices()
            AT = torch.sparse_coo_tensor(idx.flip(0), A.values(), A.shape).coalesce()
            A_sym = (A + AT).coalesce()
        else:
            A_sym = A

        frontier = visited.clone()
        for _ in range(step):
            if not frontier.any() or visited.all():
                break
            y = torch.sparse.mm(
                A_sym,
                frontier.to(dtype=A_sym.dtype).unsqueeze(1),
            ).squeeze(1)
            new_nodes = y > 0
            new_frontier = new_nodes & (~visited)
            visited = visited | new_nodes
            frontier = new_frontier

        return visited

    def local_algorithm(self,
                        adj: torch.Tensor,
                        features: Optional[torch.Tensor],
                        limited: bool = False,
                        labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        timing_info = {'conversion_time' : 0.0}
        self.local_algorithm_calls += 1

        with print_zone(self.verbose >= 3):
            if self.method == "magi":
                from baselines.magi_model import magi
                res = magi(
                    adj,
                    features,
                    labels,
                    n_epochs=self.baseline_iter,
                    timing_info=timing_info,
                )
            elif self.method in ("prgpt:infomap", "prgpt:locale"):
                from baselines.rough_PRGPT import rough_prgpt
                refine = self.method.split(":")[1]
                res = rough_prgpt(adj, refine=refine, timing_info=timing_info)
            elif self.method == "leidenalg":
                from baselines.leiden import leidenalg_partition
                res = leidenalg_partition(
                    adj,
                    init_partition=labels,
                    timing_info=timing_info,
                    resolution=self.resolution,
                )
            elif self.method in ("ldleiden", "dfleiden"):
                from baselines.dgc import _run_leiden
                res = _run_leiden(self.method, adj, init_partition = labels, timing_info = timing_info)
            elif self.method == "dmon":
                from baselines.dmon import adapted_dmon
                res = adapted_dmon(
                    adj,
                    features,
                    labels,
                    epochs=self.baseline_iter,
                    timing_info=timing_info,
                )
            elif self.method == "mfc":
                from baselines.mfc import mfc_adopted
                if not self.has_real_features:
                    features = None
                res = mfc_adopted(
                    adj=adj,
                    features=features,
                    network_type="MFC",
                    timing_info=timing_info,
                    num_epoch=self.baseline_iter,
                    initial_partition = labels
                )

            elif self.method == "flmig":
                from baselines.flmig import flmig_adopted
                res = flmig_adopted(
                    adj=adj,
                    Number_iter=self.baseline_iter,
                    return_labels=True,
                    timing_info=timing_info,
                    initial_labels=labels
                )
            elif self.method == "s2cag":
                from baselines.s2cag import s2cag
                if not self.has_real_features:
                    features = None
                res = s2cag(
                    adj,
                    features,
                    labels,
                    T=self.baseline_iter,
                    timing_info=timing_info,
                )
            else:
                raise ValueError("Unsupported baseline method name")
        self.conversion_time += timing_info.get('conversion_time', 0.0)
        self.last_timing_info = timing_info
        return res

    @staticmethod
    def aggregate(adj: torch.Tensor, pattern: torch.Tensor) -> torch.Tensor:
        return torch.sparse.mm(pattern, torch.sparse.mm(adj, pattern.t()))

    @staticmethod
    def cut_by_partition(
        adj: torch.Tensor,
        node_mask: torch.Tensor,
        node_labels: torch.Tensor,
        inplace: bool = True,
    ) -> torch.Tensor:
        """
        Cut edges that violate partition constraints.

        If inplace=True (default), zeroes disallowed entries in-place.
        If inplace=False, rebuilds sparse tensor keeping only allowed edges.
        """
        indices = adj.indices()
        row, col = indices

        keep = node_mask[row] & node_mask[col]
        keep = keep & (node_labels[row] == node_labels[col])

        if inplace:
            adj.values().masked_fill_(~keep, 0)
            return adj

        return torch.sparse_coo_tensor(
            indices[:, keep],
            adj.values()[keep],
            adj.size(),
            device=adj.device,
        ).coalesce()

    @staticmethod
    def _restrict_adjacency(
        adj: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Keep edges whose two endpoints are selected by ``node_mask``.

        The input COO tensor is already coalesced, and filtering preserves its
        lexicographic index order.  Marking the result coalesced therefore
        avoids both ``torch.isin`` and an unnecessary sort, giving one linear
        pass over the stored edges.
        """
        adj = adj.coalesce()
        if node_mask.dim() != 1 or node_mask.numel() != adj.size(0):
            raise ValueError("node_mask must contain one entry per graph vertex")
        if node_mask.device != adj.device:
            node_mask = node_mask.to(adj.device)
        node_mask = node_mask.to(dtype=torch.bool)
        indices = adj.indices()
        keep = node_mask[indices[0]] & node_mask[indices[1]]
        return torch.sparse_coo_tensor(
            indices[:, keep],
            adj.values()[keep],
            adj.size(),
            dtype=adj.dtype,
            device=adj.device,
            is_coalesced=True,
        )

    def _community_closure(
        self,
        labels: torch.Tensor,
        touched_vertices: torch.Tensor,
    ) -> torch.Tensor:
        """Return the union of label blocks meeting ``touched_vertices``.

        Canonical labels are original vertex IDs, so a boolean label lookup
        computes closure in ``O(n + |B|)`` time without ``torch.isin``.
        """
        if labels.numel() != self.nodes_num:
            raise ValueError("community labels must cover every graph vertex")
        if labels.numel() and (
            int(labels.min().item()) < 0
            or int(labels.max().item()) >= self.nodes_num
        ):
            raise ValueError(
                "community labels must be canonical vertex IDs in [0, n - 1]"
            )
        touched_labels = labels.index_select(0, touched_vertices)
        label_mask = torch.zeros(
            self.nodes_num, dtype=torch.bool, device=labels.device
        )
        label_mask[touched_labels] = True
        return label_mask[labels]

    @staticmethod
    def _order_groups_by_minimum_vertex(
        temporary_groups: torch.Tensor,
        original_vertices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compact local group IDs in minimum-original-vertex order."""
        if temporary_groups.numel() == 0:
            empty = temporary_groups.to(dtype=torch.long)
            return empty, empty
        group_count = int(temporary_groups.max().item()) + 1
        sentinel = int(original_vertices.max().item()) + 1
        representatives = torch.full(
            (group_count,),
            sentinel,
            dtype=torch.long,
            device=temporary_groups.device,
        )
        representatives.scatter_reduce_(
            0,
            temporary_groups,
            original_vertices,
            reduce="amin",
            include_self=True,
        )
        order = torch.argsort(representatives)
        compact_by_group = torch.empty_like(order)
        compact_by_group[order] = torch.arange(
            group_count, dtype=torch.long, device=order.device
        )
        inverse = compact_by_group[temporary_groups]
        counts = torch.bincount(inverse, minlength=group_count)
        return inverse, counts

    def _level_zero_grouping(
        self,
        ext_nodes: torch.Tensor,
        affected_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build level-zero atoms without writing provisional global labels.

        Radius-affected vertices are singleton atoms. Remaining vertices are
        grouped by their pre-update coarsest stored block, matching the former
        propagation semantics while avoiding temporary namespace collisions.
        """
        affected_in_scope = affected_mask.index_select(0, ext_nodes)
        residual_in_scope = ~affected_in_scope
        temporary = torch.empty(
            ext_nodes.numel(), dtype=torch.long, device=ext_nodes.device
        )
        residual_group_count = 0
        if residual_in_scope.any():
            residual_labels = self.coms[-1, ext_nodes[residual_in_scope]]
            _, residual_inverse = torch.unique(
                residual_labels, sorted=True, return_inverse=True
            )
            temporary[residual_in_scope] = residual_inverse
            residual_group_count = int(residual_inverse.max().item()) + 1
        affected_count = int(affected_in_scope.sum().item())
        if affected_count:
            temporary[affected_in_scope] = residual_group_count + torch.arange(
                affected_count, dtype=torch.long, device=ext_nodes.device
            )
        return self._order_groups_by_minimum_vertex(temporary, ext_nodes)
       
    def run(
        self,
        nodes_mask: torch.Tensor,
        *,
        closure_enabled: bool = True,
        base_atom_policy: str = "hierarchical",
        collect_profile: bool = False,
    ) -> None:
        """
        Run the local hierarchical update on selected vertices.


        Parameters
        ----------
        nodes_mask : torch.Tensor
            Boolean mask of radius-expanded vertices.
        closure_enabled : bool
            Expand the mask to every pre-update stored block that it touches.
            ``False`` is a research ablation with radius-only scopes.
        base_atom_policy : {"hierarchical", "singleton"}
            Level-zero contraction policy. ``"hierarchical"`` uses affected
            singletons plus coarsest residual groups; ``"singleton"`` keeps
            every scoped original vertex separate. Higher levels always use
            parent quotients of the already updated preceding level.
        collect_profile : bool
            Record phase times and workload sizes in ``last_run_profile``.
        """

        if base_atom_policy not in {"hierarchical", "singleton"}:
            raise ValueError(
                "base_atom_policy must be 'hierarchical' or 'singleton'"
            )

        compute_device = self.device
        needs_features = self._local_algorithm_requires_features()

        def profile_clock() -> float:
            if not collect_profile:
                return 0.0
            if compute_device.type == "cuda":
                torch.cuda.synchronize(compute_device)
            return time.perf_counter()

        def profile_elapsed(start: float) -> float:
            if not collect_profile:
                return 0.0
            if compute_device.type == "cuda":
                torch.cuda.synchronize(compute_device)
            return time.perf_counter() - start

        profile = None
        if collect_profile:
            profile = {
                "closure_enabled": bool(closure_enabled),
                "base_atom_policy": base_atom_policy,
                "closure_time": 0.0,
                "prepare_time": 0.0,
                "reset_time": 0.0,
                "aggregation_time": 0.0,
                "backend_time": 0.0,
                "backend_conversion_time": 0.0,
                "projection_time": 0.0,
                "certificate_time": 0.0,
                # Parent quotients rebuild from A_t[U_l]; no destructive cut
                # is part of the repaired production algorithm.
                "cut_time": 0.0,
                "instrumented_wall_time": 0.0,
                "total_profiled_time": 0.0,
                "levels": [],
            }
        total_start = profile_clock()
        self.last_run_profile = None

        coms_work = self.coms
        adj_base = self.adj
        features_work = self.features if needs_features else None

        nodes_mask_work = nodes_mask.to(
            device=compute_device,
            dtype=torch.bool,
        )

        # Find indices of affected nodes.
        nodes = torch.nonzero(nodes_mask_work, as_tuple=True)[0]
        if nodes.numel() == 0:
            if collect_profile:
                assert profile is not None
                wall_time = profile_elapsed(total_start)
                profile["instrumented_wall_time"] = wall_time
                profile["total_profiled_time"] = wall_time
                self.last_run_profile = profile
            return

        # Frozen pre-update closure at every stored level.
        phase_start = profile_clock()
        ext_mask_work = torch.zeros_like(coms_work, dtype=torch.bool)
        if closure_enabled:
            for l in range(self.subcoms_depth):
                ext_mask_work[l] = self._community_closure(coms_work[l], nodes)
        else:
            ext_mask_work[:] = nodes_mask_work.unsqueeze(0)
        if collect_profile:
            assert profile is not None
            profile["closure_time"] = profile_elapsed(phase_start)

        for l in range(self.subcoms_depth):
            phase_start = profile_clock()
            level_ext_mask = ext_mask_work[l]
            ext_nodes = torch.nonzero(level_ext_mask, as_tuple=True)[0]

            if l == 0:
                if base_atom_policy == "singleton":
                    inverse = torch.arange(
                        ext_nodes.numel(),
                        dtype=torch.long,
                        device=compute_device,
                    )
                    counts = torch.ones_like(inverse)
                else:
                    inverse, counts = self._level_zero_grouping(
                        ext_nodes, nodes_mask_work
                    )
            else:
                # Parent quotient: atoms are exactly the blocks of the already
                # updated preceding finer level on U_l.
                _, inverse, counts = torch.unique(
                    coms_work[l - 1, level_ext_mask],
                    sorted=True,
                    return_counts=True,
                    return_inverse=True,
                )
            n = counts.numel()
            prepare_time = profile_elapsed(phase_start)
            if collect_profile:
                assert profile is not None
                profile["prepare_time"] += prepare_time
            if n == 0:
                if collect_profile:
                    assert profile is not None
                    profile["levels"].append(
                        {
                            "level": l,
                            "closure_vertices": int(ext_nodes.numel()),
                            "contracted_nodes": 0,
                            "contracted_edges": 0,
                            "prepare_time": prepare_time,
                            "reset_time": 0.0,
                            "aggregation_time": 0.0,
                            "backend_time": 0.0,
                            "backend_conversion_time": 0.0,
                            "projection_time": 0.0,
                            "certificate_time": 0.0,
                            "boundary_certificate": None,
                            "ranking_certificate": None,
                            "cut_time": 0.0,
                        }
                    )
                continue

            # Every level receives a fresh restriction A_t[U_l]. No destructive
            # cut is carried from a preceding backend call.
            phase_start = profile_clock()
            level_adj = self._restrict_adjacency(adj_base, level_ext_mask)
            reset_time = profile_elapsed(phase_start)
            if collect_profile:
                assert profile is not None
                profile["reset_time"] += reset_time

            phase_start = profile_clock()
            aggr_idx = torch.stack((inverse, ext_nodes))
            aggr_adj_ptn = sparse.tensor(
                aggr_idx,
                (n, self.nodes_num),
                level_adj.dtype,
            )
            aggr_adj = self.aggregate(level_adj, aggr_adj_ptn)
            del aggr_adj_ptn

            aggr_features = None
            if needs_features:
                ext_features = features_work.index_select(0, ext_nodes)
                aggr_features = torch.zeros(
                    (n, ext_features.size(1)),
                    dtype=ext_features.dtype,
                    device=ext_features.device,
                )
                aggr_features.index_add_(0, inverse, ext_features)
                if self.aggregation_mode == "normalized":
                    aggr_features /= counts.to(dtype=ext_features.dtype).unsqueeze(1)
            aggregation_time = profile_elapsed(phase_start)
            if collect_profile:
                assert profile is not None
                profile["aggregation_time"] += aggregation_time

            # Apply local algorithm for aggregated graph
            phase_start = profile_clock()
            coms = self.local_algorithm(aggr_adj, aggr_features, l > 0).to(
                device=compute_device,
                dtype=torch.long,
            )
            backend_time = profile_elapsed(phase_start)
            backend_conversion_time = 0.0
            if collect_profile:
                assert profile is not None
                backend_conversion_time = float(
                    (self.last_timing_info or {}).get("conversion_time", 0.0)
                )
                profile["backend_time"] += backend_time
                profile["backend_conversion_time"] += backend_conversion_time
            if coms.numel() != n:
                raise ValueError(
                    f"backend returned {coms.numel()} labels for {n} groups"
                )

            # Project to original vertices and use the minimum original vertex
            # as each stored block's deterministic, collision-free label.
            phase_start = profile_clock()
            projected = coms[inverse]
            if closure_enabled:
                # U_l is a union of complete pre-update level-l blocks.  The
                # minimum vertex of every retained outside block is therefore
                # outside U_l, whereas every projected representative is in
                # U_l, so scope-local canonical labels are disjoint and the
                # stored entries outside U_l remain byte-for-byte unchanged.
                coms_work[l, level_ext_mask] = self.canonicalize_partition(
                    projected, ext_nodes
                )
            else:
                # The radius-only research control can cut through an old
                # block.  Reusing its canonical label on the retained outside
                # remainder would merge the two sides accidentally.  Give the
                # inside blocks a temporary namespace disjoint from all stored
                # labels, then canonicalize the complete row.  This preserves
                # the partition induced outside the radius-only scope (though
                # an outside remainder's numeric representative can change),
                # keeps the inside/outside blocks disjoint, and lets higher
                # parent quotients operate on a nested partition.
                _, inside_inverse = torch.unique(
                    projected,
                    sorted=True,
                    return_inverse=True,
                )
                separated = coms_work[l].clone()
                separated[level_ext_mask] = self.nodes_num + inside_inverse
                coms_work[l] = self.canonicalize_partition(separated)
            projection_time = profile_elapsed(phase_start)
            if collect_profile:
                assert profile is not None
                profile["projection_time"] += projection_time

            certificate_time = 0.0
            boundary_certificate = None
            ranking_certificate = None
            if collect_profile:
                phase_start = profile_clock()
                boundary_certificate = restricted_boundary_objective_certificate(
                    adj_base,
                    level_ext_mask,
                    coms_work[l],
                    gamma=self.resolution,
                )
                identity_assignments = coms_work[l].clone()
                if closure_enabled:
                    identity_assignments[level_ext_mask] = (
                        self.canonicalize_partition(inverse, ext_nodes)
                    )
                else:
                    # Preserve the same fixed outside partition while giving
                    # every quotient atom a separate inside block.
                    identity_assignments[level_ext_mask] = (
                        self.nodes_num + inverse
                    )
                    identity_assignments = self.canonicalize_partition(
                        identity_assignments
                    )
                ranking_certificate = restricted_ranking_certificate(
                    adj_base,
                    level_ext_mask,
                    coms_work[l],
                    identity_assignments,
                    gamma=self.resolution,
                    candidate_certificate=boundary_certificate,
                    reference="identity atom partition",
                )
                certificate_time = profile_elapsed(phase_start)
                assert profile is not None
                profile["certificate_time"] += certificate_time

            if collect_profile:
                contracted_edges = (
                    int(aggr_adj.coalesce()._nnz())
                    if aggr_adj.is_sparse
                    else int(torch.count_nonzero(aggr_adj).item())
                )
            else:
                contracted_edges = 0
            if collect_profile:
                assert profile is not None
                profile["levels"].append(
                    {
                        "level": l,
                        "closure_vertices": int(ext_nodes.numel()),
                        "contracted_nodes": int(n),
                        "contracted_edges": contracted_edges,
                        "prepare_time": prepare_time,
                        "reset_time": reset_time,
                        "aggregation_time": aggregation_time,
                        "backend_time": backend_time,
                        "backend_conversion_time": backend_conversion_time,
                        "projection_time": projection_time,
                        "certificate_time": certificate_time,
                        "boundary_certificate": boundary_certificate,
                        "ranking_certificate": ranking_certificate,
                        "cut_time": 0.0,
                    }
                )

        if collect_profile:
            assert profile is not None
            wall_time = profile_elapsed(total_start)
            profile["instrumented_wall_time"] = wall_time
            # The certificate is an article diagnostic, not part of the
            # production update. Keep the observed wall clock, but remove its
            # separately timed pass from the production-path total.
            profile["total_profiled_time"] = max(
                0.0, wall_time - profile["certificate_time"]
            )
            self.last_run_profile = profile
