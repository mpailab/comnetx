# Baseline Implementation Audit

This audit tracks methods that would be natural empirical comparisons for the
ComNetX paper, but are not all equally suitable for the current measurement
protocol. The criterion used in the article is intentionally narrow: an
empirical baseline should be callable from the same Python-controlled batched
workflow, consume the same edge-update sequence, and return a complete hard
partition for every measured snapshot.

## Summary

- Already included or implemented in the current experiments: Leidenalg, FLMIG,
  DMoN, MAGI, S2CAG, PRGPT-Infomap, PRGPT-Locale, DF-Leiden, and MFC.
- Relevant dynamic full-partition methods that are not currently executable in
  the common pipeline generally fall into one of three groups: non-Python or
  external-tool implementations, no maintained public wrapper found during the
  audit, or a protocol that would require substantial reimplementation.
- Several available Python libraries for temporal community discovery are
  useful related work, but they mostly implement two-stage matching/smoothing,
  temporal communities, or stream/link-community objectives rather than the
  local batched full-partition maintenance protocol used here.

## Candidate Methods

| Candidate family | Representative refs | Protocol match | Implementation status from audit | Current decision |
| --- | --- | --- | --- | --- |
| DynaMo-style incremental modularity | `ZCL19` | High: dynamic full-partition modularity maintenance. | Public sample repository is Java/Maven/Eclipse-oriented and describes sample experiment code rather than a native Python package. `tnetwork` exposes a Python call, but it is an external Java bridge. Direct Java runs were attempted on the selected graph streams; an attempted code-level correction did not make the runs complete the protocol reproducibly. | Cite and discuss, but do not include as an empirical baseline unless a stable bridge is repaired and validated. |
| Adaptive label propagation (ALPA) | `HLS17` | Medium-high: dynamic full-partition label propagation over graph events. | Public implementation is `ALPA.jl`, a Julia package/command-line workflow. No maintained native Python wrapper was found during the audit. | Cite and discuss; exclude from the current baseline tables because integration would require a separate Julia execution path. |
| LabelRankT and related dynamic label-propagation methods | `XCS13`, `SLX14`, `MZW20` | Medium-high: incremental label-propagation partitions. | No maintained Python package matching the paper algorithms was found during the audit. Some secondary/tutorial implementations exist, but not a reproducible research baseline for the current protocol. | Cite as related dynamic label-propagation work; do not include empirically without a validated implementation. |
| C-Blondel | `Seif20` | High: dynamic Louvain-style full partition. | Public repository metadata points to a Java implementation. No maintained Python wrapper was found during the audit. | Cite if needed; omit empirically unless a reliable Java bridge is built. |
| Delta-Screening and DF-Louvain | `ZK21`, `SLu24` | High: dynamic Louvain-style full partition. | Delta-Screening itself did not appear as a standalone maintained Python baseline in the audit. A recent `df-louvain` Python binding exists and includes DF-Louvain plus Delta-Screening variants. | DF-Leiden is already included as the stronger Leiden-family native dynamic baseline. DF-Louvain is an optional supplementary comparison if time allows. |
| Grappolo dynamic CD | `HKT17` | High: scalable static/dynamic modularity-oriented CD. | Public descriptions point to HPC-oriented C/C++ code. No maintained Python wrapper matching the dynamic baseline protocol was found during the audit. | Cite as scalable dynamic CD; omit empirically in the current Python-controlled pipeline. |
| DyComPar / DyG-DPCD and permanence-based dynamic CD | `Sattar23`, `Sattar25` | Medium-high: dynamic full-partition/community-evolution algorithms, but with a different permanence-oriented objective and parallel/distributed emphasis. | The paper reports implemented code and parallel algorithms, but no maintained Python package or wrapper was found during the audit. | Cite and classify; do not include empirically without an executable public interface or author-provided code. |
| DynComm | DynComm R package | Medium-high: dynamic community maintenance for evolving graphs. | Available as an R package with R/Rcpp interfaces, not as a native Python package. | Related software only for the current submission; not used as a Python-controlled baseline. |
| `tnetwork` dynamic community detection | tnetwork DCD | Medium: Python implementations exist, but many are two-stage matching/smoothing or temporal-community methods; DynaMo is external Java. | Pure-Python functions include iterative matching, label smoothing, smoothed Louvain, rolling CPM, and MSSCD; external methods may require Java/Matlab. | Potential rebuttal-only baseline if time permits, but not a like-for-like high-performance local-update comparator. |
| CDlib temporal clustering | CDlib dynamic discovery | Medium-low for speed comparison: primarily dynamic community discovery and two-stage identify/match workflows. | Python library available; dynamic module applies static methods per step or temporal trade-off methods such as TILES. | Useful for related work and possible appendix/rebuttal, but not selected for the main speedup tables. |
| Anchor/seed local dynamic community discovery | `DGB17`, `CBT23`, `LSS21` | Low for our tables: output is local/query/anchor-centered communities rather than a full hard partition of every snapshot. | Implementation availability is secondary because the output protocol is not directly comparable to global NMI/modularity tables. | Cite and contrast; no empirical full-partition comparison. |
| Online overlapping communities | `BCA18` and TILES/OLCPM-style methods | Low for our tables: overlapping communities and often link-stream/online objectives. | OLCPM has public Java code; CDlib includes TILES-like temporal tools. | Cite and contrast; no empirical comparison with non-overlapping full partitions. |
| Continuous-time / link-stream methods | LAGO, flow stability, MSSCD-like methods | Low for our current protocol: continuous-time or link-stream objectives rather than batched snapshot maintenance. | Some Python software exists, but the task and metrics differ. | Related work only unless the paper scope is expanded. |
| Static high-performance graph clustering | NetworKit PLM/PLP, GVE-Leiden | Medium: full partitions, but static full-recomputation baselines rather than dynamic methods. | Python frontends exist for NetworKit; GVE-Leiden availability is implementation-specific. | Optional static speed baseline. Current paper uses Leidenalg as the main high-quality static recomputation baseline and DF-Leiden as the native dynamic baseline. |

## Conservative Article Wording

The article should not claim that no Python wrappers exist for all omitted
methods. The accurate claim is narrower: some methods have no maintained native
Python interface for the common batched full-partition pipeline; some have only
external Java/Matlab/R/Julia execution paths; and some solve a different output
protocol. This is the wording now used in the Baselines subsection.

## Source Pointers Used In The Audit

Detailed external URLs are omitted from this anonymous reproduction bundle.
