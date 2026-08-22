# Reviewer-Issue Register for the IEEE Access Revision

This register retains only conference-review comments that improve the
correctness, evidential support, reproducibility, or positioning of the journal
article. Conference routing, ranking, presentation format, and workshop-specific
comments are intentionally omitted.

Status: incorporated where the archived implementation and measurements permit;
unresolved items are stated as limitations or future measurement requirements.

## Incorporated in the Manuscript

### 1. Describe the implemented hierarchy and update path exactly

Source: Reviewer 1, detailed comment 1.

The method section now follows the code rather than an intended invariant. It
defines the Leiden-built initial hierarchy, frozen pre-update masks, provisional
labels, descending label propagation, ascending backend calls, one-time
restriction to the finest affected set, feature averaging, and absence of a
post-update nesting repair. The article does not claim that provisional labels
are globally unique or that updated levels remain nested.

### 2. Narrow backend-independence and quality claims

Sources: Reviewers 1 and 3; Reviewer 2, weak point 2.

ComNetX is presented as a backend-independent interface for a local hierarchical
approximation, not as a semantics-preserving wrapper. The text explains that
boundary removal, contraction, and feature averaging can alter the backend
input and objective. Quality claims are restricted to the measured backends,
datasets, and operating regimes. Short- and long-horizon results are reported
separately.

### 3. Report temporal evidence without overstating robustness

Sources: Reviewer 1, detailed comments 4 and 5; Reviewers 2 and 3.

The revision includes the archived 500-update PubMed and arxivmath series and
the five-seed DSBM stress test. It distinguishes fixed-stream runtime
repeatability from generalization. Because the archive contains only one real
window per graph and no long-horizon per-batch NMI or agreement series, the
article does not claim robustness across independent temporal windows.

### 4. Treat fallback as an unimplemented policy

Sources: all three reviewers.

No adaptive Local/Full controller exists in the current code or measurement
archive. The paper therefore does not present one. It uses the DSBM break-even
and workload profiles only to identify possible future policy inputs and states
that thresholds, held-out windows, and a drift or periodic-refresh safeguard
would be required.

### 5. Make feature-aware comparisons defensible

Sources: Reviewer 1, detailed comment 3; Reviewer 3.

DMoN, MAGI, MFC, FLMIG, and PRGPT are excluded from the journal evidence because
the earlier aggregate mixes graph directions, feature modes, update counts, or
invalid configurations. Missing executions are not classified as outcomes.
S$^2$CAG is retained only as a fixed-budget interface/runtime case. Each
invocation performs five internal runs with $T=10$; the full call uses previous
labels only to select the requested cluster count, whereas label-free local
calls request as many clusters as contracted vertices. These cluster-count,
bootstrap, and graph-size confounds are disclosed, so the rows are not treated
as a tuned quality comparison.

### 6. Use the topology sweep as exploratory evidence only

Source: Reviewer 3.

The available radius/depth grid has mostly one run per configuration and no
held-out selection window. The revision reports it as an exploratory ablation
and does not claim that the fixed $L=3$, $r=1$ setting is universally optimal.

The journal revision also retains the measured resolution sweep, the
direction-preserving Leiden control, the S$^2$CAG feature-mode ablation, and
the cut-based partition summaries. These controls are included because they
test the scope of the central empirical claim; they are explicitly identified
as single-run or fixed-budget evidence where appropriate.

### 7. Strengthen novelty and related-work positioning

Source: Reviewer 2, weak points 1 and 4.

The related-work section now separates dynamic modularity methods,
representation-learning methods, local-update methods, and hierarchical graph
coarsening. It includes and distinguishes LD-Leiden (the work identified as
arXiv:2502.18497) by scope, backend coupling, and empirical protocol. The
article contains no anonymous-conference artifact language.

## Explicitly Unresolved Without New Measurements

The following additions would materially strengthen a later revision, but the
current archive cannot support them and no result is fabricated:

- non-overlapping or independently sampled real temporal windows;
- long-horizon per-batch NMI or Local-versus-Full partition agreement;
- deletion, vertex-arrival, evolving-feature, and directed-stream tests;
- a predeclared adaptive Local/Full policy selected on pilot data and evaluated
  on held-out windows or seeds;
- repeated radius/depth sweeps with a held-out selection protocol;
- tuned multi-seed feature-aware baselines with valid cluster-count and budget
  selection;
- portable timing experiments with recorded hardware, software versions,
  timestamps, and repeat identifiers.

## Evidence Used

- `results/icdm-2026-1/measurements/experiment_measurements.json`: short,
  repeatability, topology, long-horizon, and DSBM records;
- `results/icdm-2026-1/measurements/workload_profiles.json`: affected, closed,
  and contracted workloads plus closure/contraction ablations;
- `results/icdm-2026-1/measurements/neighborhood_measurements.json`:
  neighborhood-growth profiles;
- `results/icdm-2026-1/measurements/cut_metrics.json`: paired conductance and
  normalized-cut controls;
- `journal/ieee-access/analysis/validated_results.json`: asserted selection and
  unrounded values used by the manuscript generator.
