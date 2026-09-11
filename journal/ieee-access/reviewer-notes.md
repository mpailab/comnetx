# Reviewer-Issue Register for the IEEE Access Revision

This register retains only conference-review comments that improve the
correctness, evidential support, reproducibility, or positioning of the journal
article. Conference routing, ranking, presentation format, and workshop-specific
comments are intentionally omitted.

Status: every comment is filtered through the article-level scientific
argument. A comment is incorporated only when it improves correctness,
explanation, or evidence relative to comparable Q1 work. The hierarchy and
label-namespace audit exposed genuine implementation defects; these have been
repaired, and historical smart-mode measurements cannot be relabeled as
measurements of the changed implementation. The namespace defect can also
affect depth one. See `code-change-audit-20260911.md` for exact counterexamples,
their limitations, and the distinction between a bug fix and a changed
algorithmic policy.

## Incorporated in the Manuscript

### 1. Describe the implemented hierarchy and update path exactly

Source: Reviewer 1, detailed comment 1.

The repaired method constructs the initial hierarchy from exact parent
quotients of the original adjacency, canonicalizes every block by its minimum
original vertex, and rejects a deep cache whose rows are not nested. For each
update it freezes all closures from the pre-update hierarchy, rebuilds every
level from the resident induced adjacency, uses the already updated finer
blocks as parent-quotient atoms, and projects backend blocks through a
collision-free canonical namespace. The accompanying argument now establishes
non-strict fine-to-coarse refinement and scope isolation. Strictly distinct
levels are not claimed because an admissible backend can return the identity
partition.

### 2. Narrow backend-independence and quality claims

Sources: Reviewers 1 and 3; Reviewer 2, weak point 2.

ComNetX is presented as a backend-neutral input transformation, not as a
semantics-preserving wrapper for arbitrary objectives. Quotient contraction is
exact for block flows and modularity on the group-constant candidate family;
boundary localization, indivisible atoms, heuristic optimization, and
feature aggregation remain separate approximation sources. The boundary term
is quantified by an exact mismatch decomposition and a restricted
global-objective loss/ranking certificate. Quality claims remain restricted to
the measured backends, datasets, and regimes.

### 3. Report temporal evidence without overstating robustness

Sources: Reviewer 1, detailed comments 4 and 5; Reviewers 2 and 3.

RQ5 retains the 500-update PubMed and arxivmath trajectories and the five-seed
DSBM stress design. Fixed-stream execution repeats are described only as timing
repeatability; independent DSBM seeds support stream-level uncertainty. The
article does not infer robustness across independent real temporal windows
that were not measured.

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

The five original experimental questions are preserved as a second scientific
pillar rather than compressed into a generic benchmark section: RQ1 tests the
paired quality--time consequence, RQ2 backend generality and the
specialization--reuse gap, RQ3 the radius--closure--quotient mechanism, RQ4
the causal controls, and RQ5 robustness and the operating envelope. This
structure matches accepted Q1 evidence patterns in DynaMo, CL-OND, Fusion3M,
and CoD\AE N; it is not retained merely because it appeared in the conference
submission.

### 7. Strengthen novelty and related-work positioning

Source: Reviewer 2, weak points 1 and 4.

The related-work section now separates dynamic modularity methods,
representation-learning methods, local-update methods, and hierarchical graph
coarsening. It includes and distinguishes LD-Leiden (the work identified as
arXiv:2502.18497) by scope, backend coupling, and empirical protocol. The
article contains no anonymous-conference artifact language.

## Explicitly Unresolved Without New Measurements

The following additions have not been measured:

- non-overlapping or independently sampled real temporal windows;
- long-horizon per-batch NMI or Local-versus-Full partition agreement;
- deletion, vertex-arrival, and evolving-feature tests;
- a predeclared adaptive Local/Full policy selected on pilot data and evaluated
  on held-out windows or seeds;
- repeated radius/depth sweeps with a held-out selection protocol;
- tuned multi-seed feature-aware baselines with valid cluster-count and budget
  selection.

The former 24/72-hour campaign frameworks and runtime certificates were
removed in the code audit. Future runs use the existing launch tools, retained
JSON configurations, invariant checks, and phase/work profiles; source, inputs,
environment and shared initialization still need to be recorded. The completed standalone LD-Leiden diagnostic
is retained as internal evidence but is not admitted to the manuscript: its
implementation is not publicly redistributable, its initialization and timing
interfaces do not match the paired ComNetX protocol, and its internal and
common-evaluator modularity values require further diagnosis. These results
must not be silently pooled with the affected historical smart-mode rows.

## Evidence Used

Historical artifacts below remain observations of the earlier implementation;
they do not measure the changed update procedure. New measurements belong under
`results/ieee-access-2026-1/raw/repaired-comnetx/`. Candidate LD-Leiden records
remain separate and do not enter the manuscript under the present evidence
policy. New ComNetX results enter only after checking source/input identity,
paired initialization and timing, structural correctness, and result values.

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
