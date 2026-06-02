# Reviewer Remarks Registry

This registry tracks reviewer remarks, planned corrections, and current closure
status for the ComNetX paper. Percentages are implementation-oriented: they
reflect how much of the required article/result update is already present in
the repository, not whether a future reviewer will necessarily agree.

## Review R1: Preliminary Desk-Rejection Assessment

- Received from: user-supplied preliminary reviewer assessment.
- Registered on: 2026-06-02.
- Overall status: 83% addressed.
- Overall risk: moderate. The main textual concerns are mostly covered, but the
  directed-graph limitation and empirical comparison with local dynamic
  frameworks still carry residual risk.
- Bibliography audit note: the reviewer wording appears to mix author names for
  two local/overlapping-community references. The 2017 paper titled "Local
  Community Detection in Dynamic Networks" is by DiTursi, Ghosh, and Bogdanov,
  not Han et al.; OLCPM is by Boudebza, Cazabet, Azouaou, and Nouali, while
  Palla et al. are relevant predecessors for dynamic overlapping-community
  analysis rather than OLCPM authors.

### Non-Actionable Checks

| ID | Reviewer Point | Status | Registry Note |
| --- | --- | --- | --- |
| R1-N1 | Paper length: pass. | 100% addressed | No action required. |
| R1-N2 | Topic compatibility: pass. | 100% addressed | No action required. |
| R1-N3 | Minimum quality: pass. | 100% addressed | No action required. |
| R1-N4 | Prompt-injection/hidden manipulation: pass. | 100% addressed | No action required. |
| R1-N5 | Relevance, presentation, ethics, GenAI content: generally positive. | 100% addressed | No action required unless later reviews conflict. |

### Actionable Weaknesses

| ID | Reviewer Concern | Current Status | Corrections Already Made | Evidence | Remaining Work |
| --- | --- | --- | --- | --- | --- |
| R1-W1 | Utility of features in GNN backends is unclear; random features appear to replace semantic attributes. | 85% addressed | Added a more cautious interpretation of S2CAG feature modes, explicitly stating that real features can help and that random features are not a replacement for attributes. Added dataset-feature repeated rows for S2CAG. | `article/article.tex`: S2CAG dataset-feature stability discussion; S2CAG feature-mode ablation text; `results/icdm-2026-0`. | Revisit after the single-container rerun finishes; replace unstable timing values if needed. |
| R1-W2 | S2CAG speedup is smaller than contracted workload reduction; overhead source is unclear. | 85% addressed | Added an explicit runtime interpretation: ComNetX overhead is small, while S2CAG is dominated by neural backend cost, feature materialization, sparse tensor construction, and GPU training. The text also explains why A100 80GB GPUs can mask graph-size reduction. | `article/article.tex`: locality/workload reduction discussion around S2CAG overhead; workload profile results in `results/icdm-2026-0`. | If larger-graph S2CAG follow-up results arrive, update the evidence or state limits more explicitly. |
| R1-W3 | Missing discussion/comparison with local dynamic frameworks. | 82% addressed | Added related-work discussion for batch-incremental, local, anchor-centered, overlapping, Delta-Screening, and dynamic Leiden methods. Clarified that ComNetX differs by wrapping arbitrary snapshot/dynamic solvers via contracted local subproblems. Audited the reviewer-mentioned "Han et al. 2017" item and resolved it as DiTursi, Ghosh, and Bogdanov's ICDM 2017 paper with the same title. | `article/article.tex`: Related Work, dynamic and local community detection; bibliography entries `CT13`, `DGB17`, `CBT23`, `BCA18`, `ZK21`, `SLe24`. | No empirical comparison has been added; keep this as the remaining risk. |
| R1-W4 | Limitation to undirected structures and loss of directionality are insufficiently discussed. | 55% addressed | Clarified that main real-data runs explicitly symmetrize graphs for cross-backend comparability. Prepared a final Leiden directed-control measurement block for `dyn_pubmed` and `arxivmath`. | `article/article.tex`: experimental setup and limitations; `scripts/paper/cn69_single_container_rerun_20260602/12_leiden_directed_control_99910.json`. | Complete directed-control measurements, package results, and add a concise article sentence/table if results are favorable. Current article still treats direction-preserving objectives as outside the main evaluation. |
| R1-W5 | Lack of discussion/evaluation of modularity resolution limits under local aggregation. | 90% addressed | Added a targeted Leiden gamma-sensitivity check for `gamma in {0.5, 1, 2}` on `dyn_pubmed` and `arxivmath`. The text states that speedups remain large while high resolution can increase quality sensitivity. | `article/article.tex`: modularity definition, ablation text, limitations; `results/icdm-2026-1/measurements/gamma_sweep_measurements.json`. | Optional: add a compact appendix/rebuttal table if page budget permits. Main article currently uses text only. |

### Potentially Missing Related Work

| ID | Reviewer-Suggested Work | Current Status | Current Handling | Remaining Work |
| --- | --- | --- | --- | --- |
| R1-RW1 | Chong and Teow, "An incremental batch technique for community detection", 2013. | 100% addressed | Cited as `CT13` and discussed as batch incremental maintenance. | None unless a fuller rebuttal paragraph is needed. |
| R1-RW2 | Christopoulos et al., "Local Community Detection in Graph Streams with Anchors", 2023. | 100% addressed | Cited as `CBT23` and contrasted as anchor-centered local stream detection. | None unless empirical comparison is requested. |
| R1-RW3 | Reviewer-listed "Han et al., Local Community Detection in Dynamic Networks, 2017." | 100% addressed | Bibliography audit found that the paper with this exact title is DiTursi, Ghosh, and Bogdanov, ICDM 2017. The article now cites it as `DGB17` with the conference venue and pages. | None, unless a later review names a different Han et al. paper explicitly. |
| R1-RW4 | Reviewer-listed "Palla et al./OLCPM online overlapping communities." | 100% addressed | Bibliography audit found that OLCPM is Boudebza, Cazabet, Azouaou, and Nouali, 2018. The article cites OLCPM as `BCA18` and contrasts it as online overlapping-community maintenance; Palla et al. remain relevant predecessors but are not the OLCPM authors. | None. |

### Rebuttal Questions

| ID | Reviewer Question | Current Status | Answer Strategy |
| --- | --- | --- | --- |
| R1-Q1 | What is the practical value of GNN backends if random features are fastest? | 85% addressed | Point to dataset-feature evidence: real features help on `dyn_cora`, and dataset-feature S2CAG is accelerated on `arxivmath`. State that random features are a stress/deployment recipe, not a semantic replacement. |
| R1-Q2 | What bottlenecks S2CAG when the contracted graph is small? | 85% addressed | Point to workload-profile text: neural backend execution, feature materialization, sparse tensor construction, and training dominate; ComNetX wrapper overhead is small. |
| R1-Q3 | Does local aggregation alter modularity resolution behavior? | 90% addressed | Point to `icdm-2026-1` gamma sweep: speedups remain large for `gamma in {0.5, 1, 2}`, but high resolution can increase quality sensitivity. |
| R1-Q4 | Why omit local dynamic related work, and how does ComNetX compare? | 78% addressed | Related Work now discusses batch-incremental, local/anchor, overlapping, Delta-Screening, and dynamic Leiden methods. Emphasize solver-agnostic contracted local subproblems; empirical comparison remains limited. |

### Follow-Up Queue

| Priority | Task | Status |
| --- | --- | --- |
| High | Complete and analyze directed Leiden control on `dyn_pubmed` and `arxivmath`. | 35% addressed |
| High | Replace Table III timing values after the single-container sequential rerun finishes, if variation confirms memory contention. | 45% addressed |
| Medium | Verify exact bibliographic target for the reviewer-mentioned Han et al. 2017 local dynamic work. | 100% addressed |
| Medium | Decide whether gamma sweep should remain text-only or become a compact table/appendix/rebuttal artifact. | 75% addressed |
