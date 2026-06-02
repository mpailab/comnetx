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
| R1-W3 | Missing discussion/comparison with local dynamic frameworks. | 86% addressed | Added related-work discussion for batch-incremental, local, anchor-centered, overlapping, Delta-Screening, and dynamic Leiden methods. Clarified that ComNetX differs by wrapping arbitrary snapshot/dynamic solvers via contracted local subproblems. Audited the reviewer-mentioned "Han et al. 2017" item and resolved it as DiTursi, Ghosh, and Bogdanov's ICDM 2017 paper with the same title. Added a baseline implementation audit and noted that DynaMo was attempted through its available Java implementation but did not complete the common protocol reproducibly. | `article/article.tex`: Baselines and Related Work; `article/baseline_implementation_audit.md`; bibliography entries `CT13`, `DGB17`, `CBT23`, `BCA18`, `ZK21`, `SLe24`, `ZCL19`. | No new empirical local-dynamic baseline has been added; keep this as the remaining risk. |
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

## Review R2: Preliminary Reviewer Assessment

- Received from: user-supplied preliminary reviewer assessment.
- Registered on: 2026-06-02.
- Overall status: 78% addressed.
- Overall risk: moderate. This review is positive on scope and
  experimental breadth, but it asks for stronger operational guidance:
  automatic or adaptive choice of $L,r$, explicit fallback behavior under
  hub-heavy updates, additional structural-quality metrics, and a more precise
  complexity discussion. The current revision addresses the first, second,
  fourth, and related-work issues with compact text changes; cut-based
  structural metrics and empirical comparison with mismatched local-query
  baselines remain the main open questions.
- Note on reviewer metadata: the pasted text ends with an `UNVERIFIED` note for
  "Communities in Networks" by Porter, Onnela, and Mucha. This is treated as a
  literature-check reminder rather than as a direct criticism of the current
  manuscript.

### Non-Actionable Checks

| ID | Reviewer Point | Status | Registry Note |
| --- | --- | --- | --- |
| R2-N1 | Paper length: pass. | 100% addressed | No action required. |
| R2-N2 | Topic compatibility: pass. | 100% addressed | No action required. |
| R2-N3 | Minimum quality: pass. | 100% addressed | No action required. |
| R2-N4 | Prompt-injection/hidden manipulation: pass. | 100% addressed | No action required. |
| R2-N5 | Relevance, novelty, reproducibility, ethics, GenAI content: generally acceptable. | 100% addressed | No action required unless later reviews conflict. |

### Actionable Weaknesses

| ID | Reviewer Concern | Current Status | Corrections Already Made | Evidence | Remaining Work |
| --- | --- | --- | --- | --- | --- |
| R2-W1 | Figure 3/topology frontier shows that $L$ and $r$ strongly affect performance, but the paper lacks a heuristic or automatic selection rule for unseen graphs. | 90% addressed | The article now gives a stronger factual discussion and Figure 3 includes measured grids for `dyn_cora`, `dyn_pubmed`, and `arxivmath`; together with neighborhood growth this supports the statement that selection is topology-dependent and no monotone rule was observed. It then recommends a pilot-and-budget protocol rather than claiming automatic optimality. | `article/article.tex`: topology ablation discussion and Figure 3 interpretation; `article/topology_ablation_pareto.pdf`; `results/icdm-2026-0`. | Optional only: add a numeric threshold if future measurements justify one. Current text deliberately avoids an unsupported universal rule. |
| R2-W2 | Table IV shows that larger radii can engulf much of the graph; high-degree or central updates may cause near-full recomputation without graceful degradation. | 88% addressed | Complexity and limitations now make the fallback operational through monitored fractions $b_t/|V|$, $\max_\ell h_\ell/|V|$, and $\max_\ell\bar m_\ell/\mathrm{nnz}(A_t)$. DSBM stress results already show where the local regime breaks. | `article/article.tex`: complexity/locality discussion, topology ablation, DSBM stress-study discussion, limitations. | Remaining risk is empirical threshold calibration; current paper intentionally treats the guardrail as budget-based rather than universal. |
| R2-W3 | Modularity dominates the evaluation; structural metrics such as conductance or normalized cut could give a broader view of cluster quality. | 50% addressed | The article now states only the metrics actually computed in the result logs: NMI, ARI, macro-F1, and modularity. The weak defensive sentence about conductance/normalized cut was removed. | `article/article.tex`: metrics; `results/icdm-2026-0`; `results/icdm-2026-1`. | Open measurement question: if this concern is important enough to answer in the paper, conductance/normalized-cut values should be measured and then reported or discussed. Otherwise, do not mention them in the manuscript. |
| R2-W4 | Complexity analysis is superficial because $k_\ell$ and $\bar m_\ell$ are not bounded in terms of graph density, degree distribution, or batch properties. | 88% addressed | The paper now gives meaningful instance-dependent bounds: $b_t$ is tied to degrees around $S_t$, $h_\ell$ to sizes of touched previous communities, $k_\ell$ to the number of touched communities, and $\bar m_\ell$ to the induced local edge count. It also states why only full-graph worst-case bounds exist without topology/update assumptions. | `article/article.tex`: complexity/locality section; workload and neighborhood tables. | A formal probabilistic bound would require assumptions on degree distribution/update process and is not added to avoid overclaiming. |
| R2-W5 | Several local/dynamic community-detection works are missing from literature review and baseline comparisons. | 92% addressed | Related Work now cites and classifies Sattar et al. 2023, Han et al. 2017 adaptive label propagation, Liu et al. 2021 seed-based multiple local community detection, and the already cited local/anchor/overlapping/delta-screening lines. The text distinguishes full-partition maintenance from query/anchor/seed local discovery and solver-specific update rules. A baseline implementation audit was added and the Baselines subsection now states the empirical inclusion criterion: same Python-controlled batched workflow, same edge-update stream, and complete hard partitions. The DynaMo audit now records the direct Java attempt and the unresolved failure after an attempted code-level correction. | `article/article.tex`: Baselines and Related Work; `article/baseline_implementation_audit.md`; refs `CT13`, `DGB17`, `CBT23`, `LSS21`, `BCA18`, `ZK21`, `SLe24`, `HLS17`, `Sattar23`, `Sattar25`, `ZCL19`. | Remaining risk: a reviewer may still request an extra executable Python temporal baseline such as `tnetwork` smoothed Louvain or `df-louvain`. The authors' concurrent arXiv work remains citation-deferred. |

### Potentially Missing Related Work

| ID | Reviewer-Suggested Work | Current Status | Current Handling | Remaining Work |
| --- | --- | --- | --- | --- |
| R2-RW1 | Bokov et al., "A Parallel Hierarchical Approach for Community Detection on Large-scale Dynamic Networks", 2025. | 60% addressed / citation deferred | This is the authors' own concurrent, unpublished arXiv manuscript intended for the same conference cycle. The current article already discusses parallel and dynamic Leiden-style work at a general level. | Do not add this citation mechanically. Add it only if the submission policy, anonymity constraints, and positioning strategy clearly permit it; otherwise address the reviewer's concern through concise discussion of the broader parallel/dynamic setting without overloading Related Work. |
| R2-RW2 | Sattar et al., "Exploring temporal community evolution: algorithmic approaches and parallel optimization for dynamic community detection", 2023. | 100% addressed | Added as `Sattar23` with corrected bibliographic metadata and classified under dynamic CD methods balancing quality, temporal consistency, and update efficiency. | None. |
| R2-RW3 | Liu et al., "Multiple Local Community Detection via High-Quality Seed Identification over Both Static and Dynamic Networks", 2021. | 100% addressed | Added as `LSS21` and classified as seed-based multiple local community detection, i.e., related but closer to query/local-community discovery than full-partition snapshot maintenance. | None. |
| R2-RW4 | Liu et al., "Local Community Detection in Graph Streams with Anchors", 2021. | 80% addressed | The exact reviewer metadata appears inconsistent with the currently verified anchor-stream paper, which is Christopoulos et al. 2023 and is already cited as `CBT23`. The local/anchor regime is explicitly covered and contrasted. | Re-check if the reviewer later provides a precise Liu 2021 anchor citation. |
| R2-RW5 | Han et al., "Community detection in dynamic networks via adaptive label propagation", 2017. | 100% addressed | Added as `HLS17` with corrected bibliographic metadata and classified with incremental/adaptive label-propagation methods. | None. |
| R2-RW6 | Porter, Onnela, and Mucha, "Communities in Networks", 2009. | 100% addressed | Confirmed as `POM09` and kept as background on community-detection quality and limitations rather than as a local-dynamic baseline. | None. |

### Rebuttal Questions

| ID | Reviewer Question | Current Status | Answer Strategy |
| --- | --- | --- | --- |
| R2-Q1 | Can the authors provide theoretical or empirical guidelines to adapt $r$ and $L$ dynamically? | 90% addressed | Article now gives measured evidence from three topology grids that useful $L,r$ choices differ by graph topology, states that no monotone rule was found, and recommends a conservative pilot-and-budget protocol. |
| R2-Q2 | How does ComNetX handle anomalous super-hub updates that inflate the $r=1$ neighborhood? | 85% addressed | Article now ties the fallback interpretation to monitored workload fractions; DSBM hub-centered stress streams provide empirical support. |
| R2-Q3 | How are edge weights handled during feature aggregation for GNNs? Is structural importance weighted? | 95% addressed | Article now states explicitly that structural edge weights are preserved in $\bar A$ and do not additionally reweight $\bar X$ in the reported feature modes. |
| R2-Q4 | How does ComNetX compare to missing local-update and anchor-based strategies? | 90% addressed | Related Work now cites the verified external works and states that they target specific update rules, label-propagation heuristics, query/anchor/seed regimes, or overlapping-community models. The Baselines subsection now adds a reproducibility criterion for empirical inclusion, and the separate implementation audit records which candidates lack a native Python path or use incompatible outputs. |

### Follow-Up Queue

| Priority | Task | Status |
| --- | --- | --- |
| High | Add an operational fallback/adaptive-parameter paragraph based on monitored affected-neighborhood, closure, and contracted-edge fractions. | 88% addressed |
| High | Clarify weighted adjacency versus unweighted/normalized feature aggregation in the method section. | 95% addressed |
| Medium | Verify and triage external R2 missing related works (`Sattar23`, `Liu21` seed/anchor, `Han17` adaptive label propagation); keep the authors' concurrent Bokov et al. arXiv work as a deferred citation unless policy and positioning make it necessary. | 90% addressed |
| High | Decide whether to compute conductance/normalized-cut. Current manuscript should not mention them unless measured or backed by a clear methodological reason. | 50% addressed |
| Medium | Improve complexity discussion by connecting $k_\ell,\bar m_\ell$ to monitored empirical quantities and worst-case full-refresh behavior. | 88% addressed |
