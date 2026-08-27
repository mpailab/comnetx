# IEEE Access Requirements and Article-Style Review

This note records the editorial decisions used for the ComNetX journal
revision. It is not manuscript text.

## Official requirements checked

The working source follows the IEEE Access LaTeX template dated May 13, 2026
and the journal's current author guidance:

- [Submission guidelines](https://ieeeaccess.ieee.org/authors/submission-guidelines/)
- [Preparing your article](https://ieeeaccess.ieee.org/authors/preparing-your-article/)
- [Peer-review stages](https://ieeeaccess.ieee.org/authors/stages-of-peer-review/)
- [Reproducibility guidance](https://ieeeaccess.ieee.org/authors/reproducibility/)
- [Post-acceptance guide](https://ieeeaccess.ieee.org/authors/post-acceptance-guide/)
- IEEE Editorial Style Manual for Authors

The resulting constraints are: use the official two-column, single-spaced
class; submit matching source and PDF files below 40 MB; keep a research
article preferably below 20 pages; use a single self-contained abstract of
150--250 words; provide searchable keywords; number references in citation
order; put funding in the first-page footnote; identify a corresponding author
and e-mail address; include author biographies; verify reference metadata; and
disclose substantive generative-AI assistance. A graphical abstract is not
part of the initial manuscript package.

## Related-article sample

The style review covered 20 IEEE Access research articles on dynamic community
detection, graph clustering, attributed networks, and scalable graph
analytics, published from 2019 through 2026, plus two IEEE Access survey
articles. Representative close examples include:

- *Dynamic Network Community Detection With Coherent Neighborhood
  Propinquity*, doi: 10.1109/ACCESS.2020.2970483.
- *Model Local and Global Evolution Relationship for Dynamic Networks*,
  doi: 10.1109/ACCESS.2019.2920237.
- *Density Sensitive Random Walk Based Community Detection in Large Scale
  Networks*, doi: 10.1109/ACCESS.2021.3058908.
- *Large-Scale Network Community Detection Using Similarity-Guided Merge*,
  doi: 10.1109/ACCESS.2021.3083971.
- *A Survey of Community Detection Methods in Static and Dynamic Networks*,
  doi: 10.1109/ACCESS.2020.2996595.
- *Local Community Detection: A Survey*,
  doi: 10.1109/ACCESS.2022.3213980.

Across the 20 research articles, the median length was 14 pages and the
interquartile range was approximately 13--18 pages. The recurring organization
was Introduction; Related Work; Problem or Preliminaries; Method; Experimental
Methodology; Results and Discussion; Limitations or Threats to Validity; and
Conclusion.

## Q1 scientific-contribution calibration

The contribution structure was separately compared with full primary texts of
closely related Q1-journal papers; journal prestige is not inferred from the
writing style of the IEEE Access sample.

- [DynaMo](https://doi.org/10.1109/TKDE.2019.2951419) in IEEE TKDE couples
  event-specific propositions and explicit complexity with six real networks,
  10,000 synthetic networks, five dynamic baselines, and quality/runtime
  metrics. It is the appropriate bar for a theory-led algorithmic claim.
- [CL-OND](https://doi.org/10.1016/j.neucom.2025.129548) in Neurocomputing is
  organized around four explicit questions that test feasibility, comparative
  performance, mechanism, and generality. It shows that a Q1 contribution can
  be RQ-led rather than theorem-heavy when every experiment tests a concrete
  premise.
- [Fusion3M](https://doi.org/10.1016/j.inffus.2025.103308) in Information
  Fusion uses five RQs, seven real datasets, nine baselines, component studies,
  and cross-model transfer. Only RQ1--RQ3 and the RQ4/RQ5 section titles are
  visible without subscription, so inaccessible question wording is not
  quoted or reconstructed.
- [Evolutionary NMF](https://doi.org/10.1109/TKDE.2017.2657752) in IEEE TKDE
  makes equivalences between temporal-smoothness formulations a headline
  result because they enable the proposed algorithm. This confirms that basic
  closure and quotient identities should support ComNetX rather than be sold
  as novelty by themselves.
- [CoD\AE N](https://doi.org/10.1145/3718988) in ACM TWEB makes a controlled,
  reproducible evaluation framework the main contribution. TWEB is Q1 in JCR
  Software Engineering and Q2 in Information Systems; it is used with that
  category qualification. Its independent generated streams, diagnostic
  metrics, and uncertainty intervals provide the relevant empirical standard.

The five original ComNetX questions are therefore retained as a scientific
program, ordered as effect, backend generality, localization mechanism,
component controls, and operating envelope. Their value depends on matched
metrics, nonredundant evidence, and explicit failure/break-even regimes—not on
the number of questions. The formal section is complementary: the main text
keeps one hierarchy-correctness proposition and explicit worst-case time and memory
bounds. Elementary closure and quotient facts support those results, while the
full boundary-objective and tensor-primitive derivations remain in appendices
rather than being advertised as independent novelty.

## Style decisions applied to ComNetX

- The abstract follows context--gap--method--protocol--numeric-result order.
- The introduction ends with three non-overlapping, testable contributions.
- Related work is organized by method family and includes an explicit contrast
  with LD-Leiden rather than a chronological catalogue.
- The method starts with notation and states the implementation order,
  approximation boundaries, and cost model precisely.
- Experiments are organized by research question. Dataset provenance,
  ordering, actual update counts, repetition scope, metrics, and fixed
  parameters precede the results.
- Results use claim--evidence--implication paragraphs. Runtime repeatability is
  not presented as statistical robustness, and exploratory parameter sweeps
  are not presented as a validated selection policy.
- Reproducibility and experimental scope are consolidated after the discussion
  instead of being repeated as defensive caveats throughout the article. The
  manuscript still avoids unsupported claims about an implemented fallback,
  post-update hierarchy nesting, local warm starts, objective preservation, or
  deletion-stream validation.

## Evidence policy

Every quantitative table and figure in the journal manuscript is generated
from a narrow, asserted record selection by `analysis/validate_results.py`.
The invalid conference-wide aggregate that mixed graph-direction settings,
feature modes, and unequal update counts is deliberately excluded. Absence of
a measurement record is not interpreted as an out-of-memory event.
