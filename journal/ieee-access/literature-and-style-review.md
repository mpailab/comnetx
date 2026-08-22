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
- A dedicated limitations section separates what the implementation does from
  proposed future safeguards. In particular, the article does not claim an
  implemented fallback, post-update hierarchy nesting, local warm starts,
  objective preservation, or deletion-stream validation.

## Evidence policy

Every quantitative table and figure in the journal manuscript is generated
from a narrow, asserted record selection by `analysis/validate_results.py`.
The invalid conference-wide aggregate that mixed graph-direction settings,
feature modes, and unequal update counts is deliberately excluded. Absence of
a measurement record is not interpreted as an out-of-memory event.

