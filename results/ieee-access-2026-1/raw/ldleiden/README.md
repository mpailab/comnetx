# Raw LD-Leiden Measurements

This namespace preserves historical LD-Leiden diagnostics. They are not part
of the journal's quantitative benchmark and must not be pooled with ComNetX
results. Existing raw files and archives are retained unchanged.

The eight launch scripts used for the August 2026 diagnostic, including their
split-clock and priming metadata, remain recoverable from commit `79a2e74`.
The superseded campaign frameworks were removed during the September code
audit. The current general dynamic launcher uses its original combined
`apply()` time; newly generated records therefore do not have the same timing
contract as the historical diagnostic.
