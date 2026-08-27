# Scientific Argument and 72-Hour Evidence Plan

Status: working design record. This document fixes the scientific claims before
the manuscript or experiment set is reduced. No existing result should be
removed merely because it is not central to the first draft of this map.

## Central thesis

ComNetX is a local hierarchical adaptation method for giving a compatible
existing community detector a batched local-update path without reimplementing
the detector's objective or optimizer. The method is implemented as an adapter,
and its characterization has three parts:

1. the method constructs weighted quotient calls through the adapter and
   restores a complete partition of the original graph;
2. the analysis shows method-specific correctness properties and explicit
   worst-case time and memory requirements, including the graph-wide case; and
3. a structured empirical study demonstrates the resulting quality--time
   trade-off and identifies when the adapter cost is or is not recovered.

Community closure and quotient contraction remain important parts of the
mechanism, but the distinction between their roles is not a standalone
contribution. Closure supplies complete touched-community context, while the
quotient reduces a compatible detector call when the stored hierarchy admits a
compact representation. The exact quotient identity supports correctness on
the represented search space; the full-graph boundary derivation is retained as
technical analysis rather than presented as the main scientific result.

## Formal results and their placement

The main article should contain only the statements needed to trust or size the
method: exact scoped quotienting, one combined update-correctness proposition, and
explicit worst-case time and memory bounds parameterized by backend cost.
Elementary closure facts belong in explanatory prose. Full boundary and
primitive-level tensor derivations belong in appendices.

### Proposition 1: minimal community closure

Let \(\mathcal P\) be a partition of \(V\), let \(B\subseteq V\), and define

\[
U_{\mathcal P}(B)=\bigcup_{P\in\mathcal P:P\cap B\ne\varnothing}P.
\]

Then \(U_{\mathcal P}(B)\) is the unique smallest set that contains \(B\) and
is a union of blocks of \(\mathcal P\).

**Proof.** It is a union of blocks and contains \(B\). Every union of blocks
that contains \(B\) must contain the entire block intersected by each element
of \(B\), hence it contains \(U_{\mathcal P}(B)\).

This result gives closure a precise role: it is the least partition-saturated
label-update scope, not an arbitrary enlargement of a radius neighborhood.

### Proposition 2: nested closures of a nested pre-update hierarchy

If \(\mathcal P_f\) refines \(\mathcal P_c\), then for every \(B\subseteq V\),

\[
U_{\mathcal P_f}(B)\subseteq U_{\mathcal P_c}(B).
\]

**Proof.** Every fine block intersecting \(B\) is contained in a coarse block
that also intersects \(B\). Taking the corresponding unions proves the
inclusion.

This result applies to the stored hierarchy immediately before an update when
that hierarchy is nested. It does not by itself prove that the hierarchy
remains nested after an arbitrary multilevel update procedure.

The implemented weighted-SpMV neighborhood equals a radius ball in the
symmetrized support graph only while resident edge weights are nonnegative.
Signed updates can cancel during the multiplication; supporting them would
require an explicit Boolean support adjacency or an equivalent endpoint
traversal.

### Theorem 1: exact quotient sufficient statistics

Let \(\bm A\) be the nonnegative weighted adjacency of the current working
graph. Let \(G_1,\ldots,G_k\) be nonempty, pairwise disjoint contracted groups
covering every working vertex; equivalently, if \(\bm A\) retains a larger
ambient index space, all rows and columns outside their union are zero. Define
\(P_{ai}=1\) if \(i\in G_a\), and zero otherwise. Thus every active column of
\(\bm P\) contains exactly one unit entry. Let

\[
\bar{\bm A}=\bm P\bm A\bm P^{\mathsf T}.
\]

Then

\[
\bar A_{ab}=\sum_{i\in G_a}\sum_{j\in G_b}A_{ij}.
\]

Therefore total edge weight and directed weighted degrees are preserved by
aggregation:

\[
\bar W=W,\qquad
\bar d_a^{\mathrm{out}}=\sum_{i\in G_a}d_i^{\mathrm{out}},\qquad
\bar d_a^{\mathrm{in}}=\sum_{i\in G_a}d_i^{\mathrm{in}}.
\]

The same statement covers an undirected graph represented by a symmetric
adjacency. Self-loops created inside contracted groups must be retained, and
the same matrix-based self-loop and degree convention must be used before and
after contraction.

### Theorem 2: exact objective value on the group-constant search space

Assume \(W=\sum_{ij}A_{ij}>0\). Abstract partitions of the quotient vertices
are in bijection with partitions of the working vertices that do not split any
\(G_a\). For a quotient partition \(z\), define its lift by \(c_i=z_a\) for
\(i\in G_a\). Under the same directed/symmetric adjacency convention,
self-loop convention, and resolution \(\gamma\),

\[
Q_\gamma(\bar{\bm A},z)=Q_\gamma(\bm A,c).
\]

**Proof sketch.** For any two lifted blocks \(r,s\), the block-flow matrix

\[
F_{rs}(c)=\sum_{i:c_i=r}\sum_{j:c_j=s}A_{ij}
\]

is equal to the corresponding quotient block flow after expanding every
\(\bar A_{ab}\) by Theorem 1. Internal weight and total outgoing and incoming
volume are therefore identical for every lifted block. Substitution into the
weighted modularity expression proves the equality term by term.

The same block-flow identity exactly preserves directed cut weights and
matrix-defined volumes of every union of contracted groups. Conductance and
normalized-cut values therefore agree only when both sides use the same
self-loop, volume, direction, and zero-volume conventions.

The theorem does **not** claim equivalence to the full \(\bm A_t\) after
boundary masking, equivalence for partitions that split contracted groups, an
identical solution from a heuristic backend, or preservation of an arbitrary
feature-aware objective under mean/sum feature aggregation.

### Corollary 1: associative hierarchical contraction

If a binary membership matrix \(\bm R\), with exactly one unit entry per
column, groups the vertices of an existing quotient on the **same underlying
working vertex set**, then quotienting in two stages is exactly the same as
quotienting the original working graph by the composed membership:

\[
\bm R(\bm P\bm A\bm P^{\mathsf T})\bm R^{\mathsf T}
=(\bm R\bm P)\bm A(\bm R\bm P)^{\mathsf T}.
\]

Thus a hierarchy-preserving parent quotient does not accumulate an algebraic
edge-weight approximation. Sparse-kernel rounding aside, direct and recursive
construction contain the same sufficient statistics. This corollary cannot be
used to recover a quotient on a larger scope \(U'\supsetneq U\) from a quotient
built only on \(U\): the smaller object no longer contains the added vertices
or their incident edges. Expanding scopes must be rebuilt from the resident
adjacency, or from an exact quotient of that same larger vertex set.

### Proposition 3: exact boundary-objective mismatch decomposition

Let \(U\) be an updated scope in a nonnegative weighted graph with
\(W>0\), \(W_U>0\), and \(\gamma\ge0\). Require its new labels to be disjoint
from the retained labels on \(V\setminus U\), and keep the outside partition
fixed. For an inside partition \(z\), write \(e_a\) for the weight internal to
its inside block \(a\), and define

\[
x_a=d_{U,a}^{\rm out},\qquad y_a=d_{U,a}^{\rm in},
\]

\[
p_a=\sum_{i\in a,\,j\notin U}A_{ij},\qquad
q_a=\sum_{i\notin U,\,j\in a}A_{ij}.
\]

Thus the full-graph degrees of block \(a\) are \(x_a+p_a\) and
\(y_a+q_a\). Under the same directed-adjacency convention used by the
modularity definition,

\[
Q_G(C_z)=K_{V\setminus U}+
\frac{1}{W}\sum_a\left(e_a-
\gamma\frac{(x_a+p_a)(y_a+q_a)}{W}\right),
\]

where \(K_{V\setminus U}\) is independent of \(z\), whereas

\[
\frac{W_U}{W}Q_{G[U]}(z)=
\frac{1}{W}\sum_a\left(e_a-
\gamma\frac{x_a y_a}{W_U}\right).
\]

Consequently, the exact objective mismatch for the same lifted partition is

\[
\begin{aligned}
&Q_G(C_z)-K_{V\setminus U}-\frac{W_U}{W}Q_{G[U]}(z)\\
&\quad=\frac{\gamma}{W^2}\left[
\frac{W-W_U}{W_U}\sum_a x_a y_a
-\sum_a\left(x_aq_a+y_ap_a+p_aq_a\right)
\right].
\end{aligned}
\]

Let \(\beta_{\rm out}=\sum_a p_a\) and
\(\beta_{\rm in}=\sum_a q_a\), and write the mismatch above as \(D(z)\).
Nonnegativity gives the sharper partition-independent interval

\[
-B_-\le D(z)\le B_+,
\]

where

\[
B_+=\frac{\gamma}{W^2}W_U(W-W_U)
\]

and

\[
B_-=\frac{\gamma}{W^2}\left[
W_U(\beta_{\rm out}+\beta_{\rm in})
+\beta_{\rm out}\beta_{\rm in}\right].
\]

Indeed, \(\sum_a x_a y_a\le W_U^2\),
\(\sum_a x_aq_a\le W_U\beta_{\rm in}\),
\(\sum_a y_ap_a\le W_U\beta_{\rm out}\), and
\(\sum_a p_aq_a\le\beta_{\rm out}\beta_{\rm in}\). Consequently,
\(|D(z)|\le\max(B_-,B_+)\), while the maximum possible difference between
the mismatches of two candidate partitions is bounded by the tighter interval
width \(B_-+B_+\).

The internal-edge term is identical. The positive mismatch term records the
change from global normalization \(W\) to local normalization \(W_U\); the
negative terms record outgoing and incoming boundary degree. Even when the
boundary degree is zero, a disconnected outside component can leave
\(W_U\ne W\), so the null models can still differ. This separates exact
contraction on the group-constant working-graph space from the objective
change introduced by boundary masking.

### Corollary 2: restricted global-objective loss and ranking certificate

Let \(\mathcal S\) be one fixed, nonempty family of admissible inside
partitions: the outside partition is fixed and disjointly labeled, and every
member of \(\mathcal S\) obeys the same declared contraction atoms. Define

\[
F(z)=Q_G(C_z)-K_{V\setminus U},\qquad
L(z)=\frac{W_U}{W}Q_{G[U]}(z).
\]

If \(z_U\in\arg\max_{z\in\mathcal S}L(z)\) and
\(z_G\in\arg\max_{z\in\mathcal S}F(z)\), then

\[
0\le F(z_G)-F(z_U)\le B_-+B_+.
\]

More generally, if a backend output \(\widehat z\) has a certified local
suboptimality

\[
Q_{G[U]}(z_U)-Q_{G[U]}(\widehat z)\le\varepsilon,
\]

then

\[
F(z_G)-F(\widehat z)
\le B_-+B_++\frac{W_U}{W}\varepsilon.
\]

For two candidates \(z,z'\in\mathcal S\), the ordering is certified whenever

\[
L(z)-L(z')>B_-+B_+,
\]

because the boundary mismatch cannot reverse a gap larger than its entire
admissible interval. If the gap is reported in unscaled local modularity, the
corresponding threshold is \((W/W_U)(B_-+B_+)\).

**Proof.** Since \(F=L+D\) and every \(D(z)\) lies in
\([-B_-,B_+]\), the difference between two mismatch values is at most
\(B_-+B_+\). Local optimality makes \(L(z_G)-L(z_U)\le0\); the certified
\(\varepsilon\) case replaces this term by at most
\((W_U/W)\varepsilon\). The ranking statement follows from the same interval
width.

This is a localization certificate only relative to the same restricted
candidate family \(\mathcal S\), not a regret bound against an unrestricted
full-graph partition. A heuristic backend supplies no usable \(\varepsilon\)
unless it exposes an independent optimality certificate. The empirical study
must report \(B_-+B_+\) (and preferably the realized mismatch) to establish
whether the certificate is informative rather than vacuous on the measured
streams.

### Proposition 4: canonical collision-free projection

Let the pre-update partition be stored canonically: every block \(H\) has label
\(\kappa(H)=\min H\). Let \(U\) be a union of complete pre-update blocks, let a
backend return one label for every contracted vertex, and let
\(H_1,\ldots,H_k\) be the abstract blocks induced on \(U\) after projection
through the contraction membership. Retain every label on \(V\setminus U\)
and assign

\[
\kappa(H_a)=\min H_a
\]

to every vertex of \(H_a\). The projected vector has length \(n\), uses labels
in \(\{0,\ldots,n-1\}\), is invariant to every one-to-one integer relabeling
of the backend partition, writes no entry outside \(U\), and has no partition
block crossing the boundary between \(U\) and \(V\setminus U\).

**Proof.** Canonicalizing the backend output by its induced blocks makes the
construction independent of the raw backend namespace. Distinct nonempty
blocks have distinct minimum vertices. Every new label is a vertex of \(U\),
whereas every retained outside label is the minimum vertex of a complete old
block contained in \(V\setminus U\). Hence the two namespaces are disjoint.
The construction writes exactly the entries indexed by \(U\).

This concrete representative rule needs neither an unbounded integer namespace
nor a monotone allocator. It relies on canonicalizing the complete stored
hierarchy at initialization and after loading a cache. A schema-valid deep
cache is accepted only after the canonical rows pass the fine-to-coarse nesting
test; otherwise the hierarchy is rebuilt from the snapshot. The legacy production
projection `old_idx[backend_labels[inverse]]` additionally assumed that raw
backend labels were compact indices in \([0,k-1]\); the repaired construction
first forms backend blocks and then labels their lifted original-vertex blocks
by their minima.

### Legacy-only note: monotone destructively cut support

The legacy update zeroes every working edge that is outside the current mask
or crosses the updated partition. Its mathematical nonzero support is therefore
monotonically nonincreasing across passes. The in-place COO tensor nevertheless
retains explicit zero-valued entries, so this is not a claim about stored
`_nnz()` or sparse-kernel cost.

This property is deliberately **not** an invariant of the repaired
parent-quotient construction. Each repaired level is rebuilt from the resident
adjacency on its own, generally larger closure, so its support can grow from
one level to the next.

## Hierarchy invariant audit

### What is proved at initialization

In implementation order, row 0 is the reported fine partition. Each following
row is obtained by clustering a quotient whose vertices are the communities of
the preceding row and lifting the quotient labels back to the original
vertices. Each lifted row is therefore a function of the preceding row. The
initial rows are nested from fine to coarse.

### Why the legacy update did not guarantee post-update nesting

The legacy update computed all affected masks from the pre-update state,
propagated provisional labels, and then called a general partition-producing
backend independently at successive levels. Cutting the working adjacency
removed cross-label weights, but it did not require the next backend partition
to keep every community produced by the preceding pass indivisible.

A four-vertex counterexample uses the initially nested rows

\[
C^0=(0,0,2,2),\qquad C^1=(0,0,0,0),
\]

where row 0 is fine and row 1 is coarse, and marks vertex 1 as affected. The
open/propagate stage gives two groups on vertices \(\{0,1\}\). Let the first
admissible backend call merge these groups and let the second return the
identity partition on its two input groups. The final fine row contains block
\(\{0,1\}\), while the final coarse row assigns different labels to vertices 0
and 1. Thus the fine block is split at the nominally coarse level.

This refutes an unconditional post-update nesting theorem for independently
recomputed levels under the stated general backend contract.

### Empirical audit of the legacy implementation

A deterministic quotient audit passes the exact weight, degree, modularity,
cut, and volume identities at zero numerical error for both directed and
symmetric weighted graphs. The contraction algebra is therefore not the
problem.

The update-state audit exposes a concrete namespace defect. On the local
`dyn_pubmed` (999{:}10) stream, using the same three-level, radius-one protocol
and a Leiden bootstrap, adjacent-level refinement fails on updates 2--9. The
same updates contain 20 labels shared across a precomputed affected-scope
boundary and exactly 20 violating fine blocks, all at the first fine-to-coarse
transition. Entries outside the scope are not written, but equality of a new
inside label and a retained outside label merges the two sides in the induced
partition. Thus legacy entrywise locality holds while partition-level
isolation does not.

A namespace-only research prototype assigns fresh provisional and projected
labels while leaving the backend calls, contractions, and absence of explicit
hierarchy repair unchanged. Across nine deterministic repetitions of the same
stream it has zero boundary-label collisions, zero pre- or post-update
refinement failures, and zero outside entry writes. Final modularity changes
from \(0.7567715645\) to \(0.7813629508\). The median paired
`Optimizer.run` time ratio is \(1.0032\), within the much larger run-to-run
range. The quality difference is an observed outcome, not a pure causal
estimate: fresh labels also change the sorted quotient-vertex order seen by a
permutation-sensitive heuristic. The structural result is cleaner: on this
stream the observed refinement defect disappears with namespace repair alone.
The general backend counterexample above still precludes an unconditional
nesting theorem for the legacy update rule.

This audit used the locally available Python Leiden bootstrap because the
optional native dynamic-community wheel is not installed in the development
container. The exact first failing update can therefore differ on the
measurement server, but the deterministic counterexample and the collision
mechanism do not depend on that package.

### Collision-free hierarchy-preserving construction

Merely weakening the prose is no longer sufficient: collision-free canonical
projection is a correctness repair. For an updated scope \(U\), form the
backend-induced blocks, retain the labels on \(V\setminus U\), and apply
Proposition 4 by labeling every lifted inside block with its minimum original
vertex.

For a scientifically meaningful hierarchy, every coarser pass should then use
the blocks of the immediately preceding updated finer partition as its
quotient vertices. In implementation order, let

\[
\mathcal P_t^0\preceq\mathcal P_t^1\preceq\cdots\preceq
\mathcal P_t^{L-1}.
\]

Compute all closures from the immutable pre-update hierarchy. Update

1. level 0 on the induced graph \(\bm A_t[U_t^0]\), using the explicit base
   atom partition
   \[
   \mathcal G_t^0=
   \{\{v\}:v\in B_t\}\cup
   \{(H\cap U_t^0)\setminus B_t:
     H\in\mathcal P_{t-1}^{L-1}\}\setminus\{\varnothing\};
   \]
   that is, every expanded vertex is free and each nonempty unaffected
   remainder of a coarsest stored block is contracted; and
2. every level \(\ell>0\) on \(\bm A_t[U_t^\ell]\), contracted directly by
   the blocks of the already updated \(\mathcal P_t^{\ell-1}\).

Each level is rebuilt from the resident adjacency rather than from a
destructively cut working matrix. The same rule applies at initialization:
each coarser quotient is formed from the original adjacency or, because every
initial level uses the same full vertex set, recursively from the preceding
exact quotient. During an update the scopes can expand with \(\ell\), so a
quotient built only on \(U_t^{\ell-1}\) cannot supply the added vertices and
edges of \(U_t^\ell\); each such level must be rebuilt from the resident graph
or an exact representation of that same larger scope.

**Base-atom equivalence lemma.** Assume the pre-update hierarchy is nested and
all provisional relabelings are injective and collision-free. Singletonize
every \(v\in B_t\) on the coarsest stored row and propagate only the induced
partition relation down through the nested scopes. The atoms presented to the
first, reported-level quotient are exactly \(\mathcal G_t^0\) above.

**Proof.** Proposition 2 gives
\(U_t^0\subseteq\cdots\subseteq U_t^{L-1}\). Every vertex of \(U_t^0\)
therefore participates in every downward propagation step. An expanded vertex
retains its unique singleton identity. Every other vertex retains the identity
of its pre-update coarsest block, restricted to \(U_t^0\). Injective fresh
relabeling changes only numeric names, not this equivalence relation.

The displayed base atoms define an aggressive hierarchy-driven restriction.
The implemented vertex-level comparison uses

\[
\mathcal G_{t,\mathrm{vertex}}^0=
\{\{v\}:v\in U_t^0\}.
\]

It refines \(\mathcal G_t^0\), so its group-constant feasible partitions are a
superset of those available under hierarchical base atoms. Consequently, the
exact modularity optimum on the same induced scope cannot be lower, although a
heuristic backend output need not be monotone. This policy removes only the
level-0 contraction restriction; higher passes still use parent quotients of
the updated finer partition and therefore preserve the hierarchy theorem. It
also makes the reported-level search space independent of the auxiliary
hierarchy depth, at the cost of \(k_0=h_0\), typically larger \(q_0\), and more
backend work. In the evaluation this is called **vertex-level base atoms**, not
the ambiguous legacy phrase “no contraction.”

**Updated-scope saturation lemma.** For every \(\ell>0\), the immutable
closure \(U_t^\ell\) is a union of complete blocks of the already updated
partition \(\mathcal P_t^{\ell-1}\).

**Proof.** Pre-update refinement makes \(U_t^\ell\) a union of complete
\(\mathcal P_{t-1}^{\ell-1}\)-blocks, and Proposition 2 gives
\(U_t^{\ell-1}\subseteq U_t^\ell\). New level-\(\ell-1\) blocks created inside
\(U_t^{\ell-1}\) remain inside \(U_t^\ell\). Outside
\(U_t^{\ell-1}\), the level-\(\ell-1\) blocks are unchanged old blocks and
therefore lie either wholly inside or wholly outside \(U_t^\ell\). Canonical
collision-free projection prevents either kind of block from crossing the
smaller update boundary.

**Hierarchy theorem.** Suppose
\(\mathcal P_{t-1}^0\preceq\cdots\preceq
\mathcal P_{t-1}^{L-1}\), all closures are computed from that immutable
hierarchy, and every level uses canonical collision-free projection. Let the
level-0 result on \(U_t^0\) be the lift of a backend partition of a declared
base atom partition, with \(V\setminus U_t^0\) unchanged. For every
\(\ell>0\), let the backend partition quotient vertices that are exactly the
blocks of \(\mathcal P_t^{\ell-1}\) contained in \(U_t^\ell\), lift that
partition inside \(U_t^\ell\), and leave the old level-\(\ell\) partition
outside. Then

\[
\mathcal P_t^0\preceq\mathcal P_t^1\preceq\cdots\preceq
\mathcal P_t^{L-1},
\]

and no block of \(\mathcal P_t^\ell\) crosses the boundary of
\(U_t^\ell\).

**Proof.** The updated-scope saturation lemma makes the level-\(\ell-1\)
blocks valid quotient atoms on \(U_t^\ell\). A quotient partition can merge
these atoms but cannot split them, so its lift coarsens
\(\mathcal P_t^{\ell-1}\) inside the scope. Outside the scope, an unchanged
old fine block is contained in an unchanged old coarse block because the old
hierarchy is nested and \(U_t^\ell\) is a union of complete old coarse blocks.
No fine block crosses the scope by the saturation lemma, and no new coarse
block crosses it by canonical collision-free projection. Induction over
\(\ell\) proves both conclusions.

**Supporting lemma for the radius-only control (arbitrary-scope split
projection).** Let \(U\subseteq V\) be arbitrary rather than a union of old
blocks. Give every backend-induced block inside \(U\) a temporary label that
is disjoint from every stored label, retain the old labels on
\(V\setminus U\), and then canonicalize the complete row by block minima.
The inside backend blocks may merge vertices originating in different old
blocks. Independently, every old block \(H\) that crosses the boundary leaves
the single outside remainder \(H\setminus U\). The induced partition on
\(V\setminus U\) is unchanged, no resulting block crosses the boundary, and
numeric representatives outside \(U\) may change.

If an initially nested hierarchy applies this split projection at level 0 and
every higher level partitions quotient atoms that are the complete blocks of
the already updated preceding row inside the same \(U\), the resulting rows
remain nested. Inside \(U\), a quotient can merge but cannot split the updated
finer blocks. Outside \(U\), each remainder of an old finer block is contained
in the remainder of its old coarser parent. These two facts prove the claim by
induction. This is a separate safety result for the radius-only experimental
control: it does not imply entrywise locality outside \(U\), and it is not the
minimal-closure hierarchy theorem used by the full method.

The theorem guarantees non-strict refinement. A general backend may return the
identity partition, so distinct adjacent levels cannot be guaranteed without
an explicit multiresolution schedule or a backend-provided hierarchy. Forcing
an arbitrary merge solely to make levels visually different would weaken the
method.

A post-hoc partition-lattice join could enforce compatibility while preserving
the current backend calls, but it is not the preferred repair. A single
crossing block can trigger transitive over-merging, and the joined result is no
longer the partition optimized by the backend. The parent-quotient construction
makes nesting part of the backend search space instead.

All multibatch smart-mode measurements must be treated as provisional after
this repair. Level 0 can remain unchanged in the first batch, but corrected
coarser state changes later closures and hence later reported partitions. No
old smart result will be silently attributed to the repaired algorithm; the
minimum claim-bearing suite must be rerun before manuscript numbers are
changed.

## Explicit cost model

The cost model below describes the repaired parent-quotient construction, not
the legacy destructively cut prototype. Distinguish mathematical nonzero
support from stored sparse entries:

- \(n=|V|\);
- \(s_t\): stored COO entries in the resident \(\bm A_t\), including any
  explicit zeros, and \(m_t=|\operatorname{supp}\bm A_t|\le s_t\);
- \(\delta_t\): stored entries in the incoming update batch;
- \(b_t=|B_t|\) after radius expansion;
- \(r_t\le r\): the number of SpMV iterations actually executed before the
  radius loop stops, and \(h_\ell=|U_t^\ell|\);
- \(e_\ell\): stored coalesced COO entries in the restriction
  \(\bm A_t[U_t^\ell]\), including any explicit zero values;
- \(k_\ell\): quotient vertices at pass \(\ell\), where \(k_0\) is the size
  of the declared base atom partition and, for \(\ell>0\), \(k_\ell\) is the
  number of updated parent blocks contained in \(U_t^\ell\);
- \(q_\ell\): stored COO entries in \(\bar{\bm A}_\ell\), again including
  explicit zeros if the sparse kernel retains them;
- \(d\): dense feature dimension, \(f_\ell\): stored sparse-feature entries
  selected at level \(\ell\), and \(F\): resident feature size, equal to
  \(nd\) for dense features or \(\operatorname{nnz}(\bm X)\) for sparse
  features;
- \(T_{\mathcal B}(k,q,d)\), \(C_{\mathcal B}(k,q,d)\), and
  \(M_{\mathcal B}(k,q,d)\): backend algorithm time, backend-format conversion
  time, and backend workspace under the selected timing convention.

For the explicit coarsest-remainder base atoms,

\[
k_0=|B_t|+
\left|\left\{H\in\mathcal P_{t-1}^{L-1}:
(H\cap U_t^0)\setminus B_t\ne\varnothing\right\}\right|.
\]

For vertex-level base atoms, \(k_0=h_0\).

### Primitive-level bound for the PyTorch implementation

PyTorch does not publish a uniform asymptotic guarantee for `torch.unique`,
generic sparse--sparse `mm`, or device-specific coalescing. A rigorous
implementation bound must therefore expose those primitives. Let
\(C_{\rm unique}(x)\), \(C_{\rm sort}(x)\),
\(C_{\rm contract}(e,h,q)\), and
\(C_{\rm feat}(h,k,d,f)\) denote the actual costs of label compaction, group
ordering, sparse contraction, and feature aggregation. For dense features,
\(C_{\rm feat}=O(hd+kd)\); general sparse features retain an explicit
primitive cost, while sparse identity features reduce to the membership
pattern. Canonical block labels lie in \(\{0,\ldots,n-1\}\), so closure uses a
Boolean lookup table in \(O(n+b_t)\) per level. The induced graph uses the
Boolean edge predicate `scope[row] & scope[col]`, which is one
\(O(s_t)\) pass and preserves the already coalesced COO order. The constructor
and every sparse adjacency update explicitly restore this resident coalescing
invariant. After the
resident adjacency has already been updated, one repaired parent-quotient
update has front-end plus backend time

\[
\begin{aligned}
T_{\rm PQ}={}&C_{\rm sym}(s_t)+r_tC_{\rm SpMV}(s_t,n)\\
&+O\!\left(L(n+b_t)\right)\\
&+\sum_{\ell=0}^{L-1}\Big[
C_{\rm unique}(h_\ell)+C_{\rm sort}(k_\ell)+O(h_\ell+s_t+e_\ell)
+C_{\rm contract}(e_\ell,h_\ell,q_\ell)\\
&\hspace{28mm}+C_{\rm feat}(h_\ell,k_\ell,d,f_\ell)+O(h_\ell)
+C_{\rm unique}(h_\ell)
+T_{\mathcal B}(k_\ell,q_\ell,d)\Big].
\end{aligned}
\]

Here \(C_{\rm contract}\) includes construction of the one-hot membership
pattern and the two sparse products in
\(\bm P_\ell\bm A_t[U_t^\ell]\bm P_\ell^{\mathsf T}\). The first
\(C_{\rm unique}(h_\ell)+C_{\rm sort}(k_\ell)\) covers atom compaction and
minimum-vertex group ordering (including the hierarchical level-0 grouping);
the second \(C_{\rm unique}(h_\ell)\) covers canonicalization of the projected
backend blocks. These primitives are retained explicitly because their
device-specific costs are not assumed linear.

This bound is for the closure-enabled full method and the vertex-level-base
control. The safe radius-only counterfactual has an additional per-level
\(O(n)+C_{\rm unique}(n)\) projection cost: it clones and canonicalizes the
complete stored row so that remainders of boundary-crossing old blocks receive
new, disjoint representatives. Its timing must therefore be interpreted as the
cost of a valid arbitrary-scope update, not as the cost of merely deleting the
closure lookup from the full algorithm.

### Explicit full-scan bound

For an explicit algorithm that uses Boolean edge filtering, comparison sorting
for deterministic label/edge compaction, one-hot membership matrices, and
sequential level processing, the following conservative bound is realizable:

\[
\begin{aligned}
T_{\rm PQ}=O\Bigg(&s_t\log s_t+r_t(s_t+n)+L(n+b_t+s_t)\\
&+\sum_{\ell=0}^{L-1}\Big[
e_\ell\log e_\ell+h_\ell\log h_\ell
+h_\ell d+k_\ell d+q_\ell
+T_{\mathcal B}(k_\ell,q_\ell,d)\Big]\Bigg).
\end{aligned}
\]

This is an algorithmic comparison-model bound, not a claimed guarantee for the
opaque generic `torch.sparse.mm` primitive. If sparse sorting and coalescing
factors are explicitly suppressed, it
reduces to

\[
O\!\left(
(r_t+L)s_t+(r_t+L)n+Lb_t+
\sum_{\ell=0}^{L-1}
\left[e_\ell+h_\ell\log h_\ell+h_\ell d+q_\ell+
T_{\mathcal B}(k_\ell,q_\ell,d)\right]
\right),
\]

where \(k_\ell d\le h_\ell d\) has been absorbed and the feature terms assume
a dense \(h_\ell\times d\) scope. General sparse features must instead retain
\(C_{\rm feat}(h_\ell,k_\ell,d,f_\ell)\). Neighborhood expansion scans the
resident sparse graph at every executed SpMV, canonical closure scans all
\(n\) labels plus the \(b_t\) touched vertices at every level, and rebuilding
each induced level graph scans all \(s_t\) stored adjacency entries. The
repaired algorithm therefore has \(Ls_t\), not
the legacy \(Ls_0\), because it deliberately does not reuse a destructively
shrinking working tensor. Sparse contraction then additionally depends on the
induced \(e_\ell\).

### Mathematical local work versus implementation overhead

With adjacency lists, community-to-member indices, and indexed incident edges,
let \(\mu_r\) be the number of support edges actually traversed by the radius
search and let \(\rho_\ell\) be the incident-edge volume inspected around
\(U_t^\ell\). The corresponding mathematical local work is

\[
O\!\left(
\mu_r+
\sum_{\ell=0}^{L-1}
\left[h_\ell+\rho_\ell+C_{\rm feat}(h_\ell,k_\ell,d,f_\ell)+q_\ell+
T_{\mathcal B}(k_\ell,q_\ell,d)\right]
\right),
\]

up to the selected label and duplicate-edge aggregation strategy. Expected
hash-based aggregation can be linear in \(e_\ell+q_\ell\); a deterministic
comparison model retains an \(e_\ell\log e_\ell\) term. Conditional on
executing \(r_t\) SpMVs and all \(L\) closure/restriction passes, the current
PyTorch tensor organization incurs
\(\Omega(r_ts_t+L(n+s_t))\) full-scan memory traffic, even when
\(h_\ell\), \(k_\ell\), and \(q_\ell\) are small. This distinction prevents
the backend reduction from being misreported as a sublinear end-to-end update
theorem.

### Resident and peak space

The resident state is

\[
M_{\rm resident}=O(s_t+Ln+F).
\]

If all immutable closures are stored and levels are processed sequentially, a
conservative additional-workspace bound is

\[
\begin{aligned}
M_{\rm extra}=O\Bigg(Ln+\max\Big\{&s_t+n,\\
&\max_\ell\big[s_t+e_\ell+h_\ell+q_\ell
+M_{{\rm feat},\ell}+M_{{\rm SpMM},\ell}
+M_{\mathcal B}(k_\ell,q_\ell,d)\big]\Big\}\Bigg).
\end{aligned}
\]

The \(s_t+n\) branch covers transpose/symmetrization and neighborhood vectors;
the per-level branch covers the full edge filter, induced adjacency,
membership pattern, sparse-product workspace, dense feature aggregation, and
backend. For dense features,
\(M_{{\rm feat},\ell}=O(h_\ell d+k_\ell d)\); sparse inputs require their
selected entries and sparse-product workspace to be counted explicitly.
Ignoring hidden sparse-kernel workspace, restricting the simplification to
dense features, and using
\(e_\ell\le s_t\) gives the readable simplification

\[
O\!\left(
Ln+s_t+n+
\max_\ell\left[h_\ell(1+d)+k_\ell d+q_\ell+
M_{\mathcal B}(k_\ell,q_\ell,d)\right]
\right).
\]

### End-to-end additions and break-even condition

The article's principal adapter clock begins after applying the sparse batch
and excludes backend-format conversion and metric evaluation. A complete
end-to-end complexity must additionally include

\[
T_{\rm add}(s_{t-1},\delta_t)+O(n+\delta_t\log\delta_t)
+\sum_\ell C_{\mathcal B}(k_\ell,q_\ell,d)
+O(s_t+n\log n),
\]

for sparse addition/coalescing, affected-endpoint extraction and mask
construction, backend conversion, and the current final-modularity
calculation, respectively.

Under one fixed clock definition, the measurable break-even condition against
one full backend call is

\[
T_{\mathrm{front}}(t)+
\sum_{\ell=0}^{L-1}T_{\mathcal B}(k_\ell,q_\ell,d)
<T_{\mathcal B}(n,s_t,d).
\]

There is no unconditional speed theorem. The profiles already show why: the
backend share is high for expensive Leiden/S\(^{2}\)CAG calls and can be small
for a cheap native dynamic backend. The inequality is meaningful only when
both sides include and exclude the same update, conversion, and metric phases.

## Q1-journal calibration of the contribution

The five experimental questions are not merely headings. Read together, they
form the second scientific pillar of the article:

\[
\text{effect}\longrightarrow\text{generality}\longrightarrow
\text{mechanism}\longrightarrow\text{controls}\longrightarrow
\text{operating envelope}.
\]

This structure is calibrated against full primary texts of closely related
papers in Q1 venues rather than against a subjective preference for more
theorems or more tables.

| Q1 paper | Accepted contribution pattern | Calibration for ComNetX |
|---|---|---|
| [DynaMo](https://doi.org/10.1109/TKDE.2019.2951419), IEEE TKDE 33(5), 2021 | Eight update-event propositions drive merge/split/no-change decisions; explicit best/worst-case complexity; six real networks, 10,000 synthetic networks, five dynamic baselines, and quality plus runtime evidence | This is the theory-heavy reference point. A formal statement deserves headline status only when it changes or certifies the algorithm. Complexity must be connected to measured local work. |
| [CL-OND](https://doi.org/10.1016/j.neucom.2025.129548), Neurocomputing 626, 2025 | Four explicit RQs test the spectral premise, overall performance, contrastive mechanism, and generality; one controlled synthetic suite, two real datasets, seven baselines, and mechanism-specific loss ablations | Q1 work need not be theorem-heavy when each RQ tests a concrete premise and the ablations explain the mechanism rather than select a favorite setting. |
| [Fusion3M](https://doi.org/10.1016/j.inffus.2025.103308), Information Fusion 123, 2025 | Five RQs, seven real datasets, nine baselines, multiple downstream tasks, component/generalization/sensitivity studies, and transfer of the memory module across models | RQ organization and generality are accepted scientific arguments, but “the API runs several backends” is insufficient: transfer must have a measured effect. Only RQ1--RQ3 and the official RQ4/RQ5 section titles are publicly visible; inaccessible wording is not reconstructed as a quotation. |
| [Evolutionary NMF](https://doi.org/10.1109/TKDE.2017.2657752), IEEE TKDE 29(5), 2017 | Nontrivial equivalences connect several temporal-smoothness formulations and enable a new algorithm; synthetic/real tasks, event/noise sensitivity, and scalability support the derivation | Standard quotient algebra does not reach this novelty bar. The boundary certificate and hierarchy-preserving construction are stronger candidates because they delimit or correct the actual method. |
| [CoD\AE N](https://doi.org/10.1145/3718988), ACM TWEB 19(3), 2025 | No theorem package: the headline contribution is a reproducible benchmark with nine transformations and correctness, stability, and delay metrics; independent graph instances, repeated runs, bootstrap intervals, sensitivity, scalability, and real graphs | A controlled empirical program can itself be a contribution. Its lesson is to expose trade-offs and the absence of a universal winner rather than manufacture a leaderboard. TWEB is Q1 in the JCR Software Engineering category and Q2 in Information Systems, so it is used with this category qualification. |

The comparison yields three admissible profiles for a strong journal article:
a theory-driven method such as DynaMo, a premise-and-RQ-driven method such as
CL-OND, and an evaluation-methodology contribution such as CoD\AE N. ComNetX
should combine the first two without imitating either superficially: a compact
method-specific correctness and resource analysis, followed by a five-RQ
empirical evaluation of effectiveness, transfer, mechanism, and limits.

The present empirical breadth is already comparable to CL-OND and Fusion3M.
The main weakness relative to DynaMo and CoD\AE N is not the number of datasets
but independent repetitions, systematic regimes, and uncertainty reporting.
Within the 72-hour budget, controlled DSBM seeds, paired fixed-input repeats,
two long trajectories, and an explicit break-even analysis improve that weak
point more than another family of expensive baseline runs.

## Two-part claim-to-evidence structure for the revision

The journal article should not force the formal analysis into the role of an
experimental research question. Its scientific argument has two complementary
parts:

1. an **analytical characterization of the proposed method** showing that the
   returned state is a valid complete nested partition and giving worst-case time and memory
   bounds, while using quotient and boundary algebra only as supporting detail;
2. an **empirical characterization structured by five research questions**,
   retained from the ICDM design, that tests effectiveness, generality,
   mechanism, causal components, and the operating envelope.

The analytical part precedes the evaluation. Exact quotient algebra, minimal
closure, canonical labels, associativity, and the break-even inequality are
supporting foundations rather than novelty claims. The main analytical
proposition concerns correctness of the complete hierarchy update; the useful
quantitative result is
the explicit time and memory envelope, including graph-wide degeneration and
the current implementation's full scans. The boundary mismatch and its
restricted certificate are retained in an appendix because no empirical
non-vacuity claim is made. A deterministic invariant audit validates the
implementation and is not advertised as a sixth RQ.

### RQ1: Does ComNetX preserve community quality while reducing update time
relative to full-snapshot recomputation?

- Keep the six short Leiden streams and the two 500-update trajectories.
- Treat modularity/NMI and like-for-like clock definitions jointly; neither a
  speedup without quality nor quality without cost answers the question.
- Explain outcomes using the measured front-end/backend split and the
  break-even inequality rather than presenting a leaderboard alone.

### RQ2: Can the same adaptation improve or complement different community
backends, including a specialized dynamic method?

- Keep the multi-backend evidence for topology-only and feature-aware methods,
  with each protocol boundary stated locally.
- Add native LD-Leiden as a specialized reference. This tests the
  specialization--reuse trade-off, not whether a Python adapter must beat a
  solver-specific C++ implementation.
- A faster LD-Leiden result remains scientifically useful if it quantifies the
  price of backend reuse and separates implementation specialization from the
  adapter's broader applicability.

### RQ3: How much of the graph is inspected after neighborhood expansion,
community closure, and quotient contraction?

- Keep neighborhood growth and make the radius-to-closure-to-quotient funnel
  the central mechanism table.
- Report \(b_t,h_\ell,e_\ell,k_\ell,q_\ell\) beside phase times so the
  asymptotic quantities are connected to observed work.
- Distinguish the smaller backend instance from the present full-scan tensor
  overhead; do not call the complete implementation sublinear.

### RQ4: Which components and controls determine the quality--time trade-off?

- Retain the radius, hierarchy depth, topology/feature mode, closure, and
  contraction studies rather than collapsing them into one aggregate ablation.
- Use the repaired labels **radius-only scope** and **vertex-level base atoms**
  for the two mechanism controls. Vertex-level base atoms retain the complete
  community closures and therefore remain covered by the hierarchy theorem.
  Radius-only scopes are not partition-saturated and are **not** covered by
  that theorem: their safe control path temporarily separates the inside
  blocks, splits every old boundary-crossing block, and canonicalizes the full
  row. Evaluate it as a paired one-step counterfactual from the production
  pre-update state, then discard it and advance the persistent trajectory only
  with the valid full method. This avoids both namespace collisions and
  path-dependent contamination while making the loss of community-saturated
  locality explicit.
- Interpret each intervention through the search space or work quantities it
  changes, not merely through a better/worse final score.

### RQ5: How stable is the method across update streams, long horizons, and
large affected regions, and where does its useful operating envelope end?

- Keep repeated short runs, 500-update trajectories, directed/update-type
  controls, and the controlled DSBM update-rate study.
- Distinguish repeated execution on a fixed stream from independent-stream or
  seed robustness; only the latter supports distributional stability claims.
- Report regime boundaries constructively through the break-even condition and
  affected-region growth. A failure regime is evidence about applicability,
  not a defect to hide or an invitation to overstate universality.

Together the five questions form a mechanistic evaluation program: RQ1 tests
the practical consequence, RQ2 tests generality, RQ3 exposes the mechanism,
RQ4 identifies its causal controls, and RQ5 tests robustness and boundary
conditions. Their scientific value comes from this nonredundant coverage and
from answering each question with predeclared metrics and matching evidence,
not from the number five itself.

### Claim-adjudication rules

The five questions are retained independently of whether their answers favor
ComNetX. They are judged by the following predeclared rules rather than by a
post-hoc selection of attractive cells:

- **RQ1:** report paired quality differences together with optimizer, update,
  conversion, and end-to-end clocks. Fixed-stream repetitions support timing
  repeatability only. Do not use “quality preservation” as a binary conclusion
  unless a scientifically justified non-inferiority margin is declared before
  inspecting the repaired results; otherwise report the measured Pareto
  trade-off directly.
- **RQ2:** include the LD-Leiden result regardless of rank. A faster specialized
  C++ implementation measures the specialization--reuse gap; it does not
  falsify backend reuse. Treat S\(^{2}\)CAG as an interface/compatibility case
  unless full and local calls use matched cluster-count and initialization
  semantics.
- **RQ3:** require profile/production parity within $10^{-6}$ in final $Q$
  and NMI, then report every level's $b,h,e,k,q$, phase times, conversion,
  backend share, and memory. Promote the boundary ranking certificate only if
  its measured comparison with the identity atom partition is informative in
  a non-negligible share of eligible levels; report the raw count and
  denominator even when it is vacuous.
- **RQ4:** estimate closure and base-atom effects as paired one-step
  counterfactuals from the same pre-update state. The radius-only result is
  discarded before advancing the production trajectory. Parameter grids and
  feature modes remain supporting sensitivity evidence, not causal substitutes
  for these interventions.
- **RQ5:** use paired independent DSBM seeds for uncertainty and same-rate
  contrasts between update types. Long fixed streams establish persistence on
  those two retained windows, not arbitrary-horizon or independent-stream
  robustness. An operating-boundary claim requires valid regimes on both sides
  of break-even; otherwise report robustness within the observed regime.

All legacy multilevel smart values remain blocked from quantitative claims
until the repaired campaign regenerates them. Raw legacy artifacts remain
archived for provenance and are not relabeled as repaired evidence.

## Pre-registered LD-Leiden measurement

The comparison is useful because LD-Leiden is the closest specialized method
and its omission would be more conspicuous than an expected speed loss. It does
not test whether ComNetX is the fastest implementation of dynamic Leiden. It
tests the specialization--reuse trade-off.

Before measurement, the native launcher must split LD-Leiden timing into:

1. internal update-application time;
2. optimization time excluding that update application (the article's
   principal backend clock); and
3. end-to-end wall time from submitting the batch through completion.

The present launcher records `apply()` as one number even though LD-Leiden
1.3.3 can return `(update_ms, run_ms)`. Historical direct LD-Leiden values are
therefore not admissible for the new principal comparison.

Fixed protocol:

- LD-Leiden with one worker, because this is the current default and matches
  the method's primary single-worker comparison;
- the same undirected representations, initial Leiden partition, update chunks,
  and final common modularity/NMI computation as the paired rows;
- one `999:10` smoke run on all six graphs;
- five fixed-input repetitions on dyn_pubmed and arxivmath;
- three `9:500` repetitions on dyn_pubmed and arxivmath;
- record package/build version, machine, worker count, actual update count,
  cumulative split clocks, end-to-end time, final modularity, and NMI.

Inclusion is independent of outcome:

- if LD-Leiden is much faster, report the specialized solver as the lower-cost
  point and quantify the price paid for backend reuse;
- if ComNetX is close, report that the general adapter approaches the
  specialized reference in this regime;
- if quality differs, discuss the distinct update/search spaces rather than
  treating runtime alone as a ranking.

Stop conditions:

- no result is accepted unless exactly 10/500 updates are processed;
- stop if the installed wheel cannot expose split timing;
- investigate graph representation if internal and common final modularity
  differ by more than \(10^{-4}\);
- use cumulative 500-update time rather than per-update ratios when millisecond
  rounding produces zeros;
- repeat anomalous cells; target a cumulative-time coefficient of variation at
  or below 5% for the three long runs;
- stop a single two-graph long repeat after one hour and diagnose the build
  rather than consuming the full budget.

## 72-hour allocation

| Window | Work | Go/no-go result |
|---|---|---|
| 0--4 h | Split-clock patch, unit tests, version capture, LD smoke test | No measurement if split timing or priming is invalid |
| 4--10 h | Short suite and invariant/quotient audits | Freeze protocol after record-count and representation checks |
| 10--20 h | Three 500-update LD repeats on the two principal graphs | Stop any run exceeding one hour and diagnose |
| 20--36 h | Re-run only anomalous cells; optional one-worker sensitivity checks tied to a manuscript claim | No new exploratory baseline family |
| 36--60 h | Validate, reconcile clocks, generate claim-level summaries | Include outcomes regardless of ranking |
| 60--72 h | Freeze measurements, update manuscript, render and cross-check all numbers | No new hypothesis after the freeze |

The expected LD-Leiden runtime is minutes to a few hours. Historical queues
show that Full Leiden and S\(^{2}\)CAG, not native LD-Leiden, consumed the long
wall-clock budget.

## Retention decisions after the argument is complete

No current artifact is unconditionally useless.

| Artifact | Scientific role | Current decision |
|---|---|---|
| Method workflow | Defines closure, contraction, projection, and stored state | Keep; distinguish the legacy counterexample from the repaired hierarchy theorem and state its assumptions |
| Dataset/protocol tables | Reproducibility and scope | Keep |
| Six-stream Leiden table | Main applied quality/time result | Keep |
| Neighborhood growth | Explains radius cost | Keep as supporting evidence |
| Closure/contraction profile and ablation | Direct mechanism evidence | Keep and elevate |
| Topology grid | Exploratory sensitivity of \(L,r\) | Keep as supporting evidence; do not treat as hierarchy proof |
| Resolution and direction controls | Robustness of the main claim | Keep |
| 500-update table and trajectory | Persistence/drift evidence | Keep |
| DSBM table | Controlled break-even test | Keep |
| DSBM figure | Seed-range detail partly overlaps the table | Decide only after page layout; appendix candidate |
| Cut metrics | Independent structural diagnostic | Keep if space permits |
| Backend table | Reuse and backend-dependent overhead | Keep; add specialized LD reference separately |
| Feature-mode table | Interface coverage with strong confounds | Appendix candidate, not a central quality ranking |

An artifact should be deleted only if, after the claim map and page layout are
complete, it supplies neither a distinct claim nor necessary reproducibility
context. Ambiguous cases move to supplementary material rather than being
discarded.
