# Changelog

All notable user-visible changes to MoonOT will be recorded here. Entries describe repository capabilities and do not imply publication.

## Unreleased

- Added validated dense-coupling diagnostics for row/column marginals and unregularized transport cost, with shape, numeric-domain, and overflow errors.
- Added conditional barycentric projection from coupling rows to finite target feature vectors, with explicit zero-mass-row and shape errors.
- Exposed balanced transport-cost and entropy-regularizer components alongside the combined objective, with analytic accounting tests.
- Implemented balanced entropy-regularized Sinkhorn transport and KL-unbalanced Sinkhorn transport with explicit objectives and termination/residual diagnostics.
- Implemented balanced Sinkhorn divergence from cross and self-transport objectives, with caller-supplied cost matrices.
- Implemented fixed-support regularized barycenters using strictly positive, internally normalized weights over equal-total input measures; the output preserves their common total mass.
- Added validated rectangular point-cloud cost builders for squared-Euclidean, Euclidean, and Manhattan metrics, with stable Euclidean accumulation and typed shape/numeric errors.
- Added a runnable point-cloud transport example that feeds squared-Euclidean costs to the balanced solver.
- Synchronized the numerical contract and README with barycenter objective, mass/weight constraints, residual semantics, and regularization-bias limits. No release, performance, or external compatibility claim is made.
