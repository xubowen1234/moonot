# MoonOT

MoonOT is a MoonBit library for dense, finite-support, entropy-regularized optimal transport. Its public API provides `balanced_sinkhorn`, `unbalanced_sinkhorn`, `balanced_sinkhorn_divergence`, and `regularized_barycenter` for reusable transport plans, debiased comparisons, and fixed-support aggregation.

The package reports objective values, termination states, and numerical residuals with explicit zero-mass and validation semantics. Residuals are diagnostics, not optimality-gap certificates. See the [numerical contract](docs/algorithm-contract.md) and [algorithm derivation](docs/algorithm-derivation.md) for the exact objectives, updates, assumptions, and limits.