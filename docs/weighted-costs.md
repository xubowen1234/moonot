# Feature-weighted ground costs

`weighted_squared_euclidean_cost_matrix(source, target, weights)` constructs a
rectangular source-by-target cost matrix with diagonal nonnegative feature
weights. This lets alignment pipelines express physical units, feature
confidence and intentionally ignored dimensions before a transport solve.

Weights are finite and match point dimension; zero weights define a semimetric.
All coordinates are checked even in ignored dimensions. The implementation uses
square-root weights and hypot accumulation and returns a typed error if the
final squared norm is unrepresentable; it does not learn weights, whiten data,
accept a full covariance matrix or promise numerical scale invariance.
