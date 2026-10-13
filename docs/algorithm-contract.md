# Balanced and unbalanced solver contracts

This document defines the observable mathematics and numerical behavior of MoonOT's public solvers. A future solver must document its objective and edge cases here before its API is treated as stable.

## Shared inputs and numerical behavior

- A measure is a finite, nonnegative `Array[Double]`; a cost matrix is a dense rectangular matrix with rows corresponding to source support and columns to target support.
- Empty supports, ragged matrices, non-finite or negative masses/costs, invalid regularization parameters, and non-positive iteration budgets are rejected with structured errors.
- The implementation uses log-domain scaling and stable log-sum-exp. It never relies on materializing the Gibbs kernel as its only numerical representation.
- Returned coupling entries and objective components must be finite and nonnegative where mathematically required. Underflow of negligible entries may produce zero; overflow or invalid arithmetic returns `NumericalFailure` rather than a non-finite result.
- Iteration exhaustion is returned as `IterationLimit`, distinct from convergence. Results expose iteration count and residual diagnostics; diagnostics are not silently replaced by a success claim.

## Balanced entropic transport

Given source masses `a`, target masses `b`, and costs `C`, find a nonnegative coupling `P` with row sums `a` and column sums `b`. Total masses must agree up to the documented relative floating-point input-rounding tolerance; they are not normalized implicitly.

The minimized objective is

`sum_ij P[i,j] * C[i,j] + epsilon * sum_ij P[i,j] * (log(P[i,j]) - 1)`.

The entropy summand is zero for `P[i,j] = 0`, and `epsilon` is positive. The `-P` term is intentional: omitting it changes reported objective values by `epsilon * total_mass`, though not the optimizer.

Zero-mass rows and columns receive zero coupling. An all-zero problem is valid and returns a zero plan, zero objective, and zero marginal residual. The result reports maximum absolute row/column marginal error, iteration count, and `Converged` or `IterationLimit`.

## KL-unbalanced entropic transport

The unbalanced solver allows unequal source and target totals. For nonnegative finite `rho_source` and `rho_target`, positive finite `epsilon`, and nonnegative finite `cost_weight`, it minimizes

`<P, cost_weight * C> + epsilon * KL(P || a ⊗ b)`

`+ rho_source * KL(P 1 || a) + rho_target * KL(Pᵀ 1 || b)`.

Here `KL(x || y) = sum_k (x_k * log(x_k / y_k) - x_k + y_k)` is the generalized I-divergence, including its boundary conventions `KL(0 || 0) = 0`, `KL(0 || y) = y` for `y > 0`, and `KL(x || 0) = +∞` for `x > 0`. The coupling reference `a ⊗ b` therefore forces rows/columns corresponding to zero input mass to remain zero. A zero total on either side has the zero coupling; objective terms and unmatched positive reference mass remain explicit in the returned KL components.

The unbalanced solver uses asymmetric generalized scaling exponents `rho / (rho + epsilon)` for the source and target updates. It reports the objective and its four components, maximum absolute differences from the supplied marginals (diagnostics only), maximum log-scaling update, iteration count, and termination status. No normalization or balanced marginal constraint is implied.

## Dense coupling diagnostics

A coupling plan is a non-empty rectangular dense matrix `P` of finite, nonnegative `Double` entries. Zero entries are valid. `coupling_marginals(P)` returns the row totals as the source marginal and the column totals as the target marginal, preserving that orientation. It does not assert that either total matches any input measure. If either accumulated total is not representable as a finite `Double`, the call returns `CouplingError::NumericalOverflow` at the cell where the overflow is detected.

`coupling_transport_cost(P, C)` requires a non-empty rectangular finite, nonnegative cost matrix with exactly the same row and column counts as `P`; it returns the unregularized dot product `sum_ij P[i,j] * C[i,j]`. A non-finite product or accumulated dot product returns `NumericalOverflow`. It does not compute or approximate entropy, reference KL, marginal penalties, a regularized objective, or an optimality certificate. These utilities can inspect plans returned by balanced or unbalanced solvers; in an unbalanced solve, the computed row and column totals are not required to match the supplied marginals.

## Conditional barycentric projection

`coupling_barycentric_projection(P, Y)` accepts a coupling `P` and a rectangular target feature matrix `Y` with one non-empty feature vector per coupling column. All target values must be finite; feature vectors must have a common dimension. For source row `i` with mass `r_i = sum_j P[i,j] > 0`, it returns `T[i,:] = sum_j (P[i,j] / r_i) * Y[j,:]`. Thus each output is a conditional mean in target-feature space and is invariant to positive rescaling of a coupling row. A zero-mass source row returns `CouplingProjectionError::ZeroMassSourceRow(i)` because its conditional mean is undefined. A zero-mass target column has zero weight and contributes nothing. For each feature, the implementation scales active values by their largest absolute magnitude before weighted accumulation and rescales the result, preserving the convex-hull bound for finite inputs. This is a soft numeric projection, not an integer assignment or a guarantee of an inverse/one-to-one map.

## Sinkhorn divergence

The debiased value uses the same balanced entropic objective for cross and self problems:

`S_epsilon(a,b) = OT_epsilon(a,b; C_ab) - 0.5 * OT_epsilon(a,a; C_aa) - 0.5 * OT_epsilon(b,b; C_bb)`.

Callers provide all three cost matrices. Self-cost matrices must be square and symmetric. A nonnegative divergence additionally depends on suitable ground-cost assumptions; the API does not claim nonnegativity for arbitrary symmetric matrices.

## Fixed-support regularized barycenter

Let `a^(k)` be input measures on one support, let each supplied weight `w_k` be finite and strictly positive, and define `lambda_k = w_k / sum_l w_l`. With common total mass `M` and shared square cost matrix `C`, the intended objective is

`min_{x >= 0, sum_i x[i] = M} sum_k lambda_k * OT_epsilon(x, a^(k); C)`,

where `OT_epsilon` is the balanced objective above and `x` is the source marginal. Weights are normalized internally. Input totals must agree within relative tolerance `1.0e-12 * max(total_k, M)`; masses are not normalized. The output barycenter preserves `M`, so it is a probability vector only when `M = 1`. If `M = 0`, the result is the zero barycenter with zero objective/residuals, zero iterations, and `Converged`.

All measures must have the same nonempty support size. `C` must be finite, nonnegative, and square for that support; symmetry is not required. The API rejects zero or negative weights. `weighted_objective` is evaluated after the outer iteration as the normalized weighted sum of balanced-solver objectives at the returned barycenter. `maximum_transport_residual` is the largest marginal residual among those evaluation solves.

`log_barycenter_residual` is the maximum coordinatewise absolute change in the log barycenter between successive outer iterations. The outer loop reports `Converged` when that update is at most `tolerance`; if the outer iteration budget is exhausted, or an objective-evaluation transport solve reaches its own iteration limit, the result reports `IterationLimit`. The transport residual is diagnostic, not the outer stopping criterion. Neither residual is an objective-gap certificate or a bound on distance to an exact optimizer.

Entropy regularization changes the target relative to unregularized transport: the regularized barycenter need not equal an unregularized Wasserstein barycenter, and even a single input measure need not be returned unchanged. No quantitative regularization-bias bound or optimality-gap certificate is provided.

## Validation oracle

Tests use analytic single-cell and tiny matrix solutions; input/shape/zero-mass and iteration-limit cases; permutation and symmetry checks where assumptions apply; objective recomputation from the returned plan; and difficult finite scale/regularization cases. Comparisons against Python Optimal Transport (POT) are valid only after aligning objective conventions. Printed snapshots are never the sole correctness oracle.
