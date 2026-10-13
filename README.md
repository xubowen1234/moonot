# MoonOT

MoonOT is a pure MoonBit library for regularized discrete optimal transport on finite supports. It provides balanced and KL-unbalanced log-domain Sinkhorn solvers, a debiased Sinkhorn divergence, and a fixed-support regularized barycenter. The numerical scope is independent and bounded, informed by the mathematical conventions of [Python Optimal Transport (POT)](https://github.com/PythonOT/POT); MoonOT is not a full POT port and does not copy upstream implementation code.

> **Development status:** pre-release and under active development. Balanced and KL-unbalanced transport, Sinkhorn divergence, and fixed-support regularized barycenters are implemented with explicit objective conventions, structured validation/numerical errors, and iteration diagnostics. This pre-release README makes no measured performance or external compatibility claim; native test execution and a public release are not claimed.

## Current capability

- Balanced entropy-regularized transport with log-domain scaling and equal total-mass validation (no implicit normalization).
- KL-unbalanced transport with separate source/target marginal penalties, a product-reference coupling KL, and unequal input totals.
- Debiased Sinkhorn divergence using explicit cross/source-self/target-self cost matrices.
- Fixed-support regularized barycenters over a shared square cost matrix, with strictly positive internally normalized weights and an output total equal to the common input total.
- Rectangular dense cost matrices, finite nonnegative masses/costs, and zero-support handling.
- Optional pairwise squared-Euclidean, Euclidean, and Manhattan cost builders for finite point supports; callers can still provide costs from any domain-specific metric.
- Structured errors, iteration-limit status, objective components, marginal residuals, and log-scaling residual for the unbalanced solver.

The balanced objective is

`<P,C> + epsilon * sum_ij P[i,j] * (log(P[i,j]) - 1)`,

with the entropy summand defined as zero when `P[i,j] = 0`.

The balanced divergence subtracts half of each self-transport objective from the cross-transport objective. Its value is not promised nonnegative unless callers provide suitable mutually consistent ground costs. For common input total `M`, the fixed-support barycenter objective is `min_{x >= 0, sum(x) = M} sum_k lambda_k * OT_epsilon(x, a_k; C)`, where positive input weights are normalized to `lambda_k`. The result preserves `M` (it is a probability vector only when `M = 1`); input totals must agree within the documented relative tolerance. Entropic regularization can bias it away from the unregularized barycenter, even for a single measure. Its log-barycenter update residual determines outer convergence; the reported transport residual is diagnostic, and neither is an optimality-gap or bias bound. See [the numerical contract](docs/algorithm-contract.md) for exact equations, validation rules, termination semantics, and scope boundaries; see [the bounded duplication review](docs/competition/duplicate-check.md) for recorded ecosystem checks.

The KL-unbalanced objective is

`<P, cost_weight*C> + epsilon*KL(P || a⊗b) + rho_source*KL(P1 || a) + rho_target*KL(Pᵀ1 || b)`,

where `KL(x || y) = sum(x*log(x/y) - x + y)` with explicit zero-mass boundary conventions. See [the numerical contract](docs/algorithm-contract.md) for equations, input rules, convergence semantics, and scope boundaries; see [the bounded duplication review](docs/competition/duplicate-check.md) for recorded ecosystem checks.

## Point-cloud cost construction

The core solvers accept caller-supplied costs. For finite point clouds with a shared coordinate dimension, the package also provides three rectangular builders:

```moonbit
let source_points = [[0.0, 0.0], [3.0, 4.0]]
let target_points = [[0.0, 2.0], [3.0, 4.0]]
let squared = @moonot.squared_euclidean_cost_matrix(source_points, target_points)
let euclidean = @moonot.euclidean_cost_matrix(source_points, target_points)
let manhattan = @moonot.manhattan_cost_matrix(source_points, target_points)
```

Rows correspond to source points and columns to target points. Inputs are validated for empty or ragged supports, dimension mismatch, non-finite coordinates, and unrepresentable results. Euclidean accumulation uses a scaled hypot operation; squared-Euclidean and Manhattan costs return a typed overflow error when their mathematically defined result cannot be represented as a finite `Double`.

## Balanced example

```moonbit
let result = @moonot.balanced_sinkhorn(
  [0.5, 0.5],
  [0.5, 0.5],
  [[0.0, 1.0], [1.0, 0.0]],
  { epsilon: 0.25, tolerance: 1.0e-9, max_iterations: 1000 },
)
match result {
  Err(error) => println(error)
  Ok(solution) => {
    println(solution.regularized_objective)
    println(solution.marginal_residual)
  }
}
```

## Runnable workflow examples

Each example is a real MoonBit executable package:

```sh
moon run examples/point-cloud-transport
moon run examples/ml-reweight
moon run examples/color-histogram
moon run examples/divisible-allocation
moon run examples/unbalanced-observations
moon run examples/fixed-support-barycenter
```

They demonstrate coordinate-derived point-cloud costs, distribution reweighting, caller-prepared color histograms, divisible supply allocation, asymmetric missing/extra observation mass, and fixed-support weighted histogram aggregation. Image decoding, machine-learning estimators, and integer optimization remain the caller's responsibility.

## Verification

From the repository root:

```sh
moon info
moon fmt --check
moon check --deny-warn
moon test --deny-warn
moon run cmd/main
```

The configured CI workflow checks generated interfaces and formatting, then runs check, build, test, and each runnable example for wasm-gc, wasm, JavaScript, and native targets. Workflow configuration does not imply that a remote CI run has passed. A local native test is not claimed when the host lacks a native C toolchain.

## Boundaries

MoonOT does not implement a generic linear/integer programming solver, indivisible assignment, route planning, image decoding, automatic differentiation, GPU kernels, or a machine-learning framework. Couplings represent divisible mass plans. Domain applications can use the built-in point metrics or supply domain-specific costs, then interpret transport outputs.

## Development and provenance

See `THIRD_PARTY_NOTICES.md` for upstream provenance and license boundaries, `AI_USAGE.md` for AI assistance disclosure, and `CHANGELOG.md` for release history.

## License

Apache-2.0. See [`LICENSE`](LICENSE).


