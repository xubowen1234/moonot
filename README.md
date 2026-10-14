# MoonOT

MoonOT is a pure MoonBit library for regularized discrete optimal transport on finite supports. It provides balanced and KL-unbalanced log-domain Sinkhorn solvers, a debiased Sinkhorn divergence, and a fixed-support regularized barycenter. The numerical scope is independent and bounded, informed by the mathematical conventions of [Python Optimal Transport (POT)](https://github.com/PythonOT/POT); MoonOT is not a full POT port and does not copy upstream implementation code.

> **Scope:** finite-support, dense CPU calculations with explicit failure and iteration-limit results. Public delivery evidence is recorded in `docs/competition/delivery-audit.md`; a residual or convergence flag is not an unregularized optimality certificate.

## Current capability

- Balanced entropy-regularized transport with log-domain scaling and equal total-mass validation (no implicit normalization).
- KL-unbalanced transport with separate source/target marginal penalties, a product-reference coupling KL, and unequal input totals.
- Debiased Sinkhorn divergence using explicit cross/source-self/target-self cost matrices.
- Fixed-support regularized barycenters with either one shared square cost matrix or one square ground-cost matrix per measure; weights are strictly positive and normalized internally, and the output preserves the common input total.
- Rectangular dense cost matrices, finite nonnegative masses/costs, and zero-support handling.
- Optional pairwise squared-Euclidean, Euclidean, and Manhattan cost builders for finite point supports; callers can still provide costs from any domain-specific metric.
- Balanced objective components and structured termination diagnostics; unbalanced objective components, marginal diagnostics, and log-scaling residual.

The balanced objective is

`<P,C> + epsilon * sum_ij P[i,j] * (log(P[i,j]) - 1)`,

with the entropy summand defined as zero when `P[i,j] = 0`.

The balanced divergence subtracts half of each self-transport objective from the cross-transport objective. Its value is not promised nonnegative unless callers provide suitable mutually consistent ground costs. For common input total `M`, the fixed-support barycenter objective is `min_{x >= 0, sum(x) = M} sum_k lambda_k * OT_epsilon(x, a_k; C_k)`. The legacy `regularized_barycenter` uses the same `C` for every measure; `regularized_barycenter_with_costs` accepts a distinct `C_k` for each one. In the latter API, each matrix row indexes the common barycenter support and its columns align with the paired measure. All matrices share the same row ordering and must be finite, nonnegative, and `n × n`; symmetry and metric properties are not required. Epsilon is shared, and matrices are not averaged or independently rescaled. Positive input weights are normalized to `lambda_k`. The result preserves `M` (it is a probability vector only when `M = 1`); input totals must agree within the documented relative tolerance. Entropic regularization can bias the result away from the unregularized barycenter. The log-barycenter update residual determines outer convergence; the reported transport residual is diagnostic, and neither is an optimality-gap or bias bound. See [the numerical contract](docs/algorithm-contract.md) for equations, validation rules, termination semantics, and scope boundaries; see [the bounded duplication review](docs/competition/duplicate-check.md) for recorded ecosystem checks.

The KL-unbalanced objective is

`<P, cost_weight*C> + epsilon*KL(P || a⊗b) + rho_source*KL(P1 || a) + rho_target*KL(Pᵀ1 || b)`,

where `KL(x || y) = sum(x*log(x/y) - x + y)` with explicit zero-mass boundary conventions. See [the numerical contract](docs/algorithm-contract.md) for equations, input rules, convergence semantics, and scope boundaries; see [the bounded duplication review](docs/competition/duplicate-check.md) for recorded ecosystem checks.

## Heterogeneous-cost barycenter

Use `regularized_barycenter_with_costs` when each input distribution has its own matching geometry but all distributions map to one shared, fixed barycenter support. For example, independently calibrated domains can keep their own costs while contributing to one reference distribution:

```moonbit
let measures = [[1.0, 0.0], [0.0, 1.0]]
let costs = [
  [[0.0, 9.0], [2.0, 0.0]],
  [[0.0, 4.0], [3.0, 0.0]],
]
match @moonot.regularized_barycenter_with_costs(
  measures,
  [1.0, 1.0],
  costs,
  { epsilon: 1.0, tolerance: 1.0e-10, max_iterations: 2000 },
) {
  Err(_) => println("invalid measure/cost set or solver configuration")
  Ok(solution) => println(solution.barycenter)
}
```

The cost-matrix index pairs with the measure index. Matrix rows use the common barycenter support ordering; columns use the paired measure ordering. This API does not optimize support locations, accept rectangular costs or infinite forbidden edges, learn costs, or provide a separate epsilon per measure. The shared-cost API remains available and stores a single kernel rather than expanding it into one copy per input.

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

## Coupling diagnostics

Both transport solvers return their dense coupling plan. `coupling_marginals` computes its source-row and target-column totals, while `coupling_transport_cost` computes the unregularized inner product `<P, C>` for the same-shaped plan and cost matrix. These utilities reject empty, ragged, or mismatched shapes and negative or non-finite entries with typed `CouplingError` variants; non-finite accumulated totals, products, or dot products return a numerical-overflow error. They do not add the entropy or KL terms from a solver objective. In an unbalanced solve, the plan's marginals are diagnostics and need not equal the input marginals.

## Coupling barycentric projection

For a coupling `P` and one finite feature vector per target column, `coupling_barycentric_projection` returns the conditional weighted mean at each source row. This turns a point-cloud coupling into a soft source-to-target map without selecting a single target or changing the solver's plan:

```moonbit
let projected = @moonot.coupling_barycentric_projection(
  solution.coupling,
  target_points,
)
```

The target feature matrix must be rectangular and non-empty. Zero-mass source rows are reported as an undefined projection rather than silently filled with a sentinel. A zero-mass target column contributes nothing. Each feature is scaled by its largest active magnitude before accumulation, keeping the convex weighted mean finite for extreme finite coordinates. This operation works for numeric features; it does not decode images or perform categorical/integer assignment.

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
    println(solution.transport_cost)
    println(solution.entropy_regularizer)
    println(solution.regularized_objective)
    println(solution.marginal_residual)
  }
}
```

## Coupling utilities and continuation

- `weighted_squared_euclidean_cost_matrix`: diagonal feature weights, including zero weights; every coordinate must remain finite.
- `balanced_sinkhorn_continuation`: prevalidated strictly decreasing epsilon schedule, independent stage budgets, recorded progress and opaque seeds.
- `coupling_transpose` and `coupling_permute`: fresh validated plans and complete output-to-input index permutations.
- `coupling_entropy` and `coupling_kl`: entropy `sum p(log p-1)` and generalized KL on non-normalized plans, with explicit zero-reference errors.
- `round_balanced_coupling`: row/column clipping and deficit repair; reports actual marginal feasibility, not optimality.
- `coupling_compose`: Markov gluing on compatible bridge marginals; not a direct optimal-transport solve.

Contracts and examples: [feature costs](docs/weighted-costs.md), [continuation](docs/continuation.md), [orientation](docs/orientation.md), [information](docs/information.md), [rounding](docs/rounding.md), [composition](docs/composition.md), and [multi-domain workflow](docs/alignment-pipeline.md). Exact file names and public signatures are listed in `pkg.generated.mbti`.

## Balanced warm starts

For a sequence of related balanced problems, retain the opaque dual-potential seed from one solve and pass it to the next. The seed is expressed in cost units and may be reused at another epsilon; the new solve still applies its own tolerance and iteration limit. Warm starts can reduce iterations for nearby problems, but do not guarantee a speedup for every input:

```moonbit
match @moonot.balanced_sinkhorn_with_warm_start(
  source,
  target_at_coarse_epsilon,
  cost,
  { epsilon: 2.0, tolerance: 1.0e-9, max_iterations: 2000 },
  None,
) {
  Err(_) => println("transport solve failed")
  Ok((coarse, seed)) => {
    println(coarse.iterations)
    match @moonot.balanced_sinkhorn_with_warm_start(
      source,
      target_at_fine_epsilon,
      cost,
      { epsilon: 0.5, tolerance: 1.0e-9, max_iterations: 2000 },
      Some(seed),
    ) {
      Err(_) => println("transport solve failed")
      Ok((fine, _)) => println(fine.iterations)
    }
  }
}
```

`SinkhornResult` remains the common diagnostics type; the opt-in API returns it together with an opaque seed so callers that do not use warm starts keep the ordinary result shape.

## KL-unbalanced warm starts

`unbalanced_sinkhorn_with_warm_start` accepts `None` for a cold solve or an opaque seed returned by an earlier unbalanced solve. It returns the ordinary `UnbalancedSinkhornResult` plus a fresh seed, including when the iteration budget is exhausted. The existing `unbalanced_sinkhorn` signature and result remain unchanged.

Seeds contain private finite log scalings and the epsilon used to produce them. On an epsilon change, the solver converts each initialization by `log_scale_new = (epsilon_old / epsilon_new) * log_scale_old`; same-epsilon reuse copies the scaling directly. Source and target dimensions must match, and callers must preserve what each coordinate means and its ordering. Masses, support, costs, penalties, cost weight, epsilon, and iteration budget may otherwise change: the seed is an initializer, not a cached plan. Newly inactive coordinates and sides with zero marginal penalty start with neutral log scaling. A zero-support problem still returns its zero plan and exact zero-support KL constants; it also returns a neutral seed. Invalid seed structure yields `InvalidWarmStart`; an unrepresentable epsilon conversion is a numerical failure, never a silent cold fallback.

Iterations and scaling residual describe only the current call. The solver always performs a sweep for a nonzero-support problem and re-evaluates the plan and objective using the current inputs. A seed may help or hinder convergence; no iteration reduction or speedup is guaranteed. The runnable `examples/unbalanced-continuation` reports its actual warm and cold counts for its fixed input.

## Runnable workflow examples

Each example is a real MoonBit executable package:

```sh
moon run examples/point-cloud-transport
moon run examples/ml-reweight
moon run examples/color-histogram
moon run examples/divisible-allocation
moon run examples/unbalanced-observations
moon run examples/fixed-support-barycenter
moon run examples/epsilon-scaling
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
