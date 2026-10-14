# Balanced epsilon continuation

`balanced_sinkhorn_continuation(source, target, cost, schedule, warm_start)`
solves the same balanced transport problem at each caller-selected epsilon,
using the existing opaque `SinkhornWarmStart` between stages. It does not invent
an epsilon schedule or modify the marginals, costs, options, or initial seed.

## API and schedule contract

The schedule is an `Array[SinkhornOptions]`. Each entry specifies:

- `epsilon`: finite and strictly positive; each subsequent epsilon must be
  strictly smaller than its predecessor.
- `tolerance`: finite and strictly positive, in **absolute mass units**. A stage
  converges when the maximum absolute row or column marginal residual against
  the original supplied marginals is at most this value. It is neither relative
  tolerance nor a dual/objective change threshold.
- `max_iterations`: positive `Int`, independently applied to that stage. One
  iteration is a full row update followed by a full column update. It is not a
  total continuation budget. As in the underlying solver, a nonzero-mass stage
  performs at least one iteration even with a warm seed; an all-zero problem
  converges without iterations.

An empty schedule is invalid; a single epsilon is valid. The complete schedule
is validated before any solver runs. Validation proceeds in schedule order,
checking epsilon validity, strictly decreasing order, tolerance, then iteration
limit at each index. Indices in errors are zero-based. Schedule errors therefore
have precedence over solver input errors, including invalid initial seeds.

```moonbit
let schedule : Array[@moonot.SinkhornOptions] = [
  { epsilon: 2.0, tolerance: 1.0e-6, max_iterations: 20 },
  { epsilon: 0.5, tolerance: 1.0e-12, max_iterations: 1000 },
]
let outcome = @moonot.balanced_sinkhorn_continuation(
  [0.5, 0.5],
  [0.5, 0.5],
  [[0.0, 1.0], [1.0, 0.0]],
  schedule,
  None, // Or Some(seed) from a compatible previous balanced solve.
)
```

Each stage invokes `balanced_sinkhorn_with_warm_start`. Returned seeds are
already in cost units and are passed unchanged; the driver does not expose,
rescale, or reconstruct their potentials. All balanced-solver contracts still
apply: finite nonnegative costs and masses, compatible dimensions, totals
agreeing within the solver's relative `1.0e-12` threshold, and exact zero
rows/columns for zero support masses. No mass normalization takes place.

## Results and convergence

`Ok(SinkhornContinuationResult)` means **every scheduled solve returned a
result**, not necessarily that the final solve converged. Its public fields are:

- `stages: Array[SinkhornContinuationStage]`: one entry per epsilon, in order.
  Each entry contains the supplied `options` and the complete `SinkhornResult`.
- `total_iterations: Int64`: sum of the completed stages' iteration counts.
  A 64-bit count avoids overflowing a 32-bit sum of stage budgets.
- `last_seed: SinkhornWarmStart`: seed returned by the final stage, including
  when that stage reached its iteration limit. It can initialize another solve.
- `termination: SinkhornTermination`: **only the final stage's termination**.

The final plan and objective are `stages[stages.length() - 1].result`; no
objectives from different epsilons are combined. An intermediate
`IterationLimit` is retained in its stage diagnostics, and continuation proceeds
with its seed. Final `IterationLimit` is an ordinary successful-but-unconverged
result, clearly distinguished from final `Converged` and from errors. There is
no speedup or convergence guarantee for an arbitrary schedule and budget.

## Errors and partial progress

`SinkhornContinuationError` exposes these variants:

- `EmptySchedule`
- `InvalidEpsilon(index)` (zero, negative, NaN, or infinity)
- `NonDecreasingEpsilon(index)` (equal to or larger than the previous epsilon)
- `InvalidTolerance(index)` (zero, negative, NaN, or infinity)
- `InvalidIterationLimit(index)` (zero or negative)
- `StageFailed(index, cause, progress)`

`StageFailed` stops immediately. `cause` is the exact underlying `SinkhornError`,
including shape, unequal-total-mass, invalid-warm-start, and numerical errors;
none is converted to `IterationLimit` or silently skipped. Its
`SinkhornContinuationProgress` contains:

- `stages`: only the stages completed **before** the failure;
- `total_iterations: Int64`: only those completed stages' iterations;
- `last_seed: SinkhornWarmStart?`: their last returned seed, or the caller's
  initial optional seed if no stage completed.

The underlying solver does not return partial iterates, seeds, or iteration
counts on an error. Consequently the failing stage's work cannot be reported
and is excluded from progress. A failure can occur at a later, smaller epsilon
because scaling is no longer representable even though earlier stages worked.

The driver only reads caller-owned arrays and never mutates opaque seeds.
Returned arrays are ordinary mutable MoonBit arrays, not immutable snapshots;
stages have separate solver-created coupling arrays. Reusing or modifying a
returned plan does not alter the saved seed. A one-entry run uses the warm-start
solver, so it has that solver's representable-seed requirements, which can be
stricter than the cold solver at extreme floating-point scales.
