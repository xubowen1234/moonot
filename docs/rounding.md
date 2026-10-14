# Balanced coupling feasibility rounding

`round_balanced_coupling(coupling, source, target, tolerance)` rounds a dense,
nonnegative plan toward caller-supplied balanced marginals. It returns
`Result[CouplingRoundingResult, CouplingRoundingError]`. The result contains:

- `coupling: Array[Array[Double]]`: a freshly allocated corrected plan;
- `marginal_residual: Double`: the actual maximum absolute row or column
  marginal error against the **original** supplied marginals.

This is a **feasibility operation only**. It does not compute an optimal plan,
minimize a transport cost, or certify Sinkhorn convergence. No costs or epsilon
are needed, and inputs are never secretly normalized.

```moonbit
let outcome = @moonot.round_balanced_coupling(
  [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
  [0.25, 0.75],
  [0.125, 0.375, 0.5],
  1.0e-12,
)
```

## Input and tolerance contract

The plan must be nonempty and rectangular, with at least one column and only
finite nonnegative entries. The source length must equal the number of rows;
the target length must equal the number of columns. Every marginal mass must be
finite and nonnegative. Marginal totals are summed in index order using Double
arithmetic; an overflowing total is an error.

Let `A = sum(source)` and `B = sum(target)`. Both zero is valid. Otherwise the
balanced-total requirement is:

```
abs(A - B) / max(A, B) <= 1.0e-12
```

There is no absolute scale floor. In particular, tiny but relatively different
totals are rejected. The quotient formulation avoids underflowing the threshold
`1.0e-12 * max(A, B)` for extremely small totals.

The caller's `tolerance` must be finite and strictly positive. It is an
**absolute mass tolerance**, not relative tolerance. The final measured maximum
marginal residual must be at most this tolerance. Passing the balanced-total
check does not guarantee passing the final feasibility check: small differences
between the supplied totals and floating-point roundoff can still matter.

Validation order is plan shape/entries, source length, target length, tolerance,
source masses/total, target masses/total, then balanced-total agreement. Even an
all-zero-marginal request validates the entire input plan; NaN, infinity, and
negative entries are not accepted on the zero-total path.

## Algorithm

The driver works only on a deep copy of the plan:

1. For each row with positive sum `r_i`, multiply it by
   `min(source[i] / r_i, 1)`. Leave a zero row unchanged.
2. For each column of that row-clipped plan with positive sum `c_j`, multiply it
   by `min(target[j] / c_j, 1)`. Leave a zero column unchanged.
3. Recompute the clipped marginals. Form nonnegative deficits
   `u_i = max(source[i] - row_sum[i], 0)` and
   `v_j = max(target[j] - column_sum[j], 0)`.
4. Set `D = max(sum(u), sum(v))`. If `D > 0`, add the nonnegative rank-one
   correction `u_i * v_j / D` to every entry. If `D == 0`, add nothing.
5. Recompute checked row/column sums of the corrected plan. Return success
   **only** when the actual maximum residual against the original marginals is
   at most `tolerance`.

For exactly balanced real arithmetic, both deficit totals agree, and this is
the usual rank-one deficit repair that makes both marginals exact. With
near-equal input totals or floating-point differences, using the larger deficit
total prevents the repair from intentionally exceeding either deficit vector.
Any remaining error is exposed by the final check, not hidden by changing the
marginals. Deficits are clamped only to maintain a nonnegative correction;
small clipping overshoots are not excused by the reported residual.

If both marginal totals are zero, validation is followed by a fresh all-zero
plan with residual zero. Zero source or target entries produce exact zero rows
or columns. A feasible plan is unchanged in exact arithmetic; repeated rounding
in floating-point arithmetic is expected to be idempotent within numerical
tolerance, not necessarily bitwise identical.

## Numerical scaling

Clipping does not need to form an overflowing original row/column sum. For a
vector with maximum entry `M > 0`, it computes `s = sum(value / M)`, tests
`M > capacity / s`, and applies the clipped factor `(capacity / M) / s`.
If this factor underflows to zero, it instead computes each clipped entry as
`((value / M) / s) * capacity`. This preserves representable output mass even
when the factor itself cannot be represented. When the factor is representable,
multiplying the original entry by it also preserves tiny entries that would be
lost by always normalizing them against an enormous maximum.

Rank-one repair never forms the product of two deficits first. It normalizes
the **larger** of `u_i` and `v_j` by `D`, then multiplies by the smaller. This
avoids both an overflowing large product and an underflowing tiny product; it
also avoids losing a representable repair by first dividing a tiny deficit by
an enormous total. Terms below the Double representable range can still vanish.
Checked final sums and the caller's residual bound determine whether the result
is usable. There is no guarantee for arbitrarily strict tolerances or extreme
floating-point data.

Running time is O(rows * columns); the returned matrix and temporary vectors
use O(rows * columns + rows + columns) space. Every returned row is independently
allocated. Mutating the returned plan cannot change the input plan or marginals.

## Explicit errors

`CouplingRoundingError` exposes:

- `InvalidCoupling(CouplingError)`: preserves existing plan validation errors;
- `SourceCountMismatch`, `TargetCountMismatch`;
- `NonFiniteSourceMass(index)`, `NonFiniteTargetMass(index)`;
- `NegativeSourceMass(index)`, `NegativeTargetMass(index)`;
- `NonFiniteSourceTotal`, `NonFiniteTargetTotal`;
- `UnequalTotalMass`;
- `InvalidTolerance`;
- `NumericalFailure`: intermediate/final arithmetic or checked sums could not
  produce finite diagnostics;
- `FeasibilityFailure(residual)`: a finite corrected plan's measured residual
  exceeded the supplied tolerance. No successful-feasibility claim is returned.

Indices are zero-based. A finite plan whose unscaled row sums overflow can
still be clipped successfully using scaled arithmetic; this is distinct from
an overflowing marginal total, which is always rejected.
