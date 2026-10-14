# Dense coupling entropy and generalized KL

These diagnostics inspect generic dense coupling plans; they do not solve a
transport problem. They accept arbitrary finite nonnegative masses, without
normalizing them or enforcing balanced marginals. Both functions read their
inputs without changing either the outer arrays or any rows, including on errors.

## API and conventions

```moonbit
coupling_entropy(P : Array[Array[Double]])
  -> Result[Double, CouplingInformationError]

coupling_kl(P : Array[Array[Double]], Q : Array[Array[Double]])
  -> Result[Double, CouplingInformationError]
```

### Entropy regularizer

`coupling_entropy(P)` returns the **unscaled regularizer**

```text
sum_ij P[i,j] * (log(P[i,j]) - 1).
```

All logarithms are natural. A zero entry contributes zero, including signed
zero. This is not Shannon entropy: the sign and the `-P` term are intentional.
The result may be negative. For example, `[[0, 1], [2, 1]]` returns
`2*log(2) - 4`, not `-2*log(2)`.

To reconstruct the balanced solver's entropy contribution, multiply this
result by its `epsilon`. This multiplication is outside the diagnostic and
must be checked separately for overflow.

### Generalized KL (I-divergence)

`coupling_kl(P, Q)` returns

```text
sum_ij (p * log(p/q) - p + q), where p=P[i,j], q=Q[i,j].
```

The two plans must have **exactly the same numbers of rows and columns**.
Equal total mass is not required. The diagnostic is asymmetric and is
nonnegative; identical valid plans give exactly zero, even if their total mass
would overflow a `Double`.

| Entry pair | Contribution / result |
| --- | --- |
| `p = 0, q = 0` | `0` |
| `p = 0, q > 0` | `q` |
| `p > 0, q = 0` | `Err(ZeroReferenceSupport(row, column))` |
| `p > 0, q > 0` | Finite generalized KL contribution, or typed overflow |

The unsupported positive-mass case is mathematically infinite. The API reports
it as an explicit error rather than returning infinity as a successful `Double`.
For example, `KL([[0]], [[3]]) = 3`, whereas `KL([[3]], [[0]])` is a support error.

This uses the same generalized KL convention as the unbalanced solver, but
compares two already-materialized plans. It does not create the solver's
`a tensor b` reference, compute marginal penalties, or modify solver semantics.

## Validation and errors

Both plans must be non-empty rectangular matrices with at least one column.
Every entry must be finite and nonnegative. **Every finite negative entry is
rejected**, including negative subnormal values; there is no tolerance-based
clipping. Signed zero is accepted.

`CouplingInformationError` is defined in `information.mbt`:

- `InvalidCoupling(CouplingError)`: input `P` failed the existing coupling
  validator. Its empty, ragged, non-finite, and negative-entry errors are
  preserved, with the original zero-based coordinates.
- `InvalidReference(CouplingError)`: input `Q` failed that same validator.
- `RowCountMismatch(p_rows, q_rows)`: both plans are valid but row counts differ.
- `ColumnCountMismatch(p_columns, q_columns)`: valid plans with equal row counts
  have different column counts.
- `ZeroReferenceSupport(row, column)`: positive `P` entry with zero `Q` entry.
- `NumericalOverflow(row, column)`: a term or checked accumulation cannot be
  represented by finite `Double` arithmetic, at the indicated entry.

Validation order is all of `P`, then all of `Q`, then row count, then column
count. Only after validation and shape checks does row-major numerical evaluation
begin. Support and overflow errors during evaluation identify the first
encountered error in that order. Entropy only validates `P`.

## Numerical approach and limits

For positive entries away from equality, the implementation computes
`delta = log(p) - log(q)`, **not** `log(p/q)`. Thus an overflowing or underflowing
ratio is not itself an error. It evaluates the contribution as
`p*(delta - 1) + q`, folding the `-p` term into the multiplication; for instance,
`p=1e308, q=1e307` has finite KL although `p*delta` overflows.

When `abs(p-q) <= q/2`, let `t = (p-q)/q`. Instead of subtracting nearly equal
logarithms or terms, use

```text
KL(p || q) = (p-q) * t * A(t)
A(t) = sum(n=2..infinity, (-t)^(n-2) / (n*(n-1))).
```

The branch checks the bounded relative difference directly, with division guarded
by `abs(p-q) <= q`; it does not rely on a potentially rounded-up subnormal `q/2`.
In this range the subtraction `p-q` is exact for binary floating point
(Sterbenz's lemma). Horner evaluation through `n=64` at `abs(t)<=1/2` leaves a
coefficient tail bounded by `2^-63/(64*65)`, below ordinary Double rounding.
The coefficient is positive throughout the interval; the factors `(p-q)` and
`t` have the same sign. This avoids negative KL from naive cancellation without
clamping a negative answer to zero. It also preserves near-equal differences
when the two rounded logarithms happen to coincide. Exact equality is handled
before logarithms or division.

Both diagnostics use checked Kahan accumulation to retain small contributions
when added to larger ones. Runtime is linear in the number of entries, with at
most 63 fixed series steps per near-equal positive pair, and constant auxiliary
space. No intermediate dense plans or marginal totals are allocated.

Results remain floating-point approximations, not arbitrary-precision values.
Subnormal results are allowed; a positive mathematical divergence below the
representable range may round to zero. Non-finite arithmetic returns a typed
error, not `Ok(NaN)` or `Ok(infinity)`. Entropy's row-major partial sum can overflow
even if hypothetical later cancellation would make an exact final sum finite;
such an intermediate overflow is also an error.

## Local verification

The blackbox tests in `information_test.mbt` cover analytic entropy and KL,
zero/support rules, arbitrary mass, identical plans, near-equal entries at small
and huge scales, series boundaries, extreme ratios, representable subnormal
results, term and sum overflow, invalid/ragged/mismatched plans, strict negative
rejection, non-finite entries, nonnegativity, compensated summation, and input
immutability. They use assertions rather than snapshots.

Run locally without dependency synchronization:

```powershell
moon check --frozen information.mbt
moon test --frozen information_test.mbt
moon test --frozen --target js information_test.mbt
```

Formatting can be limited to the two new source files using `moonfmt -w`.
No existing sources, generated interfaces, dependencies, or solver contracts
need to be edited to use this feature.

## Verification results

Local strict checking (--deny-warn) and all 17 focused tests passed in the
shared workspace on Wasm and JavaScript. A separate temporary local copy of the
tracked root package plus the two new source files passed all 116 root-package
tests. That copy was used to avoid interference from concurrent unrelated edits.
Generated interfaces were not regenerated, and no remote operations were used.
