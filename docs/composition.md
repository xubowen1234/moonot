# Markov composition of dense coupling plans

## API

```moonbit
coupling_compose(
  left : Array[Array[Double]],
  right : Array[Array[Double]],
  tolerance : Double,
) -> Result[Array[Array[Double]], CouplingCompositionError]
```

`left` has shape **source × bridge**; `right` has shape **bridge × target**.
The returned fresh dense matrix has shape **source × target**. Both plans
must be non-empty rectangular matrices of finite nonnegative entries, and
all their row and column marginals must be representable as finite `Double`s.
Every finite negative entry is rejected, including negative subnormals;
signed zero is accepted. Arbitrary total mass is supported, not just unit mass.

The tolerance must be **finite and strictly positive**, and is an **absolute
per-bridge mass tolerance**, not a relative tolerance or a final marginal
residual tolerance. Zero, negative values, NaN, and either infinity are invalid.

Neither input is changed, rescaled, clipped, or globally normalized, on success
or failure. Returned rows are separately allocated and do not alias either
input or each other.

## Formula and conditional interpretation

Write

```text
ell[j] = sum_i left[i][j]
r[j]   = sum_k right[j][k]
T[j,k] = right[j][k] / r[j]  when r[j] > 0.

composed[i][k] = sum_j left[i][j] * T[j,k].
```

Bridge counts must match and each `abs(ell[j] - r[j])` must be at most the
supplied tolerance, inclusively. Matching only the total mass is not enough:
each bridge coordinate must refer to the same intermediate state, in the
same order, and pass its own compatibility check.

When `ell[j] = r[j]`, this is the source–target marginal of the joint measure

```text
G[i,j,k] = left[i][j] * right[j][k] / r[j].
```

The construction makes source and target **conditionally independent given the
bridge**: source → bridge → target is a Markov chain, and `T` is its second
transition kernel. For exact bridges, the source–bridge and bridge–target
marginals of `G` are the supplied plans.

This selects one particular gluing, not every possible joint measure with
those pairwise marginals. It is **not an optimal transport solver** and does
**not prove optimality** for any cost, entropic objective, or direct
source–target transport problem. No cost matrix is inspected.

### Zero bridge support

- If both bridge masses are zero, that bridge contributes nothing and is
  skipped without division.
- If `ell[j] > 0` but `r[j] = 0`, composition returns
  `UndefinedBridgeConditional(j)` **even when tolerance would accept the mass
  difference**. A right conditional cannot be inferred from an empty row.
- If `ell[j] = 0` and `r[j] > 0`, the bridge is allowed only if its mass
  discrepancy passes tolerance. It contributes zero, so that portion of the
  right plan does not survive in the composed plan.

An all-zero compatible pair is valid and returns a correctly shaped zero plan.

## Marginals and approximate bridge matches

The following identities and bounds describe **exact real arithmetic**.
Floating-point rounding, including underflow, adds numerical error; the
implementation checks finiteness, not a final marginal-residual guarantee.

For every accepted pair, each used right conditional sums to one, so

```text
sum_k composed[i][k] = sum_j left[i][j].
```

The left source marginal and total mass are therefore preserved. The composed
target marginal is

```text
c[k] = sum_{j:r[j]>0} ell[j] * T[j,k],
b[k] = sum_{j:r[j]>0} r[j]   * T[j,k]  (right target marginal).
```

**Exact bridge matches guarantee preservation of both exterior marginals.**
Near matches preserve the left source marginal, but do not generally preserve
the right target marginal or total mass. In particular,

```text
c[k] - b[k] = sum_{j:r[j]>0} (ell[j] - r[j]) * T[j,k].

abs(c[k] - b[k]) <= sum_j abs(ell[j] - r[j]) * T[j,k]
                 <= sum_j abs(ell[j] - r[j])
                 <= bridge_count * tolerance.

sum_k abs(c[k] - b[k]) <= sum_j abs(ell[j] - r[j])
                      <= bridge_count * tolerance.
```

The total-mass discrepancy is likewise bounded by the sum of absolute bridge
mass discrepancies. A target coordinate's error may exceed `tolerance`:
multiple bridge discrepancies can accumulate into the same target.

For approximate bridges the joint measure's bridge–target marginal is
`ell[j] * T[j,k]`, i.e. the right conditional reweighted by left bridge mass.
This is a consequence of the documented formula, **not** a hidden mutation or
normalization of the original right plan. The function does not average bridge
masses or enforce agreement by repairing either input.

### Exact associativity

For three compatible plans `A[source, bridge1]`, `B[bridge1, bridge2]`, and
`C[bridge2, target]` with exact bridge matches, Markov composition is associative
in real arithmetic. Both parenthesizations yield

```text
sum_{j,k} A[i,j] * (B[j,k]/mass_bridge1[j])
                 * (C[k,l]/mass_bridge2[k]).
```

Zero shared states can be omitted. Floating-point results need not agree
bit-for-bit for arbitrary data; the tests use exact dyadic conditionals so
that their three-stage example does agree exactly. Associativity is not
promised for tolerance-accepted chains: reweighting can change a later bridge
compatibility check, so both parenthesizations need not be accepted.

## Typed errors and evaluation order

`CouplingCompositionError` is defined entirely in `composition.mbt`:

| Error | Meaning |
| --- | --- |
| `InvalidTolerance` | Tolerance is not finite and strictly positive. |
| `InvalidLeft(CouplingError)` | Left validation or marginal computation failed. |
| `InvalidRight(CouplingError)` | Right validation or marginal computation failed. |
| `BridgeDimensionMismatch(left_columns, right_rows)` | Bridge counts differ. |
| `BridgeMassMismatch(j, left_mass, right_mass)` | Bridge mass discrepancy exceeds tolerance. |
| `UndefinedBridgeConditional(j)` | Positive left bridge mass and a zero right row. |
| `NumericalOverflow(i, j, k)` | A conditional, product, or accumulated output entry is non-finite. |
| `InvalidOutput(CouplingError)` | Output validation or marginal finiteness check failed. |

Coordinates are zero-based. The existing coupling validator's empty, ragged,
non-finite, negative-entry, and marginal-overflow errors are preserved in their
wrappers.

Order is tolerance validation; all left entries; all right entries; bridge
count; left marginals; right marginals; each bridge's support then mass check
in increasing `j`; numerical composition; and output marginal checks. An
undefined conditional takes precedence over a mass mismatch at the same bridge,
not over an earlier bridge's failure. No partial matrix is returned on failure.

## Numerical behavior

- Each right conditional is computed **before** multiplying by left mass.
  The conditional lies between zero and one. This avoids overflowing
  `left[i][j] * right[j][k]` when its normalized contribution is finite, and
  also avoids many unnecessary tiny-product underflows.
- The code never forms `ell[j]/r[j]`, which can overflow for permissively
  tolerated bridge differences even while the requested composition is finite.
- Output entries use checked Kahan accumulation. Conditional division,
  multiplication, accumulation, and compensation must stay finite.
- The existing `coupling_marginals` validates both input marginals and the
  returned matrix's row and column sums. Finite individual entries do not
  suffice: a target marginal can overflow after accepted bridge reweighting.
  These checks do not require the grand total to be finite when every individual
  marginal is finite.
- Results are finite `Double` approximations, not arbitrary precision. Division
  or multiplication can underflow to zero. In particular, a tiny conditional
  can underflow before multiplication by a large left mass, even if that final
  real-valued contribution would be representable. There is no underflow repair
  or final marginal adjustment. A successful result never contains NaN or infinity.

For `n` sources, `b` bridge states and `m` targets, numerical composition costs
`O(n*b*m)` time. Output storage is `O(n*m)` plus input/output marginal arrays;
compensation uses constant space per output entry. No three-dimensional joint
measure is allocated.

## Examples and local verification

```text
left  = [[1,2,0], [3,0,4]]
right = [[1,1,2,0], [0,1,0,1], [2,0,1,1]]

compose(left, right, 1e-12)
  = [[0.25,1.25,0.5,1], [2.75,0.75,2.5,1]]

source marginal = [3,7]
target marginal = [3,2,3,2]
```

The bridge masses on both sides are `[4,2,4]`. Total mass is ten, not one.
By contrast, `compose([[1.25,1.25]], [[1],[1]], 0.25)` returns `[[2.5]]`,
not `[[2]]`; the target difference is `2 * tolerance`.

Focused blackbox assertion tests cover the rectangle and exact marginal
conservation; three-stage associativity; shared zeros and undefined
conditionals; near-match boundary acceptance and target bounds; dimension and
mass incompatibility; huge and tiny arithmetic; input and output marginal
overflow; tolerance validation; invalid entries and shapes; immutability and
non-aliasing.

Run using local dependencies without synchronization:

```powershell
moon check --frozen --deny-warn composition.mbt
moon test --frozen composition_test.mbt
moon test --frozen --target js composition_test.mbt
```

Format only the new sources with `moonfmt -w composition.mbt` and
`moonfmt -w composition_test.mbt`. No existing files or generated interfaces
need to be changed, and no remote operations are required.
