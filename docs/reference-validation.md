# Independent reference validation

`reference_vectors_test.mbt` supplies dependency-free analytic regression tests.
`scripts/verify_pot.py` executes MoonOT's current public root APIs and compares
couplings **and objective components** with the separately implemented Python
Optimal Transport (POT), pinned to **0.9.6.post1**. This is a finite differential
check, not a proof, a performance benchmark, or a claim of full POT compatibility.
No POT source code or upstream fixtures are copied into these files.

## Reproduce (optional dependencies)

Requires MoonBit on `PATH`, Python 3.10+ (3.12 tested), and a platform supported by
the pinned wheels. Python dependencies are optional; ordinary MoonBit tests do
not import or install Python packages. The harness never installs anything.

PowerShell, from the MoonOT repository:

```powershell
# Use a trusted 64-bit Python 3.12 executable. Keep the venv OUTSIDE the repository.
$Python = 'C:\path\to\python.exe'
$Venv = Join-Path $env:USERPROFILE '.cache\moonot-reference-pot-0.9.6.post1'
& $Python -m venv $Venv
$ReferencePython = Join-Path $Venv 'Scripts\python.exe'
& $ReferencePython -m pip install -r scripts/requirements-reference.txt
& $ReferencePython scripts/verify_pot.py
if ($LASTEXITCODE -ne 0) { throw 'Reference validation failed or is blocked' }
moon test reference_vectors_test.mbt
```

POSIX CI example (make this an optional job, not a mandatory library dependency):

```sh
python3 -m venv "$HOME/.cache/moonot-reference-pot-0.9.6.post1"
PY="$HOME/.cache/moonot-reference-pot-0.9.6.post1/bin/python"
"$PY" -m pip install -r scripts/requirements-reference.txt
"$PY" scripts/verify_pot.py --report "$HOME/moonot-reference-report.json"
moon test reference_vectors_test.mbt
```

The requirements file pins POT, NumPy, and SciPy. These pins are not a fully
hashed, platform-independent lockfile; the report records the actual versions.
Do not install them globally or into a system Python. Existing compatible
private libraries may be reused; the harness enforces the exact POT version,
but records rather than rejects alternative NumPy/SciPy versions.

Options: `--moon /path/to/moon`, `--target wasm|wasm-gc|js|native` (default `wasm`),
and `--report /outside/repo/report.json`. A report path inside the repository is
rejected. Exit **0** means all checks passed; **1** means a numerical, compilation,
runtime, warning, or comparison failure; **2** means missing optional dependencies,
a wrong POT version, or invalid CLI usage. CI must not treat exit 2 as validation
success. Missing POT is reported as `BLOCKED`, not a claimed comparison.

The bridge copies root production `.mbt` files and `moon.mod`/`moon.pkg` to a
system temporary directory, omitting every `_test.mbt` and `_wbtest.mbt`. It
creates a tiny MoonBit executable there, calls `balanced_sinkhorn` and
`unbalanced_sinkhorn`, and parses their returned coupling/components. No solver
implementation is ported to Python or substituted for MoonOT. Compilation,
generated interfaces, and bridge files stay outside the repository; temporary
files are cleaned up. SHA-256 hashes of the copied production inputs are included
in the report. Do not concurrently change those inputs when reproducing a run.

## Exact objective conventions

Write `A = sum(a)`, `B = sum(b)`, `m = sum(P)`, `R = a outer b`, and

- `H(P) = sum(P * (log(P) - 1))`, with zero summands at zero;
- `D(x || y) = sum(x * log(x/y) - x + y)`;
- `D(0 || 0) = 0`, `D(0 || y) = y`, and `D(x || 0) = infinity` for `x > 0`.

### Balanced: entropy, not reference KL

MoonOT minimizes `<P,C> + epsilon H(P)` with the supplied equal-total marginals,
without normalizing them. The harness calls
`ot.solve(C, a, b, reg=epsilon, reg_type='entropy', ...)` on positive supports.
POT 0.9.6.post1 returns `value_linear = <P,C>` and
`value = <P,C> + epsilon sum(P log(P + 1e-16))`. Thus the comparable objective is
**`POT.value - epsilon * m`**. The tiny `1e-16` stabilizer is covered by the stated
comparison tolerance, not by changing MoonOT's mathematics. Independently
recomputed `transport`, `epsilon H(P)`, and their sum are checked as well.

Omitting `-epsilon*m` leaves the optimizer unchanged but reports the wrong
MoonOT objective (including for non-unit total mass). Replacing the entropy
regularizer with `D(P || a outer b)` also changes the reported value; the balanced
constraints can hide this mistake if one checks only the coupling.

### Unbalanced: generalized KL relative to `a outer b`

MoonOT's objective is

`<P, weight*C> + epsilon D(P || R) + rho_source D(P 1 || a)`
`+ rho_target D(P^T 1 || b)`.

The main POT call explicitly passes `reg_type='kl'`, `c=a outer b`,
`reg_m=(rho_source, rho_target)`, and `weight*C`; it uses
`ot.unbalanced.sinkhorn_unbalanced(..., method='sinkhorn', log=True)`.
Zero, asymmetric, and unequal penalties are intentional. Finite zero penalties
are not converted into balanced constraints, and inputs are not normalized.

**Important pinned-version reporting detail:** POT 0.9.6.post1's Sinkhorn
`log['total_cost']` calls its backend `kl_div` with the default `mass=False`.
That logged quantity includes `sum(x log(x/y + 1e-16))`, but omits `-x+y` in
*all three KL terms*, even though the scaling solution is for generalized KL.
The harness does not mistake the raw log for MoonOT's objective. It restores

**`epsilon*(A*B-m) + rho_source*(A-m) + rho_target*(B-m)`**

to the raw `total_cost`, and compares that corrected value to both MoonOT's
`objective` and an independently evaluated generalized-KL decomposition.
The raw transport cost is also checked against POT's `log['cost']`.
These corrections are necessary, not arbitrary tolerances or fitted offsets.

### Explicit entropy-versus-KL cross-check for unbalanced transport

On positive supports, the harness also calls the same POT unbalanced solver
with **`reg_type='entropy'`**. POT sets its coupling reference to the all-ones
matrix in this mode. Using the original cost would generally produce a different
unbalanced optimizer; entropy and reference KL are **not** interchangeable.
To represent MoonOT's problem exactly, use the shifted linear cost

**`C_entropy = weight*C - epsilon*log(a outer b)`**.

Then `<P,C_entropy> + epsilon D(P || ones)` differs from
`<P,weight*C> + epsilon D(P || R)` by the constant
`epsilon*(N-A*B)`, where `N` is the number of active matrix cells.
First restore the entropy run's missing mass terms
`epsilon*(N-m) + rho_source*(A-m) + rho_target*(B-m)` to its raw log, then add
**`epsilon*(A*B-N)`**. Both adjusted objective and coupling must match the KL run.
The shifted cost can be negative; that is a valid POT-side algebraic conversion,
not a negative input sent to MoonOT.

POT emits an unconditional `UserWarning` announcing its entropy reference
replacement. Only that exact message is ignored. All other POT numerical or
convergence warnings are failures. POT's unbalanced final logged error must meet
the requested tolerance; MoonOT must return `Converged` and meet its own stopping
diagnostic. These error measures differ, so iteration counts are diagnostic,
not expected to match.

## Zero supports and finite corpus

Zero-mass rows/columns are removed **only on the POT side**, then the result is
embedded in the original rectangle with exact zeros. MoonOT receives the full
inputs, and every forbidden row/column must be exactly zero. This avoids POT's
zero divisions and preserves the generalized-KL constants: omitted reference
masses are zero, so their contribution is zero. Entropy cross-checks use the
number of *active* cells for `N`, not the original padded rectangle.

If either positive support is empty, the forced plan is zero. POT is not called
on that empty subproblem. These cases are explicitly labeled **analytic boundary**:
balanced objective zero; unbalanced objective `rho_source*A + rho_target*B`.
They are counted separately and never advertised as POT comparisons.

The corpus contains 60 finite cases:

- 29 balanced: five explicit scalar/symmetric/rectangular/zero-support/zero-total
  problems and 24 deterministic generated problems;
- 31 unbalanced: seven explicit scalar, no-penalty, no-cost, zero-support,
  zero-source, zero-target, and zero-both problems, plus 24 generated problems;
- generated shapes `1x4`, `4x1`, `2x3`, `3x2`, `3x5`, `5x3`;
- epsilons `0.15`, `0.4`, `1.2`, `2.5`; balanced totals `0.3`, `1.0`, `2.7`;
- penalty pairs `(0,0)`, `(0,2)`, `(3,0)`, `(0.05,7)`, `(8,0.1)`, `(1,1)`;
- cost weights `0`, `0.3`, `1`, `1.7`; unequal unbalanced totals and embedded zeros.

Generation is portable: seed `20261014`; 32-bit integer LCG
`state = (1664525*state + 1013904223) mod 2^32`; draw
`(state mod 997 + 1)/998`. Exact draw order and scaling are in `cases()`; no
NumPy RNG, downloaded datasets, random time seed, or upstream fixture is used.
This is deliberately a moderate finite test set; it does not cover overflow,
subnormal/extreme dynamic ranges, invalid inputs, warm starts, infinite penalties,
other POT methods/backends, or other MoonOT APIs.

Solver tolerance is `1e-13`, maximum iterations `100000`. Every compared entry
must satisfy **`abs(actual-expected) <= 2e-9 + 2e-8*abs(expected)`** and be finite;
negative plans, shape mismatches, missing bridge records, iteration exhaustion,
and exact-zero support violations fail. Component/objective checks are distinct
from coupling checks. A JSON report records per-case errors, iteration counts,
counts by reference type, and maximum absolute errors per check category.

## Dependency-free analytic vectors and derivations

Nine MoonBit tests check returned couplings and all reported objective components
using formulas independent of the MoonOT implementation:

1. Balanced one cell: its coupling is the supplied total mass.
2. Balanced one active row: the constraints force its entries to equal `b`;
   other rows and zero target columns vanish.
3. Balanced zero cost: `P = (a outer b)/A` for equal nonzero totals.
4. Balanced symmetric `2x2` with marginals `(1/2,1/2)`, unit off-diagonal cost,
   and epsilon one: diagonal `1/[2*(1+exp(-1))]`; off-diagonal `1/2-diagonal`.
5. Balanced zero total: the forced zero plan and zero objective.
6. Unbalanced one cell, `a,b > 0`: differentiate the generalized-KL objective to
   obtain `P = exp(((epsilon+rho_source)*log(a) +`
   `(epsilon+rho_target)*log(b) - weight*C)/(epsilon+rho_source+rho_target))`.
   Four asymmetric/zero penalty pairs are checked in that test.
7. No marginal penalties: independent cells give
   `P_ij = a_i*b_j*exp(-weight*C_ij/epsilon)`; zero masses force zero entries.
8. Zero weighted cost: stationarity gives rank-one `P = k*(a outer b)`, with
   `k = exp(-(rho_source*log(B)+rho_target*log(A))/`
   `(epsilon+rho_source+rho_target))`, including unequal totals and a zero column.
9. Empty positive supports: check zero-source, zero-target, and zero-both plans,
   including the unmatched positive marginal KL constants.

Each analytic expected plan is used to compute transport, entropy or generalized
KL, and the total objective. The tests use assertions, not snapshots, and require
convergence. Their tolerance is `2e-10*(1+abs(expected))`, with exact equality for
expected zero coupling entries. No values are derived from a MoonOT solve, and
there are no frozen POT-produced tables to disguise as independent analysis.

## Provenance and license

Reference dependency: [Python Optimal Transport 0.9.6.post1](https://pypi.org/project/POT/0.9.6.post1/),
[upstream pinned revision](https://github.com/PythonOT/POT/tree/85113e9a380f5fcf684c50c73c1ff6a164a7366e).
POT is MIT-licensed, copyright the POT contributors (2016–2023 as noted in
MoonOT's existing `THIRD_PARTY_NOTICES.md`). The pinned installed wheel's public
APIs and reporting semantics were inspected in `ot/solvers.py`,
`ot/unbalanced/_sinkhorn.py`, and `ot/backend.py`; no implementation text was
copied. NumPy and SciPy are separately installed reference dependencies with
their own distribution licenses. The new harness, analytic tests, and locally
generated inputs follow this repository's existing Apache-2.0 license. This
addition does not change existing third-party notices or package dependencies.

## Executed results (2026-10-14)

Actual execution on Windows, MoonBit `0.1.20260920 (914d7da)`, `wasm` target,
Python `3.12.14`, POT `0.9.6.post1`, NumPy `2.2.6`, SciPy `1.15.3`:

- **PASS: 60/60 cases** (29 balanced, 31 unbalanced).
- **56 actual POT comparisons**, plus **4 separately labeled analytic boundaries**.
- **28/28 explicit entropy-versus-KL POT cross-checks** passed.

Maximum measured absolute errors (not tolerance bounds):

| Quantity | Maximum absolute error |
| --- | ---: |
| Balanced coupling entry | `8.570921750106208e-14` |
| Balanced objective | `8.171241461241152e-14` |
| Balanced reported components | `8.304468224196171e-14` |
| Unbalanced coupling entry | `2.6756374893466273e-14` |
| Unbalanced objective | `1.0658141036401503e-14` |
| Unbalanced reported components | `1.3322676295501878e-13` |
| POT entropy/KL coupling cross-check | `4.440892098500626e-16` |
| POT entropy/KL adjusted objective cross-check | `5.329070518200751e-15` |

The private installation used
`C:\Users\42673\.cache\moonot-reference-pot-0.9.6.post1`; no global/system
packages were installed. `pip check` passed. The full JSON report, with per-case
results and production-input hashes, was written outside the repository to
`C:\Users\42673\.cache\moonot-reference-wasm-report.json`.

`moon test reference_vectors_test.mbt`: **9 passed, 0 failed**. An isolated copy
also ran `moon info`, `moon fmt`, and the full tests: **141 passed, 0 failed** at
that snapshot. Only the formatted new reference test was copied back; existing
source files and generated interfaces were not edited by this validation task.
Concurrent repository work can change the full-suite count or compilation state.

Failure sensitivity was actually checked by injecting `+0.001` into the first
MoonBit objective *in memory*, outside repository files: the harness rejected it
at `balanced_components:balanced/one_cell`, measured error
`0.00099999999999988987`. A Python environment without SciPy was also exercised:
it reported `BLOCKED` and exited 2, rather than claiming a POT comparison.
