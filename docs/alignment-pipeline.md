# Three-domain alignment workflow

Run `moon run examples/alignment-pipeline` (choose a supported target with `--target`).

Two feature distributions are aligned through a shared three-point bridge. Feature weights make the second coordinate less influential, without normalizing any marginal. A strictly decreasing two-stage epsilon schedule solves each leg. The workflow rejects exhausted iteration budgets, repairs both plans to the supplied marginals, then glues their bridge conditionals and maps the source into target coordinates with a barycentric projection.

The executable checks source/target marginal preservation and the projection's coordinate-wise convex-hull bounds, and aborts on every failed contract. The bridge composition is a feasible multi-domain correspondence, **not** a claim that it minimizes direct source-to-target transport cost. Repair modifies the original solver plan and its objective; the original optimality diagnostics must not be attached to the repaired/composed result. Ground-cost feature weights are not distribution masses.

All inputs are small synthetic authored examples, not private data or generated source-code scale. This demonstrates reusable APIs for domain alignment; it does not implement a machine-learning model or application-specific preprocessing.
