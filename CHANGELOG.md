# Changelog

All notable user-visible changes to MoonOT will be recorded here. Entries describe repository capabilities and do not imply publication.

## 0.1.3 — 2026-10-20

- 修复条件重心投影在零权重大坐标上的伪溢出，以及极端指数条件权重先下溢造成的可表示结果丢失。
- 修复 Markov 耦合拼接的中间比值下溢，并延迟恢复最终二进制指数，保留多个次正规桥接项合计后可表示的质量。
- 修复 KL 非平衡传输费用计算中三个因子连乘的次正规前缀舍入偏差；公共 API 与依赖不变。
- 增加极端指数、最小次正规数、符号、矩形输出和输入不变性回归；独立验证方法与适用边界见算法契约。

## 0.1.2 — 2026-10-14

- 修正 README 示例的错误与数组输出，使其通过严格编译；增加文档代码示例的自动校验，算法与公共 API 不变。

## 0.1.1 — 2026-10-14

- 将主 README 与包简介改为中文，补齐安装说明、全部可运行示例和数值边界；公共 API 与算法实现未改变。

## 0.1.0 — 2026-10-14

- Added feature-weighted ground costs, bounded epsilon schedules, coupling orientation, entropy/generalized KL, balanced feasibility rounding, and Markov composition with typed validation failures.
- Added an assertion-bearing three-domain alignment workflow and independently reproducible analytic/POT reference validation.
- Added audited source/history gates and pinned multi-target CI.

- Added opt-in KL-unbalanced Sinkhorn continuation with a distinct opaque seed/error API, checked epsilon migration, per-call diagnostics, and a runnable changed-input example; no speedup is guaranteed.
- Added `regularized_barycenter_with_costs` for fixed-support balanced barycenters with one validated square ground-cost matrix per measure, with per-matrix log-domain kernels and original-cost objective evaluation; the shared-cost API remains unchanged.

- Added opt-in balanced Sinkhorn continuation with opaque cost-unit dual-potential seeds, preserving the existing `SinkhornResult` shape and reporting ordinary convergence diagnostics for each solve.
- Added validated dense-coupling diagnostics for row/column marginals and unregularized transport cost, with shape, numeric-domain, and overflow errors.
- Added conditional barycentric projection from coupling rows to finite target feature vectors, with explicit zero-mass-row and shape errors.
- Exposed balanced transport-cost and entropy-regularizer components alongside the combined objective, with analytic accounting tests.
- Implemented balanced entropy-regularized Sinkhorn transport and KL-unbalanced Sinkhorn transport with explicit objectives and termination/residual diagnostics.
- Implemented balanced Sinkhorn divergence from cross and self-transport objectives, with caller-supplied cost matrices.
- Implemented fixed-support regularized barycenters using strictly positive, internally normalized weights over equal-total input measures; the output preserves their common total mass.
- Added validated rectangular point-cloud cost builders for squared-Euclidean, Euclidean, and Manhattan metrics, with stable Euclidean accumulation and typed shape/numeric errors.
- Added a runnable point-cloud transport example that feeds squared-Euclidean costs to the balanced solver.
- Synchronized the numerical contract and README with barycenter objective, mass/weight constraints, residual semantics, and regularization-bias limits. No release, performance, or external compatibility claim is made.
