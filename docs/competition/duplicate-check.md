# MoonOT — duplicate-check and selection evidence

**Check date:** 2026-10-13 (Asia/Shanghai)  
**State:** selected direction; local implementation and verification are in progress. No public release is claimed.  
**Working project name:** MoonOT (provisional; repository destination is unresolved because the supplied `xubowen1234/moontsinspect` repository already contains the distinct MoonTSInspect project and the authenticated account has read-only permission).

## 1. Capability fingerprint

- **Problem domain:** reusable discrete optimal-transport numerics for finite nonnegative measures on finite supports.
- **Primary downstream users:** machine-learning/data-alignment libraries; image-analysis/color-transfer pipelines; resource-allocation simulators and scientific applications.
- **Workflows:** (1) source and target sample masses plus pairwise costs → a transport coupling, objective and marginal residual for downstream reweighting; (2) histograms and a color-distance matrix → a coupling/map for a separate color-transfer stage; (3) divisible supply/demand masses and costs → a regularized allocation plan with convergence/budget diagnostics. The library will not promise downstream model quality, decode images, or return integer-optimal assignments.
- **Core data/API:** finite mass vectors, dense cost matrix, regularization/problem configuration and explicit iteration/tolerance/resource budget; return coupling, objective, marginal residuals and a structured termination status/error.
- **Implemented capabilities as of 2026-10-13:** log-domain balanced and KL-unbalanced transport; Sinkhorn divergence; shared- and heterogeneous-cost fixed-support barycenters; opaque balanced warm starts; rectangular point-cloud costs and coupling barycentric projection. Each public objective and termination/residual convention is documented in the library contract.
- **Not in scope:** full POT parity, generic LP/MIP or simplex, integer assignment, route optimization, entity resolution, GPU/autodiff, image decoding, or a machine-learning framework.
- **Acceptance evidence recorded after the first implementation pass:** 82 tests pass on each of `wasm-gc`, `wasm`, and `js`; barycenter coverage includes an asymmetric analytic Dirac solution/objective and an independent all-positive 2×2 scalar reference. The configured eight examples ran on all three targets. No POT differential-test result, native execution result, or remote CI result is claimed.

## 2. Search scope and method

The searches were performed on 2026-10-13. MoonCakes was queried using `moon search --limit 100 --json` across optimal transport, Sinkhorn, Wasserstein, earth mover / EMD, coupling, entropic transport, unbalanced transport, barycenter, `sinkhorn_log`, `transport_plan`, `marginal_error`, and Chinese equivalents. Results were inspected semantically rather than treating an exact-name miss as evidence. GitHub MoonBit repository/code searches used transport/Sinkhorn/optimal-transport terms and API concepts; GitHub indexing and unauthenticated/public search coverage are incomplete, so this is not an exhaustive search.

The permanent registry was read in full before selection (3324 lines; SHA-256 `15ff21a0b6f1602eb9106d57b33119601a18690d9fe9b10bbbf6d2cb5cc50f38`) and then rechecked for `moonopt`, `moon_diffusion_lab`, Sinkhorn, Wasserstein, transport and barycenter. Existing `moon_diffusion_lab` entries describe a 1D/2D toy diffusion lab, not an OT solver. The registry search did not reveal a pre-existing reserved/completed OT solver fingerprint.

## 3. Upstream and adjacent projects inspected

| Project | Fixed revision / published release | Evidence inspected | Capability relationship |
|---|---|---|---|
| [PythonOT/POT](https://github.com/PythonOT/POT/tree/85113e9a380f5fcf684c50c73c1ff6a164a7366e) | tag `0.9.6.post1`, commit `85113e9a380f5fcf684c50c73c1ff6a164a7366e`, MIT | `ot/bregman/_sinkhorn.py` (`sinkhorn_log`), `test/test_bregman.py`; release license and project overview | Upstream algorithm/reference, not a MoonBit replacement. The proposed subset is an independent implementation; no upstream source is copied. Full POT parity is explicitly excluded. |
| [toadium/owl-mbt](https://github.com/toadium/owl-mbt/tree/7be43e3c24abd06a80a6dd50640492caf89108a) | `7be43e3c24abd06a80a6dd50640492caf89108a` | `src/linalg/linalg.mbt`, linalg tests, `src/optimise/core.mbt`, `types.mbt`, optimizer tests | Arrays, linear algebra and gradient optimization; useful substrate, not a transport-coupling solver. |
| [mizchi/numbt](https://github.com/mizchi/numbt/tree/328c89315d8130ddf76d5226d3323fec3ec92839) | `328c89315d8130ddf76d5226d3323fec3ec92839` | `src/vec.mbt`, `vec_test.mbt`, `fmat_accelerate_test.mbt` | Vector/matrix and BLAS/LAPACK-oriented numerics; no specialized entropic coupling workflow observed. |
| [Luna-Flow/linear-algebra](https://github.com/Luna-Flow/linear-algebra/tree/01696669d88faf16d073a8f0b78aa644ecb696b3) | `01696669d88faf16d073a8f0b78aa644ecb696b3` | generated public interface, `matrix.mbt`, `matrix_wbtest.mbt` | Matrix/vector backend; adjacent numerical substrate rather than an OT model/solver. |
| [didiLjf/moon-record-linkage](https://github.com/didiLjf/moon-record-linkage/tree/121503d3464802488a150af56d2e10aa1398e6f3) | `121503d3464802488a150af56d2e10aa1398e6f3` | blocking and `candidate_planner` implementation/tests | Entity matching and deduplication; not mass-conserving distribution transport. |
| [phjphj676/moon-online-models](https://github.com/phjphj676/moon-online-models/tree/48e220855a8171048bd1f2fa400f7fd0fe7959e4) | `48e220855a8171048bd1f2fa400f7fd0fe7959e4` | public interface, online clustering implementation/tests | Online learning/clustering; no general coupling API observed. |
| [pkdxtools/nash-mbt](https://github.com/pkdxtools/nash-mbt/tree/4917b4c026c15886e33c9b1c609ed16d5b7fe32) | `4917b4c026c15886e33c9b1c609ed16d5b7fe32` | matrix-game, simplex, solver and divergence sources/tests | Matrix games and strategy distributions; not an entropic OT solver. |
| [Freon793/moonopt](https://skills.mooncakes.io/docs/Freon793/moonopt%400.2.0) | MoonCakes `0.2.0` | published `solve`/`solve_with` API, `moonopt.mbt`, `simplex/simplex.mbt`, `docs/api.md` | General LP/MIP simplex can model classical unregularized transport; inspected API/source did not expose a specialized entropic/Sinkhorn coupling API. |
| [hxiuzheng/moon_diffusion_lab](https://mooncakes.io/docs/hxiuzheng/moon_diffusion_lab) | MoonCakes `0.1.6` | `benchmark/benchmark.mbt` (two call regions), `core/pkg.generated.mbti`, `core/core_test.mbt`; repository archive unavailable | Public `sliced_wasserstein_approx(Array[Vec2], Array[Vec2]) -> Double` is a sliced-Wasserstein scalar metric, not the proposed general coupling result. Implementation body was not available, so no claim is made about its internals. |

In addition, breadth screening found direct overlap for two *other* candidates: [`CAIMEOX/moon_egg`](https://mooncakes.io/docs/CAIMEOX/moon_egg) already exposes the central e-graph/equality-saturation workflow; [`CaptainK-65/moon-collate`](https://github.com/CaptainK-65/moon-collate) implements Unicode 17 UCA/DUCET comparison and sort keys. Those directions were rejected rather than renamed or wrapped. Their presence does not overlap the selected OT fingerprint.

## 4. Risk analysis and verdict

Potential risks are numerical stability at small regularization, zero-support handling, ambiguous unbalanced/KL conventions, solver termination semantics, and the breadth of the proposed algorithm list. These must be controlled through a precisely stated objective, scale-aware tolerances, explicit budgets/statuses, POT differential tests on small cases, and a staged core before optional barycenter/divergence features.

**Verdict:** Within the recorded MoonCakes, GitHub MoonBit, adjacent-source/API/test and registry checks, no mature direct replacement for the proposed reusable entropic transport-coupling API was found. This is a bounded search conclusion, not a claim that no similar implementation exists anywhere. MoonCakes/GitHub indexing is not exhaustive; the MoonDiffusionLab implementation archive was inaccessible, so its conclusion is restricted to its inspected public API and tests.

## 5. Implementation progress and verification — 2026-10-13

This section updates the pre-implementation acceptance plan above without changing the recorded search scope or its limitations. MoonOT now includes the original shared-cost fixed-support barycenter API and `regularized_barycenter_with_costs`, which pairs each measure with its own finite, nonnegative square cost matrix while keeping a common ordered barycenter support. The shared-cost implementation retains one log kernel; the heterogeneous path stores one per measure. Both paths use the documented balanced entropy objective, common epsilon and normalized positive weights, and evaluate the returned objective against the original paired matrices.

The heterogeneous path has an asymmetric analytic Dirac case with an independently derived objective, plus an independent all-positive 2×2 reference that bisects each scalar coupling subproblem and uses golden-section search for the outer barycenter objective. Validation, zero-mass, cost-shift, paired permutation, shared-API compatibility, termination, and input-immutability cases are also tested. On 2026-10-13, strict check/build/test passed for `wasm-gc`, `wasm`, and `js`; each target passed **82/82** tests, and all eight configured examples were run successfully on each of these targets. `moon info`, `moon fmt`, and `git diff --check` passed. The core library contains 2,068 physical `.mbt` lines excluding test files. Native `moon check --deny-warn --target native` passed; native build/test could not run locally because `cc`, `gcc`, and `clang` are unavailable. The configured CI matrix includes native verification, but no remote CI or Release is claimed. The supplied GitHub repository remains unusable as a destination because it contains a different project; no remote write has occurred.

**Bounded duplicate-check conclusion:** 在已记录检索范围内未发现成熟直接替代。 This remains a limited conclusion: MoonCakes/GitHub indexing is not exhaustive, and the MoonDiffusionLab implementation archive was inaccessible, so its public API and inspected tests define the boundary of that comparison.
## Pre-delivery recheck — 2026-10-14

Repeated MoonCakes CLI searches with limit 100 for Sinkhorn, Wasserstein, optimal transport, barycenter and coupling, plus account-bound GitHub MoonBit repository searches for solver and API concepts. The original `moon_diffusion_lab` and `moonopt` remain adjacent rather than direct solver substitutes. Barycenter hits additionally described graph-layout ordering heuristics, barycentric coordinates/interpolation and physical-measurement interpolation (`Milky2018/diago`, `bobzhang/elk`, `Milky2018/moon_elk`, `wjyupl/measure`, `bobzhang/games`, `zhaoyuanjuns/moonbit_extractor`); the returned API summaries do not describe measure-to-measure entropy-regularized transport. Coupling hits described parasitic RC, filesystem and hydrology coupling. These are semantic false positives, not proof of exhaustiveness. The four GitHub repository queries returned no results; code outside indexed repository descriptions may be missed. Previous README/API/source/test comparisons and their archive-access limitation still apply. New weighted costs, feasibility repair and Markov gluing remain within the reserved finite-support transport fingerprint.

**Conclusion:** 在已记录检索范围内未发现成熟直接替代。

## 修复交付前语义复核 — 2026-10-20

本节是对既有选题与原有三个不同问题域候选记录的补充，不覆盖早期查重或阶段性测试结论。再次执行无凭据 MoonCakes CLI 检索（每词上限 100，覆盖 Sinkhorn、Wasserstein、optimal transport、barycenter、coupling），并通过指定账号只读查询 GitHub MoonBit 仓库。仓库描述检索零命中并不证明源码中没有实现；搜索索引、未收录和私有项目仍构成覆盖限制。

本次重新检查当前永久注册表相关完整指纹和以下固定官方发布归档的 README、公开接口、实现及测试。相邻测试仅做静态审阅，未宣称运行。此前 MoonDiffusionLab 归档不可访问的局限已由本次精确版本源码读取补足；历史记录仍保留。

| 固定相邻版本 | 实际核心与重叠 | 与 MoonOT 的功能边界 |
|---|---|---|
| [Freon793/moonopt@0.3.0](https://mooncakes.io/docs/Freon793/moonopt) | LP/MIP、预求解、revised simplex；运输 LP 示例与传统平衡运输存在真实交集。较 0.2.0 增加 LP 松弛、Devex 定价和强化等。 | 模型/证书/运输示例与相关源码测试复核未见增加熵正则 Sinkhorn、KL 非平衡耦合、去偏散度或固定支持 OT 重心。不是仅因算法名不同判为不重复。 |
| [hxiuzheng/moon_diffusion_lab@0.1.6](https://mooncakes.io/docs/hxiuzheng/moon_diffusion_lab) | `core/stats.mbt` 的 `sliced_wasserstein_approx` 是二维样本两轴排序的近似标量指标，测试覆盖相同及平移样本。 | 不接收任意代价与源/目标质量模型，不输出稠密耦合、KL 目标组成、边际残差或 OT 重心。 |
| [walkzzz/moon-geometry@0.231.0](https://mooncakes.io/docs/walkzzz/moon-geometry) | `src/emd` 与根 reexport 确有一维 EMD/均匀或权重距离接口，不能把它说成没有最优传输功能。 | 一维标量距离与 MoonOT 的任意有限矩形代价、稠密正则耦合及派生组合有局部领域交集，但输入、输出、目标和主要任务不同。 |

本次归档 SHA-256 依次为 `431a83e4ed5a28885bb37459dbb8b539ca4270912d9c4dc7fc04fc659a014485`、`e3a0d542610466f1ac4accfb3c27b299c459cbf1323915a1e71245fd157ab9cc`、`258ac806a532c8062f06f15ba09b78fe1f65b8b108fb6a301c04f3f5909cae01`。逐文件摘要及关键源码行号保留在仓库外验收证据中，不上传私人报告。

永久注册表当前快照 SHA-256 为 `fcd2fb7c6ac490a6bc23bd36c9fad4a511d85e1406981fdc62dc0ec6a5d73f76`。MoonOT 自身 reserved/completed/README 同步记录为同一项目历史；对 MoonQP、MoonChoice、MoonCutStock、MoonIsotonic、MoonDOE、MoonBelief、MoonMultigrid、MoonKinetics 等保留指纹按真实目标、数据模型及工作流比较，不因共有“质量、分布、矩阵、分配”等词就认定等价，也不复用其它 reserved/completed 项目。

现有数据对齐、颜色直方图、可分资源配置三类完整示例继续使用同一公共 API；本轮只修复已复现的数值边界，不扩展产品方向。native 本地仍只有严格检查证据，修复提交的真实 CI 必须另行核验，不能借用旧 CI。

**查重结论：在已记录检索范围内未发现成熟直接替代。**
