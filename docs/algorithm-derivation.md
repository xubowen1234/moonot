# MoonOT 算法推导与数值语义

本文将当前实现的迭代式写成数学记号，说明每个更新对应的目标和诊断量。约定与公开边界以[算法契约](algorithm-contract.md)为准；这里不把有限迭代的数值输出解释为额外的误差保证。

## 记号与共同假设

- 源支持大小为 `m`、目标支持大小为 `n`；成本矩阵 `C` 是稠密的 `m × n` 矩阵，行对应源、列对应目标。
- 输入质量与成本有限且非负。需要取对数的质量只在严格为正时取对数；非正支持项按各算法的零质量规则处理。
- 平衡算法的 `epsilon > 0`。所有更新在 log-domain 中用稳定的 `logsumexp` 计算；最终耦合仍会指数化为普通 `Double`，因此极小质量可能下溢为零，数值异常会作为错误返回。
- 下列推导说明理想实数运算下的缩放关系，不是对有限精度、迭代收敛速度或最优性误差的额外定理。

## Balanced Sinkhorn

### 目标与缩放形式

输入边缘 `a ∈ R_+^m`、`b ∈ R_+^n` 的总质量必须在实现规定的相对容差内相等，且不会被隐式归一化。实现最小化

```text
F(P) = sum_ij P_ij C_ij
     + epsilon * sum_ij P_ij (log(P_ij) - 1)
```

约束为 `P 1 = a`、`P^T 1 = b`、`P >= 0`；熵项在 `P_ij = 0` 时按零贡献处理。对正质量支持上的一阶条件给出缩放表示

```text
P_ij = u_i K_ij v_j,      K_ij = exp(-C_ij / epsilon).
```

为避免构造容易下溢的 `K`，代码保存 `log K` 并用 `logsumexp` 做矩阵-向量归约。它在活跃源/目标单元中将成本减去最小值 `c_min`，即使用 `log K_ij = -(C_ij - c_min) / epsilon`。总运输质量固定为 `M`，这个常数平移只给目标增加/减去与 `P` 无关的常数 `c_min M`，不改变最优耦合；返回的目标值仍按原始 `C` 计算。

### 交替边缘更新

从 `log u = log v = 0` 开始，每轮先令各行满足源边缘，再令各列满足目标边缘：

```text
log u_i = log a_i - logsumexp_j(log K_ij + log v_j)
log v_j = log b_j - logsumexp_i(log u_i + log K_ij)
```

仅对正质量的行、列执行这些更新。完成一轮后，代码从 `log P_ij = log u_i + log K_ij + log v_j` 生成耦合，并计算

```text
r = max(max_i |sum_j P_ij - a_i|,
        max_j |sum_i P_ij - b_j|).
```

当 `r <= tolerance` 时报告 `Converged`；达到迭代预算则报告 `IterationLimit`。目标值为原始成本加熵项的直接求和；`r` 是边缘可行性残差，不是目标函数差距证明。

### 零质量边界

若 `a_i = 0`，第 `i` 行返回零；若 `b_j = 0`，第 `j` 列返回零。正边缘质量项参与 log-domain 更新。双方总质量均为零时，直接返回全零计划、零目标与零残差、零次迭代及 `Converged`。不等总质量（超出容差）会被拒绝，而不是通过缩放输入来修复。

## KL-unbalanced Sinkhorn

### 目标与一阶条件

令 `D_ij = cost_weight * C_ij`。实现使用如下广义 KL（I-divergence）目标：

```text
G(P) = sum_ij P_ij D_ij
     + epsilon * KL(P || a ⊗ b)
     + rho_source * KL(P 1 || a)
     + rho_target * KL(P^T 1 || b)

KL(x || y) = sum_k (x_k log(x_k / y_k) - x_k + y_k).
```

边界约定是 `KL(0 || 0) = 0`、`KL(0 || y) = y`（`y > 0`），而 `x > 0, y = 0` 的 KL 为无穷。耦合参考是乘积测度 `(a ⊗ b)_ij = a_i b_j`。在正支持上对 `P_ij` 求导，可写成

```text
P_ij = a_i b_j K_ij u_i v_j,
K_ij = exp(-D_ij / epsilon).
```

记 `alpha_s = rho_source / (rho_source + epsilon)`、`alpha_t = rho_target / (rho_target + epsilon)`。将缩放形式代回行、列边缘的驻点关系，得到广义缩放：

```text
log u_i = -alpha_s * logsumexp_j(log b_j + log K_ij + log v_j)
log v_j = -alpha_t * logsumexp_i(log a_i + log K_ij + log u_i)
```

这解释了源、目标罚系数各自作用于对应一侧的更新：当两侧 `rho` 不同时，两个指数也不同。`rho = 0` 时对应指数为零、该侧缩放为 `1`。实现计算 `K` 的 log 值，不减去成本常数；生成计划时使用

```text
log P_ij = log a_i + log b_j + log K_ij + log u_i + log v_j.
```

### 收敛诊断与零质量

代码每轮比较新旧 `log u`、`log v`，以所有坐标更新的最大绝对值作为 `scaling_residual`。不超过 `tolerance` 时报告 `Converged`。输出还包含相对输入边缘的最大绝对差，但这个边缘差是诊断值，并非停止条件；非平衡问题本来就不要求计划边缘等于输入。

若 `a_i = 0` 或 `b_j = 0`，乘积参考在对应行/列为零，有限的耦合 KL 因而强制该行/列耦合为零。任一侧总质量为零时实现直接使用零计划；结果仍计算广义 KL 分量，所以另一侧正质量带来的未匹配边缘惩罚不会被边界处理悄然抹去。`rho_source`、`rho_target` 可为零，`epsilon` 必须严格为正。

## Fixed-support regularized barycenter

### 约束目标

有 `K` 个测度 `a^(k)`，均定义在同一 `n` 点支持上，总质量共同为 `M`；输入权重 `w_k` 必须有限且严格为正，代码归一化为 `lambda_k = w_k / sum_l w_l`。使用同一个有限、非负、方阵成本 `C` 和同一个 `epsilon > 0`，目标为

```text
min_x  sum_k lambda_k OT_epsilon(x, a^(k); C)
subject to x >= 0,  sum_i x_i = M.
```

这里 `OT_epsilon` 就是前述 balanced 目标。输入总质量在 `1.0e-12 * max(total_k, M)` 的相对尺度容差内相等；输入不会归一化，输出总质量为 `M`，所以只有 `M = 1` 时它才是概率向量。

### 缩放与加权几何平均

对每个输入测度 `a^(k)`，平衡耦合写作 `P^(k)_ij = u^(k)_i K_ij v^(k)_j`，其中实现使用

```text
log K_ij = -(C_ij - c_min) / epsilon.
```

记 `q^(k)_i = sum_j K_ij v^(k)_j`。给定当前列缩放，代码先按总质量 `M` 归一化 `q` 的加权几何平均：

```text
log x_i = sum_k lambda_k log q^(k)_i + c
c = log(M) - logsumexp_i(sum_k lambda_k log q^(k)_i).
```

接着令 `u^(k)_i = x_i / q^(k)_i`，再按每个输入测度的列边缘更新

```text
v^(k)_j = a^(k)_j / sum_i K_ij u^(k)_i.
```

代码以 log-sum-exp 形式实现这两个缩放步骤，反复重算加权几何平均。其固定点关系来自共享源边缘 `x` 下每个平衡问题的行缩放，并由总质量约束确定常数 `c`；这里不将该推导解释成任意有限迭代均已达到全局最优的证明。共同成本减去 `c_min` 不改变固定总质量计划的解，最终加权目标则由各个返回 `x` 与原始 `C` 的 balanced 求解结果计算。

### 残差、返回目标与零总质量

`log_barycenter_residual` 是相邻两次外迭代中 `log x` 坐标的最大绝对变化；外迭代在其不超过 `tolerance` 时报告收敛。迭代后，代码对每个 `a^(k)` 再运行 balanced 求解，计算 `weighted_objective = sum_k lambda_k * OT_epsilon(x, a^(k); C)`，并取这些求解的最大边缘残差作为 `maximum_transport_residual`。若这些内层求解触及自身迭代上限，最终状态也报告 `IterationLimit`。外层更新残差与内层边缘残差都不是 objective-gap 证书。

总质量为零时，因输入非负，各测度都是全零；实现直接返回全零重心、零目标、零残差、零次迭代与 `Converged`。总质量为正时，各输入中的零坐标会在对应平衡耦合里形成零列；代码在更新对应列缩放时跳过这些坐标。熵正则会改变相对于非正则 OT 的重心；当前 API 不提供定量正则化偏差界或最优性差距界。

## 稠密复杂度

以下按一次标量运算、`exp`、`log` 和 `logsumexp` 归约均记为常数工作量估算；不利用稀疏结构，也不包含调用端构造成本矩阵的费用。

- **Balanced / unbalanced transport：**每次迭代对行、列做稠密归约，并生成耦合、计算残差，均为 `O(mn)` 时间；保存 log-kernel 和稠密 plan 需要 `O(mn)` memory，缩放向量另需 `O(m+n)`。返回目标分量的遍历也是 `O(mn)`。
- **固定支持重心：**设输入测度数为 `K`，支持大小为 `n`。一次外迭代对每个测度做稠密核归约和列缩放更新，时间为 `O(K n^2)`；之后另有 `K` 次 balanced 目标评估，每次最多运行其迭代预算、每轮 `O(n^2)`。因此若外层和内层预算均为 `I`，实现的最坏迭代工作量为 `O(K I n^2)`，常数项包含外循环与目标评估两部分。重心阶段保存共享 `n × n` log-kernel 和 `O(Kn)` 缩放/工作向量；每次目标评估只需一个 `n × n` plan，不会同时保留 `K` 个 plan。一般而言每个稠密运输 plan 的 memory 是 `O(mn)`。

## 不能从残差推出的结论

这些状态与残差描述的是实现实际检查的量：平衡解的边缘误差、非平衡解的 log 缩放更新及边缘诊断、重心的 log 外迭代更新及内层边缘诊断。它们不构成 objective-gap 或距精确最优解的上界。熵正则相对非正则问题的偏差还取决于问题数据；本项目未据此提供数值偏差承诺。