# MoonOT

MoonOT 是纯 MoonBit 实现的有限离散分布最优传输库，提供平衡与 KL 非平衡 log-domain Sinkhorn、去偏 Sinkhorn divergence，以及固定支持上的正则化重心。数学约定参考 [Python Optimal Transport（POT）](https://github.com/PythonOT/POT)，核心实现独立重写，不复制上游实现代码，也不承诺完整 POT API 兼容。

> **适用范围：** 有限支持、密集矩阵、CPU 数值计算；显式报告输入错误、数值失败与迭代预算耗尽。残差和收敛标志不是无正则最优性证书。交付验证说明见 [交付审计](docs/competition/delivery-audit.md)。

## 安装与快速开始

MoonCakes 模块为 `xubowen1234/moonot`。在使用方项目中添加依赖：

```sh
moon add xubowen1234/moonot@0.1.3
```

在使用方的 `moon.pkg` 中声明包导入：

```moonbit
import {
  "xubowen1234/moonot",
}
```

下面计算两个等总质量分布之间的平衡传输计划：

```moonbit
let result = @moonot.balanced_sinkhorn(
  [0.5, 0.5],
  [0.5, 0.5],
  [[0.0, 1.0], [1.0, 0.0]],
  { epsilon: 0.25, tolerance: 1.0e-9, max_iterations: 1000 },
)
match result {
  Err(_) => println("传输求解失败")
  Ok(solution) => {
    for row in solution.coupling {
      for value in row {
        println(value)
      }
    }
    println(solution.transport_cost)
    println(solution.entropy_regularizer)
    println(solution.regularized_objective)
    println(solution.marginal_residual)
  }
}
```

调用者应同时检查结果的终止状态和残差，不应把成功返回直接理解为已经收敛。完整公开类型与函数签名见 [生成接口](pkg.generated.mbti)。

## 可以复用在哪些系统中

- **数据分析与机器学习组件：** 输入样本质量和特征坐标，构造代价、求解耦合，再做条件重心投影，用于跨域分布对齐或重加权；不依赖特定训练框架。
- **图像统计流水线：** 输入调用方准备的颜色直方图和颜色距离，生成软映射，或在固定支持上聚合多组直方图；图像读取和映射应用由调用方完成。
- **资源配置与科学仿真：** 输入可分割供给、需求和运输代价，求解平衡配置；总量不相等或观测缺失时使用 KL 非平衡求解器，并读取实际运输量和目标分量。这不是整数指派求解器。

## 已实现能力

- 平衡熵正则传输：log-domain 缩放、等总质量校验，不暗中归一化输入。
- KL 非平衡传输：独立的源侧与目标侧质量惩罚、乘积参考耦合 KL，允许输入总质量不同。
- 去偏 Sinkhorn divergence：显式接收跨分布、源自身与目标自身三个代价矩阵。
- 固定支持正则重心：支持共享代价或逐分布独立代价，内部归一化严格正权重，输出保留共同输入总质量。
- 有限非负质量和代价、矩形密集传输矩阵、零质量支持的显式处理。
- 平方欧氏、欧氏、曼哈顿及特征加权平方欧氏代价构造。
- 平衡与非平衡不透明热启动、严格递减 epsilon 的分阶段延续。
- 耦合边缘量、运输成本、重心投影、转置、重排、熵、广义 KL、平衡可行性修复和 Markov 拼接。
- 结构化错误、目标分量、终止原因、迭代次数与数值残差。

## 数值目标与边界

### 平衡传输与 divergence

平衡求解器的目标为：

`<P,C> + epsilon * sum_ij P[i,j] * (log(P[i,j]) - 1)`。

当 `P[i,j] = 0` 时，熵项定义为零。divergence 从跨分布目标中减去两项自身传输目标各自的一半；只有调用方提供适当且相互一致的底层代价时，才有相应非负性保证，不对任意三个代价矩阵承诺非负。

### KL 非平衡传输

非平衡目标为：

`<P, cost_weight*C> + epsilon*KL(P || a⊗b) + rho_source*KL(P1 || a) + rho_target*KL(Pᵀ1 || b)`，

其中广义 `KL(x || y) = sum(x*log(x/y) - x + y)`，零质量边界显式处理。非平衡耦合的边缘量不必等于输入质量；其残差不是平衡可行性证书。

### 固定支持与异构代价重心

共同总质量为 `M` 时，目标为 `min_{x >= 0, sum(x) = M} sum_k lambda_k * OT_epsilon(x, a_k; C_k)`。

`regularized_barycenter` 为所有分布使用同一个代价矩阵；`regularized_barycenter_with_costs` 将每个输入分布与一个独立的 `C_k` 配对：

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
  Err(_) => println("输入分布、代价矩阵或求解配置无效")
  Ok(solution) => {
    for value in solution.barycenter {
      println(value)
    }
  }
}
```

矩阵行对应共同的重心支持，列对应配对分布的支持；所有矩阵必须具有相同的行顺序，并为有限、非负的 `n × n` 方阵，不要求对称或满足度量性质。各分布共享 epsilon，矩阵不被平均或单独缩放。严格正权重归一化为 `lambda_k`；输入总量须在文档规定的相对容差内相等，输出仅在 `M = 1` 时才是概率向量。

该 API 不优化支持点位置，不接受矩形重心代价、无限禁止边或逐分布独立 epsilon，也不学习代价。熵正则化可能引入相对无正则重心的偏差。外层收敛由 log 重心更新残差决定；传输残差仅作诊断，两者均不是最优间隙或偏差界。完整公式、校验和终止语义见 [数值契约](docs/algorithm-contract.md) 与 [算法推导](docs/algorithm-derivation.md)。

## 代价构造与耦合工具

### 点云代价

`squared_euclidean_cost_matrix`、`euclidean_cost_matrix` 和 `manhattan_cost_matrix` 接收源、目标点集，返回行对应源点、列对应目标点的矩形代价矩阵。调用者也可以自行提供领域代价。空支持、参差维度、维度不匹配、非有限坐标与不可表示结果均有显式错误。欧氏距离采用缩放 hypot 累积；平方欧氏与曼哈顿结果超出有限 `Double` 范围时返回溢出错误。

`weighted_squared_euclidean_cost_matrix` 接收对角特征权重，允许零权重，但所有坐标仍必须有限。见 [特征加权代价](docs/weighted-costs.md)。

### 边缘量、成本与重心投影

`coupling_marginals` 计算行、列总量；`coupling_transport_cost` 计算同形计划和代价的无正则内积 `<P,C>`，不增加熵或 KL 项。它们通过 `CouplingError` 拒绝空、参差、不匹配、负值和非有限输入，累积或乘积溢出也返回错误。

`coupling_barycentric_projection` 对每个源行计算目标数值特征的条件加权均值，形成软映射，不选择单一目标或修改计划。目标特征矩阵须非空且矩形；零质量源行返回投影未定义错误，而不是填充哨兵值，零质量目标列不参与。每个特征按活跃最大幅度缩放后累积，以处理极端有限坐标。此操作不做图像解码、类别指派或整数优化。

### 重排、信息量、可行性与拼接

- `coupling_transpose`、`coupling_permute`：返回新的校验后计划；置换必须是完整的输出到输入索引排列。见 [方向与重排](docs/orientation.md)。
- `coupling_entropy`、`coupling_kl`：用于未归一化计划的 `sum p(log p-1)` 和广义 KL，零参考量显式报错。见 [信息量](docs/information.md)。
- `round_balanced_coupling`：通过行列裁剪和缺额修复报告实际边缘可行性，不承诺最优性。见 [可行性修复](docs/rounding.md)。
- `coupling_compose`：在兼容的桥接边缘量上作 Markov 拼接，不等同于直接求解端到端最优传输。见 [耦合拼接](docs/composition.md)。

## 热启动与分阶段延续

`balanced_sinkhorn_with_warm_start` 返回普通 `SinkhornResult` 和不透明的代价单位对偶势种子。相关问题可以复用种子，也可以改变 epsilon；每次调用仍独立遵守容差和迭代上限。不使用热启动时，原有结果结构不变。

`unbalanced_sinkhorn_with_warm_start` 接受 `None` 或先前非平衡求解返回的种子，返回普通 `UnbalancedSinkhornResult` 和新种子，预算耗尽时也返回种子。种子保存私有有限 log 缩放和原 epsilon；epsilon 改变时使用 `log_scale_new = (epsilon_old / epsilon_new) * log_scale_old` 转换，同 epsilon 则直接复用。

源、目标维度须匹配，调用者必须保持坐标语义与顺序。质量、支持、代价、惩罚、代价权重、epsilon 和预算可以变化：种子只是初始值，不是缓存的计划。新失活坐标及零边缘惩罚侧采用中性缩放；零支持问题返回零计划、精确零支持 KL 常量和中性种子。非法种子返回 `InvalidWarmStart`，不可表示的 epsilon 转换返回数值失败，不会静默退回冷启动。非零支持每次至少执行一轮，并以当前输入重新计算计划与目标；次数及残差只描述本次调用。

`balanced_sinkhorn_continuation` 校验严格递减 epsilon 日程，为各阶段提供独立预算、进度及种子。热启动可能帮助也可能阻碍收敛，不保证减少迭代或提速。见 [分阶段延续](docs/continuation.md) 与可运行的 `examples/unbalanced-continuation`。

## 可运行示例

每个目录都是实际可执行的 MoonBit 包，从仓库根目录运行：

```sh
moon run --target wasm examples/point-cloud-transport
moon run --target wasm examples/ml-reweight
moon run --target wasm examples/color-histogram
moon run --target wasm examples/divisible-allocation
moon run --target wasm examples/unbalanced-observations
moon run --target wasm examples/fixed-support-barycenter
moon run --target wasm examples/epsilon-scaling
moon run --target wasm examples/unbalanced-continuation
moon run --target wasm examples/alignment-pipeline
moon run --target wasm cmd/main
```

这些示例覆盖点云代价、分布重加权、调用方准备的颜色直方图、可分割供给配置、缺失或多余观测质量、固定支持聚合、参数延续和带断言的三域对齐。[三域对齐流程](docs/alignment-pipeline.md) 串联加权代价、求解、可行性修复、拼接和投影。图像解码、训练模型与整数优化由调用方负责。

## 检查与验证

```sh
moon info
moon fmt
moon fmt --check
moon check --deny-warn --target wasm
moon build --deny-warn --target wasm
moon test --deny-warn --target wasm
moon run --target wasm cmd/main
python3 -m unittest discover -s scripts -p test_audit_project.py
python3 scripts/audit_project.py --require-delivery
python3 scripts/verify_readme.py --target wasm
```

CI 检查格式、生成接口、源码规模及人工审阅历史，并对 wasm-gc、wasm、JavaScript、native 执行严格检查、构建、测试与全部示例。仅有配置不代表 CI 已通过，以 [实际运行记录](https://github.com/xubowen1234/moonot/actions) 为准。本机缺少 C 工具链时不声称 native 本地构建或测试成功，由真实 CI 验证。

数值参考验证采用固定 POT 版本的独立对照脚本，具体样本、依赖、误差及范围见 [参考验证](docs/reference-validation.md)。有限范围的生态查重及其检索局限见 [查重记录](docs/competition/duplicate-check.md)。

## 不支持的范围

本项目不实现通用线性／整数规划、不可分割指派、路线规划、图像解码、自动微分、GPU 内核或机器学习框架；也不承诺大规模稀疏求解、无正则精确最优解或对 POT 的完整兼容。耦合表示可分割质量计划，业务层自行解释输出。

## 来源、AI 使用与许可证

上游来源、版权及独立重写边界见 [第三方归属](THIRD_PARTY_NOTICES.md)，AI 辅助说明见 [AI_USAGE.md](AI_USAGE.md)，版本变更见 [更新日志](CHANGELOG.md)。

项目使用 Apache-2.0，全文见 [LICENSE](LICENSE)。
