# MoonOT

MoonOT 是纯 MoonBit 实现的密集、有限支持熵正则最优传输库。公开 API 提供 `balanced_sinkhorn`、`unbalanced_sinkhorn`、`balanced_sinkhorn_divergence` 和共享／异构代价固定支持重心，可复用于传输计划、去偏分布比较和分布聚合。

库显式报告目标分量、终止状态、数值残差及零质量边界。残差不是最优间隙证书。安装、示例与范围见 [中文 README](README.md)，公式与数值限制见 [数值契约](docs/algorithm-contract.md) 和 [算法推导](docs/algorithm-derivation.md)。
