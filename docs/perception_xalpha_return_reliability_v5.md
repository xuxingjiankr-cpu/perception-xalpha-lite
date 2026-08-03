# Perception-XAlpha V5：收益预测可靠性闸门

## 状态

`research-only / shadow-only / diagnostic_only`。V5 不修改实盘配置、`build_decision()`、BUY/SELL gate、仓位、下单、risk gate、三重锁或 overlay；`orders` 永远为空。

## 为什么不能继续调 V4 权重

V4 在 shadow 将平均十日收益预测为正，实际却为负，OOS R² 和 Rank IC 同时失效。此时继续寻找更漂亮的三头权重等价于在已经看过的窗口上拟合噪声。V5 的改动不是搜索新权重，而是给预期回报头增加一个严格的、训练窗内部的可靠性许可机制。

## 三层时间隔离

每个外层滚动训练窗按时间顺序切成：

1. `base-fit`：拟合 Ridge 原始预期回报模型，目标只在该段按1%/99%缩尾。
2. 十个交易日 purge。
3. `calibration-fit`：63日，只拟合一维 Ridge 校准器。
4. 十个交易日 purge。
5. `reliability-audit`：63日，既不拟合原始模型，也不拟合校准器，只决定收益头是否获准启用。
6. 外层再 purge 十个交易日，才预测下一个21日区块。

校准预测也限制在 base-fit 目标边界内，防止 V4 式极端外推。所有特征严格使用信号日及以前数据；未来十日收益只存在于离线标签表。

## 收益头启用条件

可靠性审计段必须同时满足：

- 相对 base-fit 训练均值的 OOS R² > 0；
- 每日截面 Rank IC 的 Newey-West `t >= 1.65`；
- Top50 日均收益方向准确率至少 50%；
- 校准斜率为正。

任何一项失败，预期回报头立即归零，而不是降低阈值或换参数。

## 自适应整体模型

收益头通过时：

```text
整体分数 = 40% × V2相对排名
         + 40% × 校准预期回报排名
         + 20% × 反向尾风险排名
```

Top10 平均校准预期回报必须严格高于30 bps往返成本。

收益头未通过时：

```text
整体分数 = 100% × V2相对排名
```

交易日闸门退回冻结的 V2 条件：20日市场收益为正且20日上涨宽度至少50%。该回退是 fail-closed，不会继续消费不可靠的收益预测。

## 消融与通过要求

固定比较：冻结因子 Top10、V2排名 Top10、V2排名+risk-on、固定等权三头、可靠性自适应不加闸门、可靠性自适应主策略。

Validation 和 shadow 必须分别同时满足：

- 主策略至少25个信号日和8个独立事件；
- 至少10个主策略信号日真实启用了收益头；
- 主策略在完全相同日期相对 V2 提高平均收益、胜率不降、尾亏不升，收益差 HAC `t >= 1.65`；
- 只看收益头启用日期时，也必须满足同样方向与显著性要求；
- 十日平均收益和扣成本累计收益均为正。

减少交易次数、长期禁用收益头或只靠 V2 回退，均不能被记为“预期回报模型成功”。即使全部通过，由于 V4 后已看过 validation/shadow，本次仍不能自动晋升，只能产生新的前向 shadow 假设。

## 运行

```powershell
$env:PYTHONIOENCODING='utf-8'
py -3.13 scripts/research_perception_xalpha_return_reliability_v5.py `
  --config configs/research/perception_xalpha_return_reliability_v5.json `
  --run-id run_20260803_preregistered_return_reliability_v5
```

输出目录：

`outputs/edge_research/perception_xalpha_return_reliability_v5/<run_id>/`

最新排名会记录原始/校准预期收益、收益头启用状态、审计 OOS R²、审计 Rank IC t、方向准确率、校准斜率、回退模式和最终闸门。
