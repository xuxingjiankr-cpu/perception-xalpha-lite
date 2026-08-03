# Perception-XAlpha V4：整体预期回报模型

## 状态

`research-only / shadow-only / diagnostic_only`。V4 是 V1–V3 之外的独立研究版本，不回写旧 artifact，不修改实盘配置、`build_decision()`、BUY/SELL gate、仓位、下单、risk gate、三重锁或 overlay，且 `orders` 永远为空。

## 目标

V2 只预测股票在每日候选池中的相对排名，不能回答整个市场下跌时“相对较强的股票是否仍有正绝对回报”。V3 增加尾亏风险，但其绝对概率未稳定校准。V4 因此保留两个旧头并新增一个独立的十日绝对预期回报头：

1. V2 相对排名头：回答“同日 Top50 中谁更强”。
2. V4 预期回报头：回答“下一交易日开盘至第十个交易日，绝对收益的条件均值是多少”。
3. V3 尾风险头：只用于同日相对风险排名，不再把未校准绝对概率直接当真实概率阈值。

预期回报是条件均值的点估计，不是保证涨幅，也不是收益上下界。

## 预注册模型

- 目标：`open[t+11] / open[t+1] - 1`。
- 模型：Ridge，`alpha=10`。
- 训练：滚动 756 个交易日，至少 504 日，每 21 日重训。
- 防泄漏：预测块与训练标签之间 purge 10 个完整交易日。
- 异常值：每个训练折内按目标的 1%/99% 分位缩尾；validation/shadow 不参与分位数计算。
- 缺失值与标准化：只用训练折的中位数和均值/标准差。
- 特征：V2/V3 已冻结的 17 个股票/因子特征与 4 个当日及以前市场特征。

## 整体评分

每日冻结因子 Top50 内：

```text
整体分数 = 1/3 × V2相对排名百分位
         + 1/3 × V4预期回报百分位
         + 1/3 × 反向尾亏概率百分位
```

选择整体分数 Top10。只有 Top10 的平均预测十日收益严格高于往返成本 0.30% 时，主策略才允许产生观察选择；否则为零只。该闸门允许现金且不强凑十只。

等权不是通过历史窗口优化出的“最佳权重”，而是只测试一次的透明预注册假设。

## 固定消融

1. 冻结因子 Top10。
2. V2 Ridge 相对排名 Top10。
3. V3 尾惩罚 Top10。
4. V4 绝对预期回报 Top10。
5. 三头整体 Top10，不加闸门。
6. 三头整体 Top10 + 正净预期回报闸门（主策略）。

## 通过要求

Validation 与 shadow 必须分别同时满足：

- 至少 25 个信号日、8 个间隔不少于十日的独立事件；
- 预期回报模型相对滚动训练均值的 OOS R² 为正；
- 每日预期回报截面 Rank IC 的 Newey-West `t >= 1.65`；
- 主策略相对 V2 Top10 的同日平均收益严格提高，且收益差 `t >= 1.65`；
- 同日胜率不得下降；
- 同日尾亏率不得上升；
- 组合十日平均收益与扣成本累计收益均为正；
- 对照必须使用主策略完全相同的交易日，减少交易次数本身不能算改善。

历史窗口即使全部通过，也只能成为新的前向 shadow 假设，不能自动连接交易。

## 学术依据

Gu、Kelly 与 Xiu 将股票风险溢价定义为未来实现收益的条件期望，并强调正则化和严格样本外比较是防止高维收益预测过拟合的核心。V4 采用低复杂度 Ridge 和滚动 purge，而没有继续复用已经失败的非线性预期回报模型或搜索多组权重。

- Gu, Kelly & Xiu (2020), *Empirical Asset Pricing via Machine Learning*, Review of Financial Studies.

## 运行

```powershell
$env:PYTHONIOENCODING='utf-8'
py -3.13 scripts/research_perception_xalpha_expected_utility_v4.py `
  --config configs/research/perception_xalpha_expected_utility_v4.json `
  --run-id run_20260803_preregistered_expected_utility_v4
```

输出独立写入：

`outputs/edge_research/perception_xalpha_expected_utility_v4/<run_id>/`

包含 `summary.json`、`report.md`、`latest_ranking.csv`。其中最新排名会同时给出预期十日收益、扣30 bps后的预期净收益、尾亏概率、三头整体分数与主闸门状态。
