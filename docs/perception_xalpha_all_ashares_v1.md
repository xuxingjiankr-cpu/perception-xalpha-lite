# Perception-XAlpha：全 A 股研究宇宙 v1

## 结论边界

本扩展把原先只覆盖 ETF 文件的自主因子挖掘器，扩展到当前可发现的沪、
深、北 A 股。它仍然是 `research-only / shadow-only`，不读取交易账户，
不修改交易配置、观察池、仓位、BUY/SELL gate、risk gate、策略 overlay
或三重执行锁，也不会自动推广任何因子。

“全 A 股”在本版本中的准确含义是：

- 当前能够从沪深 TDX 主数据和北交所公开主数据发现的 A 股；
- 包含沪市主板、科创板、深市主板、创业板和北交所；
- 不是历史时点成分数据库。已经退市的证券与历史 ST 状态不完整，因此
  存在幸存者偏差和成员偏差。报告会始终保留这项警告。

## 数据层

入口脚本：

```powershell
py -3.13 scripts\collect_ashare_research_daily.py --mode master
py -3.13 scripts\collect_ashare_research_daily.py --mode backfill --workers 4 --max-pages 4
py -3.13 scripts\collect_ashare_research_daily.py --mode audit
```

数据保存在：

```text
data/market/ashare_research/master/ashare_master_latest.jsonl
data/market/ashare_research/bars_1d_raw/SH_600000.jsonl
data/market/ashare_research/bars_1d_raw/SZ_000001.jsonl
data/market/ashare_research/bars_1d_raw/BJ_920000.jsonl
```

沪深 OHLCV/成交额来自 mootdx/TDX。北交所 OHLCV 来自新浪公开 K 线；
由于该接口不返回成交额，北交所成交额以 `OHLC4 × volume` 估算，并在每
行和审计报告中明确标记，不能伪装成交易所原始成交额。

所有价格按不复权原始价格保存。加载器不前向填充，并拒绝：

- OHLC 关系错误；
- 历史条数不足；
- 中位成交额不足；
- 停牌比例或缺失 bar 比例过高；
- 原始价格中极端跳变占比异常（可能是公司行动或坏数据）。

## 失败关闭

全 A 股历史验证只有在以下条件同时满足时才运行：

- 主数据文件覆盖率至少 95%；
- 至少 3,000 只股票通过历史、成交额和数据质量门；
- 沪、深、北三个交易所均有合格样本。

任何一项不足，研究器直接报错，不会用少量便利样本冒充“全 A 股”结果。
初始采集可用 `--max-codes` 做采集器烟雾测试，但烟雾数据不能进入验证。

## 研究口径

配置：

```text
configs/research/perception_xalpha_all_ashares_v1.json
```

相对 ETF 版本，因子 DSL、机制假设、Primary/Counter/Placebo、train-only
进化、purged walk-forward、PBO/DSR 和验证隔离全部复用。股票研究按预
注册的 30 bps 往返成本计算，避免沿用 ETF 的 15.5 bps 低成本假设。

运行：

```powershell
py -3.13 scripts\research_perception_xalpha_autonomous.py `
  --config configs\research\perception_xalpha_all_ashares_v1.json run
```

独立输出：

```text
outputs/edge_research/perception_xalpha_all_ashares/
```

ETF 版本的数据库、父代记忆和输出不会与股票版本混合。

## 自动循环

`scripts/run_perception_xalpha_all_ashares_weekly.ps1` 按顺序刷新主数据、
补充日线、执行数据审计，然后才运行研究循环。已有文件只拉取最近一页并
与历史原子合并；首次采集才拉取完整注册页数。

本版本不自动创建计划任务。计划任务属于部署动作，应在主数据首次完整
回填、审计达到 95% 且人工确认运行耗时后单独创建。

## 仍未解决

1. 当前主数据造成的退市股幸存者偏差；
2. 历史 ST 状态与不同涨跌停制度的精确时点数据；
3. 北交所真实成交额缺失；
4. 原始价格公司行动与分红拆股的完备事件表；
5. 行业分类的历史时点变化。

所以即使某因子通过历史验证，也只能成为新的前向 shadow 假设，不能直接
进入纸面或正式交易。
