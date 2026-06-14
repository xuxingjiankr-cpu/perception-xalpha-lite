# A 股 ETF Paper Trading Agent 的安全执行、日内信号与离线自适应框架

版本：2026-06-14  
状态：内部技术论文稿 / 审查稿  
项目路径：`C:\Users\XU XINGJIAN\Documents\Codex`  

## 摘要

本文描述一个用于 A 股 ETF 模拟交易竞赛的 paper trading agent 系统。系统目标不是构建可实盘部署策略，而是在严格安全约束下，验证日内 ETF 交易信号、风控门禁、多 agent 共享执行守卫、离线回放和收盘后参数进化的组合是否能提高 paper trading 决策质量。系统由 T+0 日内 ETF agent、低频 ETF paper agent、共享执行守卫、离线回放器、风险审计脚本和参数进化器组成。核心设计原则是：盘中只执行已锁定规则；参数迭代只在收盘后基于历史分钟快照离线回放；任何自动 overlay 只能覆盖白名单内的 `strategy` 参数，不能覆盖执行锁、风控硬门槛或下单路径。当前回放样本显示，最新一轮 cs.NE 风格候选参数没有优于 baseline，因此系统保留原参数并将进化结果标记为 `diagnostic_only`。本文最后列出已知局限、审查重点和后续研究方向。

关键词：ETF paper trading；T+0；日内动量；ORB；sell score；离线回放；参数进化；执行安全；A 股模拟交易

## 1. 研究定位与边界

本系统服务于 A 股 ETF paper trading 竞赛。它可以在满足三重执行锁和风控检查后提交模拟盘订单，但不涉及真实资金交易。

明确边界如下：

- 这是 paper trading research system。
- 不是 live-ready。
- 不是 formal strategy。
- 不是 investment recommendation。
- 不允许泄露 API key。
- 不允许绕过 `mode / execution_enabled / --execute` 三重锁。
- 不允许参数进化器修改 `risk`、`shared_execution`、`mode`、`execution_enabled` 或下单代码路径。

因此，本文中所有“交易”“下单”“执行”均指模拟盘 API 或离线回放语境，不构成任何实盘交易建议。

## 2. 系统架构

系统主要由七类模块组成。

### 2.1 T+0 日内 ETF Agent

主文件：

- `scripts/run_t0_intraday_agent.py`
- `configs/t0_intraday_paper_agent.json`

功能：

- 采集 T+0 ETF 实时 quote。
- 写入分钟快照。
- 计算日内信号。
- 执行入场评分和统一 sell score 出场评分。
- 在三重锁和风控全部通过后提交模拟盘订单。

当前配置中：

- `mode = paper_execute`
- `execution_enabled = true`
- 仍必须由 CLI `--execute` 触发执行路径。

### 2.2 低频 ETF Paper Agent

主文件：

- `scripts/run_etf_paper_trading_agent.py`
- `configs/etf_paper_trading_agent.json`

功能：

- ETF 低频轮动 / 配置。
- 使用历史价格得分和防御现金规则。
- 与 T+0 agent 共用执行安全原则。

低频 agent 中新增的 defensive cash rule 只阻止新建 BUY，不阻止已有仓位 SELL。

### 2.3 共享执行守卫

主文件：

- `scripts/shared_paper_trading_guard.py`

运行文件：

- `outputs/shared_order_router/shared_execution.lock`
- `outputs/shared_order_router/order_intents.jsonl`
- `outputs/shared_order_router/shared_execution_ledger.json`

作用：

- 防止 T+0 agent 与低频 agent 同时对同一账户产生冲突委托。
- 为订单打 owner tag。
- 在 agent 原有 `approved_for_submit` 之后才进入守卫。

共享执行守卫是额外保护层，不替代原有风控。

### 2.4 离线回放器

主文件：

- `scripts/replay_t0_decisions.py`

功能：

- 读取 `outputs/t0_intraday_agent/minute_quotes.jsonl`。
- 使用虚拟时钟重放 `build_decision()`。
- 本地模拟成交。
- 不调用 API。
- 不提交订单。
- 不写真实 `outputs/t0_intraday_agent/t0_state.json`。

回放器用于策略审计、参数比较和 bug 复现。

### 2.5 参数进化器

主文件：

- `scripts/run_t0_strategy_evolution.py`

输出：

- `outputs/t0_strategy_evolution/latest_strategy_overlay.json`
- `outputs/t0_strategy_evolution/latest_strategy_evolution.md`
- `outputs/t0_strategy_evolution/*candidate_results.csv`

功能：

- 每天收盘后离线回放候选参数。
- 根据 objective score 排序。
- 若候选严格优于 baseline 且满足安全闸门，则生成可次日自动应用的 overlay。
- 若无候选优于 baseline，则输出 `diagnostic_only`。

计划任务：

- 名称：`ETF T0 Strategy Evolution`
- 运行时间：周一至周五韩国时间 `16:20`
- 命令包装：`C:\CodexTasks\t0_strategy_evolution.cmd`

### 2.6 风险审计工具

已实现的离线审计包括：

- missingness stress replay：模拟行情缺失、quote stale、坏点。
- post-selection Sharpe audit：评估候选选择偏差。
- CVaR risk audit：估算尾部风险。
- shared execution guard validation：检查多 agent 执行冲突。

这些工具只用于离线诊断，不直接触发交易。

### 2.7 运行输出与状态

T+0 agent 主要输出：

- `latest_t0_decision.json`
- `t0_agent_runs.jsonl`
- `minute_quotes.jsonl`
- `minute_quotes.csv`
- `t0_state.json`
- `t0_orders_planned.csv`
- `t0_order_blotter.csv`

这些 runtime outputs 不纳入 git 追踪。

## 3. 执行安全模型

系统最重要的不变量是：任何提交模拟盘订单的路径都必须同时满足三类条件。

### 3.1 三重执行锁

订单提交需要：

1. `mode == "paper_execute"`
2. `execution_enabled == true`
3. CLI 参数 `--execute == true`

如果任一条件不满足，系统只能生成计划或观察结果，不能提交订单。

### 3.2 风控门禁

即使三重锁满足，仍需通过风险检查，包括但不限于：

- regular trading session
- open quiet period
- no new entry afternoon cutoff
- quote freshness
- quote liquidity
- broad market not declining
- daily loss limit
- daily order limit
- daily round trip limit
- daily entry limit
- reentry cooldown
- shared execution guard
- pending cancel safety

### 3.3 BUY-only 与 SELL 权限分离

系统设计要求：

- 入场相关风控只能阻止 BUY。
- SELL 出场不应被 entry-only check 阻断。
- 即使出现市场弱势、冷却期、下午新开仓禁止，也不能阻断合理出场。

这是当前代码审查中最关键的不变量之一。

## 4. 信号体系

T+0 agent 使用多源日内信号，但不强制建仓。

### 4.1 动量信号

基础信号为 snapshot momentum：

```text
momentum = current_price / anchor_price - 1
```

当历史快照不足时，使用当日 `change_pct` 作为冷启动 fallback：

```text
signal_type = "change_pct_fallback"
```

当历史足够时：

```text
signal_type = "snapshot_momentum_Nm"
```

### 4.2 ORB 开盘区间

系统在 09:30-09:44 记录开盘区间：

- `high`
- `low`
- `midpoint`
- `finalized`

09:45 后 ORB 可用于突破判断。突破条件关注价格是否超过 ORB 高点、动量是否为正、盘口中点是否上移。

### 4.3 盘整突破

系统计算最近若干快照的窄幅区间。如果区间足够窄且价格突破上沿，则给 entry score 加分。

盘整突破不单独决定入场，仍需通过全局风险检查和 entry score gate。

### 4.4 盘口压力与加速度

quote 中扩展字段包括：

- `midpoint`
- `bid_pressure_3m_pct`
- `acceleration`

这些字段用于确认短期买盘是否持续，以及动量是否出现衰减。

### 4.5 技术过滤

已加入：

- rolling VWAP entry filter
- intraday ATR stop-distance sanity check
- Bollinger squeeze breakout
- market correlation stress filter

其中 market correlation stress filter 会在 ETF 间高度相关且市场广度很弱时阻止新 BUY。

## 5. 入场逻辑

系统当前入场采用评分制。候选 ETF 需要通过：

- 非债券 ETF 过滤；
- quote ok；
- 未停牌；
- spread 与成交量过滤；
- momentum available；
- BUY-only 风控；
- entry score gate。

entry score 组件包括：

- ORB breakout
- consolidation breakout
- momentum strength
- bid pressure positive
- acceleration positive
- tight spread
- Bollinger squeeze breakout
- cross ETF divergence
- market breadth positive

只有 `entry_score >= entry_score_threshold` 且所有入场风控通过时，才会生成 BUY 订单。

## 6. 出场逻辑

系统当前采用统一 sell score 出场引擎，不再“一亏损就平仓”。

### 6.1 亏损出场

亏损持仓需要多个证据共同确认，例如：

- loss depth
- structure break
- momentum reversal
- negative bid pressure
- negative acceleration
- liquidity deterioration
- VWAP breakdown
- near close

如果亏损但 sell score 不达标，系统允许继续持有：

```text
reason = carry_allowed_sell_score_not_met
```

### 6.2 盈利出场

盈利持仓同样使用 sell score，重点关注：

- profit drawdown
- momentum negative
- acceleration negative
- bid pressure negative
- trailing drawdown

### 6.3 硬止损

`emergency_stop_pct` 是最后防线。它不依赖普通 sell score。

### 6.4 收盘附近

收盘前不再无条件强平。near close 只作为 sell score 权重项，避免机械地在不利价位卖出。

## 7. 风控体系

当前主要风控参数包括：

- `max_daily_submitted_orders`
- `max_daily_round_trips`
- `max_daily_loss_pct`
- `max_daily_api_runs`
- `max_single_order_pct`
- `min_order_quantity`
- `max_daily_cancels`
- `min_minutes_between_cancels`
- `kill_switch_file`
- `auto_cancel_pending`
- `reconcile_fills_from_trade_history`

关键原则：

- 新 BUY 更严格；
- SELL 出场不能被 BUY-only 过滤器误伤；
- cancel pending 失败时应阻断新下单；
- API 配额耗尽时进入 backoff，不继续打 API；
- kill switch 一旦存在，阻断执行。

## 8. 离线参数进化

参数进化器借鉴 recent cs.NE 中几类思想，但没有引入不可解释黑盒优化器。

### 8.1 理念来源

当前候选设计借鉴：

- mixed categorical / continuous black-box optimization：同时测试布尔开关与连续阈值。
- dynamic environment EA：只在当前策略附近做小幅突变。
- multi-objective evolutionary selection：同时看 PnL、亏损交易、最大日亏、未平仓风险。
- CMA-ES stopping criteria caution：样本小时固定预算，不无限搜索。

### 8.2 候选集合

当前候选包括：

- `baseline_current`
- `precision_entry_gate`
- `patient_exit`
- `balanced_precision`
- `tight_risk_cut`
- `trend_quality`
- `cs_ne_local_mutation_entry_plus`
- `cs_ne_local_mutation_exit_plus`
- `cs_ne_mixed_categorical_vwap_relaxed`
- `cs_ne_quality_diversity_gold_hk`
- `cs_ne_risk_first_low_budget`
- `cs_ne_patient_profit_capture`

### 8.3 自动应用条件

候选要自动应用，需要：

1. replay 成功；
2. 候选不是 baseline；
3. 交易数达到最低门槛；
4. 入场数达到最低门槛；
5. objective score 严格优于 baseline；
6. total PnL 不差于 baseline；
7. losing trades 不多于 baseline；
8. open positions 不多于 baseline；
9. overlay status 写为 `approved_for_paper_auto_apply`；
10. overlay 只包含 allowlist 路径。

否则输出：

```text
status = diagnostic_only
```

### 8.4 可覆盖参数范围

参数进化器只允许覆盖 `strategy` 白名单路径，例如：

- entry momentum threshold
- entry score threshold
- loss/profit sell score threshold
- min hold minutes
- consolidation parameters
- VWAP entry toggle
- ATR stop sanity parameters
- Bollinger squeeze parameters
- market correlation threshold
- bracket risk budget
- bracket target R multiples

绝对不能覆盖：

- `mode`
- `execution_enabled`
- `risk`
- `shared_execution`
- `skill`
- API key
- submit/cancel code path

## 9. 当前实验结果

最新 NE-style 进化回放基于已有 `minute_quotes.jsonl`，共 717 轮历史快照。

| Candidate | Entries | Trades | Win Rate | Total PnL | Max Day Loss | Open Positions | Objective |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_current | 3 | 3 | 0.00% | -1719.60 | -1201.60 | 0 | -3450.88 |
| cs_ne_patient_profit_capture | 3 | 3 | 0.00% | -1719.60 | -1201.60 | 0 | -3450.88 |
| balanced_precision | 2 | 1 | 0.00% | -1739.00 | -1739.00 | 1 | -3480.20 |
| precision_entry_gate | 3 | 3 | 0.00% | -1793.60 | -1201.60 | 0 | -3524.88 |
| cs_ne_mixed_categorical_vwap_relaxed | 3 | 3 | 0.00% | -1828.90 | -1201.60 | 0 | -3560.18 |

主要结论：

- 当前样本中 baseline 仍最优。
- `cs_ne_patient_profit_capture` 仅与 baseline 打平，不足以切换。
- 更耐心出场候选可能留下未平仓，风险评分更差。
- 当前 overlay 状态为 `diagnostic_only`。
- 次日不会自动修改参数。

## 10. 已修复的重要缺陷

历史回放曾发现两个重要线上风险，并已修复：

1. SELL 被入场检查误拦截：部分 BUY-only 风控曾可能影响 SELL 出场。已扩充卖单白名单。
2. 0 价止损事故：坏点 `currentPrice=0` 可能触发错误止损。已加入 `current > 0` 守卫和卖价兜底链。

这些修复说明离线回放器对执行安全有实际价值。

## 11. 局限性

当前系统存在明显局限：

1. 真实交易样本极小，不足以证明盈利能力。
2. 回放成交价使用 limit price 近似，不能完全代表模拟盘撮合。
3. paper trading API 与真实市场执行质量不同。
4. 参数进化容易对少量交易过拟合。
5. 当前 objective score 仍是单目标加权形式，不是真正 Pareto selection。
6. 当前没有严格 walk-forward 分日训练/验证切分。
7. API 配额耗尽可能影响行情采集和退出判断。
8. 盘中突发断网、任务调度失败、系统休眠仍需外部监控。
9. 该系统仍不能升级为 paper monitoring 之外的任何实盘形态。

## 12. 审查问题

建议 Claude 或人工审查重点检查以下问题：

1. SELL 是否仍可能被 BUY-only check 误拦截。
2. shared execution guard 是否可能阻断必要出场。
3. overlay allowlist 是否足够严格。
4. 进化器是否存在绕过安全锁的路径。
5. objective score 是否应拆成 Pareto 多目标。
6. 是否需要 walk-forward / rolling recent weighting。
7. 是否应提高最少交易数和最少交易日数。
8. 是否应新增 replay invariant tests。
9. 是否应把 quote stale 下的 SELL 逻辑单独处理。
10. 是否需要单独审计 API quota backoff 对出场的影响。
11. 是否需要将收盘前 near close 权重下调或分场景处理。
12. 是否需要为 T+0 ETF 与非 T+0 ETF 建立更明确的 inventory schema。

## 13. 后续工作

建议下一阶段只做以下增强：

1. 增加 replay invariant tests：
   - replay 不写真实 state；
   - no API calls；
   - SELL 不被 BUY-only check 阻断；
   - bad quote 不触发 0 价卖单；
   - overlay 不能覆盖非 allowlist。

2. 引入 rolling walk-forward：
   - 用前 N 日选择候选；
   - 用最近 1 日或次日验证；
   - 避免全样本后验选择。

3. 拆分多目标：
   - total PnL；
   - max day loss；
   - losing trades；
   - open positions；
   - trades count；
   - tail loss。

4. 扩大但约束候选集：
   - 只允许小幅局部突变；
   - 不允许直接调安全锁；
   - 不允许盘中在线改参数。

5. 强化报告：
   - 每日收盘报告；
   - 参数变更报告；
   - 次日启用 overlay 摘要；
   - 被拒绝候选说明。

## 14. 结论

本文描述的 ETF paper trading agent 是一个以安全执行和可审计迭代为核心的模拟交易系统。系统已经具备实时 quote 采集、T+0 日内信号、统一 sell score 出场、多 agent 共享执行守卫、离线回放和收盘后参数进化能力。当前数据尚不足以证明盈利能力，且最新 NE-style 参数候选没有优于 baseline，因此系统没有自动切换参数。这一行为符合安全优先原则。

最终结论：

- 系统可以继续作为 paper trading 竞赛研究平台运行。
- 当前不能声称 live-ready。
- 当前不能称为正式策略。
- 当前不能输出投资建议。
- 下一步应优先加强 replay invariant tests、walk-forward 评估和参数进化审计，而不是扩大下单权限。

## 附录 A：关键文件索引

| 文件 | 作用 |
|---|---|
| `scripts/run_t0_intraday_agent.py` | T+0 日内 ETF agent |
| `configs/t0_intraday_paper_agent.json` | T+0 agent 配置 |
| `scripts/run_etf_paper_trading_agent.py` | 低频 ETF paper agent |
| `configs/etf_paper_trading_agent.json` | 低频 agent 配置 |
| `scripts/shared_paper_trading_guard.py` | 多 agent 共享执行守卫 |
| `scripts/replay_t0_decisions.py` | 离线回放器 |
| `scripts/run_t0_strategy_evolution.py` | 收盘后参数进化器 |
| `outputs/t0_intraday_agent/` | T+0 runtime 输出 |
| `outputs/t0_replay/` | 回放输出 |
| `outputs/t0_strategy_evolution/` | 进化输出 |

## 附录 B：当前 Git 状态

当前分支：

```text
wip/unified-sell-score-scaffold
```

当前重要提交：

```text
4dd7ba7 Add cs.NE inspired T0 evolution candidates
24a8a61 Add paper-safe T0 strategy evolution
a278264 Add offline ETF risk audit pack
db99c15 Add market correlation stress filter
91df708 Add shared execution guard for paper agents
617a6e3 Add technical filters to T0 paper agent
```

当前标签：

```text
monday-wip-active-20260615
```

