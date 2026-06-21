# Replay Patch Audit — 2026-06-21

状态：`diagnostic_only`
目标：在现有 `scripts/ + configs/` 架构内修复研究口径；没有新增信号、没有优化参数、没有 alpha 声明。

## 1. 改了哪些文件

### 治理与配置

- `configs/t0_intraday_paper_agent.json`
  - 只新增治理标签：`contest_mode=true`、`risk_posture=risk_seeking`、`research_baseline=false`。
  - 没有改变该文件现有竞赛参数、执行锁或下单路径。
- `configs/t0_intraday_research_safe.json`
  - 新增独立 research-safe 配置；`mode=paper_research`、`execution_enabled=false`。
  - `auto_apply_changes=false`、`requires_human_approval=true`。
  - 恢复行业分散、相关性压力过滤、5 持仓、15% 单仓、8% 单笔、-2% 日损、0.4%/0.2% 单笔风险。
  - 固定官方确认的 131 个 T+0 代码；其余默认 T+1。
  - 路径内单边成本 5bps、单边滑点 2bps，合计至少 14bps 往返。

### Replay 与 pipeline

- `scripts/replay_t0_decisions.py`
  - 禁止同快照成交；订单最早在下一可交易 snapshot 评估。
  - 增加 order lifecycle、保守限价判断、拒单/过期/部分成交。
  - lot 级 T+0/T+1 可卖时间和 available quantity。
  - 成本内生到 cash、realized PnL、equity 和日损路径。
  - 每快照 mark-to-market，输出 cash、market value、equity、realized/unrealized PnL、drawdown。
- `scripts/run_t0_backtest_pipeline.py`
  - 强制输出 `execution_model`、`data_quality`、`result_trust_level`。
  - same-snapshot 或 full-day liquidity 自动标 `contaminated`。
  - 执行/数据证明不足自动标 `diagnostic_only`。
  - 成本已内生时，成本情景只增加额外压力，不再返还已扣成本。
- `scripts/analyze_oos_variance.py`、`scripts/run_l4_forward_validation.py`
  - 修复下游重复扣费/返还成本问题。

### Point-in-time liquidity

- 新增小型共享 helper：`scripts/point_in_time_liquidity.py`。
- 修复：
  - `research_early_entry.py`
  - `research_timing.py`
  - `research_regime.py`
  - `research_lead_lag.py`
  - `research_stops.py`
  - `research_sizing.py`
  - `fetch_yahoo_5m_quotes.py`
  - `convert_june_to_quotes.py`
  - `build_t0_replay_quotes_from_minute_data.py`
  - `run_t0_strategy_evolution.py`
- `run_t0_intraday_agent.py` 只增加未来分钟行情文件的 liquidity provenance，不改变信号。

### 测试

- `scripts/test_replay_invariants.py`
  - 新增 T51 执行、结算、成本、MTM 和信任等级测试。
  - T27 改为禁止使用后来成交额回填早盘历史。

## 2. 没改哪些文件/行为

- 没有新增交易信号。
- 没有优化 entry/exit 参数。
- 没有把 L4、i03 或任何研究结果接入实盘。
- 没有写 `latest_strategy_overlay.json`。
- 没有弱化 `mode / execution_enabled / --execute` 三重锁。
- 没有修改 SELL 前券商持仓核验或确认成交才记实盘 PnL 的逻辑。
- 没有删除旧研究文件。
- 没有把高方差竞赛配置当作 research baseline。

## 3. 修复了哪些污染

1. **同快照成交污染**：信号 snapshot 只产生 intent；下一 snapshot 才可能成交。
2. **T+1 当日可卖污染**：默认 T1；只有配置中明确列出的 T0 code 当日可卖。
3. **成本事后扣除污染**：费用与滑点直接影响现金、头寸、净实现损益和 equity 路径。
4. **成本重复处理污染**：下游 12bps 指标不再在已有 14bps 路径成本时返还 2bps。
5. **成本价估值污染**：未平仓头寸使用最新已知价格 mark-to-market。
6. **full-day liquidity 前视**：盘中研究改用当时累计 amount 与已过交易时段比例。
7. **进化缓存回填前视**：代码只从首次时点合格后进入缓存；不再把晚间合格状态回填到早盘。
8. **结果信任等级模糊**：pipeline 现在 fail-closed 输出 clean/diagnostic_only/contaminated。

## 4. 仍然存在的限制

- 旧 `yahoo_60d_quotes.jsonl` 已经在生成时删除了当日最终成交额不合格的 ETF；代码修复不能恢复这些缺失行。
- Yahoo/TDX 历史盘口为合成或缺失，仍无真实排队、深度、冲击成本。
- 简化模型只在下一 snapshot 做一次限价可成交判断；没有复杂撤改单、队列位置和逐笔撮合。
- T0 allowlist 仅包括现有官方确认的 131 个本地代码；未确认品种 fail-closed 为 T1。
- 没有 point-in-time 上市、退市、清盘、合并 ETF master，仍有幸存者偏差。
- 只有 3 天 Eastmoney 研究快照可用于 timing/regime/lead-lag/stops/sizing。
- 当前前瞻 L4 数据仍为 0/20 天。
- 当前 live 历史分钟文件没有 liquidity provenance；未来新记录会带标签，旧文件不会被倒填为已知。

## 5. 新旧 replay 的关键差异

| 项目 | 旧 replay | 新 replay |
|---|---|---|
| 成交时点 | 同 snapshot 强制成交 | 最早下一可交易 snapshot |
| 限价 | 直接按订单价成交 | 下一 snapshot 盘口满足限价才成交 |
| 生命周期 | 无 | pending/filled/partially_filled/rejected/expired |
| T 规则 | 新买数量立即 available | 默认 T1，allowlist 才 T0 |
| 成本 | replay gross，报告事后扣 | 买卖时直接进入 cash/equity |
| 未平仓估值 | 成本价 | 最新已知市价 |
| 风险路径 | 不含费用与浮亏 | 含费用、浮亏和 snapshot drawdown |
| 数据信任 | 无强制等级 | clean/diagnostic_only/contaminated |

2026-06-18 research-safe smoke replay：

- entries：3
- completed sell trades：0
- rejected orders：6
- expired orders：1
- ending open positions：3 个 T1 carry
- total equity PnL：+930.94
- max drawdown：-0.0926%
- trust：`diagnostic_only`

该单日数字不是盈利证据；它只证明新生命周期、T1 carry 和 MTM 路径运行。

## 6. 测试

命令：

```text
py -3.13 scripts/test_replay_invariants.py
```

结果：`ALL INVARIANTS PASSED`

新增要求全部覆盖：

- 同快照不成交；
- T1 当日不可卖；
- allowlist T0 当日可卖；
- 成本减少 cash/equity；
- 市价变化改变 equity/unrealized PnL；
- full-day liquidity 强制 contaminated；
- 下一 snapshot 无效价格 rejected；
- sell fill 不超过 available quantity；
- pipeline 输出 execution_model/data_quality。

## 7. 旧研究修复前后对比

| 研究 | old_result | patched_result | difference | trust_level | edge survives |
|---|---|---|---|---|---|
| opening_oos | gap-up chase OOS -0.0968%；其他规则更差 | 完全相同 | 已经使用独立 point-in-time opening 数据，不受 replay patch 影响 | diagnostic_only | 否 |
| timing | pullback forward +0.586%，1109 entries，3 天 | +0.607%，1286 entries，3 天 | universe 改为时点成交额；正点估计略升，但样本仍只有 3 天 | diagnostic_only | 未验证/否 |
| regime | trend-up momentum-reversion +0.2634%；trend-up momentum +0.0024% net | spread +0.031%；trend-up momentum -0.208% net | 原来的主要 regime 差异大幅衰减，方向性净收益转负 | diagnostic_only | 否 |
| overseas_gap | 513100/513500/518880 OOS 均为负，DSR fail | 完全相同 | 日线对齐研究不依赖 replay patch | diagnostic_only | 否 |
| lead_lag | 平均最佳净收益 -0.0193%，3 天 | +0.0064%，3 天 | 约等于 0；leader 改为各时点当时最活跃 ETF | diagnostic_only | 否 |
| stops | hold +0.5193%；stop1% +0.3933%，3 天 | hold +0.5183%；stop1% +0.3794%，3 天 | stop 仍降低均值；样本为偏上涨的 3 天 | diagnostic_only | 否 |
| sizing | equal weight -0.1669%；inverse-vol -0.1889% | -0.1854%；-0.2023% | 全部仍为负 | diagnostic_only | 否 |
| i03 full | std -24.31%；worst day +1910；PnL 12923.9 | std -8.53%；worst day +337；PnL 16803.07 | 降方差/最差日优势明显缩小；PnL 因执行口径变化不可直接横比 | contaminated | 否，旧 Yahoo universe 不可恢复 |
| L4 group | std -3.28%；worst day +1405 | std -6.05%；worst day +0 | 最差日改善消失 | contaminated | 否 |
| L4 forward | 0/20 | 0/20 | 无新前瞻日 | diagnostic_only | 样本不足 |

注意：i03 patched PnL 并不证明 edge 增强。成交时点、T1 carry、费用和 MTM 同时改变，且输入 universe 仍被旧 full-day turnover 污染。只能看作口径敏感性诊断。

## 8. 当前是否可标记 clean

**不可以。**

pipeline smoke 强制字段如下：

```json
{
  "execution_model": {
    "same_snapshot_fill": false,
    "next_snapshot_fill": true,
    "cost_in_path": true,
    "mark_to_market": true,
    "t_rule_enforced": true
  },
  "data_quality": {
    "point_in_time_liquidity": false,
    "full_day_liquidity_used": false,
    "survivor_bias_warning": true,
    "missing_data_count": 12,
    "rejected_order_count": 6
  },
  "result_trust_level": "diagnostic_only"
}
```

执行模型核心修复已生效，但旧数据 provenance、幸存者偏差、缺失记录和未平仓 T1 carry 仍阻止 clean。

## 9. 下一步还需要修什么

1. 用修复后的 Yahoo/TDX converter 从原始全量数据重新生成 point-in-time quote archive；不能复用旧筛选后文件。
2. 建立 point-in-time ETF master，包含上市、退市、清盘、合并和 T0/T1 生效日期。
3. 为真实盘口/成交回报建立 fill calibration，校准限价命中率、部分成交和滑点。
4. 对新 archive 重新冻结 baseline/OOS；旧 OOS 不再用于候选选择。
5. 累计至少 20 个 2026-06-18 之后的前瞻交易日再评估 L4。
6. 在所有 point-in-time、执行与数据质量门通过前，保持 `result_trust_level != clean`，不允许 promotion。

## 最终结论

本次修复使现有 replay 从“同快照必成交、T1 当日可卖、成本事后扣、成本价估值”变成了保守的下一快照、lot 结算、路径内费用和 MTM 模型；盘中 full-day turnover 代码路径已清除。

但旧 Yahoo60 universe 的缺失历史不可逆，幸存者偏差和真实盘口仍未解决。因此当前系统比修复前更适合产生可审计研究结果，但**现有结果仍不能标 clean，也没有发现或确认 alpha**。
