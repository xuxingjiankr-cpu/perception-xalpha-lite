<h1>
  <img src="logo.svg" alt="" height="76" align="left" />
  Perception-XAlpha Lite
</h1>

[![ci](https://github.com/xuxingjiankr-cpu/perception-xalpha-lite/actions/workflows/ci.yml/badge.svg)](https://github.com/xuxingjiankr-cpu/perception-xalpha-lite/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Research Status](https://img.shields.io/badge/status-research--only-7C3AED)](#研究完整性约定)
[![License](https://img.shields.io/badge/license-MIT-111827)](../LICENSE)

[English](../README.md) | **简体中文**

[检验你的回测](#检验你自己的回测) &middot; [实时记录](#预测在它对应的那个交易日之前公布) &middot; [实测偏差](#为什么门槛设得这么严) &middot; [目前活下来的](#目前活下来的) &middot; [快速开始](#快速开始) &middot; [Wiki](https://github.com/xuxingjiankr-cpu/perception-xalpha-lite/wiki)


> **量化挖因子表面上是个优化问题,实际上是个多重比较问题。** 这个框架**故意**找到更少的因子。

一个会先试图推翻自己结论的研究框架:生成公式化因子,在 point-in-time 数据上按真实成本回测,
然后在相信它之前,想尽办法证明它是假的。

它把**提出假设**和**接受证据**分开。它可以合成因子,但每个候选都必须活过时间隔离、
反事实对照、置换检验和多重检验校正。历史证据只能产出一个"待前瞻验证的假设",
永远产不出一个订单。它**刻意不是**一个交易引擎。

## 研究记录

- **[Wiki](https://github.com/xuxingjiankr-cpu/perception-xalpha-lite/wiki)** — 每个设计决定背后的实测数据,包括被否掉的假设
- **[研究看板](https://github.com/users/xuxingjiankr-cpu/projects/1)** — 在测什么、否掉了什么、什么在等前瞻数据
- **[站点](https://xuxingjiankr-cpu.github.io/perception-xalpha-lite/)** — 同样的内容,渲染版

## 检验你自己的回测

Fork 这个仓库,把你自己的日收益放进 `audit/returns.csv`,在 `audit/audit.json` 里
写清你**实际试过**多少个变体,推上去。检验在**你自己的 fork 里**运行,结论写进 Actions 摘要。
你的收益数据不离开你的仓库 —— 这边永远看不到。

```bash
python tools/audit_returns.py
```

它问三个**并不是同一个**的问题:

| | |
|---|---|
| **赢家是不是靠运气挑出来的?** | CSCV 把样本切成很多种组合,每次取样本内最优的那个变体,看它落到样本外的什么位置。超过 0.5,说明干活的是"挑选"这个动作本身。 |
| **这个 Sharpe 大到能扛住产生它的那次搜索吗?** | Deflated Sharpe 用"同样试验次数下纯噪声能给出的最好 Sharpe"去折算你观察到的值。 |
| **把整个变体族一起算进去,还有谁跑赢零吗?** | White's Reality Check,用保留序列相关性的 stationary bootstrap。 |

自带的例子是 **24 组纯随机数**。其中最好的一组年化 Sharpe **1.11** —— 这个数字大多数人会拿去实盘。
而 24 次试验下,纯噪声的期望最高值是 **1.18**。结论:`SELECTION IS DOING THE WORK`。

试验次数指你**试过**的变体数量,**包括你删掉的每一个**,不是文件里的列数。
少报它是一个回测通过本该通不过的检验的头号方式 —— 当这两个数字相等时,工具会直接警告你。

## 预测在它对应的那个交易日之前公布

[![每日轮动记录](daily-rotation.svg)](https://xuxingjiankr-cpu.github.io/perception-xalpha-lite/#live)

每天收盘后,一个冻结的规则从约 4700 只合格股票里选出一只,带时间戳提交到这里,
**在那个市场开盘之前**。文件是 append-only 的。

这个先后顺序就是全部主张。事后打分的记录永远会被问"结果出来之后你有没有偷偷改规则";
事前公开的不会 —— 它可以被证明是错的,但不能被修改。

**左边那段虚线灰底的部分什么都证明不了**,而且是故意画成那样的。那几个因子是从 456 个候选里
挑出来的,而挑选用的数据和这段窗口重叠;在这个面板上,**光是"挑选"这一步就值约 3 bps/天**。
一条样本内跑赢所有指数的曲线,正是一个被挑选出来的策略必然的样子。只有实线那段才是证据,
而今天它是空的。

两个经常被混为一谈的数字,在回测段上:

| | 累计 |
|---|--:|
| 策略,**原始收益**扣成本 —— 这个才能和指数比 | ≈ +60% |
| 可选股票池,等权 | +24.7% |
| 沪深300 | +20.3% |
| 上证50(富时A50的境内替代) | +8.6% |
| 策略,**相对股票池的超额** —— 这才是研究指标 | +18.5% |

拿超额去比原始指数是这个行当里最老的把戏,所以图上把它们画成两条线,并写明各自回答什么问题。
GitHub Action 每天独立重拉指数,任何一个已提交的基准收益漂移超过 5bp 就拒绝重绘。

## 为什么门槛设得这么严

四个偏差,每一个都是在搭这条管线的过程中,在真实股票面板上**实测**出来的,
每一个单独拿出来都足以凭空造出一个"策略":

![四个实测偏差](measured-corrections.png)

| 缺陷 | 它报告的 | 修正后剩下的 | 单位 |
|---|--:|--:|---|
| 用后见之明挑因子 | **+2.00** | −1.24 | bps/天,同面板同成本 |
| 把涨跌停封死的腿当成能成交 | **+6.05** | +0.38 | 这些腿的前瞻收益 % |
| 用全历史过滤股票池 | **391** | 77 | 第一年的合格股票数 |
| 把重叠标签当独立样本 | **−5.79** | −2.25 | 纯噪声上的 t 值 |

第一行值得停下来想一想。同样的数据、同样的成本模型、同样的组合构造,
**只有挑因子的规则不同**,差距约 3 bps/天。这比大多数已发表的股票因子结果还大 ——
意味着一条**无法审计自己挑选步骤**的管线,根本分不清"发现"和"挑选的副产品"。

这个机制在完全没有信号的合成数据上十秒就能复现:

```bash
python examples/selection_artifact.py          # 从纯噪声里造出 IR 4.53
python examples/make_corrections_figure.py     # 重新生成上面那张图
```

## 目前活下来的

一次完整的搜索:456 个已发表的公式化因子(Kakushadze 101、GTJA 191、Qlib 158、学术),
在 point-in-time A股面板上,**只用训练窗口**按十只股票组合的扣费超额排序,
然后在完全没碰过的测试窗口上报告结果。十个交易日持有,30bp 双边成本按实际换手计,
涨跌停封死的腿直接剔除而不是按价格成交。

| # | 因子 | 公式 | 读作 | 测试段净值 | >10%概率 | 最大回撤 |
|---|---|---|---|--:|--:|--:|
| 1 | `qlib158/rsqr60` | `ts_corr(close, t, 60)²` | 过去60天的走势有多"线性" —— 趋势质量,不是方向 | **+0.36%** | 1.13× | −17.9% |
| 2 | `gtja191/alpha_120` | `rank(vwap−close) / rank(vwap+close)` | 收盘价相对当日均价的位置 | −0.33% | 0.42× | −21.1% |
| 3 | `alpha101/alpha_042` | `rank(vwap−close) / rank(vwap+close)` | *和 #2 是同一个公式* | −0.33% | 0.42× | −21.1% |
| 4 | `qlib158/ma60` | `ts_mean(close, 60) / close` | 距 60 日均线下方多远 | −0.44% | 1.31× | −23.5% |
| 5 | `qlib158/sumn60` | `Σ max(−Δclose,0) / Σ \|Δclose\|` | 近 60 天的波动里有多大比例是向下的 | −0.50% | 1.31× | −28.1% |

**五个里只有一个扛住了成本。** 这就是一次干净的训练集排序的真实产出率,
也是大多数因子库从来不报的数字。

这张表里有三件事比排名本身更值钱:

- **第 2 行和第 3 行是同一个因子。** GTJA-191 的 #120 和 Kakushadze 的 #42 是逐字符相同的公式,
  发表在不同的库里、用不同的名字。把多个因子库合并、再把总数当成独立试验次数,
  会高估搜索的广度 —— deflated Sharpe 的分母该数**行为**,不是数文件。
- **第 4 行的 rank-IC t 值是 9.95,照样亏钱。** 信号是真的,而且极其显著;
  它只是换手太快,成本把它吃光了。统计显著和可交易是两个问题,只有一个能赚钱。
- **第 1 行不是方向信号。** `rsqr60` 是价格对时间回归的 R²,它衡量"趋势有多干净",不是"往哪个方向"。
  唯一扛住成本的那个候选,是个趋势质量过滤器,不是收益预测。

## 它跑的就是它描述的那套东西

这不是一个摆在研究旁边的方法库。它所服务的 A股项目的冻结前瞻记录,就跑在这个包上 ——
`build_panel`、`point_in_time_eligibility`、`long_only_book`、`score_log`,由定时任务每天追加。

把它对准真实工作,才发现了那些缺口。一次就四个:

| 用它才发现的 | 是什么 |
|---|---|
| `build_panel` 把 `limit_up`/`limit_down` 默认填 `False` | 纯 OHLCV 面板会**在封死的板上回测成交**,这种腿 +6.05% 而可成交的腿只有 +0.38% |
| 没有 point-in-time 股票池规则 | 只能手搓,而幸存者偏差正是从这里进来的 |
| 只有美元中性组合 | 几乎所有人实际在交易的构造 —— 纯多头 top-N —— 表达不出来 |
| DSL 里没有时间趋势 | 整个已发表的因子族(`ts_corr(close, t, w)²`)无法表达 |

四个现在都进库了。迁移是**验证过**而不是假设的:每个因子用两种方式在 400 个交易日 × 5169 只股票上
重算,截面 rank 差最大 **0.00e+00**;移植后的规则产出的第一个组合,和原管线的十只里有九只相同。

前瞻记录的价值完全在于它**拒绝**做什么 —— `freeze_spec` 不覆盖、`load_spec` 拒绝加载冻结后被改过的文件、
已记录的交易日再写入无效、`score_log` 只统计完全走完的持有窗口。CI 断言这四条仍然会触发,
因为一个守卫失效的前瞻记录,就只是个回测。

```bash
python examples/run_forward_record_synthetic.py
```

这个例子在**完全没有漂移的价格**上跑出 **+0.32% 净收益**,并在 n=6 时判定为
`insufficient_forward_sample`。这正是重点:那个数字是噪声,而框架会直接说出来,
而不是把它当成结果打印。

## 快速开始

```bash
pip install -e .

python examples/run_synthetic.py                    # 完整发现流程,合成数据
python examples/selection_artifact.py               # 从纯噪声造出 IR 4.53
python examples/run_forward_record_synthetic.py     # 冻结规则的前瞻记录契约
python tools/audit_returns.py                       # 检验一组收益是否过拟合
```

## 数据契约

**行情**:必须提供 `date, symbol, open, high, low, close, volume, amount`。
`limit_up`/`limit_down` 如果数据源没给,用 `apply_sealed_bar_limits` 从K线自身推断 ——
一根 `high == low` 的K线就是封板,方向看前收,不假设任何百分比阈值,所以 10%/20%/30% 三种板都能处理。

**财务**:`notice_date` 是**强制**的。一个值最早出现在 `max(notice_date, update_date)` **之后**的
第一个交易日,绝不用报告期末日期对齐。缺 `notice_date` 的行直接拒绝,不做猜测。

## 研究完整性约定

- 研究代码里**不存在**任何下单路径,CI 每次提交都会机械地断言这一点。
- 所有输出的 `orders` 字段恒为空。
- 历史证据只能产出"待前瞻验证的假设",永远产不出结论。
- 一个冻结的规则不会被修改,只会被新规则取代,而旧规则连同它已发布的记录一起留着。

## 局限

- 示例是合成数据,不构成任何盈利主张。
- 公开财务接口不一定保留每一次历史重述的版本。
- 用当期的证券主表会带来幸存者偏差,除非换成真正的 point-in-time 成分股历史。
- 零投资研究组合在只能做多的现金市场里不能直接执行。
- 等权重叠 tranche 是对持有期的近似,不建模排队优先级、融券可得性、市场冲击或场所容量。
- HMM、EWS、BOCPD、DMD、LPPLS、Black–Litterman、Triple Barrier 都是最小可审计原语,
  不是生产级求解器,也不构成任何预测能力的主张。
- Stationary bootstrap 推断依赖一个站得住脚的弱平稳近似;
  **没有任何重采样方法能修复被污染的数据或一本不完整的试验台账。**

## 许可

MIT。仅供研究使用。这里没有任何内容构成投资建议。
