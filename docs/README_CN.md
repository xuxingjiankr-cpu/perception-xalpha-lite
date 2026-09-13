# Perception-XAlpha Lite

[English](../README.md) | **简体中文**

**发现因子，审计证据，识别过拟合回测。**

用于 point-in-time 因子研究与回测审计的 Python 工具。支持有界公式生成、因果数据对齐、
对照试验和多重检验；仅研究用途，不连接券商，不下单，不自动晋升交易策略。

[**打开45秒演示**](https://xuxingjiankr-cpu.github.io/perception-xalpha-lite/demo.html) ·
[Colab 运行](https://colab.research.google.com/github/xuxingjiankr-cpu/perception-xalpha-lite/blob/main/examples/perception_xalpha_quickstart.ipynb) ·
[样例报告](examples/audit-cases.md) · [论文与实现](PAPERS.md)

[![三个合成审计案例：噪声选优、财报披露时间、价格复权口径。](audit-demo.svg)](https://xuxingjiankr-cpu.github.io/perception-xalpha-lite/demo.html)

演示讲解真实运行产生的**合成数据结果**，不是实时推理，也不是45秒完成计算的性能承诺。
无需账户、API Key、行情数据或上传文件。

## 运行三个案例

新案例请安装当前源码版本，不假定旧版 PyPI 包已有此入口：

~~~bash
python -m pip install "perception-xalpha-lite @ git+https://github.com/xuxingjiankr-cpu/perception-xalpha-lite.git@main"
python -m xalpha_lite.audit_cases --output-dir audit-cases-output
~~~

Windows 可将 `python` 换为 `py -3.13`，无需另配命令入口的 PATH。
Colab 执行可能需要 Google 账户；本地运行和静态演示不需要。

| 案例 | 检查什么 | 不能证明什么 |
|---|---|---|
| [噪声选优](tutorials/01-noise-selection.md) | 64组随机收益也会出现高 Sharpe；用现有 DSR、CSCV/PBO 检查 | DSR 不是未来盈利概率 |
| [财报日期](tutorials/02-disclosure-timing.md) | 公告时间和修订时间不能回填到报告期末 | 对齐程序不能找回供应商遗失的历史版本 |
| [复权口径](tutorials/03-price-basis.md) | 原始金额/成交量与复权 OHLC 混用会改变截面排名 | OHLC4不是成交 VWAP；修正排名不代表有预测力 |

输出包括合成输入、逐行对比、JSON/Markdown 报告和 SHA-256 清单。
[完整复现说明](tutorials/README.md)。

## 完整研究流程

~~~bash
python -m xalpha_lite.demo_cli --output-dir xalpha-demo-output
~~~

已有演示会生成合成市场，运行数据检查、因子生成、counter/placebo、purged walk-forward、
PBO和DSR。没有幸存因子也是合法结果；合成案例通过不等于发现真实 alpha。

实际数据先运行 [data doctor](DATA_DOCTOR.md)。财报强制提供
`notice_date`，数值只能在 `max(notice_date, update_date)` 之后的首个交易日可见。

已有回测收益可在仓库副本内用 `python tools/audit_returns.py` 审计。
先填写真实试验总数，包含被丢弃的变体。本地运行不上传数据；公开 fork 中提交的数据
会公开可见，切勿提交私有策略收益。

## 机制与边界

- [架构与数学定义](ARCHITECTURE.md)：DSL、组合构建、校准、purge和研究生命周期。
- [论文到代码映射](PAPERS.md)：区分已接入流程的机制与可独立测试的原语。
- [Evidence Lab](EVIDENCE_LAB.md)：依赖感知重采样和多重检验。
- [研究贡献规范](CONTRIBUTOR_BENCHMARK.md)：评价可复现性，不按收益排名。

HMM、BOCPD、DMD、LPPLS等可用原语不等于默认流程全部启用，也不构成盈利主张。
任何历史结论都不能自动授权交易。

## 数据来源与既有记录

此次三个案例及完整流程演示使用合成数据。此前已公开的[实测研究记录](RESEARCH_RECORD_CN.md)
和 `docs/data/` 独立保留，**不是合成数据，也不是已认证业绩**。
本次未加入私人选股结果或新实测结论。[来源边界](DATA_PROVENANCE.md)。

如果工具帮你发现了研究错误，欢迎 Star 或提交可复现 issue。
MIT 开源；仅研究与教育用途，不构成投资建议。
