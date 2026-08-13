const $ = (selector) => document.querySelector(selector);

const pct = (value, digits = 2) => {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return `${(Number(value) * 100).toFixed(digits)}%`;
};

const metricClass = (value, threshold = 0) => {
  if (value === null || value === undefined) return "neutral";
  return Number(value) > threshold ? "positive" : Number(value) < threshold ? "negative" : "neutral";
};

const escapeHtml = (value) => String(value ?? "")
  .replaceAll("&", "&amp;")
  .replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;")
  .replaceAll('"', "&quot;")
  .replaceAll("'", "&#039;");

function sourceLabel(row) {
  if (!row.completeTwelveFactorEstimate) return "非完整12因子（拒绝）";
  return row.imputedFactorCount > 0
    ? `完整12因子（中性补全${row.imputedFactorCount}项）`
    : "完整12因子（无缺失）";
}

const signedPct = (value, digits = 3) => {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  const number = Number(value) * 100;
  return `${number >= 0 ? "+" : ""}${number.toFixed(digits)}%`;
};

function shadowMetric(row, field, deltaField, digits = 2, tail = false) {
  const shadow = row.fundamentalInteractionShadow;
  if (!shadow) return '<small class="shadow-metric unavailable">基本面交互：不可用</small>';
  const delta = Number(shadow[deltaField]);
  const improved = tail ? delta < 0 : delta > 0;
  return `<small class="shadow-metric ${improved ? "improved" : "weakened"}">交互 ${pct(shadow[field], digits)} <em>${signedPct(delta, digits + 1)}</em></small>`;
}

function renderRows(rows) {
  $("#top10Body").innerHTML = rows.map((row) => `
    <tr>
      <td><span class="rank">#${row.rank}</span></td>
      <td><span class="stock-name">${escapeHtml(row.name)}</span><span class="stock-code">${row.securityId}</span></td>
      <td><span class="metric ${metricClass(row.probabilityUp, .5)}">${pct(row.probabilityUp)}</span>${shadowMetric(row, "probabilityUp", "deltaProbabilityUp")}</td>
      <td><span class="metric ${metricClass(row.expectedGrossReturn)}">${pct(row.expectedGrossReturn, 3)}</span>${shadowMetric(row, "expectedGrossReturn", "deltaExpectedGrossReturn", 3)}</td>
      <td><span class="metric ${row.probabilityTailLoss > .08 ? "negative" : "neutral"}">${pct(row.probabilityTailLoss)}</span>${shadowMetric(row, "probabilityTailLoss", "deltaProbabilityTailLoss", 2, true)}</td>
      <td><span class="source-badge ${row.completeTwelveFactorEstimate ? "" : "fallback"}">${sourceLabel(row)}</span>${row.fundamentalInteractionShadow ? '<small class="shadow-badge">+ 基本面×价量影子</small>' : ''}</td>
    </tr>
  `).join("");
}

function renderSearch(rows) {
  if (!rows?.length) {
    $("#searchResult").className = "search-result empty";
    $("#searchResult").textContent = "没有找到该股票，或它不在当前有效研究横截面中。";
    return;
  }
  $("#searchResult").className = "search-result";
  $("#searchResult").innerHTML = rows.slice(0, 8).map((row) => `
    <div class="result-card">
      <div class="result-cell result-stock">
        <small>横截面排名</small>
        <strong>#${row.rank} · ${escapeHtml(row.name)}</strong>
        <span>${row.securityId} · ${row.nameInitials || "—"} · ${sourceLabel(row)}</span>
      </div>
      <div class="result-cell"><small>预计上涨概率</small><strong class="${metricClass(row.probabilityUp, .5)}">${pct(row.probabilityUp)}</strong>${shadowMetric(row, "probabilityUp", "deltaProbabilityUp")}</div>
      <div class="result-cell"><small>预计毛涨幅</small><strong class="${metricClass(row.expectedGrossReturn)}">${pct(row.expectedGrossReturn, 3)}</strong>${shadowMetric(row, "expectedGrossReturn", "deltaExpectedGrossReturn", 3)}</div>
      <div class="result-cell"><small>尾亏概率</small><strong class="${row.probabilityTailLoss > .08 ? "negative" : "neutral"}">${pct(row.probabilityTailLoss)}</strong>${shadowMetric(row, "probabilityTailLoss", "deltaProbabilityTailLoss", 2, true)}</div>
    </div>`).join("");
}

async function loadLatest() {
  const response = await fetch("/api/latest", { cache: "no-store" });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const data = await response.json();
  renderRows(data.top10);
  $("#signalDate").textContent = data.signalDate;
  $("#effectiveDate").textContent = data.intendedTradingSession;
  $("#universeCount").textContent = `${data.securityCount.toLocaleString()} 只有效股票`;
  $("#freshness").className = "freshness";
  $("#freshness").innerHTML = `<strong>适用 ${data.intendedTradingSession}</strong>信号收盘日 ${data.signalDate}`;
  const reliability = data.forecastReliability;
  const low = reliability.status.includes("low_confidence");
  const distribution = data.forecastDistribution || {};
  const spread = distribution.top10ExpectedReturnSpread;
  const probabilitySpread = distribution.top10ProbabilityUpSpread;
  const directionWeak = probabilitySpread === null || probabilitySpread === undefined || probabilitySpread < .05;
  const interaction = data.fundamentalInteractionShadow;
  $("#reliabilityTitle").textContent = low ? "当前预测可信度较低" : "历史区分度通过，仍需前向验证";
  const interactionText = interaction
    ? ` 基本面交互已作为影子层加入：验证/影子 Top10 尾亏 AUC ${Number(interaction.validationTop10TailAuc ?? 0).toFixed(3)}/${Number(interaction.shadowTop10TailAuc ?? 0).toFixed(3)}；历史总门未通过，不改变排名。`
    : " 基本面交互影子层尚无同日结果。";
  $("#reliabilityText").textContent = `原12因子影子期上涨 AUC ${Number(reliability.shadowUpAuc ?? 0).toFixed(3)}，尾亏 AUC ${Number(reliability.shadowTailAuc ?? 0).toFixed(3)}；Top10 预计涨幅跨度 ${pct(spread, 3)}，上涨概率跨度 ${pct(probabilitySpread, 3)}。${directionWeak ? "方向概率区分力不足，不能判断确定上涨或下跌。" : "方向概率存在横截面差异，仍不代表确定涨跌。"}${interactionText}`;
  $("#definitions").textContent = `${data.forecastHorizon}；尾亏：${data.tailLossDefinition}。`;
}

$("#searchForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = $("#searchInput").value.trim();
  if (!query) return;
  $("#searchResult").className = "search-result empty loading";
  $("#searchResult").textContent = "查询中…";
  try {
    const response = await fetch(`/api/search?q=${encodeURIComponent(query)}`, { cache: "no-store" });
    const data = await response.json();
    renderSearch(data.matches || []);
  } catch (error) {
    $("#searchResult").className = "search-result empty";
    $("#searchResult").textContent = `查询失败：${error.message}`;
  }
});

loadLatest().catch((error) => {
  $("#freshness").className = "freshness";
  $("#freshness").textContent = `加载失败：${error.message}`;
  $("#top10Body").innerHTML = '<tr><td colspan="6" class="loading-row">尚无可用快照，请先运行每日预测。</td></tr>';
});
