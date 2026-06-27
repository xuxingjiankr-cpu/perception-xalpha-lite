"""Generate a self-contained HTML dashboard for the isolated OBI paper account, from
outputs/l2_depth/obi_audit.json. Open outputs/l2_depth/obi_dashboard.html in a browser to
monitor: equity curve, daily P&L, taker-vs-maker net bps, scope half-spreads, recent blotter.
Regenerated nightly by run_research_suite.ps1. Diagnostic/monitor only; no orders.

Run: py -3.13 scripts/build_obi_dashboard.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from run_etf_paper_trading_agent import ROOT

SRC = ROOT / "outputs" / "l2_depth" / "obi_audit.json"
OUT = ROOT / "outputs" / "l2_depth" / "obi_dashboard.html"

HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OBI paper account monitor</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
 :root{--bg:#faf9f5;--card:#fff;--bd:#e7e5dd;--tx:#2c2c2a;--mut:#5f5e5a}
 @media(prefers-color-scheme:dark){:root{--bg:#26231f;--card:#302d28;--bd:#403c35;--tx:#ece9e2;--mut:#a8a59c}}
 body{font-family:system-ui,'Segoe UI',sans-serif;background:var(--bg);color:var(--tx);margin:0;padding:24px;line-height:1.6}
 .wrap{max-width:920px;margin:0 auto}
 h1{font-size:20px;font-weight:500;margin:0 0 4px}
 .sub{color:var(--mut);font-size:13px;margin:0 0 20px}
 .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:12px;margin-bottom:24px}
 .c{background:var(--card);border:.5px solid var(--bd);border-radius:10px;padding:14px}
 .c .l{font-size:12px;color:var(--mut)} .c .v{font-size:23px;font-weight:500;margin-top:2px}
 .panel{background:var(--card);border:.5px solid var(--bd);border-radius:12px;padding:16px;margin-bottom:18px}
 .panel h2{font-size:15px;font-weight:500;margin:0 0 12px}
 table{width:100%;border-collapse:collapse;font-size:13px} td,th{padding:6px 8px;text-align:right;border-bottom:.5px solid var(--bd)}
 th:first-child,td:first-child{text-align:left} .pos{color:#3b6d11} .neg{color:#a32d2d} .mut{color:var(--mut)}
 .wrapchart{position:relative;height:280px}
</style></head><body><div class="wrap">
<h1>OBI paper account — isolated monitor</h1>
<p class="sub">Notional __CAP__ · NOT the competition account · auto-updated nightly · __GEN__</p>
<div class="cards" id="cards"></div>
<div class="panel"><h2>Equity curve</h2><div class="wrapchart"><canvas id="eq"></canvas></div></div>
<div class="panel"><h2>Daily P&amp;L (¥)</h2><div class="wrapchart"><canvas id="pnl"></canvas></div></div>
<div class="panel"><h2>Taker vs maker — net per trade (bps)</h2><div class="wrapchart"><canvas id="trade"></canvas></div></div>
<div class="panel"><h2>Recent trades</h2><div id="blotter"></div></div>
</div>
<script>const D=__DATA__;
const pa=D.paper_account||{}, curve=pa.daily_curve||[];
const f=(n,d=0)=>n==null?'—':Number(n).toLocaleString(undefined,{maximumFractionDigits:d});
const cards=[['Equity','¥'+f(pa.final_equity)],['Total return',pa.total_return_pct==null?'—':pa.total_return_pct+'%'],
 ['Max DD',pa.max_drawdown_pct==null?'—':pa.max_drawdown_pct+'%'],['Days',pa.days||0],['Trades',pa.total_trades||0],
 ['Status',D.sufficient?'sufficient':'accumulating']];
document.getElementById('cards').innerHTML=cards.map(c=>`<div class="c"><div class="l">${c[0]}</div><div class="v">${c[1]}</div></div>`).join('');
const dk=matchMedia('(prefers-color-scheme:dark)').matches, gc=dk?'rgba(255,255,255,.1)':'rgba(0,0,0,.08)';
const base={responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{x:{grid:{color:gc}},y:{grid:{color:gc}}}};
if(curve.length){
 new Chart(eq,{type:'line',data:{labels:curve.map(c=>c.date),datasets:[{data:curve.map(c=>c.equity),borderColor:'#1D9E75',backgroundColor:'rgba(29,158,117,.12)',fill:true,tension:.2,pointRadius:2}]},options:base});
 new Chart(pnl,{type:'bar',data:{labels:curve.map(c=>c.date),datasets:[{data:curve.map(c=>c.day_pnl_yuan),backgroundColor:curve.map(c=>c.day_pnl_yuan>=0?'#639922':'#E24B4A'),borderRadius:3}]},options:base});
}else{for(const id of['eq','pnl'])document.getElementById(id).parentNode.innerHTML='<p class="mut" style="text-align:center;padding-top:110px">no trades yet — accumulates each session</p>';}
const tr=D.obi_trade||{}, hz=Object.keys(tr);
new Chart(trade,{type:'bar',data:{labels:hz,datasets:[
 {label:'taker',data:hz.map(h=>(tr[h].taker||{}).mean_bps??null),backgroundColor:'#378ADD',borderRadius:3},
 {label:'maker*',data:hz.map(h=>(tr[h].maker_optimistic||{}).mean_bps??null),backgroundColor:'#EF9F27',borderRadius:3}]},
 options:{...base,plugins:{legend:{display:true,labels:{boxWidth:10}}}}});
const bl=pa.recent_blotter||[];
document.getElementById('blotter').innerHTML=bl.length?`<table><tr><th>date</th><th>code</th><th>time</th><th>OBI</th><th>net bps</th></tr>${bl.map(b=>`<tr><td>${b.date}</td><td>${b.code}</td><td>${b.entry_ts}</td><td>${b.obi}</td><td class="${b.net_bps>=0?'pos':'neg'}">${b.net_bps}</td></tr>`).join('')}</table>`:'<p class="mut">no trades yet.</p>';
</script></body></html>"""


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    data = json.loads(SRC.read_text(encoding="utf-8")) if SRC.exists() else {}
    pa = data.get("paper_account", {})
    from datetime import datetime
    html = (HTML.replace("__DATA__", json.dumps(data, ensure_ascii=False))
            .replace("__CAP__", f"¥{pa.get('capital', 200000):,}")
            .replace("__GEN__", datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")))
    OUT.write_text(html, encoding="utf-8")
    print(f"OBI dashboard -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
