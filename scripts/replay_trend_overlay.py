"""Replay the current strategy, then overlay the date-varying trend filter to show its effect on
return/drawdown. Baseline replay = flat deployment (trend dormant); overlay scales each day's PnL
by that day's deploy_factor (沪深300 vs 50d MA). Honest approximation; diagnostic."""
from __future__ import annotations
import json, subprocess, sys, urllib.request
from pathlib import Path
from run_etf_paper_trading_agent import ROOT, as_float
QUOTES=ROOT/"outputs"/"t0_replay"/"yahoo_60d_quotes.jsonl"; BASE=ROOT/"configs"/"t0_intraday_paper_agent.json"
OUT=ROOT/"outputs"/"trend_overlay"; OUT.mkdir(parents=True,exist_ok=True)
def main():
    sys.stdout.reconfigure(encoding="utf-8")
    c=json.loads(BASE.read_text(encoding="utf-8"))
    c["strategy"].setdefault("t0_entry_eligibility",{})["enabled"]=False  # replay: yahoo lacks asset_class
    c["strategy"].setdefault("trend_deployment",{})["enabled"]=False      # baseline = flat (overlay applied here)
    cfg=OUT/"config.json"; cfg.write_text(json.dumps(c,ensure_ascii=False),encoding="utf-8")
    print("running baseline replay (flat deployment)...",flush=True)
    subprocess.run([sys.executable,str(ROOT/"scripts"/"replay_t0_decisions.py"),"--config",str(cfg),
        "--quotes",str(QUOTES),"--label","trend_base","--output-dir",str(OUT),"--output-detail","summary"],
        cwd=str(ROOT),capture_output=True,text=True,check=False)
    s=json.loads((OUT/"trend_base_summary.json").read_text(encoding="utf-8"))
    pd=s.get("per_day",{}); dates=sorted(pd); daily=[as_float(pd[d].get("net_pnl")) for d in dates]
    final=as_float(s.get("final_assets")); init=final-as_float(s.get("total_pnl")) or 1_000_000
    # 沪深300 daily -> per-date 50d MA regime
    sym="510300.SS"; url=f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?interval=1d&range=1y"
    d=json.loads(urllib.request.urlopen(urllib.request.Request(url,headers={"User-Agent":"Mozilla/5.0"}),timeout=20).read())
    res=d["chart"]["result"][0]; ts=res["timestamp"]; cl=res["indicators"]["quote"][0]["close"]
    import datetime
    idx={}  # date -> (close, ma50)
    closes=[(datetime.datetime.utcfromtimestamp(t).strftime("%Y-%m-%d"),c) for t,c in zip(ts,cl) if c]
    vals=[c for _,c in closes]
    for i,(dt,c) in enumerate(closes):
        if i>=49: idx[dt]=(c, sum(vals[i-49:i+1])/50)
    def factor(dt):
        # nearest prior index date
        for k in sorted([x for x in idx if x<=dt],reverse=True):
            cl_,ma=idx[k]; return 1.0 if cl_>=ma*0.99 else 0.3
        return 1.0
    facs=[factor(dt) for dt in dates]
    # cumulative equity baseline vs trend-overlaid
    eqb=init; eqt=init; cb=[]; ct=[]; ddb=ddt=0; pkb=pkt=init
    for p,f in zip(daily,facs):
        eqb+=p; eqt+=p*f
        pkb=max(pkb,eqb); pkt=max(pkt,eqt); ddb=min(ddb,eqb/pkb-1); ddt=min(ddt,eqt/pkt-1)
        cb.append(round((eqb/init-1)*100,2)); ct.append(round((eqt/init-1)*100,2))
    out={"dates":dates,"baseline_cum":cb,"trend_cum":ct,"deploy_factors":facs,
         "baseline":{"return_pct":round((eqb/init-1)*100,2),"maxDD_pct":round(ddb*100,2)},
         "trend":{"return_pct":round((eqt/init-1)*100,2),"maxDD_pct":round(ddt*100,2)},
         "downtrend_days":sum(1 for f in facs if f<1)}
    (OUT/"overlay.json").write_text(json.dumps(out,ensure_ascii=False),encoding="utf-8")
    print(f"\nBASELINE: return {out['baseline']['return_pct']}% maxDD {out['baseline']['maxDD_pct']}%")
    print(f"TREND   : return {out['trend']['return_pct']}% maxDD {out['trend']['maxDD_pct']}%")
    print(f"downtrend days (de-risked): {out['downtrend_days']}/{len(dates)}")
if __name__=="__main__": main()
