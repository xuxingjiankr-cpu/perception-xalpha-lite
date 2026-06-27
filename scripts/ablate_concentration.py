"""Concentration return/risk frontier: sweep holdings-count + position cap + sector cap on the
60d replay, measure total return vs max drawdown vs daily Sharpe. Shows the risk/return tradeoff
of concentration so a risk point can be chosen -- NOT alpha, just exposure sizing. Offline."""
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
from statistics import mean, pstdev
from run_etf_paper_trading_agent import ROOT, as_float

QUOTES=ROOT/"outputs"/"t0_replay"/"yahoo_60d_quotes.jsonl"
BASE=ROOT/"configs"/"t0_intraday_paper_agent.json"
OUT=ROOT/"outputs"/"concentration_frontier"
REPLAY=ROOT/"scripts"/"replay_t0_decisions.py"

VARIANTS={
 "conc2_max":   {"th":2,"pos":0.45,"sector":False,"mps":2},
 "conc3":       {"th":3,"pos":0.33,"sector":True, "mps":2},
 "conc5_current":{"th":5,"pos":0.25,"sector":True,"mps":2},
 "conc8_div":   {"th":8,"pos":0.15,"sector":True, "mps":1},
}

def run(name,v):
    vdir=OUT/name; vdir.mkdir(parents=True,exist_ok=True)
    c=json.loads(BASE.read_text(encoding="utf-8"))
    c["strategy"]["target_holdings"]=v["th"]
    c["strategy"]["max_position_pct"]=v["pos"]
    c["risk"]["max_single_order_pct"]=v["pos"]
    c["sector_diversification"]["enabled"]=v["sector"]
    c["sector_diversification"]["max_per_sector"]=v["mps"]
    c["strategy"].setdefault("t0_entry_eligibility",{})["enabled"]=False  # replay: yahoo quotes lack asset_class
    cfg=vdir/"config.json"; cfg.write_text(json.dumps(c,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"[{name}] th={v['th']} pos={v['pos']} sector={v['sector']}/{v['mps']} running...",flush=True)
    subprocess.run([sys.executable,str(REPLAY),"--config",str(cfg),"--quotes",str(QUOTES),
        "--label",name,"--output-dir",str(vdir),"--output-detail","summary"],
        cwd=str(ROOT),capture_output=True,text=True,check=False)
    sp=vdir/f"{name}_summary.json"; s=json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {}
    pd=s.get("per_day",{}) if isinstance(s.get("per_day"),dict) else {}
    dp=[as_float(pd[d].get("net_pnl")) for d in sorted(pd)]
    tot=as_float(s.get("total_pnl")); final=as_float(s.get("final_assets"))
    init=final-tot if final else 1_000_000
    return {"name":name,"th":v["th"],"pos":v["pos"],"sector":f"{v['sector']}/{v['mps']}",
        "total_pnl":round(tot,0),"return_pct":round(tot/init*100,2) if init else None,
        "max_dd_pct":round(as_float(s.get("max_drawdown"))*100,2),
        "daily_sharpe":round(mean(dp)/pstdev(dp),3) if len(dp)>1 and pstdev(dp) else None,
        "ann_vol_pct":round(pstdev(dp)/init*100* (240**0.5),1) if len(dp)>1 and init else None,
        "open_at_end":sum(s.get("open_positions_at_end",{}).values()) if isinstance(s.get("open_positions_at_end"),dict) else None}

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True,exist_ok=True)
    res=[run(n,v) for n,v in VARIANTS.items()]
    (OUT/"frontier.json").write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding="utf-8")
    print("\n=== CONCENTRATION FRONTIER (60d replay) ===")
    print(f"{'variant':16}{'holdings':9}{'return%':9}{'maxDD%':9}{'dSharpe':9}{'annVol%':9}")
    for r in res:
        print(f"{r['name']:16}{r['th']:<9}{str(r['return_pct']):9}{str(r['max_dd_pct']):9}{str(r['daily_sharpe']):9}{str(r['ann_vol_pct']):9}")
    print("\nNote: higher return at fewer holdings = MORE beta exposure (cuts both ways), NOT edge. Sample (Mar-Jun) was up-biased.")

if __name__=="__main__": raise SystemExit(main())
