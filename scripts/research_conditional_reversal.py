"""Conditional reversal: is the (real, OOS-stable) short-term reversal edge concentrated in the
EXTREME dislocations enough to clear cost? Buy the biggest short-term losers (by mom_5m decile),
hold h bars, net of cost. If the extreme decile beats cost OOS, there's a conditional edge.
Diagnostic only."""
from __future__ import annotations
import sys
import numpy as np
sys.path.insert(0, 'scripts')
from research_signal_ic import load_bars, TRAIN_END

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    bars = load_bars()
    dates = sorted({d for d,_ in bars})
    cbd = {}
    for (d,c) in bars: cbd.setdefault(d, []).append(c)
    def fwd(b,i,h):
        if i+h>=len(b): return None
        p,pj=b[i][1],b[i+h][1]
        return pj/p-1.0 if p>0 and pj>0 else None
    # per panel: rank by mom_5m (ascending => biggest losers = strongest reversal); decile forward ret
    for h,label in [(1,'5m'),(3,'15m'),(6,'30m')]:
        # collect per (date) the mean forward ret of bottom-decile (biggest losers) and top-decile
        dl_bot=[]; dl_2=[]
        for d in dates:
            bret=[]; t2=[]
            mins=sorted({b[0] for c in cbd[d] for b in bars[(d,c)]})
            for m in mins:
                xs=[]
                for c in cbd[d]:
                    bb=bars[(d,c)]
                    idx=next((k for k,x in enumerate(bb) if x[0]==m),None)
                    if idx is None or idx<2: continue
                    mom=bb[idx][1]/bb[idx-1][1]-1.0
                    fr=fwd(bb,idx,h)
                    if fr is not None: xs.append((mom,fr))
                if len(xs)<20: continue
                xs.sort(key=lambda z:z[0])  # ascending mom: front = biggest losers
                n=len(xs); k=max(1,n//10)
                bret.append(np.mean([f for _,f in xs[:k]]))      # bottom decile (biggest losers) -> buy
                t2.append(np.mean([f for _,f in xs[n-k:]]))      # top decile (biggest winners)
            if bret: dl_bot.append((d,float(np.mean(bret)))); dl_2.append(float(np.mean(t2)))
        arr=np.array([v for _,v in dl_bot]); te=np.array([v for dd,v in dl_bot if dd>TRAIN_END])
        def stt(a): return f'{a.mean()*1e4:+.2f}bp t={a.mean()/a.std()*np.sqrt(len(a)):.1f}' if len(a)>1 and a.std() else 'na'
        print(f'reversal h={label}: BUY biggest-loser decile gross full[{stt(arr)}] TEST[{stt(te)}]')
        for cost in [0.0006,0.0010,0.0015]:
            net=arr.mean()-cost
            print(f'    net @ {cost*1e4:.0f}bp rt: {net*1e4:+.2f}bp  -> {"POSITIVE" if net>0 else "negative"}')
    print('\nbar: extreme-loser decile gross must exceed round-trip cost (6bp commission floor, ~15bp aggressive).')

if __name__=='__main__': main()
