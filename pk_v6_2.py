import os, json, time, hashlib, logging
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TOKEN=os.getenv("TELEGRAM_BOT_TOKEN","").strip()
CHAT=os.getenv("TELEGRAM_CHAT_ID","").strip()
STATE=Path("pk_v6_2_state.json")
MIN_RR=4.0
ARM_BARS=30
SYMBOLS={
 "XAUUSD":"GC=F", "BTCUSD":"BTC-USD",
 "EURUSD":"EURUSD=X", "GBPUSD":"GBPUSD=X", "AUDUSD":"AUDUSD=X",
 "USDJPY":"JPY=X", "USDCAD":"CAD=X", "NZDUSD":"NZDUSD=X", "USDCHF":"CHF=X"
}
TFS={"5m":"5min","15m":"15min","30m":"30min","1h":"1h","2h":"2h","4h":"4h","1d":"1D"}
logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")

def get_data(ticker):
    # Yahoo intraday history is limited; obtain daily history separately for 1D signals.
    intra=yf.download(ticker,period="60d",interval="5m",auto_adjust=False,progress=False,threads=False)
    daily=yf.download(ticker,period="2y",interval="1d",auto_adjust=False,progress=False,threads=False)
    def clean(x):
        if x is None or x.empty: return pd.DataFrame()
        if isinstance(x.columns,pd.MultiIndex): x.columns=x.columns.get_level_values(0)
        x=x.rename(columns=str.title)
        x=x[[c for c in ["Open","High","Low","Close"] if c in x.columns]].dropna()
        x.index=pd.to_datetime(x.index,utc=True)
        return x[~x.index.duplicated(keep="last")].sort_index()
    return clean(intra),clean(daily)

def bars_for(intra,daily,tf):
    src=daily if tf=="1d" else intra
    if src.empty:return pd.DataFrame()
    rule=TFS[tf]
    if rule=="1D": return src.copy()
    return src.resample(rule,label="right",closed="right").agg(
      {"Open":"first","High":"max","Low":"min","Close":"last"}).dropna()

def ema(s,n): return s.ewm(span=n,adjust=False).mean()
def rsi(s,n=14):
    d=s.diff(); up=d.clip(lower=0).ewm(alpha=1/n,adjust=False).mean()
    dn=(-d.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean()
    rs=up/dn.replace(0,np.nan)
    return 100-100/(1+rs)
def rvi(df,n=8):
    co=df.Close-df.Open; hl=df.High-df.Low
    num=(co+2*co.shift(1)+2*co.shift(2)+co.shift(3))/6
    den=(hl+2*hl.shift(1)+2*hl.shift(2)+hl.shift(3))/6
    raw=num.rolling(n).mean()/den.rolling(n).mean().replace(0,np.nan)
    return raw.rolling(4).mean()
def atr(df,n=14):
    pc=df.Close.shift(1)
    tr=pd.concat([df.High-df.Low,(df.High-pc).abs(),(df.Low-pc).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n,adjust=False).mean()
def up(s,l,i): return s.iloc[i-1]<=l and s.iloc[i]>l
def down(s,l,i): return s.iloc[i-1]>=l and s.iloc[i]<l

def htf_ok(twoh,side):
    if len(twoh)<205:return False,"2H EMA200 unavailable"
    e=ema(twoh.Close,200).iloc[-1]; c=twoh.Close.iloc[-1]
    ok=c>e if side=="BUY" else c<e
    return bool(ok),f"2H close {c:.5g} {'>' if side=='BUY' else '<'} EMA200 {e:.5g}"

def evaluate(df,twoh,symbol,tf,strategy):
    if len(df)<220:return None
    d=df.copy(); d["E20"]=ema(d.Close,20); d["E100"]=ema(d.Close,100)
    d["RSI"]=rsi(d.Close); d["RVI"]=rvi(d); d["ATR"]=atr(d)
    i=len(d)-1; side=reason=None; filt="Not required"
    if strategy=="EMA":
        if d.E20.iloc[i-1]<=d.E100.iloc[i-1] and d.E20.iloc[i]>d.E100.iloc[i]:
            side="BUY"; reason="EMA20 crossed above EMA100"
        elif d.E20.iloc[i-1]>=d.E100.iloc[i-1] and d.E20.iloc[i]<d.E100.iloc[i]:
            side="SELL"; reason="EMA20 crossed below EMA100"
    elif strategy=="RSI":
        s=d.RSI; recent=s.iloc[max(0,i-ARM_BARS):i]
        if up(s,15,i) and (recent<15).any(): side="BUY"; reason="RSI recovered above 15 after oversold"
        elif down(s,85,i) and (recent>85).any(): side="SELL"; reason="RSI fell below 85 after overbought"
    elif strategy=="RVI":
        s=d.RVI; recent=s.iloc[max(0,i-ARM_BARS):i]
        if down(s,.5,i) and (recent>.5).any(): side="BUY"; reason="RVI returned below +0.500 after excursion"
        elif up(s,-.5,i) and (recent<-.5).any(): side="SELL"; reason="RVI returned above -0.500 after excursion"
    if not side:return None
    if strategy in ("RSI","RVI"):
        ok,filt=htf_ok(twoh,side)
        if not ok:return None
    a=float(d.ATR.iloc[i]); entry=float(d.Close.iloc[i])
    if not np.isfinite(a) or a<=0:return None
    recent=d.iloc[max(0,i-9):i+1]; pad=max(a*.15,entry*.00005)
    sl=float(recent.Low.min()-pad) if side=="BUY" else float(recent.High.max()+pad)
    risk=entry-sl if side=="BUY" else sl-entry
    if risk<=0:return None
    tp=entry+4*risk if side=="BUY" else entry-4*risk
    return dict(symbol=symbol,tf=tf,strategy=strategy,side=side,reason=reason,
      filter=filt,entry=entry,sl=sl,tp=tp,rr=4.0,stamp=str(d.index[i]),data=d.tail(100))

def draw(sig):
    d=sig["data"]; fig,(ax,ind)=plt.subplots(2,1,figsize=(12,8),sharex=True,
       gridspec_kw={"height_ratios":[3,1]})
    ax.plot(d.index,d.Close,label="Close",lw=1)
    ax.plot(d.index,d.E20,label="EMA20"); ax.plot(d.index,d.E100,label="EMA100")
    for y,label,style in [(sig["entry"],"ENTRY","-"),(sig["sl"],"SL","--"),(sig["tp"],"TP","--")]:
        ax.axhline(y,ls=style,label=f"{label} {y:.5g}")
    ax.scatter([d.index[-1]],[sig["entry"]],marker="^" if sig["side"]=="BUY" else "v",s=130)
    ax.set_title(f"PK BOT V6.2 | {sig['symbol']} {sig['tf']} | {sig['strategy']} {sig['side']} | RR 1:4")
    ax.grid(alpha=.2); ax.legend(fontsize=8,ncol=3)
    if sig["strategy"]=="RSI":
        ind.plot(d.index,d.RSI,label="RSI14"); ind.axhline(15,ls="--"); ind.axhline(85,ls="--"); ind.set_ylim(0,100)
    elif sig["strategy"]=="RVI":
        ind.plot(d.index,d.RVI,label="RVI8"); ind.axhline(.5,ls="--"); ind.axhline(-.5,ls="--")
    else:
        ind.plot(d.index,d.E20-d.E100,label="EMA20 - EMA100")
    ind.legend(fontsize=8); ind.grid(alpha=.2); fig.tight_layout()
    p=Path("/tmp")/f"{sig['symbol']}_{sig['tf']}_{sig['strategy']}.png"
    fig.savefig(p,dpi=130); plt.close(fig); return p

def send(sig,state):
    key=hashlib.sha256("|".join([sig["symbol"],sig["tf"],sig["strategy"],sig["side"],sig["stamp"]]).encode()).hexdigest()
    if key in state["sent"]:return
    path=draw(sig)
    caption=(f"PK BOT V6.2 — {sig['side']}\\nPair: {sig['symbol']} | TF: {sig['tf']}\\n"
      f"Strategy: {sig['strategy']}\\nEntry: {sig['entry']:.6g}\\nSL: {sig['sl']:.6g}\\n"
      f"TP: {sig['tp']:.6g}\\nR:R 1:4\\n{sig['reason']}\\n2H Filter: {sig['filter']}\\n"
      f"Closed candle: {sig['stamp']}\\nAlert only; TP is not guaranteed.")
    with open(path,"rb") as photo:
        r=requests.post(f"https://api.telegram.org/bot{TOKEN}/sendPhoto",
          data={"chat_id":CHAT,"caption":caption},files={"photo":photo},timeout=45)
    r.raise_for_status()
    if not r.json().get("ok"):raise RuntimeError(r.text)
    state["sent"].append(key); state["sent"]=state["sent"][-2000:]
    STATE.write_text(json.dumps(state,indent=2))
    logging.info("Signal sent: %s %s %s",sig["symbol"],sig["tf"],sig["strategy"])

def main():
    if not TOKEN or not CHAT: raise SystemExit("Missing TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID secrets")
    try: state=json.loads(STATE.read_text())
    except Exception: state={"sent":[]}
    for symbol,ticker in SYMBOLS.items():
      try:
        intra,daily=get_data(ticker)
        twoh=bars_for(intra,daily,"2h")
        for tf in TFS:
          df=bars_for(intra,daily,tf)
          if df.empty: continue
          for strat in ("EMA","RSI","RVI"):
            try:
              sig=evaluate(df,twoh,symbol,tf,strat)
              if sig:send(sig,state)
            except Exception: logging.exception("%s %s %s failed",symbol,tf,strat)
      except Exception: logging.exception("Data fetch failed: %s (%s)",symbol,ticker)
      time.sleep(.5)
    STATE.write_text(json.dumps(state,indent=2))
if __name__=="__main__": main()
