import os, time, logging, requests, yfinance as yf, pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

TOKEN=os.getenv("TELEGRAM_BOT_TOKEN","").strip()
CHAT=os.getenv("TELEGRAM_CHAT_ID","").strip()
RR=4.0
PAIRS={"XAUUSD":"GC=F","BTCUSD":"BTC-USD","EURUSD":"EURUSD=X","GBPUSD":"GBPUSD=X","AUDUSD":"AUDUSD=X","USDJPY":"JPY=X","USDCAD":"CAD=X","USDNZD":"NZD=X","USDCHF":"CHF=X","XAGUSD":"SI=F","USOIL":"CL=F","ETHUSD":"ETH-USD"}
TFS={"5m":("5m","5d"),"15m":("15m","60d"),"30m":("30m","60d"),"1h":("60m","730d"),"2h":("60m","730d"),"4h":("60m","730d"),"1d":("1d","10y")}
logging.basicConfig(level=logging.INFO)

def getdf(ticker,tf):
    interval,period=TFS[tf]
    d=yf.download(ticker,period=period,interval=interval,progress=False,threads=False)
    if d is None or d.empty:return pd.DataFrame()
    if isinstance(d.columns,pd.MultiIndex): d.columns=d.columns.get_level_values(0)
    d=d.rename(columns=str.title)
    if not all(c in d for c in ["Open","High","Low","Close"]):return pd.DataFrame()
    d=d[["Open","High","Low","Close"]].dropna()
    if tf in ("2h","4h"):
        rule="2h" if tf=="2h" else "4h"
        d=d.resample(rule).agg({"Open":"first","High":"max","Low":"min","Close":"last"}).dropna()
    return d.iloc[:-1] if len(d)>1 else d

def rsi(c,n=14):
    x=c.diff(); g=x.clip(lower=0).ewm(alpha=1/n,adjust=False).mean()
    l=(-x.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean()
    return 100-100/(1+g/l.replace(0,np.nan))

def rvi(d,n=8):
    o=d.Open; h=d.High; l=d.Low; c=d.Close
    num=(c-o+2*(c.shift(1)-o.shift(1))+2*(c.shift(2)-o.shift(2))+(c.shift(3)-o.shift(3)))/6
    den=(h-l+2*(h.shift(1)-l.shift(1))+2*(h.shift(2)-l.shift(2))+(h.shift(3)-l.shift(3)))/6
    return (num/den.replace(0,np.nan)).rolling(n).mean()

def getfilter(ticker):
    d=yf.download(ticker,period="730d",interval="60m",progress=False,threads=False)
    if d is None or d.empty:return None
    if isinstance(d.columns,pd.MultiIndex):d.columns=d.columns.get_level_values(0)
    c=d["Close"].dropna().resample("2h").last().dropna()
    if len(c)>1:c=c.iloc[:-1]
    if len(c)<205:return None
    return float(c.iloc[-1]),float(c.ewm(span=200,adjust=False).mean().iloc[-1])

def scan(pair,tf,d,f):
    if len(d)<205:return None
    c=d.Close; e20=c.ewm(span=20,adjust=False).mean(); e100=c.ewm(span=100,adjust=False).mean()
    rs=rsi(c); rv=rvi(d)
    prev= c.shift(1)
    tr=pd.concat([d.High-d.Low,(d.High-prev).abs(),(d.Low-prev).abs()],axis=1).max(axis=1)
    a=tr.rolling(14).mean().iloc[-1]
    if not np.isfinite(a) or a<=0:return None
    out=[]
    if e20.iloc[-2]<=e100.iloc[-2] and e20.iloc[-1]>e100.iloc[-1]:out.append(("BUY","EMA 20/100","EMA20 crossed above EMA100"))
    if e20.iloc[-2]>=e100.iloc[-2] and e20.iloc[-1]<e100.iloc[-1]:out.append(("SELL","EMA 20/100","EMA20 crossed below EMA100"))
    if f and len(rs.dropna())>3:
        if min(rs.iloc[-4:-1])<15 and rs.iloc[-2]<=15<rs.iloc[-1] and f[0]>f[1]:out.append(("BUY","RSI 14","RSI recovered above 15"))
        if max(rs.iloc[-4:-1])>85 and rs.iloc[-2]>=85>rs.iloc[-1] and f[0]<f[1]:out.append(("SELL","RSI 14","RSI returned below 85"))
    if f and len(rv.dropna())>3:
        if min(rv.iloc[-4:-1])<-.5 and rv.iloc[-2]<=-.5<rv.iloc[-1] and f[0]>f[1]:out.append(("BUY","RVI 8","RVI recovered above -0.500"))
        if max(rv.iloc[-4:-1])>.5 and rv.iloc[-2]>=.5>rv.iloc[-1] and f[0]<f[1]:out.append(("SELL","RVI 8","RVI returned below +0.500"))
    if not out:return None
    side,strategy,why=out[-1]; entry=float(c.iloc[-1]); dist=max(float(a)*1.5,entry*.0005)
    sl,tp=(entry-dist,entry+dist*4) if side=="BUY" else (entry+dist,entry-dist*4)
    if min(sl,tp)<=0:return None
    return dict(pair=pair,tf=tf,side=side,strategy=strategy,why=why,entry=entry,sl=sl,tp=tp,filter=f,time=d.index[-1])

def send(s,d):
    view=d.tail(65); fig,ax=plt.subplots(figsize=(10,5))
    for i,(_,r) in enumerate(view.iterrows()):
        col="#15966a" if r.Close>=r.Open else "#d94b4b"
        ax.vlines(i,r.Low,r.High,color=col)
        ax.add_patch(Rectangle((i-.25,min(r.Open,r.Close)),.5,max(abs(r.Close-r.Open),1e-8),color=col))
    ax.plot(range(len(view)),view.Close.ewm(span=20).mean(),label="EMA20")
    ax.plot(range(len(view)),view.Close.ewm(span=100).mean(),label="EMA100")
    for val,label,col in [(s["entry"],"ENTRY","#2878d0"),(s["sl"],"SL","#d94b4b"),(s["tp"],"TP","#15966a")]:
        ax.axhline(val,ls="--",color=col,label=f"{label} {val:.6g}")
    ax.annotate(s["side"],(len(view)-1,s["entry"]),xytext=(len(view)-12,s["entry"]),arrowprops={"arrowstyle":"->"})
    ax.set_title(f'{s["pair"]} {s["tf"]} | {s["strategy"]} | R:R 1:4');ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout()
    path="/tmp/pk_v62.png";fig.savefig(path,dpi=130);plt.close(fig)
    f=s["filter"]
    fs="Not required for EMA" if s["strategy"].startswith("EMA") else (f"2H close {f[0]:.6g} {'>' if f[0]>f[1] else '<'} EMA200 {f[1]:.6g}" if f else "Unavailable")
    t=pd.Timestamp(s["time"]);t=t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    msg=(f'{"🟢 BUY" if s["side"]=="BUY" else "🔴 SELL"} | PK BOT V6.2\\n━━━━━━━━━━━━━━\\n'
         f'📌 Pair: {s["pair"]}    ⏱ TF: {s["tf"]}\\n🧭 Strategy: {s["strategy"]}\\n━━━━━━━━━━━━━━\\n'
         f'🎯 Entry: {s["entry"]:.6g}\\n🛑 SL: {s["sl"]:.6g}\\n💰 TP: {s["tp"]:.6g}\\n⚖️ R:R = 1:4\\n'
         f'🔎 Confirmation: {s["why"]}\\n📐 2H Filter: {fs}\\n🕒 Closed candle: {t:%Y-%m-%d %H:%M UTC}\\n⚠️ Alert only; TP not guaranteed.').replace("\\n","\n")
    with open(path,"rb") as p:
        r=requests.post(f"https://api.telegram.org/bot{TOKEN}/sendPhoto",data={"chat_id":CHAT,"caption":msg},files={"photo":p},timeout=35)
    r.raise_for_status()
    if not r.json().get("ok"):raise RuntimeError(r.text)

def main():
    if not TOKEN or not CHAT:raise SystemExit("Missing Telegram GitHub secrets")
    for pair,ticker in PAIRS.items():
        for tf in TFS:
            try:
                d=getdf(ticker,tf)
                if d.empty:continue
                f=getfilter(ticker)
                s=scan(pair,tf,d,f)
                if s:send(s,d);logging.info("Sent %s %s %s",pair,tf,s["side"]);time.sleep(1)
            except Exception:logging.exception("Skip %s %s",pair,tf)
if __name__=="__main__":main()
