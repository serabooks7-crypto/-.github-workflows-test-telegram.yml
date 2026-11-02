
import time
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore")

# =====================================================
# PK V6.1 | FIXED 1:4 R:R BACKTEST
# =====================================================

DAYS = 90
MIN_RR = 4.0
TIMEFRAMES = ["5m", "15m", "30m", "1h", "4h", "1d"]
MARKETS = ["XAUUSD", "BTCUSD"]

# User-provided XAUUSD Standard assumptions
XAU_SPREAD = 0.30
XAU_SLIPPAGE = 0.05

# Assumed BTC Futures costs; change to actual venue fees
BTC_FEE = 0.0005
BTC_SLIPPAGE = 0.0001
BTC_SPREAD = 0.0

OUT = Path("backtest_results")
OUT.mkdir(exist_ok=True)


# ---------------- DATA ----------------

def get_btc(interval):
    url = "https://fapi.binance.com/fapi/v1/klines"
    end = int(time.time() * 1000)
    start = end - DAYS * 86400000
    rows = []

    while start < end:
        r = requests.get(url, params={
            "symbol": "BTCUSDT",
            "interval": interval,
            "startTime": start,
            "endTime": end,
            "limit": 1000
        }, timeout=30)
        r.raise_for_status()
        batch = r.json()

        if not batch:
            break

        rows.extend(batch)
        start = batch[-1][0] + 1
        if len(batch) < 1000:
            break
        time.sleep(0.12)

    if not rows:
        return pd.DataFrame()

    d = pd.DataFrame(rows, columns=[
        "time", "Open", "High", "Low", "Close", "Volume",
        "close_time", "qv", "trades", "tb", "tq", "ignore"
    ])
    d["time"] = pd.to_datetime(d["time"], unit="ms", utc=True)
    d["close_time"] = pd.to_datetime(
        d["close_time"], unit="ms", utc=True
    )
    d = d.set_index("time")
    d = d[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    now = pd.Timestamp.now(tz="UTC")
    d = d[d.index + pd.Timedelta(milliseconds=1) <= now]
    return d[~d.index.duplicated(keep="last")].sort_index()


def get_gold(interval):
    period = "60d" if interval != "1d" else "2y"
    d = yf.download(
        "GC=F", period=period, interval=interval,
        auto_adjust=False, progress=False, threads=False
    )
    if d.empty:
        return pd.DataFrame()

    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)

    d = d[["Open", "High", "Low", "Close", "Volume"]]
    idx = pd.to_datetime(d.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    d.index = idx
    d = d.dropna(subset=["Open", "High", "Low", "Close"])
    d = d[~d.index.duplicated(keep="last")].sort_index()

    # Apply the 90-day cutoff to DAILY data too.
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=DAYS)
    d = d[d.index >= cutoff]

    # Exclude currently forming candle.
    duration = pd.Timedelta(interval)
    d = d[d.index + duration <= pd.Timestamp.now(tz="UTC")]
    return d.astype(float)


def resample_data(d, rule):
    if d.empty:
        return d
    return d.resample(rule, label="left", closed="left").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna(subset=["Open", "High", "Low", "Close"])


def load_market(market):
    result = {}
    if market == "BTCUSD":
        for tf in TIMEFRAMES:
            result[tf] = get_btc(tf)
    else:
        for tf in ["5m", "15m", "30m", "1h", "1d"]:
            interval = "60m" if tf == "1h" else tf
            result[tf] = get_gold(interval)
        # Gold 4H is built from hourly candles.
        result["4h"] = resample_data(result["1h"], "4h")
    return result


# ---------------- INDICATORS ----------------

def indicators(d):
    d = d.copy()

    for n in [20, 50, 200]:
        d[f"EMA{n}"] = d.Close.ewm(
            span=n, adjust=False
        ).mean()

    change = d.Close.diff()
    gain = change.clip(lower=0)
    loss = -change.clip(upper=0)
    ag = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    al = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = ag / al.replace(0, np.nan)
    d["RSI"] = 100 - 100 / (1 + rs)

    d["RVI"] = (
        (d.Close - d.Open).rolling(8).mean()
        / (d.High - d.Low).rolling(8).mean().replace(0, np.nan)
    )
    return d


# ---------------- HTF BIAS ----------------

def htf_row(d, timestamp, duration):
    if d.empty:
        return None
    # Only candles whose closing time is at or before timestamp.
    valid = d[d.index + duration <= timestamp]
    if valid.empty:
        return None
    return valid.iloc[-1]


def bias_at(h1, h4, signal_close):
    a = htf_row(h1, signal_close, pd.Timedelta("1h"))
    b = htf_row(h4, signal_close, pd.Timedelta("4h"))

    if a is None or b is None:
        return "UNKNOWN"

    bull = (
        a.EMA20 > a.EMA50 > a.EMA200
        and b.EMA20 > b.EMA50 > b.EMA200
    )
    bear = (
        a.EMA20 < a.EMA50 < a.EMA200
        and b.EMA20 < b.EMA50 < b.EMA200
    )

    if bull:
        return "BULLISH"
    if bear:
        return "BEARISH"
    return "UNKNOWN"


# ---------------- LIQUIDITY ----------------

def liquidity(history):
    levels = []
    h = history.High.to_numpy()
    l = history.Low.to_numpy()

    # Only confirmed pivots; right-side candles are already closed.
    for i in range(3, len(history) - 3):
        if h[i] == max(h[i-3:i+4]):
            levels.append(("HIGH", float(h[i])))
        if l[i] == min(l[i-3:i+4]):
            levels.append(("LOW", float(l[i])))

    daily = resample_data(history, "1D")
    weekly = resample_data(history, "1W")

    if len(daily) >= 2:
        levels += [
            ("HIGH", float(daily.High.iloc[-2])),
            ("LOW", float(daily.Low.iloc[-2]))
        ]

    if len(weekly) >= 2:
        levels += [
            ("HIGH", float(weekly.High.iloc[-2])),
            ("LOW", float(weekly.Low.iloc[-2]))
        ]

    return levels


# ---------------- SIGNAL ----------------

def find_signal(hist, bias, diag):
    if len(hist) < 220:
        diag["insufficient_history"] += 1
        return None

    p = hist.iloc[-2]
    c = hist.iloc[-1]

    if bias == "UNKNOWN":
        diag["unknown_bias"] += 1
        return None

    side = "BUY" if bias == "BULLISH" else "SELL"

    if side == "BUY":
        ema_ok = (
            c.Low <= c.EMA20 and c.Close > c.EMA20
            and c.EMA20 > c.EMA50 > c.EMA200
        )
        rsi_armed = (hist.RSI.iloc[-51:-1] < 10).any()
        rsi_ok = p.RSI <= 10 and c.RSI > 10
        rvi_armed = (hist.RVI.iloc[-51:-1] > 0.5).any()
        rvi_ok = p.RVI >= 0.5 and c.RVI < 0.5
    else:
        ema_ok = (
            c.High >= c.EMA20 and c.Close < c.EMA20
            and c.EMA20 < c.EMA50 < c.EMA200
        )
        rsi_armed = (hist.RSI.iloc[-51:-1] > 90).any()
        rsi_ok = p.RSI >= 90 and c.RSI < 90
        rvi_armed = (hist.RVI.iloc[-51:-1] < -0.5).any()
        rvi_ok = p.RVI <= -0.5 and c.RVI > -0.5

    if not ema_ok:
        diag["ema_retest_fail"] += 1
        return None
    if not (rsi_armed and rsi_ok):
        diag["rsi_fail"] += 1
        return None
    if not (rvi_armed and rvi_ok):
        diag["rvi_fail"] += 1
        return None

    diag["signals_passed"] += 1
    return side


# ---------------- COSTS ----------------

def entry_fill(market, price, side):
    if market == "XAUUSD":
        cost = XAU_SPREAD / 2 + XAU_SLIPPAGE
    else:
        cost = price * (BTC_SPREAD / 2 + BTC_SLIPPAGE)

    return price + cost if side == "BUY" else price - cost


def exit_fill(market, price, side):
    if market == "XAUUSD":
        cost = XAU_SPREAD / 2 + XAU_SLIPPAGE
    else:
        cost = price * (BTC_SPREAD / 2 + BTC_SLIPPAGE)

    return price - cost if side == "BUY" else price + cost


# ---------------- BACKTEST ----------------

def run_test(market, tf, raw, h1, h4):
    trades = []
    diag = {
        "unknown_bias": 0,
        "insufficient_history": 0,
        "ema_retest_fail": 0,
        "rsi_fail": 0,
        "rvi_fail": 0,
        "signals_passed": 0,
        "liquidity_fail": 0,
        "rr_fail": 0
    }

    if raw.empty or len(raw) < 250:
        return trades, diag

    d = indicators(raw)
    h1 = indicators(h1) if not h1.empty else h1
    h4 = indicators(h4) if not h4.empty else h4

    position = None

    for i in range(220, len(d) - 1):
        bar = d.iloc[i]
        ts = d.index[i]
        candle_duration = pd.Timedelta(tf)
        signal_close = ts + candle_duration

        # Manage an existing position first.
        if position is not None:
            side = position["side"]
            exit_price = None
            reason = None

            if side == "BUY":
                if bar.Open <= position["sl"]:
                    exit_price, reason = bar.Open, "SL_GAP"
                elif bar.Open >= position["tp"]:
                    exit_price, reason = bar.Open, "TP_GAP"
                elif bar.Low <= position["sl"]:
                    exit_price, reason = position["sl"], "SL"
                elif bar.High >= position["tp"]:
                    exit_price, reason = position["tp"], "TP"
            else:
                if bar.Open >= position["sl"]:
                    exit_price, reason = bar.Open, "SL_GAP"
                elif bar.Open <= position["tp"]:
                    exit_price, reason = bar.Open, "TP_GAP"
                elif bar.High >= position["sl"]:
                    exit_price, reason = position["sl"], "SL"
                elif bar.Low <= position["tp"]:
                    exit_price, reason = position["tp"], "TP"

            if exit_price is not None:
                fill = exit_fill(market, float(exit_price), side)
                pnl = (
                    fill - position["entry"]
                    if side == "BUY"
                    else position["entry"] - fill
                )
                net_r = pnl / position["risk"]

                if market == "BTCUSD":
                    fee = (
                        position["entry"] + fill
                    ) * BTC_FEE / position["risk"]
                    net_r -= fee

                trades.append({
                    "market": market,
                    "timeframe": tf,
                    "side": side,
                    "entry_time": position["time"],
                    "exit_time": ts,
                    "entry": position["entry"],
                    "sl": position["sl"],
                    "tp": position["tp"],
                    "planned_RR": MIN_RR,
                    "actual_net_R": round(net_r, 4),
                    "exit_reason": reason
                })
                position = None
            continue

        hist = d.iloc[:i+1]
        bias = bias_at(h1, h4, signal_close)
        side = find_signal(hist, bias, diag)

        if side is None:
            continue

        levels = liquidity(hist)
        raw_entry = float(d.iloc[i+1].Open)

        if side == "BUY":
            opposing = [
                x for kind, x in levels
                if kind == "HIGH" and x > raw_entry
            ]
            stops = [
                x for kind, x in levels
                if kind == "LOW" and x < raw_entry
            ]
            if not opposing or not stops:
                diag["liquidity_fail"] += 1
                continue
            sl = max(stops)
        else:
            opposing = [
                x for kind, x in levels
                if kind == "LOW" and x < raw_entry
            ]
            stops = [
                x for kind, x in levels
                if kind == "HIGH" and x > raw_entry
            ]
            if not opposing or not stops:
                diag["liquidity_fail"] += 1
                continue
            sl = min(stops)

        entry = entry_fill(market, raw_entry, side)
        risk = abs(entry - sl)

        if risk <= 0:
            diag["liquidity_fail"] += 1
            continue

        # Exact 1:4 target. Liquidity must allow at least 4R.
        tp = entry + MIN_RR * risk if side == "BUY" else entry - MIN_RR * risk

        nearest_liquidity = (
            min(opposing) if side == "BUY" else max(opposing)
        )

        if side == "BUY" and nearest_liquidity < tp:
            diag["rr_fail"] += 1
            continue
        if side == "SELL" and nearest_liquidity > tp:
            diag["rr_fail"] += 1
            continue

        position = {
            "side": side,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "risk": risk,
            "time": d.index[i+1]
        }

    return trades, diag


# ---------------- REPORT ----------------

def stats(trades):
    if not trades:
        return {
            "trades": 0, "wins": 0, "losses": 0,
            "win_rate_pct": None, "net_R": 0,
            "profit_factor": None, "max_drawdown_R": 0,
            "average_R": None
        }

    r = np.array([x["actual_net_R"] for x in trades])
    wins = r[r > 0]
    losses = r[r < 0]
    equity = np.cumsum(r)
    peak = np.maximum.accumulate(np.maximum(equity, 0))
    dd = peak - equity

    return {
        "trades": len(r),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": round(len(wins) / len(r) * 100, 2),
        "net_R": round(r.sum(), 3),
        "profit_factor": round(
            wins.sum() / abs(losses.sum()), 3
        ) if len(losses) else None,
        "max_drawdown_R": round(dd.max(), 3),
        "average_R": round(r.mean(), 3)
    }


def main():
    all_trades = []
    summaries = []
    coverage = []
    diagnostics = []
    errors = []

    for market in MARKETS:
        print("\nLoading", market)

        try:
            data = load_market(market)
        except Exception as e:
            errors.append({"market": market, "error": str(e)})
            print("MARKET DOWNLOAD ERROR:", e)
            continue

        h1 = data.get("1h", pd.DataFrame())
        h4 = data.get("4h", pd.DataFrame())

        for tf in TIMEFRAMES:
            d = data.get(tf, pd.DataFrame())

            days = (
                (d.index[-1] - d.index[0]).total_seconds() / 86400
                if len(d) > 1 else 0
            )

            coverage.append({
                "market": market,
                "timeframe": tf,
                "candles": len(d),
                "actual_days": round(days, 2)
            })

            if d.empty:
                errors.append({
                    "market": market,
                    "error": f"No data for {tf}"
                })
                continue

            try:
                trades, diag = run_test(market, tf, d, h1, h4)
                all_trades.extend(trades)

                result = stats(trades)
                result.update({"market": market, "timeframe": tf})
                summaries.append(result)

                diag.update({"market": market, "timeframe": tf})
                diagnostics.append(diag)

                print(
                    market, tf,
                    "trades:", result["trades"],
                    "win rate:", result["win_rate_pct"],
                    "net R:", result["net_R"]
                )

            except Exception as e:
                errors.append({
                    "market": market,
                    "timeframe": tf,
                    "error": str(e)
                })
                print("TEST ERROR:", market, tf, e)

    pd.DataFrame(all_trades).to_csv(OUT / "trades.csv", index=False)
    pd.DataFrame(summaries).to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(coverage).to_csv(OUT / "data_coverage.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(OUT / "diagnostics.csv", index=False)
    pd.DataFrame(errors).to_csv(OUT / "errors.csv", index=False)

    print("\nBACKTEST FINISHED")
    print("Total closed trades:", len(all_trades))
    print("Reports:", OUT.resolve())


if __name__ == "__main__":
    main()
