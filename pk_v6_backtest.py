
import os
import time
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, timezone

# =====================================================
# PK V6.1 HISTORICAL BACKTEST ENGINE
# 90 days | BTCUSD + XAUUSD proxy | Signal-only
# =====================================================

SYMBOLS = {"BTCUSD": "BTCUSDT", "XAUUSD": "GC=F"}
TIMEFRAMES = ["5m", "15m", "30m", "1h", "4h", "1d"]

DAYS = 90
EMA_FAST, EMA_MID, EMA_SLOW = 20, 50, 200
RSI_PERIOD = 14
RVI_PERIOD = 8
RSI_BUY, RSI_SELL = 10, 90
RVI_LEVEL = 0.500
ATR_PERIOD = 14
RR_MIN = 4.0
SWING_LOOKBACK = 20
PIVOT_WINDOW = 3
ARM_LOOKBACK = 50
LIQUIDITY_BUFFER_ATR = 0.15

# Exness Standard XAUUSD cost assumptions
XAU_SPREAD = 0.30       # Total spread in price units
XAU_SLIPPAGE = 0.05     # Per side, price units
XAU_COMMISSION_PER_LOT = 0.0

# BTC Binance Futures cost assumptions
# Replace with your actual execution venue's costs if different.
BTC_SPREAD = 0.0        # Binance market data does not provide Exness spread
BTC_SLIPPAGE = 0.0      # Conservative testing should use a non-zero value
BTC_FEE_RATE = 0.0005   # Assumed 0.05% per side; configurable

BINANCE_URL = "https://fapi.binance.com"
OUTPUT_DIR = "backtest_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================= DATA =================

def fetch_binance_90d(symbol, interval):
    url = f"{BINANCE_URL}/fapi/v1/klines"
    start = int((datetime.now(timezone.utc) -
                 timedelta(days=DAYS + 15)).timestamp() * 1000)

    rows = []

    while True:
        response = requests.get(
            url,
            params={
                "symbol": symbol,
                "interval": interval,
                "startTime": start,
                "limit": 1000
            },
            timeout=30
        )
        response.raise_for_status()
        batch = response.json()

        if not batch:
            break

        rows.extend(batch)
        last_time = batch[-1][0]
        start = last_time + 1

        if len(batch) < 1000:
            break

        time.sleep(0.15)

    columns = [
        "time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ]

    df = pd.DataFrame(rows, columns=columns)

    if df.empty:
        raise ValueError("Binance returned no candles")

    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df[["time", "open", "high", "low", "close", "volume"]]
        .dropna()
        .drop_duplicates("time")
        .set_index("time")
        .sort_index()
    )

    # Remove unfinished candle
    if len(df) > 1:
        df = df.iloc[:-1]

    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=DAYS)
    return df[df.index >= cutoff]


def fetch_yahoo_90d(interval):
    # Yahoo intraday history is limited; never claim missing data is 90 days.
    source_interval = "1h" if interval == "4h" else interval
    period = "2y" if interval == "1d" else "60d"

    data = yf.download(
        "GC=F",
        period=period,
        interval=source_interval,
        progress=False,
        auto_adjust=False
    )

    if data is None or data.empty:
        raise ValueError("Yahoo returned no data")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    })

    data = data[["open", "high", "low", "close", "volume"]].dropna()
    data.index = pd.to_datetime(data.index, utc=True)
    data = data.sort_index()

    if interval == "4h":
        data = data.resample("4h").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }).dropna()

    if len(data) > 1:
        data = data.iloc[:-1]

    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=DAYS)
    return data[data.index >= cutoff]


def get_data(symbol, timeframe):
    if symbol == "BTCUSD":
        return fetch_binance_90d("BTCUSDT", timeframe)
    return fetch_yahoo_90d(timeframe)


# ================= INDICATORS =================

def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period, min_periods=period, adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period, min_periods=period, adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)

    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss > 0), 0)
    return rsi


def calculate_rvi(df, period=8):
    c, o, h, l = (
        df["close"], df["open"], df["high"], df["low"]
    )

    numerator = (
        (c-o) + 2*(c.shift(1)-o.shift(1))
        + 2*(c.shift(2)-o.shift(2))
        + (c.shift(3)-o.shift(3))
    ) / 6

    denominator = (
        (h-l) + 2*(h.shift(1)-l.shift(1))
        + 2*(h.shift(2)-l.shift(2))
        + (h.shift(3)-l.shift(3))
    ) / 6

    return (
        numerator.rolling(period).mean()
        / denominator.rolling(period).mean().replace(0, np.nan)
    )


def add_indicators(df):
    df = df.copy()

    for span, name in [
        (EMA_FAST, "ema20"),
        (EMA_MID, "ema50"),
        (EMA_SLOW, "ema200")
    ]:
        df[name] = df["close"].ewm(
            span=span, adjust=False
        ).mean()

    df["rsi"] = calculate_rsi(df["close"], RSI_PERIOD)
    df["rvi"] = calculate_rvi(df, RVI_PERIOD)

    pc = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - pc).abs(),
        (df["low"] - pc).abs()
    ], axis=1).max(axis=1)

    df["atr"] = tr.ewm(
        alpha=1/ATR_PERIOD,
        min_periods=ATR_PERIOD,
        adjust=False
    ).mean()

    return df.dropna()


# ================= HTF BIAS =================

def get_htf_bias(h1, h4, timestamp):
    a = h1.loc[:timestamp]
    b = h4.loc[:timestamp]

    if len(a) < 2 or len(b) < 2:
        return "UNKNOWN"

    x, y = a.iloc[-1], b.iloc[-1]

    if x.ema20 > x.ema50 > x.ema200 and \
       y.ema20 > y.ema50 > y.ema200:
        return "BULLISH"

    if x.ema20 < x.ema50 < x.ema200 and \
       y.ema20 < y.ema50 < y.ema200:
        return "BEARISH"

    return "NEUTRAL"


# ================= LIQUIDITY =================

def liquidity_levels(history):
    recent = history.iloc[-200:]

    highs, lows = [], []

    for i in range(PIVOT_WINDOW, len(recent)-PIVOT_WINDOW):
        h = recent.high.iloc[i]
        l = recent.low.iloc[i]

        if h == recent.high.iloc[
            i-PIVOT_WINDOW:i+PIVOT_WINDOW+1
        ].max():
            highs.append(float(h))

        if l == recent.low.iloc[
            i-PIVOT_WINDOW:i+PIVOT_WINDOW+1
        ].min():
            lows.append(float(l))

    atr = float(recent.atr.iloc[-1])
    price = float(recent.close.iloc[-1])
    tolerance = max(atr*0.15, price*0.0005)

    def equal_level(values):
        for i in range(len(values)):
            for j in range(i+1, len(values)):
                if abs(values[i]-values[j]) <= tolerance:
                    return (values[i]+values[j])/2
        return None

    # Daily/weekly levels are computed from available daily candles.
    daily = history.resample("1D").agg({
        "high": "max", "low": "min"
    }).dropna()

    weekly = history.resample("1W").agg({
        "high": "max", "low": "min"
    }).dropna()

    return {
        "EQH": equal_level(highs),
        "EQL": equal_level(lows),
        "Swing High": max(highs) if highs else None,
        "Swing Low": min(lows) if lows else None,
        "PDH": float(daily.high.iloc[-2]) if len(daily) >= 2 else None,
        "PDL": float(daily.low.iloc[-2]) if len(daily) >= 2 else None,
        "PWH": float(weekly.high.iloc[-2]) if len(weekly) >= 2 else None,
        "PWL": float(weekly.low.iloc[-2]) if len(weekly) >= 2 else None
    }


def opposing_target(entry, sl, direction, levels, atr):
    risk = abs(entry - sl)
    if risk <= 0:
        return None, "Invalid risk"

    target = entry + RR_MIN*risk if direction == "BUY" \
        else entry - RR_MIN*risk

    if direction == "BUY":
        candidates = [
            v for k, v in levels.items()
            if k in ["EQH", "Swing High", "PDH", "PWH"]
            and v is not None and v > entry
        ]

        if not candidates:
            return None, "No opposing liquidity above"

        nearest = min(candidates)

        if nearest - atr*LIQUIDITY_BUFFER_ATR < target:
            return None, "Opposing liquidity blocks 1:4"

    else:
        candidates = [
            v for k, v in levels.items()
            if k in ["EQL", "Swing Low", "PDL", "PWL"]
            and v is not None and v < entry
        ]

        if not candidates:
            return None, "No opposing liquidity below"

        nearest = max(candidates)

        if nearest + atr*LIQUIDITY_BUFFER_ATR > target:
            return None, "Opposing liquidity blocks 1:4"

    return target, "Passed"


# ================= ENTRY CONDITIONS =================

def armed(series, threshold, mode):
    prior = series.iloc[-(ARM_LOOKBACK+2):-2]

    if mode == "below":
        return bool((prior < threshold).any())

    return bool((prior > threshold).any())


def entry_setups(df, direction):
    if len(df) < ARM_LOOKBACK + 3:
        return []

    p, c = df.iloc[-2], df.iloc[-1]
    rsi, rvi = df.rsi, df.rvi
    setups = []

    if direction == "BUY":
        aligned = c.ema20 > c.ema50 > c.ema200

        if aligned and p.low <= p.ema20 \
           and p.open > p.ema20 and p.close > p.ema20 \
           and c.open > c.ema20 and c.close > c.ema20:
            setups.append("EMA20 RETEST")

        if armed(rsi, RSI_BUY, "below") \
           and rsi.iloc[-2] <= RSI_BUY \
           and rsi.iloc[-1] > RSI_BUY:
            setups.append("RSI RE-ENTRY")

        if armed(rvi, RVI_LEVEL, "above") \
           and rvi.iloc[-2] >= RVI_LEVEL \
           and rvi.iloc[-1] < RVI_LEVEL:
            setups.append("RVI RE-CROSS")

    else:
        aligned = c.ema20 < c.ema50 < c.ema200

        if aligned and p.high >= p.ema20 \
           and p.open < p.ema20 and p.close < p.ema20 \
           and c.open < c.ema20 and c.close < c.ema20:
            setups.append("EMA20 RETEST")

        if armed(rsi, RSI_SELL, "above") \
           and rsi.iloc[-2] >= RSI_SELL \
           and rsi.iloc[-1] < RSI_SELL:
            setups.append("RSI RE-ENTRY")

        if armed(rvi, -RVI_LEVEL, "below") \
           and rvi.iloc[-2] <= -RVI_LEVEL \
           and rvi.iloc[-1] > -RVI_LEVEL:
            setups.append("RVI RE-CROSS")

    return setups


# ================= COST MODEL =================

def execution_cost(symbol, price, direction, side):
    """
    Cost in price units. Half spread is charged on each side.
    BTC additionally uses assumed percentage fee per side.
    """
    if symbol == "XAUUSD":
        spread = XAU_SPREAD / 2
        slippage = XAU_SLIPPAGE
        commission = 0.0
    else:
        spread = BTC_SPREAD / 2
        slippage = BTC_SLIPPAGE
        commission = price * BTC_FEE_RATE

    return spread + slippage + commission


def fill_price(symbol, raw_price, direction, side):
    cost = execution_cost(symbol, raw_price, direction, side)

    if side == "entry":
        return raw_price + cost if direction == "BUY" \
            else raw_price - cost

    return raw_price - cost if direction == "BUY" \
        else raw_price + cost


# ================= BACKTEST SIMULATION =================

def simulate(symbol, timeframe, df, h1, h4):
    trades = []
    rejected = 0
    open_trade = None

    # Need prior history for indicators and pivots.
    for i in range(ARM_LOOKBACK + 5, len(df)-1):
        timestamp = df.index[i]
        row = df.iloc[i]

        if open_trade is not None:
            next_bar = df.iloc[i]

            direction = open_trade["direction"]
            hit_sl = (
                next_bar.low <= open_trade["sl"]
                if direction == "BUY"
                else next_bar.high >= open_trade["sl"]
            )

            hit_tp = (
                next_bar.high >= open_trade["tp"]
                if direction == "BUY"
                else next_bar.low <= open_trade["tp"]
            )

            exit_reason = None
            raw_exit = None

            # Conservative rule: if both hit in one candle, assume SL first.
            if hit_sl:
                exit_reason = "SL"
                raw_exit = open_trade["sl"]
            elif hit_tp:
                exit_reason = "TP"
                raw_exit = open_trade["tp"]

            if exit_reason:
                exit_price = fill_price(
                    symbol, raw_exit, direction, "exit"
                )

                pnl = (
                    exit_price - open_trade["entry"]
                    if direction == "BUY"
                    else open_trade["entry"] - exit_price
                )

                net_r = pnl / open_trade["risk"]

                trades.append({
                    **open_trade,
                    "exit_time": timestamp,
                    "exit_price": exit_price,
                    "result": exit_reason,
                    "gross_R": (
                        (raw_exit - open_trade["raw_entry"])
                        / open_trade["risk"]
                        if direction == "BUY"
                        else
                        (open_trade["raw_entry"] - raw_exit)
                        / open_trade["risk"]
                    ),
                    "net_R": net_r
                })

                open_trade = None

            continue

        bias = get_htf_bias(h1, h4, timestamp)

        if bias not in ["BULLISH", "BEARISH"]:
            continue

        direction = "BUY" if bias == "BULLISH" else "SELL"
        setups = entry_setups(df.iloc[:i+1], direction)

        if not setups:
            continue

        # Use only data available up to the signal candle.
        history = df.iloc[:i+1]
        levels = liquidity_levels(history)

        atr = float(row.atr)
        raw_entry = float(df.iloc[i+1].open)

        if direction == "BUY":
            sl = min(
                float(history.low.iloc[-SWING_LOOKBACK:].min()),
                raw_entry - 1.5*atr
            )
        else:
            sl = max(
                float(history.high.iloc[-SWING_LOOKBACK:].max()),
                raw_entry + 1.5*atr
            )

        target, reason = opposing_target(
            raw_entry, sl, direction, levels, atr
        )

        if target is None:
            rejected += 1
            continue

        entry = fill_price(
            symbol, raw_entry, direction, "entry"
        )

        risk = abs(entry-sl)

        if risk <= 0:
            rejected += 1
            continue

        open_trade = {
            "symbol": symbol,
            "timeframe": timeframe,
            "direction": direction,
            "setups": ",".join(setups),
            "signal_time": timestamp,
            "entry_time": df.index[i+1],
            "raw_entry": raw_entry,
            "entry": entry,
            "sl": sl,
            "tp": target,
            "risk": risk,
            "planned_RR": abs(target-entry)/risk
        }

    # Mark any unfinished position at final available close.
    if open_trade is not None:
        raw_exit = float(df.close.iloc[-1])
        exit_price = fill_price(
            symbol, raw_exit, open_trade["direction"], "exit"
        )

        direction = open_trade["direction"]
        pnl = (
            exit_price-open_trade["entry"]
            if direction == "BUY"
            else open_trade["entry"]-exit_price
        )

        trades.append({
            **open_trade,
            "exit_time": df.index[-1],
            "exit_price": exit_price,
            "result": "OPEN_AT_END",
            "gross_R": np.nan,
            "net_R": pnl/open_trade["risk"]
        })

    return trades, rejected


# ================= REPORTING =================

def summarize(trades, rejected, symbol, timeframe):
    closed = [
        t for t in trades
        if t["result"] in ["TP", "SL"]
    ]

    if not closed:
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate_pct": 0,
            "net_R": 0,
            "profit_factor": 0,
            "max_drawdown_R": 0,
            "rejected_by_liquidity": rejected
        }

    results = np.array([t["net_R"] for t in closed])
    wins = int((results > 0).sum())
    losses = int((results <= 0).sum())

    gross_profit = results[results > 0].sum()
    gross_loss = abs(results[results < 0].sum())

    pf = (
        gross_profit/gross_loss
        if gross_loss > 0 else np.inf
    )

    equity = np.cumsum(results)
    peaks = np.maximum.accumulate(
        np.maximum(equity, 0)
    )
    drawdown = peaks-equity

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "trades": len(closed),
        "wins": wins,
        "losses": losses,
        "win_rate_pct": round(100*wins/len(closed), 2),
        "net_R": round(float(results.sum()), 3),
        "average_R": round(float(results.mean()), 3),
        "profit_factor": round(float(pf), 3)
            if np.isfinite(pf) else "Infinity",
        "max_drawdown_R": round(float(drawdown.max()), 3),
        "rejected_by_liquidity": rejected
    }


def main():
    print("PK V6.1 BACKTEST STARTED")
    print("Period:", DAYS, "days")

    all_trades = []
    summaries = []
    coverage = []

    for symbol in SYMBOLS:
        print("\nMARKET:", symbol)

        # Load HTF data once for historical bias.
        try:
            h1 = add_indicators(get_data(symbol, "1h"))
            h4 = add_indicators(get_data(symbol, "4h"))
        except Exception as exc:
            print("HTF data error:", symbol, exc)
            continue

        for timeframe in TIMEFRAMES:
            print("Testing:", symbol, timeframe)

            try:
                raw = get_data(symbol, timeframe)

                actual_days = (
                    (raw.index[-1]-raw.index[0]).total_seconds()/86400
                    if len(raw) > 1 else 0
                )

                coverage.append({
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "candles": len(raw),
                    "actual_days": round(actual_days, 2),
                    "requested_days": DAYS,
                    "full_90d": actual_days >= 89
                })

                if len(raw) < 250:
                    print("Insufficient candles:", len(raw))
                    continue

                df = add_indicators(raw)

                trades, rejected = simulate(
                    symbol, timeframe, df, h1, h4
                )

                all_trades.extend(trades)

                summaries.append(
                    summarize(trades, rejected, symbol, timeframe)
                )

                print(
                    "Completed:", symbol, timeframe,
                    "| Trades:", len(trades),
                    "| Liquidity rejected:", rejected
                )

            except Exception as exc:
                print("Backtest error:", symbol, timeframe, exc)

    trade_file = f"{OUTPUT_DIR}/all_trades.csv"
    summary_file = f"{OUTPUT_DIR}/summary.csv"
    coverage_file = f"{OUTPUT_DIR}/data_coverage.csv"

    pd.DataFrame(all_trades).to_csv(trade_file, index=False)
    pd.DataFrame(summaries).to_csv(summary_file, index=False)
    pd.DataFrame(coverage).to_csv(coverage_file, index=False)

    print("\n========== BACKTEST COMPLETE ==========")
    print("Total trades:", len(all_trades))
    print("Summary:", summary_file)
    print("Trade log:", trade_file)
    print("Data coverage:", coverage_file)
    print("Cost assumptions:")
    print("XAU spread:", XAU_SPREAD)
    print("XAU slippage per side:", XAU_SLIPPAGE)
    print("BTC fee per side:", BTC_FEE_RATE)
    print("========================================")


if __name__ == "__main__":
    main()
