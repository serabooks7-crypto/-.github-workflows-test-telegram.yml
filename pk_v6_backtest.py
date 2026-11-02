
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore")

# =========================================================
# PK V6.1 - 90 DAY BACKTEST
# Separate research file. Does NOT modify the live bot.
# =========================================================

DAYS = 90
MARKETS = ["XAUUSD", "BTCUSD"]
TIMEFRAMES = ["5m", "15m", "30m", "1h", "4h", "1d"]

# XAUUSD: User-provided Exness Standard assumptions
XAU_SPREAD = 0.30          # Total spread in price units
XAU_COMMISSION_PER_LOT = 0.0
XAU_SLIPPAGE_PER_SIDE = 0.05

# BTC: Assumed Binance Futures execution costs.
# Change these if your actual exchange fees differ.
BTC_FEE_RATE_PER_SIDE = 0.0005
BTC_SPREAD_RATE_TOTAL = 0.0
BTC_SLIPPAGE_RATE_PER_SIDE = 0.0001

MIN_RR = 4.0
PIVOT_WINDOW = 3
ARM_LOOKBACK = 50
INITIAL_EQUITY_R = 0.0

OUTPUT_DIR = Path("backtest_results")
OUTPUT_DIR.mkdir(exist_ok=True)


# -------------------------
# DATA DOWNLOAD
# -------------------------

def binance_klines(interval, days=90):
    url = "https://fapi.binance.com/fapi/v1/klines"
    end_time = int(time.time() * 1000)
    start_time = end_time - days * 24 * 60 * 60 * 1000

    all_rows = []
    cursor = start_time

    while cursor < end_time:
        params = {
            "symbol": "BTCUSDT",
            "interval": interval,
            "startTime": cursor,
            "endTime": end_time,
            "limit": 1000,
        }

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        rows = response.json()

        if not rows:
            break

        all_rows.extend(rows)
        cursor = rows[-1][0] + 1

        if len(rows) < 1000:
            break

        time.sleep(0.15)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows, columns=[
        "time", "Open", "High", "Low", "Close", "Volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])

    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(
        df["close_time"], unit="ms", utc=True
    )

    df = df.set_index("time")
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df = df.astype(float)

    # Exclude candles that have not closed yet.
    now = pd.Timestamp.now(tz="UTC")
    df = df[df.index + pd.Timedelta(milliseconds=1) <= now]

    return df[~df.index.duplicated(keep="last")].sort_index()


def yahoo_gold(interval):
    # Yahoo Finance generally limits intraday history.
    period = "60d" if interval != "1d" else "2y"

    data = yf.download(
        "GC=F",
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.rename(columns={
        "Open": "Open",
        "High": "High",
        "Low": "Low",
        "Close": "Close",
        "Volume": "Volume",
    })

    required = ["Open", "High", "Low", "Close", "Volume"]
    data = data[required].dropna(subset=["Open", "High", "Low", "Close"])

    idx = pd.to_datetime(data.index)

    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")

    data.index = idx
    data = data[~data.index.duplicated(keep="last")].sort_index()

    # Remove the currently forming candle.
    duration = pd.Timedelta(interval)
    now = pd.Timestamp.now(tz="UTC")
    data = data[data.index + duration <= now]

    return data.astype(float)


def resample_ohlcv(df, rule):
    if df.empty:
        return df.copy()

    result = df.resample(rule, label="left", closed="left").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    })

    return result.dropna(subset=["Open", "High", "Low", "Close"])


def load_market_data(market):
    result = {}

    if market == "BTCUSD":
        mapping = {
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1h",
            "4h": "4h",
            "1d": "1d",
        }

        for tf, interval in mapping.items():
            print(f"Downloading BTC {tf}...")
            result[tf] = binance_klines(interval, DAYS)

    else:
        mapping = {
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "60m",
            "4h": "60m",
            "1d": "1d",
        }

        for tf, interval in mapping.items():
            print(f"Downloading Gold {tf}...")
            data = yahoo_gold(interval)

            if tf == "4h":
                data = resample_ohlcv(data, "4h")

            result[tf] = data

    return result


# -------------------------
# INDICATORS
# -------------------------

def add_indicators(df):
    df = df.copy()

    for period in [20, 50, 200]:
        df[f"EMA{period}"] = df["Close"].ewm(
            span=period, adjust=False
        ).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14, adjust=False, min_periods=14
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14, adjust=False, min_periods=14
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # Approximate RVI(8), normalized around zero.
    num = (df["Close"] - df["Open"]).rolling(8).mean()
    den = (df["High"] - df["Low"]).rolling(8).mean()
    df["RVI"] = num / den.replace(0, np.nan)

    return df


# -------------------------
# HTF BIAS - NO LOOKAHEAD
# -------------------------

def closed_htf_row(htf, timestamp, timeframe):
    if htf.empty:
        return None

    duration = pd.Timedelta(timeframe)

    # Candle is usable only after its closing time.
    available = htf.loc[
        htf.index + duration <= timestamp
    ]

    if available.empty:
        return None

    return available.iloc[-1]


def get_htf_bias(h1, h4, timestamp):
    row1 = closed_htf_row(h1, timestamp, "1h")
    row4 = closed_htf_row(h4, timestamp, "4h")

    if row1 is None or row4 is None:
        return "UNKNOWN"

    bullish1 = (
        row1["EMA20"] > row1["EMA50"] > row1["EMA200"]
    )
    bullish4 = (
        row4["EMA20"] > row4["EMA50"] > row4["EMA200"]
    )

    bearish1 = (
        row1["EMA20"] < row1["EMA50"] < row1["EMA200"]
    )
    bearish4 = (
        row4["EMA20"] < row4["EMA50"] < row4["EMA200"]
    )

    if bullish1 and bullish4:
        return "BULLISH"

    if bearish1 and bearish4:
        return "BEARISH"

    return "UNKNOWN"


# -------------------------
# LIQUIDITY ESTIMATION
# -------------------------

def liquidity_levels(history):
    levels = []

    if len(history) < 30:
        return levels

    # Confirmed swing highs and lows.
    highs = history["High"].to_numpy()
    lows = history["Low"].to_numpy()

    for i in range(PIVOT_WINDOW, len(history) - PIVOT_WINDOW):
        if highs[i] == max(
            highs[i - PIVOT_WINDOW:i + PIVOT_WINDOW + 1]
        ):
            levels.append(("HIGH", float(highs[i])))

        if lows[i] == min(
            lows[i - PIVOT_WINDOW:i + PIVOT_WINDOW + 1]
        ):
            levels.append(("LOW", float(lows[i])))

    # Previous completed daily and weekly levels.
    daily = resample_ohlcv(history, "1D")
    weekly = resample_ohlcv(history, "1W")

    if len(daily) >= 2:
        levels.append(("HIGH", float(daily["High"].iloc[-2])))
        levels.append(("LOW", float(daily["Low"].iloc[-2])))

    if len(weekly) >= 2:
        levels.append(("HIGH", float(weekly["High"].iloc[-2])))
        levels.append(("LOW", float(weekly["Low"].iloc[-2])))

    return levels


def nearest_target(levels, direction, entry):
    candidates = []

    for kind, price in levels:
        if direction == "BUY" and kind == "HIGH" and price > entry:
            candidates.append(price)

        if direction == "SELL" and kind == "LOW" and price < entry:
            candidates.append(price)

    if not candidates:
        return None

    return min(candidates) if direction == "BUY" else max(candidates)


# -------------------------
# SIGNAL MODEL
# -------------------------

def signal_at(history, bias):
    if len(history) < 220:
        return None

    prev = history.iloc[-2]
    cur = history.iloc[-1]

    if bias == "UNKNOWN":
        return None

    direction = "BUY" if bias == "BULLISH" else "SELL"

    # EMA20 retest. Candle body must not close across EMA20.
    if direction == "BUY":
        ema_retest = (
            cur["Low"] <= cur["EMA20"]
            and cur["Close"] > cur["EMA20"]
        )
        ema_alignment = (
            cur["EMA20"] > cur["EMA50"] > cur["EMA200"]
        )
    else:
        ema_retest = (
            cur["High"] >= cur["EMA20"]
            and cur["Close"] < cur["EMA20"]
        )
        ema_alignment = (
            cur["EMA20"] < cur["EMA50"] < cur["EMA200"]
        )

    if not ema_retest or not ema_alignment:
        return None

    rsi_history = history["RSI"].iloc[-(ARM_LOOKBACK + 1):-1]
    rvi_history = history["RVI"].iloc[-(ARM_LOOKBACK + 1):-1]

    if direction == "BUY":
        rsi_armed = (rsi_history < 10).any()
        rsi_trigger = prev["RSI"] <= 10 and cur["RSI"] > 10

        rvi_armed = (rvi_history > 0.5).any()
        rvi_trigger = prev["RVI"] >= 0.5 and cur["RVI"] < 0.5

    else:
        rsi_armed = (rsi_history > 90).any()
        rsi_trigger = prev["RSI"] >= 90 and cur["RSI"] < 90

        rvi_armed = (rvi_history < -0.5).any()
        rvi_trigger = prev["RVI"] <= -0.5 and cur["RVI"] > -0.5

    if not (rsi_armed and rsi_trigger and rvi_armed and rvi_trigger):
        return None

    return direction


# -------------------------
# EXECUTION COSTS
# -------------------------

def fill_price(market, price, side, is_entry):
    if market == "XAUUSD":
        half_spread = XAU_SPREAD / 2
        slip = XAU_SLIPPAGE_PER_SIDE

        if side == "BUY":
            adjustment = half_spread + slip
            return price + adjustment if is_entry else price - adjustment

        adjustment = half_spread + slip
        return price - adjustment if is_entry else price + adjustment

    spread = price * BTC_SPREAD_RATE_TOTAL / 2
    slip = price * BTC_SLIPPAGE_RATE_PER_SIDE

    if side == "BUY":
        adjustment = spread + slip
        return price + adjustment if is_entry else price - adjustment

    adjustment = spread + slip
    return price - adjustment if is_entry else price + adjustment


# -------------------------
# BACKTEST ENGINE
# -------------------------

def simulate(market, timeframe, df, h1, h4):
    trades = []

    if df.empty or len(df) < 250:
        return trades

    df = add_indicators(df)
    h1 = add_indicators(h1) if not h1.empty else h1
    h4 = add_indicators(h4) if not h4.empty else h4

    position = None

    for i in range(220, len(df) - 1):
        timestamp = df.index[i]
        candle = df.iloc[i]

        if position is not None:
            # Evaluate the candle after entry.
            bar = candle
            exit_price = None
            reason = None

            if position["side"] == "BUY":
                # Gap-aware stop/target execution.
                if bar["Open"] <= position["sl"]:
                    exit_price = bar["Open"]
                    reason = "SL_GAP"
                elif bar["Open"] >= position["tp"]:
                    exit_price = position["tp"]
                    reason = "TP_GAP"
                elif bar["Low"] <= position["sl"]:
                    exit_price = position["sl"]
                    reason = "SL"
                elif bar["High"] >= position["tp"]:
                    exit_price = position["tp"]
                    reason = "TP"

            else:
                if bar["Open"] >= position["sl"]:
                    exit_price = bar["Open"]
                    reason = "SL_GAP"
                elif bar["Open"] <= position["tp"]:
                    exit_price = position["tp"]
                    reason = "TP_GAP"
                elif bar["High"] >= position["sl"]:
                    exit_price = position["sl"]
                    reason = "SL"
                elif bar["Low"] <= position["tp"]:
                    exit_price = position["tp"]
                    reason = "TP"

            if exit_price is not None:
                exit_fill = fill_price(
                    market, exit_price, position["side"], False
                )

                if position["side"] == "BUY":
                    pnl = exit_fill - position["entry"]
                else:
                    pnl = position["entry"] - exit_fill

                net_r = pnl / position["risk"]

                if market == "BTCUSD":
                    fee_r = (
                        position["entry"] * BTC_FEE_RATE_PER_SIDE
                        + exit_fill * BTC_FEE_RATE_PER_SIDE
                    ) / position["risk"]
                    net_r -= fee_r

                trades.append({
                    "market": market,
                    "timeframe": timeframe,
                    "side": position["side"],
                    "entry_time": position["time"],
                    "exit_time": timestamp,
                    "entry": position["entry"],
                    "exit": exit_fill,
                    "stop_loss": position["sl"],
                    "take_profit": position["tp"],
                    "exit_reason": reason,
                    "net_R": net_r,
                })

                position = None

            continue

        history = df.iloc[:i + 1]
        bias = get_htf_bias(h1, h4, timestamp)
        direction = signal_at(history, bias)

        if direction is None:
            continue

        levels = liquidity_levels(history)
        raw_entry = float(df.iloc[i + 1]["Open"])

        if direction == "BUY":
            swing_lows = [
                price for kind, price in levels
                if kind == "LOW" and price < raw_entry
            ]
            if not swing_lows:
                continue
            sl = max(swing_lows)
        else:
            swing_highs = [
                price for kind, price in levels
                if kind == "HIGH" and price > raw_entry
            ]
            if not swing_highs:
                continue
            sl = min(swing_highs)

        entry = fill_price(market, raw_entry, direction, True)
        risk = abs(entry - sl)

        if risk <= 0:
            continue

        target = nearest_target(levels, direction, raw_entry)

        if target is None:
            continue

        reward = (
            target - raw_entry
            if direction == "BUY"
            else raw_entry - target
        )

        if reward / risk < MIN_RR:
            continue

        # Target is fixed at the nearest opposing liquidity level.
        tp = target

        position = {
            "side": direction,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "risk": risk,
            "time": df.index[i + 1],
        }

    return trades


# -------------------------
# REPORTING
# -------------------------

def summarize(trades):
    if not trades:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate_pct": 0,
            "net_R": 0,
            "profit_factor_R": 0,
            "max_drawdown_R": 0,
            "average_R": 0,
        }

    results = np.array([t["net_R"] for t in trades], dtype=float)

    wins = results[results > 0]
    losses = results[results < 0]

    equity = np.cumsum(results)
    peaks = np.maximum.accumulate(np.maximum(equity, 0))
    drawdown = peaks - equity

    gross_profit = wins.sum()
    gross_loss = abs(losses.sum())

    return {
        "trades": len(results),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": round(len(wins) / len(results) * 100, 2),
        "net_R": round(results.sum(), 3),
        "profit_factor_R": round(
            gross_profit / gross_loss, 3
        ) if gross_loss else None,
        "max_drawdown_R": round(drawdown.max(), 3),
        "average_R": round(results.mean(), 3),
    }


def main():
    all_trades = []
    summaries = []
    coverage = []

    for market in MARKETS:
        print(f"\n========== {market} ==========")

        try:
            market_data = load_market_data(market)
        except Exception as exc:
            print(f"Download failed for {market}: {exc}")
            continue

        h1 = market_data.get("1h", pd.DataFrame())
        h4 = market_data.get("4h", pd.DataFrame())

        for timeframe in TIMEFRAMES:
            df = market_data.get(timeframe, pd.DataFrame())

            actual_days = (
                (df.index[-1] - df.index[0]).total_seconds() / 86400
                if len(df) > 1 else 0
            )

            coverage.append({
                "market": market,
                "timeframe": timeframe,
                "candles": len(df),
                "actual_days": round(actual_days, 2),
                "first_candle": str(df.index[0]) if len(df) else "",
                "last_candle": str(df.index[-1]) if len(df) else "",
            })

            print(
                f"{market} {timeframe}: "
                f"{len(df)} candles, {actual_days:.1f} days"
            )

            if df.empty:
                continue

            try:
                trades = simulate(
                    market, timeframe, df, h1, h4
                )
            except Exception as exc:
                print(f"Simulation error {market} {timeframe}: {exc}")
                continue

            all_trades.extend(trades)

            stats = summarize(trades)
            stats["market"] = market
            stats["timeframe"] = timeframe
            summaries.append(stats)

            print(
                f"Trades={stats['trades']} | "
                f"Win rate={stats['win_rate_pct']}% | "
                f"Net R={stats['net_R']} | "
                f"PF={stats['profit_factor_R']}"
            )

    pd.DataFrame(all_trades).to_csv(
        OUTPUT_DIR / "trades.csv", index=False
    )

    pd.DataFrame(summaries).to_csv(
        OUTPUT_DIR / "summary.csv", index=False
    )

    pd.DataFrame(coverage).to_csv(
        OUTPUT_DIR / "data_coverage.csv", index=False
    )

    print("\n====================================")
    print("PK V6.1 BACKTEST COMPLETED")
    print("====================================")
    print(f"Total closed trades: {len(all_trades)}")
    print(f"Reports saved in: {OUTPUT_DIR.resolve()}")
    print("Files: trades.csv, summary.csv, data_coverage.csv")


if __name__ == "__main__":
    main()
