
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ==========================================
# RLA MULTI-STRATEGY RESEARCH ENGINE V1
# XAUUSD proxy: GC=F (NOT broker XAUUSD)
# Research only - no live trading signals
# ==========================================

SYMBOL = "GC=F"
INTERVAL = "5m"
PERIOD = "60d"

ATR_PERIOD = 14
EMA_FAST = 20
EMA_SLOW = 50
RSI_PERIOD = 14

MIN_RR = 4.0
MAX_HOLDING_BARS = 60

OUTPUT_DIR = Path("research_output")
OUTPUT_DIR.mkdir(exist_ok=True)


def download_data():
    print("Downloading market data...")

    df = yf.download(
        SYMBOL,
        period=PERIOD,
        interval=INTERVAL,
        auto_adjust=False,
        progress=False
    )

    if df.empty:
        raise RuntimeError("No market data received.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close", "Volume"]
    df = df[required].copy()
    df = df.dropna()
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()

    if len(df) < 500:
        raise RuntimeError("Insufficient historical candles.")

    # Remove the latest potentially incomplete candle.
    df = df.iloc[:-1].copy()

    age = pd.Timestamp.now(tz="UTC") - pd.Timestamp(
        df.index[-1]
    ).tz_convert("UTC")

    if age.total_seconds() > 30 * 60:
        raise RuntimeError(
            "Market data is stale. Research stopped."
        )

    return df


def calculate_indicators(df):
    df = df.copy()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    df["EMA20"] = close.ewm(
        span=EMA_FAST, adjust=False
    ).mean()

    df["EMA50"] = close.ewm(
        span=EMA_SLOW, adjust=False
    ).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / RSI_PERIOD,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / RSI_PERIOD,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    previous_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    df["ATR"] = tr.ewm(
        alpha=1 / ATR_PERIOD,
        adjust=False
    ).mean()

    df["PREV_HIGH"] = high.shift(1).rolling(20).max()
    df["PREV_LOW"] = low.shift(1).rolling(20).min()

    # Liquidity sweep and reclaim.
    df["SWEEP_LOW"] = (
        (low < df["PREV_LOW"]) &
        (close > df["PREV_LOW"])
    )

    df["SWEEP_HIGH"] = (
        (high > df["PREV_HIGH"]) &
        (close < df["PREV_HIGH"])
    )

    return df.dropna()


def build_signals(df):
    df = df.copy()

    # Model 1: Trend + Pullback
    df["TREND_BUY"] = (
        (df["EMA20"] > df["EMA50"]) &
        (df["Close"] > df["EMA20"]) &
        (df["RSI"].between(50, 68))
    )

    df["TREND_SELL"] = (
        (df["EMA20"] < df["EMA50"]) &
        (df["Close"] < df["EMA20"]) &
        (df["RSI"].between(32, 50))
    )

    # Model 2: Liquidity Sweep + Reclaim
    df["SWEEP_BUY"] = (
        df["SWEEP_LOW"] &
        (df["RSI"] > 45)
    )

    df["SWEEP_SELL"] = (
        df["SWEEP_HIGH"] &
        (df["RSI"] < 55)
    )

    # Model 3: Range Breakout
    df["BREAKOUT_BUY"] = (
        (df["Close"] > df["PREV_HIGH"]) &
        (df["RSI"] > 60)
    )

    df["BREAKOUT_SELL"] = (
        (df["Close"] < df["PREV_LOW"]) &
        (df["RSI"] < 40)
    )

    return df


def backtest_model(df, model, direction):
    results = []

    buy_col = model + "_BUY"
    sell_col = model + "_SELL"

    for i in range(100, len(df) - MAX_HOLDING_BARS):
        row = df.iloc[i]

        if direction == "BUY" and not row[buy_col]:
            continue

        if direction == "SELL" and not row[sell_col]:
            continue

        entry = float(row["Close"])
        atr = float(row["ATR"])

        if not np.isfinite(atr) or atr <= 0:
            continue

        # Fixed risk distance for initial research.
        risk = 1.5 * atr

        if direction == "BUY":
            sl = entry - risk
            tp = entry + (MIN_RR * risk)
        else:
            sl = entry + risk
            tp = entry - (MIN_RR * risk)

        outcome = "TIMEOUT"
        exit_price = float(df.iloc[i + MAX_HOLDING_BARS]["Close"])
        exit_index = i + MAX_HOLDING_BARS

        for j in range(i + 1, i + MAX_HOLDING_BARS + 1):
            candle = df.iloc[j]

            if direction == "BUY":
                hit_sl = candle["Low"] <= sl
                hit_tp = candle["High"] >= tp
            else:
                hit_sl = candle["High"] >= sl
                hit_tp = candle["Low"] <= tp

            # Conservative rule: if both hit in one candle,
            # count SL first.
            if hit_sl:
                outcome = "LOSS"
                exit_price = sl
                exit_index = j
                break

            if hit_tp:
                outcome = "WIN"
                exit_price = tp
                exit_index = j
                break

        results.append({
            "time": df.index[i],
            "model": model,
            "direction": direction,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "rr": MIN_RR,
            "outcome": outcome,
            "exit_price": exit_price,
            "exit_time": df.index[exit_index]
        })

        # Avoid overlapping positions in this first test.
        if exit_index > i + 1:
            pass

    return results


def main():
    print("RLA Research Engine V1 started.")

    data = download_data()
    data = calculate_indicators(data)
    data = build_signals(data)

    models = [
        "TREND",
        "SWEEP",
        "BREAKOUT"
    ]

    all_results = []

    for model in models:
        for direction in ["BUY", "SELL"]:
            trades = backtest_model(
                data, model, direction
            )
            all_results.extend(trades)

    if not all_results:
        print("No historical setups found.")
        return

    results = pd.DataFrame(all_results)

    # Chronological out-of-sample split.
    split_time = data.index[int(len(data) * 0.70)]

    in_sample = results[results["time"] < split_time]
    out_sample = results[results["time"] >= split_time]

    results.to_csv(
        OUTPUT_DIR / "all_trades.csv",
        index=False
    )

    def report(group, title):
        print("\n", title)
        print("-" * 45)

        if group.empty:
            print("No trades.")
            return

        for model in models:
            subset = group[group["model"] == model]
            if subset.empty:
                continue

            wins = (subset["outcome"] == "WIN").sum()
            losses = (subset["outcome"] == "LOSS").sum()
            total = wins + losses

            win_rate = (
                wins / total * 100 if total else 0
            )

            print(
                f"{model}: "
                f"Trades={len(subset)}, "
                f"Wins={wins}, "
                f"Losses={losses}, "
                f"Timeouts={len(subset) - total}, "
                f"Win Rate={win_rate:.2f}%"
            )

    report(in_sample, "IN-SAMPLE RESULTS")
    report(out_sample, "OUT-OF-SAMPLE RESULTS")

    print("\nResearch completed.")
    print("Symbol:", SYMBOL)
    print("Data:", INTERVAL, PERIOD)
    print("Minimum RR:", MIN_RR)
    print("Important: Costs and broker-feed differences")
    print("are not yet included.")
    print("This is NOT a production trading signal.")


if __name__ == "__main__":
    main()
