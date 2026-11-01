
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ==========================================
# RLA MULTI-STRATEGY RESEARCH ENGINE V2
# XAUUSD research proxy: GC=F
# NOT FundingPips MT5 execution data
# ==========================================

SYMBOL = "GC=F"
INTERVAL = "5m"
PERIOD = "60d"

ATR_PERIOD = 14
EMA_FAST = 20
EMA_SLOW = 50
RSI_PERIOD = 14

# Test multiple reward targets.
RR_LEVELS = [4, 5, 6, 8, 10]

STOP_ATR_MULTIPLIER = 1.5
MAX_HOLDING_BARS = 60

# Set these after checking actual broker costs.
# All values are in price units, not pips.
SPREAD_PRICE = 0.0
SLIPPAGE_PRICE = 0.0
COMMISSION_PRICE = 0.0

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

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()

    if len(df) < 1000:
        raise RuntimeError(
            f"Insufficient candles: {len(df)}"
        )

    # Remove potentially incomplete latest candle.
    df = df.iloc[:-1].copy()

    if len(df) < 500:
        raise RuntimeError("Not enough completed candles.")

    print("Candles downloaded:", len(df))
    print("First candle:", df.index[0])
    print("Last completed candle:", df.index[-1])

    return df


def calculate_indicators(df):
    df = df.copy()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    df["EMA20"] = close.ewm(
        span=EMA_FAST,
        adjust=False
    ).mean()

    df["EMA50"] = close.ewm(
        span=EMA_SLOW,
        adjust=False
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

    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ],
        axis=1
    ).max(axis=1)

    df["ATR"] = tr.ewm(
        alpha=1 / ATR_PERIOD,
        adjust=False
    ).mean()

    # Previous 20-candle range.
    # Current candle is excluded.
    df["RANGE_HIGH"] = high.shift(1).rolling(20).max()
    df["RANGE_LOW"] = low.shift(1).rolling(20).min()

    # Liquidity sweep and reclaim.
    df["SWEEP_BUY"] = (
        (low < df["RANGE_LOW"]) &
        (close > df["RANGE_LOW"])
    )

    df["SWEEP_SELL"] = (
        (high > df["RANGE_HIGH"]) &
        (close < df["RANGE_HIGH"])
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
        df["SWEEP_BUY"] &
        (df["RSI"] > 45)
    )

    df["SWEEP_SELL"] = (
        df["SWEEP_SELL"] &
        (df["RSI"] < 55)
    )

    # Model 3: Range Breakout
    df["BREAKOUT_BUY"] = (
        (df["Close"] > df["RANGE_HIGH"]) &
        (df["RSI"] > 60)
    )

    df["BREAKOUT_SELL"] = (
        (df["Close"] < df["RANGE_LOW"]) &
        (df["RSI"] < 40)
    )

    return df


def backtest_model(df, model, direction, rr):
    results = []

    signal_col = model + "_" + direction
    next_allowed_index = 0

    total_cost = (
        SPREAD_PRICE +
        SLIPPAGE_PRICE +
        COMMISSION_PRICE
    )

    for i in range(100, len(df) - MAX_HOLDING_BARS - 1):

        # No overlapping positions in this model.
        if i < next_allowed_index:
            continue

        if not bool(df.iloc[i][signal_col]):
            continue

        signal_candle = df.iloc[i]
        entry_candle = df.iloc[i + 1]

        # Entry at next candle open.
        entry = float(entry_candle["Open"])
        atr = float(signal_candle["ATR"])

        if not np.isfinite(entry) or not np.isfinite(atr):
            continue

        if entry <= 0 or atr <= 0:
            continue

        risk = STOP_ATR_MULTIPLIER * atr

        if direction == "BUY":
            sl = entry - risk
            tp = entry + (rr * risk)
        else:
            sl = entry + risk
            tp = entry - (rr * risk)

        outcome = "TIMEOUT"
        exit_price = None
        exit_index = None

        last_index = min(
            i + MAX_HOLDING_BARS + 1,
            len(df) - 1
        )

        for j in range(i + 1, last_index + 1):

            candle = df.iloc[j]

            if direction == "BUY":
                hit_sl = candle["Low"] <= sl
                hit_tp = candle["High"] >= tp
            else:
                hit_sl = candle["High"] >= sl
                hit_tp = candle["Low"] <= tp

            # Conservative same-candle rule.
            if hit_sl:
                outcome = "LOSS"
                exit_price = sl
                exit_index = j
                break

            if hit_tp:
                outcome = "TP"
                exit_price = tp
                exit_index = j
                break

        if exit_index is None:
            exit_index = last_index
            exit_price = float(df.iloc[last_index]["Close"])

        if direction == "BUY":
            gross_pnl = exit_price - entry
        else:
            gross_pnl = entry - exit_price

        net_pnl = gross_pnl - total_cost
        net_r = net_pnl / risk

        results.append({
            "signal_time": df.index[i],
            "entry_time": df.index[i + 1],
            "exit_time": df.index[exit_index],
            "model": model,
            "direction": direction,
            "rr_target": rr,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "exit_price": exit_price,
            "outcome": outcome,
            "gross_pnl": gross_pnl,
            "estimated_cost": total_cost,
            "net_pnl": net_pnl,
            "net_r": net_r
        })

        # Next trade only after this trade exits.
        next_allowed_index = exit_index + 1

    return results


def calculate_metrics(group):
    if group.empty:
        return None

    total = len(group)

    tp_hits = (group["outcome"] == "TP").sum()
    sl_hits = (group["outcome"] == "LOSS").sum()
    timeouts = (group["outcome"] == "TIMEOUT").sum()

    # Positive net outcome, including profitable timeouts.
    profitable = (group["net_pnl"] > 0).sum()

    gross_profit = group.loc[
        group["net_pnl"] > 0, "net_pnl"
    ].sum()

    gross_loss = abs(
        group.loc[
            group["net_pnl"] < 0, "net_pnl"
        ].sum()
    )

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0 else np.inf
    )

    equity_r = group["net_r"].cumsum()
    drawdown = equity_r - equity_r.cummax()

    return {
        "trades": total,
        "TP_hits": int(tp_hits),
        "SL_hits": int(sl_hits),
        "timeouts": int(timeouts),
        "TP_hit_rate_pct": round(tp_hits / total * 100, 2),
        "profitable_rate_pct": round(
            profitable / total * 100, 2
        ),
        "average_net_R": round(
            group["net_r"].mean(), 3
        ),
        "profit_factor": round(
            profit_factor, 3
        ) if np.isfinite(profit_factor) else "Inf",
        "max_drawdown_R": round(
            abs(drawdown.min()), 3
        )
    }


def create_report(results, data):
    if results.empty:
        print("No trades found.")
        return

    split_index = int(len(data) * 0.70)
    split_time = data.index[split_index]

    results["sample"] = np.where(
        results["signal_time"] < split_time,
        "IN_SAMPLE",
        "OUT_OF_SAMPLE"
    )

    results.to_csv(
        OUTPUT_DIR / "all_trades.csv",
        index=False
    )

    reports = []

    for sample in ["IN_SAMPLE", "OUT_OF_SAMPLE"]:
        subset = results[results["sample"] == sample]

        for model in sorted(subset["model"].unique()):
            for direction in ["BUY", "SELL"]:
                for rr in RR_LEVELS:

                    group = subset[
                        (subset["model"] == model) &
                        (subset["direction"] == direction) &
                        (subset["rr_target"] == rr)
                    ]

                    metrics = calculate_metrics(group)

                    if metrics is None:
                        continue

                    reports.append({
                        "sample": sample,
                        "model": model,
                        "direction": direction,
                        "target_RR": rr,
                        **metrics
                    })

    report = pd.DataFrame(reports)

    report.to_csv(
        OUTPUT_DIR / "performance_report.csv",
        index=False
    )

    print("\n========== PERFORMANCE REPORT ==========")

    if not report.empty:
        print(
            report.to_string(index=False)
        )

    print("\nFiles saved in:", OUTPUT_DIR)


def main():
    print("RLA Research Engine V2 started.")

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
            for rr in RR_LEVELS:

                print(
                    f"Testing {model} {direction} "
                    f"at 1:{rr}"
                )

                trades = backtest_model(
                    data,
                    model,
                    direction,
                    rr
                )

                all_results.extend(trades)

    if not all_results:
        print("No historical trades found.")
        return

    results = pd.DataFrame(all_results)

    create_report(results, data)

    print("\nResearch completed.")
    print("Symbol:", SYMBOL)
    print("Interval:", INTERVAL)
    print("Period:", PERIOD)
    print("RR levels:", RR_LEVELS)

    print("\nIMPORTANT:")
    print("GC=F is not FundingPips MT5 XAUUSD.")
    print("Costs are estimates configured by the user.")
    print("No live signals are generated.")
    print("This is a preliminary research engine.")


if __name__ == "__main__":
    main()
