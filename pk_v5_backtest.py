
import os
import yfinance as yf
import pandas as pd
import numpy as np

SYMBOLS = {
    "BTCUSD": "BTC-USD",
    "XAUUSD": "GC=F"
}

PERIOD = "60d"
INTERVAL = "15m"

RSI_PERIOD = 14
RVI_PERIOD = 8
ATR_PERIOD = 14

RSI_BUY = 10
RSI_SELL = 90
RVI_BUY = -0.500
RVI_SELL = 0.500

EMA_FAST = 50
EMA_MID = 100
EMA_SLOW = 200

RR = 4.0
SWING_LOOKBACK = 10
ATR_MULTIPLIER = 1.5

OUTPUT_DIR = "pk_backtest_results"


def download_data(ticker):
    df = yf.download(
        ticker,
        period=PERIOD,
        interval=INTERVAL,
        auto_adjust=False,
        progress=False
    )

    if df.empty:
        raise ValueError("No Yahoo Finance data returned.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close"]].copy()
    df = df.dropna()

    if len(df) < 300:
        raise ValueError(f"Insufficient candles: {len(df)}")

    return df


def indicators(df):
    d = df.copy()

    for period, name in [
        (EMA_FAST, "EMA50"),
        (EMA_MID, "EMA100"),
        (EMA_SLOW, "EMA200")
    ]:
        d[name] = d["Close"].ewm(
            span=period, adjust=False
        ).mean()

    delta = d["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / RSI_PERIOD,
        min_periods=RSI_PERIOD,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / RSI_PERIOD,
        min_periods=RSI_PERIOD,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    d["RSI"] = 100 - (100 / (1 + rs))
    d.loc[avg_loss == 0, "RSI"] = 100
    d.loc[avg_gain == 0, "RSI"] = 0

    numerator = d["Close"] - d["Open"]
    denominator = (d["High"] - d["Low"]).replace(0, np.nan)

    nw = (
        numerator + 2 * numerator.shift(1)
        + 2 * numerator.shift(2) + numerator.shift(3)
    ) / 6

    dw = (
        denominator + 2 * denominator.shift(1)
        + 2 * denominator.shift(2) + denominator.shift(3)
    ) / 6

    d["RVI"] = (nw / dw).rolling(
        RVI_PERIOD, min_periods=RVI_PERIOD
    ).mean()

    prev_close = d["Close"].shift(1)

    tr = pd.concat([
        d["High"] - d["Low"],
        (d["High"] - prev_close).abs(),
        (d["Low"] - prev_close).abs()
    ], axis=1).max(axis=1)

    d["ATR"] = tr.ewm(
        alpha=1 / ATR_PERIOD,
        min_periods=ATR_PERIOD,
        adjust=False
    ).mean()

    return d.dropna()


def backtest(symbol, df):
    trades = []
    active_trade = None

    for i in range(1, len(df) - 1):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        next_row = df.iloc[i + 1]

        # First manage any existing trade.
        if active_trade is not None:
            direction = active_trade["direction"]
            sl = active_trade["sl"]
            tp = active_trade["tp"]

            hit_sl = (
                row["Low"] <= sl if direction == "BUY"
                else row["High"] >= sl
            )

            hit_tp = (
                row["High"] >= tp if direction == "BUY"
                else row["Low"] <= tp
            )

            if hit_sl or hit_tp:
                # Conservative assumption: SL first if both touched.
                if hit_sl:
                    active_trade["exit"] = sl
                    active_trade["result_r"] = -1.0
                    active_trade["outcome"] = "LOSS"
                else:
                    active_trade["exit"] = tp
                    active_trade["result_r"] = RR
                    active_trade["outcome"] = "WIN"

                active_trade["exit_time"] = df.index[i]
                trades.append(active_trade)
                active_trade = None

            # Do not open another trade on the same candle.
            continue

        # EMA alignment transition
        bull = (
            row["EMA50"] > row["EMA100"] > row["EMA200"]
        )
        bear = (
            row["EMA50"] < row["EMA100"] < row["EMA200"]
        )

        prev_bull = (
            prev["EMA50"] > prev["EMA100"] > prev["EMA200"]
        )
        prev_bear = (
            prev["EMA50"] < prev["EMA100"] < prev["EMA200"]
        )

        ema_buy = bull and not prev_bull
        ema_sell = bear and not prev_bear

        # RSI threshold crossing
        rsi_buy = (
            row["RSI"] <= RSI_BUY
            and prev["RSI"] > RSI_BUY
        )
        rsi_sell = (
            row["RSI"] >= RSI_SELL
            and prev["RSI"] < RSI_SELL
        )

        # RVI threshold crossing
        rvi_buy = (
            row["RVI"] <= RVI_BUY
            and prev["RVI"] > RVI_BUY
        )
        rvi_sell = (
            row["RVI"] >= RVI_SELL
            and prev["RVI"] < RVI_SELL
        )

        buy_count = sum([ema_buy, rsi_buy, rvi_buy])
        sell_count = sum([ema_sell, rsi_sell, rvi_sell])

        if buy_count and sell_count:
            continue

        if buy_count:
            direction = "BUY"
            count = buy_count
        elif sell_count:
            direction = "SELL"
            count = sell_count
        else:
            continue

        # Entry on the next candle's open.
        entry = float(next_row["Open"])
        atr = float(row["ATR"])

        recent = df.iloc[
            max(0, i - SWING_LOOKBACK + 1):i + 1
        ]

        if direction == "BUY":
            swing_sl = float(recent["Low"].min())
            atr_sl = entry - ATR_MULTIPLIER * atr
            sl = min(swing_sl, atr_sl)
            risk = entry - sl
            tp = entry + RR * risk
        else:
            swing_sl = float(recent["High"].max())
            atr_sl = entry + ATR_MULTIPLIER * atr
            sl = max(swing_sl, atr_sl)
            risk = sl - entry
            tp = entry - RR * risk

        if risk <= 0 or not np.isfinite(risk):
            continue

        active_trade = {
            "symbol": symbol,
            "direction": direction,
            "signal_time": df.index[i],
            "entry_time": df.index[i + 1],
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "risk": risk,
            "rr": RR,
            "setup_count": f"{count}/3",
            "ema_trigger": bool(ema_buy or ema_sell),
            "rsi_trigger": bool(rsi_buy or rsi_sell),
            "rvi_trigger": bool(rvi_buy or rvi_sell)
        }

    # Close any remaining open trade at the final available close.
    if active_trade is not None:
        final_price = float(df.iloc[-1]["Close"])
        direction = active_trade["direction"]

        pnl = (
            final_price - active_trade["entry"]
            if direction == "BUY"
            else active_trade["entry"] - final_price
        )

        active_trade["exit"] = final_price
        active_trade["result_r"] = pnl / active_trade["risk"]
        active_trade["outcome"] = "OPEN_AT_END"
        active_trade["exit_time"] = df.index[-1]
        trades.append(active_trade)

    return trades


def summarize(symbol, trades):
    if not trades:
        return {
            "Symbol": symbol,
            "Trades": 0,
            "Wins": 0,
            "Losses": 0,
            "Win Rate %": 0,
            "Profit Factor": 0,
            "Net R": 0,
            "Max Drawdown R": 0
        }

    t = pd.DataFrame(trades)

    # Open-at-end trades are reported separately, not counted
    # as completed wins or losses.
    completed = t[t["outcome"].isin(["WIN", "LOSS"])].copy()

    wins = int((completed["outcome"] == "WIN").sum())
    losses = int((completed["outcome"] == "LOSS").sum())
    total = len(completed)

    gross_profit = completed.loc[
        completed["result_r"] > 0, "result_r"
    ].sum()

    gross_loss = abs(completed.loc[
        completed["result_r"] < 0, "result_r"
    ].sum())

    pf = (
        gross_profit / gross_loss
        if gross_loss > 0
        else (float("inf") if gross_profit > 0 else 0)
    )

    equity = completed["result_r"].cumsum()
    peak = equity.cummax()
    drawdown = equity - peak
    max_dd = abs(float(drawdown.min())) if len(drawdown) else 0

    win_rate = wins / total * 100 if total else 0

    return {
        "Symbol": symbol,
        "Trades": total,
        "Wins": wins,
        "Losses": losses,
        "Open at end": int((t["outcome"] == "OPEN_AT_END").sum()),
        "Win Rate %": round(win_rate, 2),
        "Profit Factor": round(pf, 2) if np.isfinite(pf) else "INF",
        "Net R": round(float(completed["result_r"].sum()), 2),
        "Max Drawdown R": round(max_dd, 2)
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_trades = []
    summaries = []

    print("PK V5 BACKTEST STARTED")
    print("Period:", PERIOD, "| Timeframe:", INTERVAL)

    for symbol, ticker in SYMBOLS.items():
        try:
            raw = download_data(ticker)
            df = indicators(raw)

            trades = backtest(symbol, df)
            all_trades.extend(trades)

            summary = summarize(symbol, trades)
            summaries.append(summary)

            print("\n", summary)

        except Exception as e:
            print(f"{symbol} ERROR: {e}")

    if all_trades:
        pd.DataFrame(all_trades).to_csv(
            f"{OUTPUT_DIR}/all_trades.csv",
            index=False
        )

    if summaries:
        pd.DataFrame(summaries).to_csv(
            f"{OUTPUT_DIR}/summary.csv",
            index=False
        )

    print("\nBACKTEST FINISHED")
    print("Results saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
