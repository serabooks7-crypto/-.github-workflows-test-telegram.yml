
import os
import requests
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone

# ==============================
# RLA XAUUSD AUTO SIGNAL BOT
# Experimental Version 1.0
# ==============================

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

SYMBOL = "GC=F"  # Gold Futures proxy, NOT spot XAUUSD

RVI_PERIOD = 10
RSI_PERIOD = 14
ATR_PERIOD = 14

def send_message(message):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    response = requests.post(
        url,
        data={"chat_id": CHAT_ID, "text": message},
        timeout=20
    )
    response.raise_for_status()

def get_data(interval, period):
    data = yf.download(
        SYMBOL,
        interval=interval,
        period=period,
        progress=False,
        auto_adjust=False
    )

    if data.empty:
        raise ValueError(f"No market data for {interval}")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data.dropna()

def calculate_indicators(df):
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    open_price = df["Open"]

    df["MA21"] = close.rolling(21).mean()
    df["MA50"] = close.rolling(50).mean()
    df["MA200"] = close.rolling(200).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/RSI_PERIOD, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/RSI_PERIOD, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    numerator = (
        (close - open_price) +
        2 * (close.shift(1) - open_price.shift(1)) +
        2 * (close.shift(2) - open_price.shift(2)) +
        (close.shift(3) - open_price.shift(3))
    ) / 6

    denominator = (
        (high - low) +
        2 * (high.shift(1) - low.shift(1)) +
        2 * (high.shift(2) - low.shift(2)) +
        (high.shift(3) - low.shift(3))
    ) / 6

    df["RVI"] = (
        numerator.rolling(RVI_PERIOD).mean() /
        denominator.rolling(RVI_PERIOD).mean()
    )

    df["RVI_SIGNAL"] = df["RVI"].rolling(4).mean()

    previous_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - previous_close).abs(),
        (low - previous_close).abs()
    ], axis=1).max(axis=1)

    df["ATR"] = tr.ewm(
        alpha=1/ATR_PERIOD,
        adjust=False
    ).mean()

    return df.dropna()

def create_chart(df, signal, entry, sl):
    chart = df.tail(60)

    plt.figure(figsize=(12, 7))
    plt.plot(chart.index, chart["Close"], label="Gold Price")
    plt.plot(chart.index, chart["MA21"], label="MA21")
    plt.plot(chart.index, chart["MA50"], label="MA50")
    plt.plot(chart.index, chart["MA200"], label="MA200")

    plt.axhline(entry, linestyle="--", label=f"Entry: {entry:.2f}")
    plt.axhline(sl, linestyle=":", label=f"SL: {sl:.2f}")

    plt.title(f"RLA GOLD SIGNAL - {signal}")
    plt.xlabel("Time")
    plt.ylabel("Gold Futures Price")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("signal_chart.png", dpi=140)
    plt.close()

def main():
    if not TOKEN or not CHAT_ID:
        raise ValueError("Telegram secrets are missing")

    # Get M15 and M5 data
    m15 = calculate_indicators(get_data("15m", "60d"))
    m5 = calculate_indicators(get_data("5m", "5d"))

    if len(m15) < 205 or len(m5) < 10:
        raise ValueError("Not enough market data")

    # Use last completed candles only
    trend = m15.iloc[-2]
    previous = m5.iloc[-3]
    current = m5.iloc[-2]

    buy_trend = (
        trend["MA21"] > trend["MA50"] >
        trend["MA200"]
    )

    sell_trend = (
        trend["MA21"] < trend["MA50"] <
        trend["MA200"]
    )

    buy_cross = (
        previous["RVI"] <= previous["RVI_SIGNAL"]
        and current["RVI"] > current["RVI_SIGNAL"]
    )

    sell_cross = (
        previous["RVI"] >= previous["RVI_SIGNAL"]
        and current["RVI"] < current["RVI_SIGNAL"]
    )

    signal = None

    if buy_trend and buy_cross and current["RSI"] > 50:
        signal = "BUY"

    elif sell_trend and sell_cross and current["RSI"] < 50:
        signal = "SELL"

    if signal is None:
        send_message(
            "RLA GOLD BOT\nNo confirmed signal.\n"
            "Market data: GC=F Gold Futures proxy."
        )
        return

    entry = float(current["Close"])
    atr = float(current["ATR"])

    if not np.isfinite(entry) or not np.isfinite(atr) or atr <= 0:
        raise ValueError("Invalid entry or ATR")

    if signal == "BUY":
        sl = entry - (2 * atr)
    else:
        sl = entry + (2 * atr)

    create_chart(m5, signal, entry, sl)

    message = (
        f"RLA GOLD SIGNAL: {signal}\n\n"
        f"Entry: {entry:.2f}\n"
        f"Stop Loss: {sl:.2f}\n"
        f"RSI: {current['RSI']:.2f}\n"
        f"RVI: {current['RVI']:.4f}\n"
        f"ATR: {atr:.2f}\n\n"
        "Data: GC=F Gold Futures proxy\n"
        "Not verified Exness spot XAUUSD.\n"
        "Experimental strategy - not backtested."
    )

    url = f"https://api.telegram.org/bot{TOKEN}/sendPhoto"

    with open("signal_chart.png", "rb") as photo:
        response = requests.post(
            url,
            data={"chat_id": CHAT_ID, "caption": message},
            files={"photo": photo},
            timeout=30
        )

    response.raise_for_status()

if __name__ == "__main__":
    main()
