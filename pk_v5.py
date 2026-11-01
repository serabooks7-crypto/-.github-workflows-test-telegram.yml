
import os
import io
import json
import time
import requests
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone

BOT_NAME = "PK"
TIMEFRAME = "15m"
SYMBOLS = {
    "BTCUSD": "BTC-USD",
    "XAUUSD": "GC=F"
}

RSI_PERIOD = 14
RSI_BUY = 10
RSI_SELL = 90
RVI_PERIOD = 8
RVI_BUY = -0.500
RVI_SELL = 0.500
EMA_PERIODS = [50, 100, 200]
RR = 4.0

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


def telegram_send(text, image=None):
    if not TOKEN or not CHAT_ID:
        print("Telegram secrets missing")
        return

    url = f"https://api.telegram.org/bot{TOKEN}/"

    try:
        if image:
            requests.post(
                url + "sendPhoto",
                data={"chat_id": CHAT_ID, "caption": text},
                files={"photo": image},
                timeout=30
            ).raise_for_status()
        else:
            requests.post(
                url + "sendMessage",
                data={"chat_id": CHAT_ID, "text": text},
                timeout=30
            ).raise_for_status()
    except Exception as e:
        print("Telegram error:", e)


def calculate_indicators(df):
    df = df.copy()

    for p in EMA_PERIODS:
        df[f"EMA{p}"] = df["Close"].ewm(
            span=p, adjust=False
        ).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / RSI_PERIOD, adjust=False
    ).mean()
    avg_loss = loss.ewm(
        alpha=1 / RSI_PERIOD, adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # Normalized Relative Vigor Index, range approximately -1 to +1
    num = (
        (df["Close"] - df["Open"]) +
        2 * (df["Close"].shift(1) - df["Open"].shift(1)) +
        2 * (df["Close"].shift(2) - df["Open"].shift(2)) +
        (df["Close"].shift(3) - df["Open"].shift(3))
    ) / 6

    den = (
        (df["High"] - df["Low"]) +
        2 * (df["High"].shift(1) - df["Low"].shift(1)) +
        2 * (df["High"].shift(2) - df["Low"].shift(2)) +
        (df["High"].shift(3) - df["Low"].shift(3))
    ) / 6

    df["RVI"] = (
        num.rolling(RVI_PERIOD).mean() /
        den.rolling(RVI_PERIOD).mean().replace(0, np.nan)
    )

    return df.dropna()


def get_data(ticker):
    df = yf.download(
        ticker,
        period="60d",
        interval=TIMEFRAME,
        auto_adjust=True,
        progress=False
    )

    if df.empty:
        raise ValueError("No market data received")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close"]
    df = df[required].dropna()

    return calculate_indicators(df)


def detect_signal(df):
    row = df.iloc[-1]
    previous = df.iloc[-2]

    buy = []
    sell = []

    # EMA: full alignment transition
    bull_now = (
        row["EMA50"] > row["EMA100"] > row["EMA200"]
    )
    bull_before = (
        previous["EMA50"] > previous["EMA100"] >
        previous["EMA200"]
    )

    bear_now = (
        row["EMA50"] < row["EMA100"] < row["EMA200"]
    )
    bear_before = (
        previous["EMA50"] < previous["EMA100"] <
        previous["EMA200"]
    )

    if bull_now and not bull_before:
        buy.append("EMA 50/100/200 Bullish Crossover")

    if bear_now and not bear_before:
        sell.append("EMA 50/100/200 Bearish Crossover")

    # RSI independent trigger
    if row["RSI"] <= RSI_BUY:
        buy.append("RSI(14) <= 10")

    if row["RSI"] >= RSI_SELL:
        sell.append("RSI(14) >= 90")

    # RVI independent trigger
    if row["RVI"] <= RVI_BUY:
        buy.append("RVI(8) <= -0.500")

    if row["RVI"] >= RVI_SELL:
        sell.append("RVI(8) >= +0.500")

    if buy and sell:
        return "CONFLICT", buy, sell

    if buy:
        return "BUY", buy, []

    if sell:
        return "SELL", [], sell

    return None, [], []


def make_chart(df, symbol, direction, entry, sl, tp):
    recent = df.tail(100)

    fig, axes = plt.subplots(
        3, 1, figsize=(14, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1, 1]}
    )

    ax = axes[0]
    x = np.arange(len(recent))

    for i, (_, r) in enumerate(recent.iterrows()):
        color = "green" if r["Close"] >= r["Open"] else "red"
        ax.vlines(i, r["Low"], r["High"], color=color)
        ax.bar(
            i, abs(r["Close"] - r["Open"]),
            bottom=min(r["Open"], r["Close"]),
            color=color, width=0.65
        )

    for col in ["EMA50", "EMA100", "EMA200"]:
        ax.plot(x, recent[col], label=col)

    ax.axhline(entry, linestyle="--", label="Entry")
    ax.axhline(sl, linestyle=":", label="Stop Loss")
    ax.axhline(tp, linestyle=":", label="Take Profit")
    ax.set_title(f"{BOT_NAME} | {symbol} | {direction}")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.2)

    axes[1].plot(x, recent["RSI"], label="RSI 14")
    axes[1].axhline(90, linestyle="--")
    axes[1].axhline(10, linestyle="--")
    axes[1].set_ylim(0, 100)
    axes[1].legend()
    axes[1].grid(alpha=0.2)

    axes[2].plot(x, recent["RVI"], label="RVI 8")
    axes[2].axhline(0.5, linestyle="--")
    axes[2].axhline(-0.5, linestyle="--")
    axes[2].legend()
    axes[2].grid(alpha=0.2)

    plt.tight_layout()

    image = io.BytesIO()
    fig.savefig(image, format="png", dpi=130)
    plt.close(fig)
    image.seek(0)

    return image


def process_symbol(symbol, ticker):
    try:
        df = get_data(ticker)

        if len(df) < 210:
            print(symbol, "Not enough candles")
            return

        direction, buy, sell = detect_signal(df)

        if direction is None:
            print(symbol, "No signal")
            return

        if direction == "CONFLICT":
            msg = (
                f"⚠️ {BOT_NAME} SIGNAL CONFLICT\n"
                f"Symbol: {symbol}\n"
                f"BUY triggers: {', '.join(buy)}\n"
                f"SELL triggers: {', '.join(sell)}\n"
                "No trade recommendation."
            )
            telegram_send(msg)
            return

        row = df.iloc[-1]
        entry = float(row["Close"])

        # ATR-based structural risk approximation
        tr = pd.concat([
            df["High"] - df["Low"],
            (df["High"] - df["Close"].shift()).abs(),
            (df["Low"] - df["Close"].shift()).abs()
        ], axis=1).max(axis=1)

        atr = float(tr.rolling(14).mean().iloc[-1])

        if not np.isfinite(atr) or atr <= 0:
            return

        recent_low = float(df["Low"].tail(10).min())
        recent_high = float(df["High"].tail(10).max())

        if direction == "BUY":
            sl = min(recent_low, entry - 1.5 * atr)
            risk = entry - sl
            tp = entry + RR * risk
            triggers = buy
        else:
            sl = max(recent_high, entry + 1.5 * atr)
            risk = sl - entry
            tp = entry - RR * risk
            triggers = sell

        if risk <= 0:
            return

        count = len(triggers)

        message = (
            f"🚨 {BOT_NAME} V5 SIGNAL\n\n"
            f"Symbol: {symbol}\n"
            f"Timeframe: {TIMEFRAME}\n"
            f"Direction: {direction}\n"
            f"Entry: {entry:.5f}\n"
            f"Stop Loss: {sl:.5f}\n"
            f"Take Profit: {tp:.5f}\n"
            f"Risk Reward: 1:{RR:.0f}\n\n"
            f"SETUP COUNT: {count}/3\n"
            f"Triggered:\n" +
            "\n".join(f"✅ {t}" for t in triggers) +
            f"\n\nRSI: {row['RSI']:.2f}"
            f"\nRVI: {row['RVI']:.3f}"
            f"\nEMA50: {row['EMA50']:.5f}"
            f"\nEMA100: {row['EMA100']:.5f}"
            f"\nEMA200: {row['EMA200']:.5f}"
            f"\n\nData: Yahoo Finance"
            f"\nMode: Signal only, no auto execution"
        )

        chart = make_chart(
            df, symbol, direction, entry, sl, tp
        )

        telegram_send(message, chart)

    except Exception as e:
        print(symbol, "ERROR:", str(e))


def main():
    print(f"{BOT_NAME} V5 started")

    for symbol, ticker in SYMBOLS.items():
        process_symbol(symbol, ticker)


if __name__ == "__main__":
    main()
