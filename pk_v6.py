
import os
import time
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone

# =========================================================
# PK V6 — FREE MULTI-TIMEFRAME SIGNAL ENGINE
# =========================================================

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

TIMEFRAMES = ["15m", "30m", "1h", "4h", "1d"]
SYMBOLS = {
    "BTCUSD": "BTCUSDT",
    "XAUUSD": "GC=F"
}

EMA_FAST = 20
EMA_MID = 50
EMA_SLOW = 200

RSI_PERIOD = 14
RSI_BUY = 10
RSI_SELL = 90

RVI_PERIOD = 8
RVI_LEVEL = 0.500

ATR_PERIOD = 14
RR_MIN = 4.0
SWING_LOOKBACK = 10
PIVOT_WINDOW = 3

BINANCE_URL = "https://fapi.binance.com"
YAHOO_PERIOD = "60d"

# =========================================================
# TELEGRAM
# =========================================================

def telegram_send(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram secrets missing.")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"

    try:
        response = requests.post(
            url,
            data={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message,
                "disable_web_page_preview": True
            },
            timeout=20
        )
        result = response.json()
        print("Telegram:", result.get("ok"), result.get("description", ""))
        return bool(result.get("ok"))
    except Exception as exc:
        print("Telegram error:", exc)
        return False


# =========================================================
# DATA COLLECTION
# =========================================================

def fetch_binance(symbol, interval, limit=1000):
    url = f"{BINANCE_URL}/fapi/v1/klines"

    response = requests.get(
        url,
        params={
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        },
        timeout=25
    )
    response.raise_for_status()

    rows = response.json()

    columns = [
        "time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ]

    df = pd.DataFrame(rows, columns=columns)

    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[["time", "open", "high", "low", "close", "volume"]]
    df = df.dropna().set_index("time").sort_index()

    # Remove unfinished candle
    if len(df) > 1:
        df = df.iloc[:-1]

    return df


def fetch_yahoo(interval):
    yahoo_interval = interval

    # Yahoo does not reliably provide native 4h candles.
    if interval == "4h":
        source_interval = "1h"
    else:
        source_interval = interval

    data = yf.download(
        "GC=F",
        period=YAHOO_PERIOD,
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

    data = data[["open", "high", "low", "close", "volume"]]
    data = data.dropna()

    if interval == "4h":
        data = data.resample("4h").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }).dropna()

    data.index = pd.to_datetime(data.index, utc=True)

    # Remove current unfinished candle
    if len(data) > 1:
        data = data.iloc[:-1]

    return data


def get_data(symbol, interval):
    if symbol == "BTCUSD":
        return fetch_binance(SYMBOLS[symbol], interval)

    return fetch_yahoo(interval)


# =========================================================
# INDICATORS
# =========================================================

def calculate_rsi(close, period=14):
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss > 0), 0)

    return rsi


def calculate_rvi(df, period=8):
    # Normalized RVI approximation.
    # This is NOT guaranteed to match TradingView RVI.

    close = df["close"]
    open_ = df["open"]
    high = df["high"]
    low = df["low"]

    numerator = (
        (close - open_) +
        2 * (close.shift(1) - open_.shift(1)) +
        2 * (close.shift(2) - open_.shift(2)) +
        (close.shift(3) - open_.shift(3))
    ) / 6

    denominator = (
        (high - low) +
        2 * (high.shift(1) - low.shift(1)) +
        2 * (high.shift(2) - low.shift(2)) +
        (high.shift(3) - low.shift(3))
    ) / 6

    numerator_smooth = numerator.rolling(period).mean()
    denominator_smooth = denominator.rolling(period).mean()

    return numerator_smooth / denominator_smooth.replace(0, np.nan)


def add_indicators(df):
    df = df.copy()

    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()

    df["rsi"] = calculate_rsi(df["close"], RSI_PERIOD)
    df["rvi"] = calculate_rvi(df, RVI_PERIOD)

    previous_close = df["close"].shift(1)

    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - previous_close).abs(),
        (df["low"] - previous_close).abs()
    ], axis=1).max(axis=1)

    df["atr"] = tr.ewm(
        alpha=1 / ATR_PERIOD,
        min_periods=ATR_PERIOD,
        adjust=False
    ).mean()

    return df.dropna()


# =========================================================
# HIGHER TIMEFRAME BIAS
# =========================================================

def get_bias(symbol):
    try:
        h1 = add_indicators(get_data(symbol, "1h"))
        h4 = add_indicators(get_data(symbol, "4h"))

        if len(h1) < 5 or len(h4) < 5:
            return "UNKNOWN"

        a = h1.iloc[-1]
        b = h4.iloc[-1]

        bullish = (
            a.ema20 > a.ema50 > a.ema200
            and b.ema20 > b.ema50 > b.ema200
        )

        bearish = (
            a.ema20 < a.ema50 < a.ema200
            and b.ema20 < b.ema50 < b.ema200
        )

        if bullish:
            return "BULLISH"

        if bearish:
            return "BEARISH"

        return "NEUTRAL"

    except Exception as exc:
        print("Bias error:", symbol, exc)
        return "UNKNOWN"


# =========================================================
# LIQUIDITY DETECTION
# =========================================================

def detect_liquidity(df):
    if len(df) < 30:
        return {}

    recent = df.iloc[-100:].copy()

    highs = recent["high"]
    lows = recent["low"]

    current = recent.iloc[-1]
    previous = recent.iloc[-2]

    # Approximate pivot highs/lows
    pivot_highs = []
    pivot_lows = []

    for i in range(PIVOT_WINDOW, len(recent) - PIVOT_WINDOW):
        h = highs.iloc[i]
        l = lows.iloc[i]

        if h == highs.iloc[
            i-PIVOT_WINDOW:i+PIVOT_WINDOW+1
        ].max():
            pivot_highs.append(float(h))

        if l == lows.iloc[
            i-PIVOT_WINDOW:i+PIVOT_WINDOW+1
        ].min():
            pivot_lows.append(float(l))

    atr = float(current.atr) if pd.notna(current.atr) else 0

    tolerance = max(atr * 0.15, float(current.close) * 0.0005)

    def find_equal_levels(values):
        if len(values) < 2:
            return None

        for i in range(len(values) - 1):
            for j in range(i + 1, len(values)):
                if abs(values[i] - values[j]) <= tolerance:
                    return round((values[i] + values[j]) / 2, 5)

        return None

    eqh = find_equal_levels(pivot_highs)
    eql = find_equal_levels(pivot_lows)

    swing_high = max(pivot_highs) if pivot_highs else None
    swing_low = min(pivot_lows) if pivot_lows else None

    # Previous completed daily/weekly levels from available data
    daily = recent.resample("1D").agg({
        "high": "max",
        "low": "min"
    }).dropna()

    weekly = recent.resample("1W").agg({
        "high": "max",
        "low": "min"
    }).dropna()

    pdh = float(daily.iloc[-2]["high"]) if len(daily) >= 2 else None
    pdl = float(daily.iloc[-2]["low"]) if len(daily) >= 2 else None

    pwh = float(weekly.iloc[-2]["high"]) if len(weekly) >= 2 else None
    pwl = float(weekly.iloc[-2]["low"]) if len(weekly) >= 2 else None

    bullish_sweep = (
        previous.low < (swing_low if swing_low is not None else previous.low)
        and current.close > previous.low
    )

    bearish_sweep = (
        previous.high > (swing_high if swing_high is not None else previous.high)
        and current.close < previous.high
    )

    # Relative volume approximation
    average_volume = recent["volume"].rolling(20).mean().iloc[-1]

    relative_volume = (
        float(current.volume / average_volume)
        if pd.notna(average_volume) and average_volume > 0
        else None
    )

    return {
        "EQH": eqh,
        "EQL": eql,
        "Swing High": swing_high,
        "Swing Low": swing_low,
        "PDH": pdh,
        "PDL": pdl,
        "PWH": pwh,
        "PWL": pwl,
        "Bullish Sweep": bool(bullish_sweep),
        "Bearish Sweep": bool(bearish_sweep),
        "Relative Volume": relative_volume
    }


# =========================================================
# SETUP ENGINE
# =========================================================

def ema_retest(df, direction):
    if len(df) < 3:
        return False

    previous = df.iloc[-2]
    current = df.iloc[-1]

    bullish_alignment = (
        current.ema20 > current.ema50 > current.ema200
    )

    bearish_alignment = (
        current.ema20 < current.ema50 < current.ema200
    )

    if direction == "BUY":
        # Previous candle touches EMA20.
        # Current candle body closes above EMA20.
        touched = previous.low <= previous.ema20
        body_valid = (
            current.open > current.ema20
            and current.close > current.ema20
        )

        return bullish_alignment and touched and body_valid

    touched = previous.high >= previous.ema20
    body_valid = (
        current.open < current.ema20
        and current.close < current.ema20
    )

    return bearish_alignment and touched and body_valid


def rsi_reentry(df, direction):
    if len(df) < 3:
        return False

    rsi = df["rsi"]

    # First cross arms the setup.
    # Return through the threshold creates the entry.
    if direction == "BUY":
        armed = (rsi.iloc[:-1] < RSI_BUY).any()
        reentry = rsi.iloc[-2] <= RSI_BUY and rsi.iloc[-1] > RSI_BUY
        return bool(armed and reentry)

    armed = (rsi.iloc[:-1] > RSI_SELL).any()
    reentry = rsi.iloc[-2] >= RSI_SELL and rsi.iloc[-1] < RSI_SELL

    return bool(armed and reentry)


def rvi_reentry(df, direction):
    if len(df) < 3:
        return False

    rvi = df["rvi"]

    if direction == "BUY":
        armed = (rvi.iloc[:-1] > RVI_LEVEL).any()
        reentry = (
            rvi.iloc[-2] >= RVI_LEVEL
            and rvi.iloc[-1] < RVI_LEVEL
        )
        return bool(armed and reentry)

    # User-confirmed SELL Option B:
    # First below -0.500, then cross back above -0.500.
    armed = (rvi.iloc[:-1] < -RVI_LEVEL).any()
    reentry = (
        rvi.iloc[-2] <= -RVI_LEVEL
        and rvi.iloc[-1] > -RVI_LEVEL
    )

    return bool(armed and reentry)


# =========================================================
# RISK ENGINE
# =========================================================

def calculate_trade_levels(df, direction):
    current = df.iloc[-1]

    entry = float(current.close)
    atr = float(current.atr)

    if not np.isfinite(atr) or atr <= 0:
        return None

    recent = df.iloc[-SWING_LOOKBACK:]

    if direction == "BUY":
        swing = float(recent["low"].min())
        sl = min(swing, entry - 1.5 * atr)
        risk = entry - sl

        if risk <= 0:
            return None

        tp = entry + RR_MIN * risk

    else:
        swing = float(recent["high"].max())
        sl = max(swing, entry + 1.5 * atr)
        risk = sl - entry

        if risk <= 0:
            return None

        tp = entry - RR_MIN * risk

    return {
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "rr": RR_MIN
    }


# =========================================================
# SCAN ONE SYMBOL / TIMEFRAME
# =========================================================

def scan_one(symbol, timeframe, bias):
    try:
        raw = get_data(symbol, timeframe)

        if len(raw) < 250:
            return None, f"Insufficient data: {len(raw)} candles"

        df = add_indicators(raw)

        if len(df) < 10:
            return None, "Insufficient indicator history"

        liquidity = detect_liquidity(df)

        setups = []

        for direction in ["BUY", "SELL"]:
            bias_ok = (
                (direction == "BUY" and bias == "BULLISH")
                or
                (direction == "SELL" and bias == "BEARISH")
            )

            if not bias_ok:
                continue

            if ema_retest(df, direction):
                setups.append("EMA20 RETEST")

            if rsi_reentry(df, direction):
                setups.append("RSI RE-ENTRY")

            if rvi_reentry(df, direction):
                setups.append("RVI RE-CROSS")

            if setups:
                levels = calculate_trade_levels(df, direction)

                if levels:
                    return {
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "direction": direction,
                        "setups": list(setups),
                        "levels": levels,
                        "bias": bias,
                        "price": float(df.iloc[-1].close),
                        "rsi": float(df.iloc[-1].rsi),
                        "rvi": float(df.iloc[-1].rvi),
                        "ema20": float(df.iloc[-1].ema20),
                        "ema50": float(df.iloc[-1].ema50),
                        "ema200": float(df.iloc[-1].ema200),
                        "liquidity": liquidity,
                        "data_time": str(df.index[-1])
                    }, "Signal"

        return None, "No confirmed setup"

    except Exception as exc:
        print("Scan error:", symbol, timeframe, exc)
        return None, str(exc)


# =========================================================
# TELEGRAM FORMAT
# =========================================================

def format_signal(signal):
    levels = signal["levels"]
    liquidity = signal["liquidity"]

    direction_icon = "🟢" if signal["direction"] == "BUY" else "🔴"

    def show(value):
        return "N/A" if value is None else f"{value:.5f}"

    return f"""
{direction_icon} PK V6 SIGNAL

Symbol: {signal['symbol']}
Timeframe: {signal['timeframe']}
Direction: {signal['direction']}

Entry: {levels['entry']:.5f}
Stop Loss: {levels['sl']:.5f}
Take Profit: {levels['tp']:.5f}
Risk Reward: 1:{levels['rr']:.1f}

SETUPS:
{chr(10).join('✅ ' + x for x in signal['setups'])}

HTF BIAS: {signal['bias']}

EMA20: {signal['ema20']:.5f}
EMA50: {signal['ema50']:.5f}
EMA200: {signal['ema200']:.5f}

RSI(14): {signal['rsi']:.2f}
RVI(8): {signal['rvi']:.4f}

LIQUIDITY:
EQH: {show(liquidity.get('EQH'))}
EQL: {show(liquidity.get('EQL'))}
Swing High: {show(liquidity.get('Swing High'))}
Swing Low: {show(liquidity.get('Swing Low'))}
PDH: {show(liquidity.get('PDH'))}
PDL: {show(liquidity.get('PDL'))}
PWH: {show(liquidity.get('PWH'))}
PWL: {show(liquidity.get('PWL'))}

Bullish Sweep: {liquidity.get('Bullish Sweep')}
Bearish Sweep: {liquidity.get('Bearish Sweep')}
Relative Volume: {liquidity.get('Relative Volume')}

Data: {'Binance Futures' if signal['symbol'] == 'BTCUSD' else 'Yahoo GC=F'}
Mode: Signal only — no auto execution
"""


# =========================================================
# MAIN
# =========================================================

def main():
    print("PK V6 started:", datetime.now(timezone.utc).isoformat())

    all_signals = []
    report = ["PK V6 MARKET SCAN", ""]

    for symbol in SYMBOLS:
        print("Scanning:", symbol)

        bias = get_bias(symbol)
        report.append(f"{symbol} | HTF Bias: {bias}")

        if bias in ("UNKNOWN", "NEUTRAL"):
            report.append("No directional bias — entries blocked.")
            report.append("")
            continue

        for timeframe in TIMEFRAMES:
            signal, status = scan_one(symbol, timeframe, bias)

            if signal:
                all_signals.append(signal)
                report.append(
                    f"{timeframe}: {signal['direction']} — "
                    f"{', '.join(signal['setups'])}"
                )
            else:
                report.append(f"{timeframe}: {status}")

        report.append("")

    # Send each signal separately
    for signal in all_signals:
        telegram_send(format_signal(signal))
        time.sleep(1)

    # Always send a scan summary
    report.append(f"Signals found: {len(all_signals)}")
    report.append("CVD: Disabled")
    report.append("Mode: Signal only")
    report.append("")

    telegram_send("\n".join(report))

    print("PK V6 finished. Signals:", len(all_signals))


if __name__ == "__main__":
    main()
