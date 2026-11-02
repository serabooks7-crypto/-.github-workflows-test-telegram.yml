
import os
import time
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone

# ================= CONFIGURATION =================

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

SYMBOLS = {"BTCUSD": "BTCUSDT", "XAUUSD": "GC=F"}

TIMEFRAMES = ["5m", "15m", "30m", "1h", "4h", "1d"]
BIAS_TIMEFRAMES = ["1h", "4h"]

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
SWING_LOOKBACK = 20
PIVOT_WINDOW = 3
ARM_LOOKBACK = 50

# Buffer around opposing liquidity, expressed in ATR
LIQUIDITY_BUFFER_ATR = 0.15

BINANCE_URL = "https://fapi.binance.com"
INTRADAY_PERIOD = "60d"
DAILY_PERIOD = "2y"


# ================= TELEGRAM =================

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
        response.raise_for_status()
        result = response.json()
        print("Telegram:", result.get("ok"), result.get("description", ""))
        return bool(result.get("ok"))
    except Exception as exc:
        print("Telegram error:", exc)
        return False


# ================= DATA FETCHING =================

def fetch_binance(symbol, interval, limit=1000):
    url = f"{BINANCE_URL}/fapi/v1/klines"

    response = requests.get(
        url,
        params={"symbol": symbol, "interval": interval, "limit": limit},
        timeout=25
    )
    response.raise_for_status()

    columns = [
        "time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ]

    df = pd.DataFrame(response.json(), columns=columns)

    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df[["time", "open", "high", "low", "close", "volume"]]
        .dropna()
        .set_index("time")
        .sort_index()
    )

    # Remove currently forming candle
    if len(df) > 1:
        df = df.iloc[:-1]

    return df


def fetch_yahoo(interval):
    period = DAILY_PERIOD if interval == "1d" else INTRADAY_PERIOD

    # Yahoo does not provide native 4h bars reliably.
    source_interval = "1h" if interval == "4h" else interval

    data = yf.download(
        "GC=F",
        period=period,
        interval=source_interval,
        progress=False,
        auto_adjust=False
    )

    if data is None or data.empty:
        raise ValueError(f"Yahoo returned no data for {interval}")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    })

    required = ["open", "high", "low", "close", "volume"]

    data = data[required].dropna()
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

    # Remove current unfinished candle
    if len(data) > 1:
        data = data.iloc[:-1]

    return data


def get_data(symbol, interval):
    if symbol == "BTCUSD":
        return fetch_binance(SYMBOLS[symbol], interval)

    return fetch_yahoo(interval)


def get_daily_data(symbol):
    if symbol == "BTCUSD":
        return fetch_binance("BTCUSDT", "1d")

    return fetch_yahoo("1d")


# ================= INDICATORS =================

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
    close = df["close"]
    open_ = df["open"]
    high = df["high"]
    low = df["low"]

    numerator = (
        (close - open_)
        + 2 * (close.shift(1) - open_.shift(1))
        + 2 * (close.shift(2) - open_.shift(2))
        + (close.shift(3) - open_.shift(3))
    ) / 6

    denominator = (
        (high - low)
        + 2 * (high.shift(1) - low.shift(1))
        + 2 * (high.shift(2) - low.shift(2))
        + (high.shift(3) - low.shift(3))
    ) / 6

    numerator_smooth = numerator.rolling(period).mean()
    denominator_smooth = denominator.rolling(period).mean()

    return numerator_smooth / denominator_smooth.replace(0, np.nan)


def add_indicators(df):
    df = df.copy()

    df["ema20"] = df["close"].ewm(span=EMA_FAST, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=EMA_MID, adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=EMA_SLOW, adjust=False).mean()

    df["rsi"] = calculate_rsi(df["close"], RSI_PERIOD)
    df["rvi"] = calculate_rvi(df, RVI_PERIOD)

    previous_close = df["close"].shift(1)

    true_range = pd.concat([
        df["high"] - df["low"],
        (df["high"] - previous_close).abs(),
        (df["low"] - previous_close).abs()
    ], axis=1).max(axis=1)

    df["atr"] = true_range.ewm(
        alpha=1 / ATR_PERIOD,
        min_periods=ATR_PERIOD,
        adjust=False
    ).mean()

    return df.dropna()


# ================= HTF BIAS =================

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


# ================= LIQUIDITY LEVELS =================

def find_pivots(df):
    highs = []
    lows = []

    for i in range(PIVOT_WINDOW, len(df) - PIVOT_WINDOW):
        high = df["high"].iloc[i]
        low = df["low"].iloc[i]

        high_window = df["high"].iloc[
            i - PIVOT_WINDOW:i + PIVOT_WINDOW + 1
        ]

        low_window = df["low"].iloc[
            i - PIVOT_WINDOW:i + PIVOT_WINDOW + 1
        ]

        if high == high_window.max():
            highs.append(float(high))

        if low == low_window.min():
            lows.append(float(low))

    return highs, lows


def equal_level(values, tolerance):
    if len(values) < 2:
        return None

    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            if abs(values[i] - values[j]) <= tolerance:
                return (values[i] + values[j]) / 2

    return None


def get_period_levels(symbol):
    try:
        daily = get_daily_data(symbol)

        if len(daily) < 3:
            return {}

        daily = daily.copy()

        pdh = float(daily["high"].iloc[-2])
        pdl = float(daily["low"].iloc[-2])

        weekly = daily.resample("W").agg({
            "high": "max",
            "low": "min"
        }).dropna()

        pwh = float(weekly["high"].iloc[-2]) if len(weekly) >= 2 else None
        pwl = float(weekly["low"].iloc[-2]) if len(weekly) >= 2 else None

        return {
            "PDH": pdh,
            "PDL": pdl,
            "PWH": pwh,
            "PWL": pwl
        }

    except Exception as exc:
        print("Daily/weekly liquidity error:", symbol, exc)
        return {}


def detect_liquidity(df, period_levels=None):
    period_levels = period_levels or {}

    recent = df.iloc[-200:].copy()

    current = recent.iloc[-1]
    previous = recent.iloc[-2]

    pivot_highs, pivot_lows = find_pivots(recent)

    atr = float(current.atr)
    tolerance = max(atr * 0.15, float(current.close) * 0.0005)

    eqh = equal_level(pivot_highs, tolerance)
    eql = equal_level(pivot_lows, tolerance)

    swing_high = max(pivot_highs) if pivot_highs else None
    swing_low = min(pivot_lows) if pivot_lows else None

    # Sweep is checked against the previous candle's available structure.
    reference_highs = [
        x for x in pivot_highs
        if x >= float(previous.high)
    ]

    reference_lows = [
        x for x in pivot_lows
        if x <= float(previous.low)
    ]

    bullish_sweep = False
    bearish_sweep = False

    if swing_low is not None:
        bullish_sweep = (
            previous.low < swing_low
            and current.close > swing_low
        )

    if swing_high is not None:
        bearish_sweep = (
            previous.high > swing_high
            and current.close < swing_high
        )

    average_volume = recent["volume"].rolling(20).mean().iloc[-1]

    relative_volume = None

    if pd.notna(average_volume) and average_volume > 0:
        relative_volume = float(current.volume / average_volume)

    levels = {
        "EQH": eqh,
        "EQL": eql,
        "Swing High": swing_high,
        "Swing Low": swing_low,
        "PDH": period_levels.get("PDH"),
        "PDL": period_levels.get("PDL"),
        "PWH": period_levels.get("PWH"),
        "PWL": period_levels.get("PWL"),
        "Bullish Sweep": bool(bullish_sweep),
        "Bearish Sweep": bool(bearish_sweep),
        "Relative Volume": relative_volume
    }

    return levels


# ================= ENTRY MODELS =================

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
        touched = previous.low <= previous.ema20

        # Retest candle body must remain above EMA20.
        previous_body_valid = (
            previous.open > previous.ema20
            and previous.close > previous.ema20
        )

        confirmation = (
            current.open > current.ema20
            and current.close > current.ema20
        )

        return bool(
            bullish_alignment
            and touched
            and previous_body_valid
            and confirmation
        )

    touched = previous.high >= previous.ema20

    previous_body_valid = (
        previous.open < previous.ema20
        and previous.close < previous.ema20
    )

    confirmation = (
        current.open < current.ema20
        and current.close < current.ema20
    )

    return bool(
        bearish_alignment
        and touched
        and previous_body_valid
        and confirmation
    )


def recent_armed(series, threshold, direction):
    """
    Require a recent threshold extreme before the latest re-entry.
    The arming event cannot be an old event from the full history.
    """
    if len(series) < 3:
        return False

    prior = series.iloc[-(ARM_LOOKBACK + 2):-2]

    if direction == "BELOW":
        return bool((prior < threshold).any())

    return bool((prior > threshold).any())


def rsi_reentry(df, direction):
    if len(df) < 3:
        return False

    rsi = df["rsi"]

    if direction == "BUY":
        armed = recent_armed(rsi, RSI_BUY, "BELOW")
        crossed_back = (
            rsi.iloc[-2] <= RSI_BUY
            and rsi.iloc[-1] > RSI_BUY
        )
        return bool(armed and crossed_back)

    armed = recent_armed(rsi, RSI_SELL, "ABOVE")

    crossed_back = (
        rsi.iloc[-2] >= RSI_SELL
        and rsi.iloc[-1] < RSI_SELL
    )

    return bool(armed and crossed_back)


def rvi_reentry(df, direction):
    if len(df) < 3:
        return False

    rvi = df["rvi"]

    if direction == "BUY":
        armed = recent_armed(rvi, RVI_LEVEL, "ABOVE")

        crossed_back = (
            rvi.iloc[-2] >= RVI_LEVEL
            and rvi.iloc[-1] < RVI_LEVEL
        )

        return bool(armed and crossed_back)

    # User-confirmed Option B:
    # First below -0.500, then cross back above -0.500.
    armed = recent_armed(rvi, -RVI_LEVEL, "BELOW")

    crossed_back = (
        rvi.iloc[-2] <= -RVI_LEVEL
        and rvi.iloc[-1] > -RVI_LEVEL
    )

    return bool(armed and crossed_back)


# ================= OPPOSING LIQUIDITY CHECK =================

def opposing_liquidity_check(entry, direction, sl, liquidity, atr):
    risk = abs(entry - sl)

    if risk <= 0:
        return None, "Invalid risk"

    target_distance = RR_MIN * risk

    if direction == "BUY":
        mathematical_target = entry + target_distance
        candidate_levels = [
            liquidity.get("EQH"),
            liquidity.get("Swing High"),
            liquidity.get("PDH"),
            liquidity.get("PWH")
        ]

        opposing = [
            float(x) for x in candidate_levels
            if x is not None and float(x) > entry
        ]

        if not opposing:
            return None, "No verified opposing liquidity above entry"

        nearest = min(opposing)
        buffer = atr * LIQUIDITY_BUFFER_ATR
        available_target = nearest - buffer

        if available_target < mathematical_target:
            return None, (
                f"Opposing liquidity too close: {nearest:.5f}; "
                f"required target: {mathematical_target:.5f}"
            )

        return mathematical_target, f"Clear to target; nearest liquidity {nearest:.5f}"

    mathematical_target = entry - target_distance

    candidate_levels = [
        liquidity.get("EQL"),
        liquidity.get("Swing Low"),
        liquidity.get("PDL"),
        liquidity.get("PWL")
    ]

    opposing = [
        float(x) for x in candidate_levels
        if x is not None and float(x) < entry
    ]

    if not opposing:
        return None, "No verified opposing liquidity below entry"

    nearest = max(opposing)
    buffer = atr * LIQUIDITY_BUFFER_ATR
    available_target = nearest + buffer

    if available_target > mathematical_target:
        return None, (
            f"Opposing liquidity too close: {nearest:.5f}; "
            f"required target: {mathematical_target:.5f}"
        )

    return mathematical_target, f"Clear to target; nearest liquidity {nearest:.5f}"


# ================= TRADE LEVELS =================

def calculate_trade_levels(df, direction, liquidity):
    current = df.iloc[-1]

    entry = float(current.close)
    atr = float(current.atr)

    if not np.isfinite(atr) or atr <= 0:
        return None, "Invalid ATR"

    recent = df.iloc[-SWING_LOOKBACK:]

    if direction == "BUY":
        swing = float(recent["low"].min())
        sl = min(swing, entry - 1.5 * atr)

    else:
        swing = float(recent["high"].max())
        sl = max(swing, entry + 1.5 * atr)

    risk = abs(entry - sl)

    if risk <= 0:
        return None, "Invalid stop loss"

    target, reason = opposing_liquidity_check(
        entry, direction, sl, liquidity, atr
    )

    if target is None:
        return None, reason

    actual_rr = abs(target - entry) / risk

    if actual_rr < RR_MIN:
        return None, "Target does not satisfy minimum RR"

    return {
        "entry": entry,
        "sl": sl,
        "tp": target,
        "rr": actual_rr,
        "risk": risk,
        "target_check": reason
    }, "Valid"


# ================= SCANNER =================

def scan_one(symbol, timeframe, bias, period_levels):
    try:
        raw = get_data(symbol, timeframe)

        if len(raw) < 250:
            return None, f"Insufficient data: {len(raw)} candles"

        df = add_indicators(raw)

        if len(df) < 10:
            return None, "Insufficient indicator history"

        liquidity = detect_liquidity(df, period_levels)

        for direction in ["BUY", "SELL"]:

            bias_ok = (
                (direction == "BUY" and bias == "BULLISH")
                or
                (direction == "SELL" and bias == "BEARISH")
            )

            if not bias_ok:
                continue

            setups = []

            if ema_retest(df, direction):
                setups.append("EMA20 RETEST")

            if rsi_reentry(df, direction):
                setups.append("RSI RE-ENTRY")

            if rvi_reentry(df, direction):
                setups.append("RVI RE-CROSS")

            if not setups:
                continue

            levels, reason = calculate_trade_levels(
                df, direction, liquidity
            )

            if levels is None:
                print(
                    symbol, timeframe, direction,
                    "Rejected:", reason
                )
                continue

            current = df.iloc[-1]

            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "direction": direction,
                "setups": setups,
                "levels": levels,
                "bias": bias,
                "price": float(current.close),
                "rsi": float(current.rsi),
                "rvi": float(current.rvi),
                "ema20": float(current.ema20),
                "ema50": float(current.ema50),
                "ema200": float(current.ema200),
                "atr": float(current.atr),
                "liquidity": liquidity,
                "data_time": str(df.index[-1])
            }, "Signal"

        return None, "No confirmed setup"

    except Exception as exc:
        print("Scan error:", symbol, timeframe, exc)
        return None, str(exc)


# ================= MESSAGE FORMATTING =================

def format_signal(signal):
    levels = signal["levels"]
    liquidity = signal["liquidity"]

    icon = "🟢" if signal["direction"] == "BUY" else "🔴"

    def show(value):
        if value is None:
            return "N/A"
        return f"{value:.5f}"

    return f"""
{icon} PK V6.1 SIGNAL

Symbol: {signal['symbol']}
Timeframe: {signal['timeframe']}
Direction: {signal['direction']}

Entry: {levels['entry']:.5f}
Stop Loss: {levels['sl']:.5f}
Take Profit: {levels['tp']:.5f}
Risk Reward: 1:{levels['rr']:.2f}

SETUPS:
{chr(10).join('✅ ' + x for x in signal['setups'])}

HTF BIAS: {signal['bias']}

EMA20: {signal['ema20']:.5f}
EMA50: {signal['ema50']:.5f}
EMA200: {signal['ema200']:.5f}

RSI(14): {signal['rsi']:.2f}
RVI(8): {signal['rvi']:.4f}
ATR(14): {signal['atr']:.5f}

OPPOSING LIQUIDITY CHECK:
{levels['target_check']}

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
CVD: Disabled
"""


# ================= MAIN =================

def main():
    print("PK V6.1 started:", datetime.now(timezone.utc).isoformat())

    all_signals = []
    report = ["PK V6.1 MARKET SCAN", ""]

    for symbol in SYMBOLS:
        print("Scanning:", symbol)

        bias = get_bias(symbol)
        report.append(f"{symbol} | HTF Bias: {bias}")

        period_levels = get_period_levels(symbol)

        if bias in ("UNKNOWN", "NEUTRAL"):
            report.append("No directional bias — entries blocked.")
            report.append("")
            continue

        for timeframe in TIMEFRAMES:
            signal, status = scan_one(
                symbol, timeframe, bias, period_levels
            )

            if signal:
                all_signals.append(signal)
                report.append(
                    f"{timeframe}: {signal['direction']} — "
                    f"{', '.join(signal['setups'])}"
                )
            else:
                report.append(f"{timeframe}: {status}")

        report.append("")

    for signal in all_signals:
        telegram_send(format_signal(signal))
        time.sleep(1)

    report.append(f"Signals found: {len(all_signals)}")
    report.append("Opposing liquidity target check: Enabled")
    report.append("CVD: Disabled")
    report.append("Mode: Signal only")

    telegram_send("\n".join(report))

    print("PK V6.1 finished. Signals:", len(all_signals))


if __name__ == "__main__":
    main()
