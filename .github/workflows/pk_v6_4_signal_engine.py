
"""
PK BOT V6.4 - SIGNAL ENGINE
Research and signal-generation module only.
No live orders. No Telegram messages.

Required sibling modules:
- pk_v6_4_indicators.py
- pk_v6_4_price_action.py

Input:
data_by_timeframe = {
    "15m": dataframe,
    "30m": dataframe,
    "1h": dataframe,
    "2h": dataframe,
    "4h": dataframe
}

All input candles must be CLOSED candles only.
"""

from pathlib import Path
import sys
import hashlib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pk_v6_4_indicators import calculate_indicators
from pk_v6_4_price_action import calculate_price_action


VERSION = "6.4"

SIGNAL_TIMEFRAMES = ["15m", "30m", "1h", "2h", "4h"]
EMA_TIMEFRAMES = ["30m", "1h", "2h", "4h"]
RSI_TIMEFRAMES = ["15m", "30m", "1h", "2h", "4h"]
RVI_TIMEFRAMES = ["15m", "30m", "1h", "2h", "4h"]

TIMEFRAME_MINUTES = {
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "2h": 120,
    "4h": 240
}

MINIMUM_RR = 4.0
ATR_SL_BUFFER = 0.12

RSI_PERIOD = 14
RSI_BUY_LEVEL = 15.0
RSI_SELL_LEVEL = 85.0

RVI_BUY_LEVEL = -0.400
RVI_SELL_LEVEL = 0.400


# --------------------------------------------------
# BASIC VALIDATION
# --------------------------------------------------

def validate_timeframe_data(data_by_timeframe):

    if not isinstance(data_by_timeframe, dict):
        raise TypeError("Input must be a timeframe dictionary.")

    required = ["15m", "30m", "1h", "2h", "4h"]

    for tf in required:
        if tf not in data_by_timeframe:
            raise ValueError(f"Missing timeframe: {tf}")

        df = data_by_timeframe[tf]

        if df is None or df.empty:
            raise ValueError(f"No data for timeframe: {tf}")

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(f"{tf}: DatetimeIndex required.")

        if not df.index.is_monotonic_increasing:
            raise ValueError(f"{tf}: Data must be time-sorted.")

        if df.index.has_duplicates:
            raise ValueError(f"{tf}: Duplicate timestamps found.")

    return True


def prepare_engine_data(raw_data):

    indicators = calculate_indicators(raw_data)
    price_action = calculate_price_action(raw_data)

    pa_columns = [
        "LAST_CONFIRMED_HIGH",
        "LAST_CONFIRMED_LOW",
        "CONFIRMED_PIVOT_HIGH",
        "CONFIRMED_PIVOT_LOW",
        "BSL_SWEEP",
        "SSL_SWEEP",
        "BULLISH_LIQUIDITY_SWEEP",
        "BEARISH_LIQUIDITY_SWEEP",
        "BULLISH_FVG",
        "BEARISH_FVG",
        "BULLISH_IFVG",
        "BEARISH_IFVG",
        "BULLISH_DISPLACEMENT",
        "BEARISH_DISPLACEMENT",
        "BULLISH_EMA100_RETEST",
        "BEARISH_EMA100_RETEST"
    ]

    for col in pa_columns:
        if col not in price_action.columns:
            raise ValueError(f"Price Action column missing: {col}")

    for col in pa_columns:
        indicators[col] = price_action[col].reindex(indicators.index)

    return indicators


# --------------------------------------------------
# CLOSED HIGHER-TIMEFRAME FILTER
# --------------------------------------------------

def attach_closed_htf_filter(
    signal_df,
    htf_df,
    signal_tf,
    htf_tf,
    output_prefix
):

    signal_minutes = TIMEFRAME_MINUTES[signal_tf]
    htf_minutes = TIMEFRAME_MINUTES[htf_tf]

    signal_close_times = (
        signal_df.index
        + pd.to_timedelta(signal_minutes, unit="m")
    )

    htf_available_times = (
        htf_df.index
        + pd.to_timedelta(htf_minutes, unit="m")
    )

    htf_values = htf_df[["Close", "EMA200"]].copy()
    htf_values["AVAILABLE_AT"] = htf_available_times
    htf_values = htf_values.reset_index(drop=True)

    available_ns = htf_values["AVAILABLE_AT"].astype("int64").to_numpy()
    signal_ns = pd.DatetimeIndex(signal_close_times).astype("int64").to_numpy()

    positions = np.searchsorted(
        available_ns,
        signal_ns,
        side="right"
    ) - 1

    htf_close = np.full(len(signal_df), np.nan)
    htf_ema = np.full(len(signal_df), np.nan)

    valid = positions >= 0

    if valid.any():
        htf_close[valid] = (
            htf_values["Close"].to_numpy()[positions[valid]]
        )
        htf_ema[valid] = (
            htf_values["EMA200"].to_numpy()[positions[valid]]
        )

    signal_df[f"{output_prefix}_CLOSE"] = htf_close
    signal_df[f"{output_prefix}_EMA200"] = htf_ema

    signal_df[f"{output_prefix}_BULLISH"] = (
        signal_df[f"{output_prefix}_CLOSE"]
        > signal_df[f"{output_prefix}_EMA200"]
    )

    signal_df[f"{output_prefix}_BEARISH"] = (
        signal_df[f"{output_prefix}_CLOSE"]
        < signal_df[f"{output_prefix}_EMA200"]
    )

    return signal_df


# --------------------------------------------------
# RSI / RVI CROSS-BACK STATE
# --------------------------------------------------

def detect_crossback(series, level, direction):

    values = series.to_numpy(dtype=float)
    result = np.zeros(len(values), dtype=bool)

    armed = False

    for i in range(1, len(values)):

        previous = values[i - 1]
        current = values[i]

        if not np.isfinite(previous) or not np.isfinite(current):
            continue

        if direction == "BUY":

            if previous >= level and current < level:
                armed = True

            elif armed and previous <= level and current > level:
                result[i] = True
                armed = False

        elif direction == "SELL":

            if previous <= level and current > level:
                armed = True

            elif armed and previous >= level and current < level:
                result[i] = True
                armed = False

    return pd.Series(result, index=series.index)


# --------------------------------------------------
# STRATEGY CONDITIONS
# --------------------------------------------------

def evaluate_ema_strategy(df):

    bullish_regime = df["EMA20"] > df["EMA100"]
    bearish_regime = df["EMA20"] < df["EMA100"]

    bullish_confirmation = (
        df["BULLISH_DISPLACEMENT"].fillna(False)
        | df["BULLISH_FVG"].fillna(False)
        | df["BULLISH_IFVG"].fillna(False)
    )

    bearish_confirmation = (
        df["BEARISH_DISPLACEMENT"].fillna(False)
        | df["BEARISH_FVG"].fillna(False)
        | df["BEARISH_IFVG"].fillna(False)
    )

    buy = (
        bullish_regime
        & df["BULLISH_EMA100_RETEST"].fillna(False)
        & bullish_confirmation
        & df["H4_BULLISH"].fillna(False)
    )

    sell = (
        bearish_regime
        & df["BEARISH_EMA100_RETEST"].fillna(False)
        & bearish_confirmation
        & df["H4_BEARISH"].fillna(False)
    )

    return buy, sell


def evaluate_rsi_strategy(df):

    buy_crossback = detect_crossback(
        df["RSI14"],
        RSI_BUY_LEVEL,
        "BUY"
    )

    sell_crossback = detect_crossback(
        df["RSI14"],
        RSI_SELL_LEVEL,
        "SELL"
    )

    buy = (
        buy_crossback
        & df["SSL_SWEEP"].fillna(False)
        & df["H4_BULLISH"].fillna(False)
    )

    sell = (
        sell_crossback
        & df["BSL_SWEEP"].fillna(False)
        & df["H4_BEARISH"].fillna(False)
    )

    return buy, sell


def evaluate_rvi_strategy(df):

    buy_crossback = detect_crossback(
        df["RVI_SIGNAL"],
        RVI_BUY_LEVEL,
        "BUY"
    )

    sell_crossback = detect_crossback(
        df["RVI_SIGNAL"],
        RVI_SELL_LEVEL,
        "SELL"
    )

    buy = (
        buy_crossback
        & df["SSL_SWEEP"].fillna(False)
        & df["H2_BULLISH"].fillna(False)
    )

    sell = (
        sell_crossback
        & df["BSL_SWEEP"].fillna(False)
        & df["H2_BEARISH"].fillna(False)
    )

    return buy, sell


# --------------------------------------------------
# SIGNAL ID
# --------------------------------------------------

def create_signal_id(symbol, timeframe, strategy, direction, signal_time):

    raw = (
        f"{symbol}|{timeframe}|{strategy}|"
        f"{direction}|{pd.Timestamp(signal_time).isoformat()}"
    )

    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]

    return f"PK64-{digest.upper()}"


# --------------------------------------------------
# STOP LOSS AND TAKE PROFIT
# --------------------------------------------------

def calculate_trade_levels(
    direction,
    entry,
    atr,
    pivot,
    tp2_candidates=None
):

    if not all(np.isfinite(x) for x in [entry, atr, pivot]):
        return None

    if entry <= 0 or atr <= 0:
        return None

    buffer = atr * ATR_SL_BUFFER

    if direction == "BUY":
        stop_loss = pivot - buffer
        risk = entry - stop_loss

        if risk <= 0:
            return None

        tp1 = entry + (risk * MINIMUM_RR)

        valid_tp2 = [
            item for item in (tp2_candidates or [])
            if np.isfinite(item) and item > entry
        ]

        tp2 = min(valid_tp2) if valid_tp2 else None

    elif direction == "SELL":
        stop_loss = pivot + buffer
        risk = stop_loss - entry

        if risk <= 0:
            return None

        tp1 = entry - (risk * MINIMUM_RR)

        valid_tp2 = [
            item for item in (tp2_candidates or [])
            if np.isfinite(item) and item < entry
        ]

        tp2 = max(valid_tp2) if valid_tp2 else None

    else:
        return None

    if tp1 <= 0:
        return None

    return {
        "ENTRY": float(entry),
        "SL": float(stop_loss),
        "TP1": float(tp1),
        "TP2": float(tp2) if tp2 is not None else None,
        "RISK_PRICE": float(risk),
        "RR_TP1": MINIMUM_RR,
        "SL_ATR_BUFFER": ATR_SL_BUFFER
    }


# --------------------------------------------------
# TP2: CONFIRMED SWING LIQUIDITY
# --------------------------------------------------

def get_tp2_candidates(df, signal_position, direction):

    history = df.iloc[:signal_position + 1]

    if direction == "BUY":
        values = history["CONFIRMED_PIVOT_HIGH"].dropna()
        return values.astype(float).tolist()

    values = history["CONFIRMED_PIVOT_LOW"].dropna()
    return values.astype(float).tolist()


# --------------------------------------------------
# BUILD SIGNAL ROW
# --------------------------------------------------

def build_signal_row(
    symbol,
    timeframe,
    strategy,
    direction,
    df,
    position
):

    signal_time = df.index[position]

    next_position = position + 1

    row = {
        "SIGNAL_ID": create_signal_id(
            symbol,
            timeframe,
            strategy,
            direction,
            signal_time
        ),
        "SYMBOL": symbol,
        "TIMEFRAME": timeframe,
        "STRATEGY": strategy,
        "DIRECTION": direction,
        "SIGNAL_CANDLE_TIME": signal_time,
        "ENTRY_TIME": None,
        "ENTRY": None,
        "SL": None,
        "TP1": None,
        "TP2": None,
        "RISK_PRICE": None,
        "RR_TP1": None,
        "STATUS": "WAITING_NEXT_OPEN",
        "TP2_SOURCE": "CONFIRMED_SWING_LIQUIDITY",
        "CHOP_FILTER_STATUS": "PENDING_BACKTEST_CALIBRATION"
    }

    if next_position >= len(df):
        return row

    entry = float(df["Open"].iloc[next_position])
    atr = float(df["ATR14"].iloc[position])

    if direction == "BUY":
        pivot_series = (
            df["CONFIRMED_PIVOT_LOW"]
            .iloc[:position + 1]
            .dropna()
        )
    else:
        pivot_series = (
            df["CONFIRMED_PIVOT_HIGH"]
            .iloc[:position + 1]
            .dropna()
        )

    if pivot_series.empty:
        row["STATUS"] = "CANCELLED_NO_CONFIRMED_SL_PIVOT"
        return row

    pivot = float(pivot_series.iloc[-1])

    tp2_candidates = get_tp2_candidates(
        df,
        position,
        direction
    )

    levels = calculate_trade_levels(
        direction,
        entry,
        atr,
        pivot,
        tp2_candidates
    )

    if levels is None:
        row["STATUS"] = "CANCELLED_INVALID_RISK"
        return row

    row.update(levels)
    row["ENTRY_TIME"] = df.index[next_position]
    row["STATUS"] = "ENTRY_READY"

    return row


# --------------------------------------------------
# MAIN SIGNAL ENGINE
# --------------------------------------------------

def build_signals(data_by_timeframe, symbol):

    validate_timeframe_data(data_by_timeframe)

    prepared = {}

    for timeframe in SIGNAL_TIMEFRAMES:
        prepared[timeframe] = prepare_engine_data(
            data_by_timeframe[timeframe]
        )

    all_signals = []

    for timeframe in SIGNAL_TIMEFRAMES:

        df = prepared[timeframe].copy()

        # EMA and RSI use completed 4H EMA200.
        df = attach_closed_htf_filter(
            df,
            prepared["4h"],
            timeframe,
            "4h",
            "H4"
        )

        # RVI uses completed 2H EMA200.
        df = attach_closed_htf_filter(
            df,
            prepared["2h"],
            timeframe,
            "2h",
            "H2"
        )

        strategy_conditions = {}

        if timeframe in EMA_TIMEFRAMES:
            strategy_conditions["EMA"] = evaluate_ema_strategy(df)

        if timeframe in RSI_TIMEFRAMES:
            strategy_conditions["RSI"] = evaluate_rsi_strategy(df)

        if timeframe in RVI_TIMEFRAMES:
            strategy_conditions["RVI"] = evaluate_rvi_strategy(df)

        for strategy, conditions in strategy_conditions.items():

            buy_series, sell_series = conditions

            for position in np.flatnonzero(buy_series.to_numpy()):
                all_signals.append(
                    build_signal_row(
                        symbol,
                        timeframe,
                        strategy,
                        "BUY",
                        df,
                        int(position)
                    )
                )

            for position in np.flatnonzero(sell_series.to_numpy()):
                all_signals.append(
                    build_signal_row(
                        symbol,
                        timeframe,
                        strategy,
                        "SELL",
                        df,
                        int(position)
                    )
                )

    columns = [
        "SIGNAL_ID",
        "SYMBOL",
        "TIMEFRAME",
        "STRATEGY",
        "DIRECTION",
        "SIGNAL_CANDLE_TIME",
        "ENTRY_TIME",
        "ENTRY",
        "SL",
        "TP1",
        "TP2",
        "RISK_PRICE",
        "RR_TP1",
        "STATUS",
        "TP2_SOURCE",
        "CHOP_FILTER_STATUS"
    ]

    result = pd.DataFrame(all_signals, columns=columns)

    if not result.empty:
        result = result.sort_values(
            ["SIGNAL_CANDLE_TIME", "SYMBOL", "TIMEFRAME"]
        ).reset_index(drop=True)

    return result


# --------------------------------------------------
# INITIALIZATION
# --------------------------------------------------

def initialize():

    print("PK BOT V6.4 SIGNAL ENGINE")
    print("Version:", VERSION)
    print("Strategies: EMA / RSI / RVI")
    print("Signal timeframes: 15m / 30m / 1h / 2h / 4h")
    print("EMA HTF filter: 4H EMA200")
    print("RSI HTF filter: 4H EMA200")
    print("RVI HTF filter: 2H EMA200")
    print("TP1 minimum R:R: 1:4")
    print("SL buffer: ATR14 x 0.12")
    print("Entry: Next candle open")
    print("No 1D timeframe")
    print("No live orders")
    print("No Telegram messages")
    print("Signal Engine module ready.")


if __name__ == "__main__":
    initialize()
