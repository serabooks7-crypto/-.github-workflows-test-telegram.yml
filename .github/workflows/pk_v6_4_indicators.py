
"""
PK BOT V6.4
Indicator Calculation Engine

Timeframes:
15M, 30M, 1H, 2H, 4H

1D is excluded.
All calculations use closed candles.
"""

import numpy as np
import pandas as pd


VERSION = "6.4"


# --------------------------------------------------
# DATA VALIDATION
# --------------------------------------------------

def prepare_ohlc(df):

    required = ["Open", "High", "Low", "Close"]

    if df is None or df.empty:
        raise ValueError("No market data")

    df = df.copy()

    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna(subset=required)

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    invalid = (
        (df["High"] < df["Low"]) |
        (df["High"] < df["Open"]) |
        (df["High"] < df["Close"]) |
        (df["Low"] > df["Open"]) |
        (df["Low"] > df["Close"]) |
        (df[required] <= 0).any(axis=1)
    )

    df = df.loc[~invalid].copy()

    if len(df) < 250:
        raise ValueError("Insufficient candles")

    return df


# --------------------------------------------------
# EMA
# --------------------------------------------------

def calculate_ema(df):

    close = df["Close"]

    df["EMA20"] = close.ewm(
        span=20,
        adjust=False,
        min_periods=20
    ).mean()

    df["EMA100"] = close.ewm(
        span=100,
        adjust=False,
        min_periods=100
    ).mean()

    df["EMA200"] = close.ewm(
        span=200,
        adjust=False,
        min_periods=200
    ).mean()

    return df


# --------------------------------------------------
# RSI 14 - WILDER
# --------------------------------------------------

def calculate_rsi(df, period=14):

    close = df["Close"]

    change = close.diff()

    gain = change.clip(lower=0)
    loss = -change.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    rs = avg_gain / avg_loss

    rsi = 100 - (100 / (1 + rs))

    rsi = rsi.mask(
        (avg_loss == 0) & (avg_gain > 0),
        100
    )

    rsi = rsi.mask(
        (avg_gain == 0) & (avg_loss > 0),
        0
    )

    rsi = rsi.mask(
        (avg_gain == 0) & (avg_loss == 0),
        50
    )

    df["RSI14"] = rsi

    return df


# --------------------------------------------------
# RVI 8 - STANDARD FORMULA
# --------------------------------------------------

def weighted_average(series, period=4):

    weights = np.array([1, 2, 2, 1], dtype=float)

    return series.rolling(
        period,
        min_periods=period
    ).apply(
        lambda x: np.dot(x, weights) / weights.sum(),
        raw=True
    )


def calculate_rvi(df):

    numerator = df["Close"] - df["Open"]
    denominator = df["High"] - df["Low"]

    denominator = denominator.replace(0, np.nan)

    weighted_num = weighted_average(numerator)
    weighted_den = weighted_average(denominator)

    df["RVI"] = weighted_num / weighted_den

    df["RVI_SIGNAL"] = weighted_average(df["RVI"])

    return df


# --------------------------------------------------
# ATR 14 - WILDER
# --------------------------------------------------

def calculate_atr(df, period=14):

    previous_close = df["Close"].shift(1)

    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - previous_close).abs(),
            (df["Low"] - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    df["ATR14"] = tr.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    return df


# --------------------------------------------------
# ADX 14 - WILDER
# --------------------------------------------------

def calculate_adx(df, period=14):

    up_move = df["High"].diff()

    down_move = -df["Low"].diff()

    plus_dm = pd.Series(
        np.where(
            (up_move > down_move) & (up_move > 0),
            up_move,
            0.0
        ),
        index=df.index
    )

    minus_dm = pd.Series(
        np.where(
            (down_move > up_move) & (down_move > 0),
            down_move,
            0.0
        ),
        index=df.index
    )

    atr = df["ATR14"]

    plus_smoothed = plus_dm.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    minus_smoothed = minus_dm.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    plus_di = 100 * plus_smoothed / atr
    minus_di = 100 * minus_smoothed / atr

    denominator = plus_di + minus_di

    dx = (
        100 * (plus_di - minus_di).abs()
        / denominator.replace(0, np.nan)
    )

    df["PLUS_DI"] = plus_di
    df["MINUS_DI"] = minus_di

    df["ADX14"] = dx.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    return df


# --------------------------------------------------
# FIVE-CANDLE CONFIRMED PIVOT
# --------------------------------------------------

def calculate_pivots(df):

    high = df["High"]
    low = df["Low"]

    pivot_high = pd.Series(
        np.nan,
        index=df.index,
        dtype=float
    )

    pivot_low = pd.Series(
        np.nan,
        index=df.index,
        dtype=float
    )

    # Center candle becomes confirmed
    # only after two right-side candles close.

    for i in range(4, len(df)):

        center = i - 2

        left_highs = high.iloc[i-4:center]
        right_highs = high.iloc[center+1:i+1]

        left_lows = low.iloc[i-4:center]
        right_lows = low.iloc[center+1:i+1]

        center_high = high.iloc[center]
        center_low = low.iloc[center]

        if (
            center_high > left_highs.max()
            and center_high > right_highs.max()
        ):
            pivot_high.iloc[i] = center_high

        if (
            center_low < left_lows.min()
            and center_low < right_lows.min()
        ):
            pivot_low.iloc[i] = center_low

    df["CONFIRMED_PIVOT_HIGH"] = pivot_high
    df["CONFIRMED_PIVOT_LOW"] = pivot_low

    return df


# --------------------------------------------------
# COMPLETE INDICATOR ENGINE
# --------------------------------------------------

def calculate_indicators(data):

    df = prepare_ohlc(data)

    df = calculate_ema(df)
    df = calculate_rsi(df, 14)
    df = calculate_rvi(df)
    df = calculate_atr(df, 14)
    df = calculate_adx(df, 14)
    df = calculate_pivots(df)

    return df


# --------------------------------------------------
# QUICK TEST
# --------------------------------------------------

def initialize():

    print("PK BOT V6.4 INDICATOR ENGINE")
    print("Version:", VERSION)

    print("EMA: 20 / 100 / 200")
    print("RSI: 14")
    print("RVI: 8 + Signal Line")
    print("ATR: 14")
    print("ADX: 14")
    print("Pivot: 5 candles")
    print("Closed candles only")
    print("1D: EXCLUDED")

    print("Indicator module ready.")


if __name__ == "__main__":
    initialize()
