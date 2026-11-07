
"""
PK BOT V6.4
Liquidity & Price Action Detection Engine

Features:
1. Confirmed 5-candle swing pivots
2. Liquidity sweep
3. Fair Value Gap (FVG)
4. Inverse Fair Value Gap (IFVG)
5. Displacement
6. EMA100 retest

Input must contain CLOSED candles only.
1D timeframe is excluded.
"""

import numpy as np
import pandas as pd


VERSION = "6.4"


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

MIN_FVG_ATR = 0.05
DISPLACEMENT_BODY_RATIO = 0.60
DISPLACEMENT_ATR_MULTIPLE = 1.00
EMA_RETEST_ATR = 0.15

ATR_PERIOD = 14
PIVOT_LOOKBACK = 5


# --------------------------------------------------
# DATA PREPARATION
# --------------------------------------------------

def prepare_data(data):

    required = ["Open", "High", "Low", "Close"]

    if data is None or data.empty:
        raise ValueError("No OHLC data")

    df = data.copy()

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

    if len(df) < 10:
        raise ValueError("Insufficient candles")

    return df


# --------------------------------------------------
# ATR 14
# --------------------------------------------------

def calculate_atr(df):

    previous_close = df["Close"].shift(1)

    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - previous_close).abs(),
            (df["Low"] - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    return tr.ewm(
        alpha=1 / ATR_PERIOD,
        adjust=False,
        min_periods=ATR_PERIOD
    ).mean()


# --------------------------------------------------
# EMA 100
# --------------------------------------------------

def calculate_ema100(df):

    return df["Close"].ewm(
        span=100,
        adjust=False,
        min_periods=100
    ).mean()


# --------------------------------------------------
# CONFIRMED FIVE-CANDLE PIVOTS
# --------------------------------------------------

def calculate_confirmed_pivots(df):

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

    for i in range(4, len(df)):

        center = i - 2

        left_high = high.iloc[i-4:center]
        right_high = high.iloc[center+1:i+1]

        left_low = low.iloc[i-4:center]
        right_low = low.iloc[center+1:i+1]

        center_high = high.iloc[center]
        center_low = low.iloc[center]

        if (
            center_high > left_high.max()
            and center_high > right_high.max()
        ):
            pivot_high.iloc[i] = center_high

        if (
            center_low < left_low.min()
            and center_low < right_low.min()
        ):
            pivot_low.iloc[i] = center_low

    df["CONFIRMED_PIVOT_HIGH"] = pivot_high
    df["CONFIRMED_PIVOT_LOW"] = pivot_low

    # Only pivots confirmed before the current candle.
    df["LAST_CONFIRMED_HIGH"] = pivot_high.ffill().shift(1)
    df["LAST_CONFIRMED_LOW"] = pivot_low.ffill().shift(1)

    return df


# --------------------------------------------------
# LIQUIDITY SWEEP
# --------------------------------------------------

def detect_liquidity_sweep(df):

    previous_high = df["LAST_CONFIRMED_HIGH"]
    previous_low = df["LAST_CONFIRMED_LOW"]

    # Buy-side liquidity taken, then close below level.
    df["BSL_SWEEP"] = (
        (df["High"] > previous_high) &
        (df["Close"] < previous_high)
    ).fillna(False)

    # Sell-side liquidity taken, then close above level.
    df["SSL_SWEEP"] = (
        (df["Low"] < previous_low) &
        (df["Close"] > previous_low)
    ).fillna(False)

    # Bullish reversal candidate after SSL sweep.
    df["BULLISH_LIQUIDITY_SWEEP"] = df["SSL_SWEEP"]

    # Bearish reversal candidate after BSL sweep.
    df["BEARISH_LIQUIDITY_SWEEP"] = df["BSL_SWEEP"]

    return df


# --------------------------------------------------
# FAIR VALUE GAP
# --------------------------------------------------

def detect_fvg(df, min_fvg_atr=MIN_FVG_ATR):

    high_two_bars_ago = df["High"].shift(2)
    low_two_bars_ago = df["Low"].shift(2)

    bullish_gap = df["Low"] - high_two_bars_ago
    bearish_gap = low_two_bars_ago - df["High"]

    minimum_size = df["ATR14"] * min_fvg_atr

    df["BULLISH_FVG"] = (
        (bullish_gap > 0) &
        (bullish_gap >= minimum_size)
    ).fillna(False)

    df["BEARISH_FVG"] = (
        (bearish_gap > 0) &
        (bearish_gap >= minimum_size)
    ).fillna(False)

    df["BULLISH_FVG_LOW"] = np.where(
        df["BULLISH_FVG"],
        high_two_bars_ago,
        np.nan
    )

    df["BULLISH_FVG_HIGH"] = np.where(
        df["BULLISH_FVG"],
        df["Low"],
        np.nan
    )

    df["BEARISH_FVG_LOW"] = np.where(
        df["BEARISH_FVG"],
        df["High"],
        np.nan
    )

    df["BEARISH_FVG_HIGH"] = np.where(
        df["BEARISH_FVG"],
        low_two_bars_ago,
        np.nan
    )

    df["FVG_SIZE"] = np.where(
        df["BULLISH_FVG"],
        bullish_gap,
        np.where(
            df["BEARISH_FVG"],
            bearish_gap,
            np.nan
        )
    )

    return df


# --------------------------------------------------
# INVERSE FAIR VALUE GAP
# --------------------------------------------------

def detect_ifvg(df):

    bullish_ifvg = pd.Series(
        False,
        index=df.index
    )

    bearish_ifvg = pd.Series(
        False,
        index=df.index
    )

    # Each zone is tracked independently.
    zones = []

    for i in range(len(df)):

        current_high = df["High"].iloc[i]
        current_low = df["Low"].iloc[i]
        current_close = df["Close"].iloc[i]

        # Process previously created FVG zones.
        for zone in zones:

            if zone["used"]:
                continue

            if zone["type"] == "bullish":

                # Bullish FVG invalidated by a close
                # below its lower boundary.
                if not zone["invalidated"]:

                    if current_close < zone["lower"]:
                        zone["invalidated"] = True

                # Retest from below creates bearish IFVG.
                elif (
                    current_high >= zone["lower"]
                    and current_close < zone["lower"]
                ):
                    bearish_ifvg.iloc[i] = True
                    zone["used"] = True

            elif zone["type"] == "bearish":

                # Bearish FVG invalidated by a close
                # above its upper boundary.
                if not zone["invalidated"]:

                    if current_close > zone["upper"]:
                        zone["invalidated"] = True

                # Retest from above creates bullish IFVG.
                elif (
                    current_low <= zone["upper"]
                    and current_close > zone["upper"]
                ):
                    bullish_ifvg.iloc[i] = True
                    zone["used"] = True

        # Add newly confirmed FVG zones after processing
        # old zones, preventing same-candle look-ahead.
        if df["BULLISH_FVG"].iloc[i]:

            zones.append({
                "type": "bullish",
                "lower": df["BULLISH_FVG_LOW"].iloc[i],
                "upper": df["BULLISH_FVG_HIGH"].iloc[i],
                "invalidated": False,
                "used": False
            })

        if df["BEARISH_FVG"].iloc[i]:

            zones.append({
                "type": "bearish",
                "lower": df["BEARISH_FVG_LOW"].iloc[i],
                "upper": df["BEARISH_FVG_HIGH"].iloc[i],
                "invalidated": False,
                "used": False
            })

    df["BULLISH_IFVG"] = bullish_ifvg
    df["BEARISH_IFVG"] = bearish_ifvg

    return df


# --------------------------------------------------
# DISPLACEMENT
# --------------------------------------------------

def detect_displacement(
    df,
    body_ratio=DISPLACEMENT_BODY_RATIO,
    atr_multiple=DISPLACEMENT_ATR_MULTIPLE
):

    candle_range = df["High"] - df["Low"]
    candle_body = (df["Close"] - df["Open"]).abs()

    body_ratio_actual = (
        candle_body / candle_range.replace(0, np.nan)
    )

    strong_candle = (
        (body_ratio_actual >= body_ratio) &
        (candle_range >= df["ATR14"] * atr_multiple)
    )

    df["BODY_RATIO"] = body_ratio_actual

    df["BULLISH_DISPLACEMENT"] = (
        strong_candle &
        (df["Close"] > df["Open"])
    ).fillna(False)

    df["BEARISH_DISPLACEMENT"] = (
        strong_candle &
        (df["Close"] < df["Open"])
    ).fillna(False)

    return df


# --------------------------------------------------
# EMA100 RETEST
# --------------------------------------------------

def detect_ema100_retest(
    df,
    tolerance=EMA_RETEST_ATR
):

    ema = df["EMA100"]
    atr = df["ATR14"]

    tolerance_value = atr * tolerance

    # Candle range must reach the EMA100 tolerance zone.
    touches_zone = (
        (df["Low"] <= ema + tolerance_value) &
        (df["High"] >= ema - tolerance_value)
    )

    df["BULLISH_EMA100_RETEST"] = (
        touches_zone &
        (df["Close"] > ema) &
        (df["Close"] > df["Open"])
    ).fillna(False)

    df["BEARISH_EMA100_RETEST"] = (
        touches_zone &
        (df["Close"] < ema) &
        (df["Close"] < df["Open"])
    ).fillna(False)

    return df


# --------------------------------------------------
# COMPLETE PRICE ACTION ENGINE
# --------------------------------------------------

def calculate_price_action(
    data,
    min_fvg_atr=MIN_FVG_ATR,
    displacement_body_ratio=DISPLACEMENT_BODY_RATIO,
    displacement_atr_multiple=DISPLACEMENT_ATR_MULTIPLE,
    ema_retest_tolerance=EMA_RETEST_ATR
):

    df = prepare_data(data)

    df["ATR14"] = calculate_atr(df)
    df["EMA100"] = calculate_ema100(df)

    df = calculate_confirmed_pivots(df)
    df = detect_liquidity_sweep(df)

    df = detect_fvg(
        df,
        min_fvg_atr=min_fvg_atr
    )

    df = detect_ifvg(df)

    df = detect_displacement(
        df,
        body_ratio=displacement_body_ratio,
        atr_multiple=displacement_atr_multiple
    )

    df = detect_ema100_retest(
        df,
        tolerance=ema_retest_tolerance
    )

    return df


# --------------------------------------------------
# INITIALIZATION
# --------------------------------------------------

def initialize():

    print("PK BOT V6.4 PRICE ACTION ENGINE")
    print("Version:", VERSION)
    print("Liquidity Sweep: Confirmed pivot")
    print("FVG: Three-candle imbalance")
    print("IFVG: Invalidation + retest")
    print("Displacement: Body + ATR")
    print("EMA Retest: EMA100 + ATR tolerance")
    print("Pivot: Five candles")
    print("Closed candles only")
    print("1D: EXCLUDED")
    print("Price Action module ready.")


if __name__ == "__main__":
    initialize()
