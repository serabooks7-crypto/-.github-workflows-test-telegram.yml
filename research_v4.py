
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ==========================================================
# RLA V4 | XAUUSD A+ RESEARCH ENGINE
# Research and alert preparation only. No live execution.
# ==========================================================

VERSION = "RLA_V4"
SYMBOL = os.getenv("RLA_SYMBOL", "GC=F")
INTERVAL = "5m"
PERIOD = os.getenv("RLA_PERIOD", "60d")

MIN_RR = 4.0
ATR_PERIOD = 14
MAX_HOLD_BARS = 48
OUTPUT = Path("research_v4_output")
OUTPUT.mkdir(exist_ok=True)

ROUND_TURN_COST = float(
    os.getenv("RLA_ROUND_TURN_COST_PRICE", "0")
)
SLIPPAGE_PER_SIDE = float(
    os.getenv("RLA_SLIPPAGE_PRICE_PER_SIDE", "0")
)

NY_TZ = "America/New_York"

STRATEGIES = [
    "CRT_LIQUIDITY_SWEEP",
    "TBS_ENTRY",
    "MSS_FVG",
    "HTF_TREND_PULLBACK",
    "OB_DISPLACEMENT",
    "SESSION_JUDAS_AMD",
]


# ==========================================================
# 1. DATA ENGINE
# ==========================================================

def download_data():
    print("Downloading market data:", SYMBOL)

    data = yf.download(
        SYMBOL,
        period=PERIOD,
        interval=INTERVAL,
        auto_adjust=False,
        progress=False,
    )

    if data.empty:
        raise RuntimeError("Market data download failed.")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data[
        ["Open", "High", "Low", "Close", "Volume"]
    ].copy()

    data = data.dropna(
        subset=["Open", "High", "Low", "Close"]
    )

    if data.index.tz is None:
        data.index = data.index.tz_localize("UTC")

    return data.sort_index()


# ==========================================================
# 2. TIMEFRAME RESAMPLING
# ==========================================================

def resample_ohlc(data, timeframe):
    result = data.resample(
        timeframe,
        label="right",
        closed="right",
    ).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    })

    return result.dropna()


def add_htf_context(data):
    data = data.copy()

    h1 = resample_ohlc(data, "1h")
    h4 = resample_ohlc(data, "4h")

    for higher, prefix in [
        (h1, "H1"),
        (h4, "H4"),
    ]:
        higher[prefix + "_EMA20"] = (
            higher["Close"].ewm(span=20).mean()
        )
        higher[prefix + "_EMA50"] = (
            higher["Close"].ewm(span=50).mean()
        )
        higher[prefix + "_EMA200"] = (
            higher["Close"].ewm(span=200).mean()
        )

        higher[prefix + "_BULL"] = (
            (higher["Close"] > higher[prefix + "_EMA50"]) &
            (higher[prefix + "_EMA50"] >
             higher[prefix + "_EMA200"])
        )

        higher[prefix + "_BEAR"] = (
            (higher["Close"] < higher[prefix + "_EMA50"]) &
            (higher[prefix + "_EMA50"] <
             higher[prefix + "_EMA200"])
        )

    # Backward as-of join: only completed HTF candles are used.
    left = data.reset_index()
    time_column = left.columns[0]
    left = left.rename(columns={time_column: "Timestamp"})

    for higher, prefix in [
        (h1, "H1"),
        (h4, "H4"),
    ]:
        right = higher[
            [
                prefix + "_BULL",
                prefix + "_BEAR",
            ]
        ].reset_index()

        right = right.rename(
            columns={right.columns[0]: "Timestamp"}
        )

        left = pd.merge_asof(
            left.sort_values("Timestamp"),
            right.sort_values("Timestamp"),
            on="Timestamp",
            direction="backward",
        )

    left = left.set_index("Timestamp")
    return left


# ==========================================================
# 3. INDICATORS AND MARKET STRUCTURE
# ==========================================================

def add_market_features(data):
    df = data.copy()

    df["EMA20"] = df["Close"].ewm(span=20).mean()
    df["EMA50"] = df["Close"].ewm(span=50).mean()
    df["EMA200"] = df["Close"].ewm(span=200).mean()

    prev_close = df["Close"].shift(1)

    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["ATR"] = tr.rolling(ATR_PERIOD).mean()

    df["SwingHigh"] = (
        df["High"].rolling(12).max().shift(1)
    )
    df["SwingLow"] = (
        df["Low"].rolling(12).min().shift(1)
    )

    df["RangeHigh"] = (
        df["High"].rolling(20).max().shift(1)
    )
    df["RangeLow"] = (
        df["Low"].rolling(20).min().shift(1)
    )

    df["Body"] = (
        df["Close"] - df["Open"]
    ).abs()

    df["BullDisplacement"] = (
        (df["Close"] > df["Open"]) &
        (df["Body"] > df["ATR"] * 0.8)
    )

    df["BearDisplacement"] = (
        (df["Close"] < df["Open"]) &
        (df["Body"] > df["ATR"] * 0.8)
    )

    df["BullFVG"] = (
        df["Low"] > df["High"].shift(2)
    )

    df["BearFVG"] = (
        df["High"] < df["Low"].shift(2)
    )

    df["MSS_BULL"] = (
        df["Close"] > df["SwingHigh"]
    )

    df["MSS_BEAR"] = (
        df["Close"] < df["SwingLow"]
    )

    df["HTF_BULL"] = (
        df["H1_BULL"].fillna(False) &
        df["H4_BULL"].fillna(False)
    )

    df["HTF_BEAR"] = (
        df["H1_BEAR"].fillna(False) &
        df["H4_BEAR"].fillna(False)
    )

    return df


# ==========================================================
# 4. SESSION AND LIQUIDITY CONTEXT
# ==========================================================

def add_session_context(data):
    df = data.copy()
    local = df.index.tz_convert(NY_TZ)

    df["NY_DATE"] = local.date
    df["NY_HOUR"] = local.hour
    df["NY_MINUTE"] = local.hour * 60 + local.minute

    # New York morning research window
    df["NY_MORNING"] = (
        (df["NY_HOUR"] >= 7) &
        (df["NY_HOUR"] < 11)
    )

    # Prior day high/low using completed previous NY day.
    daily = df.groupby("NY_DATE").agg(
        PDH=("High", "max"),
        PDL=("Low", "min"),
    )

    daily["PDH"] = daily["PDH"].shift(1)
    daily["PDL"] = daily["PDL"].shift(1)

    df["PDH"] = df["NY_DATE"].map(daily["PDH"])
    df["PDL"] = df["NY_DATE"].map(daily["PDL"])

    df["SWEEP_LOW"] = (
        (df["Low"] < df["RangeLow"]) &
        (df["Close"] > df["RangeLow"])
    )

    df["SWEEP_HIGH"] = (
        (df["High"] > df["RangeHigh"]) &
        (df["Close"] < df["RangeHigh"])
    )

    df["PDL_SWEEP"] = (
        (df["Low"] < df["PDL"]) &
        (df["Close"] > df["PDL"])
    )

    df["PDH_SWEEP"] = (
        (df["High"] > df["PDH"]) &
        (df["Close"] < df["PDH"])
    )

    return df


# ==========================================================
# 5. SIX STRATEGY SIGNAL MODELS
# ==========================================================

def generate_strategy_signals(data):
    df = data.copy()

    for strategy in STRATEGIES:
        df[strategy + "_BUY"] = False
        df[strategy + "_SELL"] = False

    # Strategy 1: CRT + Liquidity Sweep
    df["CRT_LIQUIDITY_SWEEP_BUY"] = (
        df["SWEEP_LOW"] &
        df["BullDisplacement"] &
        df["HTF_BULL"]
    )

    df["CRT_LIQUIDITY_SWEEP_SELL"] = (
        df["SWEEP_HIGH"] &
        df["BearDisplacement"] &
        df["HTF_BEAR"]
    )

    # Strategy 2: TBS sweep then confirmation
    df["TBS_ENTRY_BUY"] = (
        df["SWEEP_LOW"].shift(1).fillna(False) &
        df["MSS_BULL"] &
        df["BullDisplacement"] &
        df["HTF_BULL"]
    )

    df["TBS_ENTRY_SELL"] = (
        df["SWEEP_HIGH"].shift(1).fillna(False) &
        df["MSS_BEAR"] &
        df["BearDisplacement"] &
        df["HTF_BEAR"]
    )

    # Strategy 3: MSS + FVG
    df["MSS_FVG_BUY"] = (
        df["MSS_BULL"] &
        df["BullFVG"] &
        df["BullDisplacement"] &
        df["HTF_BULL"]
    )

    df["MSS_FVG_SELL"] = (
        df["MSS_BEAR"] &
        df["BearFVG"] &
        df["BearDisplacement"] &
        df["HTF_BEAR"]
    )

    # Strategy 4: HTF Trend Pullback
    df["HTF_TREND_PULLBACK_BUY"] = (
        df["HTF_BULL"] &
        (df["Low"] <= df["EMA20"]) &
        (df["Close"] > df["EMA20"]) &
        (df["Close"] > df["Open"])
    )

    df["HTF_TREND_PULLBACK_SELL"] = (
        df["HTF_BEAR"] &
        (df["High"] >= df["EMA20"]) &
        (df["Close"] < df["EMA20"]) &
        (df["Close"] < df["Open"])
    )

    # Strategy 5: Order Block + Displacement proxy
    previous_bear = (
        df["Close"].shift(1) < df["Open"].shift(1)
    )
    previous_bull = (
        df["Close"].shift(1) > df["Open"].shift(1)
    )

    df["OB_DISPLACEMENT_BUY"] = (
        previous_bear &
        df["BullDisplacement"] &
        (df["Close"] > df["High"].shift(1)) &
        df["HTF_BULL"]
    )

    df["OB_DISPLACEMENT_SELL"] = (
        previous_bull &
        df["BearDisplacement"] &
        (df["Close"] < df["Low"].shift(1)) &
        df["HTF_BEAR"]
    )

    # Strategy 6: Session Judas / AMD proxy
    df["SESSION_JUDAS_AMD_BUY"] = (
        df["NY_MORNING"] &
        (df["PDL_SWEEP"] | df["SWEEP_LOW"]) &
        df["BullDisplacement"] &
        df["HTF_BULL"]
    )

    df["SESSION_JUDAS_AMD_SELL"] = (
        df["NY_MORNING"] &
        (df["PDH_SWEEP"] | df["SWEEP_HIGH"]) &
        df["BearDisplacement"] &
        df["HTF_BEAR"]
    )

    return df

# ==========================================================
# 6. A+ SETUP QUALITY FILTER
# ==========================================================

def is_a_plus_setup(row, strategy, direction):
    """
    A+ means a clean, context-aligned setup.
    This is a rule-based research filter, not a guaranteed
    prediction or a claim of 80% win rate.
    """

    if direction == "BUY":
        if not bool(row.get("HTF_BULL", False)):
            return False
        if not bool(row.get("BullDisplacement", False)):
            return False
    else:
        if not bool(row.get("HTF_BEAR", False)):
            return False
        if not bool(row.get("BearDisplacement", False)):
            return False

    # Strategy-specific confirmation
    if strategy == "CRT_LIQUIDITY_SWEEP":
        return bool(
            row["SWEEP_LOW"] if direction == "BUY"
            else row["SWEEP_HIGH"]
        )

    if strategy == "TBS_ENTRY":
        return bool(
            row["MSS_BULL"] if direction == "BUY"
            else row["MSS_BEAR"]
        )

    if strategy == "MSS_FVG":
        return bool(
            row["BullFVG"] if direction == "BUY"
            else row["BearFVG"]
        )

    if strategy == "HTF_TREND_PULLBACK":
        return bool(
            row["Close"] > row["EMA20"] if direction == "BUY"
            else row["Close"] < row["EMA20"]
        )

    if strategy == "OB_DISPLACEMENT":
        return True

    if strategy == "SESSION_JUDAS_AMD":
        return bool(row["NY_MORNING"])

    return False


# ==========================================================
# 7. ENTRY, STOP LOSS AND TAKE PROFIT
# ==========================================================

def build_trade_levels(data, signal_index, direction):
    """
    Signal is confirmed at candle close.
    Entry is the NEXT candle open.
    """

    if signal_index + 1 >= len(data):
        return None

    signal = data.iloc[signal_index]
    next_bar = data.iloc[signal_index + 1]

    atr = signal["ATR"]

    if pd.isna(atr) or atr <= 0:
        return None

    entry = float(next_bar["Open"])

    if direction == "BUY":
        swing_stop = float(
            data["Low"].iloc[
                max(0, signal_index - 5):signal_index + 1
            ].min()
        )

        stop = swing_stop - float(atr) * 0.15
        risk = entry - stop

        if risk <= 0:
            return None

        target = entry + risk * MIN_RR

    else:
        swing_stop = float(
            data["High"].iloc[
                max(0, signal_index - 5):signal_index + 1
            ].max()
        )

        stop = swing_stop + float(atr) * 0.15
        risk = stop - entry

        if risk <= 0:
            return None

        target = entry - risk * MIN_RR

    if risk < float(atr) * 0.10:
        return None

    return {
        "entry": entry,
        "stop": stop,
        "target": target,
        "risk": risk,
        "planned_rr": MIN_RR,
    }


# ==========================================================
# 8. TRADE SIMULATION
# ==========================================================

def simulate_trade(data, signal_index, strategy, direction):
    levels = build_trade_levels(
        data, signal_index, direction
    )

    if levels is None:
        return None

    entry_index = signal_index + 1
    last_index = min(
        entry_index + MAX_HOLD_BARS - 1,
        len(data) - 1
    )

    entry = levels["entry"]
    stop = levels["stop"]
    target = levels["target"]
    risk = levels["risk"]

    exit_price = None
    exit_reason = "TIMEOUT"
    exit_index = last_index

    for j in range(entry_index, last_index + 1):
        bar = data.iloc[j]

        if direction == "BUY":
            # Gap-aware stop
            if float(bar["Open"]) <= stop:
                exit_price = float(bar["Open"])
                exit_reason = "SL_GAP"
                exit_index = j
                break

            # Gap-aware target
            if float(bar["Open"]) >= target:
                exit_price = target
                exit_reason = "TP_GAP"
                exit_index = j
                break

            hit_sl = float(bar["Low"]) <= stop
            hit_tp = float(bar["High"]) >= target

        else:
            if float(bar["Open"]) >= stop:
                exit_price = float(bar["Open"])
                exit_reason = "SL_GAP"
                exit_index = j
                break

            if float(bar["Open"]) <= target:
                exit_price = target
                exit_reason = "TP_GAP"
                exit_index = j
                break

            hit_sl = float(bar["High"]) >= stop
            hit_tp = float(bar["Low"]) <= target

        # Conservative assumption:
        # SL is first when both levels are touched in one bar.
        if hit_sl:
            exit_price = stop
            exit_reason = "SL"
            exit_index = j
            break

        if hit_tp:
            exit_price = target
            exit_reason = "TP"
            exit_index = j
            break

    if exit_price is None:
        exit_price = float(data["Close"].iloc[exit_index])

    if direction == "BUY":
        gross_r = (exit_price - entry) / risk
    else:
        gross_r = (entry - exit_price) / risk

    total_cost = (
        ROUND_TURN_COST + 2 * SLIPPAGE_PER_SIDE
    )

    net_r = gross_r - total_cost / risk

    signal_time = data.index[signal_index]
    entry_time = data.index[entry_index]
    exit_time = data.index[exit_index]

    return {
        "strategy": strategy,
        "direction": direction,
        "signal_time": signal_time,
        "entry_time": entry_time,
        "exit_time": exit_time,
        "entry": entry,
        "stop": stop,
        "target": target,
        "exit_price": exit_price,
        "planned_rr": MIN_RR,
        "exit_reason": exit_reason,
        "gross_r": gross_r,
        "net_r": net_r,
        "win": net_r > 0,
        "tp_hit": exit_reason in ("TP", "TP_GAP"),
    }


# ==========================================================
# 9. INDIVIDUAL STRATEGY BACKTEST
# ==========================================================

def backtest_strategy(data, strategy, direction):
    signal_column = strategy + "_" + direction
    trades = []

    start_index = 210
    end_index = len(data) - MAX_HOLD_BARS - 1

    for i in range(start_index, end_index):
        if not bool(data[signal_column].iloc[i]):
            continue

        row = data.iloc[i]

        if not is_a_plus_setup(row, strategy, direction):
            continue

        trade = simulate_trade(
            data, i, strategy, direction
        )

        if trade is not None:
            trades.append(trade)

    return pd.DataFrame(trades)


def run_strategy_backtests(data):
    results = []

    for strategy in STRATEGIES:
        for direction in ("BUY", "SELL"):
            trades = backtest_strategy(
                data, strategy, direction
            )

            if not trades.empty:
                results.append(trades)

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)


# ==========================================================
# 10. COMBINED PORTFOLIO
# ==========================================================

def build_combined_portfolio(candidates):
    if candidates.empty:
        return candidates.copy()

    df = candidates.copy()

    df["entry_time"] = pd.to_datetime(
        df["entry_time"], utc=True
    )
    df["exit_time"] = pd.to_datetime(
        df["exit_time"], utc=True
    )

    # Priority only resolves simultaneous competing signals.
    priority = {
        "CRT_LIQUIDITY_SWEEP": 1,
        "TBS_ENTRY": 2,
        "MSS_FVG": 3,
        "OB_DISPLACEMENT": 4,
        "SESSION_JUDAS_AMD": 5,
        "HTF_TREND_PULLBACK": 6,
    }

    df["priority"] = df["strategy"].map(priority)

    df = df.sort_values(
        ["entry_time", "priority"]
    ).reset_index(drop=True)

    selected = []
    last_exit = None
    recent = []

    for _, trade in df.iterrows():
        entry_time = trade["entry_time"]

        # Remove signals older than 15 minutes.
        recent = [
            item for item in recent
            if (entry_time - item[0]).total_seconds()
            <= 900
        ]

        duplicate = any(
            item[1] == trade["direction"]
            for item in recent
        )

        if duplicate:
            continue

        # Avoid overlapping portfolio positions.
        if last_exit is not None and entry_time < last_exit:
            continue

        selected.append(trade.to_dict())

        recent.append(
            (entry_time, trade["direction"])
        )

        last_exit = trade["exit_time"]

    result = pd.DataFrame(selected)

    if "priority" in result.columns:
        result = result.drop(columns=["priority"])

    return result


# ==========================================================
# 11. PERFORMANCE METRICS
# ==========================================================

def calculate_metrics(trades, period_name):
    if trades.empty:
        return {
            "period": period_name,
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate_pct": 0.0,
            "tp_hit_rate_pct": 0.0,
            "profit_factor": 0.0,
            "average_net_r": 0.0,
            "total_net_r": 0.0,
            "max_drawdown_r": 0.0,
        }

    net = trades["net_r"].astype(float)

    wins = int((net > 0).sum())
    losses = int((net <= 0).sum())

    gross_profit = float(net[net > 0].sum())
    gross_loss = abs(float(net[net < 0].sum()))

    if gross_loss == 0:
        pf = float("inf") if gross_profit > 0 else 0.0
    else:
        pf = gross_profit / gross_loss

    equity = net.cumsum()
    drawdown = equity - equity.cummax()

    return {
        "period": period_name,
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate_pct": round(
            wins / len(trades) * 100, 2
        ),
        "tp_hit_rate_pct": round(
            trades["tp_hit"].mean() * 100, 2
        ),
        "profit_factor": round(pf, 3),
        "average_net_r": round(float(net.mean()), 3),
        "total_net_r": round(float(net.sum()), 3),
        "max_drawdown_r": round(
            abs(float(drawdown.min())), 3
        ),
    }


def make_strategy_report(trades, period_name):
    rows = []

    if trades.empty:
        return pd.DataFrame()

    for strategy, group in trades.groupby("strategy"):
        row = calculate_metrics(group, period_name)
        row["strategy"] = strategy
        rows.append(row)

    return pd.DataFrame(rows)


# ==========================================================
# 12. MAIN RESEARCH PIPELINE
# ==========================================================

def main():
    print("=" * 60)
    print(VERSION, "START")
    print("=" * 60)

    data = download_data()
    data = add_htf_context(data)
    data = add_market_features(data)
    data = add_session_context(data)
    data = generate_strategy_signals(data)

    data = data.dropna(
        subset=["ATR", "H1_BULL", "H4_BULL"]
    )

    if len(data) < 500:
        raise RuntimeError(
            "Insufficient data after feature preparation."
        )

    # Chronological 70/30 split.
    split = int(len(data) * 0.70)

    in_sample = data.iloc[:split].copy()
    out_sample = data.iloc[split:].copy()

    print("Backtesting In-Sample...")
    candidates_in = run_strategy_backtests(in_sample)

    print("Backtesting Out-of-Sample...")
    candidates_out = run_strategy_backtests(out_sample)

    portfolio_in = build_combined_portfolio(
        candidates_in
    )

    portfolio_out = build_combined_portfolio(
        candidates_out
    )

    portfolio_in.to_csv(
        OUTPUT / "portfolio_in_sample_v4.csv",
        index=False
    )

    portfolio_out.to_csv(
        OUTPUT / "portfolio_out_of_sample_v4.csv",
        index=False
    )

    metrics_in = calculate_metrics(
        portfolio_in, "IN_SAMPLE"
    )

    metrics_out = calculate_metrics(
        portfolio_out, "OUT_OF_SAMPLE"
    )

    pd.DataFrame(
        [metrics_in, metrics_out]
    ).to_csv(
        OUTPUT / "portfolio_performance_v4.csv",
        index=False
    )

    make_strategy_report(
        portfolio_in, "IN_SAMPLE"
    ).to_csv(
        OUTPUT / "strategy_in_sample_v4.csv",
        index=False
    )

    make_strategy_report(
        portfolio_out, "OUT_OF_SAMPLE"
    ).to_csv(
        OUTPUT / "strategy_out_of_sample_v4.csv",
        index=False
    )

    print("\nIN-SAMPLE")
    print(metrics_in)

    print("\nOUT-OF-SAMPLE")
    print(metrics_out)

    print("\nReports saved in:", OUTPUT.resolve())
    print("Research completed. No live trades placed.")


if __name__ == "__main__":
    main()
