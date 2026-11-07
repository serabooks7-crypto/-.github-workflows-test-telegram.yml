
"""
PK BOT V6.4 - HISTORICAL BACKTEST
Research only. No live orders. No Telegram messages.

Uses the existing V6.4 Signal Engine.
Yahoo Finance data: 15m, 30m, 1h.
2h and 4h are resampled from 1h.
Backtest period: latest 60 days (Yahoo intraday limit).
TP2 is informational and never affects trade outcome.
"""

from pathlib import Path
import sys
import time
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from pk_v6_4_signal_engine import build_signals

warnings.filterwarnings("ignore")

VERSION = "6.4"
PERIOD = "60d"

SYMBOLS = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "JPY=X",
    "USDCHF": "CHF=X",
    "USDCAD": "CAD=X",
    "AUDUSD": "AUDUSD=X",
    "NZDUSD": "NZDUSD=X",
    "EURJPY": "EURJPY=X",
    "GBPJPY": "GBPJPY=X",
    "EURGBP": "EURGBP=X",
    "EURCHF": "EURCHF=X",
    "AUDJPY": "AUDJPY=X",
    "XAUUSD": "GC=F",
    "AUDCAD": "AUDCAD=X",
}

INTERVALS = ["15m", "30m", "1h"]
TIMEFRAMES = ["15m", "30m", "1h", "2h", "4h"]

TRADE_COLUMNS = [
    "SIGNAL_ID", "SYMBOL", "TIMEFRAME", "STRATEGY",
    "DIRECTION", "SIGNAL_CANDLE_TIME", "ENTRY_TIME",
    "ENTRY", "SL", "TP1", "TP2", "OUTCOME",
    "EXIT_TIME", "EXIT_PRICE", "R_MULTIPLE",
    "BARS_TO_EXIT"
]


def download_market_data():
    """Download all symbols in three Yahoo requests."""
    downloaded = {}

    for interval in INTERVALS:
        print(f"Downloading {interval} data...")

        raw = None
        for attempt in range(3):
            try:
                raw = yf.download(
                    tickers=list(SYMBOLS.values()),
                    period=PERIOD,
                    interval=interval,
                    group_by="ticker",
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                    timeout=30
                )
                if raw is not None and not raw.empty:
                    break
            except Exception as exc:
                print(f"Download attempt {attempt + 1} failed: {exc}")
                time.sleep(3)

        downloaded[interval] = raw
        time.sleep(2)

    return downloaded


def extract_symbol_frame(raw, ticker):
    if raw is None or raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        level0 = raw.columns.get_level_values(0)
        level1 = raw.columns.get_level_values(1)

        if ticker in level0:
            frame = raw[ticker].copy()
        elif ticker in level1:
            frame = raw.xs(ticker, axis=1, level=1).copy()
        else:
            return pd.DataFrame()
    else:
        frame = raw.copy()

    required = ["Open", "High", "Low", "Close"]
    if any(col not in frame.columns for col in required):
        return pd.DataFrame()

    frame = frame[required].copy()

    for col in required:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=required)

    if frame.index.tz is None:
        frame.index = frame.index.tz_localize("UTC")
    else:
        frame.index = frame.index.tz_convert("UTC")

    frame = frame.sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]

    invalid = (
        (frame["High"] < frame["Low"]) |
        (frame["High"] < frame["Open"]) |
        (frame["High"] < frame["Close"]) |
        (frame["Low"] > frame["Open"]) |
        (frame["Low"] > frame["Close"]) |
        (frame[required] <= 0).any(axis=1)
    )

    return frame.loc[~invalid].copy()


def keep_closed_candles(frame, minutes, now):
    if frame.empty:
        return frame

    close_time = frame.index + pd.Timedelta(minutes=minutes)
    return frame.loc[close_time <= now].copy()


def resample_closed_ohlc(hourly, hours, now):
    """Build only complete 2h/4h candles from 1h candles."""
    if hourly.empty:
        return pd.DataFrame()

    rule = f"{hours}h"

    grouped = hourly.resample(
        rule,
        label="left",
        closed="left",
        origin="start_day"
    )

    result = grouped.agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last"
    })

    count = grouped["Open"].count()
    result = result.loc[count == hours].dropna()

    return keep_closed_candles(result, hours * 60, now)


def build_timeframes(symbol, downloaded, now):
    frames = {}

    for interval, minutes in [
        ("15m", 15),
        ("30m", 30),
        ("1h", 60)
    ]:
        raw = downloaded.get(interval)
        frame = extract_symbol_frame(raw, SYMBOLS[symbol])
        frames[interval] = keep_closed_candles(
            frame, minutes, now
        )

    frames["2h"] = resample_closed_ohlc(
        frames["1h"], 2, now
    )
    frames["4h"] = resample_closed_ohlc(
        frames["1h"], 4, now
    )

    for tf in TIMEFRAMES:
        if frames[tf].empty or len(frames[tf]) < 250:
            raise ValueError(
                f"{symbol} {tf}: insufficient closed candles "
                f"({len(frames[tf])})"
            )

    return frames


def resolve_trade(signal, frame):
    """Resolve TP1/SL using conservative OHLC execution."""
    entry_time = pd.Timestamp(signal["ENTRY_TIME"])

    if entry_time not in frame.index:
        return None

    start = frame.index.get_loc(entry_time)

    if not isinstance(start, (int, np.integer)):
        return None

    direction = signal["DIRECTION"]
    entry = float(signal["ENTRY"])
    stop = float(signal["SL"])
    target = float(signal["TP1"])
    risk = abs(entry - stop)

    if risk <= 0:
        return None

    for position in range(start, len(frame)):
        candle = frame.iloc[position]
        candle_time = frame.index[position]

        o = float(candle["Open"])
        h = float(candle["High"])
        l = float(candle["Low"])

        if direction == "BUY":
            # Stop gap: assume worse fill at candle Open.
            if o <= stop:
                exit_price = o
                outcome = "SL"
            elif o >= target:
                exit_price = target
                outcome = "TP1"
            elif l <= stop and h >= target:
                # Both levels touched: conservative SL first.
                exit_price = stop
                outcome = "SL"
            elif l <= stop:
                exit_price = stop
                outcome = "SL"
            elif h >= target:
                exit_price = target
                outcome = "TP1"
            else:
                continue

            r_multiple = (exit_price - entry) / risk

        else:
            if o >= stop:
                exit_price = o
                outcome = "SL"
            elif o <= target:
                exit_price = target
                outcome = "TP1"
            elif h >= stop and l <= target:
                exit_price = stop
                outcome = "SL"
            elif h >= stop:
                exit_price = stop
                outcome = "SL"
            elif l <= target:
                exit_price = target
                outcome = "TP1"
            else:
                continue

            r_multiple = (entry - exit_price) / risk

        return {
            "OUTCOME": outcome,
            "EXIT_TIME": candle_time,
            "EXIT_PRICE": exit_price,
            "R_MULTIPLE": round(r_multiple, 4),
            "BARS_TO_EXIT": position - start + 1
        }

    return {
        "OUTCOME": "OPEN",
        "EXIT_TIME": None,
        "EXIT_PRICE": None,
        "R_MULTIPLE": np.nan,
        "BARS_TO_EXIT": len(frame) - start
    }


def backtest_symbol(symbol, frames):
    signals = build_signals(frames, symbol)

    if signals is None or signals.empty:
        return []

    trades = []

    for _, signal in signals.iterrows():
        if signal.get("STATUS") != "ENTRY_READY":
            continue

        tf = signal["TIMEFRAME"]
        frame = frames[tf]

        result = resolve_trade(signal, frame)
        if result is None:
            continue

        row = {
            key: signal.get(key)
            for key in [
                "SIGNAL_ID", "SYMBOL", "TIMEFRAME",
                "STRATEGY", "DIRECTION",
                "SIGNAL_CANDLE_TIME", "ENTRY_TIME",
                "ENTRY", "SL", "TP1", "TP2"
            ]
        }
        row.update(result)
        trades.append(row)

    return trades


def calculate_metrics(group):
    closed = group[group["OUTCOME"].isin(["TP1", "SL"])]

    wins = int((closed["OUTCOME"] == "TP1").sum())
    losses = int((closed["OUTCOME"] == "SL").sum())
    open_trades = int((group["OUTCOME"] == "OPEN").sum())
    total_closed = wins + losses

    win_rate = (
        wins / total_closed * 100
        if total_closed else 0.0
    )

    net_r = (
        float(closed["R_MULTIPLE"].sum())
        if total_closed else 0.0
    )

    avg_r = (
        float(closed["R_MULTIPLE"].mean())
        if total_closed else 0.0
    )

    gross_profit = float(
        closed.loc[closed["R_MULTIPLE"] > 0, "R_MULTIPLE"].sum()
    )
    gross_loss = abs(float(
        closed.loc[closed["R_MULTIPLE"] < 0, "R_MULTIPLE"].sum()
    ))

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0 else np.nan
    )

    if total_closed:
        ordered = closed.sort_values("ENTRY_TIME")
        equity = ordered["R_MULTIPLE"].cumsum()
        drawdown = equity - equity.cummax()
        max_drawdown_r = float(drawdown.min())
    else:
        max_drawdown_r = 0.0

    return {
        "TOTAL_SIGNALS": len(group),
        "CLOSED_TRADES": total_closed,
        "WINS": wins,
        "LOSSES": losses,
        "OPEN_TRADES": open_trades,
        "WIN_RATE_PCT": round(win_rate, 2),
        "NET_R": round(net_r, 2),
        "AVERAGE_R": round(avg_r, 3),
        "PROFIT_FACTOR": (
            round(profit_factor, 3)
            if np.isfinite(profit_factor) else None
        ),
        "MAX_DRAWDOWN_R": round(max_drawdown_r, 2)
    }


def make_report(trades):
    if not trades:
        return pd.DataFrame(), pd.DataFrame()

    trade_df = pd.DataFrame(trades)
    trade_df = trade_df.sort_values(
        ["ENTRY_TIME", "SYMBOL", "TIMEFRAME"]
    ).reset_index(drop=True)

    summaries = []

    dimensions = [
        ("OVERALL", []),
        ("STRATEGY", ["STRATEGY"]),
        ("TIMEFRAME", ["TIMEFRAME"]),
        ("SYMBOL", ["SYMBOL"]),
        ("STRATEGY_TIMEFRAME", ["STRATEGY", "TIMEFRAME"])
    ]

    for label, columns in dimensions:
        if not columns:
            summaries.append({
                "GROUP": label,
                "NAME": "ALL",
                **calculate_metrics(trade_df)
            })
            continue

        for keys, group in trade_df.groupby(
            columns, dropna=False
        ):
            if not isinstance(keys, tuple):
                keys = (keys,)

            name = " | ".join(str(x) for x in keys)

            summaries.append({
                "GROUP": label,
                "NAME": name,
                **calculate_metrics(group)
            })

    return trade_df, pd.DataFrame(summaries)


def run_self_tests():
    """Small execution-engine checks before market backtest."""
    idx = pd.date_range(
        "2026-01-01",
        periods=3,
        freq="15min",
        tz="UTC"
    )

    frame = pd.DataFrame({
        "Open": [100, 100, 100],
        "High": [101, 105, 101],
        "Low": [99, 95, 99],
        "Close": [100, 100, 100]
    }, index=idx)

    base = {
        "ENTRY_TIME": idx[0],
        "DIRECTION": "BUY",
        "ENTRY": 100,
        "SL": 98,
        "TP1": 108,
    }

    both = resolve_trade(base, frame)
    assert both["OUTCOME"] == "SL"

    gap_frame = frame.copy()
    gap_frame.iloc[0, gap_frame.columns.get_loc("Open")] = 97
    gap = resolve_trade(base, gap_frame)
    assert gap["OUTCOME"] == "SL"
    assert gap["EXIT_PRICE"] == 97

    open_frame = pd.DataFrame({
        "Open": [100, 100],
        "High": [101, 101],
        "Low": [99, 99],
        "Close": [100, 100]
    }, index=idx[:2])

    opened = resolve_trade(base, open_frame)
    assert opened["OUTCOME"] == "OPEN"

    print("PASS: Same-candle TP1 and SL uses conservative SL")
    print("PASS: Stop-loss gap uses worse candle Open")
    print("PASS: Unresolved trade remains OPEN")


def main():
    print("=" * 60)
    print("PK BOT V6.4 HISTORICAL BACKTEST")
    print("Research only | No live orders | No Telegram")
    print("Period:", PERIOD)
    print("Symbols:", len(SYMBOLS))
    print("=" * 60)

    run_self_tests()

    downloaded = download_market_data()
    now = pd.Timestamp.now(tz="UTC")

    all_trades = []
    failures = []

    for number, symbol in enumerate(SYMBOLS, start=1):
        print(f"\n[{number}/{len(SYMBOLS)}] Testing {symbol}")

        try:
            frames = build_timeframes(
                symbol, downloaded, now
            )

            print(
                "Candles:",
                ", ".join(
                    f"{tf}={len(frames[tf])}"
                    for tf in TIMEFRAMES
                )
            )

            trades = backtest_symbol(symbol, frames)
            all_trades.extend(trades)

            print(f"Signals resolved: {len(trades)}")

        except Exception as exc:
            message = f"{symbol}: {type(exc).__name__}: {exc}"
            failures.append(message)
            print("SYMBOL ERROR:", message)

    output_dir = Path("backtest-results")
    output_dir.mkdir(exist_ok=True)

    trade_df, summary_df = make_report(all_trades)

    if trade_df.empty:
        trade_df = pd.DataFrame(columns=TRADE_COLUMNS)

    trade_df.to_csv(
        output_dir / "pk_v6_4_trades.csv",
        index=False
    )

    summary_df.to_csv(
        output_dir / "pk_v6_4_summary.csv",
        index=False
    )

    with open(
        output_dir / "pk_v6_4_report.txt",
        "w",
        encoding="utf-8"
    ) as report:
        report.write("PK BOT V6.4 BACKTEST REPORT\n")
        report.write(f"Period: {PERIOD}\n")
        report.write("Data: Yahoo Finance\n")
        report.write("XAUUSD source: GC=F Gold Futures proxy\n")
        report.write("TP2 excluded from all outcome calculations.\n")
        report.write(
            "Same-candle TP1/SL ambiguity: SL assumed first.\n"
        )
        report.write(
            "Spread, commission and slippage are NOT included.\n"
        )
        report.write(
            "Overlapping signals are reported independently.\n\n"
        )

        if not summary_df.empty:
            report.write(summary_df.to_string(index=False))
        else:
            report.write("No completed signals were generated.\n")

        report.write("\n\nDATA / SYMBOL ERRORS:\n")
        report.write("\n".join(failures) if failures else "None")

    print("\n" + "=" * 60)
    print("BACKTEST SUMMARY")
    print("=" * 60)

    if not summary_df.empty:
        print(
            summary_df[
                summary_df["GROUP"] == "OVERALL"
            ].to_string(index=False)
        )
    else:
        print("No eligible signals found.")

    print("\nSymbol errors:", len(failures))
    for failure in failures:
        print("WARNING:", failure)

    print("\nFiles saved in backtest-results/")
    print("1. pk_v6_4_trades.csv")
    print("2. pk_v6_4_summary.csv")
    print("3. pk_v6_4_report.txt")

    print("\nNo live orders were placed.")
    print("No Telegram messages were sent.")


if __name__ == "__main__":
    main()
