"""
RLA Research Engine V4
Research-only Gold proxy backtester. It never places orders or sends signals.

Data caveats:
- Default GC=F is a Yahoo continuous COMEX gold futures proxy, NOT broker MT5 XAUUSD.
- OHLC bars cannot provide true footprint, DOM, CVD, options positioning or institutional order flow.
- CRT Sweep is a rules-based proxy, not a complete discretionary CRT/TBS model.
- Set realistic round-turn costs in GC=F price units before interpreting net results.
"""

from pathlib import Path
import os
import warnings
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ---------------- USER CONFIG ----------------
SYMBOL = os.getenv("RLA_SYMBOL", "GC=F")
INTERVAL = os.getenv("RLA_INTERVAL", "5m")
PERIOD = os.getenv("RLA_PERIOD", "60d")
TRAIN_FRACTION = 0.70
MAX_HOLDING_BARS = 48                 # 4 hours on 5m bars
ATR_PERIOD = 14
STOP_ATR_MULTIPLIER = 1.25
RR_LEVELS = [1, 1.5, 2, 3, 4, 5]
ROUND_TURN_COST_PRICE = float(os.getenv("RLA_ROUND_TURN_COST_PRICE", "0.0"))
SLIPPAGE_PRICE_PER_SIDE = float(os.getenv("RLA_SLIPPAGE_PRICE_PER_SIDE", "0.0"))
OUTPUT_DIR = Path("research_v4_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# New York local trading windows; timezone conversion handles DST.
# NY AM window includes London/NY overlap. No signals outside these windows.
USE_SESSION_FILTER = True
NY_WINDOWS = [(2, 5), (7, 12)]       # [start_hour, end_hour), New York time


def download_data():
    print(f"Downloading {SYMBOL} {INTERVAL} {PERIOD}...")
    raw = yf.download(
        SYMBOL, period=PERIOD, interval=INTERVAL,
        auto_adjust=False, progress=False, threads=False
    )
    if raw is None or raw.empty:
        raise RuntimeError("Yahoo Finance returned no data.")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise RuntimeError(f"Missing OHLC columns: {missing}")

    cols = required + (["Volume"] if "Volume" in raw.columns else [])
    x = raw[cols].copy()
    for c in cols:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
    x = x[~x.index.duplicated(keep="last")].sort_index()
    if x.index.tz is None:
        # Yahoo intraday data is normally timezone-aware. If not, mark UTC.
        x.index = x.index.tz_localize("UTC")
    else:
        x.index = x.index.tz_convert("UTC")
    if len(x) > 1:
        x = x.iloc[:-1]  # discard possibly forming candle
    if len(x) < 1500:
        raise RuntimeError(f"Only {len(x)} completed bars; at least 1500 required.")
    print(f"Completed bars: {len(x)} | {x.index[0]} -> {x.index[-1]}")
    print("WARNING: GC=F is a futures proxy, not broker XAUUSD.")
    return x


def add_features(df):
    x = df.copy()
    c, h, l, o = x.Close, x.High, x.Low, x.Open

    # Local structure and volatility.
    x["EMA20"] = c.ewm(span=20, adjust=False).mean()
    x["EMA50"] = c.ewm(span=50, adjust=False).mean()
    x["EMA200"] = c.ewm(span=200, adjust=False).mean()
    prev = c.shift(1)
    tr = pd.concat([(h-l), (h-prev).abs(), (l-prev).abs()], axis=1).max(axis=1)
    x["ATR"] = tr.ewm(alpha=1/ATR_PERIOD, adjust=False).mean()
    delta = c.diff()
    avg_gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    x["RSI"] = 100 - 100/(1+rs)

    # Previous-bar-only liquidity references (no current-bar lookahead).
    x["PREV_H20"] = h.shift(1).rolling(20).max()
    x["PREV_L20"] = l.shift(1).rolling(20).min()
    x["PREV_H12"] = h.shift(1).rolling(12).max()
    x["PREV_L12"] = l.shift(1).rolling(12).min()
    x["PREV_H5"] = h.shift(1).rolling(5).max()
    x["PREV_L5"] = l.shift(1).rolling(5).min()

    candle_range = (h-l).replace(0, np.nan)
    x["BODY_ATR"] = (c-o).abs()/x["ATR"]
    x["CLOSE_LOC"] = (c-l)/candle_range

    # Higher-timeframe bias: completed 1h and 4h bars only.
    # Resample labels at bar close, then forward-fill onto 5m timestamps.
    for rule, tag in [("1h", "H1"), ("4h", "H4")]:
        ht = x[["Open", "High", "Low", "Close"]].resample(
            rule, label="right", closed="right"
        ).agg({"Open":"first", "High":"max", "Low":"min", "Close":"last"}).dropna()
        ht[f"{tag}_EMA20"] = ht.Close.ewm(span=20, adjust=False).mean()
        ht[f"{tag}_EMA50"] = ht.Close.ewm(span=50, adjust=False).mean()
        ht[f"{tag}_EMA200"] = ht.Close.ewm(span=200, adjust=False).mean()
        ht = ht[[f"{tag}_EMA20", f"{tag}_EMA50", f"{tag}_EMA200"]]
        x = x.join(ht.reindex(x.index, method="ffill"))

    # Session filter based on New York clock, including daylight saving.
    ny = x.index.tz_convert("America/New_York")
    hour = pd.Series(ny.hour, index=x.index)
    if USE_SESSION_FILTER:
        allowed = pd.Series(False, index=x.index)
        for start, end in NY_WINDOWS:
            allowed |= (hour >= start) & (hour < end)
        x["IN_SESSION"] = allowed
    else:
        x["IN_SESSION"] = True

    return x.replace([np.inf, -np.inf], np.nan)


def build_signals(x):
    z = x.copy()
    # HTF consensus filters reduce counter-trend entries.
    htf_bull = (
        (z.H1_EMA20 > z.H1_EMA50) &
        (z.H4_EMA20 > z.H4_EMA50)
    )
    htf_bear = (
        (z.H1_EMA20 < z.H1_EMA50) &
        (z.H4_EMA20 < z.H4_EMA50)
    )

    # 1) CRT-like liquidity sweep + reclaim + local MSS/displacement proxy.
    # A sell sweep takes prior highs, closes back below, and closes below
    # a recent micro swing low. Buy is the inverse.
    z["CRT_SWEEP_SELL"] = (
        z.IN_SESSION &
        (z.High > z.PREV_H20) & (z.Close < z.PREV_H20) &
        (z.Close < z.Open) & (z.CLOSE_LOC < 0.30) &
        (z.Close < z.PREV_L5) & (z.BODY_ATR >= 0.35) &
        htf_bear & z.RSI.between(25, 55)
    )
    z["CRT_SWEEP_BUY"] = (
        z.IN_SESSION &
        (z.Low < z.PREV_L20) & (z.Close > z.PREV_L20) &
        (z.Close > z.Open) & (z.CLOSE_LOC > 0.70) &
        (z.Close > z.PREV_H5) & (z.BODY_ATR >= 0.35) &
        htf_bull & z.RSI.between(45, 75)
    )

    # 2) Trend pullback, aligned to both higher timeframes.
    z["TREND_PULLBACK_SELL"] = (
        z.IN_SESSION & htf_bear &
        (z.EMA20 < z.EMA50) & (z.EMA50 < z.EMA200) &
        (z.High >= z.EMA20) & (z.Close < z.EMA20) &
        (z.Close < z.Open) & z.RSI.between(30, 50) &
        (z.BODY_ATR >= 0.15)
    )
    z["TREND_PULLBACK_BUY"] = (
        z.IN_SESSION & htf_bull &
        (z.EMA20 > z.EMA50) & (z.EMA50 > z.EMA200) &
        (z.Low <= z.EMA20) & (z.Close > z.EMA20) &
        (z.Close > z.Open) & z.RSI.between(50, 70) &
        (z.BODY_ATR >= 0.15)
    )

    # 3) Range expansion breakout with volatility expansion and HTF agreement.
    atr_mean = z.ATR.rolling(50).mean()
    z["RANGE_EXPANSION_SELL"] = (
        z.IN_SESSION & htf_bear &
        (z.Close < z.PREV_L12) & (z.Close < z.Open) &
        (z.BODY_ATR >= 0.60) & (z.ATR > atr_mean) &
        (z.CLOSE_LOC < 0.25) & (z.RSI < 48)
    )
    z["RANGE_EXPANSION_BUY"] = (
        z.IN_SESSION & htf_bull &
        (z.Close > z.PREV_H12) & (z.Close > z.Open) &
        (z.BODY_ATR >= 0.60) & (z.ATR > atr_mean) &
        (z.CLOSE_LOC > 0.75) & (z.RSI > 52)
    )

    signal_cols = [
        "CRT_SWEEP_SELL", "CRT_SWEEP_BUY",
        "TREND_PULLBACK_SELL", "TREND_PULLBACK_BUY",
        "RANGE_EXPANSION_SELL", "RANGE_EXPANSION_BUY"
    ]
    # Preserve warmup rows as false; only indicators required for a model matter.
    for col in signal_cols:
        z[col] = z[col].fillna(False).astype(bool)
    return z, signal_cols


def backtest_model(x, signal_col, rr, split_index):
    direction = "BUY" if signal_col.endswith("_BUY") else "SELL"
    trades = []
    next_free = 0
    n = len(x)
    valid_start = 300

    for i in range(valid_start, n - MAX_HOLDING_BARS - 2):
        if i < next_free or not bool(x.iloc[i][signal_col]):
            continue
        # Strictly purge any possible holding window crossing train/test boundary.
        if i < split_index and i + MAX_HOLDING_BARS + 1 >= split_index:
            continue

        entry_i = i + 1
        entry = float(x.iloc[entry_i].Open)
        atr = float(x.iloc[i].ATR)
        if not np.isfinite(entry) or not np.isfinite(atr) or atr <= 0:
            continue

        risk = STOP_ATR_MULTIPLIER * atr
        if direction == "BUY":
            sl, tp = entry-risk, entry+rr*risk
        else:
            sl, tp = entry+risk, entry-rr*risk

        outcome, exit_price, exit_i = "TIMEOUT", None, None
        end_i = min(i + MAX_HOLDING_BARS + 1, n-1)
        for j in range(entry_i, end_i+1):
            bar = x.iloc[j]
            op, hi, lo = float(bar.Open), float(bar.High), float(bar.Low)
            if direction == "BUY":
                if op <= sl:
                    outcome, exit_price, exit_i = "LOSS_GAP", op, j
                    break
                if op >= tp:
                    outcome, exit_price, exit_i = "TP_GAP", op, j
                    break
                hit_sl, hit_tp = lo <= sl, hi >= tp
            else:
                if op >= sl:
                    outcome, exit_price, exit_i = "LOSS_GAP", op, j
                    break
                if op <= tp:
                    outcome, exit_price, exit_i = "TP_GAP", op, j
                    break
                hit_sl, hit_tp = hi >= sl, lo <= tp

            # Conservative intrabar ambiguity: stop wins if both levels touched.
            if hit_sl:
                outcome, exit_price, exit_i = "LOSS", sl, j
                break
            if hit_tp:
                outcome, exit_price, exit_i = "TP", tp, j
                break

        if exit_i is None:
            exit_i = end_i
            exit_price = float(x.iloc[end_i].Close)

        gross = (exit_price-entry) if direction == "BUY" else (entry-exit_price)
        costs = ROUND_TURN_COST_PRICE + 2*SLIPPAGE_PRICE_PER_SIDE
        net = gross - costs
        net_r = net/risk

        trades.append({
            "signal_time_utc": x.index[i],
            "entry_time_utc": x.index[entry_i],
            "exit_time_utc": x.index[exit_i],
            "signal_model": signal_col,
            "direction": direction,
            "target_RR": rr,
            "entry": entry, "sl": sl, "tp": tp, "exit_price": exit_price,
            "outcome": outcome,
            "gross_pnl_price": gross,
            "round_turn_cost_price": ROUND_TURN_COST_PRICE,
            "slippage_price_per_side": SLIPPAGE_PRICE_PER_SIDE,
            "total_cost_price": costs,
            "net_pnl_price": net, "net_R": net_r,
            "risk_price": risk,
            "bars_held": exit_i-entry_i,
            "sample": "IN_SAMPLE" if i < split_index else "OUT_OF_SAMPLE",
            "session_filter": "NY London/NY windows"
        })
        next_free = exit_i + 1

    return trades


def calc_metrics(g):
    if g.empty:
        return {}
    n = len(g)
    tp = g.outcome.str.startswith("TP").sum()
    sl = g.outcome.str.startswith("LOSS").sum()
    wins = (g.net_R > 0).sum()
    gains = g.loc[g.net_R > 0, "net_R"].sum()
    losses = abs(g.loc[g.net_R < 0, "net_R"].sum())
    pf = gains/losses if losses > 0 else np.inf
    eq = pd.concat([pd.Series([0.0]), g.net_R.reset_index(drop=True)]).cumsum()
    dd = eq - eq.cummax()
    return {
        "trades": int(n),
        "TP_hit_rate_pct": round(tp/n*100, 2),
        "profitable_trade_rate_pct": round(wins/n*100, 2),
        "SL_rate_pct": round(sl/n*100, 2),
        "average_net_R": round(g.net_R.mean(), 4),
        "median_net_R": round(g.net_R.median(), 4),
        "profit_factor_R": round(pf, 4) if np.isfinite(pf) else "Inf",
        "max_drawdown_R": round(abs(dd.min()), 4),
        "total_net_R": round(g.net_R.sum(), 4),
        "avg_bars_held": round(g.bars_held.mean(), 2)
    }


def main():
    print("=== RLA RESEARCH ENGINE V4 | RESEARCH ONLY ===")
    print(f"Symbol={SYMBOL} | Interval={INTERVAL} | Period={PERIOD}")
    print(f"Round-turn cost={ROUND_TURN_COST_PRICE} | Slippage/side={SLIPPAGE_PRICE_PER_SIDE}")
    if ROUND_TURN_COST_PRICE == 0 and SLIPPAGE_PRICE_PER_SIDE == 0:
        print("WARNING: costs are zero. Net results are not execution-realistic.")
    print("HTF: completed 1h + 4h EMA bias | Session: NY 02-05 and 07-12")
    print("No live trading, no Telegram alerts.")

    data = add_features(download_data())
    data, models = build_signals(data)
    data = data.dropna(subset=[
        "ATR", "EMA20", "EMA50", "EMA200",
        "H1_EMA20", "H1_EMA50", "H4_EMA20", "H4_EMA50",
        "PREV_H20", "PREV_L20", "PREV_H12", "PREV_L12",
        "PREV_H5", "PREV_L5"
    ]).copy()
    split = int(len(data)*TRAIN_FRACTION)
    print(f"Usable bars={len(data)} | Train={split} | OOS={len(data)-split}")

    all_trades = []
    for model in models:
        for rr in RR_LEVELS:
            print(f"Testing {model} | 1:{rr}")
            all_trades.extend(backtest_model(data, model, rr, split))

    if not all_trades:
        raise RuntimeError("No trades generated. Review data, session window or filters.")

    trades = pd.DataFrame(all_trades)
    trades.to_csv(OUTPUT_DIR/"all_trades_v4.csv", index=False)

    rows = []
    for sample in ["IN_SAMPLE", "OUT_OF_SAMPLE"]:
        for model in models:
            direction = "BUY" if model.endswith("_BUY") else "SELL"
            for rr in RR_LEVELS:
                g = trades[
                    (trades["sample"] == sample) &
                    (trades["signal_model"] == model) &
                    (trades["target_RR"] == rr)
                ]
                m = calc_metrics(g)
                if m:
                    rows.append({
                        "sample": sample, "model": model,
                        "direction": direction, "target_RR": rr, **m
                    })

    report = pd.DataFrame(rows)
    report.to_csv(OUTPUT_DIR/"performance_report_v4.csv", index=False)

    oos = report[
        (report["sample"] == "OUT_OF_SAMPLE") &
        (report["trades"] >= 30)
    ].copy()
    if not oos.empty:
        oos["pf_sort"] = pd.to_numeric(oos.profit_factor_R, errors="coerce")
        oos = oos.sort_values(
            ["average_net_R", "pf_sort", "trades"],
            ascending=[False, False, False]
        )
        oos.to_csv(OUTPUT_DIR/"oos_research_ranking_v4.csv", index=False)

    # Signal frequency diagnostics by model, independently of RR expansion.
    diag = []
    for model in models:
        diag.append({
            "model": model,
            "signal_bars": int(data[model].sum()),
            "session_bars": int(data.IN_SESSION.sum()),
            "signal_rate_pct_of_session_bars": round(
                data[model].sum()/max(1, data.IN_SESSION.sum())*100, 3
            )
        })
    pd.DataFrame(diag).to_csv(OUTPUT_DIR/"signal_diagnostics_v4.csv", index=False)

    print("\n=== OOS RESEARCH RANKING (min 30 trades) ===")
    if not oos.empty:
        print(oos.to_string(index=False))
    else:
        print("No model reached 30 OOS trades.")
    print("\nReports saved in:", OUTPUT_DIR)
    print("IMPORTANT: Research candidates only; no model is approved for live use.")
    print("Validate against broker MT5 XAUUSD data and realistic costs before conclusions.")


if __name__ == "__main__":
    main()
