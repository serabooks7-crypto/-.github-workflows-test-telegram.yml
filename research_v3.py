"""
RLA Research Engine V3
Research-only XAUUSD proxy backtester. No live orders or Telegram alerts.

IMPORTANT:
- Default data is Yahoo Finance GC=F (continuous COMEX gold futures proxy),
  NOT the user's FundingPips MT5 XAUUSD feed.
- This engine does NOT claim to implement true footprint/order-book,
  options positioning, or broker-specific CRT/TBS data.
- Configure realistic round-trip costs before interpreting net results.
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ---------------- CONFIG ----------------
SYMBOL = "GC=F"
INTERVAL = "5m"
PERIOD = "60d"

ATR_PERIOD = 14
MAX_HOLDING_BARS = 60
STOP_ATR_MULTIPLIER = 1.5
RR_LEVELS = [1, 1.5, 2, 3, 4, 5, 6, 8, 10]

# Set these to realistic round-trip costs in the SAME price units as GC=F.
# Zero defaults are for pipeline testing ONLY and produce gross-like results.
ROUND_TURN_COST_PRICE = 0.0

# Purge trades whose holding window crosses the train/test boundary.
TRAIN_FRACTION = 0.70
OUTPUT_DIR = Path("research_v3_output")
OUTPUT_DIR.mkdir(exist_ok=True)


def download_data():
    print(f"Downloading {SYMBOL} {INTERVAL} data for {PERIOD}...")
    df = yf.download(
        SYMBOL, period=PERIOD, interval=INTERVAL,
        auto_adjust=False, progress=False, threads=False
    )
    if df is None or df.empty:
        raise RuntimeError("Yahoo Finance returned no data.")

    if isinstance(df.columns, pd.MultiIndex):
        # Single ticker download: use the price-field level.
        df.columns = df.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    keep = required + (["Volume"] if "Volume" in df.columns else [])
    df = df[keep].copy()
    for col in keep:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df[~df.index.duplicated(keep="last")].sort_index()

    # Exclude the most recent candle, which may still be forming.
    if len(df) > 1:
        df = df.iloc[:-1].copy()
    if len(df) < 1000:
        raise RuntimeError(f"Insufficient completed candles: {len(df)}")

    print("Candles:", len(df))
    print("First:", df.index[0], "| Last:", df.index[-1])
    print("Data source warning: GC=F is not broker MT5 XAUUSD.")
    return df


def add_indicators(df):
    x = df.copy()
    c, h, l = x["Close"], x["High"], x["Low"]

    x["EMA20"] = c.ewm(span=20, adjust=False).mean()
    x["EMA50"] = c.ewm(span=50, adjust=False).mean()
    x["EMA200"] = c.ewm(span=200, adjust=False).mean()

    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    x["RSI"] = 100 - (100 / (1 + rs))

    prev_close = c.shift(1)
    tr = pd.concat([
        h - l, (h - prev_close).abs(), (l - prev_close).abs()
    ], axis=1).max(axis=1)
    x["ATR"] = tr.ewm(alpha=1/ATR_PERIOD, adjust=False).mean()

    # Completed-bar range references: current candle excluded.
    x["RANGE_H20"] = h.shift(1).rolling(20).max()
    x["RANGE_L20"] = l.shift(1).rolling(20).min()
    x["RANGE_H12"] = h.shift(1).rolling(12).max()
    x["RANGE_L12"] = l.shift(1).rolling(12).min()

    # Rolling VWAP proxy using available candle volume. If volume is absent,
    # zero, or unreliable, this model is disabled rather than fabricated.
    if "Volume" in df.columns:
        vol = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)
        typical = (h + l + c) / 3
        pv = typical * vol
        rolling_vol = vol.rolling(48).sum()
        x["VWAP48"] = pv.rolling(48).sum() / rolling_vol.replace(0, np.nan)
    else:
        x["VWAP48"] = np.nan

    # Candle structure / displacement proxy.
    x["BODY"] = (c - x["Open"]).abs()
    x["RANGE"] = (h - l).replace(0, np.nan)
    x["BODY_ATR"] = x["BODY"] / x["ATR"]
    x["CLOSE_LOCATION"] = (c - l) / x["RANGE"]

    return x.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["EMA20", "EMA50", "EMA200", "RSI", "ATR",
                "RANGE_H20", "RANGE_L20", "RANGE_H12", "RANGE_L12"]
    )


def build_models(x):
    z = x.copy()
    models = {}

    # 1. Trend continuation after pullback.
    models["TREND_PULLBACK_BUY"] = (
        (z.EMA20 > z.EMA50) & (z.EMA50 > z.EMA200) &
        (z.Low <= z.EMA20) & (z.Close > z.EMA20) &
        z.RSI.between(50, 68)
    )
    models["TREND_PULLBACK_SELL"] = (
        (z.EMA20 < z.EMA50) & (z.EMA50 < z.EMA200) &
        (z.High >= z.EMA20) & (z.Close < z.EMA20) &
        z.RSI.between(32, 50)
    )

    # 2. Liquidity sweep + reclaim (OHLC proxy, not true order-flow).
    models["SWEEP_RECLAIM_BUY"] = (
        (z.Low < z.RANGE_L20) & (z.Close > z.RANGE_L20) &
        (z.Close > z.Open) & (z.RSI > 45) &
        (z.CLOSE_LOCATION > 0.60)
    )
    models["SWEEP_RECLAIM_SELL"] = (
        (z.High > z.RANGE_H20) & (z.Close < z.RANGE_H20) &
        (z.Close < z.Open) & (z.RSI < 55) &
        (z.CLOSE_LOCATION < 0.40)
    )

    # 3. Range breakout with displacement confirmation.
    models["DISPLACEMENT_BREAK_BUY"] = (
        (z.Close > z.RANGE_H20) & (z.Close > z.Open) &
        (z.BODY_ATR >= 0.60) & (z.RSI > 55)
    )
    models["DISPLACEMENT_BREAK_SELL"] = (
        (z.Close < z.RANGE_L20) & (z.Close < z.Open) &
        (z.BODY_ATR >= 0.60) & (z.RSI < 45)
    )

    # 4. Short-range expansion.
    models["RANGE_EXPANSION_BUY"] = (
        (z.Close > z.RANGE_H12) & (z.Close > z.Open) &
        (z.ATR > z.ATR.rolling(50).mean()) & (z.RSI > 52)
    )
    models["RANGE_EXPANSION_SELL"] = (
        (z.Close < z.RANGE_L12) & (z.Close < z.Open) &
        (z.ATR > z.ATR.rolling(50).mean()) & (z.RSI < 48)
    )

    # 5. VWAP reclaim proxy; only enabled if meaningful volume exists.
    models["VWAP_RECLAIM_BUY"] = (
        z.VWAP48.notna() & (z.Close > z.VWAP48) &
        (z.Close.shift(1) <= z.VWAP48.shift(1)) &
        (z.EMA20 > z.EMA50) & (z.RSI > 50)
    )
    models["VWAP_RECLAIM_SELL"] = (
        z.VWAP48.notna() & (z.Close < z.VWAP48) &
        (z.Close.shift(1) >= z.VWAP48.shift(1)) &
        (z.EMA20 < z.EMA50) & (z.RSI < 50)
    )

    # CRT/TBS-like range sweep/reclaim research proxy. This is explicitly
    # not a full CRT/TBS discretionary model or HTF narrative implementation.
    models["CRT_SWEEP_PROXY_BUY"] = (
        (z.Low < z.RANGE_H20.shift(1)) &
        (z.Close > z.RANGE_L20) & (z.Close > z.Open) &
        (z.CLOSE_LOCATION > 0.70) & (z.RSI > 48)
    )
    models["CRT_SWEEP_PROXY_SELL"] = (
        (z.High > z.RANGE_L20.shift(1)) &
        (z.Close < z.RANGE_H20) & (z.Close < z.Open) &
        (z.CLOSE_LOCATION < 0.30) & (z.RSI < 52)
    )

    for name in models:
        z[name] = models[name].fillna(False).astype(bool)
    return z, list(models.keys())


def backtest_one(x, signal_col, direction, rr, split_index):
    trades = []
    next_free = 0
    n = len(x)
    # Reserve enough bars for a full possible holding period.
    for i in range(250, n - MAX_HOLDING_BARS - 1):
        if i < next_free or not bool(x.iloc[i][signal_col]):
            continue

        # The test boundary is a hard separation. Do not carry trades
        # from training into test or let test signals use training exits.
        if i < split_index and i + MAX_HOLDING_BARS >= split_index:
            continue

        entry_bar = i + 1
        entry = float(x.iloc[entry_bar]["Open"])
        atr = float(x.iloc[i]["ATR"])
        if not np.isfinite(entry) or not np.isfinite(atr) or atr <= 0:
            continue

        risk = STOP_ATR_MULTIPLIER * atr
        if direction == "BUY":
            sl, tp = entry - risk, entry + rr * risk
        else:
            sl, tp = entry + risk, entry - rr * risk

        outcome, exit_price, exit_i = "TIMEOUT", None, None
        end_i = min(i + MAX_HOLDING_BARS + 1, n - 1)

        for j in range(entry_bar, end_i + 1):
            bar = x.iloc[j]
            o, hi, lo = float(bar.Open), float(bar.High), float(bar.Low)

            if direction == "BUY":
                # Gap handling: if open crosses a level, assume fill at open.
                if o <= sl:
                    outcome, exit_price, exit_i = "LOSS", o, j
                    break
                if o >= tp:
                    outcome, exit_price, exit_i = "TP", o, j
                    break
                hit_sl, hit_tp = lo <= sl, hi >= tp
            else:
                if o >= sl:
                    outcome, exit_price, exit_i = "LOSS", o, j
                    break
                if o <= tp:
                    outcome, exit_price, exit_i = "TP", o, j
                    break
                hit_sl, hit_tp = hi >= sl, lo <= tp

            # Conservative ambiguity rule: if both touched in one bar,
            # count stop first.
            if hit_sl:
                outcome, exit_price, exit_i = "LOSS", sl, j
                break
            if hit_tp:
                outcome, exit_price, exit_i = "TP", tp, j
                break

        if exit_i is None:
            exit_i = end_i
            exit_price = float(x.iloc[end_i]["Close"])

        gross = (exit_price - entry) if direction == "BUY" else (entry - exit_price)
        net = gross - ROUND_TURN_COST_PRICE
        net_r = net / risk

        trades.append({
            "signal_time": x.index[i],
            "entry_time": x.index[entry_bar],
            "exit_time": x.index[exit_i],
            "model": signal_col,
            "direction": direction,
            "target_RR": rr,
            "entry": entry, "sl": sl, "tp": tp,
            "exit_price": exit_price,
            "outcome": outcome,
            "gross_pnl_price": gross,
            "round_turn_cost_price": ROUND_TURN_COST_PRICE,
            "net_pnl_price": net,
            "net_R": net_r,
            "bars_held": exit_i - entry_bar,
            "sample": "IN_SAMPLE" if i < split_index else "OUT_OF_SAMPLE"
        })
        next_free = exit_i + 1
    return trades


def metrics(g):
    if g.empty:
        return None
    total = len(g)
    tp = int((g.outcome == "TP").sum())
    sl = int((g.outcome == "LOSS").sum())
    timeout = int((g.outcome == "TIMEOUT").sum())
    wins = int((g.net_pnl_price > 0).sum())
    gp = g.loc[g.net_pnl_price > 0, "net_pnl_price"].sum()
    gl = abs(g.loc[g.net_pnl_price < 0, "net_pnl_price"].sum())
    pf = gp / gl if gl > 0 else np.inf
    equity = pd.concat([pd.Series([0.0]), g.net_R.reset_index(drop=True)]).cumsum()
    dd = equity - equity.cummax()
    return {
        "trades": total,
        "TP_hits": tp,
        "SL_hits": sl,
        "timeouts": timeout,
        "TP_hit_rate_pct": round(tp / total * 100, 2),
        "profitable_trade_rate_pct": round(wins / total * 100, 2),
        "average_net_R": round(g.net_R.mean(), 4),
        "profit_factor": round(pf, 4) if np.isfinite(pf) else "Inf",
        "max_drawdown_R": round(abs(dd.min()), 4),
        "avg_bars_held": round(g.bars_held.mean(), 2)
    }


def main():
    print("RLA Research Engine V3 started.")
    print("Configured round-trip cost:", ROUND_TURN_COST_PRICE)
    if ROUND_TURN_COST_PRICE == 0:
        print("WARNING: costs are zero; results are NOT net-realistic.")

    data = add_indicators(download_data())
    data, model_names = build_models(data)
    split = int(len(data) * TRAIN_FRACTION)

    all_trades = []
    for model in model_names:
        direction = "BUY" if model.endswith("_BUY") else "SELL"
        for rr in RR_LEVELS:
            print(f"Testing {model} | 1:{rr}")
            all_trades.extend(
                backtest_one(data, model, direction, rr, split)
            )

    if not all_trades:
        print("No trades generated. No report files created.")
        return

    trades = pd.DataFrame(all_trades)
    trades.to_csv(OUTPUT_DIR / "all_trades_v3.csv", index=False)

    rows = []
    for sample in ["IN_SAMPLE", "OUT_OF_SAMPLE"]:
        for model in model_names:
            direction = "BUY" if model.endswith("_BUY") else "SELL"
            for rr in RR_LEVELS:
                g = trades[
                    (trades["sample"] == sample) &
                    (trades["model"] == model) &
                    (trades["target_RR"] == rr)
                ]
                m = metrics(g)
                if m:
                    rows.append({
                        "sample": sample, "model": model,
                        "direction": direction, "target_RR": rr, **m
                    })

    report = pd.DataFrame(rows)
    report.to_csv(OUTPUT_DIR / "performance_report_v3.csv", index=False)

    # Simple research shortlist only; not an automatic live-trading approval.
    oos = report[
        (report["sample"] == "OUT_OF_SAMPLE") &
        (report["trades"] >= 30)
    ].copy()
    if not oos.empty:
        oos["pf_numeric"] = pd.to_numeric(oos["profit_factor"], errors="coerce")
        shortlist = oos.sort_values(
            ["average_net_R", "pf_numeric", "TP_hit_rate_pct"],
            ascending=False
        )
        shortlist.to_csv(OUTPUT_DIR / "oos_research_ranking_v3.csv", index=False)

    print("\n===== V3 PERFORMANCE REPORT =====")
    print(report.to_string(index=False))
    print("\nSaved reports in:", OUTPUT_DIR)
    print("IMPORTANT: Research only. Validate costs and broker data before use.")
    print("OHLC proxies cannot reproduce true order book/footprint/options data.")


if __name__ == "__main__":
    main()
