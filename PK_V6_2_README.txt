PK BOT V6.2 corrected package.
Upload pk_v6_2.py and pk_v6_2_requirements.txt to repository root.
Upload .github/workflows/pk-v6-2.yml to the exact .github/workflows path.
Keep GitHub Secrets TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.
Rules: EMA20/100 cross has no 2H filter. RSI14 BUY: excursion below 15 then cross above 15, 2H close > EMA200; SELL: excursion above 85 then cross below 85, 2H close < EMA200. RVI8 BUY: excursion below -0.500 then cross above -0.500, 2H close > EMA200; SELL: excursion above +0.500 then cross below +0.500, 2H close < EMA200. Any one strategy can trigger. Closed candles only. RR target 1:4. Every signal sends a chart.
IMPORTANT: Yahoo Finance data is not Exness/MT5 broker data. GC=F, SI=F, CL=F are futures proxies for XAUUSD, XAGUSD, USOIL. Forex ticker conventions may be inverted or differ from broker quotes. Some symbol/timeframe history may be unavailable and skipped. RVI is an approximation and may differ from TradingView/MT5. SL uses 1.5x ATR proxy; TP is 4x stop distance, not guaranteed. Scheduled GitHub runs may be delayed. No backtest/live-feed validation has been done; test on demo first.
