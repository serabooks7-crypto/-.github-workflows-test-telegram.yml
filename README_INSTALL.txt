PK BOT V6.2 install
Add these files to the repository without changing any V6.1 files:
- pk_v6_2.py
- pk_v6_2_requirements.txt
- .github/workflows/pk-v6-2.yml

Use existing repository secrets TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.
Workflow runs every 5 minutes and can be run manually from GitHub Actions.

Data caveats:
- XAUUSD uses Yahoo GC=F Gold Futures proxy, not broker spot XAUUSD.
- BTCUSD uses BTC-USD. FX symbols use Yahoo Finance feeds; data may be delayed/unavailable.
- Yahoo intraday history is limited. If intraday history is missing, those timeframes are skipped.
- RSI/RVI require 2H EMA200 (at least 205 two-hour bars); otherwise signals are withheld.
- RVI is an implementation approximation and may differ from TradingView/broker RVI.
- Entry is last completed candle close. SL uses recent 10-bar extreme plus buffer; TP is 4R.
- Signals are alerts only, not trade execution. A 1:4 target is not guaranteed to be hit.
