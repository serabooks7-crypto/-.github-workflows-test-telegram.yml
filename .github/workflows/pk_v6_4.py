"""
PK BOT V6.4
Main signal engine
"""

VERSION = "6.4"

INSTRUMENTS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF",
    "USDCAD", "AUDUSD", "NZDUSD",
    "EURJPY", "GBPJPY", "EURGBP", "EURCHF",
    "AUDJPY", "CHFJPY", "EURAUD", "EURCAD",
    "AUDCAD", "GBPAUD", "GBPCAD", "AUDNZD",
    "NZDJPY", "CADJPY", "GBPNZD", "EURNZD",
    "CADCHF", "AUDCHF", "GBPCHF", "NZDCAD",
    "NZDCHF", "XAUUSD", "XAGUSD"
]

TIMEFRAMES = [
    "15m", "30m", "1h", "2h",
    "4h", "6h", "8h", "12h", "1d"
]

MIN_RISK_REWARD = 4.0
ATR_BUFFER = 0.20

STRATEGIES = ["EMA", "RSI", "RVI"]


def initialize():
    print("PK BOT V6.4 initialized")
    print("Instruments:", len(INSTRUMENTS))
    print("Strategies:", ", ".join(STRATEGIES))
    print("Minimum TP1 Risk/Reward:", MIN_RISK_REWARD)
    print("ATR Buffer:", ATR_BUFFER)


if __name__ == "__main__":
    initialize()
