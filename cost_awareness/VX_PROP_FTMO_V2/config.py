"""
VX PROP FTMO V2 CONFIGURATION
"""

# ==============================
# MT5 ACCOUNT
# ==============================

ACCOUNT_LOGIN = 1514154175
ACCOUNT_PASSWORD = "$rx@6j?uv55P"
ACCOUNT_SERVER = "FTMO-Demo"

TERMINAL_PATH = (
    r"C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe"
)


# ==============================
# FTMO RULES
# ==============================

START_EQUITY = 100000.0

PROFIT_TARGET = 10.0

DAILY_HALT_PCT = 4.0

TOTAL_HALT_PCT = 8.0


# ==============================
# RISK
# ==============================

RISK_PCT = 0.4

MAX_OPEN_TRADES = 3


# ==============================
# SYMBOLS
# ==============================

SYMBOLS = [
    "EURUSD",
    "AUDUSD",
    "USDJPY",
    "USDCHF",
    "XAUUSD",
]


# ==============================
# BOT
# ==============================

MAGIC = 555555

TIMEFRAME = "M5"

CANDLE_COUNT = 3000

LOOP_INTERVAL = 30


# ==============================
# EXECUTION
# ==============================

RR = 1.5

ATR_MULTIPLIER = 0.8

MIN_ATR_POINTS = 5

STOP_BUFFER_MULT = 1.5