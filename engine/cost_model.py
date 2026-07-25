"""
Shared cost model for turning raw MT5 data into real USD costs.

This exists because the project had THREE different, inconsistent spread
assumptions living in three different files:
  - engine/backtester.py         : flat 0.0002 constant, wrong units (see below)
  - engine/execution_engine.py   : flat 0.0001 constant, one-sided, entry only
  - scripts/backtest_live_logic.py: real per-bar spread, but only for 6 symbols
    and only when --include-spread is passed; commission still not modeled

Meanwhile scripts/instrument_screener.py already proved on this project's own
data that flat/guessed spread assumptions materially change conclusions
(AUDUSD: 6/6 walk-forward folds profitable on paper, 0/6 profitable once real
spread was modeled). None of that realism reaches the actual backtester used
by evaluate/walkforward/tournament scripts -- this module is the fix, in one
place, so every script uses the same numbers instead of drifting apart again.

Nothing here talks to MT5. It only converts numbers you already have
(a symbol name, a per-bar 'spread' column, an optional measured commission
map) into a single USD cost per trade.
"""

import json
from pathlib import Path

import pandas as pd

COMMISSION_MAP_PATH = Path("configs/commission_map.json")
SWAP_MAP_PATH = Path("configs/swap_map.json")

# Point size: multiply the CSV's 'spread' column (MT5's native integer point
# unit) by this to get a price-space distance. Digits-per-symbol, not a
# guess -- matches scripts/backtest_live_logic.py's POINT_SIZE table, extended
# to the other pairs mentioned in this project's own trading plan.
POINT_SIZE = {
    "EURUSD": 0.00001,
    "GBPUSD": 0.00001,
    "AUDUSD": 0.00001,
    "USDCAD": 0.00001,
    "EURGBP": 0.00001,
    "USDCHF": 0.00001,
    "USDJPY": 0.001,
    "EURJPY": 0.001,
    "GBPJPY": 0.001,
    "XAUUSD": 0.01,
    "BTCUSD": 0.01,
}
DEFAULT_POINT_SIZE = 0.00001

# Contract size per 1.0 lot, and whether the raw price-move PnL comes out in
# USD directly or needs dividing by price (USD-base pairs: USDJPY, USDCHF).
# Same approximation backtest_live_logic.py uses -- accurate for CCY/USD
# quoting and XAUUSD, approximate for USD/CCY pairs.
CONTRACT_SIZE = {
    "EURUSD": (100_000.0, False),
    "GBPUSD": (100_000.0, False),
    "AUDUSD": (100_000.0, False),
    "USDCAD": (100_000.0, True),
    "EURGBP": (100_000.0, False),
    "USDCHF": (100_000.0, True),
    "USDJPY": (100_000.0, True),
    "EURJPY": (100_000.0, False),
    "GBPJPY": (100_000.0, False),
    "XAUUSD": (100.0, False),
    "BTCUSD": (1.0, False),
}
DEFAULT_CONTRACT = (100_000.0, False)

# Fallback commission by asset class (USD per lot round trip), used only
# when configs/commission_map.json (produced by scripts/measure_commission.py
# from real deal history) has nothing for this symbol. Same figures
# scripts/instrument_screener.py already uses, so a screener run and a
# backtester run never silently disagree about cost.
CLASS_COMMISSION = {
    "fx": 5.04,
    "metal": 6.00,
    "index": 0.0,
    "energy": 0.0,
    "crypto": 0.0,
}
METAL_HINTS = ("XAU", "XAG", "XPT", "XPD", "GOLD", "SILVER")
CRYPTO_HINTS = ("BTC", "ETH", "LTC", "XRP", "SOL")


def classify(symbol):
    upper = (symbol or "").upper()
    if any(hint in upper for hint in CRYPTO_HINTS):
        return "crypto"
    if any(hint in upper for hint in METAL_HINTS):
        return "metal"
    return "fx"


def load_commission_map(path=None):
    """Real measured commission, if scripts/measure_commission.py has been
    run. Returns {} if not -- callers fall back to CLASS_COMMISSION."""
    p = Path(path) if path else COMMISSION_MAP_PATH
    if not p.exists():
        return {}
    try:
        with p.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return {}
    return {
        symbol: data.get("per_lot_round_trip", 0.0)
        for symbol, data in payload.get("symbols", {}).items()
    }


def load_swap_map(path=None):
    """Real measured swap (USD per lot per night), if
    scripts/measure_swap.py has been run against live deal history. Returns
    {} if not -- there is deliberately no CLASS_SWAP-style fallback table:
    unlike commission, swap is direction-dependent (long vs short are often
    charged/credited very differently -- see the FTMO example below) and
    varies enough by broker/symbol/week that a guessed number would be
    worse than treating it as zero and telling you to measure it.

    Expected format per symbol: {"long": usd_per_lot_per_night,
    "short": usd_per_lot_per_night}. A bare number is also accepted for
    backward compatibility and applied to both directions.
    """
    p = Path(path) if path else SWAP_MAP_PATH
    if not p.exists():
        return {}
    try:
        with p.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return {}
    return payload.get("symbols", {})


def spread_cost_price(symbol, spread_points, fallback_price=None):
    """Round-trip spread cost in PRICE units (not R, not USD) for one trade.

    Prefers the real per-bar 'spread' column MT5 already exports. Falls back
    to fallback_price (a flat price-distance you supply) only if the column
    is missing, zero, or NaN for that bar.
    """
    point = POINT_SIZE.get((symbol or "").upper(), DEFAULT_POINT_SIZE)
    if spread_points is not None and pd.notna(spread_points) and spread_points > 0:
        return float(spread_points) * point
    return float(fallback_price) if fallback_price is not None else 0.0


def spread_cost_usd(symbol, lots, entry_price, spread_points, fallback_price=None):
    """Spread cost converted to USD, given a position size in lots."""
    contract_size, needs_usd_conversion = CONTRACT_SIZE.get((symbol or "").upper(), DEFAULT_CONTRACT)
    price_cost = spread_cost_price(symbol, spread_points, fallback_price)
    raw = price_cost * contract_size * lots
    if needs_usd_conversion and entry_price:
        return raw / entry_price
    return raw


def commission_cost_usd(symbol, lots, commission_map=None, commission_per_lot=None):
    """Round-trip commission in USD for a given lot size.

    Priority: explicit commission_per_lot override > measured commission_map
    > CLASS_COMMISSION fallback by asset class.
    """
    symbol_u = (symbol or "").upper()
    if commission_per_lot is not None:
        per_lot = commission_per_lot
    elif commission_map and symbol_u in commission_map:
        per_lot = commission_map[symbol_u]
    else:
        per_lot = CLASS_COMMISSION.get(classify(symbol_u), CLASS_COMMISSION["fx"])
    return per_lot * lots


# Illustrative only -- NOT verified against your account, and not applied
# unless swap_map has nothing for the symbol. From FTMO's own published
# example explaining swap mechanics: EURUSD long -6.25 points/lot, short
# -3.00 points/lot. Swap rates are set per-broker and change weekly, so
# this exists only so a demo run isn't silently swap-free -- measure your
# own via scripts/measure_swap.py once the VPS is back up.
SWAP_FALLBACK_USD_PER_LOT_PER_NIGHT = {
    "EURUSD": {"long": -6.25 * 0.00001 * 100_000, "short": -3.00 * 0.00001 * 100_000},
}


def swap_cost_usd(symbol, lots, nights_held, swap_map=None, swap_per_lot_per_night=None, direction=1):
    """Overnight financing cost in USD, SIGNED (negative = cost, positive =
    credit -- brokers occasionally pay swap on one side of a pair).

    nights_held: number of broker rollovers (see nights_held() below) the
    position was open across. 0 for anything closed same session.

    direction: 1 for long, -1 for short -- swap is very often asymmetric
    between the two (e.g. the FTMO EURUSD example: -6.25 pts long vs -3.00
    pts short), so this is not optional if you want a realistic number.

    Priority: explicit swap_per_lot_per_night override (applied regardless
    of direction) > measured swap_map entry for this symbol/direction >
    the single illustrative EURUSD fallback above > 0.0.
    """
    if nights_held <= 0:
        return 0.0
    symbol_u = (symbol or "").upper()
    side = "long" if direction == 1 else "short"

    if swap_per_lot_per_night is not None:
        per_lot_night = swap_per_lot_per_night
    elif swap_map and symbol_u in swap_map:
        entry = swap_map[symbol_u]
        per_lot_night = entry.get(side, 0.0) if isinstance(entry, dict) else entry
    elif symbol_u in SWAP_FALLBACK_USD_PER_LOT_PER_NIGHT:
        per_lot_night = SWAP_FALLBACK_USD_PER_LOT_PER_NIGHT[symbol_u].get(side, 0.0)
    else:
        per_lot_night = 0.0

    return per_lot_night * lots * nights_held


def nights_held(entry_time, exit_time, rollover_hour_utc=21, triple_wednesday=True):
    """Count broker rollovers crossed between entry and exit.

    Approximates MT5/FX convention: swap applied once per calendar day at
    ~21:00-22:00 UTC (broker-dependent -- confirm yours), tripled on
    Wednesday to account for weekend settlement. This is a simplification
    (exact rollover time varies by broker and DST); good enough to flag
    "held overnight, cost real swap" vs "closed same session, zero swap",
    which is the distinction that actually matters for backtest realism.
    """
    if entry_time is None or exit_time is None:
        return 0

    entry_time = pd.Timestamp(entry_time)
    exit_time = pd.Timestamp(exit_time)

    count = 0
    cursor = entry_time.normalize() + pd.Timedelta(hours=rollover_hour_utc)
    if cursor <= entry_time:
        cursor += pd.Timedelta(days=1)

    while cursor <= exit_time:
        multiplier = 3 if (triple_wednesday and cursor.weekday() == 2) else 1
        count += multiplier
        cursor += pd.Timedelta(days=1)

    return count


def approx_lots(symbol, risk_usd, stop_distance_price, entry_price):
    """Back out an approximate lot size from a risk-based position (the
    Backtester sizes positions by R, not by lots). Only used to price
    commission; not used for PnL itself. Same simplification
    backtest_live_logic.py uses for its own sizing."""
    contract_size, needs_usd_conversion = CONTRACT_SIZE.get((symbol or "").upper(), DEFAULT_CONTRACT)
    if stop_distance_price <= 0 or contract_size <= 0:
        return 0.0
    loss_per_lot = stop_distance_price * contract_size
    if needs_usd_conversion and entry_price:
        loss_per_lot = loss_per_lot / entry_price
    if loss_per_lot <= 0:
        return 0.0
    return risk_usd / loss_per_lot
