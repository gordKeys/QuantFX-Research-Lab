"""
================================================================================
  VX PROP — COST-INCLUSIVE BACKTEST + WALK-FORWARD HARNESS
  A faithful, non-peeking test rig for the strategy in ftmo.py.

  It reproduces the LIVE bot's exact signal + risk logic and runs it over
  history with realistic frictions so you can see how effective it really is:

    • SPREAD        — full spread paid per round trip (per-bar if MT5 gives it,
                      else a per-symbol default), charged as an explicit line
    • COMMISSION    — $/lot/side, round-trip, explicit line
    • SWAP          — optional overnight financing per lot per night, explicit
    • SLIPPAGE      — optional, points on entry+exit, explicit

  Every trade records GROSS pnl (price only) AND each cost component AND NET,
  so the summary shows exactly what the frictions removed. A NET equity curve
  is written to CSV.

  THREE MODES
    backtest    one continuous run over all history (raw net equity of the
                strategy; FTMO halts OFF by default so you see the whole curve)
    challenge   repeated fresh FTMO windows (equity reset each window, halts ON,
                stop on +target / -total). Re-tests the "14/14 passed" claim
                UNDER costs.
    walkforward rolling out-of-sample. --optimize picks params per symbol on
                in-sample and scores OOS only (honest edge estimate).

  DATA
    Primary   : fetched live from MetaTrader5 (M5 + H1 per symbol).
    Cache/CSV : every fetch is cached to <cache>/SYMBOL_TF.csv and reused, so
                you can develop off-MT5 (e.g. on the Mac) after one fetch.
    Self-test : --selftest runs the whole engine on synthetic random-walk data
                (proves the plumbing; the NUMBERS are meaningless for edge).

  RUN
    python ftmo_backtest_wf.py --selftest
    python ftmo_backtest_wf.py backtest    --from 2023-01-01 --to 2026-07-01
    python ftmo_backtest_wf.py challenge    --challenge-days 14
    python ftmo_backtest_wf.py walkforward  --folds 5 --optimize
================================================================================
"""

from __future__ import annotations
import argparse
import itertools
import math
import sys
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

UTC = timezone.utc

# ════════════════════════════════════════════════════════════════════════════
#  STRATEGY PARAMS  — copied verbatim from ftmo.py so this tests THAT bot
# ════════════════════════════════════════════════════════════════════════════
SYM_CFG = {
    "EURUSD": {"rr": 1.5, "atr": 0.8, "min_score": 4, "fast": 5, "slow": 13, "rsi_p": 9, "ob": 68, "os": 32, "tf": "M5", "trend_tf": "H1", "is_crypto": False},
    "GBPUSD": {"rr": 1.5, "atr": 0.8, "min_score": 3, "fast": 5, "slow": 13, "rsi_p": 9, "ob": 68, "os": 32, "tf": "M5", "trend_tf": "H1", "is_crypto": False},
    "XAUUSD": {"rr": 1.5, "atr": 0.8, "min_score": 3, "fast": 5, "slow": 13, "rsi_p": 9, "ob": 68, "os": 32, "tf": "M5", "trend_tf": "H1", "is_crypto": False},
    # "ETHUSD": {"rr": 1.0, "atr": 1.2, "min_score": 3, "fast": 9, "slow": 21, "rsi_p": 14, "ob": 72, "os": 28, "tf": "M5", "trend_tf": "H4", "is_crypto": True},
}
SYMBOLS = list(SYM_CFG.keys())

VOL_FACTOR = 1.05
H_EMA = 20
FOREX_HOURS = set(range(8, 22))          # entries only 08:00–21:59 UTC for forex
STOP_BUFFER_MULT = 1.5
MIN_ATR_POINTS = 5

# ════════════════════════════════════════════════════════════════════════════
#  ACCOUNT / FTMO PARAMS  — match your challenge
# ════════════════════════════════════════════════════════════════════════════
START_EQUITY = 100_000.0
PROFIT_TARGET = 10.0     # % (Phase 1). Set 5 for Phase 2.
DAILY_HALT_PCT = 4.0     # self-halt buffer (FTMO daily fail = 5%)
TOTAL_HALT_PCT = 8.0     # self-halt buffer (FTMO total fail = 10%)
RISK_PCT = 0.4           # fixed risk per trade — never scales
MAX_OPEN_TRADES = 3

TF_SECONDS = {"M5": 300, "M15": 900, "H1": 3600, "H4": 14400, "D1": 86400}


# ════════════════════════════════════════════════════════════════════════════
#  INSTRUMENT SPECS + COST MODEL   (defaults; override from your MT5 symbol_info)
#
#  value_per_point = account-currency value of a 1-point price move for 1.0 lot.
#    EURUSD/GBPUSD (5-digit, point 0.00001): $1.0/point/lot  (=$10 per pip)
#    XAUUSD        (2-digit, point 0.01):     $1.0/point/lot  (=$100 per $1 move)
#  spread_points   = fallback spread if the M5 feed has no per-bar spread column.
#  commission      = $ per lot per side. FTMO-style ≈ $3/lot/side (=$6 round trip
#                    per standard lot). You measured ~$2.52/side — SET IT EXACTLY.
#  swap_long/short = account-currency per lot per night (negative = you pay).
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class Spec:
    point: float
    digits: int
    value_per_point: float          # $/point/lot
    spread_points: float            # fallback spread (points)
    volume_min: float = 0.01
    volume_step: float = 0.01
    volume_max: float = 100.0
    trade_stops_level: float = 0.0  # broker min stop distance (points)
    commission_per_lot_side: float = 3.0
    swap_long: float = 0.0
    swap_short: float = 0.0
    is_crypto: bool = False


SPECS = {
    "EURUSD": Spec(point=0.00001, digits=5, value_per_point=1.0, spread_points=8,  commission_per_lot_side=3.0),
    "GBPUSD": Spec(point=0.00001, digits=5, value_per_point=1.0, spread_points=12, commission_per_lot_side=3.0),
    "XAUUSD": Spec(point=0.01,    digits=2, value_per_point=1.0, spread_points=25, commission_per_lot_side=3.0),
    # "ETHUSD": Spec(point=0.01, digits=2, value_per_point=0.01, spread_points=200, commission_per_lot_side=3.0, is_crypto=True),
}


@dataclass
class Config:
    """Everything the engine reads at run time (frictions, halts, fill rules)."""
    start_equity: float = START_EQUITY
    profit_target: float = PROFIT_TARGET
    daily_halt_pct: float = DAILY_HALT_PCT
    total_halt_pct: float = TOTAL_HALT_PCT
    risk_pct: float = RISK_PCT
    max_open_trades: int = MAX_OPEN_TRADES

    # frictions
    costs_on: bool = True
    use_bar_spread: bool = True       # use per-bar spread column if present
    slippage_points: float = 0.0      # per side, in points
    rollover_hour_utc: int = 21       # night boundary for swap counting

    # fill realism
    same_bar_rule: str = "sl_first"   # if a bar spans both SL and TP: pessimistic
    forex_hours_only: bool = True

    # halts
    enforce_halts: bool = False       # set True for challenge realism
    stop_on_pass: bool = True
    stop_on_total_halt: bool = True

    # loss / giveback protection (ported from QuantFX live_runner)
    protection_on: bool = False
    protection_mode: str = "full"     # "full" (loss+giveback) or "loss_only"
    protect_ref_risk: float = 20.0    # == REF_RISK_USD (defined later)


# ════════════════════════════════════════════════════════════════════════════
#  INDICATORS  — faithful to add_indicators() in ftmo.py
# ════════════════════════════════════════════════════════════════════════════
def add_indicators(df: pd.DataFrame, scfg: dict) -> pd.DataFrame:
    c = df["close"]; v = df["tick_volume"]
    df["ema_fast"] = c.ewm(span=scfg["fast"], adjust=False).mean()
    df["ema_slow"] = c.ewm(span=scfg["slow"], adjust=False).mean()
    mf = c.ewm(span=12, adjust=False).mean(); ms = c.ewm(span=26, adjust=False).mean()
    df["macd"] = mf - ms
    df["macd_sig"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_h"] = df["macd"] - df["macd_sig"]
    rp = scfg["rsi_p"]; delta = c.diff()
    ag = delta.clip(lower=0).ewm(alpha=1 / rp, min_periods=rp, adjust=False).mean()
    al = (-delta.clip(upper=0)).ewm(alpha=1 / rp, min_periods=rp, adjust=False).mean()
    df["rsi"] = 100 - (100 / (1 + ag / al))
    df["bb_mid"] = c.rolling(20).mean(); bb_std = c.rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * bb_std
    df["bb_lower"] = df["bb_mid"] - 2 * bb_std
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - c.shift()).abs(),
                    (df["low"] - c.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    df["vol_avg"] = v.rolling(20).mean(); df["vol"] = v
    return df


# ── candle patterns, vectorised (same definitions as ftmo.py) ────────────────
def _bull_engulf(o, h, l, c, po, ph, pl, pc):
    return (pc < po) & (c > o) & (o < pc) & (c > po)


def _bear_engulf(o, h, l, c, po, ph, pl, pc):
    return (pc > po) & (c < o) & (o > pc) & (c < po)


def _bull_pin(o, h, l, c):
    body = np.abs(c - o)
    lower_wick = np.minimum(c, o) - l
    upper_wick = h - np.maximum(c, o)
    return (body > 0) & (lower_wick >= 2 * body) & (upper_wick <= body)


def _bear_pin(o, h, l, c):
    body = np.abs(c - o)
    upper_wick = h - np.maximum(c, o)
    lower_wick = np.minimum(c, o) - l
    return (body > 0) & (upper_wick >= 2 * body) & (lower_wick <= body)


# ════════════════════════════════════════════════════════════════════════════
#  SIGNAL SCORING  — vectorised port of score_signals() + the entry decision
#  Signal is computed on CLOSED bar i (bar i and i-1). Action happens at the
#  OPEN of bar i+1 (no look-ahead). Trend uses the last CLOSED trend bar.
# ════════════════════════════════════════════════════════════════════════════
def compute_signals(df: pd.DataFrame, trend: pd.Series, scfg: dict) -> pd.DataFrame:
    o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
    po, ph, pl, pc = _shift1(o), _shift1(h), _shift1(l), _shift1(c)

    ema_f, ema_s = df["ema_fast"].values, df["ema_slow"].values
    p_ema_f, p_ema_s = _shift1(ema_f), _shift1(ema_s)
    macd_h = df["macd_h"].values; p_macd_h = _shift1(macd_h)
    rsi = df["rsi"].values
    bb_u, bb_l = df["bb_upper"].values, df["bb_lower"].values
    vol, vol_avg = df["vol"].values, df["vol_avg"].values

    vol_ok = np.where(vol_avg > 0, vol >= vol_avg * VOL_FACTOR, False)

    buy = (
        ((p_ema_f <= p_ema_s) & (ema_f > ema_s)).astype(int)                       # ema cross up
        + ((macd_h > 0) & (macd_h > p_macd_h)).astype(int)                         # macd rising >0
        + ((c <= bb_l * 1.001) & (rsi < scfg["ob"])).astype(int)                   # lower band + rsi
        + (_bull_engulf(o, h, l, c, po, ph, pl, pc) | _bull_pin(o, h, l, c)).astype(int)
        + vol_ok.astype(int)
    )
    sell = (
        ((p_ema_f >= p_ema_s) & (ema_f < ema_s)).astype(int)
        + ((macd_h < 0) & (macd_h < p_macd_h)).astype(int)
        + ((c >= bb_u * 0.999) & (rsi > scfg["os"])).astype(int)
        + (_bear_engulf(o, h, l, c, po, ph, pl, pc) | _bear_pin(o, h, l, c)).astype(int)
        + vol_ok.astype(int)
    )

    # trend gate: DOWN kills buys, UP kills sells (unknown trend -> no gate)
    tr = trend.values  # 'UP' / 'DOWN' / None aligned to df.index
    buy = np.where(tr == "DOWN", 0, buy)
    sell = np.where(tr == "UP", 0, sell)

    ms = scfg["min_score"]
    direction = np.where((buy >= ms) & (buy >= sell), "BUY",
                np.where((sell >= ms) & (sell > buy), "SELL", None))

    out = pd.DataFrame(index=df.index)
    out["direction"] = direction
    return out


def _shift1(a: np.ndarray) -> np.ndarray:
    s = np.empty_like(a, dtype=float)
    s[0] = np.nan
    s[1:] = a[:-1]
    return s


# ════════════════════════════════════════════════════════════════════════════
#  STOP DISTANCE + LOT SIZE  — faithful to compute_stop_distance()/get_lot()
# ════════════════════════════════════════════════════════════════════════════
def stop_distance_series(df: pd.DataFrame, spread_pts: np.ndarray, spec: Spec, scfg: dict) -> np.ndarray:
    atr = df["atr"].values
    point = spec.point
    atr_sl = scfg["atr"] * atr
    floor = np.maximum(spec.trade_stops_level, spread_pts) * STOP_BUFFER_MULT * point
    dist = np.maximum(atr_sl, floor)
    # invalidate: atr NaN or atr/point below MIN_ATR_POINTS
    bad = np.isnan(atr) | ((atr / point) < MIN_ATR_POINTS)
    dist = np.where(bad, np.nan, dist)
    return dist


def compute_lot(equity: float, stop_dist: float, spec: Spec, cfg: Config) -> float:
    if stop_dist <= 0:
        return 0.0
    risk_amt = equity * (cfg.risk_pct / 100.0)
    sl_points = stop_dist / spec.point
    if sl_points <= 0 or spec.value_per_point <= 0:
        return spec.volume_min
    lot = risk_amt / (sl_points * spec.value_per_point)
    lot = max(spec.volume_min, min(lot, spec.volume_max))
    lot = round(round(lot / spec.volume_step) * spec.volume_step, 2)
    return lot


# ════════════════════════════════════════════════════════════════════════════
#  LOSS / GIVEBACK PROTECTION  — ported faithfully from the QuantFX-Research-Lab
#  live_runner.manage_live_position() + trade_management_params().
#
#  Per bar, while a position is open, this can: close on a dollar loss stop
#  (soft/warn/hard), quick-cut a never-profitable trade, time-stop, close on an
#  R-based or dollar-based GIVEBACK from the peak (so winners don't round-trip
#  to losers), or move the SL to breakeven / trail it. Per-symbol tiers below.
#
#  Dollar thresholds were tuned around a reference risk (REF_RISK_USD). This
#  backtest may size differently, so each trade scales those thresholds by
#  (trade_risk_$ / REF_RISK_USD) — preserving the tuned RELATIONSHIPS while
#  staying correct under whatever sizing the run uses. R-based tiers need no
#  scaling. Note (matches the live code): breakeven collapses risk to 0, after
#  which R-tiers go dormant and the DOLLAR giveback/stops do the protecting.
# ════════════════════════════════════════════════════════════════════════════
REF_RISK_USD = 20.0   # risk-per-trade the live dollar tiers were tuned around


def _normalize_symbol(symbol):
    if not symbol:
        return symbol
    s = symbol.upper()
    for suf in (".R", ".CASH", ".PRO", "M", ".A", ".C", "_SB", ".RAW"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s


def trade_management_params(symbol=None):
    base = {
        "breakeven_at_r": 1.00, "trail_at_r": 1.60, "trail_buffer_r": 0.65,
        "giveback_trigger_r": 1.40, "giveback_buffer_r": 0.50,
        "min_peak_profit_usd": 4.0, "giveback_usd_buffer": 2.0,
        "max_minutes": 180, "max_bars": 48,
        "warn_loss_per_trade_usd": 11.0, "soft_loss_per_trade_usd": 13.5,
        "max_loss_per_trade_usd": 15.0, "quick_cut_minutes": 25.0, "quick_cut_loss_usd": 7.0,
    }
    symbol = _normalize_symbol(symbol)
    if symbol == "EURUSD":
        base.update({"breakeven_at_r": 0.45, "trail_at_r": 0.80, "trail_buffer_r": 0.28,
                     "giveback_trigger_r": 0.72, "giveback_buffer_r": 0.36,
                     "min_peak_profit_usd": 2.0, "giveback_usd_buffer": 2.0,
                     "max_minutes": 120, "max_bars": 28, "quick_cut_minutes": 20.0, "quick_cut_loss_usd": 5.0})
    elif symbol == "USDCHF":
        base.update({"breakeven_at_r": 0.40, "trail_at_r": 0.75, "trail_buffer_r": 0.25,
                     "giveback_trigger_r": 0.65, "giveback_buffer_r": 0.16,
                     "min_peak_profit_usd": 2.0, "giveback_usd_buffer": 1.0,
                     "max_minutes": 100, "max_bars": 24, "quick_cut_minutes": 20.0, "quick_cut_loss_usd": 5.0})
    elif symbol == "USDJPY":
        base.update({"breakeven_at_r": 0.80, "trail_at_r": 1.45, "trail_buffer_r": 0.58,
                     "giveback_trigger_r": 1.20, "giveback_buffer_r": 0.30,
                     "min_peak_profit_usd": 3.0, "giveback_usd_buffer": 1.5,
                     "max_minutes": 150, "max_bars": 36, "quick_cut_minutes": 30.0, "quick_cut_loss_usd": 7.0})
    elif symbol == "AUDUSD":
        base.update({"breakeven_at_r": 0.70, "trail_at_r": 1.35, "trail_buffer_r": 0.52,
                     "giveback_trigger_r": 1.05, "giveback_buffer_r": 0.22,
                     "min_peak_profit_usd": 3.0, "giveback_usd_buffer": 1.5,
                     "max_minutes": 165, "max_bars": 40, "quick_cut_minutes": 30.0, "quick_cut_loss_usd": 7.0})
    elif symbol == "XAUUSD":
        base.update({"breakeven_at_r": 0.90, "trail_at_r": 1.55, "trail_buffer_r": 0.60,
                     "giveback_trigger_r": 1.30, "giveback_buffer_r": 0.32,
                     "min_peak_profit_usd": 3.0, "giveback_usd_buffer": 1.5,
                     "max_minutes": 180, "max_bars": 44, "quick_cut_minutes": 35.0, "quick_cut_loss_usd": 7.0})
    return base


def manage_position(pos, price, held_minutes, mgmt, scale, mode="full"):
    """Faithful port of manage_live_position, in backtest terms.
    pos carries: dir(+1/-1), entry, sl (mutable), peak_r, peak_profit_usd,
    pnl_per_price (=$ per 1.0 price unit for this lot). Returns (action, payload):
    ("close", reason) | ("modify_sl", new_sl) | ("hold", None).
    mode: "full" = loss stops + giveback + breakeven/trail (everything);
          "loss_only" = loss stops + quick-cut + time stops ONLY (no giveback,
          no breakeven, no trail — winners run to the structural TP/SL)."""
    is_buy = pos["sign"] == 1
    pnl_price = (price - pos["entry"]) if is_buy else (pos["entry"] - price)
    pnl_usd = pnl_price * pos["pnl_per_price"]
    risk_price = abs(pos["entry"] - pos["sl"])
    open_r = (pnl_price / risk_price) if risk_price > 0 else None   # None after breakeven

    # peaks (persist on pos)
    if open_r is not None:
        pos["peak_r"] = max(pos["peak_r"], open_r)
    pos["peak_profit_usd"] = max(pos["peak_profit_usd"], pnl_usd)
    peak_r = pos["peak_r"]
    peak_profit_usd = pos["peak_profit_usd"]
    giveback_r = (peak_r - open_r) if open_r is not None else 0.0

    # scaled dollar thresholds
    soft = mgmt["soft_loss_per_trade_usd"] * scale
    warn = mgmt["warn_loss_per_trade_usd"] * scale
    hard = mgmt["max_loss_per_trade_usd"] * scale
    min_peak = mgmt["min_peak_profit_usd"] * scale
    gb_usd = mgmt["giveback_usd_buffer"] * scale
    qc_loss = mgmt["quick_cut_loss_usd"] * scale

    # ═══ LOSS PROTECTION (both configs) ═══
    # ---- dollar loss stops ----
    if pnl_usd <= -soft:
        return "close", "soft_dollar_stop"
    if pnl_usd <= -warn:
        return "close", "warn_dollar_stop"
    if pnl_usd <= -hard:
        return "close", "hard_dollar_stop"

    # ---- quick cut: never-profitable trade bleeding past a window ----
    if (mgmt["quick_cut_minutes"] > 0 and qc_loss > 0 and peak_profit_usd <= 0
            and held_minutes >= mgmt["quick_cut_minutes"] and pnl_usd <= -qc_loss):
        return "close", "quick_cut_never_profitable"

    # ---- time-based cuts ----
    if held_minutes >= mgmt["max_minutes"] and open_r is not None and open_r <= -0.18:
        return "close", "loss_cut"
    if held_minutes >= mgmt["max_bars"] * 5 and open_r is not None and open_r < 0:
        return "close", "time_stop"

    # ═══ GIVEBACK / PROFIT PROTECTION (full config only) ═══
    if mode == "loss_only":
        return "hold", None

    # ---- R-based giveback ----
    if (open_r is not None and peak_r >= mgmt["giveback_trigger_r"]
            and pnl_usd >= min_peak and giveback_r >= mgmt["giveback_buffer_r"]):
        return "close", "profit_giveback_close"

    # ---- dollar-based giveback safety net (works after breakeven) ----
    if (gb_usd > 0 and peak_profit_usd >= min_peak
            and (peak_profit_usd - pnl_usd) >= gb_usd):
        return "close", "profit_giveback_close_usd"

    # ---- breakeven + trail (SL moves) ----
    new_sl = pos["sl"]
    if open_r is not None:
        if open_r >= mgmt["breakeven_at_r"]:
            new_sl = max(new_sl, pos["entry"]) if is_buy else min(new_sl, pos["entry"])
        if open_r >= mgmt["trail_at_r"]:
            td = risk_price * mgmt["trail_buffer_r"]
            new_sl = max(new_sl, price - td) if is_buy else min(new_sl, price + td)
    if new_sl != pos["sl"]:
        return "modify_sl", new_sl
    return "hold", None


def _normalise_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    tcol = next((x for x in ("time", "datetime", "date", "timestamp") if x in df.columns), None)
    if tcol is None:
        raise ValueError(f"no time column in {list(df.columns)}")
    t = df[tcol]
    if pd.api.types.is_numeric_dtype(t):
        df["time"] = pd.to_datetime(t, unit="s", utc=True)      # MT5 epoch seconds
    else:
        df["time"] = pd.to_datetime(t, utc=True, errors="coerce")  # cached ISO strings
    for col in ("open", "high", "low", "close", "tick_volume", "spread"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "tick_volume" not in df.columns:
        for alt in ("volume", "tickvol", "real_volume"):
            if alt in df.columns:
                df["tick_volume"] = df[alt]; break
        else:
            df["tick_volume"] = 1.0
    if "spread" not in df.columns:
        df["spread"] = np.nan
    keep = ["time", "open", "high", "low", "close", "tick_volume", "spread"]
    df = df[keep].dropna(subset=["open", "high", "low", "close"])
    return df.set_index("time").sort_index()


def load_csv(path: Path) -> pd.DataFrame:
    return _normalise_ohlc(pd.read_csv(path))


# ── MT5 session (connect once, reuse) ────────────────────────────────────────
_MT5 = {"init": False, "mod": None}


def mt5_connect(login=None, password=None, server=None, terminal=None):
    if _MT5["init"]:
        return _MT5["mod"]
    import MetaTrader5 as mt5      # only importable on the Windows box with MT5
    kwargs = {}
    if terminal:
        kwargs["path"] = terminal
    if not mt5.initialize(**kwargs):
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    if login:
        if not mt5.login(int(login), password=password, server=server):
            raise RuntimeError(f"MT5 login failed: {mt5.last_error()}")
    info = mt5.account_info()
    if info:
        print(f"  MT5 connected: {info.login} @ {info.server}  bal ${info.balance:,.2f}")
    _MT5["init"] = True
    _MT5["mod"] = mt5
    return mt5


def mt5_disconnect():
    if _MT5["init"] and _MT5["mod"] is not None:
        try:
            _MT5["mod"].shutdown()
        except Exception:
            pass
        _MT5["init"] = False


_BARS_PER_DAY = {"M5": 288, "M15": 96, "H1": 24, "H4": 6, "D1": 1}
_DATA_MEM: dict = {}   # in-memory cache so walk-forward --optimize doesn't re-read CSVs


def _mt5_history(mt5, symbol: str, tf: str, dt_from: datetime, dt_to: datetime) -> pd.DataFrame:
    """Robust pull: ensure the symbol is live, try range, then fall back to
    position-based reads (which force the terminal to download history)."""
    tf_map = {"M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
              "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4, "D1": mt5.TIMEFRAME_D1}
    tfc = tf_map[tf]

    if mt5.symbol_info(symbol) is None:
        raise RuntimeError(
            f"{symbol} not found in this terminal. Check the exact name in Market "
            f"Watch (some brokers use suffixes like {symbol}.r / {symbol}m).")
    mt5.symbol_select(symbol, True)
    for _ in range(12):                      # wait for the symbol to go live
        si = mt5.symbol_info(symbol)
        if si and si.visible:
            break
        time.sleep(0.3)

    from_ts = pd.Timestamp(dt_from, tz="UTC")
    errors = []

    def _to_df(frames):
        good = [pd.DataFrame(r) for r in frames if r is not None and len(r)]
        if not good:
            return None
        df = _normalise_ohlc(pd.concat(good, ignore_index=True))
        return df[~df.index.duplicated(keep="last")].sort_index()

    # ---- Strategy A: page copy_rates_from_pos in modest chunks (avoids the
    #      giant-count "invalid params" rejection). Pages newest -> oldest. ----
    for chunk in (50_000, 20_000, 5_000):
        frames = []
        start = 0
        for _page in range(400):
            r = mt5.copy_rates_from_pos(symbol, tfc, int(start), int(chunk))
            if r is None or len(r) == 0:
                if start == 0:
                    errors.append(f"from_pos(chunk={chunk}): {mt5.last_error()}")
                break
            frames.append(r)
            oldest = pd.Timestamp(int(r["time"][0]), unit="s", tz="UTC")
            start += chunk
            if oldest <= from_ts:            # paged back far enough
                break
            if len(r) < chunk:               # ran out of history
                break
        df = _to_df(frames)
        if df is not None and len(df):
            return df

    # ---- Strategy B: copy_rates_range in 30-day chunks (naive UTC datetimes) ----
    for tz_aware in (False, True):
        frames = []
        cur = pd.Timestamp(dt_from, tz="UTC")
        end = pd.Timestamp(dt_to, tz="UTC")
        step = pd.Timedelta(days=30)
        while cur < end:
            nxt = min(cur + step, end)
            a = cur.to_pydatetime() if tz_aware else cur.tz_localize(None).to_pydatetime()
            b = nxt.to_pydatetime() if tz_aware else nxt.tz_localize(None).to_pydatetime()
            try:
                r = mt5.copy_rates_range(symbol, tfc, a, b)
                if r is not None and len(r):
                    frames.append(r)
            except Exception as e:
                errors.append(f"range(tz={tz_aware}): {e}")
            cur = nxt
        df = _to_df(frames)
        if df is not None and len(df):
            return df
        errors.append(f"range(tz={tz_aware}) empty: {mt5.last_error()}")

    raise RuntimeError(
        f"MT5 returned no {tf} bars for {symbol}. Tried paged from_pos + chunked "
        f"range. Errors: {errors}. In the terminal, open a {symbol} {tf} chart and "
        f"scroll left once so history downloads, then retry (or pass --refresh).")


def get_data(symbol: str, tf: str, dt_from: datetime, dt_to: datetime, cache_dir: Path,
             refresh: bool = False, mt5_kwargs: dict | None = None) -> pd.DataFrame:
    """Cache-first data access. Uses <cache>/SYMBOL_TF.csv when present unless
    --refresh; otherwise pulls from MT5 and writes the cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{symbol}_{tf}.csv"

    memkey = (symbol, tf, str(dt_from), str(dt_to))
    if not refresh and memkey in _DATA_MEM:
        return _DATA_MEM[memkey]

    if cache.exists() and not refresh:
        df = load_csv(cache)
        sl = _slice(df, dt_from, dt_to)
        cov = "" if len(sl) else "  (cache has no bars in this range — try --refresh)"
        print(f"  cache {cache.name}: {len(df):,} bars, {len(sl):,} in range{cov}")
        _DATA_MEM[memkey] = sl
        return sl

    mt5 = mt5_connect(**(mt5_kwargs or {}))
    df = _mt5_history(mt5, symbol, tf, dt_from, dt_to)
    df.reset_index().to_csv(cache, index=False)
    span = f"{df.index.min()} -> {df.index.max()}"
    print(f"  fetched {symbol} {tf}: {len(df):,} bars ({span}) -> cached {cache.name}")

    sl = _slice(df, dt_from, dt_to)
    if len(sl) == 0:
        raise RuntimeError(
            f"{symbol} {tf}: MT5 gave {len(df):,} bars but none between {dt_from.date()} "
            f"and {dt_to.date()}. Available: {span}. Adjust --from/--to to that span.")
    if df.index.min() > pd.Timestamp(dt_from, tz='UTC'):
        print(f"    note: history starts {df.index.min().date()}, later than --from "
              f"{dt_from.date()} — using what's available.")
    _DATA_MEM[memkey] = sl
    return sl


def _slice(df: pd.DataFrame, dt_from, dt_to) -> pd.DataFrame:
    return df.loc[(df.index >= pd.Timestamp(dt_from, tz="UTC")) &
                  (df.index < pd.Timestamp(dt_to, tz="UTC"))]


# ════════════════════════════════════════════════════════════════════════════
#  PREPARE  — one symbol -> numpy arrays the engine iterates over
#  Everything actionable is SHIFTED to the bar where the trade would open.
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class SymData:
    symbol: str
    scfg: dict
    spec: Spec
    times: pd.DatetimeIndex
    o: np.ndarray; h: np.ndarray; l: np.ndarray; c: np.ndarray
    direction: np.ndarray      # actionable at THIS bar's open (from prior bar)
    stop_dist: np.ndarray      # actionable stop distance (price units)
    spread_pts: np.ndarray     # per-bar spread in points (for cost)
    is_crypto: bool


def prepare_symbol(symbol: str, m5: pd.DataFrame, trend_df: pd.DataFrame,
                   scfg: dict, spec: Spec, cfg: Config) -> SymData:
    m5 = add_indicators(m5.copy(), scfg)

    # trend on the higher timeframe, attached to each trend bar's CLOSE time
    tdelta = pd.Timedelta(seconds=TF_SECONDS[scfg["trend_tf"]])
    t_ema = trend_df["close"].ewm(span=H_EMA, adjust=False).mean()
    t_up = np.where(trend_df["close"].values > t_ema.values, "UP", "DOWN")
    trend_ser = pd.Series(t_up, index=trend_df.index + tdelta)
    trend_ser = trend_ser[~trend_ser.index.duplicated(keep="last")].sort_index()

    # trend known at decision time = m5 bar close (= bar open + M5)
    m5delta = pd.Timedelta(seconds=TF_SECONDS[scfg["tf"]])
    decide_time = m5.index + m5delta
    trend_at_signal = pd.Series(
        trend_ser.reindex(decide_time, method="ffill").values, index=m5.index)

    # per-bar spread in points
    if cfg.use_bar_spread and "spread" in m5.columns and m5["spread"].notna().any():
        spread_pts = m5["spread"].ffill().fillna(spec.spread_points).values.astype(float)
    else:
        spread_pts = np.full(len(m5), spec.spread_points, dtype=float)

    sig = compute_signals(m5, trend_at_signal, scfg)
    stop_dist = stop_distance_series(m5, spread_pts, spec, scfg)

    # SHIFT: decision & stop computed on bar i become actionable at bar i+1 open
    direction_act = np.empty(len(m5), dtype=object)
    direction_act[0] = None
    direction_act[1:] = sig["direction"].values[:-1]
    stop_act = _shift1(stop_dist)

    return SymData(
        symbol=symbol, scfg=scfg, spec=spec, times=m5.index,
        o=m5["open"].values, h=m5["high"].values, l=m5["low"].values, c=m5["close"].values,
        direction=direction_act, stop_dist=stop_act, spread_pts=spread_pts,
        is_crypto=scfg["is_crypto"],
    )


# ════════════════════════════════════════════════════════════════════════════
#  ENGINE  — portfolio, event-driven over the union M5 clock. One position per
#  symbol (like the live bot). Exits checked before entries each timestamp.
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class Trade:
    symbol: str; direction: str
    entry_time: pd.Timestamp; exit_time: pd.Timestamp
    entry: float; exit: float; lot: float
    reason: str
    gross: float; spread_cost: float; commission: float; swap: float; slippage: float
    net: float
    r_multiple: float


@dataclass
class RunResult:
    trades: list
    eq_times: list
    eq_vals: list
    passed: bool
    failed_total: bool
    worst_daily_dd: float
    worst_total_dd: float
    end_equity: float


class Portfolio:
    def __init__(self, syms: dict[str, SymData], union: pd.DatetimeIndex, cfg: Config):
        self.syms = syms
        self.union = union
        self.cfg = cfg
        # per-symbol row lookup by union position, -1 where the symbol has no bar
        self.row_of_u = {}
        for s, sd in syms.items():
            r = np.full(len(union), -1, dtype=np.int64)
            pos = union.get_indexer(sd.times)
            good = pos >= 0
            r[pos[good]] = np.arange(len(sd.times))[good]
            self.row_of_u[s] = r
        self.hours = union.hour.values
        ns = union.asi8                       # int64 nanoseconds, tz-safe
        self.daynum = ns // (86_400 * 1_000_000_000)
        self.rollnum = (ns // 1_000_000_000 - cfg.rollover_hour_utc * 3600) // 86_400
        self.mgmt = {s: trade_management_params(s) for s in syms}   # protection tiers

    # ---- cost helpers -------------------------------------------------------
    def _nights(self, u_in: int, u_out: int) -> int:
        return int(self.rollnum[u_out] - self.rollnum[u_in])

    def _close(self, sd: SymData, pos: dict, exit_price: float, reason: str,
               u_in: int, u_out: int) -> Trade:
        spec = sd.spec; cfg = self.cfg
        sgn = 1.0 if pos["dir"] == "BUY" else -1.0
        gross = sgn * (exit_price - pos["entry"]) / spec.point * spec.value_per_point * pos["lot"]

        if cfg.costs_on:
            sp_in = sd.spread_pts[self.row_of_u[sd.symbol][u_in]]
            sp_out = sd.spread_pts[self.row_of_u[sd.symbol][u_out]]
            spread_pts = 0.5 * (sp_in + sp_out)                     # full spread / round trip
            spread_cost = spread_pts * spec.value_per_point * pos["lot"]
            commission = 2 * spec.commission_per_lot_side * pos["lot"]
            slippage = 2 * cfg.slippage_points * spec.value_per_point * pos["lot"]
            nights = self._nights(u_in, u_out)
            swap_rate = spec.swap_long if pos["dir"] == "BUY" else spec.swap_short
            swap = nights * swap_rate * pos["lot"]                  # signed (income if +)
        else:
            spread_cost = commission = slippage = swap = 0.0

        net = gross - spread_cost - commission - slippage + swap
        risk_cash = pos["risk_cash"]
        r_mult = net / risk_cash if risk_cash > 0 else 0.0
        return Trade(sd.symbol, pos["dir"], pos["entry_time"], self.union[u_out],
                     pos["entry"], exit_price, pos["lot"], reason,
                     gross, spread_cost, commission, swap, slippage, net, r_mult)

    def _check_exit(self, pos: dict, hi: float, lo: float):
        cfg = self.cfg
        if pos["dir"] == "BUY":
            hit_sl = lo <= pos["sl"]; hit_tp = hi >= pos["tp"]
            if hit_sl and hit_tp:
                return (pos["sl"], "SL") if cfg.same_bar_rule == "sl_first" else (pos["tp"], "TP")
            if hit_sl: return pos["sl"], "SL"
            if hit_tp: return pos["tp"], "TP"
        else:
            hit_sl = hi >= pos["sl"]; hit_tp = lo <= pos["tp"]
            if hit_sl and hit_tp:
                return (pos["sl"], "SL") if cfg.same_bar_rule == "sl_first" else (pos["tp"], "TP")
            if hit_sl: return pos["sl"], "SL"
            if hit_tp: return pos["tp"], "TP"
        return None, None

    def _sl_reason(self, pos: dict) -> str:
        if pos["sl"] == pos["entry"]:
            return "breakeven_stop"
        if pos["sl"] != pos["orig_sl"]:
            return "trail_stop"
        return "SL"

    def _protect_exit(self, pos: dict, hi: float, lo: float, bar_time):
        """Walk the bar adverse-side-first, applying the ported protection
        manager alongside the (possibly trailed) SL/TP. Returns (price, reason)
        or None; may mutate pos['sl'] via breakeven/trail."""
        cfg = self.cfg
        is_buy = pos["sign"] == 1
        held_min = (bar_time - pos["entry_time"]).total_seconds() / 60.0
        scale = pos["risk_cash"] / cfg.protect_ref_risk if cfg.protect_ref_risk > 0 else 1.0
        for price in ([lo, hi] if is_buy else [hi, lo]):
            # hard SL (broker fill) — checked on the adverse checkpoint first
            if (is_buy and price <= pos["sl"]) or (not is_buy and price >= pos["sl"]):
                return pos["sl"], self._sl_reason(pos)
            # TP
            if (is_buy and price >= pos["tp"]) or (not is_buy and price <= pos["tp"]):
                return pos["tp"], "TP"
            # protection manager (dollar stops / giveback / time / trail)
            action, payload = manage_position(pos, price, held_min, pos["mgmt"],
                                              scale, cfg.protection_mode)
            if action == "close":
                return price, payload
            if action == "modify_sl":
                pos["sl"] = payload
        return None

    # ---- main loop ----------------------------------------------------------
    def run(self, u0: int, u1: int, reset_equity: bool = True) -> RunResult:
        cfg = self.cfg
        balance = cfg.start_equity
        positions: dict[str, dict] = {}
        entry_u: dict[str, int] = {}
        last_close = {s: np.nan for s in self.syms}
        trades: list[Trade] = []
        eq_t: list = []; eq_v: list = []
        day_key = None; day_start_eq = balance
        daily_halted = False; total_halted = False; passed = False
        worst_daily = 0.0; worst_total = 0.0

        def floating() -> float:
            f = 0.0
            for s, p in positions.items():
                lc = last_close[s]
                if not math.isnan(lc):
                    sgn = 1.0 if p["dir"] == "BUY" else -1.0
                    sp = self.syms[s].spec
                    f += sgn * (lc - p["entry"]) / sp.point * sp.value_per_point * p["lot"]
            return f

        for u in range(u0, u1):
            # (1) update last_close + process exits for symbols with a bar here
            for s in list(positions.keys()):
                j = self.row_of_u[s][u]
                if j < 0:
                    continue
                sd = self.syms[s]
                last_close[s] = sd.c[j]
                if cfg.protection_on:
                    res = self._protect_exit(positions[s], sd.h[j], sd.l[j], self.union[u])
                    ex, why = res if res is not None else (None, None)
                else:
                    ex, why = self._check_exit(positions[s], sd.h[j], sd.l[j])
                if ex is not None:
                    tr = self._close(sd, positions[s], ex, why, entry_u[s], u)
                    balance += tr.net; trades.append(tr)
                    del positions[s]; del entry_u[s]
            for s in self.syms:
                j = self.row_of_u[s][u]
                if j >= 0:
                    last_close[s] = self.syms[s].c[j]

            # (2) daily reset
            dk = self.daynum[u]
            if dk != day_key:
                day_key = dk
                day_start_eq = balance + floating()
                daily_halted = False

            # (3) equity, drawdowns, halts
            eq = balance + floating()
            eq_t.append(self.union[u]); eq_v.append(eq)
            total_dd = (cfg.start_equity - eq) / cfg.start_equity * 100
            daily_dd = (day_start_eq - eq) / day_start_eq * 100 if day_start_eq else 0.0
            worst_total = max(worst_total, total_dd); worst_daily = max(worst_daily, daily_dd)
            if (eq - cfg.start_equity) / cfg.start_equity * 100 >= cfg.profit_target:
                passed = True

            if cfg.enforce_halts and total_dd >= cfg.total_halt_pct and not total_halted:
                total_halted = True
                for s in list(positions.keys()):        # flatten at last close
                    sd = self.syms[s]; lc = last_close[s]
                    tr = self._close(sd, positions[s], lc, "TOTAL_HALT", entry_u[s], u)
                    balance += tr.net; trades.append(tr)
                    del positions[s]; del entry_u[s]
            if cfg.enforce_halts and daily_dd >= cfg.daily_halt_pct:
                daily_halted = True

            blocked = (cfg.enforce_halts and (daily_halted or total_halted)) or \
                      (cfg.stop_on_pass and passed)

            # (4) entries
            if not blocked:
                open_count = len(positions)
                for s, sd in self.syms.items():
                    if s in positions:
                        continue
                    if open_count >= cfg.max_open_trades:
                        break
                    j = self.row_of_u[s][u]
                    if j < 0:
                        continue
                    direction = sd.direction[j]
                    if direction not in ("BUY", "SELL"):
                        continue
                    if cfg.forex_hours_only and not sd.is_crypto and self.hours[u] not in FOREX_HOURS:
                        continue
                    dist = sd.stop_dist[j]
                    if not (dist > 0) or math.isnan(dist):
                        continue
                    entry = sd.o[j]
                    lot = compute_lot(eq, dist, sd.spec, cfg)
                    if lot <= 0:
                        continue
                    tp_dist = dist * sd.scfg["rr"]
                    if direction == "BUY":
                        sl = entry - dist; tp = entry + tp_dist
                    else:
                        sl = entry + dist; tp = entry - tp_dist
                    risk_cash = (dist / sd.spec.point) * sd.spec.value_per_point * lot
                    pos = {"dir": direction, "sign": 1 if direction == "BUY" else -1,
                           "entry": entry, "entry_time": self.union[u],
                           "sl": sl, "orig_sl": sl, "tp": tp, "lot": lot, "risk_cash": risk_cash,
                           "pnl_per_price": sd.spec.value_per_point / sd.spec.point * lot,
                           "peak_r": 0.0, "peak_profit_usd": 0.0, "mgmt": self.mgmt[s]}
                    # same-bar exit check on the entry bar itself (gap/spike realism)
                    if cfg.protection_on:
                        res = self._protect_exit(pos, sd.h[j], sd.l[j], self.union[u])
                        ex, why = res if res is not None else (None, None)
                    else:
                        ex, why = self._check_exit(pos, sd.h[j], sd.l[j])
                    if ex is not None:
                        tr = self._close(sd, pos, ex, why, u, u)
                        balance += tr.net; trades.append(tr)
                    else:
                        positions[s] = pos; entry_u[s] = u; open_count += 1

            # (5) stop conditions
            if cfg.enforce_halts and total_halted and cfg.stop_on_total_halt and not positions:
                break
            if cfg.stop_on_pass and passed and not positions:
                break

        # close leftovers at last close for clean accounting
        for s in list(positions.keys()):
            sd = self.syms[s]; lc = last_close[s]
            u_out = u1 - 1
            tr = self._close(sd, positions[s], lc, "END", entry_u[s], u_out)
            balance += tr.net; trades.append(tr)

        return RunResult(trades, eq_t, eq_v, passed, total_halted,
                         worst_daily, worst_total, balance + floating())


# ════════════════════════════════════════════════════════════════════════════
#  METRICS + REPORTING
# ════════════════════════════════════════════════════════════════════════════
def summarise(trades: list, eq_vals: list, start_equity: float,
              worst_daily=None, worst_total=None) -> dict:
    n = len(trades)
    if n == 0:
        return {"trades": 0}
    net = np.array([t.net for t in trades])
    gross = np.array([t.gross for t in trades])
    wins = net > 0
    gross_win = net[wins].sum(); gross_loss = -net[~wins].sum()
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    if eq_vals:
        eq = np.array(eq_vals)
        peak = np.maximum.accumulate(eq)
        maxdd = float(((peak - eq) / peak).max() * 100)
    else:
        maxdd = None
    r = np.array([t.r_multiple for t in trades])
    dur_h = np.array([(t.exit_time - t.entry_time).total_seconds() / 3600 for t in trades])
    return {
        "trades": n,
        "win_rate": float(wins.mean() * 100),
        "gross_pnl": float(gross.sum()),
        "spread_cost": float(sum(t.spread_cost for t in trades)),
        "commission": float(sum(t.commission for t in trades)),
        "swap": float(sum(t.swap for t in trades)),
        "slippage": float(sum(t.slippage for t in trades)),
        "net_pnl": float(net.sum()),
        "net_return_pct": float(net.sum() / start_equity * 100),
        "profit_factor": float(pf),
        "expectancy_R": float(r.mean()),
        "avg_win": float(net[wins].mean()) if wins.any() else 0.0,
        "avg_loss": float(net[~wins].mean()) if (~wins).any() else 0.0,
        "max_dd_pct": maxdd,
        "avg_dur_h": float(dur_h.mean()),
        "worst_daily_dd": worst_daily,
        "worst_total_dd": worst_total,
    }


def pseudo_equity(trades: list, start_equity: float) -> list:
    """Approx equity curve from a trade list (sorted by exit): start + cumsum(net).
    Overlapping positions make this approximate, but it gives a usable OOS DD."""
    if not trades:
        return []
    order = sorted(trades, key=lambda t: t.exit_time)
    eq = start_equity; out = []
    for t in order:
        eq += t.net; out.append(eq)
    return out


def _fmt_money(x):
    return f"${x:,.2f}" if x is not None else "n/a"


def print_summary(title: str, s: dict):
    print("\n" + "=" * 68)
    print(f"  {title}")
    print("=" * 68)
    if s.get("trades", 0) == 0:
        print("  no trades")
        return
    tot_cost = s["spread_cost"] + s["commission"] + s["slippage"] - s["swap"]
    print(f"  Trades            : {s['trades']:,}     Win rate: {s['win_rate']:.1f}%")
    print(f"  Avg duration      : {s['avg_dur_h']:.1f} h")
    print("  ---- P&L (gross -> net) ------------------------------------------")
    print(f"  Gross P&L (price) : {_fmt_money(s['gross_pnl'])}")
    print(f"    - Spread        : {_fmt_money(-s['spread_cost'])}")
    print(f"    - Commission    : {_fmt_money(-s['commission'])}")
    if abs(s["slippage"]) > 1e-9:
        print(f"    - Slippage      : {_fmt_money(-s['slippage'])}")
    if abs(s["swap"]) > 1e-9:
        print(f"    +/- Swap        : {_fmt_money(s['swap'])}")
    print(f"    = Total costs   : {_fmt_money(-tot_cost)}")
    print(f"  NET P&L           : {_fmt_money(s['net_pnl'])}   ({s['net_return_pct']:+.2f}%)")
    if s["gross_pnl"] != 0:
        eaten = tot_cost / abs(s["gross_pnl"]) * 100
        print(f"  Costs / |gross|   : {eaten:.1f}%")
    print("  ---- Risk ---------------------------------------------------------")
    print(f"  Profit factor     : {s['profit_factor']:.2f}     Expectancy: {s['expectancy_R']:+.3f} R")
    print(f"  Avg win / loss    : {_fmt_money(s['avg_win'])} / {_fmt_money(s['avg_loss'])}")
    if s["max_dd_pct"] is not None:
        print(f"  Max drawdown      : {s['max_dd_pct']:.2f}%")
    if s.get("worst_daily_dd") is not None:
        print(f"  Worst daily DD    : {s['worst_daily_dd']:.2f}%  (FTMO fail 5%)")
        print(f"  Worst total DD    : {s['worst_total_dd']:.2f}%  (FTMO fail 10%)")


def trades_to_df(trades: list) -> pd.DataFrame:
    return pd.DataFrame([t.__dict__ for t in trades])


def per_symbol_table(trades: list) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    df = trades_to_df(trades)
    g = df.groupby("symbol").agg(
        trades=("net", "size"),
        win_rate=("net", lambda x: (x > 0).mean() * 100),
        gross=("gross", "sum"),
        costs=("spread_cost", lambda x: x.sum()),  # placeholder, fixed below
        net=("net", "sum"),
    )
    # proper total costs per symbol
    df["total_cost"] = df["spread_cost"] + df["commission"] + df["slippage"] - df["swap"]
    g["costs"] = df.groupby("symbol")["total_cost"].sum()
    g["expectancy_R"] = df.groupby("symbol")["r_multiple"].mean()
    return g.round(2)


def exit_reason_table(trades: list) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    df = trades_to_df(trades)
    g = df.groupby("reason").agg(
        n=("net", "size"),
        net=("net", "sum"),
        avg=("net", "mean"),
        win_rate=("net", lambda x: (x > 0).mean() * 100),
    ).sort_values("n", ascending=False)
    return g.round(2)


# ════════════════════════════════════════════════════════════════════════════
#  MODES
# ════════════════════════════════════════════════════════════════════════════
def build_union(syms: dict[str, SymData]) -> pd.DatetimeIndex:
    idx = None
    for sd in syms.values():
        idx = sd.times if idx is None else idx.union(sd.times)
    return idx.sort_values()


def load_symbols(mode_from, mode_to, cache_dir: Path, cfg: Config,
                 mt5_kwargs: dict, refresh: bool = False) -> dict[str, SymData]:
    syms = {}
    for symbol in SYMBOLS:
        scfg = SYM_CFG[symbol]; spec = SPECS[symbol]
        spec = replace(spec, is_crypto=scfg["is_crypto"])
        print(f"[{symbol}]")
        m5 = get_data(symbol, scfg["tf"], mode_from, mode_to, cache_dir, refresh, mt5_kwargs)
        tr = get_data(symbol, scfg["trend_tf"], mode_from, mode_to, cache_dir, refresh, mt5_kwargs)
        syms[symbol] = prepare_symbol(symbol, m5, tr, scfg, spec, cfg)
    return syms


def run_ab(syms, cfg: Config, out_dir: Path):
    """Run the same data through three configs and print a side-by-side table."""
    union = build_union(syms)
    variants = [
        ("no protection",       "ab_none",      replace(cfg, protection_on=False)),
        ("loss_only",           "ab_loss_only", replace(cfg, protection_on=True, protection_mode="loss_only")),
        ("full (loss+giveback)", "ab_full",     replace(cfg, protection_on=True, protection_mode="full")),
    ]
    rows = []
    for label, tag, c in variants:
        res = Portfolio(syms, union, c).run(0, len(union))
        s = summarise(res.trades, res.eq_vals, c.start_equity, res.worst_daily_dd, res.worst_total_dd)
        cost = s["spread_cost"] + s["commission"] + s["slippage"] - s["swap"]
        rows.append({"config": label, "trades": s["trades"], "win%": round(s["win_rate"], 1),
                     "gross$": round(s["gross_pnl"]), "costs$": round(cost),
                     "net$": round(s["net_pnl"]), "net%": round(s["net_return_pct"], 1),
                     "PF": round(s["profit_factor"], 2), "exp_R": round(s["expectancy_R"], 3),
                     "maxDD%": round(s["max_dd_pct"], 1) if s["max_dd_pct"] is not None else None})
        _write_outputs(res, out_dir, tag)
    print("\n" + "=" * 78)
    print(f"  A/B — protection configs on the same data (costs {'OFF' if not cfg.costs_on else 'ON'},"
          f" ref-risk ${cfg.protect_ref_risk:.0f})")
    print("=" * 78)
    print(pd.DataFrame(rows).to_string(index=False))
    best = max(rows, key=lambda r: r["net$"])
    print(f"\n  Best net: {best['config']}  (net ${best['net$']:,.0f}, "
          f"expectancy {best['exp_R']:+.3f}R, maxDD {best['maxDD%']}%)")
    print("  (per-config trade logs + equity curves written as ab_none / ab_loss_only / ab_full)")
    return rows


def run_backtest(syms, cfg: Config, out_dir: Path):
    union = build_union(syms)
    pf = Portfolio(syms, union, cfg)
    res = pf.run(0, len(union), reset_equity=True)
    s = summarise(res.trades, res.eq_vals, cfg.start_equity, res.worst_daily_dd, res.worst_total_dd)
    tag = "PROTECTION ON" if cfg.protection_on else "no protection"
    print_summary(f"CONTINUOUS BACKTEST  (net of costs, {tag})", s)
    if cfg.protection_on:
        print(f"\n  [protection] dollar tiers scaled at ref-risk ${cfg.protect_ref_risk:.0f}/trade; "
              f"LOWER --protect-ref-risk for looser $-stops (fire at a larger R), raise for tighter.")
    print("\n  Per-symbol (net):")
    print(per_symbol_table(res.trades).to_string())
    if cfg.protection_on:
        print("\n  Exit reasons (net):")
        print(exit_reason_table(res.trades).to_string())
    _write_outputs(res, out_dir, "backtest")
    return s


def run_challenge(syms, cfg: Config, out_dir: Path, challenge_days: int):
    cfg = replace(cfg, enforce_halts=True, stop_on_pass=True, stop_on_total_halt=True)
    union = build_union(syms)
    pf = Portfolio(syms, union, cfg)
    # cut union into consecutive windows of `challenge_days`
    day0 = union[0].normalize()
    edges = []
    d = day0
    while d < union[-1]:
        edges.append(d); d = d + pd.Timedelta(days=challenge_days)
    edges.append(union[-1] + pd.Timedelta(seconds=1))
    results = []
    for a, b in zip(edges[:-1], edges[1:]):
        u0 = int(union.searchsorted(a, side="left"))
        u1 = int(union.searchsorted(b, side="left"))
        if u1 - u0 < 2:
            continue
        r = pf.run(u0, u1, reset_equity=True)
        results.append((a.date(), r))
    n = len(results)
    passed = sum(1 for _, r in results if r.passed)
    breached = sum(1 for _, r in results if r.failed_total)
    wd = max((r.worst_daily_dd for _, r in results), default=0.0)
    wt = max((r.worst_total_dd for _, r in results), default=0.0)
    print("\n" + "=" * 68)
    print(f"  FTMO CHALLENGE WINDOWS  ({challenge_days}-day, costs ON, halts ON)")
    print("=" * 68)
    print(f"  Windows tested    : {n}")
    print(f"  PASSED (+{cfg.profit_target:.0f}%)   : {passed}/{n}"
          + (f"  ({passed/n*100:.0f}%)" if n else ""))
    print(f"  Total breaches    : {breached}/{n}   (self-halt before FTMO -10%)")
    print(f"  Worst daily DD    : {wd:.2f}%   (FTMO fail 5%)")
    print(f"  Worst total DD    : {wt:.2f}%   (FTMO fail 10%)")
    # dump per-window
    rows = [{"window_start": str(a), "passed": r.passed, "breached_total": r.failed_total,
             "worst_daily_dd": round(r.worst_daily_dd, 2), "worst_total_dd": round(r.worst_total_dd, 2),
             "end_equity": round(r.end_equity, 2), "trades": len(r.trades)} for a, r in results]
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "challenge_windows.csv", index=False)
    print(f"\n  per-window CSV -> {out_dir / 'challenge_windows.csv'}")


PARAM_GRID = {
    "min_score": [2, 3, 4],
    "rr": [1.0, 1.5, 2.0],
    "atr": [0.8, 1.2],
}


def _apply_params(scfg: dict, combo: dict) -> dict:
    x = dict(scfg); x.update(combo); return x


def run_walkforward(syms_raw_data: dict, cfg: Config, out_dir: Path,
                    folds: int, optimize: bool, min_is_trades: int,
                    reprep):
    """
    reprep(symbol, scfg) -> SymData, so we can rebuild signals per param combo.
    Walk-forward is per-symbol (independent), OOS trades aggregated at the end.
    """
    all_oos = []
    per_symbol_oos = {}
    for symbol in SYMBOLS:
        scfg0 = SYM_CFG[symbol]
        base = reprep(symbol, scfg0)
        times = base.times
        # equal folds across the timeline
        bounds = np.linspace(0, len(times), folds + 1, dtype=int)
        oos_trades = []
        for k in range(folds):
            # in-sample = everything before this fold; OOS = this fold
            is_lo, is_hi = 0, bounds[k]
            oos_lo, oos_hi = bounds[k], bounds[k + 1]
            if oos_hi - oos_lo < 50:
                continue
            if optimize and is_hi - is_lo >= 200:
                best, best_combo = None, scfg0
                for vals in itertools.product(*PARAM_GRID.values()):
                    combo = dict(zip(PARAM_GRID.keys(), vals))
                    sd = reprep(symbol, _apply_params(scfg0, combo))
                    pf = Portfolio({symbol: sd}, sd.times, cfg)
                    r = pf.run(is_lo, is_hi, reset_equity=True)
                    if len(r.trades) < min_is_trades:
                        continue
                    val = sum(t.net for t in r.trades)
                    if best is None or val > best:
                        best, best_combo = val, combo
                sd = reprep(symbol, _apply_params(scfg0, best_combo))
            else:
                sd = base
            pf = Portfolio({symbol: sd}, sd.times, cfg)
            r = pf.run(oos_lo, oos_hi, reset_equity=True)
            oos_trades.extend(r.trades)
        per_symbol_oos[symbol] = oos_trades
        all_oos.extend(oos_trades)

    print("\n" + "=" * 68)
    print(f"  WALK-FORWARD  ({folds} folds, OOS only, "
          f"{'param-optimised' if optimize else 'fixed params'}, costs ON)")
    print("=" * 68)
    for symbol in SYMBOLS:
        tr = per_symbol_oos[symbol]
        s = summarise(tr, [], cfg.start_equity)
        if s.get("trades", 0) == 0:
            print(f"  {symbol}: no OOS trades"); continue
        print(f"  {symbol:8s}  trades {s['trades']:4d}  win {s['win_rate']:4.1f}%  "
              f"net {_fmt_money(s['net_pnl']):>12s}  PF {s['profit_factor']:.2f}  "
              f"exp {s['expectancy_R']:+.3f}R")
    agg = summarise(all_oos, pseudo_equity(all_oos, cfg.start_equity), cfg.start_equity)
    print_summary("WALK-FORWARD  — aggregated OOS across all symbols", agg)
    out_dir.mkdir(parents=True, exist_ok=True)
    if all_oos:
        trades_to_df(all_oos).to_csv(out_dir / "walkforward_oos_trades.csv", index=False)
        print(f"\n  OOS trade log -> {out_dir / 'walkforward_oos_trades.csv'}")


def _write_outputs(res: RunResult, out_dir: Path, tag: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    if res.trades:
        trades_to_df(res.trades).to_csv(out_dir / f"{tag}_trades.csv", index=False)
    if res.eq_vals:
        pd.DataFrame({"time": res.eq_times, "equity": res.eq_vals}).to_csv(
            out_dir / f"{tag}_equity.csv", index=False)
    print(f"\n  outputs -> {out_dir}/{tag}_trades.csv , {tag}_equity.csv")


# ════════════════════════════════════════════════════════════════════════════
#  SELF-TEST  — synthetic random-walk data. Proves the engine runs end-to-end.
#  NOTE: results are noise, not an edge. Only the plumbing is being tested.
# ════════════════════════════════════════════════════════════════════════════
def _synth(symbol: str, n_days: int, spec: Spec, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    bars = n_days * 24 * 12          # M5 bars
    start = pd.Timestamp("2024-01-01", tz="UTC")
    idx = pd.date_range(start, periods=bars, freq="5min")
    base = 1.10 if symbol != "XAUUSD" else 2000.0
    step = spec.point * (30 if symbol != "XAUUSD" else 40)
    ret = rng.normal(0, step, bars)
    close = base + np.cumsum(ret)
    spr = np.abs(rng.normal(0, step * 0.3, bars))
    high = close + np.abs(rng.normal(0, step, bars)) + spr
    low = close - np.abs(rng.normal(0, step, bars)) - spr
    openp = np.concatenate([[close[0]], close[:-1]])
    vol = rng.integers(50, 500, bars).astype(float)
    m5 = pd.DataFrame({"open": openp, "high": np.maximum.reduce([openp, high, close]),
                       "low": np.minimum.reduce([openp, low, close]), "close": close,
                       "tick_volume": vol, "spread": spec.spread_points}, index=idx)
    m5.index.name = "time"
    h1 = m5.resample("1h").agg({"open": "first", "high": "max", "low": "min",
                                 "close": "last", "tick_volume": "sum", "spread": "mean"}).dropna()
    return m5, h1


def run_selftest():
    print("SELF-TEST — synthetic random-walk data (numbers are noise; testing plumbing)")
    cfg = Config(costs_on=True)
    syms = {}
    for i, symbol in enumerate(SYMBOLS):
        scfg = SYM_CFG[symbol]; spec = replace(SPECS[symbol], is_crypto=scfg["is_crypto"])
        m5, h1 = _synth(symbol, n_days=120, spec=spec, seed=1000 + i)
        syms[symbol] = prepare_symbol(symbol, m5, h1, scfg, spec, cfg)
    out = Path("./bt_out_selftest")

    run_backtest(syms, cfg, out)
    run_backtest(syms, replace(cfg, protection_on=True), out)   # exercise protection path
    run_challenge(syms, replace(cfg), out, challenge_days=14)

    # walk-forward needs a re-prep closure bound to the synthetic frames
    frames = {}
    for i, symbol in enumerate(SYMBOLS):
        spec = replace(SPECS[symbol], is_crypto=SYM_CFG[symbol]["is_crypto"])
        frames[symbol] = _synth(symbol, n_days=120, spec=spec, seed=1000 + i)

    def reprep(symbol, scfg):
        spec = replace(SPECS[symbol], is_crypto=scfg["is_crypto"])
        m5, h1 = frames[symbol]
        return prepare_symbol(symbol, m5.copy(), h1.copy(), scfg, spec, cfg)

    run_walkforward({}, cfg, out, folds=4, optimize=True, min_is_trades=10, reprep=reprep)
    print("\nSELF-TEST complete. Engine ran backtest + challenge + walk-forward with costs.")


# ════════════════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="Cost-inclusive backtest + walk-forward for the VX PROP FTMO strategy.")
    ap.add_argument("mode", nargs="?", default="selftest",
                    choices=["backtest", "challenge", "walkforward", "selftest"])
    ap.add_argument("--selftest", action="store_true", help="alias for the 'selftest' mode")
    ap.add_argument("--from", dest="dfrom", default="2023-01-01")
    ap.add_argument("--to", dest="dto", default="2026-07-01")
    ap.add_argument("--cache", default="./mt5_cache")
    ap.add_argument("--out", default="./bt_out")
    ap.add_argument("--challenge-days", type=int, default=14)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--optimize", action="store_true")
    ap.add_argument("--min-is-trades", type=int, default=20)
    ap.add_argument("--no-costs", action="store_true", help="turn frictions OFF (see the gross picture)")
    ap.add_argument("--slippage-points", type=float, default=0.0)
    ap.add_argument("--enforce-halts", action="store_true", help="apply FTMO halts in backtest mode too")
    ap.add_argument("--refresh", action="store_true", help="ignore CSV cache and re-pull from MT5")
    ap.add_argument("--protect", action="store_true", help="enable ported loss/giveback protection")
    ap.add_argument("--protect-mode", choices=["full", "loss_only"], default="full",
                    help="full = loss + giveback/breakeven/trail; loss_only = loss stops only")
    ap.add_argument("--ab", action="store_true",
                    help="backtest mode: run no-protection vs loss_only vs full side by side")
    ap.add_argument("--protect-ref-risk", type=float, default=20.0,
                    help="risk-$ the dollar tiers were tuned around; LOWER = looser $-stops, higher = tighter")
    ap.add_argument("--start-equity", type=float, default=START_EQUITY, help="challenge account size")
    ap.add_argument("--profit-target", type=float, default=PROFIT_TARGET, help="%% target (10 P1, 5 P2)")
    # MT5 login (optional; MT5 often already logged-in via terminal)
    ap.add_argument("--mt5-login", default=None)
    ap.add_argument("--mt5-password", default=None)
    ap.add_argument("--mt5-server", default=None)
    ap.add_argument("--mt5-terminal", default=None)
    args = ap.parse_args()

    if args.selftest or args.mode == "selftest":
        run_selftest(); return

    cfg = Config(costs_on=not args.no_costs, slippage_points=args.slippage_points,
                 enforce_halts=args.enforce_halts,
                 start_equity=args.start_equity, profit_target=args.profit_target,
                 protection_on=args.protect, protection_mode=args.protect_mode,
                 protect_ref_risk=args.protect_ref_risk)
    cache = Path(args.cache); out = Path(args.out)
    dfrom = datetime.fromisoformat(args.dfrom); dto = datetime.fromisoformat(args.dto)
    mt5_kwargs = dict(login=args.mt5_login, password=args.mt5_password,
                      server=args.mt5_server, terminal=args.mt5_terminal)

    print(f"Loading {SYMBOLS} {args.dfrom} -> {args.dto} (costs {'OFF' if args.no_costs else 'ON'})")
    try:
        syms = load_symbols(dfrom, dto, cache, cfg, mt5_kwargs, refresh=args.refresh)

        if args.mode == "backtest":
            if args.ab:
                run_ab(syms, cfg, out)
            else:
                run_backtest(syms, cfg, out)
        elif args.mode == "challenge":
            run_challenge(syms, cfg, out, args.challenge_days)
        elif args.mode == "walkforward":
            def reprep(symbol, scfg):
                spec = replace(SPECS[symbol], is_crypto=scfg["is_crypto"])
                m5 = get_data(symbol, scfg["tf"], dfrom, dto, cache, args.refresh, mt5_kwargs)
                tr = get_data(symbol, scfg["trend_tf"], dfrom, dto, cache, args.refresh, mt5_kwargs)
                return prepare_symbol(symbol, m5, tr, scfg, spec, cfg)
            run_walkforward({}, cfg, out, args.folds, args.optimize, args.min_is_trades, reprep)
    finally:
        mt5_disconnect()


if __name__ == "__main__":
    main()
