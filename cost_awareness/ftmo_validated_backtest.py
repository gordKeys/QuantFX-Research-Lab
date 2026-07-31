"""
================================================================================
  FTMO — VALIDATED-STRATEGY BACKTEST
  Runs ONLY what the QuantFX-Research-Lab repo actually validated for live use,
  ported faithfully, and re-tested in the cost-inclusive engine.

  WHAT THE REPO VALIDATED (configs/live_symbols.json + scripts/strategy_router.py):
    Strategy : FiveSignalConfluenceScalper (6 components) on M5,
               SL = 1.5*ATR14, TP = 4.0*ATR14, 0.5% risk/trade.
    EURUSD   : min_score 3, TREND component DISABLED  (4/4 walk-forward folds;
               entry_quality_analyzer found the EMA-trend component was dead
               weight here — its edge is the mean-reversion components).
    USDJPY   : min_score 5, require_trend_alignment=True  (3/4 folds).
  EXPLICITLY NOT VALIDATED (kept OUT of this backtest):
    AUDUSD   : best without spread, but 0/6 folds once REAL spread is modeled.
    USDCHF   : net negative even at min_score 5.
    XAUUSD   : intraday paused; only its H1 session variant ever showed edge.
    GBPUSD   : not routed at all (was the big cost-bleeder in the ftmo.py test).

  This reuses the ftmo_backtest_wf.py engine (portfolio event loop, spread +
  commission + swap costs, FTMO halts, challenge windows, MT5 fetch + CSV cache)
  and only swaps in the validated signal + per-symbol config. It also writes an
  ENTRY -> PEAK -> EXIT report (per-trade MFE, MFE_R, giveback) so you can see
  how much of each winner was given back.

  RUN
    python ftmo_validated_backtest.py selftest
    python ftmo_validated_backtest.py backtest  --from 2023-01-01 --to 2026-07-01 --start-equity 10000
    python ftmo_validated_backtest.py challenge  --challenge-days 14 --start-equity 10000
================================================================================
"""
from __future__ import annotations
import argparse
import math
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ftmo_backtest_wf as eng
from ftmo_backtest_wf import (
    Spec, Config, SymData, Portfolio, build_union, get_data, _slice, _shift1,
    summarise, print_summary, per_symbol_table, exit_reason_table, trades_to_df,
    _write_outputs, run_challenge, mt5_disconnect,
)

# ════════════════════════════════════════════════════════════════════════════
#  VALIDATED CONFIG  (lifted from the repo)
# ════════════════════════════════════════════════════════════════════════════
SL_ATR = 1.5
TP_ATR = 4.0
RR = TP_ATR / SL_ATR                 # 2.667
RISK_PCT = 0.5                       # repo RiskManager / live_challenge default
ATR_PERIOD = 14
STOP_BUFFER_MULT = 1.5

# per-symbol validated variant (strategy_router.py)
VARIANT = {
    "EURUSD": {"min_score": 3, "disabled": {"trend"},  "require_trend_alignment": False},
    "USDJPY": {"min_score": 5, "disabled": set(),       "require_trend_alignment": True},
}
SYMBOLS = ["EURUSD", "USDJPY"]

# commission from configs/commission_map.json ($ per lot ROUND TRIP -> /2 per side);
# swaps measured zero. USDJPY not in the map -> assume same round-trip as EURUSD.
SPECS = {
    "EURUSD": Spec(point=0.00001, digits=5, value_per_point=1.0,   spread_points=8,
                   commission_per_lot_side=5.04 / 2, swap_long=0.0, swap_short=0.0),
    # USDJPY value_per_point is ~100/price and is recomputed from the data at load.
    "USDJPY": Spec(point=0.001,   digits=3, value_per_point=0.667, spread_points=15,
                   commission_per_lot_side=5.04 / 2, swap_long=0.0, swap_short=0.0),
}
JPY_PAIRS = {"USDJPY"}


# ════════════════════════════════════════════════════════════════════════════
#  VALIDATED SIGNAL  — vectorised port of FiveSignalConfluenceScalper
# ════════════════════════════════════════════════════════════════════════════
def _rsi_sma(close: pd.Series, n=ATR_PERIOD) -> np.ndarray:
    d = close.diff()
    gain = d.clip(lower=0).rolling(n).mean()
    loss = (-d.clip(upper=0)).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).values


def _atr_sma(df: pd.DataFrame, n=ATR_PERIOD) -> np.ndarray:
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift()
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean().values


def compute_validated_signals(df: pd.DataFrame, variant: dict) -> np.ndarray:
    """Returns +1 / -1 / 0 per bar (signal computed on THAT closed bar)."""
    O, H, L, C = df["open"].values, df["high"].values, df["low"].values, df["close"].values
    V = df["tick_volume"].values
    c = df["close"]
    ema_f = c.ewm(span=20, adjust=False).mean().values
    ema_s = c.ewm(span=50, adjust=False).mean().values
    mid = c.rolling(20).mean(); std = c.rolling(20).std()
    upper = (mid + 2 * std).values; lower = (mid - 2 * std).values
    rsi = _rsi_sma(c, ATR_PERIOD)
    avgvol = df["tick_volume"].rolling(20).mean().values
    support = df["low"].rolling(30).min().shift(1).values
    resistance = df["high"].rolling(30).max().shift(1).values

    pO, pC = _shift1(O), _shift1(C)
    body = np.abs(C - O); rng = np.maximum(H - L, 1e-9)
    uw = H - np.maximum(O, C); lw = np.minimum(O, C) - L
    volspike = V > avgvol * 1.2
    bull_eng = (C > O) & (pC < pO) & (C >= pO) & (O <= pC)
    bear_eng = (C < O) & (pC > pO) & (O >= pC) & (C <= pO)
    bull_pin = (lw > body * 2) & (lw / rng > 0.45) & (C > O)
    bear_pin = (uw > body * 2) & (uw / rng > 0.45) & (C < O)

    long_trend = ema_f > ema_s
    short_trend = ema_f < ema_s
    long_flags = {
        "trend": long_trend, "band_extreme": C < lower, "rsi_extreme": rsi <= 35,
        "candle_pattern": bull_eng | bull_pin, "volume_spike": volspike & (C > O),
        "support_resistance": C <= support * 1.0015,
    }
    short_flags = {
        "trend": short_trend, "band_extreme": C > upper, "rsi_extreme": rsi >= 65,
        "candle_pattern": bear_eng | bear_pin, "volume_spike": volspike & (C < O),
        "support_resistance": C >= resistance * 0.9985,
    }
    dis = variant["disabled"]
    n = len(C)
    long_score = np.zeros(n, int); short_score = np.zeros(n, int)
    for name in long_flags:
        if name in dis:
            continue
        long_score += np.nan_to_num(long_flags[name]).astype(int)
        short_score += np.nan_to_num(short_flags[name]).astype(int)

    valid = ~(np.isnan(ema_f) | np.isnan(ema_s) | np.isnan(upper) | np.isnan(lower)
              | np.isnan(rsi) | np.isnan(avgvol) | np.isnan(support) | np.isnan(resistance))
    ms = variant["min_score"]; rta = variant["require_trend_alignment"]
    long_ok = (long_score >= ms) & (long_score > short_score) & valid
    short_ok = (short_score >= ms) & (short_score > long_score) & valid
    if rta:
        long_ok = long_ok & long_trend
        short_ok = short_ok & short_trend
    d = np.zeros(n, int)
    d[long_ok] = 1
    d[short_ok] = -1
    return d


# ════════════════════════════════════════════════════════════════════════════
#  PREPARE  — build a SymData the engine can run (signal + ATR stop, shifted)
# ════════════════════════════════════════════════════════════════════════════
def prepare_validated_symbol(symbol: str, m5: pd.DataFrame, spec: Spec, cfg: Config) -> SymData:
    variant = VARIANT[symbol]
    sig = compute_validated_signals(m5, variant)
    atr = _atr_sma(m5, ATR_PERIOD)

    # per-bar spread (points)
    if cfg.use_bar_spread and "spread" in m5.columns and m5["spread"].notna().any():
        spread_pts = m5["spread"].ffill().fillna(spec.spread_points).values.astype(float)
    else:
        spread_pts = np.full(len(m5), spec.spread_points, dtype=float)

    # stop distance = SL_ATR*ATR with a spread floor; TP handled via scfg["rr"]
    floor = np.maximum(spec.trade_stops_level, spread_pts) * STOP_BUFFER_MULT * spec.point
    stop_dist = np.maximum(SL_ATR * atr, floor)
    stop_dist = np.where(np.isnan(atr), np.nan, stop_dist)

    # actionable at NEXT bar open (no look-ahead)
    dir_str = np.where(sig == 1, "BUY", np.where(sig == -1, "SELL", None)).astype(object)
    direction_act = np.empty(len(m5), dtype=object)
    direction_act[0] = None
    direction_act[1:] = dir_str[:-1]
    stop_act = _shift1(stop_dist)

    scfg = {"rr": RR, "is_crypto": False, "min_score": variant["min_score"],
            "disabled": variant["disabled"], "require_trend_alignment": variant["require_trend_alignment"]}
    return SymData(symbol=symbol, scfg=scfg, spec=spec, times=m5.index,
                   o=m5["open"].values, h=m5["high"].values, l=m5["low"].values, c=m5["close"].values,
                   direction=direction_act, stop_dist=stop_act, spread_pts=spread_pts, is_crypto=False)


def _jpy_value_per_point(spec: Spec, mean_close: float) -> float:
    # 1 lot = 100,000 base; value of 1 point in USD = 100000*point / price
    return 100_000 * spec.point / mean_close


def build_syms(dfrom, dto, cache: Path, cfg: Config, refresh: bool, mt5_kwargs: dict) -> dict:
    syms = {}
    for symbol in SYMBOLS:
        spec = SPECS[symbol]
        print(f"[{symbol}]  ({VARIANT[symbol]['min_score']} of 6"
              + (", trend off" if 'trend' in VARIANT[symbol]['disabled'] else "")
              + (", trend-aligned" if VARIANT[symbol]['require_trend_alignment'] else "") + ")")
        m5 = get_data(symbol, "M5", dfrom, dto, cache, refresh, mt5_kwargs)
        if symbol in JPY_PAIRS:
            spec = replace(spec, value_per_point=_jpy_value_per_point(spec, float(m5["close"].mean())))
            print(f"  USDJPY value/point set to ${spec.value_per_point:.4f}/lot (from mean price)")
        syms[symbol] = prepare_validated_symbol(symbol, m5, spec, cfg)
    return syms


# ════════════════════════════════════════════════════════════════════════════
#  ENTRY -> PEAK -> EXIT enrichment  (per-trade MFE / giveback)
# ════════════════════════════════════════════════════════════════════════════
def peak_report(trades: list, syms: dict) -> pd.DataFrame:
    rows = []
    idxcache = {s: sd.times for s, sd in syms.items()}
    for t in trades:
        sd = syms[t.symbol]
        times = idxcache[t.symbol]
        a = times.searchsorted(t.entry_time, side="left")
        b = times.searchsorted(t.exit_time, side="right")
        seg_h = sd.h[a:b]; seg_l = sd.l[a:b]
        if len(seg_h) == 0:
            mfe_price = 0.0
        elif t.direction == "BUY":
            mfe_price = max(0.0, float(seg_h.max()) - t.entry)
        else:
            mfe_price = max(0.0, t.entry - float(seg_l.min()))
        # risk at entry = stop distance carried on the entry bar
        risk_price = sd.stop_dist[a] if a < len(sd.stop_dist) and not math.isnan(sd.stop_dist[a]) else np.nan
        vpp = sd.spec.value_per_point
        mfe_usd = mfe_price / sd.spec.point * vpp * t.lot
        mfe_r = (mfe_price / risk_price) if (risk_price and not math.isnan(risk_price)) else np.nan
        giveback_usd = mfe_usd - t.net
        rows.append({"symbol": t.symbol, "dir": t.direction,
                     "entry_time": t.entry_time, "exit_time": t.exit_time,
                     "entry": round(t.entry, sd.spec.digits), "exit": round(t.exit, sd.spec.digits),
                     "reason": t.reason, "lot": t.lot,
                     "mfe_R": round(mfe_r, 2) if not math.isnan(mfe_r) else None,
                     "mfe_usd": round(mfe_usd, 2), "net": round(t.net, 2),
                     "giveback_usd": round(giveback_usd, 2), "R": round(t.r_multiple, 2)})
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════════════
#  MODES
# ════════════════════════════════════════════════════════════════════════════
def run_backtest(syms, cfg: Config, out_dir: Path):
    union = build_union(syms)
    res = Portfolio(syms, union, cfg).run(0, len(union))
    s = summarise(res.trades, res.eq_vals, cfg.start_equity, res.worst_daily_dd, res.worst_total_dd)
    print_summary("VALIDATED STRATEGY — CONTINUOUS BACKTEST (net of costs)", s)
    if res.trades:
        print("\n  Per-symbol (net):")
        print(per_symbol_table(res.trades).to_string())
        rep = peak_report(res.trades, syms)
        out_dir.mkdir(parents=True, exist_ok=True)
        rep.to_csv(out_dir / "validated_entry_peak_exit.csv", index=False)
        # giveback summary
        peaked = rep[rep["mfe_usd"] > 0]
        gaveback = peaked[peaked["net"] <= 0]
        print("\n  Entry -> peak -> exit:")
        print(f"    trades that reached profit in-flight : {len(peaked)}/{len(rep)}")
        if len(peaked):
            conv = (peaked["net"] > 0).mean() * 100
            print(f"    of those, closed profitable          : {conv:.1f}%")
            print(f"    peaked then round-tripped to a loss  : {len(gaveback)}  "
                  f"(gave back ${(gaveback['mfe_usd'] - gaveback['net']).sum():,.0f} of peak)")
        print(f"    per-trade report -> {out_dir / 'validated_entry_peak_exit.csv'}")
    _write_outputs(res, out_dir, "validated")
    return s


def _synth(symbol, n_days, spec, seed):
    rng = np.random.default_rng(seed)
    bars = n_days * 24 * 12
    idx = pd.date_range("2024-01-01", periods=bars, freq="5min", tz="UTC")
    base = 150.0 if symbol in JPY_PAIRS else 1.10
    step = spec.point * (30 if symbol not in JPY_PAIRS else 40)
    close = base + np.cumsum(rng.normal(0, step, bars))
    spr = np.abs(rng.normal(0, step * 0.3, bars))
    high = close + np.abs(rng.normal(0, step, bars)) + spr
    low = close - np.abs(rng.normal(0, step, bars)) - spr
    openp = np.concatenate([[close[0]], close[:-1]])
    df = pd.DataFrame({"open": openp, "high": np.maximum.reduce([openp, high, close]),
                       "low": np.minimum.reduce([openp, low, close]), "close": close,
                       "tick_volume": rng.integers(50, 500, bars).astype(float),
                       "spread": spec.spread_points}, index=idx)
    df.index.name = "time"
    return df


def run_selftest():
    print("SELF-TEST — synthetic data (numbers are noise; testing the plumbing)")
    cfg = Config(costs_on=True, risk_pct=RISK_PCT, forex_hours_only=False)
    syms = {}
    for i, symbol in enumerate(SYMBOLS):
        spec = SPECS[symbol]
        df = _synth(symbol, 120, spec, 7 + i)
        if symbol in JPY_PAIRS:
            spec = replace(spec, value_per_point=_jpy_value_per_point(spec, float(df["close"].mean())))
        syms[symbol] = prepare_validated_symbol(symbol, df, spec, cfg)
    out = Path("./bt_validated_selftest")
    run_backtest(syms, cfg, out)
    run_challenge(syms, replace(cfg, enforce_halts=True), out, challenge_days=14)
    print("\nSELF-TEST complete.")


def main():
    ap = argparse.ArgumentParser(description="Backtest of the repo's validated EURUSD/USDJPY confluence strategy.")
    ap.add_argument("mode", nargs="?", default="selftest",
                    choices=["backtest", "challenge", "selftest"])
    ap.add_argument("--from", dest="dfrom", default="2023-01-01")
    ap.add_argument("--to", dest="dto", default="2026-07-01")
    ap.add_argument("--cache", default="./mt5_cache")
    ap.add_argument("--out", default="./bt_out")
    ap.add_argument("--challenge-days", type=int, default=14)
    ap.add_argument("--start-equity", type=float, default=10000.0)
    ap.add_argument("--profit-target", type=float, default=10.0)
    ap.add_argument("--risk-pct", type=float, default=RISK_PCT)
    ap.add_argument("--no-costs", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--protect", action="store_true", help="also apply the ported loss/giveback protection")
    ap.add_argument("--protect-mode", choices=["full", "loss_only"], default="full")
    ap.add_argument("--mt5-login", default=None); ap.add_argument("--mt5-password", default=None)
    ap.add_argument("--mt5-server", default=None); ap.add_argument("--mt5-terminal", default=None)
    args = ap.parse_args()

    if args.mode == "selftest":
        run_selftest(); return

    cfg = Config(costs_on=not args.no_costs, risk_pct=args.risk_pct,
                 start_equity=args.start_equity, profit_target=args.profit_target,
                 forex_hours_only=False,                      # validated scalper trades all hours
                 protection_on=args.protect, protection_mode=args.protect_mode)
    cache = Path(args.cache); out = Path(args.out)
    dfrom = datetime.fromisoformat(args.dfrom); dto = datetime.fromisoformat(args.dto)
    mt5_kwargs = dict(login=args.mt5_login, password=args.mt5_password,
                      server=args.mt5_server, terminal=args.mt5_terminal)
    print(f"VALIDATED universe {SYMBOLS}  {args.dfrom} -> {args.dto} "
          f"(costs {'OFF' if args.no_costs else 'ON'}, risk {args.risk_pct}%)")
    try:
        syms = build_syms(dfrom, dto, cache, cfg, args.refresh, mt5_kwargs)
        if args.mode == "backtest":
            run_backtest(syms, cfg, out)
        elif args.mode == "challenge":
            run_challenge(syms, cfg, out, args.challenge_days)
    finally:
        mt5_disconnect()


if __name__ == "__main__":
    main()
