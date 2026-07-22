"""
Live trend-sleeve runner -- the evidence-backed component, on demo, with hard
risk caps.

This is what goes live today. It trades ONLY the daily-trend sleeve
(NAS100, US500, XAUUSD, long-only Donchian breakout, ATR trailing exit) because
that is the only strategy in this project that survived out-of-sample
walkforward AND the concentration/regime robustness cuts. The fast
mean-reversion signal is deliberately NOT here: it was measured to be
negative-expectancy after costs on the user's own data, and putting it on a
challenge account would fail the challenge slowly.

Every risk control the user asked for is enforced here, and each is checked on
EVERY cycle, before any order:

  - risk per trade         0.5% of equity, sized across the structural stop
  - daily circuit breaker  -3%: flatten everything and stop trading for the day
  - total circuit breaker  -7%: flatten and stop entirely (well short of -10%)
  - profit lock-in         at +8%, per-trade risk halves to protect the target

What "live" means for this sleeve: D1 breakouts are RARE. Most cycles will place
no trade and log "no signal". That is correct behaviour, not a fault. The bot's
job on most days is to do nothing and stay within the caps; its job on the few
days a real trend breaks is to be in it.

Run modes:
  --dry-run     compute signals and log intended actions, place NO orders
  --loop-once   run a single cycle and exit (for cron or testing)
  (default)     loop forever, one cycle per --interval seconds

    python run_project.py trend_live --dry-run --loop-once
    python run_project.py trend_live            # live on the connected MT5 demo
"""

from bootstrap import add_project_root

add_project_root()

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from configs.live_challenge import (
    LiveConfig, daily_circuit_breaker_hit, total_circuit_breaker_hit,
    effective_risk,
)
from mt5_broker_adapter import MT5BrokerAdapter, MT5UnavailableError

LOG_DIR = Path("logs")
STATE_PATH = Path("logs/trend_live_state.json")

# Alias table: the sleeve's canonical names vs what the broker may call them.
BROKER_ALIASES = {
    "NAS100": ["NAS100", "US100", "USTEC", "NAS100.cash", "US100.cash", "USTEC.cash"],
    "US500": ["US500", "SPX500", "US500.cash", "SPX500.cash"],
    "XAUUSD": ["XAUUSD", "GOLD", "XAUUSD.pro"],
}


def log_event(day, payload):
    LOG_DIR.mkdir(exist_ok=True)
    payload["ts"] = datetime.now(timezone.utc).isoformat()
    path = LOG_DIR / f"trend_live_{day}.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def load_state():
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"day": None, "day_start_equity": None, "peak_equity": None, "halted_day": None}


def save_state(state):
    LOG_DIR.mkdir(exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))


def atr(rates_df, period=14):
    prev = rates_df["close"].shift(1)
    tr = pd.concat([
        rates_df["high"] - rates_df["low"],
        (rates_df["high"] - prev).abs(),
        (rates_df["low"] - prev).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def resolve_symbol(broker, canonical):
    """Find the broker's real name for a sleeve symbol."""
    for candidate in BROKER_ALIASES.get(canonical, [canonical]):
        info = broker.symbol_info(candidate)
        if info is not None:
            broker.mt5.symbol_select(candidate, True)
            return candidate
    return None


def compute_signal(broker, symbol, cfg: LiveConfig):
    """
    Daily Donchian breakout on the last N completed D1 bars. Returns
    (direction, entry_price, stop_price, atr) or None.
    """
    timeframe = broker.mt5.TIMEFRAME_D1
    need = cfg.trend.entry_lookback + 30
    rates = broker.rates_copy(symbol, timeframe, need)
    if rates is None or len(rates) < cfg.trend.entry_lookback + 15:
        return None

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    a = atr(df)
    band = float(a.iloc[-1])
    if np.isnan(band) or band <= 0:
        return None

    # prior N-day high, excluding the current (still-forming) bar
    prior_high = df["high"].iloc[-(cfg.trend.entry_lookback + 1):-1].max()
    prior_low = df["low"].iloc[-(cfg.trend.entry_lookback + 1):-1].min()
    last_close = float(df["close"].iloc[-1])

    tick = broker.mt5.symbol_info_tick(symbol)
    if tick is None:
        return None

    if last_close > prior_high:
        entry = tick.ask
        stop = entry - band * cfg.trend.initial_stop_atr
        return (1, entry, stop, band)
    if not cfg.trend.long_only and last_close < prior_low:
        entry = tick.bid
        stop = entry + band * cfg.trend.initial_stop_atr
        return (-1, entry, stop, band)
    return None


def manage_trailing_stops(broker, cfg: LiveConfig, day):
    """Walk open trend-sleeve positions and ratchet their stops up (longs) using
    the D1 ATR trail. Never loosens a stop."""
    positions = broker.positions_get()
    if not positions:
        return
    for p in positions:
        if p.magic != cfg.magic_trend:
            continue
        rates = broker.rates_copy(p.symbol, broker.mt5.TIMEFRAME_D1, 30)
        if rates is None or len(rates) < 15:
            continue
        df = pd.DataFrame(rates)
        band = float(atr(df).iloc[-1])
        if np.isnan(band) or band <= 0:
            continue
        is_buy = p.type == broker.mt5.POSITION_TYPE_BUY
        tick = broker.mt5.symbol_info_tick(p.symbol)
        if tick is None:
            continue
        if is_buy:
            new_stop = tick.bid - band * cfg.trend.trail_atr
            if new_stop > p.sl:
                broker.modify_position(p.ticket, p.symbol, stop_loss=new_stop, take_profit=p.tp)
                log_event(day, {"event": "trail", "symbol": p.symbol,
                                "ticket": p.ticket, "new_sl": new_stop})
        else:
            new_stop = tick.ask + band * cfg.trend.trail_atr
            if new_stop < p.sl:
                broker.modify_position(p.ticket, p.symbol, stop_loss=new_stop, take_profit=p.tp)
                log_event(day, {"event": "trail", "symbol": p.symbol,
                                "ticket": p.ticket, "new_sl": new_stop})


def flatten_all(broker, cfg, day, reason):
    positions = broker.positions_get()
    for p in positions or []:
        if p.magic == cfg.magic_trend:
            broker.close_position(p)
            log_event(day, {"event": "flatten", "symbol": p.symbol,
                            "ticket": p.ticket, "reason": reason})


def cycle(broker, cfg: LiveConfig, state, dry_run):
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    equity = broker.account_equity()
    if equity is None:
        log_event(day, {"event": "error", "detail": "no equity from broker"})
        return state

    # ---- day rollover ----
    if state["day"] != day:
        state["day"] = day
        state["day_start_equity"] = equity
        state["halted_day"] = None
    if state["peak_equity"] is None or equity > state["peak_equity"]:
        state["peak_equity"] = equity

    log_event(day, {"event": "cycle_start", "equity": equity,
                    "day_start_equity": state["day_start_equity"],
                    "peak_equity": state["peak_equity"]})

    # ---- circuit breakers, checked BEFORE any new order ----
    if total_circuit_breaker_hit(state["peak_equity"], equity, cfg.caps):
        if not dry_run:
            flatten_all(broker, cfg, day, "total_circuit_breaker")
        log_event(day, {"event": "HALT_TOTAL",
                        "detail": "total drawdown circuit breaker hit; trading stopped"})
        save_state(state)
        return state

    if daily_circuit_breaker_hit(state["day_start_equity"], equity, cfg.caps):
        if state["halted_day"] != day and not dry_run:
            flatten_all(broker, cfg, day, "daily_circuit_breaker")
        state["halted_day"] = day
        log_event(day, {"event": "HALT_DAY",
                        "detail": "daily loss circuit breaker hit; no more trades today"})
        save_state(state)
        return state

    # ---- manage existing positions (trail stops) ----
    if not dry_run:
        manage_trailing_stops(broker, cfg, day)

    # ---- look for new entries ----
    open_trend = sum(1 for p in (broker.positions_get() or []) if p.magic == cfg.magic_trend)
    risk_frac = effective_risk(equity, cfg.caps)

    for canonical in cfg.trend.symbols:
        if open_trend >= cfg.trend.max_concurrent:
            break
        symbol = resolve_symbol(broker, canonical)
        if symbol is None:
            log_event(day, {"event": "skip", "symbol": canonical, "reason": "not offered by broker"})
            continue
        # already in this symbol?
        if any(p.symbol == symbol and p.magic == cfg.magic_trend
               for p in (broker.positions_get(symbol=symbol) or [])):
            continue

        sig = compute_signal(broker, symbol, cfg)
        if sig is None:
            log_event(day, {"event": "skip", "symbol": symbol, "reason": "no_signal"})
            continue

        direction, entry, stop, band = sig
        risk_dist = abs(entry - stop)
        info = broker.symbol_info(symbol)
        tick_value = getattr(info, "trade_tick_value", 0) or 0
        tick_size = getattr(info, "trade_tick_size", 0) or getattr(info, "point", 0)
        if not tick_value or not tick_size or risk_dist <= 0:
            log_event(day, {"event": "skip", "symbol": symbol, "reason": "cannot_size"})
            continue

        dollars_per_unit = tick_value / tick_size
        risk_dollars = equity * risk_frac
        volume = risk_dollars / (risk_dist * dollars_per_unit)
        volume = broker.normalize_volume(symbol, volume)

        intent = {"event": "signal", "symbol": symbol, "direction": direction,
                  "entry": entry, "stop": stop, "volume": volume,
                  "risk_frac": risk_frac, "risk_dollars": round(risk_dollars, 2)}

        if dry_run:
            intent["mode"] = "DRY_RUN_no_order"
            log_event(day, intent)
        else:
            result = broker.place_order(symbol=symbol, direction=direction,
                                        volume=volume, stop_loss=stop,
                                        take_profit=0.0, comment="QuantFX trend")
            intent["order_result"] = getattr(result, "retcode", None)
            log_event(day, intent)
            open_trend += 1

    log_event(day, {"event": "cycle_end", "open_trend_positions": open_trend})
    save_state(state)
    return state


def main():
    parser = argparse.ArgumentParser(description="Live trend-sleeve runner with FTMO risk caps.")
    parser.add_argument("--dry-run", action="store_true", help="log intended actions, place no orders")
    parser.add_argument("--loop-once", action="store_true", help="run one cycle and exit")
    parser.add_argument("--interval", type=int, default=900, help="seconds between cycles (default 15 min)")
    parser.add_argument("--config", default="configs/live_challenge.json")
    args = parser.parse_args()

    cfg = LiveConfig()  # defaults match the challenge; JSON is for record/override

    try:
        broker = MT5BrokerAdapter(magic_number=cfg.magic_trend)
        broker.initialize()
    except MT5UnavailableError as exc:
        raise SystemExit(f"{exc}\nThis runner must run on the VPS with MT5 installed and logged in.")

    print("Trend-sleeve live runner started.")
    print(f"  symbols   : {', '.join(cfg.trend.symbols)} (long-only={cfg.trend.long_only})")
    print(f"  risk/trade: {cfg.caps.risk_per_trade*100:.1f}%   daily breaker: "
          f"-{cfg.caps.circuit_breaker_daily*100:.0f}%   total breaker: "
          f"-{cfg.caps.circuit_breaker_total*100:.0f}%")
    print(f"  mode      : {'DRY RUN (no orders)' if args.dry_run else 'LIVE'}")
    print("  Note: D1 breakouts are rare. Most cycles will log 'no_signal'. That is normal.\n")

    state = load_state()
    try:
        while True:
            state = cycle(broker, cfg, state, args.dry_run)
            if args.loop_once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        broker.shutdown()


if __name__ == "__main__":
    main()
