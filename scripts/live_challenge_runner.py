"""
Live challenge runner -- the thin loop that turns configs/live_challenge.json
into real (or dry-run) MT5 orders, with every risk rule enforced BEFORE any
order is sent.

Design priorities, in order:
  1. Never breach an FTMO cap. The circuit breakers are checked first, every
     cycle, and halt trading before the -5% / -10% walls.
  2. Only take quality-gated trades. The throttle rejects weak signals, capped
     symbols, blocked hours, and over-frequency.
  3. Manage exits with a trailing stop so winners can exceed losers.
  4. Log everything as JSONL so the demo run can be audited trade by trade.

This file imports MetaTrader5 ONLY when actually going live. In --dry-run it
uses the adapter's simulated surface (or logs intended actions), so it can be
smoke-tested anywhere -- including here, without MT5.

Usage on the VPS (MT5 open and logged in):
    python run_project.py livechallenge --dry-run --loop-once   # smoke test
    python run_project.py livechallenge --dry-run               # paper, no orders
    python run_project.py livechallenge                         # DEMO orders

SAFETY: defaults to --dry-run unless --live is explicitly passed. You cannot
send orders by accident.
"""

from bootstrap import add_project_root

add_project_root()

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

# Config lives in configs/live_challenge.py
import importlib.util

spec = importlib.util.spec_from_file_location(
    "live_challenge", str(Path("configs/live_challenge.py"))
)
lc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lc)


LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


def log_event(event: dict):
    event["ts"] = datetime.now(timezone.utc).isoformat()
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = LOG_DIR / f"live_challenge_{day}.jsonl"
    with path.open("a") as f:
        f.write(json.dumps(event, default=str) + "\n")


class ChallengeRunner:
    def __init__(self, cfg: "lc.LiveConfig", live=False, dry_run=True):
        self.cfg = cfg
        self.live = live
        self.dry_run = dry_run

        # daily/session state, reset each broker day
        self.day = None
        self.day_start_equity = cfg.caps.account_size
        self.peak_equity = cfg.caps.account_size
        self.trades_today = 0
        self.symbol_trades_today = {}
        self.last_entry_min_by_symbol = {}
        self.halted_today = False
        self.halted_total = False

        self.broker = None  # set in connect()

    # ---- broker plumbing -------------------------------------------------

    def connect(self):
        """Attach the MT5 adapter. Only imports MT5 when going live."""
        if self.live:
            from mt5_broker_adapter import MT5BrokerAdapter, MT5UnavailableError
            try:
                self.broker = MT5BrokerAdapter()
                self.broker.initialize()
            except MT5UnavailableError as exc:
                raise SystemExit(f"Cannot go live: {exc}")
        else:
            self.broker = None  # dry-run logs intended actions instead

    def current_equity(self):
        if self.broker is not None:
            info = self.broker.mt5.account_info()
            return float(info.equity) if info else self.day_start_equity
        return self.day_start_equity  # dry-run: equity flat unless simulated

    # ---- daily lifecycle -------------------------------------------------

    def roll_day_if_needed(self):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self.day:
            self.day = today
            self.day_start_equity = self.current_equity()
            self.trades_today = 0
            self.symbol_trades_today = {}
            self.halted_today = False
            log_event({"event": "day_start", "day": today,
                       "day_start_equity": self.day_start_equity})

    # ---- the safety gate, checked every cycle ----------------------------

    def check_circuit_breakers(self):
        equity = self.current_equity()
        self.peak_equity = max(self.peak_equity, equity)

        if lc.total_circuit_breaker_hit(self.peak_equity, equity, self.cfg.caps):
            if not self.halted_total:
                log_event({"event": "HALT_TOTAL", "equity": equity,
                           "peak": self.peak_equity,
                           "reason": "total drawdown circuit breaker"})
                self.flatten_all("total_circuit_breaker")
            self.halted_total = True
            return False

        if lc.daily_circuit_breaker_hit(self.day_start_equity, equity, self.cfg.caps):
            if not self.halted_today:
                log_event({"event": "HALT_DAILY", "equity": equity,
                           "day_start": self.day_start_equity,
                           "reason": "daily loss circuit breaker"})
                self.flatten_all("daily_circuit_breaker")
            self.halted_today = True
            return False

        return True

    def flatten_all(self, reason):
        """Close every open position. On dry-run, log the intent."""
        if self.broker is None:
            log_event({"event": "flatten_all_DRYRUN", "reason": reason})
            return
        positions = self.broker.mt5.positions_get() or []
        for pos in positions:
            self.broker.close_position(pos.ticket)
            log_event({"event": "closed", "ticket": pos.ticket, "reason": reason})

    # ---- entry path ------------------------------------------------------

    def evaluate_symbol(self, symbol, signal_fn):
        """
        signal_fn(symbol) -> dict or None:
            {direction, strength, entry, atr, stop_atr, target_pips}
        Returns None if no trade, else places (or logs) the order.
        """
        if self.halted_today or self.halted_total:
            return

        sig = signal_fn(symbol)
        if not sig or sig["direction"] == 0:
            log_event({"event": "skip", "symbol": symbol, "reason": "no_signal"})
            return

        # quality gate
        if sig["strength"] < self.cfg.fast.min_signal_strength:
            log_event({"event": "skip", "symbol": symbol, "reason": "weak_signal",
                       "strength": sig["strength"]})
            return

        now = datetime.now(timezone.utc)
        minutes_now = now.hour * 60 + now.minute
        state = {
            "trades_today": self.trades_today,
            "open_positions": self.open_position_count(),
            "last_entry_min_by_symbol": self.last_entry_min_by_symbol,
            "symbol_trades_today": self.symbol_trades_today,
            "symbol": symbol,
            "minutes_now": minutes_now,
        }
        ok, reason = lc.can_take_fast_trade(state, self.cfg.fast, now.hour)
        if not ok:
            log_event({"event": "skip", "symbol": symbol, "reason": reason})
            return

        # target size gate: must allow >= min target (so >=~$12 is reachable)
        atr = sig["atr"]
        min_target = max(self.cfg.fast.min_target_atr * atr,
                         self.cfg.fast.min_target_pips * sig.get("pip", atr / 10))
        # (target is realised by the trail, not a fixed TP, but we require the
        #  ATR to be large enough that the trail can plausibly reach min_target)
        if atr * self.cfg.exit.trail_atr < min_target:
            log_event({"event": "skip", "symbol": symbol,
                       "reason": "insufficient_target_room", "atr": atr})
            return

        risk_frac = lc.effective_risk(self.current_equity(), self.cfg.caps)
        entry = sig["entry"]
        stop = entry - sig["direction"] * atr * self.cfg.exit.initial_stop_atr
        self.place_order(symbol, sig["direction"], entry, stop, risk_frac, atr)

    def open_position_count(self):
        if self.broker is None:
            return 0
        return len(self.broker.mt5.positions_get() or [])

    def place_order(self, symbol, direction, entry, stop, risk_frac, atr):
        equity = self.current_equity()
        risk_dollars = equity * risk_frac
        stop_dist = abs(entry - stop)
        # size computed by adapter using real tick value; here we pass intent
        order = {
            "event": "ORDER", "symbol": symbol,
            "direction": "buy" if direction > 0 else "sell",
            "entry": entry, "stop": stop, "risk_dollars": risk_dollars,
            "risk_frac": risk_frac, "atr": atr,
            "magic": self.cfg.magic_fast, "exit": "atr_trail",
        }

        if self.dry_run or self.broker is None:
            order["event"] = "ORDER_DRYRUN"
            log_event(order)
        else:
            result = self.broker.market_order(
                symbol=symbol, direction=direction, stop_loss=stop,
                risk_dollars=risk_dollars, magic=self.cfg.magic_fast,
            )
            order["result"] = str(result)
            log_event(order)

        self.trades_today += 1
        self.symbol_trades_today[symbol] = self.symbol_trades_today.get(symbol, 0) + 1
        now = datetime.now(timezone.utc)
        self.last_entry_min_by_symbol[symbol] = now.hour * 60 + now.minute

    # ---- exit management (trailing stop on open positions) ---------------

    def manage_open_positions(self, atr_fn):
        if self.broker is None:
            return
        for pos in (self.broker.mt5.positions_get() or []):
            if pos.magic not in (self.cfg.magic_fast, self.cfg.magic_trend):
                continue
            atr = atr_fn(pos.symbol)
            if not atr:
                continue
            tick = self.broker.mt5.symbol_info_tick(pos.symbol)
            if not tick:
                continue
            direction = 1 if pos.type == 0 else -1
            high = tick.bid  # approximation for the trailing reference
            low = tick.bid
            new_stop, _ = lc.trail_stop_update(
                direction, pos.sl or pos.price_open, pos.price_open,
                high, low, atr, self.cfg.exit, armed=True,
            )
            if (direction > 0 and new_stop > (pos.sl or 0)) or \
               (direction < 0 and (pos.sl == 0 or new_stop < pos.sl)):
                self.broker.modify_stop(pos.ticket, new_stop)
                log_event({"event": "trail_update", "ticket": pos.ticket,
                           "new_stop": new_stop})

    # ---- main loop -------------------------------------------------------

    def run(self, signal_fn, atr_fn, loop_once=False, interval=60):
        self.connect()
        log_event({"event": "runner_start", "live": self.live,
                   "dry_run": self.dry_run, "cfg": lc.asdict(self.cfg)})
        try:
            while True:
                self.roll_day_if_needed()
                if self.check_circuit_breakers():
                    self.manage_open_positions(atr_fn)
                    for symbol in self.cfg.fast_symbols:
                        self.evaluate_symbol(symbol, signal_fn)
                    # trend sleeve handled by its own (daily) cadence elsewhere
                log_event({"event": "cycle_end",
                           "equity": self.current_equity(),
                           "trades_today": self.trades_today,
                           "halted_today": self.halted_today})
                if loop_once:
                    break
                time.sleep(interval)
        finally:
            if self.broker is not None:
                self.broker.shutdown()
            log_event({"event": "runner_stop"})


def stub_signal(symbol):
    """Placeholder wired in dry-run so the loop is exercisable without the real
    signal module. Returns no trade. Replace with the real fast-signal call."""
    return {"direction": 0}


def stub_atr(symbol):
    return None


def main():
    parser = argparse.ArgumentParser(description="Live FTMO challenge runner.")
    parser.add_argument("--live", action="store_true",
                        help="Actually send orders. Without this, dry-run only.")
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--loop-once", action="store_true",
                        help="Run a single cycle and exit (smoke test).")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--config", default="configs/live_challenge.json")
    args = parser.parse_args()

    cfg = lc.LiveConfig()  # defaults match the tuned challenge config
    live = args.live
    dry_run = not live

    if live:
        print("LIVE MODE: orders WILL be sent to the connected MT5 account.")
        print("Make sure it is the DEMO account, not funded.")
    else:
        print("DRY-RUN: no orders sent. Intended actions are logged to logs/.")

    runner = ChallengeRunner(cfg, live=live, dry_run=dry_run)
    # NOTE: replace stub_signal / stub_atr with the real fast-signal + ATR feed
    # from the existing engine when wiring to production.
    runner.run(stub_signal, stub_atr, loop_once=args.loop_once, interval=args.interval)

    print(f"Done. Logs in logs/live_challenge_*.jsonl")


if __name__ == "__main__":
    main()
