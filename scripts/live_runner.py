from bootstrap import add_project_root
add_project_root()

import argparse
from datetime import datetime, timezone
import pandas as pd
import json
from pathlib import Path
from collections import Counter
from datetime import timedelta

from engine.data_loader import DataLoader
from engine.features import FeatureEngine
from engine.risk_manager import RiskManager
from ftmo_rules import FtmoRules, FtmoRiskGuard
from strategy_router import StrategyRouter
from mt5_broker_adapter import MT5BrokerAdapter, MT5UnavailableError
from timing_utils import timed


def build_data_for_symbol(symbol, broker=None):
    if broker is None:
        return FeatureEngine().add_features(DataLoader(symbol=symbol).load())

    rates = broker.rates_copy(symbol, broker.mt5.TIMEFRAME_M5, 2000)
    if rates is None or len(rates) == 0:
        return None

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.rename(columns={"tick_volume": "tick_volume"})
    df = df.set_index("time")
    df = df[["open", "high", "low", "close", "tick_volume", "spread", "real_volume"]]
    return FeatureEngine().add_features(df)


def latest_signal(symbol, data, router):
    strategy = router.get_strategy(symbol)
    signal_series = strategy.generate_signals(data)
    signal = int(signal_series.iloc[-1])
    score = None
    if signal == 1:
        score = getattr(strategy, "last_long_score", None)
    elif signal == -1:
        score = getattr(strategy, "last_short_score", None)
    return signal, strategy, score


def ensure_log_dir():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    return log_dir


def append_jsonl(path, payload):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")


def date_log_paths(log_dir, day):
    return (
        log_dir / f"live_run_{day.isoformat()}.jsonl",
        log_dir / f"daily_summary_{day.isoformat()}.json",
    )


def format_status(symbol, consecutive_losses, cooldown_until, last_closed_pnl):
    cooldown_text = "off"
    if cooldown_until is not None:
        remaining = cooldown_until - datetime.now(timezone.utc)
        if remaining.total_seconds() > 0:
            cooldown_text = f"{remaining}"
        else:
            cooldown_text = "expired"

    pnl_text = "n/a" if last_closed_pnl is None else f"{last_closed_pnl:.2f}"
    return (
        f"STATUS | symbol={symbol} | "
        f"consecutive_losses={consecutive_losses} | "
        f"cooldown_remaining={cooldown_text} | "
        f"last_closed_pnl={pnl_text}"
    )


def _normalize_symbol(symbol):
    """Strip common broker suffixes (e.g. Exness 'm'/'.raw'/'-ecn') so tier
    lookups still match when the live symbol string carries a broker suffix,
    instead of silently falling through to the default tier."""
    raw = (symbol or "").upper()
    for suffix in (".RAW", ".ECN", ".PRO", "-ECN", "_ECN", ".M", "-M", "_M", "M"):
        if raw.endswith(suffix) and len(raw) - len(suffix) >= 6:
            return raw[: -len(suffix)]
    return raw


def trade_management_params(symbol=None):
    base = {
        "breakeven_at_r": 1.00,
        "trail_at_r": 1.60,
        "trail_buffer_r": 0.65,
        "giveback_trigger_r": 1.40,
        "giveback_buffer_r": 0.50,
        "min_peak_profit_usd": 4.0,
        "giveback_usd_buffer": 2.0,
        "max_minutes": 180,
        "max_bars": 48,
        "warn_loss_per_trade_usd": 11.0,
        "soft_loss_per_trade_usd": 13.5,
        "max_loss_per_trade_usd": 15.0,
        "quick_cut_minutes": 25.0,
        "quick_cut_loss_usd": 7.0,
    }

    symbol = _normalize_symbol(symbol)
    if symbol == "EURUSD":
        base.update(
            {
                "breakeven_at_r": 0.45,
                "trail_at_r": 0.80,
                "trail_buffer_r": 0.28,
                "giveback_trigger_r": 0.72,
                "giveback_buffer_r": 0.36,
                "min_peak_profit_usd": 2.0,
                "giveback_usd_buffer": 2.0,
                "max_minutes": 120,
                "max_bars": 28,
                "quick_cut_minutes": 20.0,
                "quick_cut_loss_usd": 5.0,
            }
        )
    elif symbol == "USDCHF":
        # USDCHF was sharing USDJPY's looser tier, but live data shows it is
        # the actual worst offender: biggest aggregate loss, ~1.8x deeper
        # avg MAE than USDJPY, and the worst avg giveback of all 4 symbols.
        # It behaves more like EURUSD (fast reversals, shallow real edge) than
        # like USDJPY, so give it its own tier: lock in profit earlier, trail
        # tighter, and close on giveback sooner.
        base.update(
            {
                "breakeven_at_r": 0.40,
                "trail_at_r": 0.75,
                "trail_buffer_r": 0.25,
                "giveback_trigger_r": 0.65,
                "giveback_buffer_r": 0.16,
                "min_peak_profit_usd": 2.0,
                "giveback_usd_buffer": 1.0,
                "max_minutes": 100,
                "max_bars": 24,
                "quick_cut_minutes": 20.0,
                "quick_cut_loss_usd": 5.0,
            }
        )
    elif symbol == "USDJPY":
        base.update(
            {
                "breakeven_at_r": 0.80,
                "trail_at_r": 1.45,
                "trail_buffer_r": 0.58,
                "giveback_trigger_r": 1.20,
                "giveback_buffer_r": 0.30,
                "min_peak_profit_usd": 3.0,
                "giveback_usd_buffer": 1.5,
                "max_minutes": 150,
                "max_bars": 36,
                "quick_cut_minutes": 30.0,
                "quick_cut_loss_usd": 7.0,
            }
        )
    elif symbol == "AUDUSD":
        # 2nd-worst avg giveback in live data behind USDCHF; tighten the
        # giveback buffer a bit so peaks get protected sooner, without
        # changing its otherwise-decent win rate by touching breakeven/trail.
        base.update(
            {
                "breakeven_at_r": 0.70,
                "trail_at_r": 1.35,
                "trail_buffer_r": 0.52,
                "giveback_trigger_r": 1.05,
                "giveback_buffer_r": 0.22,
                "min_peak_profit_usd": 3.0,
                "giveback_usd_buffer": 1.5,
                "max_minutes": 165,
                "max_bars": 40,
                "quick_cut_minutes": 30.0,
                "quick_cut_loss_usd": 7.0,
            }
        )
    elif symbol == "XAUUSD":
        # New symbol. Gold trends tend to run further once they get going
        # (matches the H1 confluence edge concentrating here), so give it
        # more room than the majors before trailing/giveback kicks in.
        base.update(
            {
                "breakeven_at_r": 0.90,
                "trail_at_r": 1.55,
                "trail_buffer_r": 0.60,
                "giveback_trigger_r": 1.30,
                "giveback_buffer_r": 0.32,
                "min_peak_profit_usd": 3.0,
                "giveback_usd_buffer": 1.5,
                "max_minutes": 180,
                "max_bars": 44,
                "quick_cut_minutes": 35.0,
                "quick_cut_loss_usd": 7.0,
            }
        )

    return base


def live_strategy_banner(router, symbols):
    lines = [
        "VX PROP LIVE MODE",
        "Entry system: 5-signal confluence branch",
        "Signals: EMA trend + Bollinger extreme + RSI extreme + candle pattern + volume spike + support/resistance",
        "Risk: hard per-trade loss cap $15 | floating cap $15 | 3-loss cooldown",
        "Exit profile: symbol-specific profit preservation + giveback close",
        f"Symbols: {', '.join(symbols)}",
    ]
    mapped = [f"{symbol}={router.get_strategy_name(symbol)}" for symbol in symbols]
    lines.append(f"Routing: {' | '.join(mapped)}")
    return lines


def entry_risk_gate(equity, floating_pnl, rules, day_start_equity=None):
    if day_start_equity is None:
        day_start_equity = rules.initial_balance

    daily_loss = day_start_equity - equity
    total_loss = rules.initial_balance - equity
    daily_ratio = daily_loss / rules.daily_loss_limit if rules.daily_loss_limit else 0.0
    total_ratio = total_loss / rules.total_loss_limit if rules.total_loss_limit else 0.0
    floating_ratio = abs(floating_pnl) / rules.max_floating_loss_usd if rules.max_floating_loss_usd else 0.0

    if daily_loss >= rules.daily_loss_limit:
        return False, "daily_loss_limit"
    if total_loss >= rules.total_loss_limit:
        return False, "total_loss_limit"
    if floating_pnl <= -rules.max_floating_loss_usd:
        return False, "max_floating_loss"

    if daily_ratio >= 0.75:
        return False, "near_daily_loss_limit"
    if total_ratio >= 0.90:
        return False, "near_total_loss_limit"
    if floating_ratio >= 0.70:
        return False, "near_floating_loss_limit"

    return True, "ok"


def floating_pnl_summary(positions):
    total = 0.0
    by_symbol = {}
    for position in positions or []:
        profit = float(getattr(position, "profit", 0.0) or 0.0)
        total += profit
        symbol = getattr(position, "symbol", "")
        by_symbol[symbol] = by_symbol.get(symbol, 0.0) + profit
    return total, by_symbol


def max_volume_for_loss(broker, symbol, direction, entry_price, stop_price, max_loss_usd):
    if broker is None:
        return None
    loss_per_lot = broker.order_calc_profit(direction, symbol, 1.0, entry_price, stop_price)
    if loss_per_lot is None:
        return None
    loss_per_lot = abs(float(loss_per_lot))
    if loss_per_lot <= 0:
        return None
    return max_loss_usd / loss_per_lot


def close_all_positions(broker, positions):
    results = []
    for position in positions or []:
        try:
            results.append(broker.close_position(position))
        except Exception as exc:
            results.append(exc)
    return results


CLOSE_ACTIONS = {
    "soft_dollar_stop",
    "warn_dollar_stop",
    "hard_dollar_stop",
    "quick_cut_never_profitable",
    "loss_cut",
    "time_stop",
    "profit_giveback_close",
    "profit_giveback_close_usd",
}


def manage_live_position(broker, position, current_price, current_time, mgmt, tracker=None):
    risk = abs(position.price_open - position.sl)
    # NOTE: risk legitimately becomes 0 once the stop is moved to breakeven
    # (sl == entry) -- that used to make this function bail out entirely via
    # "invalid_risk", which silently disabled ALL protection (dollar stops,
    # giveback close, everything) for the rest of the trade's life the
    # moment it reached breakeven. That's very likely the single biggest
    # contributor to the "peaks then gives it all back" pattern: right when
    # a trade earns active protection, it lost it. Fix: track open_r as
    # None when risk is 0 and guard every R-dependent check individually;
    # the plain $-based checks below don't need R and must keep running.
    is_buy = position.type == broker.mt5.POSITION_TYPE_BUY
    current_pnl = (current_price - position.price_open) if is_buy else (position.price_open - current_price)
    open_r = (current_pnl / risk) if risk > 0 else None
    current_pnl_usd = float(getattr(position, "profit", current_pnl) or current_pnl)

    position_time = getattr(position, "time", None)
    held_minutes = 0.0
    if position_time is not None:
        if isinstance(position_time, (int, float)):
            position_time = datetime.fromtimestamp(position_time, tz=timezone.utc)
        elif getattr(position_time, "tzinfo", None) is None:
            position_time = position_time.replace(tzinfo=timezone.utc)
        held_minutes = (current_time - position_time).total_seconds() / 60.0

    position_id = int(getattr(position, "ticket", 0) or 0)
    peak_profit_usd = current_pnl_usd
    peak_r = open_r
    giveback_r = 0.0
    if tracker is not None and position_id:
        snapshot = tracker.setdefault(
            position_id,
            {"peak_r": open_r if open_r is not None else 0.0, "peak_profit_usd": current_pnl_usd, "last_seen": current_time},
        )
        if open_r is not None:
            snapshot["peak_r"] = max(float(snapshot.get("peak_r", open_r)), open_r)
        snapshot["peak_profit_usd"] = max(float(snapshot.get("peak_profit_usd", current_pnl_usd)), current_pnl_usd)
        snapshot["last_seen"] = current_time
        peak_r = snapshot["peak_r"]
        peak_profit_usd = snapshot["peak_profit_usd"]
        if open_r is not None:
            giveback_r = peak_r - open_r

    if current_pnl_usd <= -mgmt["soft_loss_per_trade_usd"]:
        return broker.close_position(position), "soft_dollar_stop"

    if current_pnl_usd <= -mgmt["warn_loss_per_trade_usd"]:
        return broker.close_position(position), "warn_dollar_stop"

    if current_pnl_usd <= -mgmt["max_loss_per_trade_usd"]:
        return broker.close_position(position), "hard_dollar_stop"

    # Never-profitable quick cut: this is a *separate* problem from giveback.
    # ~88% of aggregate loss in the live sample came from trades that never
    # once reached profit (mfe<=0) -- these aren't failing to protect a peak,
    # they were just wrong from the start. Waiting the full warn_loss amount
    # ($11) on a trade that's shown zero follow-through after a reasonable
    # window bleeds more than it needs to. Only applies when peak_profit_usd
    # is still <= 0 (never touched positive), so trades that dip before
    # running are not affected -- this must not fire once a trade has ever
    # been in profit.
    quick_cut_minutes = float(mgmt.get("quick_cut_minutes", 0.0) or 0.0)
    quick_cut_loss_usd = float(mgmt.get("quick_cut_loss_usd", 0.0) or 0.0)
    if (
        quick_cut_minutes > 0
        and quick_cut_loss_usd > 0
        and peak_profit_usd <= 0
        and held_minutes >= quick_cut_minutes
        and current_pnl_usd <= -quick_cut_loss_usd
    ):
        return broker.close_position(position), "quick_cut_never_profitable"

    if held_minutes >= mgmt["max_minutes"] and open_r is not None and open_r <= -0.18:
        return broker.close_position(position), "loss_cut"

    if held_minutes >= mgmt["max_bars"] * 5 and open_r is not None and open_r < 0:
        return broker.close_position(position), "time_stop"

    min_peak_profit_usd = float(mgmt.get("min_peak_profit_usd", 0.0) or 0.0)
    if (
        open_r is not None
        and peak_r >= mgmt["giveback_trigger_r"]
        and current_pnl_usd >= min_peak_profit_usd
        and giveback_r >= mgmt["giveback_buffer_r"]
    ):
        return broker.close_position(position), "profit_giveback_close"

    # Dollar-based safety net: the R-based check above only fires once the
    # peak *R-multiple* clears giveback_trigger_r, which requires a fairly
    # large move relative to the stop distance. Plenty of trades peak at a
    # modest but real dollar profit (well above min_peak_profit_usd) without
    # ever reaching that R threshold, then round-trip to a loss with nothing
    # protecting them. This checks the tracked dollar peak directly instead
    # of going through R-space, so those trades still get closed once
    # they've given back a meaningful chunk of an established profit. This
    # check does NOT depend on open_r, so it's the one thing that still
    # protects a trade once it's past breakeven and risk has collapsed to 0.
    giveback_usd_buffer = float(mgmt.get("giveback_usd_buffer", 0.0) or 0.0)
    if (
        giveback_usd_buffer > 0
        and peak_profit_usd >= min_peak_profit_usd
        and (peak_profit_usd - current_pnl_usd) >= giveback_usd_buffer
    ):
        return broker.close_position(position), "profit_giveback_close_usd"

    new_sl = position.sl
    if open_r is not None:
        if open_r >= mgmt["breakeven_at_r"]:
            new_sl = max(new_sl, position.price_open) if is_buy else min(new_sl, position.price_open)

        if open_r >= mgmt["trail_at_r"]:
            trail_distance = risk * mgmt["trail_buffer_r"]
            new_sl = max(new_sl, current_price - trail_distance) if is_buy else min(new_sl, current_price + trail_distance)

    if new_sl != position.sl:
        return broker.modify_position(position.ticket, position.symbol, new_sl, position.tp), "modify_sl"

    return None, "hold"


def profit_status_line(position, current_price, mgmt):
    risk = abs(position.price_open - position.sl)
    if risk <= 0:
        return "profit_locked=unknown | trailing=unknown | fade=unknown"

    is_buy = position.type == 0
    current_pnl = (current_price - position.price_open) if is_buy else (position.price_open - current_price)
    open_r = current_pnl / risk
    profit_locked = "yes" if open_r >= mgmt["breakeven_at_r"] else "no"
    trailing = "yes" if open_r >= mgmt["trail_at_r"] else "no"
    return f"profit_locked={profit_locked} | trailing={trailing} | fade=on"


def minutes_since_weekly_open(now, weekday=6, hour=22, minute=0):
    """Minutes elapsed since the most recent weekly market open (default:
    Sunday 22:00 UTC, the standard forex week start). Returns a large number
    if it can't find a recent one (shouldn't happen in practice, since the
    most recent occurrence is always within the last 7 days)."""
    candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    days_back = (candidate.weekday() - weekday) % 7
    candidate = candidate - timedelta(days=days_back)
    if candidate > now:
        candidate -= timedelta(days=7)
    return (now - candidate).total_seconds() / 60.0


def cooldown_delta_from_args(args):
    return timedelta(minutes=max(1, args.cooldown_candles) * 5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--symbols", nargs="+", default=["AUDUSD", "EURUSD", "USDCHF", "USDJPY", "XAUUSD"]
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--loop-once", action="store_true")
    parser.add_argument("--max-consecutive-losses", type=int, default=3)
    parser.add_argument("--cooldown-candles", type=int, default=12)
    parser.add_argument(
        "--market-open-buffer-minutes",
        type=int,
        default=30,
        help="Suppress new entries (position management still runs normally) for this many minutes after "
        "the weekly market open. Set to 0 to disable. Default 30: the first stretch after Sunday reopen "
        "tends to have wider spreads, thinner liquidity, and gap risk from Friday's close.",
    )
    parser.add_argument("--market-open-weekday", type=int, default=6, help="0=Monday .. 6=Sunday (default 6)")
    parser.add_argument("--market-open-hour", type=int, default=22, help="UTC hour of weekly open (default 22)")
    parser.add_argument("--market-open-minute", type=int, default=0, help="UTC minute of weekly open (default 0)")
    parser.add_argument("--magic-number", type=int, default=26072026)
    parser.add_argument("--mirror-signals", action="store_true", help="Invert strategy signals for null trading")
    args = parser.parse_args()

    router = StrategyRouter()
    rules = FtmoRules(
        initial_balance=10000,
        max_daily_loss_pct=0.04,
        max_total_loss_pct=0.08,
        max_risk_per_trade_pct=0.0020,
        max_open_positions=len(args.symbols),
        max_floating_loss_usd=15.0,
        max_consecutive_losses=args.max_consecutive_losses,
    )
    guard = FtmoRiskGuard(rules)
    risk = RiskManager(risk_per_trade=rules.max_risk_per_trade_pct)
    log_dir = ensure_log_dir()
    cooldown_until = None
    last_deal_check = None
    last_closed_pnl = None
    trade_trackers = {}
    # Tickets already registered with the consecutive-loss guard via the
    # immediate close-detection path above, so the delayed deal-history poll
    # below doesn't double-count the same closed trade when it eventually
    # catches up.
    registered_closed_tickets = set()

    broker = None
    if not args.dry_run:
        try:
            broker = MT5BrokerAdapter(magic_number=args.magic_number)
            broker.initialize()
            last_deal_check = datetime.now(timezone.utc) - timedelta(minutes=5)
        except MT5UnavailableError as exc:
            print(f"MT5 unavailable, falling back to dry-run: {exc}")
            args.dry_run = True

    if args.mirror_signals:
        print("NULL TRADER ACTIVE | signals are being mirrored")

    print("\n=== LIVE STRATEGY PROFILE ===")
    for line in live_strategy_banner(router, args.symbols):
        print(line)
    print("=" * 68)

    while True:
        started = datetime.now(timezone.utc)
        cycle_counts = Counter()
        current_day = started.date()
        run_log, summary_file = date_log_paths(log_dir, current_day)
        print(f"\n=== LIVE CYCLE {started.isoformat()} ===")
        append_jsonl(run_log, {"event": "cycle_start", "time": started})

        if cooldown_until and started < cooldown_until:
            remaining = cooldown_until - started
            print(f"Cooldown active for {remaining}")
            append_jsonl(
                run_log,
                {
                    "event": "cooldown_active",
                    "time": started,
                    "cooldown_until": cooldown_until,
                    "remaining_seconds": remaining.total_seconds(),
                },
            )
            if args.loop_once:
                break
            elapsed = (datetime.now(timezone.utc) - started).total_seconds()
            sleep_for = max(1, args.poll_seconds - int(elapsed))
            print(f"Sleeping {sleep_for}s")
            import time
            time.sleep(sleep_for)
            continue

        if cooldown_until and started >= cooldown_until:
            cooldown_until = None
            guard.consecutive_losses = 0
            append_jsonl(run_log, {"event": "cooldown_lifted", "time": started})

        if broker and not args.dry_run:
            positions = broker.positions_get()
            if positions:
                for position in positions:
                    symbol = getattr(position, "symbol", "")
                    mgmt = trade_management_params(symbol)
                    tick = broker.mt5.symbol_info_tick(symbol)
                    if tick is None:
                        continue
                    current_price = tick.bid if getattr(position, "type", 0) == broker.mt5.POSITION_TYPE_BUY else tick.ask
                    action_result, action = manage_live_position(broker, position, current_price, started, mgmt, trade_trackers)
                    if action_result is not None:
                        print(f"{symbol}: manage action={action} result={action_result}")
                        append_jsonl(
                            run_log,
                            {
                                "event": "position_manage",
                                "symbol": symbol,
                                "ticket": getattr(position, "ticket", None),
                                "action": action,
                                "result": str(action_result),
                                "time": started,
                            },
                        )
                        # NOTE: this must list every action manage_live_position can
                        # return that means the position actually closed. The
                        # previous list was missing "quick_cut_never_profitable"
                        # and "profit_giveback_close_usd" -- meaning trade_trackers
                        # never got cleaned up for those close types, and (more
                        # seriously, see below) the consecutive-loss guard never
                        # heard about those closes at all.
                        if action in CLOSE_ACTIONS:
                            ticket = int(getattr(position, "ticket", 0) or 0)
                            trade_trackers.pop(ticket, None)
                            # Register the loss/win with the cooldown guard RIGHT
                            # NOW, using the position's own realized profit at the
                            # moment of closing -- do not wait for the broker's
                            # deal-history poll below to notice it. That poll runs
                            # once per outer loop cycle and can lag reality by a
                            # full cycle or more; if positions open and close
                            # faster than that (exactly what happened at market
                            # open: a run of EURUSD trades opening and closing
                            # about every 60 seconds), the consecutive-loss
                            # counter falls behind and the cooldown that's
                            # supposed to kick in after 3 straight losses never
                            # fires in time -- which is how 5+ more losing entries
                            # got placed on the same symbol after the threshold
                            # was already crossed.
                            if ticket and ticket not in registered_closed_tickets:
                                realized_pnl = float(getattr(position, "profit", 0.0) or 0.0)
                                guard.register_closed_trade(realized_pnl)
                                registered_closed_tickets.add(ticket)
                                last_closed_pnl = realized_pnl
                                if guard.consecutive_losses >= rules.max_consecutive_losses:
                                    cooldown_until = started + cooldown_delta_from_args(args)
                                    print(
                                        f"{rules.max_consecutive_losses} consecutive losses reached (detected immediately "
                                        f"on close, not via delayed deal history); pausing for {args.cooldown_candles} M5 "
                                        f"candles until {cooldown_until.isoformat()}"
                                    )
                                    append_jsonl(
                                        run_log,
                                        {
                                            "event": "cooldown_started",
                                            "time": started,
                                            "cooldown_until": cooldown_until,
                                            "consecutive_losses": guard.consecutive_losses,
                                            "source": "immediate_close",
                                        },
                                    )

        open_positions = broker.positions_get() if broker and not args.dry_run else []
        floating_pnl, floating_by_symbol = floating_pnl_summary(open_positions)
        if open_positions:
            print(f"ACCOUNT STATUS | open_positions={len(open_positions)} | floating_pnl={floating_pnl:.2f} | by_symbol={floating_by_symbol}")

        if broker and not args.dry_run and open_positions and floating_pnl <= -rules.max_floating_loss_usd:
            print(
                f"ACCOUNT RISK STOP | floating_pnl={floating_pnl:.2f} "
                f"<= -{rules.max_floating_loss_usd:.2f}; closing all positions and starting cooldown"
            )
            append_jsonl(
                run_log,
                {
                    "event": "account_risk_stop",
                    "time": started,
                    "floating_pnl": floating_pnl,
                    "limit": -rules.max_floating_loss_usd,
                    "open_positions": len(open_positions),
                    "by_symbol": floating_by_symbol,
                },
            )
            close_results = close_all_positions(broker, open_positions)
            append_jsonl(
                run_log,
                {
                    "event": "account_risk_stop_close_results",
                    "time": started,
                    "results": [str(result) for result in close_results],
                },
            )
            cooldown_until = started + cooldown_delta_from_args(args)
            guard.consecutive_losses = 0
            if args.loop_once:
                break
            elapsed = (datetime.now(timezone.utc) - started).total_seconds()
            sleep_for = max(1, args.poll_seconds - int(elapsed))
            print(f"Sleeping {sleep_for}s")
            import time
            time.sleep(sleep_for)
            continue

        if broker and not args.dry_run and last_deal_check is not None:
            closed_deals = broker.history_deals_since(last_deal_check, magic=args.magic_number)
            last_deal_check = started
            if closed_deals:
                for deal in closed_deals:
                    deal_ticket = int(getattr(deal, "position_id", 0) or 0)
                    if deal_ticket and deal_ticket in registered_closed_tickets:
                        # Already registered immediately when manage_live_position
                        # closed it -- this is deal history catching up to a
                        # close we already accounted for. Consume it and move on
                        # instead of double-counting the loss/win.
                        registered_closed_tickets.discard(deal_ticket)
                        continue
                    profit = float(getattr(deal, "profit", 0.0) or 0.0)
                    if profit != 0:
                        last_closed_pnl = profit
                        guard.register_closed_trade(profit)
                        append_jsonl(
                            run_log,
                            {
                                "event": "closed_deal",
                                "symbol": getattr(deal, "symbol", ""),
                                "profit": profit,
                                "time": getattr(deal, "time", started),
                                "consecutive_losses": guard.consecutive_losses,
                                "magic": getattr(deal, "magic", None),
                            },
                        )

                if guard.consecutive_losses >= rules.max_consecutive_losses:
                    cooldown_until = started + cooldown_delta_from_args(args)
                    print(
                        f"3 consecutive losses reached; pausing for {args.cooldown_candles} M5 candles "
                        f"until {cooldown_until.isoformat()}"
                    )
                    append_jsonl(
                        run_log,
                        {
                            "event": "cooldown_started",
                            "time": started,
                            "cooldown_until": cooldown_until,
                            "consecutive_losses": guard.consecutive_losses,
                        },
                    )
                    if args.loop_once:
                        break
                    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
                    sleep_for = max(1, args.poll_seconds - int(elapsed))
                    print(f"Sleeping {sleep_for}s")
                    import time
                    time.sleep(sleep_for)
                    continue

        market_open_buffer_active = False
        if args.market_open_buffer_minutes > 0:
            minutes_elapsed = minutes_since_weekly_open(
                started, args.market_open_weekday, args.market_open_hour, args.market_open_minute
            )
            if minutes_elapsed < args.market_open_buffer_minutes:
                market_open_buffer_active = True
                print(
                    f"MARKET OPEN BUFFER ACTIVE | {minutes_elapsed:.1f}min since weekly open "
                    f"(< {args.market_open_buffer_minutes}min) -- new entries suppressed, existing positions still managed"
                )

        for symbol in args.symbols:
            print(format_status(symbol, guard.consecutive_losses, cooldown_until, last_closed_pnl))
            with timed(f"{symbol} evaluation"):
                data = build_data_for_symbol(symbol, broker=broker if not args.dry_run else None)
                if data is None or data.empty:
                    print(f"{symbol}: no data available")
                    append_jsonl(run_log, {"event": "no_data", "symbol": symbol, "time": datetime.now(timezone.utc)})
                    cycle_counts["no_data"] += 1
                    continue
                signal, strategy, entry_score = latest_signal(symbol, data, router)
                if args.mirror_signals and signal != 0:
                    signal = -signal
                if signal != 0 and market_open_buffer_active:
                    print(f"{symbol}: signal={signal} suppressed (within {args.market_open_buffer_minutes}min of weekly market open)")
                    append_jsonl(
                        run_log,
                        {
                            "event": "signal_suppressed_market_open",
                            "symbol": symbol,
                            "signal": signal,
                            "buffer_minutes": args.market_open_buffer_minutes,
                            "time": started,
                        },
                    )
                    cycle_counts["market_open_buffer"] += 1
                    continue
                price = float(data["close"].iloc[-1])
                atr = float(data["atr"].iloc[-1])
                broker_time = data.index[-1].to_pydatetime()
                mgmt = trade_management_params(symbol)

                if broker and not args.dry_run:
                    equity = broker.account_equity() or rules.initial_balance
                else:
                    equity = rules.initial_balance

                entry_allowed, entry_reason = entry_risk_gate(
                    equity,
                    floating_pnl,
                    rules,
                    day_start_equity=guard.day_start_equity,
                )
                if not entry_allowed:
                    print(
                        f"{symbol}: new entries paused ({entry_reason}) | "
                        f"equity={equity:.2f} | floating_pnl={floating_pnl:.2f}"
                    )
                    cycle_counts["skip_entry_risk_gate"] += 1
                    append_jsonl(
                        run_log,
                        {
                            "event": "skip_entry_risk_gate",
                            "symbol": symbol,
                            "reason": entry_reason,
                            "equity": equity,
                            "floating_pnl": floating_pnl,
                            "broker_time": broker_time,
                        },
                    )
                    continue

                if signal == 0:
                    print(f"{symbol}: no trade (no_signal)")
                    cycle_counts["skip_no_signal"] += 1
                    append_jsonl(
                        run_log,
                        {
                            "event": "skip",
                            "symbol": symbol,
                            "reason": "no_signal",
                            "signal": signal,
                            "equity": equity,
                            "broker_time": broker_time,
                        },
                    )
                    continue

                if broker and not args.dry_run:
                    open_positions_for_symbol = broker.positions_get(symbol=symbol) or []
                    if open_positions_for_symbol:
                        current_tick = broker.mt5.symbol_info_tick(symbol)
                        if current_tick is not None:
                            current_price_for_status = current_tick.bid if signal == -1 else current_tick.ask
                            print(f"{symbol}: {profit_status_line(open_positions_for_symbol[0], current_price_for_status, mgmt)}")

                can_trade, gate_reason = guard.can_trade_with_floating(
                    equity=equity,
                    floating_pnl=floating_pnl,
                    open_positions=len(open_positions),
                    day=current_day,
                )
                if not can_trade:
                    print(f"{symbol}: trading paused ({gate_reason})")
                    cycle_counts["skip_risk_gate"] += 1
                    append_jsonl(
                        run_log,
                        {
                            "event": "skip_risk_gate",
                            "symbol": symbol,
                            "reason": gate_reason,
                            "equity": equity,
                            "floating_pnl": floating_pnl,
                            "open_positions": len(open_positions),
                            "broker_time": broker_time,
                        },
                    )
                    continue

                stop, target = risk.calculate_sl_tp(signal, price, atr)
                size = risk.calculate_position_size(equity, price, stop, atr=atr)
                size = max(0.01, round(min(size, 0.25), 2))
                if broker and not args.dry_run:
                    hard_cap = max_volume_for_loss(
                        broker=broker,
                        symbol=symbol,
                        direction=signal,
                        entry_price=price,
                        stop_price=stop,
                        max_loss_usd=mgmt["max_loss_per_trade_usd"],
                    )
                    if hard_cap is not None:
                        size = min(size, hard_cap)
                        size = max(0.01, round(size, 2))
                        estimated_loss = broker.order_calc_profit(signal, symbol, size, price, stop)
                        estimated_loss = abs(float(estimated_loss)) if estimated_loss is not None else None
                        print(
                            f"{symbol}: risk check | size_cap={hard_cap:.2f} | "
                            f"final_size={size:.2f} | est_max_loss={estimated_loss:.2f}"
                        )
                        if estimated_loss is not None and estimated_loss > mgmt["max_loss_per_trade_usd"]:
                            print(
                                f"{symbol}: skipped because est_max_loss={estimated_loss:.2f} "
                                f"exceeds hard limit={mgmt['max_loss_per_trade_usd']:.2f}"
                            )
                            cycle_counts["skip_hard_loss_cap"] += 1
                            append_jsonl(
                                run_log,
                                {
                                    "event": "skip_hard_loss_cap",
                                    "symbol": symbol,
                                    "estimated_loss": estimated_loss,
                                    "hard_limit": mgmt["max_loss_per_trade_usd"],
                                    "broker_time": broker_time,
                                },
                            )
                            continue
                if size <= 0:
                    print(f"{symbol}: skipped due to zero size")
                    cycle_counts["skip_zero_size"] += 1
                    append_jsonl(run_log, {"event": "skip_zero_size", "symbol": symbol, "broker_time": broker_time})
                    continue

                if broker and not args.dry_run and broker.positions_total(symbol) >= 1:
                    print(f"{symbol}: trading paused (max_one_trade_per_symbol)")
                    cycle_counts["skip_open_position"] += 1
                    append_jsonl(run_log, {"event": "skip_open_position", "symbol": symbol, "reason": "max_one_trade_per_symbol", "broker_time": broker_time})
                    continue

                print(
                    f"{symbol}: strategy={strategy.__class__.__name__} signal={signal} "
                    f"price={price:.5f} size={size:.2f} sl={stop:.5f} tp={target:.5f}"
                )
                spread_points = None
                spread_price = None
                spread_usd_est = None
                if broker is not None:
                    try:
                        info = broker.symbol_info(symbol)
                        if info is not None:
                            spread_points = getattr(info, "spread", None)
                            point = getattr(info, "point", None)
                            if spread_points is not None and point:
                                spread_price = spread_points * point
                                contract_size = getattr(info, "trade_contract_size", 100000.0) or 100000.0
                                spread_usd_est = spread_price * contract_size * size
                    except Exception:
                        pass
                append_jsonl(
                    run_log,
                    {
                        "event": "signal",
                        "symbol": symbol,
                        "strategy": strategy.__class__.__name__,
                        "signal": signal,
                        "score": entry_score,
                        "price": price,
                        "size": size,
                        "stop": stop,
                        "target": target,
                        "equity": equity,
                        "broker_time": broker_time,
                        "spread_points": spread_points,
                        "spread_price": spread_price,
                        "spread_usd_est": spread_usd_est,
                    },
                )
                cycle_counts["signals"] += 1

                if broker and not args.dry_run:
                    result = broker.place_order(
                        symbol=symbol,
                        direction=signal,
                        volume=size,
                        stop_loss=stop,
                        take_profit=target,
                    )
                    print(f"{symbol}: order result={result} magic={args.magic_number}")
                    cycle_counts["orders_sent"] += 1
                    try:
                        retcode = getattr(result, "retcode", None)
                        if retcode is not None and retcode != broker.mt5.TRADE_RETCODE_DONE:
                            append_jsonl(
                                run_log,
                                {
                                    "event": "order_rejected",
                                    "symbol": symbol,
                                    "retcode": retcode,
                                    "comment": getattr(result, "comment", ""),
                                    "broker_time": broker_time,
                                },
                            )
                        else:
                            append_jsonl(
                                run_log,
                                {
                                    "event": "order_accepted",
                                    "symbol": symbol,
                                    "result": str(result),
                                    "order": getattr(result, "order", None),
                                    "deal": getattr(result, "deal", None),
                                    "score": entry_score,
                                    "magic": args.magic_number,
                                    "broker_time": broker_time,
                                },
                            )
                    except Exception:
                        pass

        append_jsonl(
            run_log,
            {
                "event": "cycle_summary",
                "time": datetime.now(timezone.utc),
                "summary": dict(cycle_counts),
            },
        )
        with summary_file.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "day": current_day.isoformat(),
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "summary": dict(cycle_counts),
                },
                handle,
                indent=2,
            )

        if args.loop_once:
            break

        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        sleep_for = max(1, args.poll_seconds - int(elapsed))
        print(f"Sleeping {sleep_for}s")

        import time
        time.sleep(sleep_for)

    if broker and not args.dry_run:
        broker.shutdown()

    append_jsonl(run_log, {"event": "runner_stop", "time": datetime.now(timezone.utc)})


if __name__ == "__main__":
    main()
