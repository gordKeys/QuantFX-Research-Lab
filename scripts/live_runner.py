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
    return int(signal_series.iloc[-1]), strategy


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


def trade_management_params():
    return {
        "breakeven_at_r": 0.25,
        "trail_at_r": 0.40,
        "trail_buffer_r": 0.18,
        "max_minutes": 30,
        "max_bars": 18,
        "profit_fade_pct": 0.35,
        "profit_floor_r": 0.50,
        "warn_loss_per_trade_usd": 11.0,
        "soft_loss_per_trade_usd": 13.5,
        "max_loss_per_trade_usd": 15.0,
    }


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


def manage_live_position(broker, position, current_price, current_time, mgmt):
    risk = abs(position.price_open - position.sl)
    if risk <= 0:
        return None, "invalid_risk"

    is_buy = position.type == broker.mt5.POSITION_TYPE_BUY
    current_pnl = (current_price - position.price_open) if is_buy else (position.price_open - current_price)
    open_r = current_pnl / risk
    current_pnl_usd = float(getattr(position, "profit", current_pnl) or current_pnl)

    peak_pnl = float(getattr(position, "profit", 0.0) or 0.0)
    if peak_pnl <= 0:
        peak_pnl = current_pnl

    position_time = getattr(position, "time", None)
    held_minutes = 0.0
    if position_time is not None:
        if isinstance(position_time, (int, float)):
            position_time = datetime.fromtimestamp(position_time, tz=timezone.utc)
        elif getattr(position_time, "tzinfo", None) is None:
            position_time = position_time.replace(tzinfo=timezone.utc)
        held_minutes = (current_time - position_time).total_seconds() / 60.0

    if current_pnl_usd <= -mgmt["soft_loss_per_trade_usd"]:
        return broker.close_position(position), "soft_dollar_stop"

    if current_pnl_usd <= -mgmt["warn_loss_per_trade_usd"]:
        return broker.close_position(position), "warn_dollar_stop"

    if peak_pnl >= risk * mgmt["profit_floor_r"] and current_pnl <= peak_pnl * (1 - mgmt["profit_fade_pct"]):
        return broker.close_position(position), "profit_fade"

    if current_pnl_usd <= -mgmt["max_loss_per_trade_usd"]:
        return broker.close_position(position), "hard_dollar_stop"

    if held_minutes >= mgmt["max_minutes"] and open_r <= -0.18:
        return broker.close_position(position), "loss_cut"

    if held_minutes >= mgmt["max_bars"] * 5 and open_r < 0:
        return broker.close_position(position), "time_stop"

    new_sl = position.sl
    if open_r >= mgmt["breakeven_at_r"]:
        new_sl = max(new_sl, position.price_open) if is_buy else min(new_sl, position.price_open)

    if open_r >= mgmt["trail_at_r"]:
        trail_distance = risk * mgmt["trail_buffer_r"]
        new_sl = max(new_sl, current_price - trail_distance) if is_buy else min(new_sl, current_price + trail_distance)

    if new_sl != position.sl:
        return broker.modify_position(position.ticket, position.symbol, new_sl, position.tp), "modify_sl"

    return None, "hold"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=["EURUSD", "GBPUSD"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--loop-once", action="store_true")
    parser.add_argument("--max-consecutive-losses", type=int, default=2)
    parser.add_argument("--cooldown-hours", type=int, default=6)
    parser.add_argument("--magic-number", type=int, default=26072026)
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

    broker = None
    if not args.dry_run:
        try:
            broker = MT5BrokerAdapter(magic_number=args.magic_number)
            broker.initialize()
            last_deal_check = datetime.now(timezone.utc) - timedelta(minutes=5)
        except MT5UnavailableError as exc:
            print(f"MT5 unavailable, falling back to dry-run: {exc}")
            args.dry_run = True

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
                mgmt = trade_management_params()
                for position in positions:
                    symbol = getattr(position, "symbol", "")
                    tick = broker.mt5.symbol_info_tick(symbol)
                    if tick is None:
                        continue
                    current_price = tick.bid if getattr(position, "type", 0) == broker.mt5.POSITION_TYPE_BUY else tick.ask
                    action_result, action = manage_live_position(broker, position, current_price, started, mgmt)
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
            cooldown_until = started + timedelta(hours=args.cooldown_hours)
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
                    cooldown_until = started + timedelta(hours=args.cooldown_hours)
                    print(f"3 consecutive losses reached; pausing until {cooldown_until.isoformat()}")
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

        for symbol in args.symbols:
            print(format_status(symbol, guard.consecutive_losses, cooldown_until, last_closed_pnl))
            with timed(f"{symbol} evaluation"):
                data = build_data_for_symbol(symbol, broker=broker if not args.dry_run else None)
                if data is None or data.empty:
                    print(f"{symbol}: no data available")
                    append_jsonl(run_log, {"event": "no_data", "symbol": symbol, "time": datetime.now(timezone.utc)})
                    cycle_counts["no_data"] += 1
                    continue
                signal, strategy = latest_signal(symbol, data, router)
                price = float(data["close"].iloc[-1])
                atr = float(data["atr"].iloc[-1])
                broker_time = data.index[-1].to_pydatetime()
                mgmt = trade_management_params()

                if broker and not args.dry_run:
                    equity = broker.account_equity() or rules.initial_balance
                else:
                    equity = rules.initial_balance

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
                append_jsonl(
                    run_log,
                    {
                        "event": "signal",
                        "symbol": symbol,
                        "strategy": strategy.__class__.__name__,
                        "signal": signal,
                        "price": price,
                        "size": size,
                        "stop": stop,
                        "target": target,
                        "equity": equity,
                        "broker_time": broker_time,
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
