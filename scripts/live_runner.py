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


def calculate_trade_volume(broker, symbol, direction, entry_price, stop_price, account_equity, risk_per_trade):
    risk_amount = account_equity * risk_per_trade
    stop_distance = abs(entry_price - stop_price)
    if stop_distance <= 0:
        return 0.0, "invalid_stop"

    info = broker.symbol_info(symbol)
    if info is None:
        return 0.0, "no_symbol_info"

    tick_value = float(getattr(info, "trade_tick_value", 0.0) or 0.0)
    tick_size = float(getattr(info, "trade_tick_size", 0.0) or 0.0)
    contract_size = float(getattr(info, "trade_contract_size", 0.0) or 0.0)
    ask_price = entry_price

    if tick_value > 0 and tick_size > 0:
        value_per_price_unit = tick_value / tick_size
        raw_volume = risk_amount / (stop_distance * value_per_price_unit)
    elif contract_size > 0:
        raw_volume = risk_amount / (stop_distance * contract_size)
    else:
        return 0.0, "no_symbol_pricing"

    volume = broker.normalize_volume(symbol, raw_volume)
    min_volume = float(getattr(info, "volume_min", 0.01) or 0.01)
    volume_step = float(getattr(info, "volume_step", 0.01) or 0.01)

    while volume >= min_volume:
        margin = broker.order_calc_margin(direction, symbol, volume, ask_price)
        if margin is not None and margin <= account_equity * 0.85:
            return volume, "ok"
        volume = broker.normalize_volume(symbol, volume - volume_step)

    return 0.0, "insufficient_margin"


def trade_management_params():
    return {
        "breakeven_at_r": 0.2,
        "trail_at_r": 0.4,
        "trail_buffer_r": 0.25,
        "max_bars": 48,
        "profit_fade_pct": 0.35,
        "profit_floor_r": 0.25,
    }


def manage_live_position(broker, position, current_price, mgmt):
    risk = abs(position.price_open - position.sl)
    if risk <= 0:
        return None, "invalid_risk"

    current_pnl = (current_price - position.price_open) if position.type == broker.mt5.POSITION_TYPE_BUY else (position.price_open - current_price)
    peak_pnl = float(getattr(position, "profit", 0.0) or 0.0)
    if peak_pnl <= 0:
        peak_pnl = current_pnl

    if peak_pnl >= risk * mgmt["profit_floor_r"] and current_pnl <= peak_pnl * (1 - mgmt["profit_fade_pct"]):
        close_result = broker.close_position(position.ticket, position.symbol, position.volume, 1 if position.type == broker.mt5.POSITION_TYPE_BUY else -1)
        return close_result, "profit_fade"

    open_r = current_pnl / risk
    new_sl = position.sl

    if open_r >= mgmt["breakeven_at_r"]:
        if position.type == broker.mt5.POSITION_TYPE_BUY:
            new_sl = max(new_sl, position.price_open)
        else:
            new_sl = min(new_sl, position.price_open)

    if open_r >= mgmt["trail_at_r"]:
        trail_distance = risk * mgmt["trail_buffer_r"]
        if position.type == broker.mt5.POSITION_TYPE_BUY:
            new_sl = max(new_sl, current_price - trail_distance)
        else:
            new_sl = min(new_sl, current_price + trail_distance)

    if new_sl != position.sl:
        mod_result = broker.modify_position(position.ticket, position.symbol, new_sl, position.tp)
        return mod_result, "modify_sl"

    return None, "hold"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=["EURUSD", "GBPUSD", "XAUUSD"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--loop-once", action="store_true")
    parser.add_argument("--max-consecutive-losses", type=int, default=3)
    parser.add_argument("--cooldown-hours", type=int, default=3)
    args = parser.parse_args()

    router = StrategyRouter()
    rules = FtmoRules(initial_balance=10000, max_consecutive_losses=args.max_consecutive_losses)
    guard = FtmoRiskGuard(rules)
    risk = RiskManager(risk_per_trade=rules.max_risk_per_trade_pct)
    log_dir = ensure_log_dir()
    cooldown_until = None
    last_deal_check = None
    last_closed_pnl = None
    active_positions = 0

    broker = None
    if not args.dry_run:
        try:
            broker = MT5BrokerAdapter()
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
                for position in positions:
                    symbol = getattr(position, "symbol", "")
                    tick = broker.mt5.symbol_info_tick(symbol)
                    if tick is None:
                        continue
                    current_price = tick.bid if getattr(position, "type", 0) == broker.mt5.POSITION_TYPE_BUY else tick.ask
                    mgmt = trade_management_params()
                    result, action = manage_live_position(broker, position, current_price, mgmt)
                    if result is not None:
                        append_jsonl(
                            run_log,
                            {
                                "event": "position_manage",
                                "symbol": symbol,
                                "ticket": getattr(position, "ticket", None),
                                "action": action,
                                "result": str(result),
                                "time": started,
                            },
                        )
                        print(f"{symbol}: manage action={action} result={result}")

        if broker and not args.dry_run and last_deal_check is not None:
            closed_deals = broker.history_deals_since(last_deal_check, magic=26072026)
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

                ok, reason = guard.can_trade(
                    equity,
                    open_positions=broker.positions_total(symbol) if broker and not args.dry_run else 0,
                    day=broker_time.date(),
                    safety_buffer_pct=0.5,
                )
                if not ok:
                    print(f"{symbol}: trading paused ({reason})")
                    append_jsonl(
                        run_log,
                        {
                            "event": "trade_paused",
                            "symbol": symbol,
                            "reason": reason,
                            "equity": equity,
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

                stop, target = risk.calculate_sl_tp(signal, price, atr)
                if broker and not args.dry_run:
                    size, size_reason = calculate_trade_volume(
                        broker=broker,
                        symbol=symbol,
                        direction=signal,
                        entry_price=price,
                        stop_price=stop,
                        account_equity=equity,
                        risk_per_trade=rules.max_risk_per_trade_pct,
                    )
                else:
                    size = risk.calculate_position_size(equity, price, stop, atr=atr)
                    size_reason = "dry_run"

                if size <= 0:
                    print(f"{symbol}: skipped due to sizing ({size_reason})")
                    cycle_counts["skip_zero_size"] += 1
                    append_jsonl(run_log, {"event": "skip_zero_size", "symbol": symbol, "reason": size_reason, "broker_time": broker_time})
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
                        "size_reason": size_reason,
                        "broker_time": broker_time,
                    },
                )
                cycle_counts["signals"] += 1

                if broker and not args.dry_run:
                    active_positions = broker.positions_total(symbol)
                    if active_positions >= 1:
                        print(f"{symbol}: skipped because position already open")
                        cycle_counts["skip_open_position"] += 1
                        continue
                    result = broker.place_order(
                        symbol=symbol,
                        direction=signal,
                        volume=size,
                        stop_loss=stop,
                        take_profit=target,
                    )
                    print(f"{symbol}: order result={result}")
                    accepted = getattr(result, "retcode", None) == broker.mt5.TRADE_RETCODE_DONE
                    cycle_counts["orders_sent"] += int(bool(accepted))
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
                                    "broker_time": broker_time,
                                },
                            )
                    except Exception:
                        pass
                    append_jsonl(
                        run_log,
                        {
                            "event": "trade_management",
                            "symbol": symbol,
                            "params": mgmt,
                            "broker_time": broker_time,
                        },
                    )

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
