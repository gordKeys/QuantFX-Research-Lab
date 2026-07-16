"""
Exit-logic backtest / walk-forward harness.

The point of this script is narrow and deliberate: it does NOT reimplement
its own simplified exit model (the way engine/backtester.py does, with its
own hardcoded breakeven/trail constants). It imports and replays the actual
`manage_live_position()` function from scripts/live_runner.py bar-by-bar over
real historical M5 data, so "does the exit-logic change help" gets answered
against the exact code that will run live, not a proxy for it.

It runs the SAME entry signals and SAME historical path through two exit
configs -- OLD (pre-fix: giveback_usd_buffer forced to 0, so only the
R-multiple-based giveback close is active) and NEW (current tiers, with the
dollar-based giveback safety net) -- and reports the entry-vs-exit
attribution for each, so you can see whether the peaked-then-gave-it-back
bucket actually shrinks before this touches a live account.

Caveats (read before trusting the numbers):
- Intrabar path is unknown from OHLC bars alone. Each bar is checked
  adverse-side-first (low before high for BUY, high before low for SELL),
  which is the standard conservative backtesting convention -- it will not
  make the new logic look better than it is by feeding it favorable-first
  paths.
- Dollar PnL is derived by normalizing risk to --risk-usd per trade (same
  idea as live position sizing: risk_amount / stop_distance), not MT5's
  actual tick value tables, so it should be treated as directionally
  correct rather than penny-accurate.
- This only validates the exit-management change. It does not and cannot
  validate entry-signal quality, which the loss_diagnostics report showed
  is the bigger lever right now.

Usage:
    python scripts/exit_logic_backtest.py --symbol EURUSD
    python scripts/exit_logic_backtest.py --symbol XAUUSD --risk-usd 20
"""
from bootstrap import add_project_root
add_project_root()

import argparse
import copy
from datetime import timezone

import pandas as pd

from engine.data_loader import DataLoader
from engine.features import FeatureEngine
from engine.risk_manager import RiskManager
from strategy_router import StrategyRouter
from live_runner import manage_live_position, trade_management_params


class _MT5Consts:
    POSITION_TYPE_BUY = 0
    POSITION_TYPE_SELL = 1


class FakeBroker:
    """Minimal stand-in for MT5BrokerAdapter -- only what manage_live_position touches."""

    mt5 = _MT5Consts()

    def __init__(self):
        self.last_close_reason = None

    def close_position(self, position):
        position.closed = True
        return "closed"

    def modify_position(self, ticket, symbol, sl, tp):
        return "modified"


class FakePosition:
    __slots__ = ["ticket", "symbol", "type", "price_open", "sl", "tp", "time", "profit", "closed"]

    def __init__(self, ticket, symbol, direction, price_open, sl, tp, open_time):
        self.ticket = ticket
        self.symbol = symbol
        self.type = 0 if direction == 1 else 1
        self.price_open = price_open
        self.sl = sl
        self.tp = tp
        self.time = open_time
        self.profit = 0.0
        self.closed = False


def _load_symbol_data(symbol):
    return FeatureEngine().add_features(DataLoader(symbol=symbol).load())


def _simulate(symbol, data, strategy, mgmt_overrides, risk_usd, sl_atr, tp_atr):
    """Replay manage_live_position bar-by-bar. Returns a list of trade dicts."""
    signals = strategy.generate_signals(data)
    risk_mgr = RiskManager()
    broker = FakeBroker()
    tracker = {}

    trades = []
    position = None
    ticket_seq = 0
    dollar_per_unit = None  # calibrated per-trade from risk_usd / stop_distance
    entry_mfe = 0.0
    entry_mae = 0.0

    index = data.index
    for i in range(len(data)):
        bar_time = index[i].to_pydatetime()
        if bar_time.tzinfo is None:
            bar_time = bar_time.replace(tzinfo=timezone.utc)
        open_ = float(data["open"].iloc[i])
        high = float(data["high"].iloc[i])
        low = float(data["low"].iloc[i])
        close = float(data["close"].iloc[i])
        atr = data["atr"].iloc[i] if "atr" in data.columns else None
        sig = int(signals.iloc[i]) if i < len(signals) else 0

        if position is None:
            if sig != 0 and pd.notna(atr) and atr > 0:
                stop, target = risk_mgr.calculate_sl_tp(sig, close, atr, sl_atr=sl_atr, tp_atr=tp_atr)
                stop_distance = abs(close - stop)
                if stop_distance <= 0:
                    continue
                dollar_per_unit = risk_usd / stop_distance
                ticket_seq += 1
                position = FakePosition(ticket_seq, symbol, sig, close, stop, target, bar_time)
                entry_mfe = 0.0
                entry_mae = 0.0
            continue

        # position is open: walk this bar's path, adverse side first
        is_buy = position.type == 0
        checkpoints = [low, high] if is_buy else [high, low]
        closed_this_bar = False
        for price in checkpoints:
            pnl = (price - position.price_open) if is_buy else (position.price_open - price)
            pnl_usd = pnl * dollar_per_unit
            entry_mfe = max(entry_mfe, pnl_usd)
            entry_mae = min(entry_mae, pnl_usd)
            position.profit = pnl_usd

            result, action = manage_live_position(broker, position, price, bar_time, mgmt_overrides, tracker)
            if position.closed:
                trades.append(
                    {
                        "symbol": symbol,
                        "direction": "BUY" if is_buy else "SELL",
                        "open_time": position.time,
                        "close_time": bar_time,
                        "entry_price": position.price_open,
                        "exit_price": price,
                        "profit_usd": pnl_usd,
                        "mfe_usd": entry_mfe,
                        "mae_usd": entry_mae,
                        "reason": action,
                    }
                )
                tracker.pop(position.ticket, None)
                position = None
                closed_this_bar = True
                break
        if closed_this_bar:
            continue

    return trades


def _attribution(trades):
    if not trades:
        return {
            "trades": 0, "total_pnl": 0.0, "never_profit_n": 0, "never_profit_pnl": 0.0,
            "gaveback_n": 0, "gaveback_pnl": 0.0, "kept_n": 0, "kept_pnl": 0.0,
            "conversion_rate": None,
        }
    df = pd.DataFrame(trades)
    never = df[df["mfe_usd"] <= 0]
    peaked = df[df["mfe_usd"] > 0]
    gaveback = peaked[peaked["profit_usd"] <= 0]
    kept = peaked[peaked["profit_usd"] > 0]
    return {
        "trades": len(df),
        "total_pnl": df["profit_usd"].sum(),
        "never_profit_n": len(never),
        "never_profit_pnl": never["profit_usd"].sum(),
        "gaveback_n": len(gaveback),
        "gaveback_pnl": gaveback["profit_usd"].sum(),
        "kept_n": len(kept),
        "kept_pnl": kept["profit_usd"].sum(),
        "conversion_rate": (len(kept) / len(peaked)) if len(peaked) else None,
    }


def _print_report(label, stats):
    print(f"\n--- {label} ---")
    print(f"Trades: {stats['trades']} | Total PnL: {stats['total_pnl']:.2f}")
    print(f"Never reached profit: {stats['never_profit_n']} trades, PnL {stats['never_profit_pnl']:.2f}")
    print(f"Peaked then gave it back: {stats['gaveback_n']} trades, PnL {stats['gaveback_pnl']:.2f}")
    print(f"Peaked and kept some/all: {stats['kept_n']} trades, PnL {stats['kept_pnl']:.2f}")
    if stats["conversion_rate"] is not None:
        print(f"Peak->profit conversion rate: {stats['conversion_rate']:.1%}")


def main():
    parser = argparse.ArgumentParser(description="Replay manage_live_position over history to A/B the exit-logic fix")
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--risk-usd", type=float, default=20.0, help="Target $ risk per trade (position-size normalizer)")
    parser.add_argument("--sl-atr", type=float, default=1.5)
    parser.add_argument("--tp-atr", type=float, default=4.0)
    args = parser.parse_args()

    data = _load_symbol_data(args.symbol)
    router = StrategyRouter()
    strategy = router.get_strategy(args.symbol)
    print(f"Symbol: {args.symbol} | bars: {len(data)} | strategy: {router.get_strategy_name(args.symbol)}")

    new_mgmt = trade_management_params(args.symbol)
    old_mgmt = copy.deepcopy(new_mgmt)
    old_mgmt["giveback_usd_buffer"] = 0.0  # pre-fix behaviour: R-based giveback close only

    old_trades = _simulate(args.symbol, data, strategy, old_mgmt, args.risk_usd, args.sl_atr, args.tp_atr)
    new_trades = _simulate(args.symbol, data, strategy, new_mgmt, args.risk_usd, args.sl_atr, args.tp_atr)

    old_stats = _attribution(old_trades)
    new_stats = _attribution(new_trades)

    _print_report("OLD (R-based giveback close only, pre-fix)", old_stats)
    _print_report("NEW (+ dollar-based giveback safety net)", new_stats)

    print("\n=== DELTA (new - old) ===")
    print(f"Total PnL: {new_stats['total_pnl'] - old_stats['total_pnl']:+.2f}")
    if old_stats["conversion_rate"] is not None and new_stats["conversion_rate"] is not None:
        print(
            f"Peak->profit conversion rate: {old_stats['conversion_rate']:.1%} -> "
            f"{new_stats['conversion_rate']:.1%}"
        )
    print(
        "\nRemember: this only tests the exit-management change on this symbol's "
        "signal + history combo. A win here doesn't validate entry quality, and a "
        "symbol with too few trades in its local CSV won't give a reliable read -- "
        "check the trade count above before trusting the delta."
    )


if __name__ == "__main__":
    main()
