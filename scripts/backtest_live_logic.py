"""
Replay the ACTUAL live exit-management code against historical bars.

Why this exists: engine/backtester.py (used by the walkforward/tournament/
sweep scripts) has its own hard-coded exit rules (breakeven_at_r=2.0,
trail_at_r=4.0, fixed 3R take-profit, etc.) that are completely different
from -- and don't share a single line with -- trade_management_params() /
manage_live_position() in live_runner.py, which is what actually runs live.
That means none of the walkforward/tournament reports validate the exit
tuning you're actually trading with. This script closes that gap: it drives
manage_live_position() itself, bar by bar, against historical M5 data, so a
change to the live exit tiers can be checked against history before it goes
live, using the exact same code path (not a re-implementation of it that can
drift out of sync).

IMPORTANT APPROXIMATIONS (read before trusting exact dollar figures):
  - Position sizing/PnL conversion to USD uses a simple contract-size model
    (100000 units for FX majors quoted as CCY/USD, 100 oz for XAUUSD) rather
    than MT5's order_calc_profit, which isn't available offline. This is
    accurate for EURUSD/GBPUSD/XAUUSD-style quoting but is NOT currently
    correct for USD-base pairs like USDJPY/USDCHF (their JPY/CHF-denominated
    PnL needs an extra FX conversion this script does not do) -- and those
    symbols don't have local CSVs to test with anyway. Only run this against
    symbols in AVAILABLE_SYMBOLS below until more data/conversion is added.
  - Fills happen at bar close, not intrabar -- a real stop/giveback trigger
    could fire a few pips earlier or later intrabar. This affects exact PnL,
    not the qualitative comparison between exit-tier configurations, which
    is what this script is for.
  - Spread/slippage/commission are not modeled.

Use this for RELATIVE comparison (old tier vs new tier, on the same bars),
not as a precise live PnL forecast.

Usage:
    python scripts/backtest_live_logic.py --symbol EURUSD --compare-old
"""
from bootstrap import add_project_root
add_project_root()

import argparse
from datetime import timezone
from types import SimpleNamespace

import pandas as pd

from engine.data_loader import DataLoader
from engine.features import FeatureEngine
from engine.risk_manager import RiskManager
from strategy_router import StrategyRouter
from live_runner import manage_live_position, trade_management_params as new_trade_management_params


# Contract-size model for converting a price move into a USD PnL, per lot.
# (contract_size, needs_usd_conversion): for CCY/USD pairs (EURUSD, GBPUSD,
# AUDUSD) and XAUUSD, PnL is already USD-denominated -- no conversion needed.
# For USD/CCY pairs (USDJPY, USDCHF), USD is the BASE currency, so the raw
# price-move PnL comes out in JPY/CHF and must be divided by the current
# price to convert back to USD. Getting this wrong is why USDJPY/USDCHF were
# excluded from AVAILABLE_SYMBOLS before -- they'd have silently produced
# PnL in the wrong currency's units.
SYMBOL_MODEL = {
    "EURUSD": (100_000.0, False),
    "GBPUSD": (100_000.0, False),
    "AUDUSD": (100_000.0, False),
    "USDCHF": (100_000.0, True),
    "USDJPY": (100_000.0, True),
    "XAUUSD": (100.0, False),
}
AVAILABLE_SYMBOLS = tuple(SYMBOL_MODEL.keys())

POSITION_TYPE_BUY = 0
POSITION_TYPE_SELL = 1


def _old_trade_management_params(symbol=None):
    """Snapshot of the tiers as they were before this round of tuning, kept
    here (not in live_runner.py) purely so this script can A/B the old vs
    new exit logic on the same historical bars. If you tune the live tiers
    again later, update THIS snapshot to whatever you just moved away from,
    not the other way around."""
    base = {
        "breakeven_at_r": 1.00,
        "trail_at_r": 1.60,
        "trail_buffer_r": 0.65,
        "giveback_trigger_r": 1.40,
        "giveback_buffer_r": 0.50,
        "min_peak_profit_usd": 4.0,
        "giveback_usd_buffer": 0.0,  # didn't exist in the old tiers
        "max_minutes": 180,
        "max_bars": 48,
        "warn_loss_per_trade_usd": 11.0,
        "soft_loss_per_trade_usd": 13.5,
        "max_loss_per_trade_usd": 15.0,
        "quick_cut_minutes": 0.0,  # didn't exist in the old tiers
        "quick_cut_loss_usd": 0.0,
    }
    symbol = (symbol or "").upper()
    if symbol == "EURUSD":
        base.update(
            {
                "breakeven_at_r": 0.45,
                "trail_at_r": 0.80,
                "trail_buffer_r": 0.28,
                "giveback_trigger_r": 0.72,
                "giveback_buffer_r": 0.18,
                "min_peak_profit_usd": 2.0,
                "max_minutes": 120,
                "max_bars": 28,
            }
        )
    elif symbol in {"USDJPY", "USDCHF"}:
        base.update(
            {
                "breakeven_at_r": 0.80,
                "trail_at_r": 1.45,
                "trail_buffer_r": 0.58,
                "giveback_trigger_r": 1.20,
                "giveback_buffer_r": 0.30,
                "min_peak_profit_usd": 3.0,
                "max_minutes": 150,
                "max_bars": 36,
            }
        )
    elif symbol == "AUDUSD":
        base.update(
            {
                "breakeven_at_r": 0.70,
                "trail_at_r": 1.35,
                "trail_buffer_r": 0.52,
                "giveback_trigger_r": 1.10,
                "giveback_buffer_r": 0.28,
                "min_peak_profit_usd": 3.0,
                "max_minutes": 165,
                "max_bars": 40,
            }
        )
    return base


class SimBroker:
    """Minimal duck-typed stand-in for MT5BrokerAdapter, just enough for
    manage_live_position() to run unmodified against it."""

    def __init__(self):
        self.mt5 = SimpleNamespace(POSITION_TYPE_BUY=POSITION_TYPE_BUY, POSITION_TYPE_SELL=POSITION_TYPE_SELL)
        self.closed = []

    def close_position(self, position):
        self.closed.append(position.ticket)
        return "closed"

    def modify_position(self, ticket, symbol, sl, tp):
        # The position object is mutated directly by the simulator after
        # this call returns (see _simulate_symbol), mirroring how a real
        # position's .sl reflects the broker's new state on the next poll.
        return "modified"


def _pnl_usd(symbol, direction, entry_price, current_price, lots):
    contract_size, needs_usd_conversion = SYMBOL_MODEL[symbol]
    diff = (current_price - entry_price) if direction == 1 else (entry_price - current_price)
    raw = diff * contract_size * lots
    if needs_usd_conversion:
        # raw is denominated in the quote currency (JPY/CHF); convert to USD
        # using the current price (USD is the base currency for these pairs).
        return raw / current_price
    return raw


def _position_size(symbol, entry_price, stop_price, max_loss_usd):
    """Approximate max_volume_for_loss()'s intent offline: size so the loss
    at the stop is roughly max_loss_usd, capped the same way live_runner
    caps it (0.01 to 0.25 lots)."""
    contract_size, needs_usd_conversion = SYMBOL_MODEL[symbol]
    stop_distance = abs(entry_price - stop_price)
    if stop_distance <= 0:
        return 0.01
    loss_per_lot = stop_distance * contract_size
    if needs_usd_conversion:
        loss_per_lot = loss_per_lot / entry_price
    size = max_loss_usd / loss_per_lot if loss_per_lot > 0 else 0.25
    return max(0.01, round(min(size, 0.25), 2))


def _simulate_symbol(symbol, data, strategy, mgmt_fn, risk, max_loss_usd_default=15.0):
    signals = strategy.generate_signals(data)
    trades = []
    tracker = {}
    broker = SimBroker()

    position = None
    ticket_counter = 0

    for i in range(len(data)):
        bar = data.iloc[i]
        current_time = data.index[i]
        if getattr(current_time, "tzinfo", None) is None:
            current_time = current_time.tz_localize(timezone.utc)
        price = float(bar["close"])
        atr = float(bar["atr"])

        if position is not None:
            mgmt = mgmt_fn(symbol)
            position.profit = _pnl_usd(symbol, 1 if position.type == POSITION_TYPE_BUY else -1, position.price_open, price, position.volume)
            result, action = manage_live_position(broker, position, price, current_time, mgmt, tracker)
            if action == "modify_sl":
                # manage_live_position computed a new SL internally and only
                # returned a broker-call result (which live gets applied by
                # the broker before the next fetch); recompute the same new
                # SL here identically and apply it directly, using the SL as
                # of THIS call (before we touch it) as the risk denominator --
                # that matches how manage_live_position computed "risk"
                # internally for this same call.
                risk_dist = abs(position.price_open - position.sl)
                is_buy = position.type == POSITION_TYPE_BUY
                current_pnl = (price - position.price_open) if is_buy else (position.price_open - price)
                open_r = current_pnl / risk_dist if risk_dist else 0.0
                new_sl = position.sl
                if open_r >= mgmt["breakeven_at_r"]:
                    new_sl = max(new_sl, position.price_open) if is_buy else min(new_sl, position.price_open)
                if open_r >= mgmt["trail_at_r"]:
                    trail_distance = risk_dist * mgmt["trail_buffer_r"]
                    new_sl = max(new_sl, price - trail_distance) if is_buy else min(new_sl, price + trail_distance)
                position.sl = new_sl
            elif action not in ("hold", "invalid_risk"):
                peak = tracker.get(position.ticket, {})
                trades.append(
                    {
                        "symbol": symbol,
                        "direction": "BUY" if position.type == POSITION_TYPE_BUY else "SELL",
                        "open_time": position.time,
                        "close_time": current_time,
                        "entry_price": position.price_open,
                        "exit_price": price,
                        "profit_usd": position.profit,
                        "peak_profit_usd": peak.get("peak_profit_usd", position.profit),
                        "close_reason": action,
                    }
                )
                tracker.pop(position.ticket, None)
                position = None

        if position is None:
            signal = int(signals.iloc[i])
            if signal == 0:
                continue
            stop, target = risk.calculate_sl_tp(signal, price, atr)
            mgmt = mgmt_fn(symbol)
            volume = _position_size(symbol, price, stop, mgmt.get("max_loss_per_trade_usd", max_loss_usd_default))
            ticket_counter += 1
            position = SimpleNamespace(
                ticket=ticket_counter,
                symbol=symbol,
                type=POSITION_TYPE_BUY if signal == 1 else POSITION_TYPE_SELL,
                price_open=price,
                sl=stop,
                tp=target,
                volume=volume,
                time=current_time,
                profit=0.0,
            )

    return pd.DataFrame(trades)


def _summarize(trades: pd.DataFrame, label: str):
    print(f"\n--- {label} ---")
    if trades.empty:
        print("No trades generated.")
        return
    total_pnl = trades["profit_usd"].sum()
    win_rate = (trades["profit_usd"] > 0).mean()
    never_profit = trades[trades["peak_profit_usd"] <= 0]
    peaked = trades[trades["peak_profit_usd"] > 0]
    gave_back = peaked[peaked["profit_usd"] <= 0]
    kept = peaked[peaked["profit_usd"] > 0]
    print(f"Trades: {len(trades)} | Total PnL: {total_pnl:.2f} | Win rate: {win_rate:.2%}")
    print(
        f"Never profitable: {len(never_profit)} (PnL {never_profit['profit_usd'].sum():.2f}) | "
        f"Peaked then lost: {len(gave_back)} (PnL {gave_back['profit_usd'].sum():.2f}) | "
        f"Peaked and kept: {len(kept)} (PnL {kept['profit_usd'].sum():.2f})"
    )
    print("Close reasons:", trades["close_reason"].value_counts().to_dict())


def _fold_report(symbol, data, strategy, mgmt_fn, risk, folds, label):
    """Split the history into N contiguous, non-overlapping folds and report
    each one separately. A single aggregate number over the whole history
    can look good just because one lucky stretch dominates; per-fold
    consistency is what actually tells you whether an edge holds up across
    different chunks of time, which is the point of walk-forward style
    checking rather than a single in-sample run."""
    fold_size = len(data) // folds
    if fold_size < 200:
        print(f"Not enough bars for {folds} folds on {symbol}; using 1 fold instead.")
        folds = 1
        fold_size = len(data)

    print(f"\n--- {label}: {folds}-fold walk-forward ---")
    fold_pnls = []
    for f in range(folds):
        start = f * fold_size
        end = len(data) if f == folds - 1 else (f + 1) * fold_size
        fold_data = data.iloc[start:end]
        if fold_data.empty:
            continue
        trades = _simulate_symbol(symbol, fold_data, strategy, mgmt_fn, risk)
        pnl = trades["profit_usd"].sum() if not trades.empty else 0.0
        win_rate = (trades["profit_usd"] > 0).mean() if not trades.empty else 0.0
        fold_pnls.append(pnl)
        span = f"{fold_data.index[0].date()} to {fold_data.index[-1].date()}"
        print(f"Fold {f + 1}/{folds} ({span}): {len(trades)} trades | PnL {pnl:.2f} | win rate {win_rate:.2%}")

    if fold_pnls:
        profitable_folds = sum(1 for p in fold_pnls if p > 0)
        print(
            f"-> {profitable_folds}/{len(fold_pnls)} folds profitable. "
            + ("Consistent across time periods." if profitable_folds == len(fold_pnls)
               else "Inconsistent -- results are being driven by a subset of the history, treat the aggregate number with caution.")
        )


def _scaled_giveback_params(scale):
    def _fn(symbol=None):
        mgmt = new_trade_management_params(symbol)
        mgmt["giveback_usd_buffer"] = mgmt.get("giveback_usd_buffer", 0.0) * scale
        mgmt["giveback_buffer_r"] = mgmt.get("giveback_buffer_r", 0.0) * scale
        return mgmt
    return _fn


def main():
    parser = argparse.ArgumentParser(description="Replay live exit logic against historical bars")
    parser.add_argument("--symbol", action="append", help=f"Symbol(s) to test, from {AVAILABLE_SYMBOLS}")
    parser.add_argument("--compare-old", action="store_true", help="Also run the pre-tuning tiers for A/B comparison")
    parser.add_argument("--folds", type=int, default=1, help="Split history into N contiguous folds for a walk-forward-style consistency check (default: 1, i.e. off)")
    parser.add_argument(
        "--giveback-scale",
        type=float,
        help="Test a giveback buffer scaled by this factor (e.g. 0.5 = tighter, 1.5 = looser) "
        "vs the current buffer, to see whether tightening/loosening it actually helps or just "
        "shifts trades between the 'kept' and 'lost' buckets without changing the total.",
    )
    args = parser.parse_args()

    symbols = args.symbol or list(AVAILABLE_SYMBOLS)
    unknown = [s for s in symbols if s.upper() not in SYMBOL_MODEL]
    if unknown:
        print(f"Skipping {unknown}: no pricing model for these yet.")
    symbols = [s.upper() for s in symbols if s.upper() in SYMBOL_MODEL]
    if not symbols:
        raise SystemExit(f"No testable symbols given. Available: {AVAILABLE_SYMBOLS}")

    router = StrategyRouter()
    risk = RiskManager(risk_per_trade=0.0020)

    for symbol in symbols:
        print(f"\n=== {symbol} ===")
        try:
            data = FeatureEngine().add_features(DataLoader(symbol=symbol).load())
        except FileNotFoundError:
            print(
                f"No local data for {symbol} (expected data/{symbol}_M5.csv). "
                f"Export it first: python run_project.py export --symbols {symbol} --timeframe M5 --bars 20000"
            )
            continue

        strategy = router.get_strategy(symbol)
        print(f"Bars: {len(data)} | Strategy: {strategy.__class__.__name__} (min_score={getattr(strategy, 'min_score', 'n/a')})")

        new_trades = _simulate_symbol(symbol, data, strategy, new_trade_management_params, risk)
        _summarize(new_trades, "NEW tiers (current live_runner.py)")

        if args.compare_old:
            old_trades = _simulate_symbol(symbol, data, strategy, _old_trade_management_params, risk)
            _summarize(old_trades, "OLD tiers (pre-tuning snapshot)")

        if args.folds > 1:
            _fold_report(symbol, data, strategy, new_trade_management_params, risk, args.folds, "NEW tiers")

        if args.giveback_scale is not None:
            scaled_trades = _simulate_symbol(symbol, data, strategy, _scaled_giveback_params(args.giveback_scale), risk)
            _summarize(scaled_trades, f"Giveback buffer x{args.giveback_scale} (vs current)")


if __name__ == "__main__":
    main()
