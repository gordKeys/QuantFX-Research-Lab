"""
Entry quality analyzer.

Where backtest_live_logic.py answers "does this EXIT tuning help", this
answers "does this ENTRY tuning help" -- using the exact same real,
exit-managed trade outcomes (same manage_live_position code path), not a
naive fixed-bar-hold classification of "was price up N bars later" that
would ignore your actual stops/giveback/quick-cut logic entirely.

Two things this does:

1. COMPONENT BREAKDOWN: FiveSignalConfluenceScalper sums 6 signal
   components into one score. Two of those are philosophically different
   things sharing one number: "trend" (EMA fast/slow) is trend-following,
   while "band_extreme"/"rsi_extreme"/"support_resistance" fire on
   oversold/overbought extremes, which often coincide with a move AGAINST
   the trend. Summing them means a counter-trend reversal setup can
   outscore a trend-aligned one, or vice versa, with no way to tell from
   the score alone which kind of trade you're taking. This reports, per
   component, the win rate and avg PnL of trades where that component was
   active at entry vs the baseline -- a poor man's feature importance,
   using real trade outcomes.

2. HYPOTHESIS TESTING: automatically tries a handful of concrete, motivated
   tweaks (requiring trend alignment, dropping the weakest component(s),
   a few min_score levels) and reports each one's aggregate result AND its
   walk-forward fold consistency. A tweak that only wins in the aggregate
   but loses money in most individual folds is flagged as likely overfit,
   not a real improvement -- the point is to avoid trading a config that
   just happened to fit this particular stretch of history.

Usage:
    python scripts/entry_quality_analyzer.py --symbol EURUSD --folds 4
    python scripts/entry_quality_analyzer.py --folds 4   # all available symbols
"""
from bootstrap import add_project_root
add_project_root()

import argparse
import copy

import pandas as pd

from engine.data_loader import DataLoader
from engine.features import FeatureEngine
from engine.risk_manager import RiskManager
from strategies.five_signal_confluence_scalper import FiveSignalConfluenceScalper
from strategy_router import StrategyRouter
from live_runner import trade_management_params
from backtest_live_logic import _simulate_symbol, SYMBOL_MODEL, AVAILABLE_SYMBOLS


def _component_breakdown(symbol, data, strategy, risk):
    """Run once, join real trade outcomes to the components active at each
    entry, and report each component's marginal win rate / avg PnL."""
    trades = _simulate_symbol(symbol, data, strategy, trade_management_params, risk)
    components_by_time = strategy.last_run_components
    if trades.empty or not components_by_time:
        print("No trades or no component data recorded; skipping component breakdown.")
        return

    trades = trades.copy()
    trades["_join_key"] = pd.to_datetime(trades["open_time"]).dt.tz_localize(None)
    components_naive = {pd.Timestamp(k).tz_localize(None) if pd.Timestamp(k).tzinfo else pd.Timestamp(k): v for k, v in components_by_time.items()}
    trades["components"] = trades["_join_key"].map(components_naive)
    trades = trades.dropna(subset=["components"])
    if trades.empty:
        print("Could not join any trades to their entry components (timestamp mismatch); skipping.")
        return

    baseline_win_rate = (trades["profit_usd"] > 0).mean()
    baseline_avg_pnl = trades["profit_usd"].mean()
    print(f"\nBaseline: {len(trades)} trades | win rate {baseline_win_rate:.2%} | avg PnL {baseline_avg_pnl:.2f}")
    print(f"{'component':>18} | {'active: n':>10} | {'win rate':>9} | {'avg pnl':>8} | {'vs baseline avg pnl':>20}")
    print("-" * 78)
    for component in FiveSignalConfluenceScalper.COMPONENTS:
        active_mask = trades["components"].apply(lambda c: c.get(component, False))
        active = trades[active_mask]
        if active.empty:
            print(f"{component:>18} | {'0':>10} | {'--':>9} | {'--':>8} | never active in a winning-side signal")
            continue
        win_rate = (active["profit_usd"] > 0).mean()
        avg_pnl = active["profit_usd"].mean()
        delta = avg_pnl - baseline_avg_pnl
        flag = "" if abs(delta) < 0.5 else (" <- notably better" if delta > 0 else " <- notably worse")
        print(f"{component:>18} | {len(active):>10} | {win_rate:>9.2%} | {avg_pnl:>8.2f} | {delta:>+20.2f}{flag}")


def _fold_eval(symbol, data, strategy, risk, folds):
    """Same fold-consistency idea as backtest_live_logic._fold_report, kept
    local here so it returns numbers (not just prints) for comparison
    across candidate configs."""
    fold_size = len(data) // folds
    if fold_size < 200:
        folds = 1
        fold_size = len(data)
    results = []
    for f in range(folds):
        start = f * fold_size
        end = len(data) if f == folds - 1 else (f + 1) * fold_size
        fold_data = data.iloc[start:end]
        if fold_data.empty:
            continue
        trades = _simulate_symbol(symbol, fold_data, strategy, trade_management_params, risk)
        results.append(trades["profit_usd"].sum() if not trades.empty else 0.0)
    return results


def _hypothesis_tests(symbol, data, risk, folds, base_min_score):
    print(f"\n=== Hypothesis tests ({folds}-fold consistency check each) ===")
    candidates = {
        "baseline": FiveSignalConfluenceScalper(min_score=base_min_score),
        "require_trend_alignment": FiveSignalConfluenceScalper(min_score=base_min_score, require_trend_alignment=True),
    }
    for component in FiveSignalConfluenceScalper.COMPONENTS:
        candidates[f"drop_{component}"] = FiveSignalConfluenceScalper(
            min_score=base_min_score, disabled_components={component}
        )
    for score in sorted({max(1, base_min_score - 1), base_min_score, base_min_score + 1, min(6, base_min_score + 2)}):
        candidates[f"min_score_{score}"] = FiveSignalConfluenceScalper(min_score=score)

    baseline_folds = None
    rows = []
    for name, strat in candidates.items():
        fold_pnls = _fold_eval(symbol, data, strat, risk, folds)
        total = sum(fold_pnls)
        profitable_folds = sum(1 for p in fold_pnls if p > 0)
        rows.append((name, total, profitable_folds, len(fold_pnls), fold_pnls))
        if name == "baseline":
            baseline_folds = (profitable_folds, len(fold_pnls), total)

    rows.sort(key=lambda r: r[1], reverse=True)
    print(f"{'config':>26} | {'total pnl':>10} | {'folds profitable':>17}")
    print("-" * 60)
    for name, total, profitable, n_folds, fold_pnls in rows:
        marker = ""
        if name != "baseline" and baseline_folds:
            base_profitable, base_n, base_total = baseline_folds
            if total > base_total and profitable >= base_profitable:
                marker = "  <- robust improvement (better AND at least as consistent)"
            elif total > base_total and profitable < base_profitable:
                marker = "  <- better total but LESS consistent, likely overfit -- be cautious"
        print(f"{name:>26} | {total:>10.2f} | {profitable:>3}/{n_folds:<13}{marker}")


def main():
    parser = argparse.ArgumentParser(description="Analyze which entry signal components predict winners, and test structural tweaks")
    parser.add_argument("--symbol", action="append", help=f"Symbol(s) to analyze, from {AVAILABLE_SYMBOLS}")
    parser.add_argument("--folds", type=int, default=4, help="Number of walk-forward folds for consistency checks (default 4)")
    args = parser.parse_args()

    symbols = args.symbol or list(AVAILABLE_SYMBOLS)
    symbols = [s.upper() for s in symbols if s.upper() in SYMBOL_MODEL]
    if not symbols:
        raise SystemExit(f"No testable symbols given. Available: {AVAILABLE_SYMBOLS}")

    router = StrategyRouter()
    risk = RiskManager(risk_per_trade=0.0020)

    for symbol in symbols:
        print(f"\n{'=' * 60}\n{symbol}\n{'=' * 60}")
        try:
            data = FeatureEngine().add_features(DataLoader(symbol=symbol).load())
        except FileNotFoundError:
            print(f"No local data for {symbol}. Export it first with: python run_project.py export --symbols {symbol}")
            continue

        current_strategy = router.get_strategy(symbol)
        if not isinstance(current_strategy, FiveSignalConfluenceScalper):
            print(f"Routed strategy is {current_strategy.__class__.__name__}, not the confluence scalper -- component breakdown doesn't apply, skipping.")
            continue
        base_min_score = current_strategy.min_score

        print(f"\n--- Component breakdown (current live config: min_score={base_min_score}) ---")
        _component_breakdown(symbol, data, FiveSignalConfluenceScalper(min_score=base_min_score), risk)

        _hypothesis_tests(symbol, data, risk, args.folds, base_min_score)


if __name__ == "__main__":
    main()
