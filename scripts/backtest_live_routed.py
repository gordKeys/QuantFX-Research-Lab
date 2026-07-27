"""
Backtest + walk-forward the EXACT strategy/config StrategyRouter routes to
each symbol live (not a generic default), with real measured costs
(configs/commission_map.json, configs/swap_map.json) and the live risk-per-
trade (0.5%, from configs/live_challenge.py's RiskCaps).

Purpose: find out whether the live losses are (a) a real, already-present
lack of edge that also shows up in backtest ("fix the strategy") or (b) a
live/backtest mismatch ("hunt for an execution bug"), per the user's request.
"""

import sys
sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

from engine.data_loader import DataLoader
from engine.features import FeatureEngine
from engine.backtester import Backtester
from strategy_router import StrategyRouter

LIVE_RISK_PER_TRADE = 0.005  # configs/live_challenge.py RiskCaps.risk_per_trade
SYMBOLS = ["EURUSD", "AUDUSD", "XAUUSD"]


def run_symbol(symbol, strategy_name, strategy, train_bars=2000, test_bars=500, step_bars=500):
    try:
        data = FeatureEngine().add_features(DataLoader(symbol=symbol).load())
    except FileNotFoundError:
        print(f"\n=== {symbol}  (live strategy: {strategy_name}) ===")
        print(f"  SKIPPED -- no data/{symbol}_M5.csv in this environment. "
              f"Run this same script on your machine (or export {symbol} via "
              f"scripts/export_mt5_data.py) to get this one.")
        return

    # --- Full-sample backtest (in-sample, but with real costs) ---
    bt = Backtester(data, strategy, symbol=symbol)
    bt.risk_per_trade = LIVE_RISK_PER_TRADE
    full = bt.run()

    # --- Walk-forward (rolling out-of-sample) ---
    start = 0
    folds = []
    while start + train_bars + test_bars <= len(data):
        test = data.iloc[start + train_bars: start + train_bars + test_bars].copy()
        wf_bt = Backtester(test, strategy, symbol=symbol)
        wf_bt.risk_per_trade = LIVE_RISK_PER_TRADE
        folds.append(wf_bt.run())
        start += step_bars

    print(f"\n=== {symbol}  (live strategy: {strategy_name}) ===")
    print(f"FULL SAMPLE ({full['total_trades']} trades):")
    print(f"  win_rate={full['win_rate']:.2%}  avg_win_R={full['avg_win_r']:.3f}  avg_loss_R={full['avg_loss_r']:.3f}")
    print(f"  net_pf={full['profit_factor']:.2f}  gross_pf={full['gross_profit_factor']:.2f}  expectancy_R={full['expectancy_r']:.4f}")
    print(f"  final_balance=${full['final_balance']:.2f}  spread=${full['total_spread_cost_usd']:.2f}  "
          f"commission=${full['total_commission_usd']:.2f}  swap=${full['total_swap_usd']:.2f}")

    if not folds:
        print("  (not enough bars for a single walk-forward fold)")
        return

    print(f"\nWALK-FORWARD ({len(folds)} folds, {test_bars} bars each, out-of-sample):")
    profitable_folds = 0
    for i, f in enumerate(folds, 1):
        tag = "OK " if f["profit_factor"] >= 1.0 else "netloss"
        if f["profit_factor"] >= 1.0:
            profitable_folds += 1
        print(f"  fold {i}: trades={f['total_trades']:4d}  win_rate={f['win_rate']:.2%}  "
              f"net_pf={f['profit_factor']:.2f}  gross_pf={f['gross_profit_factor']:.2f}  "
              f"exp_R={f['expectancy_r']:+.4f}  [{tag}]")
    print(f"  -> {profitable_folds}/{len(folds)} folds net-profitable")


def main():
    router = StrategyRouter()
    for symbol in SYMBOLS:
        strategy_name = router.get_strategy_name(symbol)
        strategy = router.get_strategy(symbol)
        run_symbol(symbol, strategy_name, strategy)


if __name__ == "__main__":
    main()
