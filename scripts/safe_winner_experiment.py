from bootstrap import add_project_root
add_project_root()

import argparse
from dataclasses import dataclass

from engine.backtester import Backtester
from strategy_batch_tools import load_symbol_data, resolve_symbol_inputs, infer_symbol_from_path
from timing_utils import timed
from strategies.mean_reversion_pullback import MeanReversionPullback
from strategies.momentum import Momentum
from strategies.mean_reversion import MeanReversion
from strategies.safe_winner_strategy import SafeWinnerStrategy


@dataclass
class SafeRow:
    symbol: str
    strategy: str
    balance: float
    trades: int
    win_rate: float
    avg_r: float
    max_dd: float


def max_drawdown(equity_curve):
    peak = equity_curve[0] if equity_curve else 0
    worst = 0.0
    for value in equity_curve:
        if value > peak:
            peak = value
        worst = min(worst, value - peak)
    return worst


def walkforward_for_data(data, strategy, train_bars=2000, test_bars=500, step_bars=500):
    start = 0
    balances = []
    trades = []
    win_rates = []
    max_dds = []

    while start + train_bars + test_bars <= len(data):
        test = data.iloc[start + train_bars : start + train_bars + test_bars].copy()
        result = Backtester(test, strategy).run()
        balances.append(result["final_balance"])
        trades.append(result["total_trades"])
        win_rates.append(result["win_rate"])
        max_dds.append(max_drawdown(result["equity_curve"]))
        start += step_bars

    folds = max(1, len(balances))
    return {
        "avg_balance": sum(balances) / folds if balances else 10000.0,
        "avg_trades": sum(trades) / folds if balances else 0.0,
        "avg_win_rate": sum(win_rates) / folds if balances else 0.0,
        "avg_max_dd": sum(max_dds) / folds if balances else 0.0,
    }


def milestone_points(balance):
    return {
        "d10500": 0.0 if balance >= 10500 else None,
        "d11000": 0.0 if balance >= 11000 else None,
        "d12000": 0.0 if balance >= 12000 else None,
    }


def build_symbol_strategy(symbol):
    symbol = symbol.upper()
    if symbol == "EURUSD":
        return SafeWinnerStrategy(MeanReversionPullback(), mode="mean_reversion")
    if symbol == "GBPUSD":
        return SafeWinnerStrategy(Momentum(), mode="momentum")
    if symbol == "XAUUSD":
        return SafeWinnerStrategy(MeanReversion(lookback=20, entry_z=1.5), mode="mean_reversion")
    return SafeWinnerStrategy(MeanReversionPullback(), mode="mean_reversion")


def run_for_inputs(inputs, train_bars, test_bars, step_bars):
    rows = []
    for item in inputs:
        if item.endswith(".csv"):
            symbol_name = infer_symbol_from_path(item)
            data = load_symbol_data(data_path=item)
        else:
            symbol_name = item
            data = load_symbol_data(symbol=item)

        strategy = build_symbol_strategy(symbol_name)
        strategy_name = strategy.base_strategy.__class__.__name__
        backtest = Backtester(data, strategy).run()
        wf = walkforward_for_data(
            data=data,
            strategy=strategy,
            train_bars=train_bars,
            test_bars=test_bars,
            step_bars=step_bars,
        )
        rows.append(
            SafeRow(
                symbol=symbol_name,
                strategy=strategy_name,
                balance=backtest["final_balance"],
                trades=backtest["total_trades"],
                win_rate=backtest["win_rate"],
                avg_r=backtest["avg_r"],
                max_dd=max_drawdown(backtest["equity_curve"]),
            )
        )
    return rows


def print_report(rows):
    print("\n=== SAFE WINNER REPORT ===")
    print(f"{'symbol':>10} | {'strategy':>24} | {'balance':>10} | {'trades':>6} | {'win_rate':>8} | {'avg_r':>8} | {'max_dd':>9}")
    print("-" * 100)
    for row in rows:
        print(
            f"{row.symbol:>10} | {row.strategy:>24} | {row.balance:10.2f} | {row.trades:6d} | {row.win_rate:8.2%} | {row.avg_r:8.4f} | {row.max_dd:9.2f}"
        )


def print_walkforward(inputs, train_bars, test_bars, step_bars):
    print("\n=== WALK-FORWARD ===")
    print(f"{'symbol':>10} | {'avg_balance':>12} | {'avg_trades':>10} | {'avg_win':>8} | {'avg_dd':>9}")
    print("-" * 70)
    for item in inputs:
        if item.endswith(".csv"):
            symbol_name = infer_symbol_from_path(item)
            data = load_symbol_data(data_path=item)
        else:
            symbol_name = item
            data = load_symbol_data(symbol=item)
        strategy = build_symbol_strategy(symbol_name)
        wf = walkforward_for_data(data, strategy, train_bars, test_bars, step_bars)
        print(
            f"{symbol_name:>10} | {wf['avg_balance']:12.2f} | {wf['avg_trades']:10.2f} | {wf['avg_win_rate']:8.2%} | {wf['avg_max_dd']:9.2f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["test", "walkforward", "milestone", "tournament"])
    parser.add_argument("--symbol", action="append")
    parser.add_argument("--data", action="append")
    parser.add_argument("--train-bars", type=int, default=2000)
    parser.add_argument("--test-bars", type=int, default=500)
    parser.add_argument("--step-bars", type=int, default=500)
    args = parser.parse_args()

    inputs = resolve_symbol_inputs((args.data or []) + (args.symbol or []) or None)
    with timed("safe_winner_report", style="days"):
        rows = run_for_inputs(inputs, args.train_bars, args.test_bars, args.step_bars)

    print_report(rows)
    print_walkforward(inputs, args.train_bars, args.test_bars, args.step_bars)

    if args.mode in {"milestone", "tournament"}:
        print("\n=== MILESTONES ===")
        print(f"{'symbol':>10} | {'d10500':>8} | {'d11000':>8} | {'d12000':>8}")
        print("-" * 46)
        for row in rows:
            milestones = milestone_points(row.balance)
            print(
                f"{row.symbol:>10} | {str(milestones['d10500']):>8} | {str(milestones['d11000']):>8} | {str(milestones['d12000']):>8}"
            )


if __name__ == "__main__":
    main()
