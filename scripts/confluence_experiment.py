from bootstrap import add_project_root
add_project_root()

import argparse
from dataclasses import dataclass

from engine.backtester import Backtester
from strategy_batch_tools import load_symbol_data, resolve_symbol_inputs, infer_symbol_from_path
from timing_utils import timed
from strategies.five_signal_confluence_scalper import FiveSignalConfluenceScalper


@dataclass
class ConfluenceRow:
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
    targets = [10500.0, 11000.0, 12000.0]
    labels = ["d10500", "d11000", "d12000"]
    return dict(zip(labels, [0.0 if balance >= target else None for target in targets]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", action="append")
    parser.add_argument("--data", action="append")
    parser.add_argument("--train-bars", type=int, default=2000)
    parser.add_argument("--test-bars", type=int, default=500)
    parser.add_argument("--step-bars", type=int, default=500)
    args = parser.parse_args()

    inputs = resolve_symbol_inputs((args.data or []) + (args.symbol or []) or None)
    strategy = FiveSignalConfluenceScalper()
    rows = []

    with timed("confluence_report", style="days"):
        for item in inputs:
            if item.endswith(".csv"):
                symbol_name = infer_symbol_from_path(item)
                data = load_symbol_data(data_path=item)
            else:
                symbol_name = item
                data = load_symbol_data(symbol=item)

            backtest = Backtester(data, strategy).run()
            walkforward = walkforward_for_data(
                data=data,
                strategy=strategy,
                train_bars=args.train_bars,
                test_bars=args.test_bars,
                step_bars=args.step_bars,
            )
            rows.append(
                ConfluenceRow(
                    symbol=symbol_name,
                    strategy="five_signal_confluence_scalper",
                    balance=backtest["final_balance"],
                    trades=backtest["total_trades"],
                    win_rate=backtest["win_rate"],
                    avg_r=backtest["avg_r"],
                    max_dd=max_drawdown(backtest["equity_curve"]),
                )
            )

    print("\n=== CONFLUENCE REPORT ===")
    print(f"{'symbol':>10} | {'strategy':>30} | {'balance':>10} | {'trades':>6} | {'win_rate':>8} | {'avg_r':>8} | {'max_dd':>9}")
    print("-" * 100)
    for row in rows:
        print(
            f"{row.symbol:>10} | {row.strategy:>30} | {row.balance:10.2f} | {row.trades:6d} | {row.win_rate:8.2%} | {row.avg_r:8.4f} | {row.max_dd:9.2f}"
        )

    print("\n=== WALK-FORWARD ===")
    print(f"{'symbol':>10} | {'avg_balance':>12} | {'avg_trades':>10} | {'avg_win':>8} | {'avg_dd':>9}")
    print("-" * 70)
    for item in inputs:
        symbol_name = infer_symbol_from_path(item) if item.endswith(".csv") else item
        data = load_symbol_data(data_path=item) if item.endswith(".csv") else load_symbol_data(symbol=item)
        wf = walkforward_for_data(
            data=data,
            strategy=strategy,
            train_bars=args.train_bars,
            test_bars=args.test_bars,
            step_bars=args.step_bars,
        )
        print(
            f"{symbol_name:>10} | {wf['avg_balance']:12.2f} | {wf['avg_trades']:10.2f} | {wf['avg_win_rate']:8.2%} | {wf['avg_max_dd']:9.2f}"
        )


if __name__ == "__main__":
    main()
