from bootstrap import add_project_root
add_project_root()

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from engine.backtester import Backtester
from strategy_batch_tools import (
    default_strategy_grid,
    infer_symbol_from_path,
    load_symbol_data,
    resolve_symbol_inputs,
)
from timing_utils import timed


@dataclass
class TournamentRow:
    symbol: str
    strategy: str
    backtest_balance: float
    backtest_trades: int
    backtest_win_rate: float
    backtest_max_dd: float
    walkforward_balance: float
    walkforward_trades: float
    walkforward_win_rate: float
    walkforward_max_dd: float
    score: float


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


def score_row(backtest_balance, backtest_max_dd, walkforward_balance, walkforward_max_dd):
    balance_gain = (backtest_balance - 10000.0) + (walkforward_balance - 10000.0)
    drawdown_penalty = abs(backtest_max_dd) + abs(walkforward_max_dd)
    return balance_gain - 1.5 * drawdown_penalty


def load_focus_map():
    config_path = Path("configs/symbol_universe.json")
    if not config_path.exists():
        return {
            "AUDUSD": "mean_reversion",
            "EURUSD": "mean_reversion_pullback",
            "NZDUSD": "mean_reversion",
            "USDCHF": "five_signal_confluence_scalper",
            "USDJPY": "five_signal_confluence_scalper",
            "USDCAD": "five_signal_confluence_scalper",
        }
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    symbols = payload.get("tournament_candidates", [])
    live_symbols = payload.get("preferred_live", [])
    basket = list(dict.fromkeys((live_symbols or []) + (symbols or [])))
    focus_map = {
        "AUDUSD": "mean_reversion",
        "EURUSD": "mean_reversion_pullback",
        "NZDUSD": "mean_reversion",
        "USDCHF": "five_signal_confluence_scalper",
        "USDJPY": "five_signal_confluence_scalper",
        "USDCAD": "five_signal_confluence_scalper",
    }
    return {symbol: focus_map[symbol] for symbol in basket if symbol in focus_map}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", action="append")
    parser.add_argument("--data", action="append")
    parser.add_argument("--train-bars", type=int, default=2000)
    parser.add_argument("--test-bars", type=int, default=500)
    parser.add_argument("--step-bars", type=int, default=500)
    parser.add_argument("--full", action="store_true", help="Run all strategies on all inputs.")
    args = parser.parse_args()

    inputs = resolve_symbol_inputs((args.data or []) + (args.symbol or []) or None)
    strategies = default_strategy_grid()
    focus_map = load_focus_map()
    rows = []

    with timed("tournament_report", style="days"):
        for item in inputs:
            if item.endswith(".csv"):
                symbol_name = infer_symbol_from_path(item)
                data = load_symbol_data(data_path=item)
            else:
                symbol_name = item
                data = load_symbol_data(symbol=item)

            if args.full:
                strategy_items = strategies.items()
            else:
                strategy_name = focus_map.get(symbol_name)
                if strategy_name is None:
                    continue
                strategy_items = [(strategy_name, strategies[strategy_name])]

            for strategy_name, strategy in strategy_items:
                backtest = Backtester(data, strategy).run()
                walkforward = walkforward_for_data(
                    data=data,
                    strategy=strategy,
                    train_bars=args.train_bars,
                    test_bars=args.test_bars,
                    step_bars=args.step_bars,
                )
                rows.append(
                    TournamentRow(
                        symbol=symbol_name,
                        strategy=strategy_name,
                        backtest_balance=backtest["final_balance"],
                        backtest_trades=backtest["total_trades"],
                        backtest_win_rate=backtest["win_rate"],
                        backtest_max_dd=max_drawdown(backtest["equity_curve"]),
                        walkforward_balance=walkforward["avg_balance"],
                        walkforward_trades=walkforward["avg_trades"],
                        walkforward_win_rate=walkforward["avg_win_rate"],
                        walkforward_max_dd=walkforward["avg_max_dd"],
                        score=score_row(
                            backtest["final_balance"],
                            max_drawdown(backtest["equity_curve"]),
                            walkforward["avg_balance"],
                            walkforward["avg_max_dd"],
                        ),
                    )
                )

    rows = sorted(rows, key=lambda row: (row.symbol, -row.score))

    print("\n=== TOURNAMENT REPORT ===")
    print(
        f"{'symbol':>10} | {'strategy':>24} | {'score':>10} | {'bt_bal':>10} | {'bt_dd':>9} | "
        f"{'wf_bal':>10} | {'wf_dd':>9} | {'bt_win':>7} | {'wf_win':>7}"
    )
    print("-" * 120)
    for row in rows:
        print(
            f"{row.symbol:>10} | {row.strategy:>24} | {row.score:10.2f} | {row.backtest_balance:10.2f} | {row.backtest_max_dd:9.2f} | "
            f"{row.walkforward_balance:10.2f} | {row.walkforward_max_dd:9.2f} | {row.backtest_win_rate:7.2%} | {row.walkforward_win_rate:7.2%}"
        )

    print("\n=== BEST PER SYMBOL ===")
    for symbol in sorted({row.symbol for row in rows}):
        symbol_rows = [row for row in rows if row.symbol == symbol]
        best = max(symbol_rows, key=lambda row: row.score)
        print(f"{symbol}: {best.strategy} | score={best.score:.2f} | bt={best.backtest_balance:.2f} | wf={best.walkforward_balance:.2f}")


if __name__ == "__main__":
    main()
