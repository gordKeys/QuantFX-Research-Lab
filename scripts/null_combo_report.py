from bootstrap import add_project_root
add_project_root()

import argparse
from dataclasses import dataclass
from pathlib import Path

from engine.backtester import Backtester
from strategy_batch_tools import infer_symbol_from_path, load_symbol_data
from strategies.mean_reversion_pullback import MeanReversionPullback
from strategies.scalp_reversion import ScalpReversion
from strategies.mirror_strategy import MirrorStrategy
from timing_utils import timed


FTMO_INITIAL_BALANCE = 10000.0
FTMO_DAILY_LOSS_LIMIT = FTMO_INITIAL_BALANCE * 0.05
FTMO_TOTAL_LOSS_LIMIT = FTMO_INITIAL_BALANCE * 0.10


@dataclass
class NullRow:
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

    @property
    def positive_walkforward(self) -> bool:
        return self.walkforward_balance > FTMO_INITIAL_BALANCE


def max_drawdown(equity_curve):
    peak = equity_curve[0] if equity_curve else 0
    worst = 0.0
    for value in equity_curve:
        if value > peak:
            peak = value
        drawdown = value - peak
        if drawdown < worst:
            worst = drawdown
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
        "avg_balance": sum(balances) / folds,
        "avg_trades": sum(trades) / folds,
        "avg_win_rate": sum(win_rates) / folds,
        "avg_max_dd": sum(max_dds) / folds,
    }


def ftmo_ratio(drawdown):
    abs_dd = abs(drawdown)
    return {
        "daily_pct": (abs_dd / FTMO_DAILY_LOSS_LIMIT) * 100.0 if FTMO_DAILY_LOSS_LIMIT else 0.0,
        "total_pct": (abs_dd / FTMO_TOTAL_LOSS_LIMIT) * 100.0 if FTMO_TOTAL_LOSS_LIMIT else 0.0,
        "breach_daily": abs_dd > FTMO_DAILY_LOSS_LIMIT,
        "breach_total": abs_dd > FTMO_TOTAL_LOSS_LIMIT,
    }


def print_row(row: NullRow):
    backtest_ratio = ftmo_ratio(row.backtest_max_dd)
    walkforward_ratio = ftmo_ratio(row.walkforward_max_dd)

    print(
        f"{row.symbol:>10} | {row.strategy:>24} | "
        f"{row.backtest_balance:10.2f} | {row.backtest_max_dd:9.2f} | "
        f"{backtest_ratio['daily_pct']:7.1f}%/{backtest_ratio['total_pct']:7.1f}% "
        f"{'B' if backtest_ratio['breach_daily'] else '-'}{'B' if backtest_ratio['breach_total'] else '-'} | "
        f"{row.walkforward_balance:10.2f} | {row.walkforward_max_dd:9.2f} | "
        f"{walkforward_ratio['daily_pct']:7.1f}%/{walkforward_ratio['total_pct']:7.1f}% "
        f"{'B' if walkforward_ratio['breach_daily'] else '-'}{'B' if walkforward_ratio['breach_total'] else '-'} | "
        f"{row.backtest_win_rate:7.2%} | {row.walkforward_win_rate:7.2%}"
    )


def main():
    parser = argparse.ArgumentParser(description="Null trader combo + walkforward with FTMO drawdown comparison")
    parser.add_argument("--symbol", action="append", help="Symbol name like EURUSD. Repeatable.")
    parser.add_argument("--data", action="append", help="CSV path like data/EURUSD_M5.csv. Repeatable.")
    parser.add_argument("--train-bars", type=int, default=2000)
    parser.add_argument("--test-bars", type=int, default=500)
    parser.add_argument("--step-bars", type=int, default=500)
    args = parser.parse_args()

    focus_map = {
        "EURUSD": "mean_reversion_pullback",
        "GBPUSD": "mean_reversion_pullback",
        "USDCHF": "mean_reversion_pullback",
        "USDJPY": "scalp_reversion",
        "USDCAD": "scalp_reversion",
    }

    strategy_registry = {
        "mean_reversion_pullback": MirrorStrategy(MeanReversionPullback()),
        "scalp_reversion": MirrorStrategy(ScalpReversion()),
    }

    requested_symbols = [item.upper() for item in (args.symbol or [])]
    requested_data = args.data or []
    requested_from_data = [infer_symbol_from_path(path) for path in requested_data]
    symbols = requested_symbols + requested_from_data
    if not symbols:
        symbols = list(focus_map.keys())

    rows = []

    with timed("null_combo_report", style="days"):
        print("\n=== NULL COMBO + WALK-FORWARD ===")
        print(
            f"{'symbol':>10} | {'strategy':>24} | {'bt_bal':>10} | {'bt_dd':>9} | {'bt FTMO':>18} | "
            f"{'wf_bal':>10} | {'wf_dd':>9} | {'wf FTMO':>18} | {'bt_win':>8} | {'wf_win':>8}"
        )
        print("-" * 150)

        seen = set()
        for symbol in symbols:
            symbol = symbol.upper()
            if symbol in seen:
                continue
            seen.add(symbol)

            strategy_name = focus_map.get(symbol)
            if strategy_name is None:
                print(f"{symbol:>10} | {'unsupported':>24} | {'n/a':>10} | {'n/a':>9} | {'n/a':>18} | {'n/a':>10} | {'n/a':>9} | {'n/a':>18} | {'n/a':>8} | {'n/a':>8}")
                continue

            csv_path = Path("data") / f"{symbol}_M5.csv"
            if not csv_path.exists():
                print(f"{symbol:>10} | {strategy_name:>24} | {'missing CSV':>10} | {'n/a':>9} | {'n/a':>18} | {'n/a':>10} | {'n/a':>9} | {'n/a':>18} | {'n/a':>8} | {'n/a':>8}")
                print(f"           ↳ expected {csv_path}")
                continue

            try:
                data = load_symbol_data(data_path=str(csv_path))
            except Exception as exc:
                print(f"{symbol:>10} | {strategy_name:>24} | {'load failed':>10} | {'n/a':>9} | {'n/a':>18} | {'n/a':>10} | {'n/a':>9} | {'n/a':>18} | {'n/a':>8} | {'n/a':>8}")
                print(f"           ↳ {exc}")
                continue

            strategy = strategy_registry[strategy_name]
            backtest = Backtester(data, strategy).run()
            walkforward = walkforward_for_data(
                data=data,
                strategy=strategy,
                train_bars=args.train_bars,
                test_bars=args.test_bars,
                step_bars=args.step_bars,
            )

            row = NullRow(
                symbol=symbol,
                strategy=strategy_name,
                backtest_balance=backtest["final_balance"],
                backtest_trades=backtest["total_trades"],
                backtest_win_rate=backtest["win_rate"],
                backtest_max_dd=max_drawdown(backtest["equity_curve"]),
                walkforward_balance=walkforward["avg_balance"],
                walkforward_trades=walkforward["avg_trades"],
                walkforward_win_rate=walkforward["avg_win_rate"],
                walkforward_max_dd=walkforward["avg_max_dd"],
            )
            rows.append(row)
            print_row(row)

        print("\n=== FTMO INTERPRETATION ===")
        print(f"Starting balance: {FTMO_INITIAL_BALANCE:.2f}")
        print(f"Max daily loss:   {FTMO_DAILY_LOSS_LIMIT:.2f}")
        print(f"Max total loss:   {FTMO_TOTAL_LOSS_LIMIT:.2f}")
        print("FTMO columns show drawdown as % of daily limit / total limit.")
        print("Breach flags are based on absolute drawdown exceeding those limits.")

        positive = [row for row in rows if row.positive_walkforward]
        if positive:
            print("\n=== WFA POSITIVE PICKS ===")
            for row in positive:
                print(
                    f"{row.symbol}: {row.strategy} | wf_bal={row.walkforward_balance:.2f} | "
                    f"wf_dd={row.walkforward_max_dd:.2f}"
                )


if __name__ == "__main__":
    main()
