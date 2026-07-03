from bootstrap import add_project_root
add_project_root()

import argparse
from dataclasses import dataclass

from engine.backtester import Backtester
from strategy_batch_tools import default_strategy_grid, load_symbol_data, resolve_symbol_inputs, infer_symbol_from_path
from timing_utils import timed


@dataclass
class MilestoneResult:
    symbol: str
    strategy: str
    final_balance: float
    trades: int
    win_rate: float
    avg_r: float
    days_to_10500: float | None
    days_to_11000: float | None
    days_to_12000: float | None
    week1_balance: float
    week2_balance: float
    max_drawdown: float


def first_hit_days(equity_curve, target, bars_per_day=288):
    for index, value in enumerate(equity_curve):
        if value >= target:
            return index / bars_per_day
    return None


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


def balance_at_days(equity_curve, days, bars_per_day=288):
    if days is None:
        return None
    index = min(len(equity_curve) - 1, int(days * bars_per_day))
    if index < 0:
        return None
    return equity_curve[index]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", action="append")
    parser.add_argument("--data", action="append")
    args = parser.parse_args()

    inputs = resolve_symbol_inputs((args.data or []) + (args.symbol or []) or None)
    strategies = default_strategy_grid()

    rows = []
    with timed("milestone_report", style="days"):
        for item in inputs:
            if item.endswith(".csv"):
                symbol_name = infer_symbol_from_path(item)
                data = load_symbol_data(data_path=item)
            else:
                symbol_name = item
                data = load_symbol_data(symbol=item)

            for strategy_name, strategy in strategies.items():
                result = Backtester(data, strategy).run()
                curve = result["equity_curve"]
                rows.append(
                    MilestoneResult(
                        symbol=symbol_name,
                        strategy=strategy_name,
                        final_balance=result["final_balance"],
                        trades=result["total_trades"],
                        win_rate=result["win_rate"],
                        avg_r=result["avg_r"],
                        days_to_10500=first_hit_days(curve, 10500),
                        days_to_11000=first_hit_days(curve, 11000),
                        days_to_12000=first_hit_days(curve, 12000),
                        week1_balance=balance_at_days(curve, 7),
                        week2_balance=balance_at_days(curve, 14),
                        max_drawdown=max_drawdown(curve),
                    )
                )

    rows = sorted(rows, key=lambda row: (row.symbol, -row.final_balance))

    print("\n=== MILESTONE REPORT ===")
    print(
        f"{'symbol':>10} | {'strategy':>18} | {'final':>10} | {'trades':>6} | {'win_rate':>8} | "
        f"{'d10500':>8} | {'d11000':>8} | {'d12000':>8} | {'w1':>8} | {'w2':>8} | {'max_dd':>10}"
    )
    print("-" * 132)
    for row in rows:
        def fmt_days(value):
            return "n/a" if value is None else f"{value:.2f}"

        print(
            f"{row.symbol:>10} | {row.strategy:>18} | {row.final_balance:10.2f} | {row.trades:6d} | {row.win_rate:8.2%} | "
            f"{fmt_days(row.days_to_10500):>8} | {fmt_days(row.days_to_11000):>8} | {fmt_days(row.days_to_12000):>8} | "
            f"{(row.week1_balance if row.week1_balance is not None else float('nan')):8.2f} | "
            f"{(row.week2_balance if row.week2_balance is not None else float('nan')):8.2f} | "
            f"{row.max_drawdown:10.2f}"
        )

    combo_rows = [row for row in rows if row.strategy in {"mean_reversion", "pullback_trend", "mean_pullback_combo"}]
    if combo_rows:
        print("\n=== COMBO VIEW ===")
        print(f"{'symbol':>10} | {'strategy':>18} | {'final':>10} | {'d10500':>8} | {'max_dd':>10}")
        print("-" * 60)
        for row in combo_rows:
            def fmt_days(value):
                return "n/a" if value is None else f"{value:.2f}"

            print(
                f"{row.symbol:>10} | {row.strategy:>18} | {row.final_balance:10.2f} | "
                f"{fmt_days(row.days_to_10500):>8} | {row.max_drawdown:10.2f}"
            )


if __name__ == "__main__":
    main()
