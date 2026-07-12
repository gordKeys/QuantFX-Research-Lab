from bootstrap import add_project_root
add_project_root()

import json
from pathlib import Path

import pandas as pd

from engine.data_loader import DataLoader
from engine.features import FeatureEngine
from strategy_router import StrategyRouter
from strategies.five_signal_confluence_scalper import FiveSignalConfluenceScalper
from strategies.h1_confluence_trend import H1ConfluenceTrend, H1SessionConfluenceTrend
from timing_utils import timed


ROOT = Path(__file__).resolve().parents[1]


def load_live_symbols():
    live_symbols_file = ROOT / "configs" / "live_symbols.json"
    if not live_symbols_file.exists():
        return ["EURUSD", "GBPUSD"]
    try:
        with live_symbols_file.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        symbols = payload.get("symbols", [])
        return [symbol.upper() for symbol in symbols] or ["EURUSD", "GBPUSD"]
    except Exception:
        return ["EURUSD", "GBPUSD"]


def load_symbol_data(symbol):
    loader = DataLoader(symbol=symbol)
    return FeatureEngine().add_features(loader.load())


def strategy_family_name(strategy):
    name = strategy.__class__.__name__
    if isinstance(strategy, FiveSignalConfluenceScalper):
        return "five_signal_confluence_scalper"
    if isinstance(strategy, H1SessionConfluenceTrend):
        return "h1_session_confluence_trend"
    if isinstance(strategy, H1ConfluenceTrend):
        return "h1_confluence_trend"
    return name


def count_recent_signals(signals, lookback=100):
    if signals is None or len(signals) == 0:
        return {"buy": 0, "sell": 0, "total": 0}
    tail = signals.tail(lookback)
    buy = int((tail == 1).sum())
    sell = int((tail == -1).sum())
    return {"buy": buy, "sell": sell, "total": buy + sell}


def main():
    router = StrategyRouter()
    live_symbols = load_live_symbols()
    five_signal = FiveSignalConfluenceScalper()

    print("\n=== STRATEGY DIAGNOSTIC REPORT ===")
    print(f"Live symbols: {', '.join(live_symbols)}")
    print(f"Cooldown rule: 3 losses -> 12 M5 candles")
    print("Loss controls: warn=$11 | soft=$13.5 | hard=$15 | floating cap=$15")
    print("Entry gate: near daily/total/floating-loss blocks before new entries")
    print("\n" + f"{'symbol':>10} | {'routed_strategy':>26} | {'uses_5sig':>9} | {'latest_routed':>13} | {'latest_5sig':>10} | {'match':>7} | {'routed_last100':>15}")
    print("-" * 116)

    with timed("strategy_diagnostic", style="days"):
        for symbol in live_symbols:
            strategy = router.get_strategy(symbol)
            routed_name = router.get_strategy_name(symbol)
            uses_five_signal = isinstance(strategy, FiveSignalConfluenceScalper)

            csv_path = ROOT / "data" / f"{symbol}_M5.csv"
            if not csv_path.exists():
                print(f"{symbol:>10} | {routed_name:>26} | {str(uses_five_signal):>9} | {'missing CSV':>13} | {'n/a':>10} | {'n/a':>7} | {'n/a':>15}")
                continue

            try:
                data = load_symbol_data(symbol)
            except Exception as exc:
                print(f"{symbol:>10} | {routed_name:>26} | {str(uses_five_signal):>9} | {'load failed':>13} | {'n/a':>10} | {'n/a':>7} | {str(exc)[:15]:>15}")
                continue

            routed_signals = strategy.generate_signals(data)
            five_signals = five_signal.generate_signals(data)
            routed_latest = int(routed_signals.iloc[-1]) if len(routed_signals) else 0
            five_latest = int(five_signals.iloc[-1]) if len(five_signals) else 0
            match = "yes" if routed_latest == five_latest else "no"
            routed_recent = count_recent_signals(routed_signals, lookback=100)

            print(
                f"{symbol:>10} | {routed_name:>26} | {str(uses_five_signal):>9} | "
                f"{routed_latest:>13} | {five_latest:>10} | {match:>7} | "
                f"{routed_recent['total']:>15}"
            )

    print("\n=== WHAT TO LOOK FOR ===")
    print("- If routed_strategy is not five_signal_confluence_scalper, live is not using the 5-signal branch.")
    print("- If latest_routed and latest_5sig differ often, the live bot is not aligned with the new entry logic.")
    print("- If routed_last100 is high, the strategy is still producing many opportunities and may be too permissive.")


if __name__ == "__main__":
    main()
