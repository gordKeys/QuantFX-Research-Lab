from bootstrap import add_project_root
add_project_root()

import argparse
import json
from pathlib import Path

import pandas as pd

from mt5_broker_adapter import MT5BrokerAdapter, MT5UnavailableError


TIMEFRAME_MAP = {
    "M5": "TIMEFRAME_M5",
    "M15": "TIMEFRAME_M15",
    "M30": "TIMEFRAME_M30",
    "H1": "TIMEFRAME_H1",
    "H4": "TIMEFRAME_H4",
    "D1": "TIMEFRAME_D1",
}


def timeframe_attr(broker, timeframe_name):
    attr = TIMEFRAME_MAP.get(timeframe_name.upper())
    if attr is None:
        raise ValueError(f"Unsupported timeframe: {timeframe_name}")
    return getattr(broker.mt5, attr)


def load_symbol_universe():
    config_path = Path("configs/symbol_universe.json")
    if not config_path.exists():
        return []
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    candidates = payload.get("tournament_candidates", [])
    preferred_live = payload.get("preferred_live", [])
    merged = []
    for symbol in preferred_live + candidates:
        if symbol not in merged:
            merged.append(symbol)
    return merged


def symbol_meta(broker, symbol):
    info = broker.symbol_info(symbol)
    if info is None:
        return {}
    return {
        "symbol": symbol,
        "digits": getattr(info, "digits", None),
        "point": getattr(info, "point", None),
        "trade_contract_size": getattr(info, "trade_contract_size", None),
        "volume_min": getattr(info, "volume_min", None),
        "volume_step": getattr(info, "volume_step", None),
        "volume_max": getattr(info, "volume_max", None),
    }


def main():
    parser = argparse.ArgumentParser(description="Export MT5 OHLC data to CSV files.")
    parser.add_argument("--symbol", action="append", help="Repeatable symbol filter.")
    parser.add_argument("--symbols", nargs="+", help="Symbols to export. Defaults to config universe.")
    parser.add_argument("--timeframe", nargs="+", default=["M15"], choices=list(TIMEFRAME_MAP.keys()),
                        help="One or more timeframes -- SMC/AMD/price-action approaches typically need "
                        "a higher timeframe for structure/bias (H4, D1) plus a lower one for entries (M15, H1).")
    parser.add_argument("--bars", type=int, default=20000)
    parser.add_argument("--output-dir", default="data")
    args = parser.parse_args()

    symbols = args.symbols or args.symbol or load_symbol_universe()
    if not symbols:
        raise SystemExit("No symbols provided and configs/symbol_universe.json was empty.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        broker = MT5BrokerAdapter()
        broker.initialize()
    except MT5UnavailableError as exc:
        raise SystemExit(str(exc))

    exported = 0

    for symbol in symbols:
        for tf_name in args.timeframe:
            timeframe = timeframe_attr(broker, tf_name)
            print(f"Fetching {symbol} {tf_name}...")
            rates = broker.rates_copy(symbol, timeframe, args.bars)
            if rates is None or len(rates) == 0:
                print(f"  skipped: no data returned (symbol may not exist on this broker/account)")
                continue

            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df = df.set_index("time")
            keep_columns = ["open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
            df = df[keep_columns]

            csv_path = output_dir / f"{symbol}_{tf_name}.csv"
            df.to_csv(csv_path)

            meta = symbol_meta(broker, symbol)
            meta.update(
                {
                    "timeframe": tf_name,
                    "bars_requested": args.bars,
                    "bars_exported": len(df),
                }
            )
            meta_path = output_dir / f"{symbol}_{tf_name}_meta.json"
            with meta_path.open("w", encoding="utf-8") as handle:
                json.dump(meta, handle, indent=2, default=str)

            print(f"  saved {len(df)} bars -> {csv_path}")
            if len(df) < args.bars:
                print(
                    f"  NOTE: got {len(df)} of {args.bars} requested bars -- likely your broker's "
                    f"history limit for this symbol/timeframe, not an error here. For more history, "
                    f"either accept what's available or ask your broker/VPS to extend chart history depth."
                )
            exported += 1

    broker.shutdown()
    print(f"\nDone. Exported {exported} symbol/timeframe combinations.")


if __name__ == "__main__":
    main()
