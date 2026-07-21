"""
Export OHLC history + symbol metadata for the structure-research universe.

Run this on the VPS with MT5 open and logged in:

    python run_project.py export_structure --preset core --timeframe M15 H4 D1 --bars 60000

Why this exists alongside export_mt5_data.py:
  1. Broker symbol resolution. Almost every broker suffixes symbols (EURUSD.a,
     XAUUSDm, US30.cash). The old exporter asks for "EURUSD", gets None back,
     prints "skipped", and you silently lose the pair. This one looks up what
     the broker actually calls each instrument before giving up.
  2. symbol_select(). A symbol can exist and still return zero bars because it
     isn't in Market Watch. That failure looks identical to "symbol not offered".
     We select it first, then retry.
  3. Chunked history. copy_rates_from_pos caps out around 5-10k bars depending
     on terminal history settings. Structure work on H4/D1 needs years, not
     weeks, so we walk backwards in chunks until the broker stops giving.
  4. Spread capture per session. The single most expensive lesson in this
     project so far was that spread killed an edge that looked real without it.
     We record spread stats at export time, per instrument, so the screener can
     rank pairs on cost economics before any strategy code gets written.

Files produced per symbol/timeframe in data/:
    <SYMBOL>_<TF>.csv          OHLC + tick_volume + spread (points) + real_volume
    <SYMBOL>_<TF>_meta.json    point, digits, contract size, volume limits,
                               live spread snapshot, resolved broker name
    _export_manifest.json      what succeeded, what failed, and why
"""

from bootstrap import add_project_root

add_project_root()

import argparse
import json
import time
from datetime import datetime, timedelta, timezone
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

TIMEFRAME_MINUTES = {
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 60,
    "H4": 240,
    "D1": 1440,
}

CONFIG_PATH = Path("configs/structure_research_universe.json")

# Fallback if the config file is missing. Kept deliberately short -- the config
# file is the source of truth and carries the reasoning for each group.
FALLBACK_PRESETS = {
    "core": [
        "XAUUSD", "GBPUSD", "GBPJPY", "USDJPY", "EURUSD",
        "USOIL", "US30", "NAS100", "BTCUSD",
    ],
    "wide": [
        "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD",
        "EURJPY", "GBPJPY", "AUDJPY", "EURGBP", "GBPAUD", "EURAUD",
        "XAUUSD", "XAGUSD", "USOIL",
        "US30", "NAS100", "US500", "GER40",
        "BTCUSD", "ETHUSD",
    ],
}


def load_universe(preset):
    """Build the symbol list from the research config, falling back if absent."""
    if not CONFIG_PATH.exists():
        return FALLBACK_PRESETS.get(preset, FALLBACK_PRESETS["core"])

    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    groups = {
        key: value.get("symbols", [])
        for key, value in payload.items()
        if isinstance(value, dict) and "symbols" in value
    }

    if preset == "wide":
        wanted = list(groups.keys())
    elif preset == "core":
        # One representative per behaviour class rather than everything at once.
        wanted = ["majors", "metals", "energy", "indices_worth_checking"]
    else:
        wanted = [preset]

    symbols = []
    for group in wanted:
        for symbol in groups.get(group, []):
            if symbol not in symbols:
                symbols.append(symbol)

    return symbols or FALLBACK_PRESETS["core"]


def build_symbol_index(broker):
    """
    Map a normalised name -> list of real broker symbol names.

    Normalisation strips non-alphanumerics and uppercases, so EURUSD.a,
    EURUSD-raw and eurusd all collapse to EURUSD. Brokers also rename
    instruments outright (USOIL vs WTI vs CRUDE, US30 vs DJ30 vs WS30), which
    the ALIASES table below handles.
    """
    all_symbols = broker.mt5.symbols_get()
    index = {}
    if not all_symbols:
        return index

    for info in all_symbols:
        raw = info.name
        key = "".join(ch for ch in raw.upper() if ch.isalnum())
        index.setdefault(key, []).append(raw)

    return index


ALIASES = {
    "USOIL": ["USOIL", "WTI", "CRUDOIL", "CRUDEOIL", "XTIUSD", "OILUSD", "USCRUDE", "UKOIL"],
    "US30": ["US30", "DJ30", "WS30", "DOW30", "USA30", "US30CASH"],
    "NAS100": ["NAS100", "USTEC", "NDX100", "USA100", "NAS100CASH", "US100"],
    "US500": ["US500", "SPX500", "USA500", "SP500", "US500CASH"],
    "GER40": ["GER40", "DE40", "DAX40", "GER30", "DE30", "DAX"],
    "BTCUSD": ["BTCUSD", "BITCOIN", "BTCUSDT"],
    "ETHUSD": ["ETHUSD", "ETHEREUM", "ETHUSDT"],
    "XAUUSD": ["XAUUSD", "GOLD", "GOLDUSD"],
    "XAGUSD": ["XAGUSD", "SILVER", "SILVERUSD"],
}


def resolve_symbol(broker, index, wanted):
    """Return the broker's actual name for `wanted`, or None."""
    candidates = ALIASES.get(wanted.upper(), [wanted])
    if wanted not in candidates:
        candidates = [wanted] + candidates

    for candidate in candidates:
        key = "".join(ch for ch in candidate.upper() if ch.isalnum())
        matches = index.get(key)
        if matches:
            # Prefer the shortest name -- usually the plain/standard contract
            # rather than a swap-free or micro variant.
            return sorted(matches, key=len)[0]

    # Last resort: prefix match, e.g. wanted="XAUUSD", broker has "XAUUSD.pro"
    base = "".join(ch for ch in wanted.upper() if ch.isalnum())
    prefix_hits = [
        names[0] for key, names in index.items() if key.startswith(base)
    ]
    if prefix_hits:
        return sorted(prefix_hits, key=len)[0]

    return None


def fetch_history(broker, symbol, timeframe, tf_name, target_bars, chunk=5000):
    """
    Walk backwards through history in chunks until we have target_bars or the
    broker stops returning new data.

    copy_rates_from(symbol, tf, dt, n) returns up to n bars ending at dt. We
    move dt back by the span we just received and ask again.
    """
    minutes = TIMEFRAME_MINUTES[tf_name]
    cursor = datetime.now(timezone.utc) + timedelta(days=1)
    frames = []
    collected = 0
    stall = 0

    while collected < target_bars and stall < 3:
        want = min(chunk, target_bars - collected)
        rates = broker.mt5.copy_rates_from(symbol, timeframe, cursor, want)

        if rates is None or len(rates) == 0:
            stall += 1
            cursor = cursor - timedelta(minutes=minutes * chunk)
            continue

        frame = pd.DataFrame(rates)
        frame["time"] = pd.to_datetime(frame["time"], unit="s")
        frames.append(frame)
        collected += len(frame)

        earliest = frame["time"].min()
        new_cursor = earliest.to_pydatetime().replace(tzinfo=timezone.utc) - timedelta(minutes=minutes)

        if new_cursor >= cursor:
            break
        cursor = new_cursor

        if len(frame) < want:
            stall += 1
        else:
            stall = 0

        time.sleep(0.05)  # be polite to the terminal

    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset="time").sort_values("time").set_index("time")

    for column in ["open", "high", "low", "close", "tick_volume", "spread", "real_volume"]:
        if column not in df.columns:
            df[column] = 0

    return df[["open", "high", "low", "close", "tick_volume", "spread", "real_volume"]]


def symbol_meta(broker, resolved, requested):
    info = broker.symbol_info(resolved)
    if info is None:
        return {"symbol": requested, "broker_symbol": resolved}

    tick = broker.mt5.symbol_info_tick(resolved)
    live_spread = None
    if tick is not None and getattr(tick, "ask", None) and getattr(tick, "bid", None):
        live_spread = round(tick.ask - tick.bid, 10)

    return {
        "symbol": requested,
        "broker_symbol": resolved,
        "description": getattr(info, "description", None),
        "digits": getattr(info, "digits", None),
        "point": getattr(info, "point", None),
        "spread_points_now": getattr(info, "spread", None),
        "spread_price_now": live_spread,
        "spread_float": getattr(info, "spread_float", None),
        "trade_contract_size": getattr(info, "trade_contract_size", None),
        "trade_tick_value": getattr(info, "trade_tick_value", None),
        "trade_tick_size": getattr(info, "trade_tick_size", None),
        "volume_min": getattr(info, "volume_min", None),
        "volume_step": getattr(info, "volume_step", None),
        "volume_max": getattr(info, "volume_max", None),
        "swap_long": getattr(info, "swap_long", None),
        "swap_short": getattr(info, "swap_short", None),
    }


def main():
    parser = argparse.ArgumentParser(description="Export structure-research OHLC data from MT5.")
    parser.add_argument("--preset", default="core",
                        help="core | wide | a group name from configs/structure_research_universe.json")
    parser.add_argument("--symbols", nargs="+", help="Explicit symbol list, overrides --preset.")
    parser.add_argument("--timeframe", nargs="+", default=["M15", "H4", "D1"],
                        choices=list(TIMEFRAME_MAP.keys()),
                        help="Structure work needs a bias timeframe (H4/D1) and an entry timeframe (M15/H1).")
    parser.add_argument("--bars", type=int, default=60000)
    parser.add_argument("--output-dir", default="data")
    args = parser.parse_args()

    symbols = args.symbols or load_universe(args.preset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        broker = MT5BrokerAdapter()
        broker.initialize()
    except MT5UnavailableError as exc:
        raise SystemExit(str(exc))

    index = build_symbol_index(broker)
    print(f"Broker exposes {sum(len(v) for v in index.values())} symbols.\n")

    manifest = {"exported": [], "failed": [], "run_at": datetime.now(timezone.utc).isoformat()}

    for requested in symbols:
        resolved = resolve_symbol(broker, index, requested)

        if resolved is None:
            print(f"{requested}: NOT OFFERED by this broker -- skipping.")
            manifest["failed"].append({"symbol": requested, "reason": "not offered by broker"})
            continue

        if resolved != requested:
            print(f"{requested}: broker calls this '{resolved}'")

        if not broker.mt5.symbol_select(resolved, True):
            print(f"{requested}: could not add '{resolved}' to Market Watch -- skipping.")
            manifest["failed"].append({"symbol": requested, "reason": "symbol_select failed"})
            continue

        meta_base = symbol_meta(broker, resolved, requested)

        for tf_name in args.timeframe:
            timeframe = getattr(broker.mt5, TIMEFRAME_MAP[tf_name])
            print(f"  fetching {requested} {tf_name}...", end=" ", flush=True)

            df = fetch_history(broker, resolved, timeframe, tf_name, args.bars)

            if df is None or df.empty:
                print("no data returned")
                manifest["failed"].append(
                    {"symbol": requested, "timeframe": tf_name, "reason": "no bars returned"}
                )
                continue

            csv_path = output_dir / f"{requested}_{tf_name}.csv"
            df.to_csv(csv_path)

            meta = dict(meta_base)
            meta.update({
                "timeframe": tf_name,
                "bars_requested": args.bars,
                "bars_exported": len(df),
                "first_bar": str(df.index.min()),
                "last_bar": str(df.index.max()),
                "median_spread_points": float(df["spread"].median()),
                "p90_spread_points": float(df["spread"].quantile(0.90)),
            })
            meta_path = output_dir / f"{requested}_{tf_name}_meta.json"
            with meta_path.open("w", encoding="utf-8") as handle:
                json.dump(meta, handle, indent=2, default=str)

            span_days = (df.index.max() - df.index.min()).days
            print(f"{len(df)} bars covering ~{span_days} days -> {csv_path.name}")

            manifest["exported"].append({
                "symbol": requested,
                "broker_symbol": resolved,
                "timeframe": tf_name,
                "bars": len(df),
                "days": span_days,
            })

    broker.shutdown()

    manifest_path = output_dir / "_export_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"\nExported {len(manifest['exported'])} symbol/timeframe combinations.")
    if manifest["failed"]:
        print(f"{len(manifest['failed'])} failed -- see {manifest_path}")
    print("\nNext: python run_project.py screen --timeframe M15")


if __name__ == "__main__":
    main()
