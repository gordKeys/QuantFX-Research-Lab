"""
Instrument screener -- run this BEFORE writing any SMC/structure strategy code.

The question this answers is not "does my strategy work". It is the question
that killed the last approach: on this instrument, is a typical structural move
big enough that spread doesn't eat it?

AUDUSD looked like the best pair in the book across 6/6 walk-forward folds. With
real spread modelled it went 0/6 and -1331. The edge was never there; the cost
model was missing. Screening instruments on cost economics first means we don't
spend three weeks building order-block detection for a pair that can't pay for
its own transaction costs.

Metrics, per symbol:

  legs_per_day        How many swing legs (structure moves) actually form per
                      day. This sets the natural trade frequency ceiling of ANY
                      structure-based approach on this instrument. If a pair
                      offers 3 legs/day, a 20-trade/day target across 5 pairs
                      means taking essentially every leg -- which is scalping
                      with different vocabulary.

  median_leg          Median swing-leg size in price terms. The raw size of a
                      structural move.

  leg_to_spread       median_leg / median_spread. The headroom ratio, and the
                      single most important number here. A round trip costs
                      roughly one spread. At 20:1 you keep ~95% of a leg. At
                      4:1 you keep ~75% and need a very high hit rate. Under
                      3:1, structure trading on that instrument is not viable
                      no matter how good the entry logic is.

  junk_leg_pct        Share of legs smaller than 3x round-trip cost -- moves
                      that are structurally real but not worth trading. High
                      values mean heavy filtering is mandatory.

  spread_by_session   Spread is not constant. Late-NY/early-Asia spread is
                      routinely 3-5x the London figure on metals and indices.
                      A 24h average hides this. Session columns show where the
                      instrument is actually tradeable.

Usage:
    python run_project.py screen --timeframe M15
    python run_project.py screen --timeframe H4 --swing-lookback 3
"""

from bootstrap import add_project_root

add_project_root()

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


# Broker server time is typically UTC+2/+3, so these are approximate windows in
# server time, not exact market hours. Good enough to expose cost asymmetry.
class MissingTickMeta(RuntimeError):
    """Raised when commission was requested but the meta lacks tick pricing."""


COMMISSION_MAP_PATH = Path("configs/commission_map.json")

# Fallback commission by asset class, in account currency per lot round trip.
# Without this, a screen ranks measured symbols against unmeasured ones and the
# ordering is meaningless: the pairs you happen to have traded look expensive
# purely because they are the only ones carrying their true cost.
CLASS_COMMISSION = {
    "fx": 5.04,       # measured: $2.52/side on both EURUSD and AUDUSD
    "metal": 6.00,    # measured on XAUUSD, thin sample; percentage-of-notional
    "index": 0.0,     # FTMO charges nothing on indices
    "energy": 0.0,    # nor on energy
    "crypto": 0.0,    # percentage of volume, usually folded into spread
}

INDEX_HINTS = ("US30", "US500", "NAS100", "GER40", "UK100", "JP225", "USTEC", "US100")
CRYPTO_HINTS = ("BTC", "ETH", "LTC", "XRP", "SOL")
METAL_HINTS = ("XAU", "XAG", "XPT", "XPD", "GOLD", "SILVER")
ENERGY_HINTS = ("OIL", "WTI", "BRENT", "NGAS", "XTI", "XBR")


def classify(symbol):
    upper = symbol.upper()
    if any(hint in upper for hint in CRYPTO_HINTS):
        return "crypto"
    if any(hint in upper for hint in METAL_HINTS):
        return "metal"
    if any(hint in upper for hint in ENERGY_HINTS):
        return "energy"
    if any(upper.startswith(hint) for hint in INDEX_HINTS):
        return "index"
    return "fx"


def load_commission_map():
    """
    Per-symbol commission measured from real deal history, if available.

    A single flat --commission-per-lot is wrong on a mixed universe: FTMO
    charges nothing on indices and energy, per-lot on FX, and a percentage of
    notional on metals and crypto. Applying one number to all of them would
    penalise indices for a fee they don't charge. This map, produced by
    measure_commission.py, takes priority over the flag.
    """
    if not COMMISSION_MAP_PATH.exists():
        return {}
    try:
        with COMMISSION_MAP_PATH.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return {}
    return {
        symbol: data.get("per_lot_round_trip", 0.0)
        for symbol, data in payload.get("symbols", {}).items()
    }


SESSIONS = {
    "asia": (0, 7),
    "london": (7, 13),
    "ny": (13, 21),
    "late": (21, 24),
}


def discover(data_dir, timeframe):
    """Find every <SYMBOL>_<TF>.csv in the data directory."""
    pattern = re.compile(rf"^(.+)_{re.escape(timeframe)}\.csv$")
    found = []
    for path in sorted(Path(data_dir).glob(f"*_{timeframe}.csv")):
        match = pattern.match(path.name)
        if match:
            found.append((match.group(1), path))
    return found


def load_meta(data_dir, symbol, timeframe):
    path = Path(data_dir) / f"{symbol}_{timeframe}_meta.json"
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return {}


def infer_point(df, meta):
    """
    Price value of one MT5 point. Prefer broker metadata; otherwise infer from
    the number of decimals actually present in the close prices.
    """
    point = meta.get("point")
    if point:
        return float(point)

    sample = df["close"].dropna().head(500)
    decimals = 0
    for value in sample:
        text = f"{value:.10f}".rstrip("0")
        if "." in text:
            decimals = max(decimals, len(text.split(".")[1]))
    decimals = min(decimals, 5)
    return 10.0 ** (-decimals)


def find_swings(df, lookback=3):
    """
    Fractal swing detection: a swing high is a bar whose high exceeds the highs
    of `lookback` bars either side. Same in reverse for lows.

    Then we enforce alternation (high, low, high, low...) keeping the most
    extreme point in each run, which is what turns raw fractals into a clean
    zigzag of structural legs.

    Deliberately simple and parameter-light. A leg here is not a trade signal --
    it is a measurement of how far this instrument travels between structural
    turning points, which is the thing we need before deciding whether the
    instrument is worth modelling at all.
    """
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    n = len(df)

    points = []  # (index, price, kind) kind: 1 = high, -1 = low

    for i in range(lookback, n - lookback):
        window_high = high[i - lookback:i + lookback + 1]
        window_low = low[i - lookback:i + lookback + 1]

        if high[i] == window_high.max() and (window_high.argmax() == lookback):
            points.append((i, high[i], 1))
        elif low[i] == window_low.min() and (window_low.argmin() == lookback):
            points.append((i, low[i], -1))

    if not points:
        return []

    # Enforce alternation, keeping the extreme of each same-kind run.
    zigzag = [points[0]]
    for idx, price, kind in points[1:]:
        last_idx, last_price, last_kind = zigzag[-1]
        if kind == last_kind:
            better = price > last_price if kind == 1 else price < last_price
            if better:
                zigzag[-1] = (idx, price, kind)
        else:
            zigzag.append((idx, price, kind))

    return zigzag


def commission_in_price(meta, commission_per_lot, point):
    """
    Convert round-trip commission ($/lot) into price terms for this instrument.

    This matters more than it looks. On a raw-spread account EURUSD might show a
    1-point (0.1 pip) spread while charging $7/lot round trip -- which is 0.7
    pips, seven times the spread. Screening on spread alone would rank that pair
    as nearly free to trade when the real cost is 8x higher. Any instrument
    ranking that ignores commission is measuring the wrong thing.
    """
    if not commission_per_lot:
        return 0.0

    tick_value = meta.get("trade_tick_value")
    tick_size = meta.get("trade_tick_size") or point

    if not tick_value or not tick_size:
        # Silently returning 0 here would make a commission-inclusive screen
        # look identical to a spread-only one. Signal it instead.
        raise MissingTickMeta(
            "no trade_tick_value/trade_tick_size in meta -- re-export with "
            "export_structure_data.py so commission can be priced"
        )

    # $ per 1.0 price unit per lot = tick_value / tick_size
    dollars_per_price_unit = float(tick_value) / float(tick_size)
    if dollars_per_price_unit <= 0:
        return 0.0

    return commission_per_lot / dollars_per_price_unit


def resolve_commission(symbol, meta, commission_map, fallback):
    """Prefer the measured figure; fall back to the flat flag."""
    if symbol in commission_map:
        return commission_map[symbol], "measured"

    broker_symbol = meta.get("broker_symbol")
    if broker_symbol and broker_symbol in commission_map:
        return commission_map[broker_symbol], "measured"

    if fallback:
        return fallback, "flag"

    asset_class = classify(symbol)
    return CLASS_COMMISSION.get(asset_class, 0.0), f"class:{asset_class}"


def screen_symbol(symbol, path, timeframe, data_dir, lookback,
                  commission_per_lot=0.0, commission_map=None):
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").sort_index()

    if len(df) < 200:
        return None

    meta = load_meta(data_dir, symbol, timeframe)
    point = infer_point(df, meta)

    # spread column is in points -> convert to price
    spread_price = df["spread"].astype(float) * point
    median_spread = float(spread_price.median())
    if median_spread <= 0:
        # Some feeds report zero spread. Fall back to broker metadata snapshot.
        median_spread = float(meta.get("spread_price_now") or 0) or np.nan

    zigzag = find_swings(df, lookback=lookback)
    if len(zigzag) < 10:
        return None

    legs = [abs(zigzag[i][1] - zigzag[i - 1][1]) for i in range(1, len(zigzag))]
    legs = np.array([leg for leg in legs if leg > 0])
    if legs.size == 0:
        return None

    span_days = max((df.index.max() - df.index.min()).days, 1)
    # Markets are closed ~2 of every 7 days for non-crypto; approximate.
    trading_days = span_days * (5 / 7) if not symbol.upper().startswith(("BTC", "ETH")) else span_days

    # ATR(14) median -- the volatility yardstick every ratio below is scaled to.
    prev_close = df["close"].shift(1)
    true_range = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr_value = float(true_range.rolling(14).mean().median())
    atr = atr_value

    median_leg = float(np.median(legs))
    p25_leg = float(np.percentile(legs, 25))
    p75_leg = float(np.percentile(legs, 75))

    per_lot, commission_source = resolve_commission(
        symbol, meta, commission_map or {}, commission_per_lot
    )
    try:
        commission = commission_in_price(meta, per_lot, point)
        commission_note = ""
    except MissingTickMeta as exc:
        commission = 0.0
        commission_note = str(exc)
    # Effective round-trip cost: cross the spread once, plus commission.
    round_trip = median_spread + commission
    leg_to_spread = median_leg / round_trip if round_trip and not np.isnan(round_trip) else np.nan
    junk_threshold = 3 * round_trip
    junk_pct = float((legs < junk_threshold).mean() * 100) if round_trip else np.nan

    row = {
        "symbol": symbol,
        "bars": len(df),
        "days": span_days,
        "median_leg": median_leg,
        "leg_iqr": f"{p25_leg:.5g}-{p75_leg:.5g}",
        "median_spread": median_spread,
        "commission_px": commission,
        "commission_per_lot": per_lot,
        "commission_source": commission_source,
        "commission_note": commission_note,
        "round_trip_cost": round_trip,
        "leg_to_spread": leg_to_spread,
        "junk_leg_pct": junk_pct,
        "legs_per_day": len(legs) / trading_days,
        "big_legs_per_day": int((legs >= atr_value * 2).sum()) / trading_days,
        "median_leg_atr": median_leg / atr_value if atr_value else np.nan,
        "atr": atr,
        "atr_to_spread": atr / round_trip if round_trip else np.nan,
    }

    hours = df.index.hour
    for name, (start, end) in SESSIONS.items():
        mask = (hours >= start) & (hours < end)
        if mask.any():
            row[f"spread_{name}"] = float(spread_price[mask].median() / point)
        else:
            row[f"spread_{name}"] = np.nan

    return row


def verdict(leg_to_spread):
    """
    Thresholds recalibrated after the first full-universe run marked all 21
    instruments "strong", which is the same as marking none of them.

    At swing scale on M15, cost simply is not the binding constraint anywhere --
    ratios come in between 20:1 and 900:1. So this now grades relative headroom
    rather than pretending there is a viability cliff:

      >=100  cost is a rounding error
      >=40   comfortable
      >=20   fine at swing targets, fatal at scalp targets
      <20    only worth trading for moves well above the median leg
    """
    if pd.isna(leg_to_spread):
        return "no cost data"
    if leg_to_spread >= 100:
        return "negligible cost"
    if leg_to_spread >= 40:
        return "comfortable"
    if leg_to_spread >= 20:
        return "swing-only"
    return "large-moves-only"


def main():
    parser = argparse.ArgumentParser(description="Rank instruments on structural-move vs spread economics.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--timeframe", default="M15")
    parser.add_argument("--swing-lookback", type=int, default=3,
                        help="Bars either side required to confirm a fractal swing. "
                             "Higher = fewer, larger legs.")
    parser.add_argument("--commission-per-lot", type=float, default=0.0,
                        help="Round-trip commission in account currency per 1.0 lot. "
                             "Check your FTMO/broker spec -- on raw-spread accounts this is "
                             "usually the LARGER half of your cost and flips the ranking.")
    parser.add_argument("--output", default="reports/instrument_screen.csv")
    args = parser.parse_args()

    found = discover(args.data_dir, args.timeframe)
    if not found:
        raise SystemExit(
            f"No *_{args.timeframe}.csv files in {args.data_dir}/. "
            f"Run the exporter first."
        )

    commission_map = load_commission_map()
    if commission_map:
        print(f"Using measured commission for {len(commission_map)} symbols "
              f"from {COMMISSION_MAP_PATH}.")
    elif not args.commission_per_lot:
        print("No commission data. Run: python run_project.py commission --days 90")

    rows = []
    for symbol, path in found:
        try:
            row = screen_symbol(symbol, path, args.timeframe, args.data_dir,
                                args.swing_lookback, args.commission_per_lot,
                                commission_map)
        except Exception as exc:
            print(f"  {symbol}: failed ({exc})")
            continue
        if row:
            rows.append(row)

    if not rows:
        raise SystemExit("No symbol produced enough swings to screen.")

    table = pd.DataFrame(rows).sort_values("leg_to_spread", ascending=False)

    if args.commission_per_lot or commission_map:
        stale = table.loc[table["commission_note"] != "", "symbol"].tolist()
        if stale:
            print(
                f"\nWARNING: commission could NOT be applied to {', '.join(stale)} "
                f"-- their meta.json is missing tick pricing, so those rows are "
                f"spread-only and their ratios are optimistic. Re-export with "
                f"export_structure_data.py."
            )
    table["verdict"] = table["leg_to_spread"].apply(verdict)

    display = table[[
        "symbol", "bars", "days", "median_leg", "round_trip_cost",
        "leg_to_spread", "median_leg_atr", "big_legs_per_day",
        "commission_source", "verdict",
    ]].copy()
    display["median_leg"] = display["median_leg"].map(lambda v: f"{v:.5g}")
    display["round_trip_cost"] = display["round_trip_cost"].map(lambda v: f"{v:.5g}")
    display["leg_to_spread"] = display["leg_to_spread"].map(lambda v: f"{v:.1f}")
    display["median_leg_atr"] = display["median_leg_atr"].map(lambda v: f"{v:.2f}")
    display["big_legs_per_day"] = display["big_legs_per_day"].map(lambda v: f"{v:.2f}")

    print(f"\n=== Instrument screen -- {args.timeframe}, swing lookback {args.swing_lookback} ===")
    if commission_map:
        print("Cost model: median spread + per-symbol measured commission\n")
    elif args.commission_per_lot:
        print(f"Cost model: median spread + assumed ${args.commission_per_lot}/lot "
              f"round trip -- measure it with 'commission' mode instead\n")
    else:
        print("Cost model: median spread ONLY -- ratios below are optimistic\n")
    print(display.to_string(index=False))

    spread_cols = [c for c in table.columns if c.startswith("spread_")]
    print(f"\n--- Median spread by session (points) ---\n")
    print(table[["symbol"] + spread_cols].to_string(index=False, float_format=lambda v: f"{v:.1f}"))

    total_big = table["big_legs_per_day"].sum()
    print(f"\nLegs >= 2x ATR per day across this universe: {total_big:.1f}")
    print("These are the moves large enough that cost is irrelevant and a")
    print("structural target is worth holding for. Raw leg counts are NOT shown")
    print("here because fractal density is a function of bar count and lookback,")
    print("not of the instrument -- that column measured the detector, not the market.")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    print(f"\nFull table -> {output}")


if __name__ == "__main__":
    main()
