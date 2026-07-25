"""
Measure YOUR actual swap (overnight financing) per symbol and direction,
from MT5 deal history -- the companion to measure_commission.py.

Needs MT5 running (i.e. your VPS back up and logged into the account). Not
runnable from this environment for the same reason measure_commission.py
isn't -- there is no live terminal to connect to here.

Why direction matters: swap is very often asymmetric between long and
short on the same symbol (FTMO's own published example: EURUSD long
-6.25 points/lot vs short -3.00 points/lot per night). A single flat swap
number per symbol would misprice whichever direction you trade more.

    python run_project.py swap --days 90

Output: configs/swap_map.json, in the same {"symbols": {...}} shape as
commission_map.json, which engine/cost_model.py reads automatically.
Symbols/directions with no trade history are left out rather than guessed.

NO TRADE HISTORY YET, OR WANT A FASTER READ?
    Right-click a symbol in Market Watch -> Specification. MT5 shows
    "Swap long" / "Swap short" directly (in points or as a %, depending on
    broker) without needing any trade history at all -- faster than this
    script if you just want the current numbers, though this script gives
    you what you actually paid, which also captures any triple-swap days
    that happened during the window.
"""

from bootstrap import add_project_root

add_project_root()

import argparse
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from mt5_broker_adapter import MT5BrokerAdapter, MT5UnavailableError


OUTPUT_PATH = Path("configs/swap_map.json")

# Only used to sanity-check what we measure -- never written to the map.
# From FTMO's own blog post on swap mechanics; verify against your own
# account and the current week's rates (these are typically republished
# weekly and move with interest-rate differentials).
PUBLISHED_REFERENCE = {
    "EURUSD": "long ~-6.25 pts/lot, short ~-3.00 pts/lot (example only)",
}

DEAL_ENTRY_IN = 0
DEAL_TYPE_BUY = 0


def measure(broker, days):
    """Group deals by position_id so swap (recorded on the closing deal)
    can be attributed to the direction set by the opening deal."""
    end = datetime.now(timezone.utc) + timedelta(days=1)
    start = end - timedelta(days=days + 1)

    deals = broker.mt5.history_deals_get(start, end)
    if deals is None:
        raise SystemExit(
            f"history_deals_get returned None: {broker.mt5.last_error()}. "
            f"Check the terminal is logged into the account you want to measure."
        )

    positions = defaultdict(lambda: {"symbol": None, "direction": None, "swap": 0.0, "volume": 0.0})

    for deal in deals:
        pos_id = getattr(deal, "position_id", None)
        symbol = getattr(deal, "symbol", "") or ""
        volume = float(getattr(deal, "volume", 0) or 0)
        swap = float(getattr(deal, "swap", 0) or 0)
        deal_type = getattr(deal, "type", None)
        entry = getattr(deal, "entry", None)

        if pos_id is None or not symbol:
            continue

        bucket = positions[pos_id]
        bucket["symbol"] = symbol
        bucket["swap"] += swap

        # The deal that OPENED the position tells us the direction. Swap
        # itself typically posts on the closing deal, which is why we
        # can't just read direction off the deal carrying the swap value.
        if entry == DEAL_ENTRY_IN:
            bucket["direction"] = "long" if deal_type == DEAL_TYPE_BUY else "short"
            bucket["volume"] = volume

    totals = defaultdict(lambda: {"swap": 0.0, "volume": 0.0, "positions": 0})
    for bucket in positions.values():
        if not bucket["symbol"] or bucket["direction"] is None or bucket["volume"] <= 0:
            continue
        key = (bucket["symbol"], bucket["direction"])
        totals[key]["swap"] += bucket["swap"]
        totals[key]["volume"] += bucket["volume"]
        totals[key]["positions"] += 1

    return totals


def main():
    parser = argparse.ArgumentParser(
        description="Derive real per-lot-per-night swap, by direction, from MT5 deal history."
    )
    parser.add_argument("--days", type=int, default=90,
                        help="How far back to look. Widen this if you trade rarely, or if you "
                             "want a window that includes at least one Wednesday rollover.")
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args()

    try:
        broker = MT5BrokerAdapter()
        broker.initialize()
    except MT5UnavailableError as exc:
        raise SystemExit(str(exc))

    account = broker.mt5.account_info()
    if account is not None:
        print(f"Account {account.login} on {account.server} ({account.currency})\n")

    totals = measure(broker, args.days)
    broker.shutdown()

    if not totals:
        print(f"No closed positions with an identifiable direction in the last {args.days} days.\n")
        print("Published reference (VERIFY against your own account, changes weekly):")
        for symbol, note in PUBLISHED_REFERENCE.items():
            print(f"  {symbol:<10} {note}")
        print(
            "\nFaster path if you don't want to wait for trade history: right-click the "
            "symbol in Market Watch -> Specification -> read Swap long / Swap short directly."
        )
        raise SystemExit(0)

    swap_map = defaultdict(dict)
    rows = []

    for (symbol, direction), bucket in sorted(totals.items()):
        volume = bucket["volume"]
        per_lot_per_position = bucket["swap"] / volume if volume else 0.0
        # This assumes ~1 night per position on average within the sample;
        # for a precise per-night figure you'd want nights_held per
        # position, but this script only has aggregate deal history, not
        # per-position hold time -- treat this as a reasonable estimate,
        # not an exact figure, and prefer a wider --days window to smooth
        # out any single anomalous multi-night hold.
        swap_map[symbol][direction] = round(per_lot_per_position, 4)
        rows.append((symbol, direction, bucket["positions"], volume, bucket["swap"], per_lot_per_position))

    print(f"{'symbol':<10}{'dir':<7}{'positions':>10}{'lots':>10}{'total_swap':>12}{'$/lot/night*':>14}")
    print("-" * 63)
    for symbol, direction, positions_count, volume, total_swap, per_lot in rows:
        print(f"{symbol:<10}{direction:<7}{positions_count:>10}{volume:>10.2f}{total_swap:>12.2f}{per_lot:>14.4f}")
    print("\n* Approximate -- assumes ~1 night held per position in the sample. Widen --days if "
          "your positions are typically held multiple nights, to average that out.")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_measured_at": datetime.now(timezone.utc).isoformat(),
        "_account": getattr(account, "login", None),
        "_server": getattr(account, "server", None),
        "_lookback_days": args.days,
        "_note": "Generated from real deal history, grouped by position to attribute swap to "
                 "the direction that opened it. engine/cost_model.py reads this automatically. "
                 "Delete and re-run after any account change or when rates are republished.",
        "symbols": dict(swap_map),
    }
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"\nWrote {output} -- engine/cost_model.py (and the backtester) will pick this up automatically.")


if __name__ == "__main__":
    main()
