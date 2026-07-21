"""
Measure YOUR actual commission per lot, per symbol, from MT5 deal history.

Published commission tables are a starting point, not an answer. The figure that
matters depends on your account type (Raw vs Normal), your region, and the
instrument class -- FTMO charges nothing on indices and energy while charging
per-lot on FX and a percentage of notional on metals and crypto. Feeding one
flat number into the screener would overstate index costs and understate metals.

This script pulls your closed deals, groups by symbol, and computes the real
figure from money you actually paid.

    python run_project.py commission --days 90

Output: configs/commission_map.json, which instrument_screener.py reads
automatically. Symbols with no trade history are left out rather than guessed at.

NO TRADE HISTORY YET?
    Place one 0.01-lot trade on each instrument class you care about (one FX
    pair, gold, one index, oil), close it immediately, then run this. The
    commission is charged on open, so even an instantly-closed micro position
    prices the whole class. Cost is a few cents and it beats assuming.
"""

from bootstrap import add_project_root

add_project_root()

import argparse
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from mt5_broker_adapter import MT5BrokerAdapter, MT5UnavailableError


OUTPUT_PATH = Path("configs/commission_map.json")

# Only used to sanity-check what we measure -- never written to the map.
# Sourced from FTMO's published schedule; verify against your own account.
PUBLISHED_REFERENCE = {
    "fx": "~$5.00 per lot round trip (Raw: ~$2.50 per side)",
    "metals": "percentage of volume (~0.0014%), not a flat per-lot fee",
    "indices": "$0",
    "energy": "$0",
    "crypto": "percentage of volume (~0.065%)",
}


def measure(broker, days):
    """Sum commission and volume per symbol across closed deals."""
    end = datetime.now(timezone.utc) + timedelta(days=1)
    start = end - timedelta(days=days + 1)

    deals = broker.mt5.history_deals_get(start, end)
    if deals is None:
        raise SystemExit(
            f"history_deals_get returned None: {broker.mt5.last_error()}. "
            f"Check the terminal is logged into the account you want to measure."
        )

    totals = defaultdict(lambda: {"commission": 0.0, "volume": 0.0, "deals": 0})

    for deal in deals:
        symbol = getattr(deal, "symbol", "") or ""
        volume = float(getattr(deal, "volume", 0) or 0)
        commission = float(getattr(deal, "commission", 0) or 0)

        # Skip balance operations, credits, and anything without a symbol.
        if not symbol or volume <= 0:
            continue

        bucket = totals[symbol]
        bucket["commission"] += commission
        bucket["volume"] += volume
        bucket["deals"] += 1

    return totals


def main():
    parser = argparse.ArgumentParser(
        description="Derive real per-lot commission from MT5 deal history."
    )
    parser.add_argument("--days", type=int, default=90,
                        help="How far back to look. Widen this if you trade rarely.")
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
        print(f"No closed deals with volume in the last {args.days} days.\n")
        print("Published FTMO reference (VERIFY against your own account):")
        for asset_class, note in PUBLISHED_REFERENCE.items():
            print(f"  {asset_class:<10} {note}")
        print(
            "\nTo measure it exactly: open and immediately close one 0.01-lot\n"
            "position on each class you plan to trade (one FX pair, XAUUSD, one\n"
            "index, USOIL), then re-run this. Commission is charged on open, so\n"
            "an instantly-closed micro trade prices the class for a few cents."
        )
        raise SystemExit(0)

    commission_map = {}
    rows = []

    for symbol, bucket in sorted(totals.items()):
        volume = bucket["volume"]
        # deal.commission is negative (money leaving the account); make it a cost.
        paid = abs(bucket["commission"])
        per_lot_side = paid / volume if volume else 0.0

        # Each round trip is two deals (one in, one out). Summing commission
        # across both and dividing by summed volume already yields the per-side
        # average, so the round trip is twice that -- unless the broker charges
        # entry-only, which shows up as roughly half the expected figure.
        round_trip = per_lot_side * 2

        commission_map[symbol] = {
            "per_lot_round_trip": round(round_trip, 4),
            "per_lot_per_side": round(per_lot_side, 4),
            "measured_from_deals": bucket["deals"],
            "measured_volume_lots": round(volume, 2),
        }

        rows.append((symbol, bucket["deals"], volume, paid, per_lot_side, round_trip))

    print(f"{'symbol':<12}{'deals':>7}{'lots':>10}{'paid':>10}{'$/lot/side':>13}{'$/lot RT':>11}")
    print("-" * 63)
    for symbol, deals, volume, paid, per_side, round_trip in rows:
        print(f"{symbol:<12}{deals:>7}{volume:>10.2f}{paid:>10.2f}{per_side:>13.2f}{round_trip:>11.2f}")

    zero = [symbol for symbol, data in commission_map.items()
            if data["per_lot_round_trip"] == 0]
    if zero:
        print(
            f"\nZero commission measured on: {', '.join(zero)}. For indices and "
            f"energy that is expected and correct. On FX or metals it more "
            f"likely means the cost is built into the spread on your account "
            f"type rather than billed separately."
        )

    thin = [symbol for symbol, data in commission_map.items()
            if data["measured_volume_lots"] < 0.1]
    if thin:
        print(
            f"\nThin sample on: {', '.join(thin)} (under 0.1 lots measured). "
            f"Rounding in the commission field can distort small samples -- "
            f"treat these as approximate."
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_measured_at": datetime.now(timezone.utc).isoformat(),
        "_account": getattr(account, "login", None),
        "_server": getattr(account, "server", None),
        "_lookback_days": args.days,
        "_note": "Generated from real deal history. instrument_screener.py reads "
                 "this automatically. Delete and re-run after any account change.",
        "symbols": commission_map,
    }
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"\nWrote {output} -- the screener will pick this up automatically.")
    print("Run: python run_project.py screen --timeframe M15")


if __name__ == "__main__":
    main()
