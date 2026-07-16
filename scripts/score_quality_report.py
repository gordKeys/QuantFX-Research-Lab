"""
Confluence score quality report.

live_runner.py now logs the confluence score (long_score/short_score,
whichever side fired) at the moment of every entry signal, in the
"order_accepted" event of logs/live_run_<day>.jsonl, along with the
resulting MT5 order/deal ticket.

trade_analyzer.py's output CSV (logs/trade_analysis.csv) has one row per
closed trade with a position_id.

This script joins the two by ticket and buckets outcomes by entry score, so
you can see whether low-confluence entries (score exactly at min_score) are
dragging down the average, or whether score doesn't actually predict
anything here -- either answer is useful and both are hypotheses, not
foregone conclusions.

Usage:
    python scripts/score_quality_report.py \
        --logs logs/live_run_2026-07-15.jsonl logs/live_run_2026-07-16.jsonl \
        --trades logs/trade_analysis.csv

Note on matching: MT5's OrderSendResult exposes both `.order` (the order
ticket) and `.deal` (the resulting deal ticket). Which one lines up with a
closed position's `position_id` can vary by broker/account type. This script
tries both and reports the match rate for each so you can tell quickly which
one is right for your setup -- if both match rates are low, check that
manually with one live ticket before trusting the buckets below.
"""
from bootstrap import add_project_root
add_project_root()

import argparse
import json
from pathlib import Path

import pandas as pd


def _load_entries(log_paths):
    rows = []
    for path in log_paths:
        path = Path(path)
        if not path.exists():
            print(f"Skipping missing log: {path}")
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if payload.get("event") == "order_accepted" and payload.get("score") is not None:
                    rows.append(payload)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Bucket closed-trade outcomes by logged entry confluence score")
    parser.add_argument("--logs", nargs="+", required=True, help="One or more live_run_*.jsonl log files")
    parser.add_argument("--trades", default="logs/trade_analysis.csv", help="trade_analyzer.py CSV output")
    args = parser.parse_args()

    entries = _load_entries(args.logs)
    if entries.empty:
        print(
            "No order_accepted events with a score found in the given logs. "
            "This log format is new -- only trades placed after this update "
            "will have a score to join against."
        )
        return

    trades_path = Path(args.trades)
    if not trades_path.exists():
        raise SystemExit(f"No such file: {trades_path}. Run trade_analyzer.py first.")
    trades = pd.read_csv(trades_path)
    if "position_id" not in trades.columns:
        raise SystemExit("trade_analysis.csv has no position_id column; re-run trade_analyzer.py to regenerate it.")

    for ticket_field in ("order", "deal"):
        if ticket_field not in entries.columns:
            continue
        merged = trades.merge(
            entries[[ticket_field, "score", "symbol"]].dropna(subset=[ticket_field]),
            left_on="position_id",
            right_on=ticket_field,
            how="inner",
            suffixes=("", "_entry"),
        )
        match_rate = len(merged) / len(trades) * 100.0 if len(trades) else 0.0
        print(f"\n=== Matched via '{ticket_field}' ticket: {len(merged)}/{len(trades)} trades ({match_rate:.0f}%) ===")
        if merged.empty:
            continue

        by_score = merged.groupby("score").agg(
            trades=("profit_usd", "count"),
            win_rate=("profit_usd", lambda s: (s > 0).mean()),
            total_pnl=("profit_usd", "sum"),
            avg_pnl=("profit_usd", "mean"),
        )
        print(f"{'score':>5} | {'trades':>6} | {'win_rate':>8} | {'total_pnl':>10} | {'avg_pnl':>8}")
        print("-" * 50)
        for score, row in by_score.sort_index().iterrows():
            print(
                f"{score:5.0f} | {int(row['trades']):6d} | {row['win_rate']:8.2%} | "
                f"{row['total_pnl']:10.2f} | {row['avg_pnl']:8.2f}"
            )

    print(
        "\nIf neither match rate above is reasonably high (most trades matched), "
        "confirm which OrderSendResult field equals position_id on your broker "
        "before trusting the score buckets -- otherwise this is comparing the "
        "wrong trades to the wrong scores."
    )


if __name__ == "__main__":
    main()
