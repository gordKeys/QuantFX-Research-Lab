"""
Horizon sweep -- does the sign flip track calendar time or bar size?

The battery produced one durable result and one confusing one.

  DURABLE   Weak-close bars mean-revert. Same sign on M15 and H4, 4-5 sigma,
            ~1.5 points. This was supposed to be a CONTROL, not a hypothesis.

  CONFUSING Decisive breaks reverse on M15 (-0.021) and continue on H4
            (+0.017). Under the pre-registered rule that fails: opposite signs
            means no result.

But the horizon was fixed at 20 BARS, which is 5 hours on M15 and 3.3 days on
H4. Those are not the same experiment. "M15 reverses, H4 continues" may simply
be "prices revert over hours and trend over days" -- one effect, sampled at two
points on a curve, rather than two contradictory results.

This script settles it by holding calendar time constant instead of bar count.
It runs the same test at matched real-time horizons on both timeframes. If the
M15 and H4 curves line up when plotted against hours, the timeframe was never
the variable and there is a genuine holding-period effect underneath -- which
is directly relevant to "few substantial trades", since it would tell you how
long a position needs to be held to sit on the right side of it.

If the curves do not line up, the effect is an artifact of bar construction and
the stopping rule applies.

    python run_project.py horizon --timeframe M15
    python run_project.py horizon --timeframe H4
"""

from bootstrap import add_project_root

add_project_root()

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.hypotheses import _control_events, _rate, causal_levels
from analysis.structure import atr

TF_MINUTES = {"M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}

# Calendar horizons to test, in hours. Chosen to span the range where the M15
# and H4 runs disagreed: 5 hours (M15's original) up to 80 hours (H4's).
TARGET_HOURS = [1, 2, 5, 12, 24, 48, 80, 160]


def events_decisive_break(df, atr_values, lookback=5, buffer_atr=0.25, cooldown=20):
    """
    A close beyond the most recent known swing extreme. Direction = the break
    direction, so a positive edge means breaks continue and a negative edge
    means they reverse.

    This is the merged version of what the battery ran as two separate tests.
    `bos_continuation` and `ctl_clean_break` summed to zero on both timeframes
    because they were the same event scored in opposite directions -- one
    result reported twice with a sign flip, which made a single finding look
    like two corroborating ones.
    """
    close = df["close"].to_numpy()
    last_high, last_low = causal_levels(df, lookback)
    events = []
    last_fired = -10 ** 9

    for i in range(lookback * 2 + 1, len(df)):
        band = atr_values[i]
        if np.isnan(band) or band <= 0:
            continue
        if i - last_fired < cooldown:
            continue
        if not np.isnan(last_high[i]) and close[i] > last_high[i] + band * buffer_atr:
            events.append((i, 1))
            last_fired = i
        elif not np.isnan(last_low[i]) and close[i] < last_low[i] - band * buffer_atr:
            events.append((i, -1))
            last_fired = i
    return events


def events_weak_close(df, atr_values, lookback=5, cooldown=20, pctile=0.25):
    """The one thing that survived the battery, carried forward for comparison."""
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    events = []
    last_fired = -10 ** 9

    for i in range(lookback * 2 + 1, len(df)):
        band = atr_values[i]
        if np.isnan(band) or band <= 0 or i - last_fired < cooldown:
            continue
        bar_range = high[i] - low[i]
        if bar_range <= 0 or bar_range < band:
            continue
        position = (close[i] - low[i]) / bar_range
        if position <= pctile:
            events.append((i, -1))
            last_fired = i
        elif position >= 1 - pctile:
            events.append((i, 1))
            last_fired = i
    return events


BUILDERS = {
    "decisive_break": events_decisive_break,
    "weak_close": events_weak_close,
}


def discover(data_dir, timeframe):
    pattern = re.compile(rf"^(.+)_{re.escape(timeframe)}\.csv$")
    return [
        (pattern.match(path.name).group(1), path)
        for path in sorted(Path(data_dir).glob(f"*_{timeframe}.csv"))
        if pattern.match(path.name)
    ]


def run(frames, builder_name, horizon_bars, k, seed=7):
    """Pooled edge across all instruments at one horizon."""
    builder = BUILDERS[builder_name]
    wins = events = c_wins = c_events = 0

    for df in frames.values():
        atr_values = atr(df).to_numpy()
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()

        evs = builder(df, atr_values)
        if len(evs) < 30:
            continue

        w, t = _rate(evs, high, low, close, atr_values, horizon_bars, k)
        directions = [d for _, d in evs]
        ctl = _control_events(len(df), atr_values, directions, horizon_bars, seed)
        cw, ct = _rate(ctl, high, low, close, atr_values, horizon_bars, k)

        wins += w
        events += t
        c_wins += cw
        c_events += ct

    if events < 100 or c_events < 100:
        return None

    rate = wins / events
    c_rate = c_wins / c_events
    std_error = float(np.sqrt(
        rate * (1 - rate) / events + c_rate * (1 - c_rate) / c_events
    ))
    edge = rate - c_rate
    return {
        "edge": edge,
        "err": std_error,
        "sigma": edge / std_error if std_error else 0.0,
        "events": events,
    }


def main():
    parser = argparse.ArgumentParser(description="Sweep holding horizon in calendar time.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--timeframe", default="M15")
    parser.add_argument("--reaction-atr", type=float, default=1.0)
    parser.add_argument("--output", default="reports/horizon_sweep.csv")
    args = parser.parse_args()

    found = discover(args.data_dir, args.timeframe)
    if not found:
        raise SystemExit(f"No *_{args.timeframe}.csv files in {args.data_dir}/.")

    minutes = TF_MINUTES.get(args.timeframe)
    if not minutes:
        raise SystemExit(f"Unknown timeframe {args.timeframe}")

    frames = {}
    for symbol, path in found:
        df = pd.read_csv(path)
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time").sort_index()
        if len(df) >= 1000:
            frames[symbol] = df

    print(f"\n=== Horizon sweep -- {args.timeframe}, {len(frames)} instruments, "
          f"{args.reaction_atr} ATR barriers ===")
    print("Same test at matched CALENDAR horizons. If the M15 and H4 tables")
    print("agree at equal hours, bar size was never the variable.\n")

    rows = []
    for builder_name in BUILDERS:
        print(f"--- {builder_name} ---")
        print(f"{'hours':>7}{'bars':>7}{'edge':>10}{'sigma':>8}{'events':>9}")
        for hours in TARGET_HOURS:
            horizon_bars = int(round(hours * 60 / minutes))
            if horizon_bars < 2 or horizon_bars > 2000:
                continue
            result = run(frames, builder_name, horizon_bars, args.reaction_atr)
            if not result:
                continue
            print(f"{hours:>7}{horizon_bars:>7}{result['edge']:>+10.4f}"
                  f"{result['sigma']:>+8.1f}{result['events']:>9}")
            rows.append({
                "timeframe": args.timeframe,
                "test": builder_name,
                "hours": hours,
                "bars": horizon_bars,
                **result,
            })
        print()

    if rows:
        table = pd.DataFrame(rows)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        # Append so the M15 and H4 runs land in one file for comparison.
        if output.exists():
            table = pd.concat([pd.read_csv(output), table], ignore_index=True)
            table = table.drop_duplicates(subset=["timeframe", "test", "hours"], keep="last")
        table.to_csv(output, index=False)
        print(f"-> {output}")
        print("\nRun the other timeframe, then compare rows at the same 'hours'.")


if __name__ == "__main__":
    main()
