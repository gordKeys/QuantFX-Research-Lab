"""
Monte Carlo null -- is the backtest return real, or does the rule produce it on
data with no predictability?

WHY THIS EXISTS

The sequential backtest returned 21 of 21 instruments profitable out-of-sample,
median +24% (H4) and +30% (M15), win rates of 57-73%. Before celebrating I ran
the identical pipeline on a pure synthetic random walk -- a series constructed
to contain no predictable structure at all.

    PURE NOISE #1 (out-of-sample)   270 trades   67.8% win
    PURE NOISE #2 (out-of-sample)   255 trades   64.3% win
    PURE NOISE #3 (out-of-sample)   280 trades   67.9% win

Those win rates are indistinguishable from the real ones. So the 60-70% win
rate is a GEOMETRIC PROPERTY OF THE ENTRY RULE, not evidence about markets. It
comes from the same family of bug as the original support/resistance error:
the rule selects an entry bar using one price (the bar's low reaching into a
zone) and then measures the outcome from another (the close), and the direction
of the bet is derived from the same recent path that triggered the selection.
Any such rule beats a direction-agnostic control on any series, including noise.

The barrier test agrees: on a synthetic random walk the same signal scored
+0.22, LARGER than the +0.10 it scored on real EURUSD. The real market is
slightly LESS exploitable by this rule than pure noise is.

WHAT IS STILL OPEN

Returns on noise came out near zero (+2.8%, -6.3%, +3.8%) while real data gave
+24-30%. That gap is not yet explained, and it is not safe to assume either
way, because my synthetic series lacked volatility clustering and used a flat
spread -- both of which affect the result.

So this script builds a PROPERLY MATCHED null. For each instrument it block-
bootstraps that instrument's own bars: blocks of consecutive bars are resampled
and chained, which preserves bar geometry, the return distribution, and short-
range volatility clustering, while destroying any longer-range predictability.
The identical signal and backtest then run on each synthetic series.

If the real return sits inside the synthetic distribution, the strategy is the
artifact and this project is finished. If it sits clearly outside on most
instruments, something real survives and is worth forward-testing.

That is the whole question, and this answers it.

    python run_project.py montecarlo --timeframe H4 --barrier 3.0 --sims 30

Runtime warning: this is 21 instruments x `sims` full backtests. Expect several
minutes on H4 and considerably longer on M15. Start with H4.
"""

from bootstrap import add_project_root

add_project_root()

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.hypotheses import _find_fvgs, _zone_events
from analysis.structure import atr

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "simulate_signal", str(Path(__file__).parent / "simulate_signal.py")
)
_sim = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sim)


def block_bootstrap(df, block=50, seed=0):
    """
    Resample the instrument's own bars in contiguous blocks and chain them.

    Each bar is stored as its open/high/low/close expressed as multiplicative
    offsets from the previous close. Resampling blocks of those offsets and
    re-accumulating produces a series with:

      - the same bar shapes (so the entry rule's geometry is unchanged)
      - the same return distribution
      - volatility clustering preserved WITHIN blocks

    but no predictable relationship ACROSS blocks. That is exactly the null we
    want: everything about the data held constant except the thing the strategy
    claims to exploit.
    """
    rng = np.random.default_rng(seed)
    close = df["close"].to_numpy()
    prev = np.concatenate([[close[0]], close[:-1]])

    rel = np.column_stack([
        df["open"].to_numpy() / prev,
        df["high"].to_numpy() / prev,
        df["low"].to_numpy() / prev,
        close / prev,
    ])
    rel = rel[1:]  # first bar has no predecessor

    n = len(rel)
    n_blocks = n // block + 1
    starts = rng.integers(0, max(n - block, 1), size=n_blocks)
    picked = np.concatenate([rel[s:s + block] for s in starts])[:n]

    out = np.empty((n, 4))
    level = close[0]
    for i in range(n):
        out[i] = picked[i] * level
        level = out[i, 3]

    synthetic = pd.DataFrame({
        "open": out[:, 0], "high": out[:, 1], "low": out[:, 2], "close": out[:, 3],
        "spread": df["spread"].to_numpy()[1:n + 1],
    }, index=df.index[1:n + 1])

    # a resampled bar can end up with close outside high/low after chaining
    synthetic["high"] = synthetic[["open", "high", "low", "close"]].max(axis=1)
    synthetic["low"] = synthetic[["open", "high", "low", "close"]].min(axis=1)
    return synthetic


def run_once(df, spec, barrier, risk, equity):
    """Out-of-sample return percentage for one series."""
    atr_values = atr(df).to_numpy()
    events = _zone_events(df, atr_values, _find_fvgs(df, atr_values))
    if len(events) < 30:
        return None
    split = int(len(df) * 0.7)
    trades = _sim.simulate(df, spec, events, barrier, risk, equity, split)
    if not trades:
        return None
    out = [t for t in trades if not t["in_sample"]]
    if len(out) < 10:
        return None
    return sum(t["pnl"] for t in out) / equity * 100


def main():
    parser = argparse.ArgumentParser(description="Monte Carlo null for the backtest.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--timeframe", default="H4")
    parser.add_argument("--barrier", type=float, default=3.0)
    parser.add_argument("--risk", type=float, default=0.005)
    parser.add_argument("--equity", type=float, default=10000.0)
    parser.add_argument("--sims", type=int, default=30)
    parser.add_argument("--block", type=int, default=50)
    parser.add_argument("--output", default="reports/montecarlo.csv")
    args = parser.parse_args()

    pattern = re.compile(rf"^(.+)_{re.escape(args.timeframe)}\.csv$")
    found = [
        (pattern.match(p.name).group(1), p)
        for p in sorted(Path(args.data_dir).glob(f"*_{args.timeframe}.csv"))
        if pattern.match(p.name)
    ]
    if not found:
        raise SystemExit(f"No *_{args.timeframe}.csv in {args.data_dir}/")

    print(f"\n=== Monte Carlo null -- {args.timeframe}, {args.barrier} ATR, "
          f"{args.sims} synthetic series per instrument ===")
    print("Synthetic series are block-bootstrapped from each instrument's OWN bars:")
    print("same bar shapes, same return distribution, volatility clustering kept")
    print("within blocks, predictability across blocks destroyed.\n")
    print(f"{'symbol':10}{'real%':>9}{'null med%':>11}{'null p95%':>11}"
          f"{'beats':>8}{'p-value':>9}")

    rows = []
    for symbol, path in found:
        df = pd.read_csv(path)
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time").sort_index()
        if len(df) < 2000:
            continue

        spec = _sim.instrument_spec(args.data_dir, symbol, args.timeframe, df)
        if not spec["dollars_per_unit"]:
            print(f"{symbol:10}   skipped -- no tick value in meta")
            continue

        real = run_once(df, spec, args.barrier, args.risk, args.equity)
        if real is None:
            continue

        null = []
        for s in range(args.sims):
            synthetic = block_bootstrap(df, block=args.block, seed=s)
            result = run_once(synthetic, spec, args.barrier, args.risk, args.equity)
            if result is not None:
                null.append(result)

        if len(null) < 5:
            print(f"{symbol:10}   skipped -- null failed to produce trades")
            continue

        null = np.array(null)
        beats = int((null >= real).sum())
        # one-sided p: how often does noise match or beat the real result
        p_value = (beats + 1) / (len(null) + 1)

        print(f"{symbol:10}{real:>9.2f}{np.median(null):>11.2f}"
              f"{np.percentile(null, 95):>11.2f}{len(null)-beats:>4}/{len(null):<3}"
              f"{p_value:>9.3f}")
        rows.append({"symbol": symbol, "timeframe": args.timeframe, "real": real,
                     "null_median": float(np.median(null)),
                     "null_p95": float(np.percentile(null, 95)),
                     "sims": len(null), "p_value": p_value})

    if not rows:
        raise SystemExit("\nNothing to report.")

    table = pd.DataFrame(rows)
    significant = table[table["p_value"] < 0.05]

    print("\n" + "=" * 66)
    print(f"{len(significant)} of {len(table)} instruments beat their own noise "
          f"at p < 0.05")
    print(f"median real return {table['real'].median():+.2f}%  vs  "
          f"median null return {table['null_median'].median():+.2f}%")
    print("=" * 66)
    if len(significant) <= max(1, len(table) * 0.1):
        print("\nThat is chance. Testing 21 instruments at p<0.05 expects ~1 hit.")
        print("The strategy is the artifact. This is the end of the line for it.")
    else:
        print("\nMore instruments clear the null than chance predicts. Worth")
        print("forward-testing on demo -- but size it as an experiment, not a bet.")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
