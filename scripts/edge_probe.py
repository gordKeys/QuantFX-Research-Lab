"""
New-input edge probes -- break the coin flip by using data the price chart
does not contain.

Every signal this project tested came from price alone (candles, levels,
z-scores), and all failed the same way: the entry could not predict direction
better than a coin flip after costs. Price on liquid instruments is the single
most heavily mined input in finance. The way out is not a seventh price pattern;
it is a DIFFERENT INPUT. Three that are real and retail-accessible, each probed
here and scored the same way so they can be compared head to head:

  1. LEAD-LAG   Does instrument B's return predict instrument A's NEXT return?
                If DXY or a yield proxy moves first and EURUSD follows a bar
                later, that lead is information not inside the EURUSD chart. We
                measure lagged cross-correlation and, more importantly, whether
                acting on it would have beaten a coin flip out-of-sample.

  2. EVENT/TIME Is there a fixed CLOCK structure -- a session open, a specific
                hour -- where the next move is biased or unusually large? The
                edge here is timing, known in advance, not a direction guess.
                Guarded with a t-stat so one big day can't fake an hourly edge.

  3. ORDER-FLOW Do tick_volume and spread dynamics (already in every CSV, never
                used by any strategy we built) predict the next return? A volume
                surge with a widening spread is a microstructure fingerprint of
                aggressive flow. Proxy, not true order flow, but it is a
                non-price input the charts carry for free.

Each probe reports a single comparable number: predictive_edge = the
out-of-sample hit rate of acting on the signal, minus 0.50. Positive and
outside noise = worth pursuing. The winner (or a combination) is what we build
next.

    python run_project.py edge_probe --timeframe M15
    python run_project.py edge_probe --timeframe H1 --anchor DXY
"""

from bootstrap import add_project_root

add_project_root()

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def load(data_dir, symbol, timeframe):
    path = Path(data_dir) / f"{symbol}_{timeframe}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time").sort_index()


def discover(data_dir, timeframe):
    pattern = re.compile(rf"^(.+)_{re.escape(timeframe)}\.csv$")
    return [pattern.match(p.name).group(1)
            for p in sorted(Path(data_dir).glob(f"*_{timeframe}.csv"))
            if pattern.match(p.name)]


def oos_hit_rate(signal, forward_ret, split=0.6):
    """
    Fit the sign of the signal->forward relationship on the first `split` of the
    data, then measure the hit rate of that rule on the held-out remainder.
    Fitting the sign in-sample and scoring out-of-sample is what stops us from
    grading a rule on the data that chose it.
    """
    n = len(signal)
    cut = int(n * split)
    if cut < 100 or n - cut < 100:
        return None, 0

    s_in, f_in = signal[:cut], forward_ret[:cut]
    valid_in = ~(np.isnan(s_in) | np.isnan(f_in))
    if valid_in.sum() < 50:
        return None, 0
    sign = np.sign(np.nanmean(np.sign(s_in[valid_in]) * f_in[valid_in])) or 1

    s_out, f_out = signal[cut:], forward_ret[cut:]
    valid_out = ~(np.isnan(s_out) | np.isnan(f_out)) & (s_out != 0)
    if valid_out.sum() < 50:
        return None, 0
    predicted = np.sign(s_out[valid_out]) * sign
    actual = np.sign(f_out[valid_out])
    hits = (predicted == actual).mean()
    return hits, int(valid_out.sum())


# --------------------------------------------------------------- 1. lead-lag

def probe_lead_lag(frames, target, anchors, max_lag=3):
    """Best out-of-sample hit rate from any anchor leading the target, lag 1..max_lag."""
    tgt = frames[target]
    tgt_ret = tgt["close"].pct_change().to_numpy()
    fwd = pd.Series(tgt_ret).shift(-1).to_numpy()  # next-bar return

    best = None
    for anchor in anchors:
        if anchor == target or anchor not in frames:
            continue
        a = frames[anchor]
        joined = tgt[["close"]].join(a[["close"]], rsuffix="_a", how="inner")
        if len(joined) < 500:
            continue
        a_ret = joined["close_a"].pct_change().to_numpy()
        t_ret = joined["close"].pct_change().to_numpy()
        t_fwd = pd.Series(t_ret).shift(-1).to_numpy()
        for lag in range(1, max_lag + 1):
            sig = pd.Series(a_ret).shift(lag - 1).to_numpy()  # anchor return leads
            hit, n = oos_hit_rate(sig, t_fwd)
            if hit is not None and (best is None or hit > best["hit"]):
                best = {"anchor": anchor, "lag": lag, "hit": hit, "n": n}
    return best


# ------------------------------------------------------------- 2. event/time

def probe_event_time(df):
    """Best out-of-sample hour-of-day directional bias."""
    df = df.copy()
    df["ret"] = df["close"].pct_change()
    df["fwd"] = df["ret"].shift(-1)
    df["hour"] = df.index.hour

    best = None
    for hour, block in df.groupby("hour"):
        if len(block) < 300:
            continue
        # direction rule: use the in-sample mean sign of this hour's forward return
        fwd = block["fwd"].to_numpy()
        cut = int(len(fwd) * 0.6)
        if cut < 100 or len(fwd) - cut < 100:
            continue
        in_sign = np.sign(np.nanmean(fwd[:cut])) or 1
        out = fwd[cut:]
        out = out[~np.isnan(out)]
        if len(out) < 100:
            continue
        hit = (np.sign(out) == in_sign).mean()
        if best is None or abs(hit - 0.5) > abs(best["hit"] - 0.5):
            best = {"hour": int(hour), "hit": hit, "n": len(out)}
    return best


# ------------------------------------------------------------- 3. order-flow

def probe_order_flow(df):
    """
    Do tick_volume and spread dynamics predict the next return?
    Signal = volume z-score times the sign of the current bar (aggression proxy),
    optionally gated by a widening spread. Scored out-of-sample like the rest.
    """
    if "tick_volume" not in df.columns:
        return None
    df = df.copy()
    ret = df["close"].pct_change()
    fwd = ret.shift(-1).to_numpy()

    vol = df["tick_volume"].astype(float)
    vol_z = ((vol - vol.rolling(50).mean()) / vol.rolling(50).std()).to_numpy()
    bar_sign = np.sign((df["close"] - df["open"]).to_numpy())

    # aggression proxy: strong-volume bar in its own direction
    sig_continuation = vol_z * bar_sign
    sig_reversion = -vol_z * bar_sign

    best = None
    for name, sig in [("flow_continuation", sig_continuation),
                      ("flow_reversion", sig_reversion)]:
        hit, n = oos_hit_rate(sig, fwd)
        if hit is not None and (best is None or abs(hit - 0.5) > abs(best["hit"] - 0.5)):
            best = {"variant": name, "hit": hit, "n": n}
    return best


def main():
    parser = argparse.ArgumentParser(description="Probe three non-price edge sources.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--timeframe", default="M15")
    parser.add_argument("--targets", nargs="+",
                        default=["EURUSD", "GBPUSD", "XAUUSD", "USDJPY"])
    parser.add_argument("--anchors", nargs="+",
                        default=["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "XAUUSD",
                                 "US500", "US30", "NAS100"],
                        help="candidate leading instruments for lead-lag")
    parser.add_argument("--output", default="reports/edge_probe.csv")
    args = parser.parse_args()

    available = discover(args.data_dir, args.timeframe)
    if not available:
        raise SystemExit(f"No *_{args.timeframe}.csv in {args.data_dir}/")

    frames = {}
    for sym in set(args.targets) | set(args.anchors):
        df = load(args.data_dir, sym, args.timeframe)
        if df is not None and len(df) >= 500:
            frames[sym] = df

    print(f"\n=== Edge probe -- {args.timeframe} ===")
    print("Out-of-sample hit rate of acting on each signal, minus 0.50.")
    print("Positive and outside ~0.02 noise = a real non-price edge.\n")
    print(f"{'target':9}{'lead-lag':>26}{'event-time':>20}{'order-flow':>24}")

    rows = []
    for target in args.targets:
        if target not in frames:
            continue
        ll = probe_lead_lag(frames, target, args.anchors)
        et = probe_event_time(frames[target])
        of = probe_order_flow(frames[target])

        ll_s = f"{ll['anchor']}@lag{ll['lag']} {ll['hit']-0.5:+.3f}" if ll else "n/a"
        et_s = f"h{et['hour']:02d} {et['hit']-0.5:+.3f}" if et else "n/a"
        of_s = f"{of['variant'].split('_')[1]} {of['hit']-0.5:+.3f}" if of else "n/a"

        print(f"{target:9}{ll_s:>26}{et_s:>20}{of_s:>24}")
        rows.append({
            "target": target,
            "leadlag_edge": (ll["hit"] - 0.5) if ll else np.nan,
            "leadlag_anchor": ll["anchor"] if ll else None,
            "leadlag_lag": ll["lag"] if ll else None,
            "event_edge": (et["hit"] - 0.5) if et else np.nan,
            "event_hour": et["hour"] if et else None,
            "flow_edge": (of["hit"] - 0.5) if of else np.nan,
            "flow_variant": of["variant"] if of else None,
        })

    if not rows:
        raise SystemExit("No target had data.")

    table = pd.DataFrame(rows)
    print("\n" + "=" * 66)
    for col, label in [("leadlag_edge", "lead-lag"), ("event_edge", "event-time"),
                       ("flow_edge", "order-flow")]:
        med = table[col].median()
        mx = table[col].max()
        print(f"  {label:12} median edge {med:+.3f}   best {mx:+.3f}")
    print("=" * 66)
    print("\nAn edge above ~+0.03 that holds across MULTIPLE targets is worth")
    print("building. A lone spike on one target is likely noise -- re-run on")
    print("another timeframe to check it persists before trusting it. The probes")
    print("are comparable, so the largest consistent one (or a combination) is")
    print("where the next real strategy comes from.")

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
