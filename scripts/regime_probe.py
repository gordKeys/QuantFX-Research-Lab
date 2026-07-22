"""
Approach 1 -- regime edge: is there a persistent structure worth positioning for?

Every directional signal we tested failed. This asks a different question: not
"which way next" but "is this instrument in a state where SOMETHING is
structurally true right now, regardless of direction".

Two things get measured, both direction-free:

  VOL PERSISTENCE   Does high volatility predict high volatility? If the ATR
                    percentile this week predicts next week's, volatility is
                    forecastable even when price is not. That is real and it is
                    the foundation of every volatility-timing strategy: size up
                    when a persistent-move regime is likely, sit out chop.
                    Measured as the autocorrelation of ATR, and as a transition
                    matrix between calm and volatile states.

  TREND PERSISTENCE Given the instrument is ALREADY trending (by a slow slope),
                    does it keep trending? This is not "predict a reversal" --
                    it is "once a trend exists, does staying with it beat
                    fading it". A positive answer means a regime FILTER has
                    value even if no entry pattern does: trade only in the
                    trending state, and only with it.

The null throughout: a random-walk series has ZERO vol persistence beyond the
mechanical and ZERO trend persistence. Anything we measure has to beat that, so
each metric is reported against its shuffled-returns control -- same returns,
random order, which destroys any real serial structure while preserving the
distribution.

    python run_project.py regime_probe --timeframe H4
    python run_project.py regime_probe --timeframe D1
"""

from bootstrap import add_project_root

add_project_root()

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def atr(df, period=14):
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def vol_persistence(df, lag=1):
    """
    Autocorrelation of ATR at `lag`. High positive value = volatility clusters,
    i.e. it is forecastable. Compared against the same series with returns
    shuffled, which sets the honest zero.
    """
    a = atr(df).dropna()
    if len(a) < 200:
        return None
    real = a.autocorr(lag=lag)

    # shuffled control: rebuild a price path from shuffled returns, recompute ATR
    rng = np.random.default_rng(3)
    rets = df["close"].pct_change().dropna().to_numpy().copy()
    rng.shuffle(rets)
    synth_close = df["close"].iloc[0] * np.cumprod(1 + rets)
    synth = pd.DataFrame({
        "high": synth_close * (1 + np.abs(rets) / 2),
        "low": synth_close * (1 - np.abs(rets) / 2),
        "close": synth_close,
    })
    shuffled = atr(synth).dropna().autocorr(lag=lag)
    return real, shuffled


def regime_transition(df, calm_pct=33, vol_pct=67):
    """
    Two-state transition probabilities on ATR percentile.

    P(vol|vol) well above the base rate of volatile bars means: once you are in
    a high-vol regime you tend to stay, so a vol-timing overlay has grip.
    """
    a = atr(df).dropna()
    if len(a) < 300:
        return None
    lo = np.percentile(a, calm_pct)
    hi = np.percentile(a, vol_pct)
    state = np.where(a >= hi, 1, np.where(a <= lo, 0, -1))  # -1 = middle, ignored

    stay_vol = trans_vol = stay_calm = trans_calm = 0
    for i in range(len(state) - 1):
        if state[i] == 1:
            stay_vol += int(state[i + 1] == 1)
            trans_vol += 1
        elif state[i] == 0:
            stay_calm += int(state[i + 1] == 0)
            trans_calm += 1

    base_vol = (state == 1).mean()
    return {
        "p_vol_given_vol": stay_vol / trans_vol if trans_vol else np.nan,
        "base_vol_rate": base_vol,
        "vol_stickiness": (stay_vol / trans_vol - base_vol) if trans_vol else np.nan,
    }


def trend_persistence(df, slope_window=20, horizon=10):
    """
    When the slow slope is up, does the next `horizon` return tend to be up too
    (and symmetrically for down)? Reported as the share of cases where the
    forward move CONTINUED the existing slope, vs a coin-flip 0.50.

    This measures whether a trend FILTER has value, independent of any entry.
    """
    close = df["close"].to_numpy()
    n = len(close)
    if n < slope_window + horizon + 50:
        return None

    ema = pd.Series(close).ewm(span=slope_window, adjust=False).mean().to_numpy()
    cont = 0
    total = 0
    for i in range(slope_window, n - horizon):
        slope = ema[i] - ema[i - slope_window]
        if slope == 0:
            continue
        fwd = close[i + horizon] - close[i]
        total += 1
        cont += int((slope > 0) == (fwd > 0))
    return cont / total if total else None


def discover(data_dir, timeframe):
    pattern = re.compile(rf"^(.+)_{re.escape(timeframe)}\.csv$")
    return [(pattern.match(p.name).group(1), p)
            for p in sorted(Path(data_dir).glob(f"*_{timeframe}.csv"))
            if pattern.match(p.name)]


def main():
    parser = argparse.ArgumentParser(description="Probe for direction-free regime edges.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--timeframe", default="H4")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--output", default="reports/regime_probe.csv")
    args = parser.parse_args()

    found = discover(args.data_dir, args.timeframe)
    if not found:
        raise SystemExit(f"No *_{args.timeframe}.csv in {args.data_dir}/")

    print(f"\n=== Regime probe -- {args.timeframe} ===")
    print("Direction-free. Each metric vs its shuffled-returns null.\n")
    print(f"{'symbol':10}{'volAC':>8}{'volAC_null':>11}{'vol_stick':>11}"
          f"{'trend_persist':>14}")

    rows = []
    for symbol, path in found:
        df = pd.read_csv(path); df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time").sort_index()
        if len(df) < 500:
            continue

        vp = vol_persistence(df)
        rt = regime_transition(df)
        tp = trend_persistence(df, horizon=args.horizon)
        if vp is None or rt is None or tp is None:
            continue

        real_ac, null_ac = vp
        rows.append({
            "symbol": symbol,
            "vol_autocorr": real_ac,
            "vol_autocorr_null": null_ac,
            "vol_stickiness": rt["vol_stickiness"],
            "trend_persist": tp,
        })
        print(f"{symbol:10}{real_ac:>8.3f}{null_ac:>11.3f}{rt['vol_stickiness']:>+11.3f}"
              f"{tp:>14.3f}")

    if not rows:
        raise SystemExit("No instrument had enough data.")

    table = pd.DataFrame(rows)
    print("\n" + "=" * 60)
    print(f"median vol autocorrelation : {table['vol_autocorr'].median():.3f} "
          f"(null {table['vol_autocorr_null'].median():.3f})")
    print(f"median vol stickiness      : {table['vol_stickiness'].median():+.3f} "
          f"(0 = no regime persistence)")
    print(f"median trend persistence   : {table['trend_persist'].median():.3f} "
          f"(0.50 = coin flip)")
    print("=" * 60)

    strong_vol = table[table["vol_autocorr"] - table["vol_autocorr_null"] > 0.2]
    strong_trend = table[table["trend_persist"] > 0.55]
    print(f"\nStrong vol persistence (beats null by >0.2): "
          f"{', '.join(strong_vol['symbol']) or 'none'}")
    print(f"Trend-persistent (>0.55 continuation): "
          f"{', '.join(strong_trend['symbol']) or 'none'}")
    print("\nvol persistence beating null -> a volatility-timing overlay has")
    print("something real to time. trend persistence >0.55 -> a trend-state")
    print("filter has value even though no ENTRY pattern did. These are overlays,")
    print("not entries: they say WHEN to be active, not WHICH way to bet.")

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
