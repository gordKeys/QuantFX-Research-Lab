"""
Structure hypothesis battery -- steps 2, 3 and 4 tested in one pass.

Why one script instead of three rounds:

The S/R result came back flat. If we now test break-of-structure, then sweeps,
then AMD, each in its own session, with the threshold chosen after seeing the
numbers, we will eventually find something that looks significant. Across 21
instruments and 2 timeframes that is 42 tests per hypothesis; at a 2-sigma
threshold roughly 2 of them come up positive by chance every single time. That
is exactly how the S/R report produced EURJPY on M15 and GER40 on H4 -- two
"winners" that reversed sign on the other timeframe.

So the rules are fixed here, in the code, before the run:

  THRESHOLD    3.0 standard errors, not 2. With ~40 tests per hypothesis,
               2 sigma expects ~2 false positives. 3 sigma expects ~0.1.

  CONSISTENCY  An instrument must clear the threshold on BOTH M15 and H4 with
               the same sign. A real structural effect does not appear on H4
               and reverse on M15.

  POOLED       The across-instrument pooled effect is the primary result. It
               has ~20x the sample of any single instrument, so if an effect
               is real and general it shows up there first and cleanest.

  DIRECTION    Significant results in BOTH directions across instruments are
               the signature of noise, not of an effect. Reported explicitly.

Every hypothesis reduces to the same measurement, so nothing differs between
them except which bars are events and which way they point:

    from the event bar, in the event's direction, does price travel
    +k*ATR before it travels -k*ATR, within the horizon?

Control is the identical test on random bars with a matched direction mix,
which gives that instrument's natural persistence rate. For a driftless series
that sits at 0.50; the control measures where it actually sits.
"""

import numpy as np
import pandas as pd

from analysis.structure import atr, find_swings

THRESHOLD_SIGMA = 3.0


def first_passage(high, low, entry, band, direction, i, horizon, k):
    """
    True if price moved k*ATR in `direction` before k*ATR against it.
    None if neither happened inside the horizon.

    Barrier order is resolved bar by bar rather than by comparing window
    extremes, because a window that contains both barriers tells you nothing
    about which was hit first -- and getting that backwards is worth several
    points of fake edge.
    """
    target = entry + direction * band * k
    stop = entry - direction * band * k

    for j in range(i + 1, min(i + 1 + horizon, len(high))):
        hit_target = high[j] >= target if direction > 0 else low[j] <= target
        hit_stop = low[j] <= stop if direction > 0 else high[j] >= stop
        if hit_target and hit_stop:
            return None  # same bar, unresolvable at this resolution
        if hit_target:
            return True
        if hit_stop:
            return False
    return None


def _rate(events, high, low, close, atr_values, horizon, k):
    wins = 0
    total = 0
    for i, direction in events:
        band = atr_values[i]
        if np.isnan(band) or band <= 0:
            continue
        outcome = first_passage(high, low, close[i], band, direction, i, horizon, k)
        if outcome is None:
            continue
        total += 1
        wins += int(outcome)
    return wins, total


def _control_events(n, atr_values, directions, horizon, seed, samples=3000):
    """Random bars, direction mix matched to the real events."""
    rng = np.random.default_rng(seed)
    valid = np.where(~np.isnan(atr_values))[0]
    valid = valid[(valid > 30) & (valid < n - horizon - 1)]
    if valid.size == 0 or not directions:
        return []
    picks = rng.choice(valid, size=min(samples, valid.size), replace=False)
    mix = rng.choice(directions, size=picks.size)
    return list(zip(picks, mix))


# ---------------------------------------------------------------- hypotheses

def events_bos(df, atr_values, lookback=5, buffer_atr=0.25):
    """
    Break of structure: a close beyond the most recent swing high/low by more
    than buffer_atr. Direction = the break direction.

    Claim under test: structure breaks continue.
    """
    close = df["close"].to_numpy()
    swings = find_swings(df, lookback=lookback)
    events = []
    for idx in range(1, len(swings)):
        bar, price, kind = swings[idx - 1]
        nxt = swings[idx][0]
        for i in range(bar + lookback + 1, min(nxt + lookback, len(df))):
            band = atr_values[i]
            if np.isnan(band) or band <= 0:
                continue
            if kind == 1 and close[i] > price + band * buffer_atr:
                events.append((i, 1))
                break
            if kind == -1 and close[i] < price - band * buffer_atr:
                events.append((i, -1))
                break
    return events


def events_sweep(df, atr_values, lookback=5, buffer_atr=0.1):
    """
    Liquidity sweep: the bar's wick pushes beyond a swing extreme but the close
    comes back inside. Direction = AGAINST the sweep.

    This is a genuinely different claim from the S/R test that came back flat.
    That one asked whether price stops at a level. This asks whether price that
    pokes through a level and rejects then reverses -- the SMC stop-hunt story.
    """
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    swings = find_swings(df, lookback=lookback)
    events = []
    for idx in range(1, len(swings)):
        bar, price, kind = swings[idx - 1]
        nxt = swings[idx][0]
        for i in range(bar + lookback + 1, min(nxt + lookback, len(df))):
            band = atr_values[i]
            if np.isnan(band) or band <= 0:
                continue
            if kind == 1 and high[i] > price + band * buffer_atr and close[i] < price:
                events.append((i, -1))
                break
            if kind == -1 and low[i] < price - band * buffer_atr and close[i] > price:
                events.append((i, 1))
                break
    return events


def events_asia_break(df, atr_values, reverse=False):
    """
    AMD: build the Asian range (00:00-07:00 server), then take the first break
    of it during London (07:00-13:00).

    reverse=False tests accumulation-then-expansion: the break runs.
    reverse=True  tests the manipulation story: the break is a trap and price
                  turns back through the range.

    Both are tested, because they are opposite predictions and the AMD
    literature is often vague about which one it is claiming.
    """
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    hours = df.index.hour
    days = df.index.normalize()

    events = []
    for _, block in pd.Series(np.arange(len(df)), index=days).groupby(level=0):
        idx = block.to_numpy()
        asia = idx[(hours[idx] >= 0) & (hours[idx] < 7)]
        london = idx[(hours[idx] >= 7) & (hours[idx] < 13)]
        if asia.size < 8 or london.size < 4:
            continue
        top = high[asia].max()
        bottom = low[asia].min()
        for i in london:
            band = atr_values[i]
            if np.isnan(band) or band <= 0:
                continue
            if close[i] > top:
                events.append((i, -1 if reverse else 1))
                break
            if close[i] < bottom:
                events.append((i, 1 if reverse else -1))
                break
    return events


HYPOTHESES = {
    "bos_continuation": (events_bos, "structure breaks continue"),
    "sweep_reversal": (events_sweep, "swept levels reverse"),
    "asia_break_runs": (lambda d, a: events_asia_break(d, a, reverse=False),
                        "Asian range break persists"),
    "asia_break_traps": (lambda d, a: events_asia_break(d, a, reverse=True),
                         "Asian range break is a trap"),
}


def test_hypothesis(df, name, horizon=20, k=1.0, seed=7):
    """Run one hypothesis on one symbol/timeframe. Returns None if too few events."""
    builder, _ = HYPOTHESES[name]
    atr_values = atr(df).to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()

    events = builder(df, atr_values)
    if len(events) < 30:
        return None

    wins, total = _rate(events, high, low, close, atr_values, horizon, k)
    if total < 30:
        return None

    directions = [direction for _, direction in events]
    control = _control_events(len(df), atr_values, directions, horizon, seed)
    c_wins, c_total = _rate(control, high, low, close, atr_values, horizon, k)
    if c_total < 30:
        return None

    rate = wins / total
    c_rate = c_wins / c_total
    # SE of a difference of two independent proportions.
    std_error = float(np.sqrt(
        rate * (1 - rate) / total + c_rate * (1 - c_rate) / c_total
    ))
    edge = rate - c_rate

    return {
        "events": total,
        "rate": rate,
        "control": c_rate,
        "edge": edge,
        "err": std_error,
        "sigma": edge / std_error if std_error else 0.0,
        "wins": wins,
        "control_events": c_total,
        "control_wins": c_wins,
    }


def pool(results):
    """
    Pooled effect across instruments: sum wins and events, then compare.

    Individual instruments are underpowered -- 300 events gives a standard
    error near 0.03, so nothing under a 9-point effect is visible. Pooling 21
    instruments gives ~6000 events and resolves effects around 2 points, which
    is the range a real but modest structural edge would live in.
    """
    wins = sum(r["wins"] for r in results)
    events = sum(r["events"] for r in results)
    c_wins = sum(r["control_wins"] for r in results)
    c_events = sum(r["control_events"] for r in results)
    if not events or not c_events:
        return None

    rate = wins / events
    c_rate = c_wins / c_events
    std_error = float(np.sqrt(
        rate * (1 - rate) / events + c_rate * (1 - c_rate) / c_events
    ))
    edge = rate - c_rate
    return {
        "events": events,
        "rate": rate,
        "control": c_rate,
        "edge": edge,
        "err": std_error,
        "sigma": edge / std_error if std_error else 0.0,
    }
