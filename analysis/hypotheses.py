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

def causal_levels(df, lookback=5):
    """
    For every bar, the most recent swing high and swing low KNOWN AT THAT BAR.

    This replaces the original event builders, which bounded their scan with
    `nxt = swings[idx][0]` -- the bar index of the *next* swing point. At the
    moment of a candidate event you cannot know where the next swing will form,
    so that window was built from the future. Every event the old code emitted
    was pre-filtered by information the trader would not have had.

    A fractal at bar b needs `lookback` bars on each side, so it only becomes
    known at bar b + lookback. That delay is respected here: last_high[i] is the
    price of the most recent swing high confirmed at or before bar i.

    Returns (last_high, last_low) as float arrays, NaN until the first
    confirmation.
    """
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    n = len(df)

    last_high = np.full(n, np.nan)
    last_low = np.full(n, np.nan)

    current_high = np.nan
    current_low = np.nan

    for i in range(n):
        # A fractal centred at bar (i - lookback) becomes confirmable now.
        centre = i - lookback
        if centre >= lookback:
            window_high = high[centre - lookback:centre + lookback + 1]
            window_low = low[centre - lookback:centre + lookback + 1]
            if high[centre] == window_high.max() and window_high.argmax() == lookback:
                current_high = high[centre]
            elif low[centre] == window_low.min() and window_low.argmin() == lookback:
                current_low = low[centre]

        last_high[i] = current_high
        last_low[i] = current_low

    return last_high, last_low


def _emit(events, i, direction, cooldown, last_fired):
    """One event per level-interaction, not one per bar while it persists."""
    if i - last_fired[0] < cooldown:
        return
    events.append((i, direction))
    last_fired[0] = i


def events_bos(df, atr_values, lookback=5, buffer_atr=0.25, cooldown=20):
    """
    Break of structure: a close beyond the most recent KNOWN swing extreme.
    Direction = the break direction. Claim: structure breaks continue.
    """
    close = df["close"].to_numpy()
    last_high, last_low = causal_levels(df, lookback)
    events = []
    last_fired = [-10**9]

    for i in range(lookback * 2 + 1, len(df)):
        band = atr_values[i]
        if np.isnan(band) or band <= 0:
            continue
        if not np.isnan(last_high[i]) and close[i] > last_high[i] + band * buffer_atr:
            _emit(events, i, 1, cooldown, last_fired)
        elif not np.isnan(last_low[i]) and close[i] < last_low[i] - band * buffer_atr:
            _emit(events, i, -1, cooldown, last_fired)
    return events


def events_sweep(df, atr_values, lookback=5, buffer_atr=0.1, cooldown=20):
    """
    Liquidity sweep: the wick pushes beyond a known swing extreme but the close
    comes back inside. Direction = AGAINST the sweep, i.e. the reversal bet.

    A negative result here means the sweep CONTINUES rather than reverses.
    """
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    last_high, last_low = causal_levels(df, lookback)
    events = []
    last_fired = [-10**9]

    for i in range(lookback * 2 + 1, len(df)):
        band = atr_values[i]
        if np.isnan(band) or band <= 0:
            continue
        level_high = last_high[i]
        level_low = last_low[i]
        if not np.isnan(level_high) and high[i] > level_high + band * buffer_atr and close[i] < level_high:
            _emit(events, i, -1, cooldown, last_fired)
        elif not np.isnan(level_low) and low[i] < level_low - band * buffer_atr and close[i] > level_low:
            _emit(events, i, 1, cooldown, last_fired)
    return events


def events_clean_break(df, atr_values, lookback=5, buffer_atr=0.1, cooldown=20):
    """
    CONTROL 1 for the sweep result.

    Same level interaction, opposite resolution: the wick goes beyond the level
    AND the close stays beyond it. Direction is set to match the sweep test's
    continuation side, so the two are directly comparable.

    If clean breaks continue just as strongly as swept ones, then closing back
    inside the level -- the entire "liquidity sweep" signature -- adds nothing,
    and what we measured is momentum after a new extreme, not a stop hunt.
    """
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    last_high, last_low = causal_levels(df, lookback)
    events = []
    last_fired = [-10**9]

    for i in range(lookback * 2 + 1, len(df)):
        band = atr_values[i]
        if np.isnan(band) or band <= 0:
            continue
        level_high = last_high[i]
        level_low = last_low[i]
        if not np.isnan(level_high) and high[i] > level_high + band * buffer_atr and close[i] >= level_high:
            _emit(events, i, -1, cooldown, last_fired)
        elif not np.isnan(level_low) and low[i] < level_low - band * buffer_atr and close[i] <= level_low:
            _emit(events, i, 1, cooldown, last_fired)
    return events


def events_low_close(df, atr_values, lookback=5, cooldown=20, pctile=0.25):
    """
    CONTROL 2 for the sweep result.

    No levels at all. Just bars that closed in the bottom (or top) quarter of
    their own range, with a range wider than average. Direction matches the
    sweep test's convention.

    A high sweep is, mechanically, a wide bar that closed well below its high.
    If ordinary bars with that same shape continue at the same rate, then the
    swing level is irrelevant and we have rediscovered short-horizon drift
    after a weak close -- a real effect perhaps, but not a structural one, and
    not what SMC claims.
    """
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    events = []
    last_fired = [-10**9]

    for i in range(lookback * 2 + 1, len(df)):
        band = atr_values[i]
        if np.isnan(band) or band <= 0:
            continue
        bar_range = high[i] - low[i]
        if bar_range <= 0 or bar_range < band:
            continue
        position = (close[i] - low[i]) / bar_range
        if position <= pctile:
            _emit(events, i, -1, cooldown, last_fired)
        elif position >= 1 - pctile:
            _emit(events, i, 1, cooldown, last_fired)
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


def _zone_events(df, atr_values, zones, cooldown=20, shift=0.0):
    """
    Given causally-discovered zones (lo, hi, direction, valid_from), emit an
    event the first time price re-enters each zone after it formed.

    `shift` displaces every zone by that many ATR, which is the control: if
    price reacts the same way at a zone moved somewhere meaningless, the zone
    was never the reason.
    """
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    n = len(df)
    events = []
    last_fired = -10 ** 9

    for lo, hi, direction, valid_from in zones:
        median_band = np.nanmedian(atr_values)
        offset = shift * median_band
        lo_z, hi_z = lo + offset, hi + offset

        for i in range(valid_from, min(valid_from + 500, n)):
            band = atr_values[i]
            if np.isnan(band) or band <= 0 or i - last_fired < cooldown:
                continue
            if low[i] <= hi_z and high[i] >= lo_z:
                events.append((i, direction))
                last_fired = i
                break
    return events


def _find_order_blocks(df, atr_values, lookback=5, impulse_atr=1.0):
    """
    Order block: the last opposite-colour candle before an impulsive move that
    breaks the most recent known swing.

    Bullish OB = last down-candle before an up-impulse. The claim is that price
    returning to it finds buyers. Zone = that candle's high-low range.
    """
    o = df["open"].to_numpy(); h = df["high"].to_numpy()
    l = df["low"].to_numpy(); c = df["close"].to_numpy()
    last_high, last_low = causal_levels(df, lookback)
    zones = []

    for i in range(lookback * 2 + 2, len(df)):
        band = atr_values[i]
        if np.isnan(band) or band <= 0:
            continue
        move = c[i] - o[i]
        if abs(move) < band * impulse_atr:
            continue
        if move > 0 and not np.isnan(last_high[i]) and c[i] > last_high[i]:
            for j in range(i - 1, max(i - 10, 0), -1):
                if c[j] < o[j]:
                    zones.append((l[j], h[j], 1, i + 1))
                    break
        elif move < 0 and not np.isnan(last_low[i]) and c[i] < last_low[i]:
            for j in range(i - 1, max(i - 10, 0), -1):
                if c[j] > o[j]:
                    zones.append((l[j], h[j], -1, i + 1))
                    break
    return zones


def _find_fvgs(df, atr_values, min_atr=0.25):
    """
    Fair value gap: a three-bar pattern where bar 1 and bar 3 do not overlap.
    Bullish gap = bar1.high < bar3.low. The claim is that price returning to
    fill the gap resumes in the gap's direction.
    """
    h = df["high"].to_numpy(); l = df["low"].to_numpy()
    zones = []
    for i in range(2, len(df)):
        band = atr_values[i]
        if np.isnan(band) or band <= 0:
            continue
        if l[i] > h[i - 2] and (l[i] - h[i - 2]) >= band * min_atr:
            zones.append((h[i - 2], l[i], 1, i + 1))
        elif h[i] < l[i - 2] and (l[i - 2] - h[i]) >= band * min_atr:
            zones.append((h[i], l[i - 2], -1, i + 1))
    return zones


def events_order_block(df, atr_values, **kw):
    """Component 4a. Claim: price reverses off order blocks."""
    return _zone_events(df, atr_values, _find_order_blocks(df, atr_values))


def events_order_block_shifted(df, atr_values, **kw):
    """CONTROL: same order blocks, displaced 1.5 ATR to meaningless prices."""
    return _zone_events(df, atr_values, _find_order_blocks(df, atr_values), shift=1.5)


def events_fvg(df, atr_values, **kw):
    """Component 4b. Claim: price resumes after filling a fair value gap."""
    return _zone_events(df, atr_values, _find_fvgs(df, atr_values))


def events_fvg_shifted(df, atr_values, **kw):
    """CONTROL: same gaps, displaced 1.5 ATR."""
    return _zone_events(df, atr_values, _find_fvgs(df, atr_values), shift=1.5)


HYPOTHESES = {
    "order_block": (events_order_block, "price reverses off order blocks"),
    "ctl_ob_shifted": (events_order_block_shifted, "CONTROL: order blocks moved 1.5 ATR"),
    "fvg_fill": (events_fvg, "price resumes after filling a fair value gap"),
    "ctl_fvg_shifted": (events_fvg_shifted, "CONTROL: gaps moved 1.5 ATR"),
    "bos_continuation": (events_bos, "structure breaks continue"),
    "sweep_reversal": (events_sweep, "swept levels reverse"),
    "ctl_clean_break": (events_clean_break, "CONTROL: clean breaks, same direction as sweep test"),
    "ctl_low_close": (events_low_close, "CONTROL: weak-close bars, no level involved"),
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
