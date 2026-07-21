"""
Structure primitives: swings -> levels -> measured reactions.

This is step 1 of the build order (S/R with confluence), plus the validation
gate you set for it: "once we know the instruments actually respond to
structure at all".

The important part is not the level detector. Anyone can cluster swing points.
The important part is measure_reactions(), which asks whether price behaves
differently at a detected level than at a random bar on the same instrument.

That comparison is the whole game. A level that produces a bounce 55% of the
time looks impressive until you learn that a randomly chosen price on the same
chart bounces 54% of the time -- at which point the level is decoration and any
strategy built on it is fitting noise. The last system passed 6/6 folds before
costs and 0/6 after. The lesson was not "model costs earlier", it was "compare
against a null before believing anything". So the null is built in here rather
than bolted on later.
"""

import numpy as np
import pandas as pd


def atr(df, period=14):
    """Average true range as a Series, used as the volatility yardstick."""
    prev_close = df["close"].shift(1)
    true_range = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return true_range.rolling(period).mean()


def find_swings(df, lookback=5):
    """
    Fractal swing points, alternating high/low.

    A swing high is a bar whose high is the maximum of the window spanning
    `lookback` bars either side. Runs of the same kind collapse to their
    extreme, which turns raw fractals into a clean zigzag.

    Returns a list of (bar_index, price, kind) with kind 1 = high, -1 = low.
    """
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    n = len(df)
    points = []

    for i in range(lookback, n - lookback):
        window_high = high[i - lookback:i + lookback + 1]
        window_low = low[i - lookback:i + lookback + 1]
        if high[i] == window_high.max() and window_high.argmax() == lookback:
            points.append((i, high[i], 1))
        elif low[i] == window_low.min() and window_low.argmin() == lookback:
            points.append((i, low[i], -1))

    if not points:
        return []

    zigzag = [points[0]]
    for idx, price, kind in points[1:]:
        _, last_price, last_kind = zigzag[-1]
        if kind == last_kind:
            better = price > last_price if kind == 1 else price < last_price
            if better:
                zigzag[-1] = (idx, price, kind)
        else:
            zigzag.append((idx, price, kind))

    return zigzag


def cluster_levels(swings, tolerance):
    """
    Group swing points that sit at effectively the same price into levels.

    `tolerance` is in price terms and should be volatility-scaled by the caller
    (a fixed pip band would make gold and EURGBP incomparable). Two swings
    within tolerance of each other belong to the same level.

    A level's strength is its touch count -- how many independent times price
    turned at that price. This is the "confluence" you asked for, in its most
    literal and least hand-wavy form: repeated independent rejection.

    Returns a list of dicts with price, touches, first/last bar index, and
    whether the touches were highs, lows, or both (both = a flip level, price
    used it as resistance then support or vice versa, which is the strongest
    variety).
    """
    if not swings:
        return []

    ordered = sorted(swings, key=lambda item: item[1])
    clusters = []
    current = [ordered[0]]

    for point in ordered[1:]:
        if point[1] - current[-1][1] <= tolerance:
            current.append(point)
        else:
            clusters.append(current)
            current = [point]
    clusters.append(current)

    levels = []
    for cluster in clusters:
        prices = [price for _, price, _ in cluster]
        kinds = {kind for _, _, kind in cluster}
        indices = [idx for idx, _, _ in cluster]
        levels.append({
            "price": float(np.mean(prices)),
            "touches": len(cluster),
            "first_bar": min(indices),
            "last_bar": max(indices),
            "kind": "flip" if len(kinds) > 1 else ("resistance" if 1 in kinds else "support"),
        })

    return sorted(levels, key=lambda level: level["touches"], reverse=True)


def _classify(entry, band, approached_from_below, window_high, window_low, reaction_atr):
    """
    Classify one touch, measuring BOTH directions from the same reference point.

    This is the fix for the bug that invalidated the first level report. The
    original measured moves from the LEVEL price while the baseline measured
    from the close. Since a touch bar's close sat a median of 0.45 ATR away
    from the level, "moved away" got that distance as a free head start while
    "moved through" had to cover it first. Against a 1.0 ATR threshold that
    head start was worth about +0.13 -- which was precisely the "edge" every
    instrument showed. Measured from the entry price instead, it went to -0.002.

    Entry price is also the honest reference: you fill at market, not at the
    idealised level.
    """
    away = entry - window_low if approached_from_below else window_high - entry
    through = window_high - entry if approached_from_below else entry - window_low
    threshold = band * reaction_atr

    if away >= threshold and away > through:
        return "respected", away / band
    if through >= threshold:
        return "broken", None
    return "neither", None


def _run_touches(df, level_prices, atr_values, horizon, reaction_atr, tolerance_atr,
                 start_bars=None):
    """
    Walk every bar that touches one of `level_prices` and classify what followed.
    Shared by the real test, the shifted control, and nothing else -- keeping one
    implementation is what makes the comparison trustworthy.
    """
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    n = len(df)

    counts = {"respected": 0, "broken": 0, "neither": 0}
    sizes = []

    for idx, price in enumerate(level_prices):
        i = (start_bars[idx] if start_bars else 0) + 1
        while i < n - horizon:
            band = atr_values[i]
            if np.isnan(band) or band <= 0:
                i += 1
                continue

            tolerance = band * tolerance_atr
            if not (low[i] <= price + tolerance and high[i] >= price - tolerance):
                i += 1
                continue

            outcome, size = _classify(
                entry=close[i],
                band=band,
                approached_from_below=close[i] < price,
                window_high=high[i + 1:i + 1 + horizon].max(),
                window_low=low[i + 1:i + 1 + horizon].min(),
                reaction_atr=reaction_atr,
            )
            counts[outcome] += 1
            if size is not None:
                sizes.append(size)

            i += horizon  # one approach shouldn't be counted bar by bar

    return counts, sizes


def measure_reactions(df, levels, min_touches=3, horizon=20, reaction_atr=1.0,
                      tolerance_atr=0.25, seed=7):
    """
    Did price react at these levels, or would any price have done the same?

    Two controls, because the first version had only the weak one and it hid a
    fatal bug:

      baseline_rate  random bars, close treated as a pseudo-level. Answers
                     "what does a coin flip look like here".
      shifted_rate   the REAL levels, displaced by 0.5-2 ATR to prices with no
                     structural meaning, then run through the identical touch
                     detection. This is the control that matters. It holds the
                     method, the instrument, the tolerance geometry and the
                     sample construction constant, and varies only whether the
                     price is a real level. If shifted levels score the same as
                     real ones, level LOCATION is irrelevant and the detector is
                     measuring the market's general tendency to move, not
                     structure.

    Only levels formed before a touch are eligible, so nothing validates itself
    on its own future.
    """
    if not levels:
        return None

    strong = [level for level in levels if level["touches"] >= min_touches]
    if not strong:
        return None

    atr_values = atr(df).to_numpy()
    n = len(df)

    prices = [level["price"] for level in strong]
    starts = [level["first_bar"] for level in strong]

    counts, sizes = _run_touches(
        df, prices, atr_values, horizon, reaction_atr, tolerance_atr, starts
    )
    total = sum(counts.values())
    if total == 0:
        return None

    respect_rate = counts["respected"] / total
    # Binomial standard error -- a +0.22 edge on 101 touches is not meaningfully
    # better than a +0.14 edge on 5000, and the first report ranked them as if
    # it were.
    std_error = float(np.sqrt(respect_rate * (1 - respect_rate) / total))

    rng = np.random.default_rng(seed)
    median_atr = float(np.nanmedian(atr_values))
    offsets = rng.uniform(0.5, 2.0, size=len(prices)) * rng.choice([-1, 1], size=len(prices))
    shifted_prices = [price + offset * median_atr for price, offset in zip(prices, offsets)]

    shifted_counts, _ = _run_touches(
        df, shifted_prices, atr_values, horizon, reaction_atr, tolerance_atr, starts
    )
    shifted_total = sum(shifted_counts.values())
    shifted_rate = shifted_counts["respected"] / shifted_total if shifted_total else None

    baseline = _baseline_reaction_rate(
        df, atr_values, horizon, reaction_atr, samples=min(2000, n // 2), seed=seed
    )

    return {
        "levels_tested": len(strong),
        "touch_events": total,
        "respected": counts["respected"],
        "broken": counts["broken"],
        "neither": counts["neither"],
        "respect_rate": respect_rate,
        "std_error": std_error,
        "break_rate": counts["broken"] / total,
        "median_reaction_atr": float(np.median(sizes)) if sizes else 0.0,
        "baseline_rate": baseline,
        "shifted_rate": shifted_rate,
        "shifted_touches": shifted_total,
        "edge_over_baseline": respect_rate - baseline if baseline is not None else None,
        "edge_over_shifted": respect_rate - shifted_rate if shifted_rate is not None else None,
    }


def _baseline_reaction_rate(df, atr_values, horizon, reaction_atr, samples, seed):
    """
    Weak control: random bars, close treated as a pseudo-level, direction
    assigned by coin flip. Entry and reference are the same price here, so this
    was never affected by the head-start bug -- which is exactly why the real
    test scoring 0.13 above it looked like a discovery instead of a defect.
    Kept for context, but `shifted_rate` is the control that decides anything.
    """
    rng = np.random.default_rng(seed)
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    n = len(df)

    valid = np.where(~np.isnan(atr_values))[0]
    valid = valid[(valid > 20) & (valid < n - horizon - 1)]
    if valid.size == 0:
        return None

    picks = rng.choice(valid, size=min(samples, valid.size), replace=False)
    respected = 0
    counted = 0

    for i in picks:
        band = atr_values[i]
        if band <= 0:
            continue
        outcome, _ = _classify(
            entry=close[i],
            band=band,
            approached_from_below=bool(rng.random() < 0.5),
            window_high=high[i + 1:i + 1 + horizon].max(),
            window_low=low[i + 1:i + 1 + horizon].min(),
            reaction_atr=reaction_atr,
        )
        counted += 1
        if outcome == "respected":
            respected += 1

    return respected / counted if counted else None


def build_levels(df, lookback=5, tolerance_atr=0.5):
    """Convenience: swings -> volatility-scaled tolerance -> clustered levels."""
    swings = find_swings(df, lookback=lookback)
    if not swings:
        return []
    median_atr = float(atr(df).median())
    if not median_atr or np.isnan(median_atr):
        return []
    return cluster_levels(swings, tolerance=median_atr * tolerance_atr)
