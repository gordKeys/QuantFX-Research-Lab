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


def measure_reactions(df, levels, min_touches=3, horizon=20, reaction_atr=1.0,
                      tolerance_atr=0.25, seed=7):
    """
    Did price actually react at these levels, or is that hindsight?

    For every bar where price enters a level's tolerance band, look forward
    `horizon` bars and classify:

        respected  price moved >= reaction_atr AWAY from the level, on the side
                   it approached from, without first closing decisively through
        broken     price moved >= reaction_atr THROUGH the level
        neither    it just sat there

    Then the same measurement is run on randomly chosen bars with no level
    present, giving a baseline bounce rate for the instrument. The edge is the
    gap between the two. A level_rate of 0.60 against a baseline of 0.58 is
    noise dressed as structure.

    Only levels formed BEFORE the touch are eligible, so a level built partly
    from future swings cannot validate itself. That lookahead is the single
    easiest way to produce a beautiful backtest of nothing.
    """
    if not levels:
        return None

    atr_series = atr(df)
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    atr_values = atr_series.to_numpy()
    n = len(df)

    strong = [level for level in levels if level["touches"] >= min_touches]
    if not strong:
        return None

    respected = 0
    broken = 0
    neither = 0
    reaction_sizes = []

    for level in strong:
        price = level["price"]
        # Only test bars after the level had formed its qualifying touches.
        start = level["first_bar"] + 1

        i = start
        while i < n - horizon:
            band = atr_values[i]
            if np.isnan(band) or band <= 0:
                i += 1
                continue

            tolerance = band * tolerance_atr
            touched = (low[i] <= price + tolerance) and (high[i] >= price - tolerance)
            if not touched:
                i += 1
                continue

            approached_from_below = close[i] < price
            window_high = high[i + 1:i + 1 + horizon].max()
            window_low = low[i + 1:i + 1 + horizon].min()
            threshold = band * reaction_atr

            if approached_from_below:
                moved_away = price - window_low
                moved_through = window_high - price
            else:
                moved_away = window_high - price
                moved_through = price - window_low

            if moved_away >= threshold and moved_away > moved_through:
                respected += 1
                reaction_sizes.append(moved_away / band)
            elif moved_through >= threshold:
                broken += 1
            else:
                neither += 1

            # Skip the horizon so one approach isn't counted bar by bar.
            i += horizon

    total = respected + broken + neither
    if total == 0:
        return None

    baseline = _baseline_reaction_rate(
        df, atr_values, horizon, reaction_atr, samples=min(2000, n // 2), seed=seed
    )

    return {
        "levels_tested": len(strong),
        "touch_events": total,
        "respected": respected,
        "broken": broken,
        "neither": neither,
        "respect_rate": respected / total,
        "break_rate": broken / total,
        "median_reaction_atr": float(np.median(reaction_sizes)) if reaction_sizes else 0.0,
        "baseline_rate": baseline,
        "edge_over_baseline": (respected / total) - baseline if baseline is not None else None,
    }


def _baseline_reaction_rate(df, atr_values, horizon, reaction_atr, samples, seed):
    """
    Null model: pick random bars, pretend the current close is a level, and
    apply exactly the same reaction test. This is what "price moved a bit after
    touching a price" looks like when the price had no special significance.
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
        price = close[i]
        # Direction assigned at random -- there is no real approach side here.
        approached_from_below = rng.random() < 0.5

        window_high = high[i + 1:i + 1 + horizon].max()
        window_low = low[i + 1:i + 1 + horizon].min()
        threshold = band * reaction_atr

        if approached_from_below:
            moved_away = price - window_low
            moved_through = window_high - price
        else:
            moved_away = window_high - price
            moved_through = price - window_low

        counted += 1
        if moved_away >= threshold and moved_away > moved_through:
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
