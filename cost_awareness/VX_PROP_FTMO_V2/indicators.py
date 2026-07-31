import pandas as pd


def add_indicators(df):

    df = df.copy()


    high = df["high"]
    low = df["low"]
    close = df["close"]


    tr = pd.concat(
        [
            high-low,
            (high-close.shift()).abs(),
            (low-close.shift()).abs()
        ],
        axis=1
    ).max(axis=1)


    df["atr"] = (
        tr
        .rolling(14)
        .mean()
    )


    return df