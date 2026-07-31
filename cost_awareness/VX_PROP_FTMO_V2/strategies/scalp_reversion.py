import pandas as pd
from strategies.base_strategy import BaseStrategy


class ScalpReversion(BaseStrategy):

    def __init__(self, lookback=8, entry_z=1.1, max_atr_ratio=0.0012):
        self.lookback = lookback
        self.entry_z = entry_z
        self.max_atr_ratio = max_atr_ratio

    def generate_signals(self, data: pd.DataFrame):
        df = data.copy()
        signals = pd.Series(0, index=df.index)

        ma = df["close"].rolling(self.lookback).mean()
        std = df["close"].rolling(self.lookback).std()
        z = (df["close"] - ma) / std
        atr_ratio = df["atr"] / df["close"]

        for i in range(self.lookback, len(df)):
            if pd.isna(z.iloc[i]) or pd.isna(atr_ratio.iloc[i]):
                continue

            if atr_ratio.iloc[i] > self.max_atr_ratio:
                continue

            if z.iloc[i] > self.entry_z:
                signals.iloc[i] = -1
            elif z.iloc[i] < -self.entry_z:
                signals.iloc[i] = 1

        return signals
