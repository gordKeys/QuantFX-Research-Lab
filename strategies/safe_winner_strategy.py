import pandas as pd

from strategies.base_strategy import BaseStrategy


class SafeWinnerStrategy(BaseStrategy):

    def __init__(
        self,
        base_strategy: BaseStrategy,
        mode: str = "mean_reversion",
        allowed_hours=None,
        min_atr_ratio=0.00045,
        max_atr_ratio=0.0040,
        max_trend_bias=0.0025,
        min_trend_bias=0.0006,
    ):
        self.base_strategy = base_strategy
        self.mode = mode
        self.allowed_hours = set(allowed_hours or [11, 13, 14, 16, 18, 20])
        self.min_atr_ratio = min_atr_ratio
        self.max_atr_ratio = max_atr_ratio
        self.max_trend_bias = max_trend_bias
        self.min_trend_bias = min_trend_bias

    def _hour_gate(self, index):
        try:
            return index.hour in self.allowed_hours
        except Exception:
            return True

    def generate_signals(self, data: pd.DataFrame):
        df = data.copy()
        base_signals = self.base_strategy.generate_signals(df)
        signals = pd.Series(0, index=df.index)

        atr_ratio = df["atr"] / df["close"]
        ema50 = df["ema50"]
        ema200 = df["ema200"]
        rsi = df["close"].diff().rolling(14).mean()

        for i in range(len(df)):
            if base_signals.iloc[i] == 0:
                continue
            if not self._hour_gate(df.index[i]):
                continue
            if pd.isna(atr_ratio.iloc[i]) or pd.isna(ema50.iloc[i]) or pd.isna(ema200.iloc[i]):
                continue
            if atr_ratio.iloc[i] < self.min_atr_ratio or atr_ratio.iloc[i] > self.max_atr_ratio:
                continue

            trend_bias = abs((ema50.iloc[i] - ema200.iloc[i]) / df["close"].iloc[i])

            if self.mode == "mean_reversion":
                if trend_bias > self.max_trend_bias:
                    continue
                signals.iloc[i] = base_signals.iloc[i]

            elif self.mode == "momentum":
                if trend_bias < self.min_trend_bias:
                    continue
                if base_signals.iloc[i] == 1 and ema50.iloc[i] < ema200.iloc[i]:
                    continue
                if base_signals.iloc[i] == -1 and ema50.iloc[i] > ema200.iloc[i]:
                    continue
                signals.iloc[i] = base_signals.iloc[i]

            else:
                signals.iloc[i] = base_signals.iloc[i]

        return signals
