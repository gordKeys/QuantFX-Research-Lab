import pandas as pd
from strategies.base_strategy import BaseStrategy
from strategies.mean_reversion import MeanReversion
from strategies.pullback_trend import PullbackTrend


class MeanReversionPullback(BaseStrategy):

    def __init__(self):
        self.mean_reversion = MeanReversion()
        self.pullback_trend = PullbackTrend()

    def generate_signals(self, data: pd.DataFrame):
        mean_signals = self.mean_reversion.generate_signals(data)
        pullback_signals = self.pullback_trend.generate_signals(data)

        signals = pd.Series(0, index=data.index)
        for i in range(len(data)):
            mean_signal = int(mean_signals.iloc[i])
            pullback_signal = int(pullback_signals.iloc[i])

            if mean_signal != 0 and pullback_signal == mean_signal:
                signals.iloc[i] = mean_signal
            elif mean_signal != 0 and pullback_signal == 0:
                signals.iloc[i] = mean_signal
            elif pullback_signal != 0 and mean_signal == 0:
                signals.iloc[i] = pullback_signal

        return signals
