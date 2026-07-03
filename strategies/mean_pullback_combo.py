import pandas as pd
from strategies.base_strategy import BaseStrategy
from strategies.mean_reversion import MeanReversion
from strategies.pullback_trend import PullbackTrend


class MeanPullbackCombo(BaseStrategy):

    def __init__(self):
        self.mean_reversion = MeanReversion(lookback=20, entry_z=1.5)
        self.pullback_trend = PullbackTrend()

    def generate_signals(self, data: pd.DataFrame):
        mean_signals = self.mean_reversion.generate_signals(data)
        pullback_signals = self.pullback_trend.generate_signals(data)

        signals = pd.Series(0, index=data.index)
        for i in range(len(data)):
            mean_signal = int(mean_signals.iloc[i])
            pull_signal = int(pullback_signals.iloc[i])
            if mean_signal != 0 and mean_signal == pull_signal:
                signals.iloc[i] = mean_signal
        return signals
