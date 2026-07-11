import pandas as pd

from strategies.base_strategy import BaseStrategy


class MirrorStrategy(BaseStrategy):
    def __init__(self, base_strategy):
        self.base_strategy = base_strategy

    def generate_signals(self, data: pd.DataFrame):
        signals = self.base_strategy.generate_signals(data)
        mirrored = signals.copy()
        mirrored = mirrored.apply(lambda value: -int(value) if int(value) != 0 else 0)
        return mirrored
