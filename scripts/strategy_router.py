from strategies.mean_reversion import MeanReversion
from strategies.mean_reversion_pullback import MeanReversionPullback
from strategies.mean_pullback_combo import MeanPullbackCombo
from strategies.momentum import Momentum
from strategies.five_signal_confluence_scalper import FiveSignalConfluenceScalper
from strategies.pullback_trend import PullbackTrend
from strategies.scalp_reversion import ScalpReversion
from strategies.trend_follow import TrendFollowing
from strategies.volatility_breakout import VolatilityBreakout


class StrategyRouter:

    def __init__(self):
        self.registry = {
            "mean_reversion": MeanReversion(lookback=20, entry_z=1.5),
            "mean_reversion_pullback": MeanReversionPullback(),
            "mean_pullback_combo": MeanPullbackCombo(),
            "momentum": Momentum(),
            "five_signal_confluence_scalper": FiveSignalConfluenceScalper(),
            "trend": TrendFollowing(),
            "pullback_trend": PullbackTrend(),
            "scalp_reversion": ScalpReversion(),
            "volatility_breakout": VolatilityBreakout(),
        }

        self.symbol_map = {
            "AUDUSD": "mean_reversion",
            "EURUSD": "five_signal_confluence_scalper",
            "USDJPY": "five_signal_confluence_scalper",
            "USDCHF": "five_signal_confluence_scalper",
        }

        self.default_strategy = "mean_reversion"

    def get_strategy_name(self, symbol: str) -> str:
        return self.symbol_map.get(symbol.upper(), self.default_strategy)

    def get_strategy(self, symbol: str):
        return self.registry[self.get_strategy_name(symbol)]

    def get_registry(self):
        return self.registry

    def update_mapping(self, symbol: str, strategy_name: str):
        if strategy_name not in self.registry:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        self.symbol_map[symbol.upper()] = strategy_name
