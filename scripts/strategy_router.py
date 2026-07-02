from strategies.mean_reversion import MeanReversion
from strategies.momentum import Momentum
from strategies.trend_follow import TrendFollowing
from strategies.volatility_breakout import VolatilityBreakout


class StrategyRouter:

    def __init__(self):
        self.registry = {
            "mean_reversion": MeanReversion(lookback=20, entry_z=1.5),
            "momentum": Momentum(),
            "trend": TrendFollowing(),
            "volatility_breakout": VolatilityBreakout(),
        }

        self.symbol_map = {
            "EURUSD": "mean_reversion",
            "GBPUSD": "mean_reversion",
            "XAUUSD": "trend",
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
