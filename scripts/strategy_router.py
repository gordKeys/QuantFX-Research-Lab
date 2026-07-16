from strategies.mean_reversion import MeanReversion
from strategies.mean_reversion_pullback import MeanReversionPullback
from strategies.mean_pullback_combo import MeanPullbackCombo
from strategies.momentum import Momentum
from strategies.five_signal_confluence_scalper import FiveSignalConfluenceScalper
from strategies.pullback_trend import PullbackTrend
from strategies.scalp_reversion import ScalpReversion
from strategies.trend_follow import TrendFollowing
from strategies.volatility_breakout import VolatilityBreakout
from strategies.h1_confluence_trend import H1ConfluenceTrend, H1SessionConfluenceTrend


def _normalize_symbol(symbol: str) -> str:
    """Strip common broker suffixes (Exness 'm', '.raw', '-ecn', etc.) so
    routing still matches when the live symbol string carries a suffix,
    instead of silently falling back to the default strategy."""
    raw = (symbol or "").upper()
    for suffix in (".RAW", ".ECN", ".PRO", "-ECN", "_ECN", ".M", "-M", "_M", "M"):
        if raw.endswith(suffix) and len(raw) - len(suffix) >= 6:
            return raw[: -len(suffix)]
    return raw


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
            "h1_confluence_trend": H1ConfluenceTrend(),
            # Session-filtered variant: restricted to the walk-forward
            # winning hours (incl. 16:00 UTC, the strongest single hour),
            # which is where the gold edge concentrates.
            "h1_session_confluence_trend": H1SessionConfluenceTrend(
                allowed_hours={11, 13, 14, 16, 18, 20}
            ),
        }

        self.symbol_map = {
            "AUDUSD": "five_signal_confluence_scalper",
            "EURUSD": "five_signal_confluence_scalper",
            "USDJPY": "five_signal_confluence_scalper",
            "USDCHF": "five_signal_confluence_scalper",
            # XAUUSD: walk-forward testing found the H1 wide/late-hours
            # confluence variant was the strongest strategy overall, with
            # gold carrying most of the edge. Route it there instead of the
            # default mean-reversion fallback.
            "XAUUSD": "h1_session_confluence_trend",
        }

        self.default_strategy = "mean_reversion"

    def get_strategy_name(self, symbol: str) -> str:
        return self.symbol_map.get(_normalize_symbol(symbol), self.default_strategy)

    def get_strategy(self, symbol: str):
        return self.registry[self.get_strategy_name(symbol)]

    def get_registry(self):
        return self.registry

    def update_mapping(self, symbol: str, strategy_name: str):
        if strategy_name not in self.registry:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        self.symbol_map[_normalize_symbol(symbol)] = strategy_name
