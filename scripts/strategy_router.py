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
            # USDCHF-specific tier: it's been the worst or near-worst
            # performer of the 4 majors in every live window checked so far
            # (day-1, week-1, and this latest day: -79.51/17 trades, 35% win
            # rate). Rather than cut it outright -- we don't have local
            # USDCHF price history to properly backtest a full exclusion --
            # require a stronger confluence score (5 of 6 signals instead of
            # 3) so it only takes higher-conviction setups. Reversible: drop
            # this back to "five_signal_confluence_scalper" in symbol_map
            # below if a few more days show this isn't the fix.
            "five_signal_confluence_scalper_strict": FiveSignalConfluenceScalper(min_score=5),
            # USDJPY: 2nd-worst performer on the post-fix day (-42.97, 35.7%
            # win), though its history is more mixed than USDCHF's -- it was
            # actually one of the stronger performers in an earlier sample.
            # Normally that mixed record would argue for waiting another day
            # before touching it. Given the hard 14-day challenge deadline,
            # cutting drag sooner is the more defensible tradeoff here.
            # Revert to "five_signal_confluence_scalper" in symbol_map below
            # if this turns out to have been the wrong call.
            "five_signal_confluence_scalper_strict_jpy": FiveSignalConfluenceScalper(min_score=5),
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
            "USDJPY": "five_signal_confluence_scalper_strict_jpy",
            "USDCHF": "five_signal_confluence_scalper_strict",
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
