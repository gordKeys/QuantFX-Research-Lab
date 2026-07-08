from bootstrap import add_project_root
add_project_root()

import os
from dataclasses import dataclass
from typing import Iterable

from engine.data_loader import DataLoader
from engine.features import FeatureEngine
from engine.backtester import Backtester

from strategies.mean_reversion import MeanReversion
from strategies.mean_reversion_pullback import MeanReversionPullback
from strategies.bollinger_mean_reversion import BollingerMeanReversion
from strategies.five_signal_confluence_scalper import FiveSignalConfluenceScalper
from strategies.h1_confluence_trend import H1ConfluenceTrend, H1SessionConfluenceTrend
from strategies.momentum import Momentum
from strategies.scalp_reversion import ScalpReversion
from strategies.trend_follow import TrendFollowing
from strategies.support_resistance_breakout import SupportResistanceBreakout
from strategies.volatility_breakout import VolatilityBreakout
from strategies.pullback_trend import PullbackTrend
from strategies.mean_pullback_combo import MeanPullbackCombo


def default_strategy_grid():
    return {
        "mean_reversion": MeanReversion(),
        "mean_reversion_pullback": MeanReversionPullback(),
        "bollinger_mean_reversion": BollingerMeanReversion(),
        "five_signal_confluence_scalper": FiveSignalConfluenceScalper(),
        "support_resistance_breakout": SupportResistanceBreakout(),
        "scalp_reversion": ScalpReversion(),
        "h1_confluence_trend": H1ConfluenceTrend(),
        "h1_session_confluence_trend": H1SessionConfluenceTrend(),
        "momentum": Momentum(),
        "trend": TrendFollowing(),
        "pullback_trend": PullbackTrend(),
        "mean_pullback_combo": MeanPullbackCombo(),
        "volatility_breakout": VolatilityBreakout(),
    }


def load_symbol_data(symbol=None, data_path=None):
    loader = DataLoader(path=data_path, symbol=symbol)
    return FeatureEngine().add_features(loader.load())


@dataclass
class StrategySummary:
    symbol: str
    strategy: str
    balance: float
    trades: int
    win_rate: float
    avg_r: float


def run_strategy_on_data(data, strategy, symbol_name, strategy_name):
    result = Backtester(data, strategy).run()
    return StrategySummary(
        symbol=symbol_name,
        strategy=strategy_name,
        balance=result["final_balance"],
        trades=result["total_trades"],
        win_rate=result["win_rate"],
        avg_r=result["avg_r"],
    )


def infer_symbol_from_path(path):
    base = os.path.basename(path)
    if "_" in base:
        return base.split("_")[0]
    return os.path.splitext(base)[0]


def resolve_symbol_inputs(symbols: Iterable[str] | None, data_dir="data"):
    if symbols:
        return list(symbols)
    return [
        os.path.join(data_dir, name)
        for name in sorted(os.listdir(data_dir))
        if name.endswith("_M5.csv")
    ]
