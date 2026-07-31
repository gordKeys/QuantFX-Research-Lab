from strategies.mean_reversion import MeanReversion
from strategies.five_signal_confluence_scalper import FiveSignalConfluenceScalper
from strategies.h1_confluence_trend import H1SessionConfluenceTrend


def _normalize_symbol(symbol):

    raw = (symbol or "").upper()

    for suffix in (
        ".RAW",
        ".ECN",
        ".PRO",
        "-ECN",
        "_ECN",
        ".M",
        "-M",
        "_M",
        "M"
    ):

        if raw.endswith(suffix) and len(raw) > len(suffix)+5:
            return raw[:-len(suffix)]

    return raw



class StrategyRouter:


    def __init__(self):

        self.registry = {


            # EURUSD
            # 5-signal confluence
            # remove trend component
            # min score 3

            "eurusd_confluence":

                FiveSignalConfluenceScalper(
                    min_score=3,
                    disabled_components={"trend"}
                ),



            # AUDUSD
            # remove candle pattern

            "audusd_confluence":

                FiveSignalConfluenceScalper(
                    min_score=3,
                    disabled_components={"candle_pattern"}
                ),



            # USDJPY
            # strict + trend alignment

            "usdjpy_strict":

                FiveSignalConfluenceScalper(
                    min_score=5,
                    require_trend_alignment=True
                ),



            # USDCHF

            "usdchf_strict":

                FiveSignalConfluenceScalper(
                    min_score=5
                ),



            # Gold

            "gold_h1":

                H1SessionConfluenceTrend(
                    allowed_hours={
                        11,
                        13,
                        14,
                        16,
                        18,
                        20
                    }
                ),



            # fallback

            "default":

                MeanReversion(
                    lookback=20,
                    entry_z=1.5
                )
        }



        self.symbol_map = {


            "EURUSD":
                "eurusd_confluence",


            "AUDUSD":
                "audusd_confluence",


            "USDJPY":
                "usdjpy_strict",


            "USDCHF":
                "usdchf_strict",


            "XAUUSD":
                "gold_h1"

        }



    def get_strategy(self, symbol):

        clean = _normalize_symbol(symbol)


        name = self.symbol_map.get(
            clean,
            "default"
        )


        return self.registry[name]



    def get_strategy_name(self, symbol):

        clean = _normalize_symbol(symbol)

        return self.symbol_map.get(
            clean,
            "default"
        )