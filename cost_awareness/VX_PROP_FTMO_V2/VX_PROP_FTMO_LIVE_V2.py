"""
============================================================

 VX PROP FTMO V2
 Strategy-Routed FTMO Live Trading Bot

 Strategies:
 EURUSD  -> 5 Signal Confluence (no trend)
 AUDUSD  -> 5 Signal Confluence (no candle)
 USDJPY  -> Strict Confluence + Trend
 USDCHF  -> Strict Confluence
 XAUUSD  -> H1 Session Confluence
 Default -> Mean Reversion z=1.5

 Risk:
 Fixed 0.4%
 Daily Halt -4%
 Total Halt -8%

============================================================
"""


import time
import logging

import MetaTrader5 as mt5
import pandas as pd


from datetime import datetime, timezone


import config

from indicators import add_indicators
from strategy_router import StrategyRouter
from risk_manager import RiskManager
from execution import Execution



UTC = timezone.utc



logging.basicConfig(

    level=logging.INFO,

    format=
    "%(asctime)s | %(levelname)s | %(message)s"

)


log = logging.getLogger()



router = StrategyRouter()



risk = RiskManager(

    config.START_EQUITY,

    config.DAILY_HALT_PCT,

    config.TOTAL_HALT_PCT,

    config.PROFIT_TARGET

)



execution = Execution(

    config.MAGIC,

    config.RISK_PCT

)





def connect():

    if not mt5.initialize(
        path=config.TERMINAL_PATH
    ):

        log.error(
            mt5.last_error()
        )

        return False



    login = mt5.login(

        config.ACCOUNT_LOGIN,

        password=config.ACCOUNT_PASSWORD,

        server=config.ACCOUNT_SERVER

    )



    if not login:

        log.error(
            mt5.last_error()
        )

        return False



    account = mt5.account_info()


    log.info(
        f"CONNECTED | "
        f"{account.login} | "
        f"Balance {account.balance}"
    )



    for symbol in config.SYMBOLS:

        mt5.symbol_select(
            symbol,
            True
        )



    return True






def get_data(symbol):


    rates = mt5.copy_rates_from_pos(

        symbol,

        mt5.TIMEFRAME_M5,

        0,

        config.CANDLE_COUNT

    )



    if rates is None:

        return None



    df = pd.DataFrame(rates)



    df["time"] = pd.to_datetime(

        df["time"],

        unit="s",

        utc=True

    )


    df.set_index(

        "time",

        inplace=True

    )


    return df






def calculate_sl_tp(symbol, direction, atr):


    info = mt5.symbol_info(symbol)


    tick = mt5.symbol_info_tick(symbol)



    if info is None or tick is None:

        return None,None



    distance = (
        atr *
        config.ATR_MULTIPLIER
    )



    if direction == 1:

        price = tick.ask

        sl = price-distance

        tp = price+(distance*config.RR)


    else:

        price=tick.bid

        sl=price+distance

        tp=price-(distance*config.RR)



    return (

        round(sl,info.digits),

        round(tp,info.digits)

    )







def has_position(symbol):


    positions = mt5.positions_get(

        symbol=symbol

    )


    if not positions:

        return False



    for p in positions:

        if p.magic == config.MAGIC:

            return True



    return False






def run():


    if not connect():

        return



    log.info(
        "=============================="
    )

    log.info(
        " VX PROP FTMO V2 STARTED "
    )

    log.info(
        "=============================="
    )



    while True:


        account = mt5.account_info()


        if account is None:

            time.sleep(10)

            continue



        equity = account.equity



        stats = risk.update(
            equity
        )



        log.info(

            f"Equity ${equity:.2f} | "
            f"Profit {stats['profit']:.2f}% | "
            f"Daily DD {stats['daily_dd']:.2f}% | "
            f"Total DD {stats['total_dd']:.2f}%"

        )



        if not risk.can_trade():

            log.warning(
                "Trading halted by FTMO rules"
            )

            time.sleep(
                config.LOOP_INTERVAL
            )

            continue




        for symbol in config.SYMBOLS:



            if has_position(symbol):

                continue



            df = get_data(symbol)



            if df is None:

                continue



            df = add_indicators(df)



            strategy = router.get_strategy(symbol)



            strategy_name = router.get_strategy_name(symbol)



            signals = strategy.generate_signals(df)



            signal = int(
                signals.iloc[-1]
            )



            if signal == 0:

                continue



            atr = df["atr"].iloc[-1]



            if pd.isna(atr):

                continue




            sl,tp = calculate_sl_tp(

                symbol,

                signal,

                atr

            )


            if sl is None:

                continue



            lot = execution.calculate_lot(

                symbol,

                equity,

                abs(sl-df["close"].iloc[-1])

            )



            log.info(

                f"{symbol} | "
                f"{strategy_name} | "
                f"SIGNAL {signal} | "
                f"LOT {lot}"

            )



            execution.send_order(

                symbol,

                signal,

                lot,

                sl,

                tp

            )




        time.sleep(

            config.LOOP_INTERVAL

        )






if __name__ == "__main__":

    run()