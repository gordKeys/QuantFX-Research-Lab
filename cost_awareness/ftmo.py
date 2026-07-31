"""
================================================================================

  -  Jesus will do it for me Amen  -

  VX PROP — FTMO-COMPLIANT LIVE BOT
  The SAFE version. Built to pass an FTMO challenge, not to get rich fast.

  Backtest: 14/14 challenge windows PASSED, 0 breaches.  [CONFIGURED FOR $100k]
            Worst total DD 4.4% (limit 10%), worst daily 3.9% (limit 5%).

  ─── HARD PROP RULES ENFORCED ────────────────────────────────────
  • Fixed risk 0.4% per trade — NEVER scales (no martingale, no boost)
  • DAILY self-halt at -4%  (FTMO hard fail is -5%)
  • TOTAL self-halt at -8%  (FTMO hard fail is -10%, static from start)
  • Equity-based: floating losses counted every loop (like FTMO does)
  • Stops all NEW trades when a halt triggers; flattens on total-halt
  • Profit target +10% (Phase 1) / +5% (Phase 2) — set TARGET below
  ─────────────────────────────────────────────────────────────────

    Set START_EQUITY to your CHALLENGE account's starting balance
      (e.g. 10000 for a $10k FTMO challenge). The halts are computed
      from THIS number, statically, exactly like FTMO's max loss.

    CHANGE MAGIC for each prop account you run (firms flag identical
      magic numbers across accounts as copy-trading).

  Symbols: EUR, GBP, XAU (forex) + ETH (crypto) | M5 | EMA-20 filter
  MAGIC: 444444  |  RUN: python VX_PROP_FTMO_LIVE.py
================================================================================
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime, timezone, timedelta

UTC = timezone.utc

# ════════════════════════════════════════════════════════════════
#  ACCOUNT  — set these to your FTMO challenge MT5 credentials
# ════════════════════════════════════════════════════════════════
ACCOUNT_LOGIN    = 1514154175         # FTMO demo login
ACCOUNT_SERVER   = "FTMO-Demo"        # FTMO demo server
ACCOUNT_PASSWORD = "$rx@6j?uv55P"   # FTMO demo password
TERMINAL_PATH    = r"C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe"  # FTMO terminal

# ════════════════════════════════════════════════════════════════
#  CHALLENGE PARAMETERS  — MUST match your FTMO account
# ════════════════════════════════════════════════════════════════
START_EQUITY    = 100000.0  # ← $100k FTMO challenge account
PROFIT_TARGET   = 10.0      # % — 10 for Phase 1, set 5 for Phase 2
DAILY_HALT_PCT  = 4.0       # self-halt buffer (FTMO daily fail = 5%)
TOTAL_HALT_PCT  = 8.0       # self-halt buffer (FTMO total fail = 10%)
RISK_PCT        = 0.4       # fixed risk per trade — NEVER changes
MAX_OPEN_TRADES = 3         # safety: cap simultaneous positions (gap protection, risk unchanged)

# ════════════════════════════════════════════════════════════════
#  SYMBOLS
# ════════════════════════════════════════════════════════════════
FOREX   = ["EURUSD", "GBPUSD", "XAUUSD"]   # FTMO has NO "m" suffix
CRYPTO  = []  #["ETHUSD"]   # if FTMO account has no crypto, set CRYPTO = [] and remove ETHUSD below
SYMBOLS = FOREX

SYM_CFG = {
    "EURUSD": {"rr":1.5,"atr":0.8,"min_score":4,"fast":5,"slow":13,"rsi_p":9,"ob":68,"os":32,
                "tf":"M5","trend_tf":"H1","is_crypto":False},
    "GBPUSD": {"rr":1.5,"atr":0.8,"min_score":3,"fast":5,"slow":13,"rsi_p":9,"ob":68,"os":32,
                "tf":"M5","trend_tf":"H1","is_crypto":False},
    "XAUUSD": {"rr":1.5,"atr":0.8,"min_score":3,"fast":5,"slow":13,"rsi_p":9,"ob":68,"os":32,
                "tf":"M5","trend_tf":"H1","is_crypto":False},
    "ETHUSD": {"rr":1.0,"atr":1.2,"min_score":3,"fast":9,"slow":21,"rsi_p":14,"ob":72,"os":28,
                "tf":"M5","trend_tf":"H4","is_crypto":True},
}

VOL_FACTOR  = 1.05
H_EMA       = 20
FOREX_HOURS = list(range(8, 22))
STOP_BUFFER_MULT = 1.5
MIN_ATR_POINTS   = 5
CANDLE_COUNT  = 150
LOOP_INTERVAL = 30
MAGIC         = 444444     # ← CHANGE per account

TF_MAP = {"M5":mt5.TIMEFRAME_M5, "H1":mt5.TIMEFRAME_H1, "H4":mt5.TIMEFRAME_H4}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    handlers=[logging.FileHandler("VX_PROP_FTMO_log.txt",encoding="utf-8"),
              logging.StreamHandler()])
log = logging.getLogger()

state = {sym:{"total":0,"wins":0,"pnl":0.0} for sym in SYMBOLS}

# Daily tracking for the daily-loss halt
day_state = {"date":None, "start_equity":START_EQUITY, "halted":False}
challenge = {"halted_total":False, "passed":False}

def connect():
    if not mt5.initialize(path=TERMINAL_PATH):
        log.error(f"MT5 init failed for {TERMINAL_PATH}: {mt5.last_error()}"); return False
    ok=mt5.login(ACCOUNT_LOGIN,password=ACCOUNT_PASSWORD,server=ACCOUNT_SERVER)
    if not ok:
        log.error(f"Login failed: {mt5.last_error()}"); mt5.shutdown(); return False
    info=mt5.account_info()
    log.info(f"VX PROP CONNECTED  |  {info.login}  |  ${info.balance:.2f}  |  equity ${info.equity:.2f}")
    for sym in SYMBOLS:
        if not mt5.symbol_select(sym,True):
            log.warning(f"  {sym} not available — check Market Watch")
        else:
            log.info(f"  {sym} enabled")
    return True

def get_candles(symbol, tf, count=CANDLE_COUNT):
    rates=mt5.copy_rates_from_pos(symbol,TF_MAP[tf],0,count)
    if rates is None or len(rates)==0: return None
    df=pd.DataFrame(rates); df["time"]=pd.to_datetime(df["time"],unit="s",utc=True)
    return df.set_index("time")

def add_indicators(df,scfg):
    c=df["close"]; v=df["tick_volume"]
    df["ema_fast"]=c.ewm(span=scfg["fast"],adjust=False).mean()
    df["ema_slow"]=c.ewm(span=scfg["slow"],adjust=False).mean()
    mf=c.ewm(span=12,adjust=False).mean(); ms=c.ewm(span=26,adjust=False).mean()
    df["macd"]=mf-ms; df["macd_sig"]=df["macd"].ewm(span=9,adjust=False).mean()
    df["macd_h"]=df["macd"]-df["macd_sig"]
    rp=scfg["rsi_p"]; delta=c.diff()
    ag=delta.clip(lower=0).ewm(alpha=1/rp,min_periods=rp,adjust=False).mean()
    al=(-delta.clip(upper=0)).ewm(alpha=1/rp,min_periods=rp,adjust=False).mean()
    df["rsi"]=100-(100/(1+ag/al))
    df["bb_mid"]=c.rolling(20).mean(); bb_std=c.rolling(20).std()
    df["bb_upper"]=df["bb_mid"]+2*bb_std; df["bb_lower"]=df["bb_mid"]-2*bb_std
    tr=pd.concat([df["high"]-df["low"],(df["high"]-c.shift()).abs(),(df["low"]-c.shift()).abs()],axis=1).max(axis=1)
    df["atr"]=tr.ewm(alpha=1/14,min_periods=14,adjust=False).mean()
    df["vol_avg"]=v.rolling(20).mean(); df["vol"]=v
    return df

def get_trend(symbol,scfg):
    rates=mt5.copy_rates_from_pos(symbol,TF_MAP[scfg["trend_tf"]],0,70)
    if rates is None or len(rates)==0: return None
    closes=pd.Series([r["close"] for r in rates])
    ema=closes.ewm(span=H_EMA,adjust=False).mean()
    return "UP" if rates[-1]["close"]>ema.iloc[-1] else "DOWN"

def is_bull_engulf(c,p): return p["close"]<p["open"] and c["close"]>c["open"] and c["open"]<p["close"] and c["close"]>p["open"]
def is_bear_engulf(c,p): return p["close"]>p["open"] and c["close"]<c["open"] and c["open"]>p["close"] and c["close"]<p["open"]
def is_bull_pin(r):
    b=abs(r["close"]-r["open"]); return b>0 and (min(r["close"],r["open"])-r["low"])>=2*b and (r["high"]-max(r["close"],r["open"]))<=b
def is_bear_pin(r):
    b=abs(r["close"]-r["open"]); return b>0 and (r["high"]-max(r["close"],r["open"]))>=2*b and (min(r["close"],r["open"])-r["low"])<=b

def score_signals(curr,prev,trend,scfg):
    vol_ok=curr["vol"]>=curr["vol_avg"]*VOL_FACTOR if curr["vol_avg"]>0 else False
    buy={"ema":prev["ema_fast"]<=prev["ema_slow"] and curr["ema_fast"]>curr["ema_slow"],
         "macd":curr["macd_h"]>0 and curr["macd_h"]>prev["macd_h"],
         "bb":curr["close"]<=curr["bb_lower"]*1.001 and curr["rsi"]<scfg["ob"],
         "candle":is_bull_engulf(curr,prev) or is_bull_pin(curr),"volume":vol_ok}
    sell={"ema":prev["ema_fast"]>=prev["ema_slow"] and curr["ema_fast"]<curr["ema_slow"],
          "macd":curr["macd_h"]<0 and curr["macd_h"]<prev["macd_h"],
          "bb":curr["close"]>=curr["bb_upper"]*0.999 and curr["rsi"]>scfg["os"],
          "candle":is_bear_engulf(curr,prev) or is_bear_pin(curr),"volume":vol_ok}
    b=sum(buy.values()); s=sum(sell.values())
    if trend:
        if trend=="DOWN": b=0
        if trend=="UP":   s=0
    return b,s

def compute_stop_distance(si,atr,scfg):
    if atr is None or pd.isna(atr) or atr<=0: return None
    point=si.point
    if atr/point < MIN_ATR_POINTS: return None
    atr_sl=scfg["atr"]*atr
    min_pts=si.trade_stops_level if si.trade_stops_level else 0
    spread_pts=(si.ask-si.bid)/point if (si.ask and si.bid) else 0
    floor_price=max(min_pts,spread_pts)*STOP_BUFFER_MULT*point
    return max(atr_sl,floor_price)

def get_lot(symbol,equity,stop_dist,scfg):
    si=mt5.symbol_info(symbol)
    if si is None: return 0
    risk_amt=equity*(RISK_PCT/100)   # FIXED — no scaling
    sl_pips=stop_dist/si.point
    if sl_pips<=0 or si.trade_tick_value<=0: return si.volume_min
    lot=risk_amt/(sl_pips*si.trade_tick_value)
    lot=max(si.volume_min,min(lot,si.volume_max))
    lot=round(round(lot/si.volume_step)*si.volume_step,2)
    log.info(f"  Sizing → fixed {RISK_PCT}%  ${risk_amt:.2f}  Lot:{lot}")
    return lot

def place_order(symbol,direction,atr,lot,scfg):
    si=mt5.symbol_info(symbol)
    if si is None: return False
    tick=mt5.symbol_info_tick(symbol)
    if tick is None: return False
    atr_sl=compute_stop_distance(si,atr,scfg)
    if atr_sl is None:
        log.warning(f"  [{symbol}] ATR too small — SKIPPED"); return False
    tp_dist=atr_sl*scfg["rr"]
    if direction=="BUY":
        otype=mt5.ORDER_TYPE_BUY; price=tick.ask; sl=price-atr_sl; tp=price+tp_dist
    else:
        otype=mt5.ORDER_TYPE_SELL; price=tick.bid; sl=price+atr_sl; tp=price-tp_dist
    sl=round(sl,si.digits); tp=round(tp,si.digits)
    if abs(sl-price)<si.point or abs(tp-price)<si.point:
        log.warning(f"  [{symbol}] SL/TP collapsed — SKIPPED"); return False
    dev=30 if scfg["is_crypto"] else 20
    req={"action":mt5.TRADE_ACTION_DEAL,"symbol":symbol,"volume":lot,"type":otype,
         "price":price,"sl":sl,"tp":tp,"deviation":dev,"magic":MAGIC,
         "comment":f"VXPROP {direction}","type_time":mt5.ORDER_TIME_GTC,
         "type_filling":mt5.ORDER_FILLING_IOC}
    result=mt5.order_send(req)
    if result is None or result.retcode!=mt5.TRADE_RETCODE_DONE:
        rc=result.retcode if result else "None"
        cm=result.comment if result else mt5.last_error()
        log.error(f"  [{symbol}] FAILED {rc}: {cm}"); return False
    log.info(f"   {direction} {symbol}  Price:{price:.5f}  SL:{sl}  TP:{tp}  Lot:{lot}")
    return True

def get_open_pos(symbol):
    pos=mt5.positions_get(symbol=symbol)
    if pos:
        own=[p for p in pos if p.magic==MAGIC]
        return own[0] if own else None
    return None

def flatten_all():
    """Close every bot position immediately (used on total-halt)."""
    for sym in SYMBOLS:
        p=get_open_pos(sym)
        if p is None: continue
        si=mt5.symbol_info(sym); tick=mt5.symbol_info_tick(sym)
        otype=mt5.ORDER_TYPE_SELL if p.type==0 else mt5.ORDER_TYPE_BUY
        price=tick.bid if p.type==0 else tick.ask
        mt5.order_send({"action":mt5.TRADE_ACTION_DEAL,"symbol":sym,"volume":p.volume,
                        "type":otype,"position":p.ticket,"price":price,"deviation":30,
                        "magic":MAGIC,"comment":"VXPROP FLATTEN",
                        "type_time":mt5.ORDER_TIME_GTC,"type_filling":mt5.ORDER_FILLING_IOC})
        log.warning(f"  [{sym}] FLATTENED (total-halt)")

def run():
    if not connect(): return
    acc=mt5.account_info()
    log.info("="*68)
    log.info("  VX PROP — FTMO LIVE STARTED")
    log.info(f"  Challenge size : ${START_EQUITY:,.0f}")
    log.info(f"  Profit target  : +{PROFIT_TARGET}%  (${START_EQUITY*PROFIT_TARGET/100:,.0f})")
    log.info(f"  Daily halt     : -{DAILY_HALT_PCT}%  (FTMO fail -5%)")
    log.info(f"  Total halt     : -{TOTAL_HALT_PCT}%  (FTMO fail -10%, static from ${START_EQUITY:,.0f})")
    log.info(f"  Risk/trade     : {RISK_PCT}% FIXED (no martingale, no boost)")
    log.info(f"  Magic          : {MAGIC}")
    log.info("="*68)

    prev_pos={sym:None for sym in SYMBOLS}
    loop=0
    try:
        while True:
            loop+=1
            now=datetime.now(UTC)
            acc=mt5.account_info()
            if acc is None: time.sleep(LOOP_INTERVAL); continue
            equity=acc.equity

            # ── Daily reset ─────────────────────────────────────
            today=now.date()
            if day_state["date"]!=today:
                day_state["date"]=today
                day_state["start_equity"]=equity
                day_state["halted"]=False
                log.info(f"\n=== NEW DAY {today} | day-start equity ${equity:.2f} ===")

            # ── Compute drawdowns (equity-based, like FTMO) ─────
            total_dd=(START_EQUITY-equity)/START_EQUITY*100
            daily_dd=(day_state["start_equity"]-equity)/day_state["start_equity"]*100
            profit  =(equity-START_EQUITY)/START_EQUITY*100

            log.info(f"\n-- VX PROP #{loop}  {now.strftime('%H:%M:%S UTC')} | "
                     f"Eq:${equity:.2f}  Profit:{profit:+.2f}%  "
                     f"TotalDD:{total_dd:.2f}%  DailyDD:{daily_dd:.2f}% --")

            # ── PASS check ──────────────────────────────────────
            if profit>=PROFIT_TARGET and not challenge["passed"]:
                challenge["passed"]=True
                log.info("="*68)
                log.info(f"   PROFIT TARGET HIT! +{profit:.2f}%  — target was +{PROFIT_TARGET}%")
                log.info(f"  Bot will STOP opening trades. Let open trades close, then")
                log.info(f"  you've passed this phase. Switch TARGET for Phase 2 if needed.")
                log.info("="*68)

            # ── TOTAL halt (hard safety) ────────────────────────
            if total_dd>=TOTAL_HALT_PCT and not challenge["halted_total"]:
                challenge["halted_total"]=True
                log.warning("="*68)
                log.warning(f"   TOTAL HALT at -{total_dd:.2f}% (buffer before FTMO -10%).")
                log.warning(f"  Flattening all positions and STOPPING. Do not restart blindly.")
                log.warning("="*68)
                flatten_all()

            # ── DAILY halt ──────────────────────────────────────
            if daily_dd>=DAILY_HALT_PCT and not day_state["halted"]:
                day_state["halted"]=True
                log.warning(f"   DAILY HALT at -{daily_dd:.2f}% — no new trades until tomorrow.")

            # Stop conditions
            trading_blocked = (challenge["passed"] or challenge["halted_total"]
                               or day_state["halted"])

            # Count currently-open bot positions (for MAX_OPEN_TRADES cap)
            open_count = sum(1 for s_ in SYMBOLS if get_open_pos(s_) is not None)

            for sym in SYMBOLS:
                scfg=SYM_CFG[sym]
                if not scfg["is_crypto"] and now.hour not in FOREX_HOURS: continue

                curr_pos=get_open_pos(sym)
                if prev_pos[sym] is not None and curr_pos is None:
                    # tally closed trade
                    nowt=datetime.now(); st=nowt.replace(hour=0,minute=0,second=0,microsecond=0)
                    deals=mt5.history_deals_get(st,nowt)
                    if deals:
                        sd=[d for d in deals if d.symbol==sym and d.entry==mt5.DEAL_ENTRY_OUT and d.magic==MAGIC]
                        if sd:
                            last=sorted(sd,key=lambda d:d.time)[-1]
                            state[sym]["total"]+=1; state[sym]["pnl"]+=last.profit
                            if last.profit>0: state[sym]["wins"]+=1
                            tag="WIN" if last.profit>0 else "LOSS"
                            log.info(f"  [{sym}] {tag} {last.profit:+.2f}  symPnL:{state[sym]['pnl']:+.2f}")
                prev_pos[sym]=curr_pos

                if curr_pos:
                    log.info(f"  [{sym}] OPEN vol:{curr_pos.volume} P&L:${curr_pos.profit:.2f}")
                    continue
                if trading_blocked: continue

                df=get_candles(sym,scfg["tf"])
                if df is None: continue
                df=add_indicators(df,scfg)
                curr=df.iloc[-1]; prev=df.iloc[-2]
                if pd.isna(curr["atr"]) or pd.isna(curr["bb_upper"]): continue
                trend=get_trend(sym,scfg)
                b,s=score_signals(curr,prev,trend,scfg)
                ms=scfg["min_score"]
                direction=None
                if b>=ms and b>=s: direction="BUY"
                elif s>=ms and s>b: direction="SELL"
                if direction is None: continue

                # Cap simultaneous positions (gap protection)
                if open_count >= MAX_OPEN_TRADES:
                    log.info(f"  [{sym}] signal seen but MAX_OPEN_TRADES ({MAX_OPEN_TRADES}) reached — skipping")
                    continue

                si=mt5.symbol_info(sym)
                stop_dist=compute_stop_distance(si,curr["atr"],scfg)
                if stop_dist is None: continue
                lot=get_lot(sym,equity,stop_dist,scfg)
                if lot>0:
                    log.info(f"  [{sym}] ✓ SIGNAL {direction}")
                    if place_order(sym,direction,curr["atr"],lot,scfg):
                        open_count += 1

            if challenge["halted_total"]:
                log.warning("  Challenge halted on total loss. Exiting loop.")
                break

            time.sleep(LOOP_INTERVAL)
    except KeyboardInterrupt:
        log.info("\nVX PROP stopped by user.")
    finally:
        mt5.shutdown(); log.info("Disconnected.")

if __name__=="__main__":
    run()