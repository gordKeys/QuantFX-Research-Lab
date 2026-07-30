"""
================================================================================
  VX COST-AWARE BACKTEST + WALK-FORWARD
  Tests VX Million v2 AND VX Monster (no-BTC) with REAL spread + commission.

  WHY COSTS MATTER ENORMOUSLY on M5:
  ─────────────────────────────────────────────────────────────────
  With 20+ trades/day, spread alone can cost 2-5% monthly.
  The previous backtests had ZERO costs — every "edge" shown was
  gross. This script deducts real costs on every trade and shows
  what the NET, after-cost edge actually is.

  COSTS MODELLED (Exness typical, conservative estimates):
  ─────────────────────────────────────────────────────────────────
  Spread (paid on entry, in pips):
    EURUSDm : 0.8 pips  (typical Exness Standard spread)
    GBPUSDm : 1.0 pips
    XAUUSDm : 25 pips   (Gold: wider spread, per 0.01 lot point)
    ETHUSDm : 2.0 pips  (USD per price unit equivalent)

  Commission: Exness Standard = $0 explicit commission, spread-only.
    (If you're on Raw Spread account: add $3.50/lot round-trip
     — toggle RAW_SPREAD_ACCOUNT = True below)

  Slippage (entry): 0.5 pip per trade (conservative estimate for M5
    fast-moving entries during signal hours).

  Swap (overnight): not modelled here — if you hold <1 day most of
    the time this is minor. Add manually if holding multi-day.
  ─────────────────────────────────────────────────────────────────

  The two bots tested:
    Bot A — VX Million v2 (Config A): EUR/GBP/XAU, M5+H1, 3%/25%/8x mart
    Bot B — VX Monster (no-BTC):      EUR/GBP/XAU+ETH, M5+H1/H4, 2%/20%/8x mart

  Output per bot:
    • Gross return (no costs) — matches your previous backtest numbers
    • Net return (with costs) — the real number
    • Cost drag — how much spread/slippage costs you
    • Per-symbol cost breakdown
    • Walk-forward: median/worst net return across rolling windows
    • Verdict: deploy / review / do not deploy

  Run:  python vx_cost_aware_bt.py
================================================================================
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

UTC = timezone.utc

# ─── Account ──────────────────────────────────────────────────────
ACCOUNT_LOGIN    = 173186259
ACCOUNT_SERVER   = "Exness-MT5Real"
ACCOUNT_PASSWORD = "Gordonpap@2023"
# TERMINAL_PATH    = r"C:\Program Files\MetaTrader 5 EXNESS\terminal64.exe"

START_BAL    = 1000.0
BT_DAYS      = 90
HISTORY_DAYS = 200
WF_WINDOW    = 45
WF_STEP      = 15

# ─── Cost model (edit these to match your real account) ───────────
RAW_SPREAD_ACCOUNT = False   # set True if you pay commission + raw spread

SPREAD_PIPS = {
    "EURUSDm": 0.8,    # pips
    "GBPUSDm": 1.0,
    "XAUUSDm": 25.0,   # XAU: pips = price points × 10
    "ETHUSDm": 2.0,
}
SLIPPAGE_PIPS = 0.5    # extra pips per entry (market impact)

# Raw spread account: commission in $ per lot, round-trip
COMMISSION_PER_LOT_RT = 3.50  # only if RAW_SPREAD_ACCOUNT = True

# Point values (USD per pip per 1 lot)
POINT_VALUE = {
    "EURUSDm": 10.0,   # 1 pip = $10/lot for pairs quoted in USD
    "GBPUSDm": 10.0,
    "XAUUSDm": 1.0,    # XAU: 1 point = $1/lot (spread in points not pips)
    "ETHUSDm": 1.0,
}

# ─── Bot A — VX Million v2 ─────────────────────────────────────────
BOT_A = {
    "name": "VX Million v2 (Config A)",
    "symbols": ["EURUSDm", "GBPUSDm", "XAUUSDm"],
    "tf": mt5.TIMEFRAME_M5,
    "tf_label": "M5",
    "trend_tf": mt5.TIMEFRAME_H1,
    "h_ema": 20,
    "session_hours": set(range(8, 22)),
    "rr": 1.5,
    "atr_mult": 0.8,
    "base_risk": 3.0,
    "max_risk":  25.0,
    "martingale": 1.8,
    "max_mart":   4.0,
    "win_boost":  2.0,
    "max_boost":  8.0,
    "sym_cfg": {
        "EURUSDm": {"min_score":4,"fast":5,"slow":13,"rsi_p":9,"ob":68,"os":32},
        "GBPUSDm": {"min_score":3,"fast":5,"slow":13,"rsi_p":9,"ob":68,"os":32},
        "XAUUSDm": {"min_score":3,"fast":5,"slow":13,"rsi_p":9,"ob":68,"os":32},
    },
    "crypto_syms": [],
    "crypto_trend_tf": None,
}

# ─── Bot B — VX Monster (no-BTC) ───────────────────────────────────
BOT_B = {
    "name": "VX Monster no-BTC",
    "symbols": ["EURUSDm", "GBPUSDm", "XAUUSDm", "ETHUSDm"],
    "tf": mt5.TIMEFRAME_M5,
    "tf_label": "M5",
    "trend_tf": mt5.TIMEFRAME_H1,
    "h_ema": 20,
    "session_hours": set(range(8, 22)),
    "rr": None,   # per-symbol
    "atr_mult": None,
    "base_risk": 2.0,
    "max_risk":  20.0,
    "martingale": 1.8,
    "max_mart":   4.0,
    "win_boost":  2.0,
    "max_boost":  8.0,
    "sym_cfg": {
        "EURUSDm": {"min_score":4,"fast":5,"slow":13,"rsi_p":9,"ob":68,"os":32,"rr":1.5,"atr":0.8},
        "GBPUSDm": {"min_score":3,"fast":5,"slow":13,"rsi_p":9,"ob":68,"os":32,"rr":1.5,"atr":0.8},
        "XAUUSDm": {"min_score":3,"fast":5,"slow":13,"rsi_p":9,"ob":68,"os":32,"rr":1.5,"atr":0.8},
        "ETHUSDm": {"min_score":3,"fast":9,"slow":21,"rsi_p":14,"ob":72,"os":28,"rr":1.0,"atr":1.2},
    },
    "crypto_syms": ["ETHUSDm"],
    "crypto_trend_tf": mt5.TIMEFRAME_H4,
}

# ═══════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════
def connect():
    if not mt5.initialize():
        print("MT5 init failed");
        return False
    # if not mt5.initialize(path=TERMINAL_PATH):
    #     if not mt5.initialize():
    #         print("MT5 init failed"); return False
    ok = mt5.login(ACCOUNT_LOGIN, password=ACCOUNT_PASSWORD, server=ACCOUNT_SERVER)
    if not ok: print(f"Login failed: {mt5.last_error()}"); mt5.shutdown(); return False
    print(f"Connected: {mt5.account_info().login}")
    all_syms = list(set(BOT_A["symbols"] + BOT_B["symbols"]))
    for s in all_syms: mt5.symbol_select(s, True)
    return True

def fetch(sym, tf, days):
    now = datetime.now(UTC); start = now - timedelta(days=days+10)
    r = mt5.copy_rates_range(sym, tf, start, now)
    if r is None or len(r)==0: return None
    df = pd.DataFrame(r)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df.set_index("time")

def add_indicators(df, scfg):
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
    df["bb_mid"]=c.rolling(20).mean(); bbs=c.rolling(20).std()
    df["bb_upper"]=df["bb_mid"]+2*bbs; df["bb_lower"]=df["bb_mid"]-2*bbs
    tr=pd.concat([df["high"]-df["low"],(df["high"]-c.shift()).abs(),(df["low"]-c.shift()).abs()],axis=1).max(axis=1)
    df["atr"]=tr.ewm(alpha=1/14,min_periods=14,adjust=False).mean()
    df["vol_avg"]=v.rolling(20).mean(); df["vol"]=v
    return df

def attach_trend(df_low, df_high, h_ema):
    ema = df_high["close"].ewm(span=h_ema, adjust=False).mean()
    dh = df_high.copy(); dh["t"]=np.where(dh["close"]>ema,"UP","DOWN")
    merged = dh["t"].reindex(df_low.index.union(dh.index)).ffill().reindex(df_low.index)
    df_low["trend"]=merged; return df_low

# ═══════════════════════════════════════════════════════════════════
#  COST CALCULATION
# ═══════════════════════════════════════════════════════════════════
def trade_cost_usd(sym, lot, is_raw_spread=RAW_SPREAD_ACCOUNT):
    """Total cost in USD for one round-trip trade (entry + exit spread + slippage)."""
    sp = SPREAD_PIPS.get(sym, 1.0) + SLIPPAGE_PIPS
    pv = POINT_VALUE.get(sym, 10.0)
    spread_cost = sp * pv * lot
    comm = (COMMISSION_PER_LOT_RT * lot) if is_raw_spread else 0.0
    return spread_cost + comm

# ═══════════════════════════════════════════════════════════════════
#  SIGNAL SCORING (shared)
# ═══════════════════════════════════════════════════════════════════
def is_bull_engulf(c,p): return p["close"]<p["open"] and c["close"]>c["open"] and c["open"]<p["close"] and c["close"]>p["open"]
def is_bear_engulf(c,p): return p["close"]>p["open"] and c["close"]<c["open"] and c["open"]>p["close"] and c["close"]<p["open"]
def is_bull_pin(r):
    b=abs(r["close"]-r["open"]); return b>0 and (min(r["close"],r["open"])-r["low"])>=2*b and (r["high"]-max(r["close"],r["open"]))<=b
def is_bear_pin(r):
    b=abs(r["close"]-r["open"]); return b>0 and (r["high"]-max(r["close"],r["open"]))>=2*b and (min(r["close"],r["open"])-r["low"])<=b

def score(curr, prev, trend, scfg, vol_factor=1.05):
    vol_ok = curr["vol_avg"]>0 and curr["vol"]>=curr["vol_avg"]*vol_factor
    b=sum([prev["ema_fast"]<=prev["ema_slow"] and curr["ema_fast"]>curr["ema_slow"],
           curr["macd_h"]>0 and curr["macd_h"]>prev["macd_h"],
           curr["close"]<=curr["bb_lower"]*1.001 and curr["rsi"]<scfg["ob"],
           is_bull_engulf(curr,prev) or is_bull_pin(curr), vol_ok])
    s=sum([prev["ema_fast"]>=prev["ema_slow"] and curr["ema_fast"]<curr["ema_slow"],
           curr["macd_h"]<0 and curr["macd_h"]<prev["macd_h"],
           curr["close"]>=curr["bb_upper"]*0.999 and curr["rsi"]>scfg["os"],
           is_bear_engulf(curr,prev) or is_bear_pin(curr), vol_ok])
    if trend=="DOWN": b=0
    if trend=="UP":   s=0
    return b, s

# ═══════════════════════════════════════════════════════════════════
#  CORE SIMULATION — COST-AWARE
# ═══════════════════════════════════════════════════════════════════
def simulate(bot, data, win_start, win_end):
    syms = [s for s in bot["symbols"] if s in data]
    # Use first available symbol as timeline reference
    ref_sym = syms[0]
    ref = sorted(t for t in data[ref_sym].index if win_start<=t<win_end)
    if len(ref)<200: return None

    bal=START_BAL; peak=START_BAL; max_dd=0.0; min_ratio=1.0
    open_pos={}; trades=[]
    ss={sym:{"mart":1.0,"boost":1.0,"streak_w":0,"streak_l":0,"total":0,"wins":0} for sym in syms}
    cost_total = {sym:0.0 for sym in syms}
    pnl_gross  = {sym:0.0 for sym in syms}
    pnl_net    = {sym:0.0 for sym in syms}

    for i, ts in enumerate(ref):
        if i<50: continue
        # ── exits ────────────────────────────────────────────────
        for sym in list(open_pos.keys()):
            if sym not in data or ts not in data[sym].index: continue
            bar=data[sym].loc[ts].to_dict(); pos=open_pos[sym]
            out=None; gross_pnl=0.0
            if pos["dir"]=="BUY":
                if bar["low"]<=pos["sl"]:  out="LOSS"; gross_pnl=-pos["risk"]
                if bar["high"]>=pos["tp"]: out="WIN";  gross_pnl=pos["risk"]*pos["rr"]
            else:
                if bar["high"]>=pos["sl"]: out="LOSS"; gross_pnl=-pos["risk"]
                if bar["low"]<=pos["tp"]:  out="WIN";  gross_pnl=pos["risk"]*pos["rr"]
            if out:
                net_pnl = gross_pnl - pos["cost"]  # deduct pre-paid cost
                bal=round(max(bal+net_pnl,0.01),2); peak=max(peak,bal)
                dd=(peak-bal)/peak*100; max_dd=max(max_dd,dd)
                min_ratio=min(min_ratio,bal/peak)
                pnl_gross[sym]+=gross_pnl; pnl_net[sym]+=net_pnl
                sv=ss[sym]; sv["total"]+=1
                if out=="WIN":
                    sv["wins"]+=1; sv["streak_w"]+=1; sv["streak_l"]=0; sv["mart"]=1.0
                    sv["boost"]=min(bot["win_boost"]**max(0,sv["streak_w"]-1),bot["max_boost"])
                else:
                    sv["streak_l"]+=1; sv["streak_w"]=0; sv["boost"]=1.0
                    sv["mart"]=min(sv["mart"]*bot["martingale"],bot["max_mart"])
                trades.append({"sym":sym,"dir":pos["dir"],"out":out,
                                "gross_pnl":gross_pnl,"net_pnl":net_pnl,"cost":pos["cost"]})
                del open_pos[sym]
        if i<1: continue
        prev_ts=ref[i-1]

        # ── entries ───────────────────────────────────────────────
        for sym in syms:
            if sym in open_pos: continue
            is_crypto = sym in bot.get("crypto_syms", [])
            if not is_crypto and ts.hour not in bot["session_hours"]: continue
            if sym not in data or ts not in data[sym].index or prev_ts not in data[sym].index: continue
            curr=data[sym].loc[ts].to_dict(); prev=data[sym].loc[prev_ts].to_dict()
            if pd.isna(curr.get("atr")) or pd.isna(curr.get("bb_upper")): continue
            scfg=bot["sym_cfg"][sym]
            trend=curr.get("trend")
            b,s_=score(curr,prev,trend,scfg)
            dir_=None
            if b>=scfg["min_score"] and b>=s_: dir_="BUY"
            elif s_>=scfg["min_score"] and s_>b: dir_="SELL"
            if not dir_: continue

            sv=ss[sym]
            pct=min(bot["base_risk"]*sv["boost"]*sv["mart"],bot["max_risk"])
            risk=bal*(pct/100)
            rr  = scfg.get("rr",  bot.get("rr",  1.5))
            atr = scfg.get("atr", bot.get("atr_mult", 0.8))
            asl = atr*curr["atr"]; en=curr["close"]
            sl=en-asl if dir_=="BUY" else en+asl
            tp=en+asl*rr if dir_=="BUY" else en-asl*rr
            # Approximate lot size (rough, for cost estimation)
            lot = max(0.01, risk/max(asl*10000,0.01))  # rough lot
            cost = trade_cost_usd(sym, lot)
            cost_total[sym]+=cost
            open_pos[sym]={"dir":dir_,"entry":en,"sl":sl,"tp":tp,
                           "risk":risk,"rr":rr,"lot":lot,"cost":cost}

    n=len(trades); wins=sum(1 for t in trades if t["out"]=="WIN")
    wr=wins/n*100 if n else 0
    gw=sum(t["gross_pnl"] for t in trades if t["out"]=="WIN")
    gl=abs(sum(t["gross_pnl"] for t in trades if t["out"]=="LOSS"))
    pf_gross=gw/gl if gl>0 else 999
    nw=sum(t["net_pnl"] for t in trades if t["out"]=="WIN")
    nl=abs(sum(t["net_pnl"] for t in trades if t["out"]=="LOSS"))
    pf_net=nw/nl if nl>0 else 999
    total_cost=sum(cost_total.values())
    gross_ret=(sum(t["gross_pnl"] for t in trades)/START_BAL)*100
    net_ret=(bal/START_BAL-1)*100
    small_low=51.0*min_ratio
    return {"gross_ret":gross_ret,"net_ret":net_ret,"max_dd":max_dd,
            "wr":wr,"pf_gross":pf_gross,"pf_net":pf_net,"n":n,
            "total_cost":total_cost,"cost_pct":total_cost/START_BAL*100,
            "small_low":small_low,"wiped":small_low<5.0,
            "pnl_gross":pnl_gross,"pnl_net":pnl_net,"cost_sym":cost_total,
            "tpd":n/((win_end-win_start).days) if n>0 else 0}

# ═══════════════════════════════════════════════════════════════════
#  PRINT HELPERS
# ═══════════════════════════════════════════════════════════════════
def print_single(r, label, syms):
    print(f"\n  {label}")
    print(f"  {'─'*60}")
    print(f"  Gross return  : {r['gross_ret']:>+8.1f}%  (NO costs — the old number)")
    print(f"  Net return    : {r['net_ret']:>+8.1f}%  (after spread + slippage)")
    print(f"  Cost drag     : {r['cost_pct']:>8.1f}%  (${r['total_cost']:,.0f} total cost on ${START_BAL:,.0f})")
    print(f"  Max drawdown  : {r['max_dd']:>8.1f}%")
    print(f"  Win rate      : {r['wr']:>8.1f}%  ({r['n']} trades, {r['tpd']:.1f}/day)")
    print(f"  PF gross      : {r['pf_gross']:>8.3f}")
    print(f"  PF net        : {r['pf_net']:>8.3f}  ← the number that matters")
    print(f"  Small-acct $51: ${r['small_low']:.2f}  ({'WIPED' if r['wiped'] else 'survived'})")
    print(f"\n  Per-symbol breakdown:")
    print(f"  {'sym':<12}{'gross $':>10}{'cost $':>10}{'net $':>10}{'net%':>8}")
    print(f"  {'─'*50}")
    for s in syms:
        g=r['pnl_gross'].get(s,0); c=r['cost_sym'].get(s,0); n=r['pnl_net'].get(s,0)
        print(f"  {s:<12}{g:>+10,.2f}{c:>10,.2f}{n:>+10,.2f}{n/START_BAL*100:>+8.1f}%")

def print_wf_summary(bot_name, rows):
    nets=[r["net_ret"] for r in rows]; dds=[r["max_dd"] for r in rows]
    pfn=[r["pf_net"] for r in rows]; costs=[r["cost_pct"] for r in rows]
    wiped=sum(1 for r in rows if r["wiped"])
    prof=sum(1 for r in rows if r["net_ret"]>0)
    print(f"\n  Walk-Forward Summary — {bot_name}")
    print(f"  {'─'*60}")
    print(f"  Windows profitable (net): {prof}/{len(rows)}")
    print(f"  Wipeouts               : {wiped}/{len(rows)}")
    print(f"  Median net return      : {np.median(nets):+.1f}% / {WF_WINDOW}d")
    print(f"  Worst net return       : {min(nets):+.1f}%")
    print(f"  Median PF (net)        : {np.median(pfn):.3f}")
    print(f"  Median drawdown        : {np.median(dds):.1f}%   Worst: {max(dds):.1f}%")
    print(f"  Median cost drag       : {np.median(costs):.1f}% per window")
    if wiped==0 and prof>=len(rows)*0.6 and np.median(pfn)>=1.10:
        print(f"  VERDICT: ✅ Net edge confirmed — consider deploying")
    elif wiped==0 and np.median(pfn)>=1.05:
        print(f"  VERDICT: ⚠️  Thin net edge — monitor closely if deployed")
    else:
        print(f"  VERDICT: ❌ Net edge too thin or negative after costs — do NOT deploy")

# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    print("\n"+"="*72)
    print("  VX COST-AWARE BACKTEST + WALK-FORWARD")
    print(f"  Spread: EUR {SPREAD_PIPS['EURUSDm']}p | GBP {SPREAD_PIPS['GBPUSDm']}p | "
          f"XAU {SPREAD_PIPS['XAUUSDm']}p | ETH {SPREAD_PIPS['ETHUSDm']}p")
    print(f"  Slippage: {SLIPPAGE_PIPS}p per entry | Commission: "
          f"{'$3.50/lot RT (raw spread)' if RAW_SPREAD_ACCOUNT else 'none (standard spread)'}")
    print(f"  Start bal: ${START_BAL:,.0f} | BT: {BT_DAYS}d | WF: {WF_WINDOW}d windows")
    print("="*72)

    if not connect(): return

    print(f"\nDownloading {HISTORY_DAYS} days of data...")
    all_syms = list(set(BOT_A["symbols"]+BOT_B["symbols"]))
    all_tfs   = {mt5.TIMEFRAME_M5, mt5.TIMEFRAME_H1, mt5.TIMEFRAME_H4}
    raw={}
    for sym in all_syms:
        for tf in all_tfs:
            r=fetch(sym,tf,HISTORY_DAYS)
            raw[(sym,tf)]=r
            if r is not None: print(f"  {sym} tf={tf}: {len(r)} bars")
    mt5.shutdown()

    # Build prepared data dicts per bot
    def prepare(bot):
        d={}
        for sym in bot["symbols"]:
            m5=raw.get((sym,mt5.TIMEFRAME_M5))
            if m5 is None: continue
            scfg=bot["sym_cfg"][sym]
            df=add_indicators(m5.copy(),scfg)
            is_crypto = sym in bot.get("crypto_syms",[])
            ttf = bot["crypto_trend_tf"] if is_crypto else bot["trend_tf"]
            if ttf:
                hi=raw.get((sym,ttf))
                if hi is not None: df=attach_trend(df,hi.copy(),bot["h_ema"])
            d[sym]=df
        return d

    data_a=prepare(BOT_A); data_b=prepare(BOT_B)

    # Single-window backtest
    tN=min(min(df.index.max() for df in data_a.values()),
           min(df.index.max() for df in data_b.values()))
    bt_start=tN-timedelta(days=BT_DAYS)

    print(f"\n{'='*72}")
    print(f"  SINGLE-WINDOW BACKTEST  ({bt_start.date()} → {tN.date()}, {BT_DAYS}d)")
    print(f"{'='*72}")

    ra=simulate(BOT_A, data_a, bt_start, tN)
    if ra: print_single(ra, BOT_A["name"], BOT_A["symbols"])
    rb=simulate(BOT_B, data_b, bt_start, tN)
    if rb: print_single(rb, BOT_B["name"], BOT_B["symbols"])

    # Walk-forward
    t0=max(min(df.index.min() for df in data_a.values()),
           min(df.index.min() for df in data_b.values()))
    windows=[]; ws=t0
    while ws+timedelta(days=WF_WINDOW)<=tN:
        windows.append((ws,ws+timedelta(days=WF_WINDOW))); ws+=timedelta(days=WF_STEP)

    print(f"\n{'='*72}")
    print(f"  WALK-FORWARD  ({WF_WINDOW}d windows, rolling {WF_STEP}d — {len(windows)} windows)")
    print(f"{'='*72}")

    for bot, data, label in [(BOT_A,data_a,"A: VX Million v2"),(BOT_B,data_b,"B: VX Monster")]:
        print(f"\n  {label}")
        print(f"  {'#':>2}  {'Window':<24}{'Gross%':>8}{'Net%':>8}{'Cost%':>7}{'PFnet':>7}{'DD%':>7}  Status")
        print("  "+"-"*68)
        rows=[]
        for idx,(ws,we) in enumerate(windows,1):
            r=simulate(bot,data,ws,we)
            if r is None: continue
            rows.append(r)
            st="WIPED" if r["wiped"] else ("✅" if r["net_ret"]>0 else "🔻")
            print(f"  {idx:>2}  {ws.date()}→{we.date()} "
                  f"{r['gross_ret']:>+7.1f}%{r['net_ret']:>+7.1f}%"
                  f"{r['cost_pct']:>6.1f}%{r['pf_net']:>7.3f}{r['max_dd']:>6.1f}%  {st}")
        if rows: print_wf_summary(bot["name"],rows)

    # Head-to-head comparison
    print(f"\n{'='*72}")
    print(f"  HEAD-TO-HEAD: Gross vs Net — what costs actually do to these bots")
    print(f"{'='*72}")
    for bot, data in [(BOT_A,data_a),(BOT_B,data_b)]:
        r=simulate(bot,data,bt_start,tN)
        if r:
            swing = r["gross_ret"]-r["net_ret"]
            print(f"  {bot['name']:<28}: gross {r['gross_ret']:>+6.1f}%  →  "
                  f"net {r['net_ret']:>+6.1f}%   (costs ate {swing:.1f}%)")
    print()
    print("  The 'cost ate' number is the key insight.")
    print("  If spread + slippage turns a positive gross into a negative net,")
    print("  the bot was NEVER profitable — the edge was illusion, not skill.")
    print("="*72+"\n")

if __name__=="__main__":
    main()