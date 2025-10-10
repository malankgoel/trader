FEATURES = """
**What you can do here**
- Define strategies as **JSON** with modular blocks: indicators → signals → logic/allocations.
- Supported **indicators**: `sma_price`, `ema_price`, `sma_return`, `std_price`, `std_return`, `rsi`, `drawdown`, `max_drawdown`.
- Supported **signals**: `gt`, `lt`, `cross_above`, `cross_below`.
- **Allocations**:
  - `conditional_weights` (2 branches: when_true / when_false)
  - `rules` (priority **if / elif / else** with a `default` branch)
- **Rebalance**: `B` (daily), `W` (weekly last trading day), `M` (month-end).
- Data source: **Yahoo Finance** (adjusted close).
"""

LIMITATIONS = """
**Limitations / assumptions**
- Only **daily** data from Yahoo is used (no intraday, futures, options, FX).
- No custom user code inside strategies (use the allowed blocks only).
- Execution model is simplified: calendar rebalances, single portfolio (no borrowing/leverage),
  slippage (bps) and per-trade fee are applied on rebalance trades.
- JSON must be **valid**: no comments, no trailing commas, strings are single-line, `null` not `None`.
- Weights in each branch should sum to **1.0**.
"""


MODEL_NAME = "gpt-5-2025-08-07"


DEFAULTS = {
    "objective": "trend",
    "risk_goal": "balanced",
    "weighting_hint": "equal",
    "hedge_intent": False,
    "start_date": "2018-01-01",
    "end_date": None,
    "rebalance_if_none": None,
    "frequency": "W",
    "slippage_bps": 2,
    "fee_per_trade": 0,
}

ALLOWED = {
    "objectives": ["trend", "mean_reversion", "hedge", "pairs", "carry"],
    "risk_goals": ["conservative", "balanced", "aggressive"],
    "signals": ["gt", "lt", "cross_above", "cross_below"],
    "rebalance": ["B", "W", "M"],
}


THEME_TICKERS = {
    "us_tech": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO", "CRM", "ADBE", "ORCL", "AMD", "NOW"],
    "ai":      ["NVDA", "MSFT", "GOOGL", "META", "AMD", "AVGO", "SNOW", "CRWD"],
    "semis":   ["NVDA", "AMD", "AVGO", "INTC", "TSM", "ASML", "MU", "QCOM"],
    "bonds":   ["SHY", "IEF", "TLT"],
    "gold":    ["GLD"],
}

HEDGE_FALLBACKS = ["BIL", "SHY", "IEF"]


INTENT_EXTRACTOR_PROMPT = """
Act as an intent extractor for trading strategies. From the user text, return a minimal JSON with only these keys and enums.
If something is ambiguous, choose the default shown in [brackets]. No prose, no code fences, no extra fields.
- objective ∈ {trend, mean_reversion, hedge, pairs, carry} [trend]
- risk_goal ∈ {conservative, balanced, aggressive} [balanced]
- themes: array of short tokens like "us_tech", "ai", "semis", "bonds", "gold". If none, []
- explicit_tickers: array of tickers found in the text (uppercase, deduped). If none, []
- hedge_intent ∈ {true,false} [false] (true if text mentions hedge/defensive/cash)
- weighting_hint ∈ {equal, cap_proxy, risk_heuristic, none} [equal]
- date_hints: { start: YYYY-MM-DD | null [null], end: YYYY-MM-DD | null [null] }
- rebalance_hint ∈ {B, W, M, none} [none]
Return compact JSON only.
""".strip()

UNIVERSE_BUILDER_PROMPT = """
You build a concrete ticker universe from themes + explicit tickers. Choose tickers by best judgment, but follow these hard rules.
Return compact JSON only: {"universe":[...], "hedge_assets":[...], "rule":"<short rule>"}

Selection rules (apply in order):
1) Start with explicit_tickers (uppercase, deduped).
2) If user wants more than explicit_tickers or has no tickers currently then follow Steps 3 to 8.
3) From themes, add 5-6 *widely traded, US-listed* tickers that clearly match each theme (e.g., us_tech, ai, semis, bonds, gold). Use primary share classes (prefer GOOGL over GOOG; avoid duplicate share classes).
4) Prefer mega/large caps and high-liquidity names (household names are OK as a proxy). Avoid OTC, penny stocks, microcaps, and obscure tickers.
5) Keep one symbol per issuer (e.g., pick GOOGL not GOOG).
6) Sort A→Z, then cap the final list to 6.
7) If explicit_tickers already cover a theme well, do not add more.
8) If no theme matches, pick a coherent, liquid set consistent with the objective (trend/mean_reversion/hedge/pairs/carry).

Hedging:
- If hedge_intent=true:
    - If any of ["BIL","SHY","IEF","SGOV","TFLO"] appear in universe, set hedge_assets to the first in this order encountered.
    - Else set hedge_assets to ["BIL"].
- If hedge_intent=false: hedge_assets = [].

Output:
- "universe": final list (1-12 tickers, A→Z)
- "hedge_assets": [] or ["BIL"]/["SHY"]/["IEF"]/["SGOV"]/["TFLO"]
- "rule": one short line stating what you did (e.g., "explicit + us_tech judgment; A→Z; capped 6; hedge=BIL").
Return compact JSON only.
""".strip()

INDICATOR_PICKER_PROMPT = """
Pick the minimal indicators needed for the objective, applied only to symbols that require them. Use these deterministic templates;
clip windows to [5,400]. IDs must be unique. Return {"indicators":[...]} only.

Templates by objective:
- trend → for each equity in universe: sma_price with windows {50,200}.
- mean_reversion → for each equity: rsi with window 14.
- hedge → for the hedge decision only: std_return with window 20 on each equity; if bonds or gold are in universe, no extra indicators for them.
- pairs → pick two tickers from universe A→Z first two; create sma_price window 100 on both.
- carry → on ETFs only (e.g., SHY/IEF/TLT/GLD): ema_price window 50.

Output item shape:
{"id":"<type>_<sym>_<win>", "type":"...", "params":{"symbol":"SYM", "window":N}}
For ema_price you may include {"adjust": false} in params if desired.
Return compact JSON only.
""".strip()

SIGNALS_LOGIC_PROMPT = """
Build signals and a single "logic" using only {gt, lt, cross_above, cross_below}.
Sources must be one of: {"kind":"price","symbol":...} | {"kind":"indicator","ref":...} | {"kind":"const","value":number}.
Deterministic patterns (round consts to 3 decimals):

- trend → per equity: gt(price, sma_200) ; logic is "or" over all per-equity signals.
- mean_reversion → per equity: lt(rsi14, 30) ; logic is "or".
- hedge → per equity: gt(std_ret20, const 0.020) ; logic is "or".
- pairs → two tickers A,B: approximate with two signals: gt(price_A, sma_A_100) AND lt(price_B, sma_B_100); logic is "and".
- carry → per ETF: gt(ema50, price) indicates exit; final logic is NOT( OR(all exits) ), i.e., stay long when none signal exit.

IDs must be unique; indicator refs must exist. Return only:
{"signals":[...], "logic": <signal-id or {"type":"and"|"or"|"not", ...}>}
Return compact JSON only.
""".strip()

ALLOCATION_WEIGHTS_PROMPT = """
Choose allocation deterministically.

If hedge_intent=true and hedge_assets nonempty, use "conditional_weights"; otherwise use "rules" with a single "default".
Policy selection:
- If weighting_hint in {equal, none} → use "equal".
- If weighting_hint = cap_proxy but no cap data → still "equal".
- If weighting_hint = risk_heuristic but no risk stats → still "equal".
- If text includes "core-satellite" → use 0.700 core / 0.300 satellite, equal-weight within groups.

Rounding/normalization: round every weight to 3 decimals, then renormalize each branch so the sum is exactly 1.000.

Return only one object:
- Conditional form:
  {"allocation":{"type":"conditional_weights","when_true":{T:W,...},"when_false":{T:W,...}},"policy":"equal"}
  where when_true = equal across "universe"; when_false = 100% to first asset in "hedge_assets" (or BIL if empty).
- Rules form:
  {"allocation":{"type":"rules","rules":[{"default":{T:W,...}}]},"policy":"equal"}
All tickers used must be in universe u hedge_assets. Return compact JSON only.
""".strip()

OPS_DATA_COSTS_REBALANCE_PROMPT = """
Fill the operational fields deterministically.

- If date_hints.start is null → "2018-01-01"; if end is null → null.
- frequency always "B".
- costs: slippage_bps = 2; fee_per_trade = 0.
- rebalance: if rebalance_hint given use it; else by risk_goal: conservative→"M", balanced→"W", aggressive→"W".

Return only:
{"data":{"source":"yahoo","start":"YYYY-MM-DD","end":null|"YYYY-MM-DD","frequency":"B"},
 "costs":{"slippage_bps":2,"fee_per_trade":0},
 "rebalance":{"frequency":"B"|"W"|"M"}}
Return compact JSON only.
""".strip()

AUDITOR_REPAIR_PROMPT = """
You are a fixer. Input is a candidate final strategy JSON and a machine-generated "errors" array.
Modify the fewest fields necessary so all checks pass:

- Unique IDs (indicators/signals), valid refs, allowed enums.
- All weight branches sum to 1.000 after rounding to 3 decimals.
- Weights only use tickers in "universe" (or hedge_assets where applicable).
- Signals reference existing indicators or price/const sources.
- Dates valid and ordered; frequency ∈ {"B"}; rebalance ∈ {"B","W","M"}.

If a reference can't be resolved, choose from provided "allowed_refs". If still impossible, set logic to the first valid signal id.
If a hedge branch requires an asset and none exists, use "BIL" if present in universe; else add "BIL" to universe and set the hedge branch to {"BIL":1.000}.
Output corrected JSON only; no prose, no code fences, no extra fields.
""".strip()




'''FEATURES = """
**What you can do here**
- Define strategies as **JSON** with modular blocks: indicators → signals → logic/allocations.
- Supported **indicators**: `sma_price`, `ema_price`, `sma_return`, `std_price`, `std_return`, `rsi`, `drawdown`, `max_drawdown`.
- Supported **signals**: `gt`, `lt`, `cross_above`, `cross_below`.
- **Allocations**:
  - `conditional_weights` (2 branches: when_true / when_false)
  - `rules` (priority **if / elif / else** with a `default` branch)
- **Rebalance**: `B` (daily), `W` (weekly last trading day), `M` (month-end).
- Data source: **Yahoo Finance** (adjusted close).
"""

LIMITATIONS = """
**Limitations / assumptions**
- Only **daily** data from Yahoo is used (no intraday, futures, options, FX).
- No custom user code inside strategies (use the allowed blocks only).
- Execution model is simplified: calendar rebalances, single portfolio (no borrowing/leverage),
  slippage (bps) and per-trade fee are applied on rebalance trades.
- JSON must be **valid**: no comments, no trailing commas, strings are single-line, `null` not `None`.
- Weights in each branch should sum to **1.0**.
"""

GPT_SYSTEM_PROMPT = """
You have to act like a strict JSON generator for trading strategies. Make sure to output only the json object and no commentary or anything else. 
The output is directly parsed as a json object for further analysis.
You will be given a trading strategy in words and you have to convert that into a json output stritly based on the rules explained below. 
Make sure you output just the json object, and it has no commentary and or trailling spaces.
Return the smallest valid JSON needed, no prose, no explanations, no code fences, no whitespace beyond what JSON requires.

Schema (the ONLY valid shapes)
{
  "version": "1.0",
  "meta": { "name": <string>, "notes": <string> },
  "universe": [<ticker>, ...],
  "data": { "source": "yahoo", "start": "YYYY-MM-DD", "end": null | "YYYY-MM-DD", "frequency": "B" },
  "costs": { "slippage_bps": <number>, "fee_per_trade": <number> },
  "indicators": [ { "id": <string>, "type": <indicator-type>, "params": { "symbol": <ticker>, ... } }, ... ],
  "signals": [ { "id": <string>, "type": <signal-type>, "left": <source>, "right": <source> }, ... ],
  "logic": <signal-id> | { "type": "and"|"or", "children": [<logic or signal-id>...] } | { "type": "not", "child": <logic or signal-id> },
  "allocation":
    { "type": "conditional_weights", "when_true": {<ticker>: <w>, ...}, "when_false": {<ticker>: <w>, ...} }
    OR
    { "type": "rules", "rules": [ { "when": <signal-id>, "weights": {<ticker>: <w>, ...} }, ..., { "default": {<ticker>: <w>, ...} } ] },
  "rebalance": { "frequency": "B"|"W"|"M" }
}

Sources used inside signals
- price source: { "kind": "price", "symbol": "QQQ" }
- indicator source: { "kind": "indicator", "ref": "<INDICATOR_ID>" }
- constant source: { "kind": "const", "value": <number> }

Indicator params (strict)
- sma_price: { symbol, window }
- ema_price: { symbol, window, adjust (optional, default false) }
- sma_return: { symbol, window }
- std_price: { symbol, window, ddof (optional, default 0) }
- std_return: { symbol, window, ddof (optional, default 0) }
- rsi: { symbol, window }
- drawdown: { symbol }
- max_drawdown: { symbol, window (optional; if omitted returns scalar over full history) }

Hard constraints & validation checklist
The assistant MUST self-check before output:
1) All ids are unique (indicators[].id, signals[].id).
2) Every signal's left/right is a valid source; indicator refs point to existing indicators.
3) Weights per branch sum to 1.0 exactly to 3 decimal places (round if needed).
4) All tickers used in weights appear in 'universe'.
5) Dates are YYYY-MM-DD; 'end' is null or a valid date; 'start' <= 'end' if both present.
6) Only allowed enums used (indicator/signal types; rebalance frequencies).
7) Strings are single-line; no trailing commas; numbers are plain (no 'NaN'/'Infinity').

Output rules (enforced)
- After showing the short capabilities list from Part A, output a SINGLE fenced code block labelled 'json'.
- No text before/after the JSON block (besides the short list in Part A). No explanations or comments inside the JSON.
- If the user asks for unsupported features, approximate using allowed blocks and clearly reflect the approximation in 'notes'.

Example (for the model to imitate)
1) Multi-branch rules (priority if/elif/else):
Plain English: If AAPL > 100SMA then 80%AAPL/20%TSLA; else if TSLA > 100EMA then 80%TSLA/20%AAPL; else 50/50. Start 2018, weekly rebalance, 2 bps.
{
  "version": "1.0",
  "meta": {
    "name": "AAPL vs TSLA SMA/EMA Strategy",
    "notes": "If AAPL > 100-day SMA then 80%AAPL/20%TSLA. Else if TSLA > 100-day EMA then 80%TSLA/20%AAPL. Else 50/50 split."
  },
  "universe": ["AAPL", "TSLA"],
  "data": {
    "source": "yahoo",
    "start": "2018-01-01",
    "end": None,
    "frequency": "B"
  },
  "costs": {
    "slippage_bps": 2,
    "fee_per_trade": 0
  },
  "indicators": [
    {
      "id": "sma_aapl_100",
      "type": "sma_price",
      "params": { "symbol": "AAPL", "window": 100 }
    },
    {
      "id": "ema_tsla_100",
      "type": "ema_price",
      "params": { "symbol": "TSLA", "window": 100 }
    }
  ],
  "signals": [
    {
      "id": "aapl_gt_sma",
      "type": "gt",
      "left": { "kind": "price", "symbol": "AAPL" },
      "right": { "kind": "indicator", "ref": "sma_aapl_100" }
    },
    {
      "id": "tsla_gt_ema",
      "type": "gt",
      "left": { "kind": "price", "symbol": "TSLA" },
      "right": { "kind": "indicator", "ref": "ema_tsla_100" }
    }
  ],
  "logic": "aapl_gt_sma",
  "allocation": {
    "type": "rules",
    "rules": [
      {
        "when": "aapl_gt_sma",
        "weights": { "AAPL": 0.8, "TSLA": 0.2 }
      },
      {
        "when": "tsla_gt_ema",
        "weights": { "AAPL": 0.2, "TSLA": 0.8 }
      },
      {
        "default": { "AAPL": 0.5, "TSLA": 0.5 }
      }
    ]
  },
  "rebalance": { "frequency": "W" }
}
"""

MODEL_NAME = "gpt-5-2025-08-07"
MODEL_NAME = "gpt-5-2025-08-07"'''


'''
import json
import math
from dataclasses import dataclass
from typing import Dict, Union, Optional, List, Tuple, Any

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px


from configuration import FEATURES, LIMITATIONS, GPT_SYSTEM_PROMPT, MODEL_NAME

from typing import Literal

# --- Block dictionaries you support (used for tolerant mapping) ---
_ALLOWED_INDICATORS = {
    "sma_price": {"req": ["symbol", "window"]},
    "ema_price": {"req": ["symbol", "window"]},
    "sma_return": {"req": ["symbol", "window"]},
    "std_price": {"req": ["symbol", "window"], "opt": ["ddof"]},
    "std_return": {"req": ["symbol", "window"], "opt": ["ddof"]},
    "rsi": {"req": ["symbol", "window"]},
    "drawdown": {"req": ["symbol", "window"]},
    "max_drawdown": {"req": ["symbol", "window"]},
}

_ALLOWED_SIGNALS = {"gt", "lt", "cross_above", "cross_below"}

def _closest_indicator(name: str) -> str:
    if name in _ALLOWED_INDICATORS:
        return name
    n = name.replace("-", "_").replace(" ", "_").lower()
    for k in _ALLOWED_INDICATORS.keys():
        if n == k or n.startswith(k.split("_")[0]):  # loose match e.g. "sma" -> "sma_price"
            return k
    # fallback to sma_price as the most common
    return "sma_price"

def _closest_signal(name: str) -> str:
    n = str(name).replace("-", "_").lower()
    if n in _ALLOWED_SIGNALS:
        return n
    if "gt" in n or "greater" in n:
        return "gt"
    if "lt" in n or "less" in n:
        return "lt"
    if "cross" in n and ("above" in n or "up" in n):
        return "cross_above"
    if "cross" in n and ("below" in n or "down" in n):
        return "cross_below"
    return "gt"

def _clean_symbol(s):
    return str(s).strip().upper()

def _safe_num(x, default):
    try:
        return int(x)
    except Exception:
        try:
            return float(x)
        except Exception:
            return default

def normalize_strategy_json(js: dict) -> dict:
    """
    Snap a possibly imperfect GPT JSON to the nearest valid schema for building blocks.
    """
    out = {
        "version": "1.0",
        "meta": {"name": "", "notes": ""},
        "universe": [],
        "data": {"source": "yahoo", "start": "2015-01-01", "end": None, "frequency": "B"},
        "costs": {"slippage_bps": 0, "fee_per_trade": 0},
        "indicators": [],
        "signals": [],
        "logic": None,
        "allocation": {"type": "conditional_weights", "when_true": {}, "when_false": {}},
        "rebalance": {"frequency": "M"},
    }

    # meta
    meta = js.get("meta", {})
    out["meta"]["name"] = str(meta.get("name", "")).strip()[:80]
    out["meta"]["notes"] = str(meta.get("notes", "")).strip()

    # universe
    univ = js.get("universe", [])
    if isinstance(univ, list):
        out["universe"] = [_clean_symbol(s) for s in univ if str(s).strip()]

    # data
    data = js.get("data", {})
    if isinstance(data, dict):
        out["data"]["source"] = (data.get("source") or "yahoo").lower()
        out["data"]["start"] = data.get("start") or "2015-01-01"
        out["data"]["end"] = data.get("end", None)
        out["data"]["frequency"] = data.get("frequency") or "B"

    # costs
    costs = js.get("costs", {})
    if isinstance(costs, dict):
        out["costs"]["slippage_bps"] = _safe_num(costs.get("slippage_bps", 0), 0)
        out["costs"]["fee_per_trade"] = _safe_num(costs.get("fee_per_trade", 0), 0)

    # indicators
    for item in js.get("indicators", []) or []:
        if not isinstance(item, dict):
            continue
        t = _closest_indicator(item.get("type", "sma_price"))
        sym = _clean_symbol(item.get("params", {}).get("symbol", (out["universe"] or ["SPY"])[0]))
        window = _safe_num(item.get("params", {}).get("window", 200), 200)
        ddof = _safe_num(item.get("params", {}).get("ddof", 0), 0)
        out["indicators"].append({
            "id": item.get("id", f"{t}_{sym}_{window}")[:64],
            "type": t,
            "params": {"symbol": sym, "window": window, **({"ddof": ddof} if "ddof" in _ALLOWED_INDICATORS[t].get("opt", []) else {})}
        })

    # signals
    for item in js.get("signals", []) or []:
        if not isinstance(item, dict):
            continue
        stype = _closest_signal(item.get("type", "gt"))
        left = item.get("left", {})
        right = item.get("right", {})
        out["signals"].append({
            "id": item.get("id", f"{stype}_{len(out['signals'])+1}")[:64],
            "type": stype,
            "left": left,
            "right": right
        })

    # logic
    # logic + allocation
    lg = js.get("logic")
    if isinstance(lg, dict):
        if "conditional_weights" in lg:
            cw = lg["conditional_weights"] or {}
            cond = cw.get("condition") or (out["signals"][0]["id"] if out["signals"] else None)
            wt_t = { _clean_symbol(k): float(v) for k, v in (cw.get("when_true") or {}).items() if k }
            wt_f = { _clean_symbol(k): float(v) for k, v in (cw.get("when_false") or {}).items() if k }
            out["logic"] = cond
            out["allocation"] = {"type":"conditional_weights","when_true":wt_t,"when_false":wt_f}
        elif "rules" in lg:
            rules_in = lg.get("rules") or []
            rules = []
            for r in rules_in:
                if "default" in r:
                    rules.append({"default": { _clean_symbol(k): float(v) for k, v in (r["default"] or {}).items() }})
                else:
                    rules.append({
                        "when": r.get("when", (out["signals"][0]["id"] if out["signals"] else "")),
                        "weights": { _clean_symbol(k): float(v) for k, v in (r.get("weights") or {}).items() }
                    })
            out["logic"] = None
            out["allocation"] = {"type":"rules","rules":rules}
    elif isinstance(lg, (str, type(None))):
        # accept string (signal id) or None and leave allocation as-is (maybe GPT set it directly)
        out["logic"] = lg

    alloc = js.get("allocation")
    if isinstance(alloc, dict):
        if alloc.get("type") == "rules" or "rules" in alloc:
            rules = []
            for r in alloc.get("rules", []):
                if "default" in r:
                    rules.append({"default": { _clean_symbol(k): float(v) for k, v in (r["default"] or {}).items() }})
                else:
                    rules.append({"when": r.get("when"), "weights": { _clean_symbol(k): float(v) for k, v in (r.get("weights") or {}).items() }})
            out["allocation"] = {"type":"rules","rules":rules}
        elif alloc.get("type") == "conditional_weights":
            wt_t = { _clean_symbol(k): float(v) for k, v in (alloc.get("when_true") or {}).items() }
            wt_f = { _clean_symbol(k): float(v) for k, v in (alloc.get("when_false") or {}).items() }
            out["allocation"] = {"type":"conditional_weights","when_true":wt_t,"when_false":wt_f}

    # rebalance
    rb = js.get("rebalance", {})
    if isinstance(rb, dict):
        f = str(rb.get("frequency", "M")).upper()
        out["rebalance"]["frequency"] = f if f in {"B", "W", "M"} else "M"

    return out

# ---------- UI building blocks ----------

def _weights_editor(label: str, tickers: List[str], value: Dict[str, float]):
    cols = st.columns(max(2, min(4, len(tickers))))
    newv = {}
    for i, t in enumerate(tickers):
        with cols[i % len(cols)]:
            newv[t] = st.number_input(f"{label} • {t}", value=float(value.get(t, 0.0)), step=0.05, min_value=0.0, max_value=1.0, key=f"{label}_{t}")
    s = sum(newv.values())
    if s and abs(s - 1.0) > 1e-6:
        st.info(f"Tip: weights sum to {s:.2f}. Most users target 1.00.")
    return newv

def _indicators_editor(universe: List[str], indicators: List[dict]):
    st.caption("Indicators")
    edited = []
    with st.container(border=True):
        count = st.number_input("How many indicators?", min_value=0, max_value=30, value=len(indicators))
        for i in range(int(count)):
            default = indicators[i] if i < len(indicators) else {}
            st.subheader(f"Indicator #{i+1}")
            c1, c2 = st.columns(2)
            with c1:
                typ = st.selectbox("Type", list(_ALLOWED_INDICATORS.keys()), index=0 if default.get("type") not in _ALLOWED_INDICATORS else list(_ALLOWED_INDICATORS.keys()).index(default["type"]), key=f"ind_type_{i}")
                sym = st.selectbox("Symbol", options=universe or ["SPY"], index=0, key=f"ind_sym_{i}")
            with c2:
                window = st.number_input("Window", min_value=1, max_value=1000, value=int(default.get("params", {}).get("window", 200)), key=f"ind_win_{i}")
                ddof = st.number_input("ddof (if applicable)", min_value=0, max_value=10, value=int(default.get("params", {}).get("ddof", 0)), key=f"ind_ddof_{i}")
            iid = st.text_input("ID", value=default.get("id", f"{typ}_{sym}_{window}")[:64], key=f"ind_id_{i}")
            params = {"symbol": sym, "window": int(window)}
            if "ddof" in _ALLOWED_INDICATORS[typ].get("opt", []):
                params["ddof"] = int(ddof)
            edited.append({"id": iid, "type": typ, "params": params})
    return edited

def _signals_editor(signals: List[dict], indicators: List[dict], universe: List[str]):
    st.caption("Signals")
    edited = []
    with st.container(border=True):
        count = st.number_input("How many signals?", min_value=0, max_value=30, value=len(signals))
        ind_ids = [i["id"] for i in indicators]
        for i in range(int(count)):
            default = signals[i] if i < len(signals) else {}
            st.subheader(f"Signal #{i+1}")
            typ = st.selectbox("Type", list(_ALLOWED_SIGNALS), index=0 if default.get("type") not in _ALLOWED_SIGNALS else list(_ALLOWED_SIGNALS).index(default["type"]), key=f"sig_type_{i}")
            # simple left/right pickers: indicator vs price vs constant
            tabs = st.tabs(["Left", "Right"])
            with tabs[0]:
                lkind = st.radio("Left side", ["indicator", "price", "const"], horizontal=True, key=f"sig_left_kind_{i}")
                if lkind == "indicator":
                    ref = st.selectbox("Indicator ref", ind_ids or ["—"], key=f"sig_left_ref_{i}")
                    left = {"kind": "indicator", "ref": ref}
                elif lkind == "price":
                    sym = st.selectbox("Price symbol", options=universe or ["SPY"], key=f"sig_left_price_{i}")
                    left = {"kind": "price", "symbol": sym}
                else:
                    val = st.number_input("Constant value", value=float(default.get("left", {}).get("value", 0.0)), key=f"sig_left_const_{i}")
                    left = {"kind": "const", "value": float(val)}
            with tabs[1]:
                rkind = st.radio("Right side", ["indicator", "price", "const"], horizontal=True, key=f"sig_right_kind_{i}")
                if rkind == "indicator":
                    ref = st.selectbox("Indicator ref ", ind_ids or ["—"], key=f"sig_right_ref_{i}")
                    right = {"kind": "indicator", "ref": ref}
                elif rkind == "price":
                    sym = st.selectbox("Price symbol ", options=universe or ["SPY"], key=f"sig_right_price_{i}")
                    right = {"kind": "price", "symbol": sym}
                else:
                    val = st.number_input("Constant value ", value=float(default.get("right", {}).get("value", 0.0)), key=f"sig_right_const_{i}")
                    right = {"kind": "const", "value": float(val)}
            sid = st.text_input("ID", value=default.get("id", f"{typ}_{i+1}")[:64], key=f"sig_id_{i}")
            edited.append({"id": sid, "type": typ, "left": left, "right": right})
    return edited

def _logic_and_alloc_editor(universe: List[str], signals: List[dict], logic: Union[None,str,dict], allocation: Dict[str, Any]):
    st.caption("Logic & Allocation")
    sig_ids = [s["id"] for s in signals]
    chosen = st.selectbox("Primary logic (signal id)", options=(sig_ids or [""]), index=0)
    logic_out = chosen if chosen else None

    mode = st.radio("Allocation type", ["conditional_weights", "rules"], horizontal=True, key="alloc_mode")
    if mode == "conditional_weights":
        st.markdown("**Weights when logic == TRUE (1)**")
        wt_true = _weights_editor("when_true", universe, (allocation.get("when_true") if allocation.get("type")=="conditional_weights" else {}) or {t:0.0 for t in universe})
        st.markdown("**Weights when logic == FALSE (0)**")
        wt_false = _weights_editor("when_false", universe, (allocation.get("when_false") if allocation.get("type")=="conditional_weights" else {}) or {t:0.0 for t in universe})
        alloc_out = {"type":"conditional_weights","when_true":wt_true,"when_false":wt_false}
    else:
        rules = allocation.get("rules", []) if allocation.get("type")=="rules" else []
        cnt = st.number_input("How many rules (including default)?", min_value=1, max_value=30, value=max(1, len(rules) or 1))
        built = []
        for i in range(int(cnt)):
            st.subheader(f"Rule #{i+1}")
            is_default = st.checkbox("Default rule (else)", value=("default" in (rules[i] if i < len(rules) else {})), key=f"rule_default_{i}")
            if is_default:
                w = _weights_editor(f"default_{i}", universe, (rules[i].get("default", {}) if i < len(rules) else {t:0.0 for t in universe}))
                built.append({"default": w})
            else:
                when = st.selectbox("When (signal id)", options=sig_ids or [""], index=0, key=f"rule_when_{i}")
                w = _weights_editor(f"weights_{i}", universe, (rules[i].get("weights", {}) if i < len(rules) else {t:0.0 for t in universe}))
                built.append({"when": when, "weights": w})
        alloc_out = {"type":"rules","rules":built}
    return logic_out, alloc_out


def _rebalance_editor(rb: dict):
    st.caption("Rebalance")
    freq = st.selectbox("Frequency", ["B", "W", "M"], index=["B","W","M"].index(rb.get("frequency","M")))
    return {"frequency": freq}

def _data_editor(data: dict):
    st.caption("Data")
    c1, c2, c3 = st.columns(3)
    with c1:
        src = st.selectbox("Source", ["yahoo"], index=0)
    with c2:
        start = st.text_input("Start (YYYY-MM-DD)", value=data.get("start","2015-01-01"))
    with c3:
        end = st.text_input("End (YYYY-MM-DD or empty)", value=data.get("end") or "")
    freq = st.selectbox("Calendar", ["B"], index=0)
    return {"source": src, "start": start, "end": (end or None), "frequency": freq}

def _costs_editor(costs: dict):
    st.caption("Costs")
    c1, c2 = st.columns(2)
    with c1:
        slp = st.number_input("Slippage (bps)", min_value=0, max_value=1000, value=int(costs.get("slippage_bps",0)))
    with c2:
        fee = st.number_input("Fee per trade ($)", min_value=0, max_value=100, value=int(costs.get("fee_per_trade",0)))
    return {"slippage_bps": slp, "fee_per_trade": fee}


REASONING_LEVEL = "low"

try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except Exception:
    _OPENAI_AVAILABLE = False


# ========================
# Data adapter (Yahoo)
# ========================

def _to_datetime(d):
    if d is None:
        return None
    if isinstance(d, str):
        return pd.to_datetime(d)
    return d

def fetch_yahoo(tickers: List[str],
                start: Optional[str] = None,
                end: Optional[str] = None) -> pd.DataFrame:
    import yfinance as yf
    start = _to_datetime(start)
    end = _to_datetime(end) or pd.Timestamp.today()

    # Pull whatever Yahoo has; don't enforce a start yet so we can detect IPO dates
    df = yf.download(tickers, start=start or "1900-01-01", end=end,
                     auto_adjust=True, progress=False, group_by="ticker")

    # Build a wide Close
    if isinstance(df.columns, pd.MultiIndex):
        out = {}
        for t in tickers:
            if (t, "Close") in df.columns:
                s = df[(t, "Close")].rename(t)
                if not s.dropna().empty:
                    out[t] = s
        wide = pd.concat(out, axis=1) if out else pd.DataFrame(index=df.index)
    else:
        # single ticker case
        wide = pd.DataFrame({tickers[0]: df["Close"]}) if "Close" in df else pd.DataFrame(index=df.index)

    wide.index = pd.to_datetime(wide.index)
    wide = wide.sort_index().dropna(how="all")

    # Drop tickers that have no data at all
    non_empty = [c for c in wide.columns if not wide[c].dropna().empty]
    wide = wide[non_empty]

    if wide.empty or len(non_empty) == 0:
        return wide  # nothing to do

    # Business-day calendar + ffill for alignment later
    wide = wide.asfreq("B").ffill()

    # Compute each symbol's first valid date, then pick the max (latest IPO among selected)
    firsts = wide.apply(lambda s: s.first_valid_index())
    # If any symbol is entirely NaN, firsts[c] will be None; drop those
    keep = [c for c in wide.columns if firsts[c] is not None]
    wide = wide[keep]
    if wide.empty:
        return wide

    common_start = max(firsts[c] for c in keep if firsts[c] is not None)

    # If the user gave a start, honor the later of (user_start, common_start)
    effective_start = max(start, common_start) if start is not None else common_start
    wide = wide[wide.index >= effective_start]

    # Trim by user end if provided
    if end is not None:
        wide = wide[wide.index <= end]

    # Final clean
    wide = wide.dropna(how="all", axis=0).dropna(how="all", axis=1)
    return wide


def get_prices(tickers: List[str], start: Optional[str], end: Optional[str], source: str = "yahoo") -> pd.DataFrame:
    if source.lower() not in ("yahoo", "yf"):
        raise ValueError("Only 'yahoo' source is supported in this app.")
    return fetch_yahoo(tickers, start, end)
    

# ========================
# Core containers & indicators (unchanged)
# ========================

ArrayLike = Union[pd.Series, pd.DataFrame]

@dataclass
class PricePanel:
    close: pd.DataFrame  # index = DatetimeIndex, columns = tickers

    @classmethod
    def from_wide_close(cls, close: pd.DataFrame) -> "PricePanel":
        if not isinstance(close.index, pd.DatetimeIndex):
            raise ValueError("close must be indexed by DatetimeIndex")
        return cls(close=close.sort_index())

    @classmethod
    def from_source(cls, tickers: List[str], start: Optional[str], end: Optional[str], source: str) -> "PricePanel":
        close = get_prices(tickers, start, end, source)
        return cls.from_wide_close(close)

    def returns(self) -> pd.DataFrame:
        return self.close.pct_change()

class Indicators:
    def __init__(self, panel: PricePanel):
        self.P = panel

    def price(self, sym: Optional[str] = None) -> ArrayLike:
        return self.P.close if sym is None else self.P.close[sym]

    def cumulative_return(self, sym: Optional[str] = None) -> ArrayLike:
        r = self.P.returns()
        cr = (1 + r).cumprod() - 1
        return cr if sym is None else cr[sym]

    def sma_price(self, window: int, sym: Optional[str] = None) -> ArrayLike:
        s = self.P.close.rolling(window).mean()
        return s if sym is None else s[sym]

    # EMA uses "window" for consistency
    def ema_price(self, window: int, sym: Optional[str] = None, adjust: bool = False) -> ArrayLike:
        e = self.P.close.ewm(span=window, adjust=adjust).mean()
        return e if sym is None else e[sym]

    def sma_return(self, window: int, sym: Optional[str] = None) -> ArrayLike:
        r = self.P.returns().rolling(window).mean()
        return r if sym is None else r[sym]

    def std_price(self, window: int, sym: Optional[str] = None, ddof: int = 0) -> ArrayLike:
        s = self.P.close.rolling(window).std(ddof=ddof)
        return s if sym is None else s[sym]

    def std_return(self, window: int, sym: Optional[str] = None, ddof: int = 0) -> ArrayLike:
        r = self.P.returns().rolling(window).std(ddof=ddof)
        return r if sym is None else r[sym]

    def rsi(self, window: int = 14, sym: Optional[str] = None) -> ArrayLike:
        c = self.P.close
        delta = c.diff()
        up = delta.clip(lower=0.0)
        dn = -delta.clip(upper=0.0)
        roll_up = up.ewm(alpha=1/window, adjust=False).mean()
        roll_dn = dn.ewm(alpha=1/window, adjust=False).mean()
        rs = roll_up / roll_dn.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi if sym is None else rsi[sym]

    def drawdown(self, sym: Optional[str] = None) -> ArrayLike:
        c = self.P.close if sym is None else self.P.close[[sym]]
        roll_max = c.cummax()
        dd = c / roll_max - 1
        return dd if sym is None else dd[sym]

    def max_drawdown(self, window: Optional[int] = None, sym: Optional[str] = None) -> ArrayLike:
        dd = self.drawdown() if sym is None else self.drawdown(sym)
        if window is None:
            return dd.min() if isinstance(dd, pd.Series) else dd.min(axis=0)
        if isinstance(dd, pd.Series):
            return dd.rolling(window).min()
        return dd.rolling(window).min()

# ========================
# Signals & allocation helpers (unchanged)
# ========================

def _to_series(x: Any, index: pd.DatetimeIndex) -> pd.Series:
    if isinstance(x, pd.Series):
        return x
    if isinstance(x, (int, float)):
        return pd.Series(x, index=index)
    raise ValueError("Signal operands must be Series or numeric constants.")

def rebalance_dates(idx: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    if freq == "B":
        return pd.Index(idx)
    if freq == "W":
        return idx.to_series().resample("W-FRI").last().dropna().index
    if freq == "M":
        return idx.to_series().resample("M").last().dropna().index
    raise ValueError("Unsupported rebalance frequency.")

@dataclass
class BrokerCosts:
    slippage_bps: float = 2.0
    fee_per_trade: float = 0.0

# ========================
# Validation & builders (unchanged, including 'rules' allocator)
# ========================

ALLOWED_INDICATORS = {
    "sma_price", "ema_price", "sma_return", "std_price", "std_return", "rsi", "drawdown", "max_drawdown"
}
ALLOWED_SIGNALS = {"gt", "lt", "cross_above", "cross_below"}
ALLOWED_REBAL = {"B", "W", "M"}

def validate_strategy(js: Dict[str, Any]) -> None:
    if not js.get("universe"):
        raise ValueError("universe must be a non-empty list of tickers.")
    data = js.get("data", {})
    if data.get("source", "yahoo").lower() not in ("yahoo", "yf"):
        raise ValueError("Only 'yahoo' is supported.")
    # indicators
    ids = set()
    for ind in js.get("indicators", []):
        t = ind.get("type")
        iid = ind.get("id")
        if t not in ALLOWED_INDICATORS:
            raise ValueError(f"Unsupported indicator: {t}")
        if iid in ids:
            raise ValueError(f"Duplicate indicator id: {iid}")
        ids.add(iid)
    # signals
    sids = set()
    for s in js.get("signals", []):
        t = s.get("type")
        sid = s.get("id")
        if t not in ALLOWED_SIGNALS:
            raise ValueError(f"Unsupported signal type: {t}")
        if sid in sids:
            raise ValueError(f"Duplicate signal id: {sid}")
        sids.add(sid)
    # allocation
    alloc = js.get("allocation", {})
    alloc_type = alloc.get("type")
    if alloc_type not in ("conditional_weights", "rules"):
        raise ValueError("allocation.type must be 'conditional_weights' or 'rules'.")
    if alloc_type == "rules":
        rules = alloc.get("rules", [])
        if not isinstance(rules, list) or len(rules) == 0:
            raise ValueError("allocation.rules must be a non-empty list.")
        has_default = any("default" in r for r in rules)
        has_when = any("when" in r for r in rules)
        if not (has_default or has_when):
            raise ValueError("rules must include at least one 'when' or a 'default'.")
    # rebalance
    freq = js.get("rebalance", {}).get("frequency", "M")
    if freq not in ALLOWED_REBAL:
        raise ValueError(f"Unsupported rebalance frequency: {freq}")

def build_indicators(js: Dict[str, Any], indicators: Indicators) -> Dict[str, ArrayLike]:
    out = {}
    for ind in js.get("indicators", []):
        iid, t, p = ind["id"], ind["type"], ind.get("params", {})
        if t == "sma_price":
            out[iid] = indicators.sma_price(p["window"], p["symbol"])
        elif t == "ema_price":
            out[iid] = indicators.ema_price(p["window"], p.get("symbol"), p.get("adjust", False))
        elif t == "sma_return":
            out[iid] = indicators.sma_return(p["window"], p["symbol"])
        elif t == "std_price":
            out[iid] = indicators.std_price(p["window"], p.get("symbol"), p.get("ddof", 0))
        elif t == "std_return":
            out[iid] = indicators.std_return(p["window"], p.get("symbol"), p.get("ddof", 0))
        elif t == "rsi":
            out[iid] = indicators.rsi(p.get("window", 14), p["symbol"])
        elif t == "drawdown":
            out[iid] = indicators.drawdown(p.get("symbol"))
        elif t == "max_drawdown":
            out[iid] = indicators.max_drawdown(p.get("window"), p.get("symbol"))
        else:
            raise ValueError(f"Unsupported indicator type: {t}")
    return out

def _resolve_source(node: Dict[str, Any], ind_map: Dict[str, ArrayLike], indicators: Indicators) -> ArrayLike:
    kind = node["kind"]
    if kind == "price":
        return indicators.price(node["symbol"])
    if kind == "indicator":
        ref = node["ref"]
        if ref not in ind_map:
            raise ValueError(f"Indicator ref not found: {ref}")
        return ind_map[ref]
    if kind == "const":
        return node["value"]
    raise ValueError(f"Unknown source kind: {kind}")

def build_signals(js: Dict[str, Any],
                  ind_map: Dict[str, ArrayLike],
                  indicators: Indicators,
                  index: pd.DatetimeIndex) -> Dict[str, pd.Series]:
    out = {}
    for s in js.get("signals", []):
        sid, typ = s["id"], s["type"]
        L = _resolve_source(s["left"], ind_map, indicators)
        R = _resolve_source(s["right"], ind_map, indicators)
        if typ in ("gt", "lt"):
            Ls = _to_series(L, index)
            Rs = _to_series(R, index) if not isinstance(R, (int, float)) else pd.Series(R, index=index)
            out[sid] = (Ls > Rs).astype(int) if typ == "gt" else (Ls < Rs).astype(int)
        elif typ in ("cross_above", "cross_below"):
            Ls = _to_series(L, index)
            Rs = _to_series(R, index)
            Ls, Rs = Ls.align(Rs, join="inner")
            if typ == "cross_above":
                prev = Ls.shift(1) <= Rs.shift(1)
                now = Ls > Rs
                out[sid] = (prev & now).astype(int).reindex(index).fillna(0).astype(int)
            else:
                prev = Ls.shift(1) >= Rs.shift(1)
                now = Ls < Rs
                out[sid] = (prev & now).astype(int).reindex(index).fillna(0).astype(int)
        else:
            raise ValueError(f"Unsupported signal type: {typ}")
    return out

def eval_logic(node: Union[str, Dict[str, Any]], signal_map: Dict[str, pd.Series]) -> pd.Series:
    if node is None:
        # optional if allocation.type == rules; return zeros
        some = next(iter(signal_map.values()))
        return pd.Series(0, index=some.index)
    if isinstance(node, str):
        if node not in signal_map:
            raise ValueError(f"Logic references unknown signal id: {node}")
        return signal_map[node].astype(int)
    t = node.get("type")
    if t == "and":
        s = None
        for ch in node["children"]:
            v = eval_logic(ch if isinstance(ch, dict) else ch, signal_map)
            s = v if s is None else ((s == 1) & (v == 1)).astype(int)
        return s
    if t == "or":
        s = None
        for ch in node["children"]:
            v = eval_logic(ch if isinstance(ch, dict) else ch, signal_map)
            s = v if s is None else ((s == 1) | (v == 1)).astype(int)
        return s
    if t == "not":
        v = eval_logic(node["child"], signal_map)
        return (1 - v).astype(int)
    raise ValueError(f"Unsupported logic type: {t}")

def conditional_weights(cond: pd.Series, w_true: Dict[str, float], w_false: Dict[str, float]) -> pd.DataFrame:
    idx = cond.index
    syms = sorted(set(list(w_true.keys()) + list(w_false.keys())))
    W = pd.DataFrame(0.0, index=idx, columns=syms)
    mask = cond.fillna(0).astype(int)
    for t in idx:
        W.loc[t, :] = w_true if mask.loc[t] == 1 else w_false
    return W.fillna(0.0)

def rules_weights(
    rules: List[dict],
    signal_map: Dict[str, pd.Series],
    index: pd.DatetimeIndex
) -> pd.DataFrame:
    # collect signal series
    sig_series: Dict[str, pd.Series] = {}
    for r in rules:
        if "when" in r:
            sid = r["when"]
            if sid not in signal_map:
                raise ValueError(f"rules_weights: unknown signal id '{sid}'")
            sig_series[sid] = signal_map[sid].reindex(index).fillna(0).astype(int)
    # universe from all branches
    syms = set()
    default_w = None
    for r in rules:
        if "weights" in r:
            syms |= set(r["weights"].keys())
        if "default" in r:
            default_w = r["default"]
            syms |= set(default_w.keys())
    syms = sorted(syms)
    W = pd.DataFrame(0.0, index=index, columns=syms)
    for t in index:
        applied = False
        for r in rules:
            if "when" in r:
                if sig_series[r["when"]].loc[t] == 1:
                    for k, v in r["weights"].items():
                        W.at[t, k] = float(v)
                    applied = True
                    break
        if not applied and default_w is not None:
            for k, v in default_w.items():
                W.at[t, k] = float(v)
    return W.fillna(0.0)

# ========================
# Backtester & Metrics (unchanged)
# ========================

def backtest(close: pd.DataFrame,
             weights: pd.DataFrame,
             costs: BrokerCosts,
             freq: str,
             initial_cash: float = 10_000.0) -> Dict[str, Any]:
    idx = close.index
    syms = list(weights.columns)
    rb_dates = rebalance_dates(idx, freq)
    positions = {s: 0.0 for s in syms}
    cash = initial_cash
    equity = []
    for dt in idx:
        px = {s: float(close.loc[dt, s]) for s in syms}
        port_val = cash + sum(positions[s] * px[s] for s in syms)
        equity.append((dt, port_val))
        if dt not in rb_dates:
            continue
        target_w = weights.loc[dt].fillna(0.0)
        target_val = {s: float(target_w[s] * port_val) for s in syms}
        trades = {}
        for s in syms:
            cur_val = positions[s] * px[s]
            delta_val = target_val[s] - cur_val
            qty = delta_val / px[s] if px[s] != 0 else 0.0
            trades[s] = qty
        traded_notional = sum(abs(trades[s]) * px[s] for s in syms)
        slip = traded_notional * (costs.slippage_bps / 10_000)
        fees = costs.fee_per_trade * sum(1 for s in syms if abs(trades[s]) > 1e-9)
        cash -= (slip + fees)
        for s in syms:
            positions[s] += trades[s]
            cash -= trades[s] * px[s]
    eq = pd.Series(dict(equity)).sort_index()
    rets = eq.pct_change().fillna(0.0)
    return {"equity": eq, "returns": rets}

def annualized_return(equity: pd.Series) -> float:
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years <= 0:
        return 0.0
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)

def sharpe_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    if returns.std(ddof=0) == 0:
        return 0.0
    ann_mean = returns.mean() * 252
    ann_std = returns.std(ddof=0) * np.sqrt(252)
    return float((ann_mean - rf) / ann_std)

def max_drawdown_series(equity: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp, pd.Series]:
    roll = equity.cummax()
    dd = equity / roll - 1
    mdd = dd.min()
    end = dd.idxmin()
    start = equity.loc[:end].idxmax()
    return float(mdd), start, end, dd

def calmar_ratio(equity: pd.Series) -> float:
    ar = annualized_return(equity)
    mdd, _, _, _ = max_drawdown_series(equity)
    return float(ar / abs(mdd)) if mdd != 0 else np.inf

def trailing_return(equity: pd.Series, days: int) -> float:
    if len(equity) <= days:
        return np.nan
    return float(equity.iloc[-1] / equity.iloc[-(days + 1)] - 1)

def alpha_beta_r2(strategy_ret: pd.Series, bench_ret: pd.Series) -> Tuple[float, float, float, float]:
    a = strategy_ret.align(bench_ret, join="inner")[0]
    b = bench_ret.align(strategy_ret, join="inner")[0]
    X = np.vstack([np.ones(len(b)), b.values]).T
    y = a.values
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    alpha, beta = coeffs[0], coeffs[1]
    yhat = X @ coeffs
    ss_res = ((y - yhat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    corr = np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else np.nan
    return float(alpha * 252), float(beta), float(r2), float(corr)

# ========================
# Plot helpers (unchanged)
# ========================

def plot_equity(eq: pd.Series, bench_eq: pd.Series, name: str):
    base = eq.iloc[0]
    base_b = bench_eq.iloc[0]
    eq_n = (eq / base).rename(name)
    bench_n = (bench_eq / base_b).rename("Benchmark (SPY)")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=eq_n.index, y=eq_n.values, mode="lines", name=name,
        hovertemplate="%{x|%Y-%m-%d}<br>Equity: %{y:.3f}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=bench_n.index, y=bench_n.values, mode="lines", name="Benchmark (SPY)",
        line=dict(color="red"),
        hovertemplate="%{x|%Y-%m-%d}<br>Equity: %{y:.3f}<extra></extra>"
    ))
    fig.update_layout(
        title="Equity Curve (normalized)",
        xaxis_title="Date",
        yaxis_title="Value (x starting equity)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(rangeslider=dict(visible=True))
    st.plotly_chart(fig, use_container_width=True)


def plot_drawdown(eq: pd.Series):
    mdd, s, e, dd = max_drawdown_series(eq)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd.values, mode="lines", name="Drawdown",
        hovertemplate="%{x|%Y-%m-%d}<br>DD: %{y:.2%}<extra></extra>"
    ))
    fig.update_layout(
        title="Drawdown",
        xaxis_title="Date",
        yaxis_title="Drawdown",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(tickformat=".0%")
    fig.update_xaxes(rangeslider=dict(visible=True))
    st.plotly_chart(fig, use_container_width=True)

# ========================
# NEW: GPT utilities
# ========================

def _reasoning_payload(level: str) -> Dict[str, Any]:
    # Some GPT-5 Thinking endpoints support a "reasoning" effort knob.
    # We keep this backend-only per your request.
    level = (level or "medium").lower()
    if level not in {"minimal", "low", "medium", "high"}:
        level = "medium"
    return {"effort": level}

# somewhere in Strat.py or a small `llm.py` you import
from openai import OpenAI

def call_openai_to_json(user_prompt: str, system_prompt: str, model: str, reasoning: Literal["low","medium","high"]="low") -> dict:
    client = OpenAI()
    resp = client.responses.create(
        model=model,
        reasoning={"effort": reasoning},
        input=[
            {"role":"system","content":system_prompt},
            {"role":"user","content":user_prompt}
        ],
        max_output_tokens=5000,  # keep tight since it must be JSON
    )
    text = resp.output_text.strip()
    # sometimes models add code fences—strip safely
    if text.startswith("```"):
        text = text.strip("`")
        # remove language hints like ```json
        text = text.split("\n",1)[1] if "\n" in text else text
    try:
        return json.loads(text)
    except Exception:
        # last-ditch: try to find the first/last braces
        l = text.find("{"); r = text.rfind("}")
        if l!=-1 and r!=-1 and r>l:
            return json.loads(text[l:r+1])
        raise



# ========================
# Streamlit App (UI wiring)
# ========================

st.set_page_config(page_title="Strategy Backtester", layout="wide")
st.title("Strategy Backtester")
if "block_state" not in st.session_state:
    st.session_state["block_state"] = None

if "step" not in st.session_state:
    st.session_state.step = 1

if st.session_state.step == 1:
    st.subheader("Overview")
    st.markdown(FEATURES)
    st.subheader("Limitations")
    st.markdown(LIMITATIONS)
    if st.button("Continue"):
        st.session_state.step = 2
        st.rerun()

elif st.session_state.step == 2:
    st.header("Build your strategy")

    tab1, tab2, tab3 = st.tabs(["Describe with GPT", "Build with Blocks", "Paste JSON"])

    with tab1:
        st.markdown("Describe the strategy in English. We'll ask GPT to generate JSON and convert it to blocks you can edit.")
        prompt = st.text_area("Strategy description", height=160, placeholder="e.g., 50/50 AAPL/TSLA; hedge to BIL when either stock is below its 200-day SMA. Weekly rebalance.")
        use_model = MODEL_NAME
        c1, c2 = st.columns([1,1])
        with c1:
            run_gpt = st.button("Generate & Convert to Blocks")
        with c2:
            st.caption(f"Model: {use_model} · Reasoning: {REASONING_LEVEL}")

        if run_gpt and prompt.strip():
            try:
                # ---- your existing OpenAI call lives in a function like this: ----
                # Replace with your Responses API call if you’ve already migrated.
                js_raw = call_openai_to_json(prompt, system_prompt=GPT_SYSTEM_PROMPT, model=use_model, reasoning=REASONING_LEVEL)
                js = normalize_strategy_json(js_raw)
                st.session_state["block_state"] = js
                st.success("Converted GPT JSON into editable blocks.")
            except Exception as e:
                st.error(f"GPT generation failed: {e}")

    with tab2:
        st.markdown("Build the strategy using blocks. No GPT.")
        # if user already generated or pasted, preload; else a small starter
        base = st.session_state.get("block_state") or {
            "version":"1.0",
            "meta":{"name":"My Strategy","notes":""},
            "universe":["AAPL","TSLA","BIL"],
            "data":{"source":"yahoo","start":"2018-01-01","end":None,"frequency":"B"},
            "costs":{"slippage_bps":2,"fee_per_trade":0},
            "indicators":[],
            "signals":[],
            "logic": None,
            "allocation": {"type":"conditional_weights","when_true":{},"when_false":{}},
            "rebalance":{"frequency":"W"},
        }
        st.subheader("Meta")
        base["meta"]["name"] = st.text_input("Name", value=base["meta"].get("name",""))
        base["meta"]["notes"] = st.text_area("Notes", value=base["meta"].get("notes",""), height=80)

        st.subheader("Universe")
        uni_str = st.text_input("Tickers (comma-separated)", value=",".join(base.get("universe") or []))
        base["universe"] = [_clean_symbol(x) for x in uni_str.split(",") if x.strip()]

        base["data"] = _data_editor(base.get("data", {}))
        base["costs"] = _costs_editor(base.get("costs", {}))

        base["indicators"] = _indicators_editor(base["universe"], base.get("indicators", []))
        base["signals"] = _signals_editor(base.get("signals", []), base["indicators"], base["universe"])
        logic_out, alloc_out = _logic_and_alloc_editor(base["universe"], base["signals"], base.get("logic", None), base.get("allocation", {}))
        base["logic"] = logic_out
        base["allocation"] = alloc_out
        base["rebalance"] = _rebalance_editor(base.get("rebalance", {}))


        st.session_state["block_state"] = base
        st.info("Blocks are live. When you backtest, we'll compile to JSON on the fly.")

    with tab3:
        st.markdown("Paste a strategy JSON; we’ll convert it to blocks you can edit.")
        txt = st.text_area("Paste JSON here", height=180, placeholder='{"version":"1.0",...}')
        if st.button("Convert JSON to Blocks"):
            try:
                js_raw = json.loads(txt)
                js = normalize_strategy_json(js_raw)
                st.session_state["block_state"] = js
                st.success("Converted JSON to blocks.")
            except Exception as e:
                st.error(f"Error: {e}")
    
    st.divider()
    if st.button("Run Backtest"):
        try:
            assert st.session_state.get("block_state") is not None, "Nothing to backtest. Use one of the three tabs first."
            js = normalize_strategy_json(st.session_state["block_state"])

            universe = list(js["universe"])
            bench = "SPY"
            tk = universe if bench in universe else universe + [bench]

            start = js.get("data", {}).get("start")
            end = js.get("data", {}).get("end")

            close_all = get_prices(tk, start, end, source=js.get("data", {}).get("source","yahoo"))
            if close_all.empty:
                st.error("No data returned. Check tickers and dates.")
                st.stop()
            
            # Inform the user if we had to shift the start or drop symbols
            effective_first = {c: close_all[c].first_valid_index() for c in close_all.columns}
            latest = max([dt for dt in effective_first.values() if dt is not None])

            if start is None or pd.to_datetime(start) < latest:
                st.info(f"Adjusted start to {latest.date()} to match earliest available data across your tickers.")

            missing = [sym for sym in tk if sym not in close_all.columns]
            if missing:
                st.warning(f"Dropped symbols with no data: {', '.join(missing)}")


            validate_strategy(js)
            panel = PricePanel.from_wide_close(close_all[universe])
            ind = Indicators(panel)

            ind_map = build_indicators(js, ind)
            index = panel.close.index
            sig_map = build_signals(js, ind_map, ind, index)

            # Build final_signal safely even if there are no signals yet
            logic_node = js.get("logic")
            if not sig_map:
                final_signal = pd.Series(0, index=index)  # treat as always false
            else:
                final_signal = eval_logic(logic_node, sig_map).reindex(index).fillna(0).astype(int)


            alloc = js["allocation"]
            if alloc.get("type") == "conditional_weights":
                weights = conditional_weights(final_signal, alloc.get("when_true", {}), alloc.get("when_false", {}))
            elif alloc.get("type") == "rules":
                weights = rules_weights(alloc.get("rules", []), sig_map, index)
            else:
                raise ValueError("Unsupported allocation type.")

            costs = js.get("costs", {})
            bt = backtest(
                close=panel.close[weights.columns].reindex(index).ffill(),
                weights=weights.reindex(index).fillna(0.0),
                costs=BrokerCosts(
                    slippage_bps=float(costs.get("slippage_bps", 2.0)),
                    fee_per_trade=float(costs.get("fee_per_trade", 0.0)),
                ),
                freq=js.get("rebalance", {}).get("frequency", "M"),
                initial_cash=10_000.0,
            )
            eq = bt["equity"]
            rets = bt["returns"]

            bench_close = close_all[[bench]].reindex(index).ffill()
            bench_eq = 10_000.0 * (bench_close[bench] / bench_close[bench].iloc[0])

            st.subheader("Results")
            c1, c2, c3, c4 = st.columns(4)
            ann = annualized_return(eq)
            vol = (rets.std(ddof=0) * (252 ** 0.5))
            shrp = sharpe_ratio(rets)
            mdd, _, _, _ = max_drawdown_series(eq)
            c1.metric("Return (annualized)", f"{ann*100:,.2f}%")
            c2.metric("Vol (annualized)", f"{vol*100:,.2f}%")
            c3.metric("Sharpe", f"{shrp:,.2f}")
            c4.metric("Max DD", f"{mdd*100:,.2f}%")

            st.subheader("Charts")
            plot_equity(eq, bench_eq, js.get("meta", {}).get("name", "Strategy"))
            plot_drawdown(eq)

            with st.expander("Show target weights (last 10 rows)"):
                st.dataframe(weights.tail(10).style.format("{:.2%}"))
        except Exception as e:
            st.error(f"Error: {e}")
'''