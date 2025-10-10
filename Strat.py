# strat.py
import json
from dataclasses import dataclass
from typing import Dict, Union, Optional, List, Tuple, Any, Literal

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from configuration import (
    FEATURES,
    LIMITATIONS,
    MODEL_NAME,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    LLM_REASONING,
    DEFAULTS,
    ALLOWED,
    HEDGE_FALLBACKS,
    DSL_JSON_SCHEMA,
    DSL_PARSE_PROMPT,
)

# --- OpenAI Responses API (single-call DSL parse) ---
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
    if isinstance(d, str) and d.strip() == "":
        return None
    if isinstance(d, str):
        return pd.to_datetime(d)
    return d

def fetch_yahoo(tickers: List[str],
                start: Optional[str] = None,
                end: Optional[str] = None) -> pd.DataFrame:
    try:
        import yfinance as yf
    except Exception as e:
        raise RuntimeError("Missing dependency: yfinance. Install it with `pip install yfinance`.") from e

    start = _to_datetime(start)
    end = _to_datetime(end) or pd.Timestamp.today()

    df = yf.download(tickers, start=start or "1900-01-01", end=end,
                     auto_adjust=True, progress=False, group_by="ticker")

    if isinstance(df.columns, pd.MultiIndex):
        out = {}
        for t in tickers:
            if (t, "Close") in df.columns:
                s = df[(t, "Close")].rename(t)
                if not s.dropna().empty:
                    out[t] = s
        wide = pd.concat(out, axis=1) if out else pd.DataFrame(index=df.index)
    else:
        wide = pd.DataFrame({tickers[0]: df["Close"]}) if "Close" in df else pd.DataFrame(index=df.index)

    wide.index = pd.to_datetime(wide.index)
    wide = wide.sort_index().dropna(how="all")

    non_empty = [c for c in wide.columns if not wide[c].dropna().empty]
    wide = wide[non_empty]
    if wide.empty or len(non_empty) == 0:
        return wide

    wide = wide.asfreq("B").ffill()

    firsts = wide.apply(lambda s: s.first_valid_index())
    keep = [c for c in wide.columns if firsts[c] is not None]
    wide = wide[keep]
    if wide.empty:
        return wide

    common_start = max(firsts[c] for c in keep if firsts[c] is not None)
    effective_start = max(start, common_start) if start is not None else common_start
    wide = wide[wide.index >= effective_start]

    if end is not None:
        wide = wide[wide.index <= end]

    wide = wide.dropna(how="all", axis=0).dropna(how="all", axis=1)
    return wide


# ========================
# Core containers & indicators
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

    def returns(self) -> pd.DataFrame:
        return self.close.pct_change()

class Indicators:
    def __init__(self, panel: PricePanel):
        self.P = panel

    def price(self, sym: Optional[str] = None) -> ArrayLike:
        return self.P.close if sym is None else self.P.close[sym]

    def sma_price(self, window: int, sym: Optional[str] = None) -> ArrayLike:
        s = self.P.close.rolling(window).mean()
        return s if sym is None else s[sym]

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
# Signals & allocation helpers
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
# Engine JSON validation
# ========================

ALLOWED_INDICATORS = set(ALLOWED["indicators"])
ALLOWED_SIGNALS = set(ALLOWED["signals"])
ALLOWED_REBAL = set(ALLOWED["rebalance"])

def _sum_close_to_one(d: Dict[str, float]) -> bool:
    return abs(sum(d.values()) - 1.0) <= 1e-3

def validate_strategy(js: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if not js.get("universe"):
        errors.append("universe must be a non-empty list of tickers.")

    data = js.get("data", {})
    if data.get("source", "yahoo").lower() not in ("yahoo","yf"):
        errors.append("Only 'yahoo' data source is supported.")

    ids = set()
    for ind in js.get("indicators", []):
        t = ind.get("type")
        iid = ind.get("id")
        if t not in ALLOWED_INDICATORS:
            errors.append(f"Unsupported indicator: {t}")
        if iid in ids:
            errors.append(f"Duplicate indicator id: {iid}")
        ids.add(iid)

    sids = set()
    valid_refs = {i["id"] for i in js.get("indicators", [])}
    for s in js.get("signals", []):
        t = s.get("type")
        sid = s.get("id")
        if t not in ALLOWED_SIGNALS:
            errors.append(f"Unsupported signal type: {t}")
        if sid in sids:
            errors.append(f"Duplicate signal id: {sid}")
        sids.add(sid)
        for side in ("left","right"):
            node = s.get(side, {})
            if not isinstance(node, dict) or "kind" not in node:
                errors.append(f"Signal {sid} side '{side}' missing kind")
                continue
            if node["kind"] == "indicator" and node.get("ref") not in valid_refs:
                errors.append(f"Signal {sid} references unknown indicator: {node.get('ref')}")
            if node["kind"] == "price" and node.get("symbol") not in js.get("universe", []):
                errors.append(f"Signal {sid} price symbol not in universe: {node.get('symbol')}")
            if node["kind"] == "const" and not isinstance(node.get("value"), (int,float)):
                errors.append(f"Signal {sid} const value invalid")

    alloc = js.get("allocation", {})
    alloc_type = alloc.get("type")
    if alloc_type not in ("conditional_weights","rules"):
        errors.append("allocation.type must be 'conditional_weights' or 'rules'.")
    universe = set(js.get("universe", []))
    if alloc_type == "conditional_weights":
        for branch in ("when_true","when_false"):
            w = alloc.get(branch, {})
            if set(w.keys()) - universe:
                errors.append(f"allocation.{branch} contains tickers not in universe: {list(set(w.keys())-universe)}")
            if w and not _sum_close_to_one(w):
                errors.append(f"allocation.{branch} weights must sum to 1.000 (±0.001); got {sum(w.values()):.3f}")
    elif alloc_type == "rules":
        rules = alloc.get("rules", [])
        if not isinstance(rules, list) or len(rules) == 0:
            errors.append("allocation.rules must be a non-empty list.")
        has_any = False
        for r in rules:
            if "weights" in r:
                has_any = True
                w = r["weights"]
                if set(w.keys()) - universe:
                    errors.append("rules.weights contains tickers not in universe")
                if w and not _sum_close_to_one(w):
                    errors.append("Each rule.weights must sum to 1.000 (±0.001).")
            if "default" in r:
                has_any = True
                w = r["default"]
                if set(w.keys()) - universe:
                    errors.append("rules.default contains tickers not in universe")
                if w and not _sum_close_to_one(w):
                    errors.append("Default weights must sum to 1.000 (±0.001).")
        if not has_any:
            errors.append("rules must include a 'weights' or 'default' branch.")

    freq = js.get("rebalance", {}).get("frequency", "W")
    if freq not in ALLOWED_REBAL:
        errors.append(f"Unsupported rebalance frequency: {freq}")

    start = data.get("start")
    end = data.get("end")
    try:
        if start is not None:
            _ = pd.to_datetime(start)
        if end is not None:
            _ = pd.to_datetime(end)
        if start and end and pd.to_datetime(start) > pd.to_datetime(end):
            errors.append("data.start must be <= data.end")
    except Exception:
        errors.append("Invalid start/end date format")
    return errors


# ========================
# Engine builders
# ========================

def _round_norm(weights: Dict[str, float]) -> Dict[str, float]:
    if not weights:
        return {}
    w = {k: float(v) for k, v in weights.items()}
    s = sum(w.values())
    if s == 0:
        n = len(w)
        if n == 0:
            return {}
        eq = {k: round(1.0/n, 3) for k in w}
        diff = round(1.000 - sum(eq.values()), 3)
        if diff != 0:
            k0 = next(iter(eq.keys()))
            eq[k0] = round(eq[k0] + diff, 3)
        return eq
    w = {k: v / s for k, v in w.items()}
    w = {k: round(v, 3) for k, v in w.items()}
    diff = round(1.000 - sum(w.values()), 3)
    if abs(diff) > 0:
        k0 = next(iter(w.keys()))
        w[k0] = round(w[k0] + diff, 3)
    return w

def _equal_weights(tickers: List[str]) -> Dict[str, float]:
    if not tickers:
        return {}
    n = len(tickers)
    base = round(1.0 / n, 3)
    w = {t: base for t in tickers}
    diff = round(1.000 - sum(w.values()), 3)
    if diff != 0:
        t0 = tickers[0]
        w[t0] = round(w[t0] + diff, 3)
    return w


# ========================
# Backtester & Metrics
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

        if dt in rb_dates:
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
            port_val = cash + sum(positions[s] * px[s] for s in syms)

        equity.append((dt, port_val))
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


# ========================
# Plot helpers
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
# LLM helper (single DSL parse)
# ========================

def _parse_responses_output(resp) -> str:
    text = ""
    try:
        text = resp.output_text or ""
    except Exception:
        pass
    if not text:
        try:
            parts = []
            for item in getattr(resp, "output", []) or []:
                for c in getattr(item, "content", []) or []:
                    if getattr(c, "type", "") == "output_text" and getattr(c, "text", None):
                        parts.append(c.text)
            text = "".join(parts).strip()
        except Exception:
            text = ""
    return text or ""

def llm_parse_dsl(user_text: str) -> Tuple[Optional[dict], Optional[str]]:
    """
    Returns (dsl_dict_or_none, error_str_or_none). Strict JSON schema enforced.
    """
    if not _OPENAI_AVAILABLE:
        return None, "OpenAI SDK not available."

    try:
        client = OpenAI()
        resp = client.responses.create(
            model=MODEL_NAME,
            reasoning={"effort": LLM_REASONING},
            input=[
                {"role": "system", "content": DSL_PARSE_PROMPT},
                {"role": "user", "content": user_text.strip()}
            ],
            max_output_tokens=LLM_MAX_TOKENS,
        )
        text = _parse_responses_output(resp).strip()
        obj = json.loads(text)
        return obj, None
    except Exception as e:
        return None, f"LLM parse error: {e}"


# ========================
# DSL → Engine compiler (deterministic)
# ========================

class CompileError(Exception):
    pass

def _wildcard_symbols(side: dict) -> bool:
    return side.get("symbol") == "*"

def _needed_indicator_nodes(sig_side: dict) -> Optional[Tuple[str,int,bool]]:
    # Return (name, window, adjust) if indicator, else None
    if sig_side.get("kind") != "indicator":
        return None
    name = sig_side.get("name")
    window = sig_side.get("window")
    adjust = bool(sig_side.get("adjust")) if sig_side.get("adjust") is not None else False
    if name not in ALLOWED_INDICATORS:
        raise CompileError(f"Indicator '{name}' not supported.")
    if name in ("sma_price","ema_price","sma_return","std_price","std_return","rsi") and window is None:
        raise CompileError(f"Indicator '{name}' requires a window.")
    return (name, window, adjust)

def _expand_signals(dsl: dict, universe: List[str]) -> Tuple[List[dict], Dict[str, List[str]], List[dict]]:
    """
    Expand wildcard signals per symbol and build indicator list.
    Returns (signals_out, expansion_map, indicators_out)
    expansion_map maps original dsl signal id -> list of expanded signal ids
    """
    indicators: List[dict] = []
    ind_seen = set()
    out_signals: List[dict] = []
    expansion_map: Dict[str, List[str]] = {}

    for s in dsl.get("signals", []):
        sid = s["id"]
        typ = s["type"]
        fold = s.get("fold", "or")
        L = s["left"]
        R = s["right"]

        expand_L = _wildcard_symbols(L)
        expand_R = _wildcard_symbols(R)
        if expand_L or expand_R:
            expansion_map[sid] = []
            for sym in universe:
                def side_to_ref(side):
                    if side["kind"] == "price":
                        return {"kind":"price","symbol":sym}
                    if side["kind"] == "indicator":
                        name, window, adjust = _needed_indicator_nodes(side)
                        ind_id = f"{name}_{sym}_{window}" if window is not None else f"{name}_{sym}"
                        if ind_id not in ind_seen:
                            ind_seen.add(ind_id)
                            params = {"symbol": sym}
                            if window is not None:
                                params["window"] = int(window)
                            if name == "ema_price" and side.get("adjust") is not None:
                                params["adjust"] = bool(adjust)
                            indicators.append({"id": ind_id, "type": name, "params": params})
                        return {"kind":"indicator","ref": ind_id}
                    if side["kind"] == "const":
                        return {"kind":"const","value": float(side["value"])}
                    raise CompileError("Unknown side kind in wildcard expansion.")

                Lref = side_to_ref(L)
                Rref = side_to_ref(R)
                new_id = f"{sid}_{sym}"
                out_signals.append({"id": new_id, "type": typ, "left": Lref, "right": Rref, "fold": fold})
                expansion_map[sid].append(new_id)
        else:
            # No wildcard expansion
            def ensure_side(side):
                if side["kind"] == "indicator":
                    name, window, adjust = _needed_indicator_nodes(side)
                    sym = side.get("symbol")
                    if sym not in universe and sym is not None:
                        raise CompileError(f"Indicator symbol {sym} not in universe.")
                    ind_id = f"{name}_{sym}_{window}" if window is not None else f"{name}_{sym}"
                    if ind_id not in ind_seen:
                        ind_seen.add(ind_id)
                        params = {"symbol": sym} if sym else {}
                        if window is not None:
                            params["window"] = int(window)
                        if name == "ema_price" and side.get("adjust") is not None:
                            params["adjust"] = bool(adjust)
                        indicators.append({"id": ind_id, "type": name, "params": params})
                    return {"kind":"indicator","ref": ind_id}
                if side["kind"] == "price":
                    sym = side.get("symbol")
                    if sym not in universe:
                        raise CompileError(f"Price symbol {sym} not in universe.")
                    return {"kind":"price","symbol": sym}
                if side["kind"] == "const":
                    return {"kind":"const","value": float(side["value"])}
                raise CompileError("Unknown side kind.")

            Lref = ensure_side(L)
            Rref = ensure_side(R)
            out_signals.append({"id": sid, "type": typ, "left": Lref, "right": Rref, "fold": s.get("fold","or")})

    return out_signals, expansion_map, indicators

def _expand_logic(node: Union[str, dict], expand_map: Dict[str, List[str]]) -> Union[str, dict]:
    if node is None:
        return None
    if isinstance(node, str):
        if node in expand_map:
            # Default fold is OR for expansions unless caller builds higher-level ANDs explicitly
            ids = expand_map[node]
            return {"type":"or","children": ids} if len(ids) > 1 else ids[0]
        return node
    t = node.get("type")
    if t in ("and","or"):
        kids = []
        for ch in node.get("children", []):
            kids.append(_expand_logic(ch, expand_map))
        return {"type": t, "children": kids}
    if t == "not":
        return {"type":"not","child": _expand_logic(node.get("child"), expand_map)}
    return node

def _choose_rebalance(dsl: dict) -> str:
    hint = dsl.get("rebalance_hint","none")
    if hint in ("B","W","M"):
        return hint
    rg = dsl.get("risk_goal","balanced")
    if rg == "conservative":
        return "M"
    return "W"

def _choose_dates(dsl: dict) -> Dict[str, Any]:
    d = dsl.get("dates", {}) or {}
    start = d.get("start") or DEFAULTS["start_date"]
    end = d.get("end", None)
    return {"source":"yahoo","start":start,"end":end,"frequency":"B"}

def _choose_allocation(dsl: dict, universe: List[str]) -> dict:
    pol = dsl.get("allocation_policy", {"type":"auto"})
    hedge = dsl.get("assets",{}).get("hedge_assets", []) or []
    # If hedging mentioned and hedge assets exist, use conditional_weights.
    if pol.get("type") in ("auto","conditional_weights") and hedge:
        wt_true: Dict[str,float]
        wtrue = pol.get("when_true","equal")
        if isinstance(wtrue, dict):
            wt_true = _round_norm(wtrue)
        elif wtrue == "core_satellite":
            core = [t for t in pol.get("core",[]) if t in universe]
            sat = [t for t in pol.get("satellite",[]) if t in universe and t not in core]
            if not core:
                core = universe[: max(1, len(universe)//2)]
            wt_core = 0.7
            wt_sat = 0.3
            c_eq = _equal_weights(core)
            s_eq = _equal_weights(sat or [t for t in universe if t not in core])
            wt_true = {k: round(v*wt_core,3) for k,v in c_eq.items()}
            for k,v in s_eq.items():
                wt_true[k] = round(wt_true.get(k,0)+ v*wt_sat, 3)
            wt_true = _round_norm(wt_true)
        else:
            wt_true = _equal_weights(universe)

        wfalse = pol.get("when_false","hedge_all")
        if isinstance(wfalse, dict):
            wt_false = _round_norm(wfalse)
        elif wfalse == "hedge_all":
            h = hedge[0] if hedge else (HEDGE_FALLBACKS[0] if HEDGE_FALLBACKS[0] in universe else None)
            if h is None:
                # Strict: no auto-adding assets. Force user edit.
                raise CompileError("Hedging requested but no hedge_assets present in universe.")
            wt_false = {h: 1.0}
        else:
            wt_false = _equal_weights(universe)

        return {"type":"conditional_weights","when_true": wt_true, "when_false": _round_norm(wt_false)}

    # Otherwise simple rules with default equal (or user weights)
    if pol.get("type") == "rules":
        rules = []
        if isinstance(pol.get("when_true"), dict):
            rules.append({"weights": _round_norm({k:v for k,v in pol["when_true"].items() if k in universe})})
        rules.append({"default": _equal_weights(universe)})
        return {"type":"rules","rules": rules}

    return {"type":"rules","rules":[{"default": _equal_weights(universe)}]}

def compile_dsl_to_engine(dsl: dict) -> dict:
    # 1) Strict coverage check
    clause_ids = {c["id"] for c in dsl.get("clauses",[])}
    coverage = dsl.get("coverage",{})
    missing_cov = [cid for cid in clause_ids if cid not in coverage or len(coverage[cid])==0]
    if missing_cov:
        raise CompileError(f"Uncovered clauses: {missing_cov}. Every sentence must map to at least one node id.")

    # 2) Universe
    universe = list(dict.fromkeys(dsl.get("assets",{}).get("tickers",[])))  # preserve order, dedupe
    if not universe:
        raise CompileError("Empty universe in DSL.")
    # Strict: do not auto-append hedge assets; they must be part of tickers if used in allocation.

    # 3) Signals & Indicators expansion
    signals_out, expand_map, indicators = _expand_signals(dsl, universe)

    # 4) Logic expansion
    logic_in = dsl.get("logic")
    logic_out = _expand_logic(logic_in, expand_map)

    # 5) Allocation
    allocation = _choose_allocation(dsl, universe)

    # 6) Ops
    data = _choose_dates(dsl)
    rebalance = {"frequency": _choose_rebalance(dsl)}
    costs = {"slippage_bps": 2, "fee_per_trade": 0}

    final = {
        "version": "1.0",
        "meta": {"name": "Compiled Strategy", "notes": ""},
        "universe": universe,
        "data": data,
        "costs": costs,
        "indicators": indicators,
        "signals": [{"id": s["id"], "type": s["type"], "left": s["left"], "right": s["right"]} for s in signals_out],
        "logic": logic_out,
        "allocation": allocation,
        "rebalance": rebalance,
    }

    # Final validation (structural)
    errs = validate_strategy(final)
    if errs:
        raise CompileError(" ; ".join(errs))
    return final


# ========================
# Logic evaluation & weights (engine runtime)
# ========================

def build_indicators(js: Dict[str, Any], indicators: Indicators) -> Dict[str, ArrayLike]:
    out = {}
    for ind in js.get("indicators", []):
        iid, t, p = ind["id"], ind["type"], ind.get("params", {})
        if t == "sma_price":
            out[iid] = indicators.sma_price(int(p["window"]), p["symbol"])
        elif t == "ema_price":
            out[iid] = indicators.ema_price(int(p["window"]), p.get("symbol"), p.get("adjust", False))
        elif t == "sma_return":
            out[iid] = indicators.sma_return(int(p["window"]), p["symbol"])
        elif t == "std_price":
            out[iid] = indicators.std_price(int(p["window"]), p.get("symbol"), int(p.get("ddof", 0)))
        elif t == "std_return":
            out[iid] = indicators.std_return(int(p["window"]), p.get("symbol"), int(p.get("ddof", 0)))
        elif t == "rsi":
            out[iid] = indicators.rsi(int(p.get("window", 14)), p["symbol"])
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

def _normalize_logic(node: Any) -> Any:
    if isinstance(node, dict):
        t = node.get("type")
        if t in ("and","or"):
            kids = []
            for ch in node.get("children", []):
                kids.append(_normalize_logic(ch))
            node["children"] = kids
        elif t == "not":
            node["child"] = _normalize_logic(node.get("child"))
    return node

def eval_logic(node: Union[str, Dict[str, Any]], signal_map: Dict[str, pd.Series]) -> pd.Series:
    if not signal_map:
        raise ValueError("signal_map is empty")
    default_idx = next(iter(signal_map.values())).index
    if node is None:
        return pd.Series(0, index=default_idx)
    if isinstance(node, str):
        if node not in signal_map:
            raise ValueError(f"Unknown logic id '{node}'.")
        return signal_map[node].astype(int)
    t = node.get("type")
    if t in ("and","or"):
        children = node.get("children", [])
        if not children:
            raise ValueError("Empty logic children.")
        s = None
        for ch in children:
            v = eval_logic(ch, signal_map)
            s = v if s is None else ((s & v).astype(int) if t == "and" else ((s | v).astype(int)))
        return s
    if t == "not":
        v = eval_logic(node.get("child"), signal_map)
        return (1 - v).astype(int)
    raise ValueError("Unknown logic node.")

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
    sig_series: Dict[str, pd.Series] = {}
    for r in rules:
        if "when" in r:
            sid = r["when"]
            if sid not in signal_map:
                raise ValueError(f"rules_weights: unknown signal id '{sid}'")
            sig_series[sid] = signal_map[sid].reindex(index).fillna(0).astype(int)
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
# Semantic assertions & round-trip
# ========================

def semantic_assertions(eq_logic: pd.Series,
                        alloc: dict,
                        index: pd.DatetimeIndex) -> List[str]:
    errors = []
    # Logic should be non-degenerate (flip at least once)
    if eq_logic.sum() == 0 or eq_logic.sum() == len(eq_logic):
        errors.append("Final decision signal is degenerate (always true/false) on the selected window.")
    # For conditional weights ensure both branches occur
    if alloc.get("type") == "conditional_weights":
        if eq_logic.sum() == 0:
            errors.append("Conditional allocation uses only 'when_false' branch on this window.")
        if eq_logic.sum() == len(eq_logic):
            errors.append("Conditional allocation uses only 'when_true' branch on this window.")
    return errors

def render_roundtrip_bullets(engine: dict) -> List[str]:
    uni = ", ".join(engine.get("universe", []))
    rb = engine.get("rebalance",{}).get("frequency","W")
    data = engine.get("data",{})
    start = data.get("start")
    end = data.get("end") or "latest"
    alloc = engine.get("allocation",{})
    if alloc.get("type") == "conditional_weights":
        wt_true = alloc.get("when_true",{})
        wt_false = alloc.get("when_false",{})
        hedgers = [k for k,v in wt_false.items() if v >= 0.99]
        hedge_str = hedgers[0] if hedgers else "hedge basket"
        bullets = [
            f"Universe: {uni}",
            "Position when logic is TRUE: weighted long across universe.",
            f"When FALSE: hedge into {hedge_str}.",
            f"Rebalance: {rb}",
            f"Data: {start} to {end}",
        ]
    else:
        bullets = [
            f"Universe: {uni}",
            "Always invested using rule defaults (unless a rule triggers).",
            f"Rebalance: {rb}",
            f"Data: {start} to {end}",
        ]
    return bullets

def coverage_gates(dsl: dict, bullets: List[str]) -> List[str]:
    # Minimal gate: every clause must have coverage and at least one keyword hits the bullets text.
    errors = []
    cov = dsl.get("coverage", {})
    clause_text = {c["id"]: c["text"] for c in dsl.get("clauses", [])}
    missing = [cid for cid, nodes in cov.items() if not nodes]
    if missing:
        errors.append(f"Clauses have empty coverage: {missing}")
    joined = " ".join(bullets).lower()
    for cid, text in clause_text.items():
        toks = [t.strip().lower() for t in text.replace(","," ").split() if len(t.strip())>=4]
        if toks and not any(t in joined for t in toks[:8]):  # cheap overlap check
            errors.append(f"Clause appears not reflected after compile: '{text[:60]}...'")
    return errors


# ========================
# Streamlit App (single-pass parse → compile → validate → backtest)
# ========================

st.set_page_config(page_title="Strategy Composer (DSL prototype)", layout="wide")
st.title("Strategy Composer (DSL prototype)")

st.subheader("What this is")
st.markdown(FEATURES)
st.subheader("Limitations")
st.markdown(LIMITATIONS)

if "dsl_text" not in st.session_state:
    st.session_state.dsl_text = ""
if "engine_text" not in st.session_state:
    st.session_state.engine_text = ""
if "debug" not in st.session_state:
    st.session_state.debug = []

st.header("1) Describe your strategy")
user_prompt = st.text_area(
    "Strategy description (English)",
    height=160,
    placeholder="e.g., Create a bundle of renewable energy names and hedge using oil ETFs when volatility spikes. Weekly rebalance."
)

c1, c2 = st.columns(2)
parse_clicked = c1.button("Parse Strategy (single LLM call)")
compile_clicked = c2.button("Compile & Backtest")

st.divider()

# ---- Parse ----
if parse_clicked:
    if not user_prompt.strip():
        st.error("Please enter a strategy description.")
        st.stop()
    st.session_state.debug = []
    dsl, err = llm_parse_dsl(user_prompt)
    if err:
        st.error(err)
        st.stop()
    if dsl.get("status") == "NEEDS_CLARIFICATION":
        st.warning("The strategy needs clarification before it can be represented.")
        if dsl.get("clarifications"):
            st.info("Requested clarifications:\n- " + "\n- ".join(dsl["clarifications"]))
    st.session_state.dsl_text = json.dumps(dsl, indent=2)
    st.success("Parsed DSL.")
    st.rerun()

# Show/allow editing the DSL
if st.session_state.dsl_text:
    st.subheader("Parsed DSL (editable)")
    st.code(st.session_state.dsl_text, language="json")
    st.session_state.dsl_text = st.text_area("Edit DSL and re-run 'Compile & Backtest' if needed", value=st.session_state.dsl_text, height=320, key="dsl_editor")

# ---- Compile & Backtest ----
if compile_clicked:
    if not st.session_state.dsl_text.strip():
        st.error("No DSL to compile. Parse first.")
        st.stop()

    try:
        dsl = json.loads(st.session_state.dsl_text)
    except Exception as e:
        st.error(f"DSL is not valid JSON: {e}")
        st.stop()

    # Hard gates before compile
    if dsl.get("status") == "NEEDS_CLARIFICATION":
        st.error("Parser requested clarifications. Please edit your text/DSL to proceed.")
        st.stop()

    # Compile
    try:
        engine = compile_dsl_to_engine(dsl)
    except CompileError as ce:
        st.error(f"Compile error: {ce}")
        st.stop()
    except Exception as e:
        st.exception(e)
        st.stop()

    # Show engine JSON
    st.subheader("Compiled Engine JSON")
    eng_str = json.dumps(engine, indent=2)
    st.code(eng_str, language="json")
    st.session_state.engine_text = eng_str

    # Data fetch
    universe = engine["universe"]
    bench = "SPY"
    tk = universe if bench in universe else (universe + [bench])

    start = engine.get("data", {}).get("start")
    end = engine.get("data", {}).get("end")
    try:
        close_all = fetch_yahoo(tk, start, end)
    except Exception as fetch_err:
        st.exception(fetch_err)
        st.stop()

    if close_all.empty:
        st.error("No data returned. Check tickers and dates.")
        st.stop()

    # Filter to available universe
    available_universe = [u for u in universe if u in close_all.columns]
    missing = [u for u in universe if u not in close_all.columns]
    if missing:
        st.error(f"Missing data for: {', '.join(missing)}. Edit DSL to replace or remove them.")
        st.stop()
    if not available_universe:
        st.error("None of the universe tickers returned data.")
        st.stop()

    # Build indicators/signals
    panel = PricePanel.from_wide_close(close_all[available_universe])
    ind = Indicators(panel)
    ind_map = build_indicators(engine, ind)
    index = panel.close.index
    sig_map = build_signals(engine, ind_map, ind, index)

    # Logic evaluation
    logic_node = _normalize_logic(engine.get("logic"))
    try:
        final_signal = eval_logic(logic_node, sig_map).reindex(index).fillna(0).astype(int)
    except Exception as e:
        st.error(f"Logic evaluation error: {e}")
        st.stop()

    # Semantic assertions
    gates = semantic_assertions(final_signal, engine.get("allocation",{}), index)
    bullets = render_roundtrip_bullets(engine)
    gates += coverage_gates(dsl, bullets)

    if gates:
        st.error("Faithfulness/semantic gates failed:\n- " + "\n- ".join(gates))
        st.info("Round-trip summary:\n• " + "\n• ".join(bullets))
        st.stop()

    # Weights
    alloc = engine["allocation"]
    if alloc.get("type") == "conditional_weights":
        weights = conditional_weights(final_signal, alloc["when_true"], alloc["when_false"])
    elif alloc.get("type") == "rules":
        weights = rules_weights(alloc.get("rules", []), sig_map, index)
    else:
        st.error("Unsupported allocation type.")
        st.stop()

    # Backtest
    costs = engine.get("costs", {})
    bt = backtest(
        close=panel.close[weights.columns].reindex(index).ffill(),
        weights=weights.reindex(index).fillna(0.0),
        costs=BrokerCosts(
            slippage_bps=float(costs.get("slippage_bps", 2.0)),
            fee_per_trade=float(costs.get("fee_per_trade", 0.0)),
        ),
        freq=engine.get("rebalance", {}).get("frequency", "W"),
        initial_cash=10_000.0,
    )
    eq = bt["equity"]
    rets = bt["returns"]

    bench_col = bench if bench in close_all.columns else available_universe[0]
    bench_close = close_all[[bench_col]].reindex(index).ffill()
    bench_eq = 10_000.0 * (bench_close[bench_col] / bench_close[bench_col].iloc[0])

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

    st.subheader("Round-trip summary (auto)")
    st.write("\n".join([f"• {b}" for b in bullets]))

    st.subheader("Charts")
    plot_equity(eq, bench_eq, engine.get("meta", {}).get("name", "Strategy"))
    plot_drawdown(eq)
