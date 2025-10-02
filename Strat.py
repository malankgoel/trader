# strat.py
import json
import math
from dataclasses import dataclass
from typing import Dict, Union, Optional, List, Tuple, Any, Literal

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# --- App config & prompts ---
from configuration import (
    FEATURES,
    LIMITATIONS,
    MODEL_NAME,
    DEFAULTS,
    ALLOWED,
    THEME_TICKERS,          # may be unused if you switched to judgment-based
    HEDGE_FALLBACKS,        # may be unused if you switched to judgment-based
    INTENT_EXTRACTOR_PROMPT,
    UNIVERSE_BUILDER_PROMPT,
    INDICATOR_PICKER_PROMPT,
    SIGNALS_LOGIC_PROMPT,
    ALLOCATION_WEIGHTS_PROMPT,
    OPS_DATA_COSTS_REBALANCE_PROMPT,
    AUDITOR_REPAIR_PROMPT,
)

# If you created a judgment-based universe builder in configuration.py, prefer it:
try:
    from configuration import UNIVERSE_BUILDER_PROMPT_JUDGMENT as _UBP_JUDGMENT
    UNIVERSE_PROMPT_ACTIVE = _UBP_JUDGMENT
except Exception:
    UNIVERSE_PROMPT_ACTIVE = UNIVERSE_BUILDER_PROMPT

# --- OpenAI setup (Responses API) ---
try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except Exception:
    _OPENAI_AVAILABLE = False

LLM_MAX_TOKENS = 2000
LLM_TEMPERATURE = 0.0
LLM_REASONING: Literal["minimal","low","medium","high"] = "low"


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
        raise RuntimeError(
            "Missing dependency: yfinance. Install it with `pip install yfinance`."
        ) from e

    start = _to_datetime(start)
    end = _to_datetime(end) or pd.Timestamp.today()

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

    @classmethod
    def from_source(cls, tickers: List[str], start: Optional[str], end: Optional[str], source: str) -> "PricePanel":
        if source.lower() not in ("yahoo","yf"):
            raise ValueError("Only yahoo is supported")
        close = fetch_yahoo(tickers, start, end)
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
# Validation & builders
# ========================

ALLOWED_INDICATORS = {
    "sma_price", "ema_price", "sma_return", "std_price", "std_return", "rsi", "drawdown", "max_drawdown"
}
ALLOWED_SIGNALS = {"gt", "lt", "cross_above", "cross_below"}
ALLOWED_REBAL = {"B", "W", "M"}

def _sum_close_to_one(d: Dict[str, float]) -> bool:
    return abs(sum(d.values()) - 1.0) <= 1e-3

def validate_strategy(js: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    # universe
    if not js.get("universe"):
        errors.append("universe must be a non-empty list of tickers.")
    # data
    data = js.get("data", {})
    if data.get("source", "yahoo").lower() not in ("yahoo", "yf"):
        errors.append("Only 'yahoo' data source is supported.")
    # indicators
    ids = set()
    for ind in js.get("indicators", []):
        t = ind.get("type")
        iid = ind.get("id")
        if t not in ALLOWED_INDICATORS:
            errors.append(f"Unsupported indicator: {t}")
        if iid in ids:
            errors.append(f"Duplicate indicator id: {iid}")
        ids.add(iid)
    # signals
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
        # check refs
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
    # logic
    if js.get("logic") is None and js.get("allocation",{}).get("type") != "rules":
        errors.append("logic is None but allocation is not 'rules'")
    # allocation
    universe = set(js.get("universe", []))
    alloc = js.get("allocation", {})
    alloc_type = alloc.get("type")
    if alloc_type not in ("conditional_weights", "rules"):
        errors.append("allocation.type must be 'conditional_weights' or 'rules'.")
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
                    errors.append(f"rules.weights contains tickers not in universe")
                if w and not _sum_close_to_one(w):
                    errors.append("Each rule.weights must sum to 1.000 (±0.001).")
            if "default" in r:
                has_any = True
                w = r["default"]
                if set(w.keys()) - universe:
                    errors.append(f"rules.default contains tickers not in universe")
                if w and not _sum_close_to_one(w):
                    errors.append("Default weights must sum to 1.000 (±0.001).")
        if not has_any:
            errors.append("rules must include a 'weights' or 'default' branch.")
    # rebalance
    freq = js.get("rebalance", {}).get("frequency", "W")
    if freq not in ALLOWED_REBAL:
        errors.append(f"Unsupported rebalance frequency: {freq}")
    # dates
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

def _prune_logic(node: Any, valid_ids: set) -> Optional[Any]:
    """Drop children that don't exist; collapse to None if nothing remains."""
    if node is None:
        return None
    if isinstance(node, str):
        return node if node in valid_ids else None
    if not isinstance(node, dict):
        return None

    t = node.get("type")
    if t in ("and", "or"):
        kids = []
        for ch in node.get("children", []):
            keep = _prune_logic(ch, valid_ids)
            if keep is not None:
                kids.append(keep)
        if not kids:
            return None
        return {"type": t, "children": kids}
    if t == "not":
        c = _prune_logic(node.get("child"), valid_ids)
        return {"type":"not","child":c} if c is not None else None
    return None


def _sync_universe_with_references(js: dict) -> dict:
    u = set(js.get("universe", []))
    # From allocation
    alloc = js.get("allocation", {})
    if alloc.get("type") == "conditional_weights":
        u |= set((alloc.get("when_true") or {}).keys())
        u |= set((alloc.get("when_false") or {}).keys())
    elif alloc.get("type") == "rules":
        for r in alloc.get("rules", []):
            if "weights" in r and isinstance(r["weights"], dict):
                u |= set(r["weights"].keys())
            if "default" in r and isinstance(r["default"], dict):
                u |= set(r["default"].keys())

                # From signals (price sides)
    for s in js.get("signals", []):
        for side in ("left", "right"):
            node = s.get(side, {})
            if isinstance(node, dict) and node.get("kind") == "price" and node.get("symbol"):
                u.add(node["symbol"])

    js["universe"] = sorted(u)
    return js

def _normalize_logic(node: Any) -> Any:
    # Accept {"type":"or","of":[...]} or {"type":"and","of":[...]}
    if isinstance(node, dict):
        t = node.get("type")
        if t in ("and", "or"):
            # Map 'of' -> 'children' if present
            if "children" not in node and "of" in node and isinstance(node["of"], list):
                node = {**node, "children": node["of"]}
                node.pop("of", None)
            # Recursively normalize children
            kids = []
            for ch in node.get("children", []):
                kids.append(_normalize_logic(ch))
            node["children"] = kids
        elif t == "not":
            # Handle alt key names defensively
            child = node.get("child", node.get("negate"))
            node["child"] = _normalize_logic(child)
    return node

def eval_logic(node: Union[str, Dict[str, Any]], signal_map: Dict[str, pd.Series]) -> pd.Series:
    if not signal_map:
        raise ValueError("signal_map is empty")

    # default index to fall back on
    default_idx = next(iter(signal_map.values())).index

    if node is None:
        return pd.Series(0, index=default_idx)

    if isinstance(node, str):
        if node not in signal_map:
            # unknown id -> treat as always-false
            return pd.Series(0, index=default_idx)
        return signal_map[node].astype(int)

    t = node.get("type")
    if t in ("and", "or"):
        children = node.get("children", [])
        if not children:  # <- guard empty
            return pd.Series(0, index=default_idx)
        s = None
        for ch in children:
            v = eval_logic(ch, signal_map)
            s = v if s is None else ((s & v).astype(int) if t == "and" else ((s | v).astype(int)))
        return s
    if t == "not":
        v = eval_logic(node.get("child"), signal_map)
        return (1 - v).astype(int)

    # unknown node -> false
    return pd.Series(0, index=default_idx)


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
# LLM helpers (Responses API)
# ========================

def _parse_responses_output(resp) -> str:
    """Robustly extract text from OpenAI Responses result."""
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

def _llm_call_json(step_name: str, system_prompt: str, user_payload: Any) -> Tuple[Optional[dict], str, Optional[str]]:
    """
    Returns (json_obj_or_none, raw_text, error_message_or_none).
    Does not raise on JSON errors; caller can decide repair/defaults.
    """
    if not _OPENAI_AVAILABLE:
        return None, "", f"OpenAI SDK not available for step '{step_name}'."

    try:
        client = OpenAI()
        resp = client.responses.create(
            model=MODEL_NAME,
            reasoning={"effort": LLM_REASONING},
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_payload if isinstance(user_payload, str) else json.dumps(user_payload, separators=(",",":"))}
            ],
            max_output_tokens=LLM_MAX_TOKENS,
        )
        text = _parse_responses_output(resp).strip()
        raw = text

        # strip code fences if any
        if text.startswith("```"):
            first_nl = text.find("\n")
            text = text[first_nl+1:] if first_nl != -1 else text
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        obj = None
        err = None
        try:
            obj = json.loads(text)
        except Exception as e:
            # try to salvage first/last braces
            l = text.find("{"); r = text.rfind("}")
            if l != -1 and r != -1 and r > l:
                try:
                    obj = json.loads(text[l:r+1])
                except Exception as e2:
                    err = f"JSON parse failed for step '{step_name}': {e2}"
            else:
                err = f"JSON parse failed for step '{step_name}': {e}"
        return obj, raw, err
    except Exception as e:
        return None, "", f"OpenAI call error for step '{step_name}': {e}"


# ========================
# Strategy assembly (pipeline glue)
# ========================

def _round_norm(weights: Dict[str, float]) -> Dict[str, float]:
    if not weights:
        return {}
    w = {k: float(v) for k, v in weights.items()}
    s = sum(w.values())
    if s == 0:
        n = len(w)
        eq = {k: round(1.0/n, 3) for k in w}
        # renormalize to 1.000 exactly
        diff = round(1.000 - sum(eq.values()), 3)
        if diff != 0:
            # adjust first key to absorb rounding diff
            k0 = next(iter(eq.keys()))
            eq[k0] = round(eq[k0] + diff, 3)
        return eq
    # rescale then round to 3 decimals
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

def assemble_final_json(
    intent: dict,
    universe_pkg: dict,
    indicators_pkg: dict,
    siglogic_pkg: dict,
    alloc_pkg: dict,
    ops_pkg: dict
) -> dict:
    uni = universe_pkg.get("universe", [])
    final = {
        "version": "1.0",
        "meta": {
            "name": "Generated Strategy",
            "notes": ""
        },
        "universe": uni,
        "data": ops_pkg.get("data", {"source":"yahoo","start":DEFAULTS["start_date"],"end":None,"frequency":"B"}),
        "costs": ops_pkg.get("costs", {"slippage_bps":2,"fee_per_trade":0}),
        "indicators": indicators_pkg.get("indicators", []),
        "signals": siglogic_pkg.get("signals", []),
        "logic": siglogic_pkg.get("logic", None),
        "allocation": alloc_pkg.get("allocation", {"type":"rules","rules":[{"default": _equal_weights(uni)}]}),
        "rebalance": ops_pkg.get("rebalance", {"frequency":"W"}),
    }
    return final


# ========================
# Streamlit App (Simple UI)
# ========================

st.set_page_config(page_title="Strategy Backtester", layout="wide")
st.title("Strategy Backtester")

if "step" not in st.session_state:
    st.session_state.step = 1
if "debug_log" not in st.session_state:
    st.session_state.debug_log = []  # list of dicts per step
if "final_json_text" not in st.session_state:
    st.session_state.final_json_text = ""
if "gen_ready" not in st.session_state:
    st.session_state.gen_ready = False
if "run_bt" not in st.session_state:
    st.session_state.run_bt = False

if st.session_state.step == 1:
    st.subheader("Overview")
    st.markdown(FEATURES)
    st.subheader("Limitations")
    st.markdown(LIMITATIONS)
    if st.button("Continue"):
        st.session_state.step = 2
        st.rerun()

elif st.session_state.step == 2:
    st.header("Describe your strategy")
    user_prompt = st.text_area(
        "Strategy description (English)",
        height=160,
        placeholder="e.g., Invest in a bundle of US tech stocks; hedge to cash when volatility spikes; weekly rebalance."
    )

    run = st.button("Generate Strategy JSON")
    st.divider()

    if run:
        if not user_prompt.strip():
            st.error("Please enter a strategy description.")
            st.stop()

        st.session_state.debug_log = []
        used_repair = False
        used_defaults = False

        # 1) Intent
        with st.status("Analyzing intent...", expanded=False) as s:
            intent_js, raw_intent, err = _llm_call_json("intent", INTENT_EXTRACTOR_PROMPT, user_prompt)
            st.session_state.debug_log.append({"step":"intent","raw":raw_intent,"error":err})
            if err or not isinstance(intent_js, dict):
                used_defaults = True
                # Fall back to minimal intent
                intent_js = {
                    "objective": DEFAULTS["objective"],
                    "risk_goal": DEFAULTS["risk_goal"],
                    "themes": [],
                    "explicit_tickers": [],
                    "hedge_intent": DEFAULTS["hedge_intent"],
                    "weighting_hint": DEFAULTS["weighting_hint"],
                    "date_hints": {"start": None, "end": None},
                    "rebalance_hint": "none"
                }
                s.update(label="Intent: defaulted due to parse error", state="error")
            else:
                s.update(label="Intent extracted", state="complete")

        # 2) Universe
        payload_u = intent_js
        with st.status("Building universe...", expanded=False) as s:
            uni_js, raw_uni, err = _llm_call_json("universe", UNIVERSE_PROMPT_ACTIVE, payload_u)
            st.session_state.debug_log.append({"step":"universe","raw":raw_uni,"error":err})
            if err or not isinstance(uni_js, dict) or not uni_js.get("universe"):
                used_defaults = True
                uni_js = {"universe": intent_js.get("explicit_tickers", [])[:12] or ["SPY","QQQ","AAPL","MSFT"],
                          "hedge_assets": ["BIL"] if intent_js.get("hedge_intent") else [],
                          "rule":"fallback: explicit or SPY/QQQ/AAPL/MSFT; hedge=BIL if needed"}
                s.update(label="Universe: defaulted due to parse error", state="error")
            else:
                s.update(label="Universe built", state="complete")

        # 3) Indicators
        payload_i = {"intent": intent_js, "universe": uni_js.get("universe", [])}
        with st.status("Selecting indicators...", expanded=False) as s:
            ind_js, raw_ind, err = _llm_call_json("indicators", INDICATOR_PICKER_PROMPT, payload_i)
            st.session_state.debug_log.append({"step":"indicators","raw":raw_ind,"error":err})
            if err or not isinstance(ind_js, dict):
                used_defaults = True
                # basic trend defaults if needed
                inds = []
                for sym in payload_i["universe"][:12]:
                    inds.append({"id":f"sma_price_{sym}_50","type":"sma_price","params":{"symbol":sym,"window":50}})
                    inds.append({"id":f"sma_price_{sym}_200","type":"sma_price","params":{"symbol":sym,"window":200}})
                ind_js = {"indicators": inds}
                s.update(label="Indicators: defaulted due to parse error", state="error")
            else:
                s.update(label="Indicators selected", state="complete")

        # 4) Signals & Logic
        payload_s = {"intent": intent_js, "universe": uni_js.get("universe", []), "indicators": ind_js.get("indicators", [])}
        with st.status("Building signals & logic...", expanded=False) as s:
            sig_js, raw_sig, err = _llm_call_json("signals_logic", SIGNALS_LOGIC_PROMPT, payload_s)
            st.session_state.debug_log.append({"step":"signals_logic","raw":raw_sig,"error":err})
            if err or not isinstance(sig_js, dict) or "signals" not in sig_js:
                used_defaults = True
                # simple default: price > 200SMA OR across all
                signals = []
                for sym in payload_s["universe"]:
                    ref = f"sma_price_{sym}_200"
                    if any(i["id"] == ref for i in payload_s["indicators"]):
                        signals.append({
                            "id": f"{sym}_gt_sma200",
                            "type": "gt",
                            "left": {"kind":"price","symbol":sym},
                            "right":{"kind":"indicator","ref":ref}
                        })
                logic = {"type":"or","children":[s["id"] for s in signals]} if signals else None
                sig_js = {"signals": signals, "logic": logic}
                s.update(label="Signals/Logic: defaulted due to parse error", state="error")
            else:
                s.update(label="Signals & logic built", state="complete")

        # 5) Allocation & Weights
        payload_a = {
            "intent": intent_js,
            "universe": uni_js.get("universe", []),
            "hedge_assets": uni_js.get("hedge_assets", []),
            "signals": sig_js.get("signals", []),
            "logic": sig_js.get("logic", None)
        }
        with st.status("Choosing allocation & weights...", expanded=False) as s:
            alloc_js, raw_alloc, err = _llm_call_json("allocation", ALLOCATION_WEIGHTS_PROMPT, payload_a)
            st.session_state.debug_log.append({"step":"allocation","raw":raw_alloc,"error":err})
            if err or not isinstance(alloc_js, dict) or "allocation" not in alloc_js:
                used_defaults = True
                if intent_js.get("hedge_intent") and payload_a.get("hedge_assets"):
                    wh = _equal_weights(payload_a["universe"])
                    wf = {payload_a["hedge_assets"][0]: 1.0}
                    alloc_js = {"allocation":{"type":"conditional_weights","when_true":_round_norm(wh),"when_false":_round_norm(wf)},"policy":"equal"}
                else:
                    alloc_js = {"allocation":{"type":"rules","rules":[{"default": _equal_weights(payload_a["universe"])}]},"policy":"equal"}
                s.update(label="Allocation: defaulted due to parse error", state="error")
            else:
                # ensure weights rounded/normalized just in case
                if alloc_js["allocation"]["type"] == "conditional_weights":
                    wt = _round_norm(alloc_js["allocation"].get("when_true", {}))
                    wf = _round_norm(alloc_js["allocation"].get("when_false", {}))
                    alloc_js["allocation"]["when_true"] = wt
                    alloc_js["allocation"]["when_false"] = wf
                else:
                    new_rules = []
                    for r in alloc_js["allocation"].get("rules", []):
                        if "weights" in r:
                            r["weights"] = _round_norm(r["weights"])
                        if "default" in r:
                            r["default"] = _round_norm(r["default"])
                        new_rules.append(r)
                    alloc_js["allocation"]["rules"] = new_rules
                s.update(label="Allocation chosen", state="complete")

        # 6) Ops (data, costs, rebalance)
        payload_o = {
            "intent": intent_js,
            "universe": uni_js.get("universe", []),
            "date_hints": intent_js.get("date_hints", {"start":None,"end":None}),
            "rebalance_hint": intent_js.get("rebalance_hint","none"),
            "risk_goal": intent_js.get("risk_goal", DEFAULTS["risk_goal"])
        }
        with st.status("Setting data window, costs & rebalance...", expanded=False) as s:
            ops_js, raw_ops, err = _llm_call_json("ops", OPS_DATA_COSTS_REBALANCE_PROMPT, payload_o)
            st.session_state.debug_log.append({"step":"ops","raw":raw_ops,"error":err})
            if err or not isinstance(ops_js, dict) or "data" not in ops_js:
                used_defaults = True
                ops_js = {
                    "data":{"source":"yahoo","start":DEFAULTS["start_date"],"end":None,"frequency":"B"},
                    "costs":{"slippage_bps":2,"fee_per_trade":0},
                    "rebalance":{"frequency":"W" if intent_js.get("risk_goal","balanced") != "conservative" else "M"}
                }
                s.update(label="Ops: defaulted due to parse error", state="error")
            else:
                s.update(label="Ops set", state="complete")

        # Combine
        with st.status("Combining & validating...", expanded=False) as s:
            final_js = assemble_final_json(intent_js, uni_js, ind_js, sig_js, alloc_js, ops_js)
            final_js = _sync_universe_with_references(final_js)
            errors = validate_strategy(final_js)
            errors = validate_strategy(final_js)
            st.session_state.debug_log.append({"step":"combine_validate","raw":json.dumps(final_js, separators=(",",":")), "error":"; ".join(errors) if errors else None})

            # Attempt repair if needed
            if errors:
                used_repair = True
                allowed_refs = [i["id"] for i in final_js.get("indicators", [])]
                repair_payload = {
                    "candidate": final_js,
                    "errors": errors,
                    "allowed_refs": allowed_refs,
                    "allowed_freqs": list(ALLOWED_REBAL),
                    "allowed_signals": list(ALLOWED_SIGNALS)
                }
                fix_js, raw_fix, err_fix = _llm_call_json("repair", AUDITOR_REPAIR_PROMPT, repair_payload)
                st.session_state.debug_log.append({"step":"repair","raw":raw_fix,"error":err_fix})
                if not err_fix and isinstance(fix_js, dict):
                    final_js = _sync_universe_with_references(fix_js)
                    errors2 = validate_strategy(final_js)
                    st.session_state.debug_log.append({"step":"post_repair_validate","raw":json.dumps(final_js, separators=(",",":")), "error":"; ".join(errors2) if errors2 else None})
                    if errors2:
                        used_defaults = True  # still failing; we will force minimal defaults below
                        # Minimal safe fallback: equal-weight default rule, keep universe/data
                        final_js["signals"] = final_js.get("signals", [])
                        final_js["logic"] = final_js.get("logic", (final_js["signals"][0]["id"] if final_js["signals"] else None))
                        final_js["allocation"] = {"type":"rules","rules":[{"default": _equal_weights(final_js.get("universe", []))}]}
                else:
                    used_defaults = True  # repair call failed

            # Final sanity normalize allocation weights
            alloc = final_js.get("allocation", {})
            if alloc.get("type") == "conditional_weights":
                alloc["when_true"] = _round_norm(alloc.get("when_true", {}))
                alloc["when_false"] = _round_norm(alloc.get("when_false", {}))
            elif alloc.get("type") == "rules":
                nr = []
                for r in alloc.get("rules", []):
                    if "weights" in r:
                        r["weights"] = _round_norm(r["weights"])
                    if "default" in r:
                        r["default"] = _round_norm(r["default"])
                    nr.append(r)
                alloc["rules"] = nr
            final_js["allocation"] = alloc

            s.update(label="Strategy ready", state="complete")
            # Store artifacts to session, then rerun so UI draws outside `if run:`
            st.session_state.final_json_text = json.dumps(final_js, indent=2)
            st.session_state.gen_ready = True
            st.session_state.run_bt = False  # reset if we’re making a new strategy
            st.session_state.debug_log = st.session_state.debug_log  # keep logs
            st.rerun()


        # Present final JSON and debug info
        st.success("Generation complete.")
        if used_repair or used_defaults:
            msg = []
            if used_repair:
                msg.append("repair prompt used")
            if used_defaults:
                msg.append("defaults applied")
            st.warning(" ; ".join(msg))

        with st.expander("Debug: model outputs & errors"):
            for row in st.session_state.debug_log:
                st.markdown(f"**Step:** {row['step']}")
                if row.get("error"):
                    st.error(row["error"])
                st.code(row.get("raw",""), language="json")

        st.subheader("Final Strategy JSON (editable)")
        st.session_state.final_json_text = json.dumps(final_js, indent=2)
        st.text_area("Strategy JSON", value=st.session_state.final_json_text, height=360, key="final_json_editor")

    # ---- Editor & Run Backtest (persists across reruns) ----
    if st.session_state.gen_ready:
        st.success("Generation complete.")

        with st.expander("Debug: model outputs & errors"):
            for row in st.session_state.debug_log:
                st.markdown(f"**Step:** {row['step']}")
                if row.get("error"):
                    st.error(row["error"])
                st.code(row.get("raw",""), language="json")

        st.subheader("Final Strategy JSON (editable)")
        # Initialize the editor value only once; avoid resetting user edits on rerun
        if "final_json_editor" not in st.session_state:
            st.session_state.final_json_editor = st.session_state.final_json_text
        st.text_area("Strategy JSON", key="final_json_editor", height=360)

        if st.button("Run Backtest"):
            st.session_state.run_bt = True
            st.rerun()
    # ---- Backtest & Results ----
    if st.session_state.gen_ready and st.session_state.run_bt:
        with st.spinner("Running backtest..."):
            try:
                js = json.loads(st.session_state.final_json_editor)

                errs = validate_strategy(js)
                if errs:
                    st.error("Validation errors:\n- " + "\n- ".join(errs))
                    st.stop()

                universe = list(js["universe"])
                if not universe:
                    st.error("Universe is empty.")
                    st.stop()

                bench = "SPY"
                tk = universe if bench in universe else universe + [bench]

                start = js.get("data", {}).get("start")
                end = js.get("data", {}).get("end")

                try:
                    close_all = fetch_yahoo(tk, start, end)
                except Exception as fetch_err:
                    st.exception(fetch_err)
                    st.stop()

                if close_all.empty:
                    st.error("No data returned. Check tickers and dates.")
                    st.stop()

                effective_first = {c: close_all[c].first_valid_index() for c in close_all.columns}
                latest = max([dt for dt in effective_first.values() if dt is not None])
                if start is None or (pd.to_datetime(start) < latest):
                    st.info(f"Adjusted start to {latest.date()} to match earliest available data across your tickers.")

                missing = [sym for sym in tk if sym not in close_all.columns]
                if missing:
                    st.warning(f"Dropped symbols with no data: {', '.join(missing)}")

                available_universe = [u for u in universe if u in close_all.columns]
                if not available_universe:
                    st.error("None of the universe tickers returned data.")
                    st.stop()

                available_set = set(available_universe)
                js["indicators"] = [
                    ind for ind in js.get("indicators", [])
                    if ind.get("params", {}).get("symbol") in available_set
                    or ind["type"] in ("drawdown", "max_drawdown") and ind.get("params", {}).get("symbol") in (available_set | {None})
                ]

                valid_ind_ids = {ind["id"] for ind in js["indicators"]}
                def _side_ok(side):
                    k = side.get("kind")
                    if k == "price":
                        return side.get("symbol") in available_set
                    if k == "indicator":
                        return side.get("ref") in valid_ind_ids
                    if k == "const":
                        return isinstance(side.get("value"), (int, float))
                    return False

                js["signals"] = [
                    s for s in js.get("signals", [])
                    if _side_ok(s.get("left", {})) and _side_ok(s.get("right", {}))
                ]

                # If logic points to a signal we just dropped, null it (rules alloc still works)
                if isinstance(js.get("logic"), str) and js["logic"] not in {s["id"] for s in js["signals"]}:
                    js["logic"] = None

                panel = PricePanel.from_wide_close(close_all[available_universe])
                ind = Indicators(panel)
                ind_map = build_indicators(js, ind)
                index = panel.close.index
                sig_map = build_signals(js, ind_map, ind, index)

                logic_node = _normalize_logic(js.get("logic"))
                logic_node = _prune_logic(logic_node, valid_ids=set(sig_map.keys()))

                final_signal = (
                    pd.Series(0, index=index)
                    if not sig_map else eval_logic(logic_node, sig_map).reindex(index).fillna(0).astype(int)
                )


                alloc = js["allocation"]
                if alloc.get("type") == "conditional_weights":
                    weights = conditional_weights(final_signal, alloc.get("when_true", {}), alloc.get("when_false", {}))
                elif alloc.get("type") == "rules":
                    weights = rules_weights(alloc.get("rules", []), sig_map, index)
                else:
                    st.error("Unsupported allocation type.")
                    st.stop()

                costs = js.get("costs", {})
                bt = backtest(
                    close=panel.close[weights.columns].reindex(index).ffill(),
                    weights=weights.reindex(index).fillna(0.0),
                    costs=BrokerCosts(
                        slippage_bps=float(costs.get("slippage_bps", 2.0)),
                        fee_per_trade=float(costs.get("fee_per_trade", 0.0)),
                    ),
                    freq=js.get("rebalance", {}).get("frequency", "W"),
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

                st.subheader("Charts")
                plot_equity(eq, bench_eq, js.get("meta", {}).get("name", "Strategy"))
                plot_drawdown(eq)

            except Exception as e:
                st.exception(e)
