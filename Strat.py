import json
import io
import math
from dataclasses import dataclass
from typing import Dict, Union, Optional, List, Tuple, Any

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ========================
# UI — Landing page blurb
# ========================

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

EXAMPLE_JSON = """{
  "version": "1.0",
  "meta": {
    "name": "IWM/XLK vs TLT/SHY with Gold Fallback",
    "notes": "Risk-on if IWM > 200SMA and RSI>=50; else if TLT > 100EMA then rates regime; else gold+cash defensive."
  },
  "universe": ["IWM", "XLK", "TLT", "SHY", "GLD"],
  "data": {
    "source": "yahoo",
    "start": "2015-01-01",
    "end": null,
    "frequency": "B"
  },
  "costs": {
    "slippage_bps": 2.0,
    "fee_per_trade": 0.0
  },
  "indicators": [
    { "id": "IWM_SMA200", "type": "sma_price", "params": { "symbol": "IWM", "window": 200 } },
    { "id": "IWM_RSI14",  "type": "rsi",       "params": { "symbol": "IWM", "window": 14 } },
    { "id": "TLT_EMA100", "type": "ema_price", "params": { "symbol": "TLT", "window": 100 } }
  ],
  "signals": [
    {
      "id": "SIG_IWM_GT_SMA",
      "type": "gt",
      "left":  { "kind": "price", "symbol": "IWM" },
      "right": { "kind": "indicator", "ref": "IWM_SMA200" }
    },
    {
      "id": "SIG_IWM_RSI_OK",
      "type": "gt",
      "left":  { "kind": "indicator", "ref": "IWM_RSI14" },
      "right": { "kind": "const", "value": 50.0 }
    },
    {
      "id": "SIG_TLT_GT_EMA",
      "type": "gt",
      "left":  { "kind": "price", "symbol": "TLT" },
      "right": { "kind": "indicator", "ref": "TLT_EMA100" }
    }
  ],
  "logic": { "type": "and", "children": ["SIG_IWM_GT_SMA", "SIG_IWM_RSI_OK"] },
  "allocation": {
    "type": "rules",
    "rules": [
      { "when": "SIG_IWM_GT_SMA", "weights": { "IWM": 0.7, "XLK": 0.3 } },
      { "when": "SIG_TLT_GT_EMA", "weights": { "TLT": 0.7, "SHY": 0.3 } },
      { "default": { "GLD": 0.5, "SHY": 0.5 } }
    ]
  },
  "rebalance": { "frequency": "M" }
}"""

# ======================================
# Data adapter (Yahoo)
# ======================================

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
    start = _to_datetime(start) or (pd.Timestamp.today() - pd.DateOffset(years=5))
    end = _to_datetime(end) or pd.Timestamp.today()
    df = yf.download(tickers, start=start, end=end, auto_adjust=True,
                     progress=False, group_by="ticker")
    if isinstance(df.columns, pd.MultiIndex):
        out = {}
        for t in tickers:
            out[t] = df[(t, "Close")].rename(t)
        wide = pd.concat(out, axis=1)
    else:
        wide = pd.DataFrame({tickers[0]: df["Close"]})
    wide.index = pd.to_datetime(wide.index)
    wide = wide.sort_index().dropna(how="all")
    wide = wide.asfreq("B").ffill().dropna(axis=1, how="all")
    return wide

def get_prices(tickers: List[str], start: Optional[str], end: Optional[str], source: str = "yahoo") -> pd.DataFrame:
    if source.lower() not in ("yahoo", "yf"):
        raise ValueError("Only 'yahoo' source is supported in this app.")
    return fetch_yahoo(tickers, start, end)

# ======================================
# Core containers & indicators
# ======================================

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

    # EMA now uses "window" for consistency
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
            out[iid] = indicators.std_price(p["window"], p.get("symbol"))
        elif t == "std_return":
            out[iid] = indicators.std_return(p["window"], p.get("symbol"))
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
# Plot helpers (Streamlit)
# ========================

def plot_equity(eq: pd.Series, bench_eq: pd.Series, name: str):
    base = eq.iloc[0]
    base_b = bench_eq.iloc[0]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    (eq / base).plot(ax=ax, label=name)
    (bench_eq / base_b).plot(ax=ax, label="Benchmark (SPY)")
    ax.set_title("Equity Curve (normalized)")
    ax.set_xlabel("Date")
    ax.legend()
    st.pyplot(fig)

def plot_drawdown(eq: pd.Series):
    mdd, s, e, dd = max_drawdown_series(eq)
    fig, ax = plt.subplots(figsize=(10, 3.2))
    dd.plot(ax=ax)
    ax.set_title("Drawdown")
    ax.set_xlabel("Date")
    st.pyplot(fig)

# ========================
# Streamlit App
# ========================

st.set_page_config(page_title="Strategy Backtester", layout="wide")

if "step" not in st.session_state:
    st.session_state.step = 1

st.title("Strategy Backtester (JSON + Yahoo Finance)")

if st.session_state.step == 1:
    st.subheader("Overview")
    st.markdown(FEATURES)
    st.subheader("Limitations")
    st.markdown(LIMITATIONS)
    if st.button("Continue"):
        st.session_state.step = 2
        st.experimental_rerun()

elif st.session_state.step == 2:
    st.subheader("Paste your Strategy JSON")
    colA, colB = st.columns([2, 1])
    with colA:
        json_text = st.text_area("JSON input", value=EXAMPLE_JSON, height=380)
        uploaded = st.file_uploader("...or upload a .json file", type=["json"])
        if uploaded is not None:
            try:
                json_text = uploaded.read().decode("utf-8")
                st.info("Loaded JSON from uploaded file.")
            except Exception as e:
                st.error(f"Failed to read file: {e}")
    with colB:
        st.markdown("**Hints**")
        st.write("- Keep strings single-line.\n- Use `null` for no end date.\n- Weights per branch must sum to 1.0.")
        run_btn = st.button("Run Backtest")

    if run_btn:
        try:
            js = json.loads(json_text)
            validate_strategy(js)

            universe = list(js["universe"])
            bench = "SPY"
            tk = universe if bench in universe else universe + [bench]

            start = js.get("data", {}).get("start")
            end = js.get("data", {}).get("end")
            source = js.get("data", {}).get("source", "yahoo")

            st.write(f"**Fetching data** for: {', '.join(tk)}")
            close_all = get_prices(tk, start, end, source)
            if close_all.empty:
                st.error("No data returned. Check tickers and dates.")
                st.stop()

            panel = PricePanel.from_wide_close(close_all[universe])
            ind = Indicators(panel)

            ind_map = build_indicators(js, ind)
            index = panel.close.index
            sig_map = build_signals(js, ind_map, ind, index)

            # 'logic' may be unused if allocation is 'rules', but we compute it for completeness
            logic_node = js.get("logic")
            final_signal = eval_logic(logic_node, sig_map).reindex(index).fillna(0).astype(int)

            alloc = js["allocation"]
            if alloc["type"] == "conditional_weights":
                weights = conditional_weights(final_signal, alloc["when_true"], alloc["when_false"])
            elif alloc["type"] == "rules":
                weights = rules_weights(alloc["rules"], sig_map, index)
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
            bench_ret = bench_eq.pct_change().fillna(0.0)

            # Metrics
            cum_ret = float(eq.iloc[-1] / eq.iloc[0] - 1)
            ann_ret = annualized_return(eq)
            shrp = sharpe_ratio(rets)
            mdd, dd_s, dd_e, dd_series = max_drawdown_series(eq)
            calmar = calmar_ratio(eq)
            tr_1m = trailing_return(eq, 21)
            tr_3m = trailing_return(eq, 63)
            alpha, beta, r2, corr = alpha_beta_r2(rets, bench_ret)

            st.success("Backtest complete.")
            st.subheader("Performance")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Cumulative Return", f"{cum_ret*100:,.2f}%")
            c2.metric("Annualized Return", f"{ann_ret*100:,.2f}%")
            c3.metric("Sharpe Ratio", f"{shrp:,.2f}")
            c4.metric("Max Drawdown", f"{mdd*100:,.2f}%")
            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Calmar", f"{calmar:,.2f}")
            c6.metric("Trailing 1M", f"{(tr_1m*100) if not np.isnan(tr_1m) else float('nan') :,.2f}%")
            c7.metric("Trailing 3M", f"{(tr_3m*100) if not np.isnan(tr_3m) else float('nan') :,.2f}%")
            c8.metric("Vol (daily std)", f"{rets.std(ddof=0):.4f}")

            st.subheader("Vs Benchmark (SPY)")
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Alpha (annualized)", f"{alpha*100:,.2f}%")
            d2.metric("Beta", f"{beta:,.2f}")
            d3.metric("R²", f"{r2:,.2f}")
            d4.metric("Correlation", f"{corr:,.2f}")

            st.subheader("Charts")
            plot_equity(eq, bench_eq, js.get("meta", {}).get("name", "Strategy"))
            plot_drawdown(eq)

            with st.expander("Show target weights (last 10 rows)"):
                st.dataframe(weights.tail(10).style.format("{:.2%}"))

        except Exception as e:
            st.error(f"Error: {e}")
