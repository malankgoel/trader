# configuration.py
import json

# ========================
# Core Model Config
# ========================

MODEL_NAME = "gpt-5-2025-08-07"
LLM_MAX_TOKENS = 2000
LLM_TEMPERATURE = 0.0
LLM_REASONING = "low"  # "minimal"|"low"|"medium"|"high"

# ========================
# App Copy
# ========================

FEATURES = """
**What this app does**
- Parse a natural-language strategy into a strict **DSL** (single LLM call).
- Compile DSL **deterministically** into engine JSON (indicators → signals → logic → allocation).
- Validate **clause coverage** (every sentence must map to logic).
- Block **silent fixes**: no defaults that change meaning.
- Run a fast backtest with Yahoo Finance adjusted close data.
- Show a round-trip summary of the compiled strategy.
"""

LIMITATIONS = """
**Limitations / assumptions**
- Only daily equities/ETFs from Yahoo (no intraday/options/FX).
- Primitive building blocks only: sma_price, ema_price, sma_return, std_price, std_return, rsi, drawdown, max_drawdown;
  and signals gt, lt, cross_above, cross_below. If a requested concept isn’t representable, the parser returns NEEDS_CLARIFICATION.
- Simple execution: periodic rebalances, slippage (bps) and per-trade fee, no borrowing/leverage.
- If the model proposes unavailable/invalid tickers, you must edit the DSL or re-parse — we don’t auto-substitute.
"""

DEFAULTS = {
    "start_date": "2018-01-01",
}

ALLOWED = {
    "signals": ["gt", "lt", "cross_above", "cross_below"],
    "indicators": ["sma_price","ema_price","sma_return","std_price","std_return","rsi","drawdown","max_drawdown"],
    "rebalance": ["B","W","M"],
}

# You asked for creativity (e.g., renewables + oil hedge), but we still keep a safe hedge fallback list.
HEDGE_FALLBACKS = ["BIL","SGOV","SHY","IEF","TFLO"]

# ========================
# DSL JSON Schema (for structured output)
# ========================
# The model must fill this. We keep it close to engine concepts but allow wildcards and alternates.
DSL_JSON_SCHEMA = {
    "name": "StrategyDSL",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "dsl_version": {"type": "string", "const": "2.0"},
            "status": {"type": "string", "enum": ["OK", "NEEDS_CLARIFICATION"]},
            "clarifications": {
                "type": "array",
                "items": {"type": "string"},
                "default": []
            },
            "clauses": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "id": {"type": "string"},
                        "text": {"type": "string"}
                    },
                    "required": ["id","text"]
                },
                "minItems": 1
            },
            "coverage": {
                "type": "object",
                "description": "Map clause_id -> array of node ids (signals or 'logic')",
                "additionalProperties": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "assets": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "tickers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 12
                    },
                    "hedge_assets": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": []
                    },
                    "alternates": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "default": {}
                    },
                    "notes": {"type": "string", "default": ""}
                },
                "required": ["tickers"]
            },
            "dates": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "start": {"type": ["string","null"]},
                    "end": {"type": ["string","null"]}
                },
                "required": ["start","end"]
            },
            "rebalance_hint": {"type": "string", "enum": ["B","W","M","none"], "default": "none"},
            "risk_goal": {"type": "string", "enum": ["conservative","balanced","aggressive"], "default": "balanced"},
            "allocation_policy": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "type": {"type": "string", "enum": ["auto","conditional_weights","rules"], "default": "auto"},
                    "when_true": {
                        "oneOf": [
                            {"type": "string", "enum": ["equal","core_satellite"]},
                            {
                                "type": "object",
                                "additionalProperties": {"type": "number"}
                            }
                        ],
                        "default": "equal"
                    },
                    "when_false": {
                        "oneOf": [
                            {"type": "string", "enum": ["hedge_all","equal"]},
                            {
                                "type": "object",
                                "additionalProperties": {"type": "number"}
                            }
                        ],
                        "default": "hedge_all"
                    },
                    "core": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": []
                    },
                    "satellite": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": []
                    }
                }
            },
            "signals": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "id": {"type": "string"},
                        "type": {"type": "string", "enum": ["gt","lt","cross_above","cross_below"]},
                        "fold": {"type": "string", "enum": ["or","and"], "default": "or"},
                        "left": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "kind": {"type": "string", "enum": ["price","indicator","const"]},
                                "symbol": {"type": ["string","null"], "description": "Use '*' to expand per ticker"},
                                "name": {"type": ["string","null"], "description": "indicator name for indicator kind"},
                                "window": {"type": ["integer","null"]},
                                "adjust": {"type": ["boolean","null"]},
                                "value": {"type": ["number","null"]}
                            },
                            "required": ["kind"]
                        },
                        "right": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "kind": {"type": "string", "enum": ["price","indicator","const"]},
                                "symbol": {"type": ["string","null"]},
                                "name": {"type": ["string","null"]},
                                "window": {"type": ["integer","null"]},
                                "adjust": {"type": ["boolean","null"]},
                                "value": {"type": ["number","null"]}
                            },
                            "required": ["kind"]
                        }
                    },
                    "required": ["id","type","left","right"]
                },
                "minItems": 1
            },
            "logic": {
                "oneOf": [
                    {"type": "string"},
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "type": {"type": "string", "enum": ["and","or","not"]},
                            "children": {
                                "type": "array",
                                "items": {"type": ["string","object"]}
                            },
                            "child": {"type": ["string","object"]}
                        },
                        "required": ["type"]
                    }
                ]
            }
        },
        "required": ["dsl_version","status","clauses","coverage","assets","dates","signals","logic","allocation_policy","rebalance_hint","risk_goal"]
    }
}

# ========================
# Parsing Prompt (single pass)
# ========================

DSL_PARSE_PROMPT = """
You parse a free-text trading strategy into a precise DSL **in one pass**.

General rules:
- Use US-listed, liquid equities and/or ETFs relevant to the text (e.g., “renewables” → TAN, ICLN, QCLN; “semis”; “AI”; “oil ETFs” → XLE, XOP, USO).
- Be practical and pick at most 6–10 primary tickers. Provide alternates where useful.
- Represent logic only with the primitives allowed (indicators: sma_price, ema_price, sma_return, std_price, std_return, rsi, drawdown, max_drawdown; signals: gt, lt, cross_above, cross_below).
- If any requested behavior isn't representable (ranking/top-k/stops), return status=NEEDS_CLARIFICATION with clarifications.
- Prefer wildcard expansion: use {"symbol":"*"} in signal sides that apply to each asset. Provide a "fold" ("or" or "and") to tell us how to combine the expansions.

Interpretation tips:
- “Trend/momentum” → price vs SMA(200) or SMA(50/200) comparisons.
- “Mean reversion/oversold” → RSI(14) < 30.
- “Volatility spike/defensive” → std_return(20) > 0.02.
- “Hedge in cash/oil/treasuries” → set hedge_assets accordingly.
- Allocation:
  - If hedging is specified, set allocation_policy.type to "conditional_weights"; when_true typically "equal"; when_false "hedge_all".
  - For “core-satellite”, set when_true="core_satellite" and identify core and satellite lists.
- Dates:
  - If no dates are present, set start to null and end to null (the app will use defaults).
- Rebalance:
  - If mentioned, set rebalance_hint accordingly ("B","W","M"), else "none".
- Coverage:
  - Split the user’s text into short clause snippets. Every clause must be covered by ≥1 node id (signals or the word "logic").

Return **strict JSON** conforming to the provided schema. No prose, no code fences.
""".strip()
