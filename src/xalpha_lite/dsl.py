"""Small declarative factor language; generated code cannot execute Python."""

from __future__ import annotations

import copy
import json
from typing import Any

import numpy as np
import pandas as pd


NODE_KINDS = ("field", "unary", "binary", "rolling", "lag", "corr", "zscore")


def canonical(expression: dict[str, Any]) -> str:
    return json.dumps(expression, sort_keys=True, separators=(",", ":"))


def fields_used(expression: dict[str, Any]) -> set[str]:
    found: set[str] = set()
    if "field" in expression:
        found.add(str(expression["field"]))
    for key in ("arg", "left", "right"):
        if isinstance(expression.get(key), dict):
            found |= fields_used(expression[key])
    return found


def validate(
    expression: Any,
    allowed_fields: set[str],
    allowed_windows: set[int],
    maximum_depth: int = 8,
) -> None:
    def walk(node: Any, depth: int) -> None:
        if not isinstance(node, dict) or depth > maximum_depth:
            raise ValueError("invalid expression or excessive depth")
        kinds = [kind for kind in NODE_KINDS if kind in node]
        if len(kinds) != 1:
            raise ValueError("each node must contain exactly one operator")
        kind = kinds[0]
        if kind == "field":
            if node["field"] not in allowed_fields:
                raise ValueError(f"field not allowed: {node['field']}")
            return
        if kind == "unary":
            if node[kind] not in {"abs", "neg", "signed_log1p", "tanh"}:
                raise ValueError("unary operator not allowed")
            walk(node["arg"], depth + 1)
            return
        if kind == "binary":
            if node[kind] not in {"add", "sub", "mul", "div"}:
                raise ValueError("binary operator not allowed")
            walk(node["left"], depth + 1)
            walk(node["right"], depth + 1)
            return
        if kind == "lag":
            if int(node[kind]) < 0 or int(node[kind]) > 252:
                raise ValueError("lag must be past-only")
            walk(node["arg"], depth + 1)
            return
        if int(node.get("window", 0)) not in allowed_windows:
            raise ValueError("window not preregistered")
        if kind == "corr":
            walk(node["left"], depth + 1)
            walk(node["right"], depth + 1)
        else:
            walk(node["arg"], depth + 1)

    walk(copy.deepcopy(expression), 1)


def evaluate(expression: dict[str, Any], panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if "field" in expression:
        return panel[expression["field"]].copy()
    if "unary" in expression:
        value = evaluate(expression["arg"], panel)
        operation = expression["unary"]
        if operation == "abs":
            return value.abs()
        if operation == "neg":
            return -value
        if operation == "tanh":
            return pd.DataFrame(np.tanh(value), index=value.index, columns=value.columns)
        return np.sign(value) * np.log1p(value.abs())
    if "binary" in expression:
        left, right = evaluate(expression["left"], panel), evaluate(expression["right"], panel)
        operation = expression["binary"]
        if operation == "add":
            return left + right
        if operation == "sub":
            return left - right
        if operation == "mul":
            return left * right
        return left / right.replace(0.0, np.nan)
    if "lag" in expression:
        return evaluate(expression["arg"], panel).shift(int(expression["lag"]))
    if "corr" in expression:
        left, right = evaluate(expression["left"], panel), evaluate(expression["right"], panel)
        window = int(expression["window"])
        return left.rolling(window, min_periods=window).corr(right)
    value = evaluate(expression["arg"], panel)
    window = int(expression["window"])
    rolling = value.rolling(window, min_periods=max(2, window // 2))
    if "rolling" in expression:
        return getattr(rolling, expression["rolling"])()
    mean, std = rolling.mean(), rolling.std().replace(0.0, np.nan)
    return (value - mean) / std

