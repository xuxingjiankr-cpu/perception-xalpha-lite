"""Shared paper-trading execution guard.

This module is deliberately narrow: it does not decide whether an order should
be submitted, and it does not bypass any existing agent risk gate. It only
serializes the final paper submit/cancel critical section and records which
agent owned each order intent.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def _shared_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    raw = cfg.get("shared_execution", {})
    if not isinstance(raw, dict):
        raw = {}
    return {
        "enabled": bool(raw.get("enabled", True)),
        "dir": str(raw.get("dir", "outputs/shared_order_router")),
        "lock_file": str(raw.get("lock_file", "shared_execution.lock")),
        "intents_jsonl": str(raw.get("intents_jsonl", "order_intents.jsonl")),
        "ledger": str(raw.get("ledger", "shared_execution_ledger.json")),
        "lock_timeout_seconds": float(raw.get("lock_timeout_seconds", 8)),
        "stale_lock_seconds": float(raw.get("stale_lock_seconds", 180)),
    }


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def _append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")


def tag_order_owner(orders: list[dict[str, Any]], agent_name: str, trade_date: str) -> list[dict[str, Any]]:
    """Tag order dicts in place so downstream blotters preserve ownership."""
    for order in orders:
        if isinstance(order, dict):
            order.setdefault("owner_agent", agent_name)
            order.setdefault("owner_trade_date", trade_date)
    return orders


class FileLock:
    def __init__(self, path: Path, timeout_seconds: float, stale_seconds: float) -> None:
        self.path = path
        self.timeout_seconds = max(0.0, timeout_seconds)
        self.stale_seconds = max(1.0, stale_seconds)
        self.fd: int | None = None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.time() + self.timeout_seconds
        while True:
            try:
                self.fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                payload = json.dumps({"pid": os.getpid(), "created_at": now_iso()}, ensure_ascii=False)
                os.write(self.fd, payload.encode("utf-8"))
                return True
            except FileExistsError:
                self._remove_stale_lock_if_needed()
                if time.time() >= deadline:
                    return False
                time.sleep(0.2)

    def _remove_stale_lock_if_needed(self) -> None:
        try:
            age = time.time() - self.path.stat().st_mtime
            if age > self.stale_seconds:
                self.path.unlink(missing_ok=True)
        except OSError:
            return

    def release(self) -> None:
        if self.fd is not None:
            try:
                os.close(self.fd)
            finally:
                self.fd = None
        try:
            self.path.unlink(missing_ok=True)
        except OSError:
            pass


class SharedExecutionGuard:
    """Context manager that serializes final paper execution across agents."""

    def __init__(
        self,
        cfg: dict[str, Any],
        *,
        agent_name: str,
        trade_date: str,
        orders: list[dict[str, Any]],
        can_execute: bool,
    ) -> None:
        self.cfg = _shared_cfg(cfg)
        self.agent_name = agent_name
        self.trade_date = trade_date
        self.orders = orders
        self.can_execute = bool(can_execute)
        self.base_dir = ROOT / self.cfg["dir"]
        self.lock = FileLock(
            self.base_dir / self.cfg["lock_file"],
            self.cfg["lock_timeout_seconds"],
            self.cfg["stale_lock_seconds"],
        )
        self.allowed = True
        self.report: dict[str, Any] = {
            "enabled": self.cfg["enabled"],
            "attempted": False,
            "allowed": True,
            "agent_name": agent_name,
            "trade_date": trade_date,
            "orders_count": len(orders),
            "reason": "not_in_approved_paper_execute_path",
        }

    def __enter__(self) -> "SharedExecutionGuard":
        if not self.cfg["enabled"]:
            self.report.update({"attempted": False, "allowed": True, "reason": "shared_execution_guard_disabled"})
            return self
        if not self.can_execute or not self.orders:
            self.report.update({"attempted": False, "allowed": True, "reason": "no_submit_path_or_no_orders"})
            return self
        self.report.update({
            "attempted": True,
            "lock_file": str(self.base_dir / self.cfg["lock_file"]),
            "intents_jsonl": str(self.base_dir / self.cfg["intents_jsonl"]),
            "ledger": str(self.base_dir / self.cfg["ledger"]),
        })
        if not self.lock.acquire():
            self.allowed = False
            self.report.update({"allowed": False, "reason": "shared_execution_lock_unavailable"})
            return self
        self.allowed = True
        self.report.update({"allowed": True, "reason": "shared_execution_lock_acquired"})
        self._record_event("execution_lock_acquired", submit_results=None, extra={})
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.report.get("attempted") and self.allowed:
            if exc is not None:
                self._record_event("execution_lock_released_after_exception", submit_results=None, extra={"exception": str(exc)})
            self.lock.release()

    def _record_event(self, event_type: str, submit_results: list[dict[str, Any]] | None, extra: dict[str, Any]) -> None:
        event = {
            "timestamp": now_iso(),
            "event_type": event_type,
            "agent_name": self.agent_name,
            "trade_date": self.trade_date,
            "orders": self.orders,
            "submit_results": submit_results or [],
            **extra,
        }
        _append_jsonl(self.base_dir / self.cfg["intents_jsonl"], event)
        _write_json(self.base_dir / self.cfg["ledger"], event)

    def record_submit_results(
        self,
        submit_results: list[dict[str, Any]],
        *,
        status: str,
        cancel_result: dict[str, Any] | None = None,
    ) -> None:
        if not self.report.get("attempted") or not self.allowed:
            return
        ok_count = sum(1 for x in submit_results if isinstance(x, dict) and x.get("ok"))
        self.report.update({
            "submit_status": status,
            "submitted_count": len(submit_results),
            "submit_ok_count": ok_count,
        })
        self._record_event(
            "submit_results_recorded",
            submit_results=submit_results,
            extra={"status": status, "cancel_result": cancel_result or {}},
        )
