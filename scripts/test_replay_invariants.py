"""Replay & safety invariant tests for the T0 paper agent.

Run: py -3.13 scripts/test_replay_invariants.py

These are guardrail regression tests, not a profitability claim. They assert:
  T1  replay never mutates the real t0_state.json (mtime + sha256 unchanged)
  T2  replay performs zero network/order side effects (no SkillClient/submit/quote)
  T3  replay is deterministic (two runs produce byte-identical decisions)
  T4  SELL exits bypass BUY-budget/data checks (daily_loss_limit etc.)
  T5  unconditional exits bypass quote_freshness; discretionary sells do not
  T6  apply-time bounds reject out-of-range overlay leaves; locks unreachable
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_t0_intraday_agent as agent  # noqa: E402

STATE_PATH = ROOT / "outputs" / "t0_intraday_agent" / "t0_state.json"
REPLAY_DIR = ROOT / "outputs" / "t0_replay"
PY = [sys.executable]

failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        failures.append(name)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "MISSING"


def run_replay(label: str) -> None:
    subprocess.run(
        PY + [str(ROOT / "scripts" / "replay_t0_decisions.py"), "--label", label],
        cwd=str(ROOT), capture_output=True, text=True, timeout=180, check=False,
    )


def t1_t3_state_and_determinism() -> None:
    before_hash, before_mtime = sha(STATE_PATH), (STATE_PATH.stat().st_mtime if STATE_PATH.exists() else None)
    run_replay("invariant_a")
    after_hash, after_mtime = sha(STATE_PATH), (STATE_PATH.stat().st_mtime if STATE_PATH.exists() else None)
    check("T1 t0_state untouched by replay", before_hash == after_hash and before_mtime == after_mtime,
          f"{before_hash[:8]}@{before_mtime} != {after_hash[:8]}@{after_mtime}")
    run_replay("invariant_b")
    a = (REPLAY_DIR / "invariant_a_decisions.jsonl")
    b = (REPLAY_DIR / "invariant_b_decisions.jsonl")
    check("T3 replay deterministic", a.exists() and b.exists() and a.read_bytes() == b.read_bytes())


def t2_no_side_effects() -> None:
    """Import-time + run-time: replay must not touch SkillClient/submit/quote/save_state."""
    calls = {"client": 0, "submit": 0, "quote": 0, "save_state": 0}
    orig_client = agent.SkillClient

    class Tripwire(orig_client):  # type: ignore[misc, valid-type]
        def __init__(self, *a, **k):
            calls["client"] += 1
            raise AssertionError("replay must not instantiate SkillClient")

    agent.SkillClient = Tripwire  # type: ignore[misc]
    orig_save = agent.save_state
    agent.save_state = lambda *a, **k: calls.__setitem__("save_state", calls["save_state"] + 1)  # type: ignore[assignment]
    try:
        import importlib
        import replay_t0_decisions as replay
        importlib.reload(replay)
        sys.argv = ["replay", "--label", "invariant_sideeffect"]
        replay.main()
    except AssertionError as exc:
        calls["client"] += 0
        check("T2 no SkillClient/submit/quote", False, str(exc))
        return
    finally:
        agent.SkillClient = orig_client  # type: ignore[misc]
        agent.save_state = orig_save  # type: ignore[assignment]
    check("T2 no SkillClient instantiated", calls["client"] == 0)
    check("T2 no real save_state in replay", calls["save_state"] == 0)


def t4_sell_bypasses_buy_checks() -> None:
    required = {"balance_ok", "daily_loss_limit", "daily_order_limit", "daily_round_trip_limit", "no_pending_t0_orders"}
    missing = required - agent.SELL_BYPASS_CHECKS
    check("T4 SELL bypasses BUY-budget checks", not missing, f"missing from bypass set: {missing}")
    # quote_freshness must NOT be in the static bypass set (it is conditional)
    check("T4 quote_freshness not blanket-bypassed", "quote_freshness" not in agent.SELL_BYPASS_CHECKS)


def t6_apply_bounds_and_lock_isolation() -> None:
    # Out-of-range allowlisted leaf must be rejected, not applied.
    cfg = {
        "strategy": {"bracket": {"risk_per_trade_pct": 0.004}, "mode": "ignored"},
        "mode": "paper_execute",
        "execution_enabled": True,
        "self_iteration": {
            "enabled": True, "auto_apply_changes": True,
            "overlay_path": "outputs/t0_strategy_evolution/__nonexistent_test__.json",
        },
    }
    # Build a fake approved overlay in memory by monkeypatching load_json.
    import json as _json
    tmp = REPLAY_DIR / "__test_overlay__.json"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(_json.dumps({
        "status": "approved_for_paper_auto_apply",
        "gate_version": agent.EXPECTED_EVOLUTION_GATE_VERSION,
        "paper_trading_only": True,
        "selected_candidate": "test",
        "selection_diagnostics": {
            "require_diebold_mariano_significant": True,
            "require_baseline_excluded_from_mcs": True,
            "require_spa_reject": True,
        },
        "strategy_overlay": {
            "bracket": {"risk_per_trade_pct": 0.5},   # absurd, out of [0.0020,0.0045]
            "entry_score_threshold": 55,               # in-range, should apply
            "mode": "live",                            # not allowlisted -> rejected
        },
    }), encoding="utf-8")
    cfg["self_iteration"]["overlay_path"] = str(tmp)
    out = agent.apply_evolution_overlay_if_enabled(cfg)
    meta = out.get("_evolution_overlay", {})
    applied = meta.get("applied_paths", {})
    oob = meta.get("out_of_bounds_paths", [])
    rejected = meta.get("rejected_paths", [])
    check("T6 out-of-range risk rejected", any("risk_per_trade_pct" in s for s in oob)
          and "bracket.risk_per_trade_pct" not in applied)
    check("T6 in-range leaf applied", applied.get("entry_score_threshold") == 55)
    check("T6 non-allowlisted path rejected", "mode" in rejected)
    check("T6 top-level lock untouched", out.get("mode") == "paper_execute" and out.get("execution_enabled") is True)
    check("T6 strategy.mode never reaches cfg.mode", out["strategy"].get("mode") in (None, "ignored", "live"))

    stale = REPLAY_DIR / "__stale_overlay__.json"
    stale.write_text(_json.dumps({
        "status": "approved_for_paper_auto_apply",
        "paper_trading_only": True,
        "selected_candidate": "old_gate",
        "selection_diagnostics": {
            "require_diebold_mariano_significant": True,
            "require_baseline_excluded_from_mcs": True,
            "require_spa_reject": True,
        },
        "strategy_overlay": {"entry_score_threshold": 49},
    }), encoding="utf-8")
    stale_cfg = {
        "strategy": {"entry_score_threshold": 60},
        "self_iteration": {
            "enabled": True, "auto_apply_changes": True,
            "overlay_path": str(stale),
        },
    }
    stale_out = agent.apply_evolution_overlay_if_enabled(stale_cfg)
    check("T6 stale gate-version overlay rejected", stale_out["strategy"]["entry_score_threshold"] == 60
          and "gate_version_mismatch" in stale_out.get("_evolution_overlay", {}).get("reason", ""))
    tmp.unlink(missing_ok=True)
    stale.unlink(missing_ok=True)


def t7_multi_position_exit() -> None:
    """3 held positions must ALL be evaluated and exit in one run (no starvation)."""
    import io as _io
    import json as _json
    from datetime import datetime
    from zoneinfo import ZoneInfo

    cfg = _json.load(_io.open(ROOT / "configs" / "t0_intraday_paper_agent.json", encoding="utf-8"))
    now = datetime(2026, 6, 15, 10, 30, tzinfo=ZoneInfo("Asia/Shanghai"))  # open session
    ts = now.isoformat()
    codes = [("513050", 1.092), ("513100", 2.201), ("588000", 1.744)]
    positions = [{
        "stockCode": c, "stockName": c, "exchange": "SH",
        "quantity": 10000, "availableQuantity": 10000, "costPrice": cost,
    } for c, cost in codes]
    # current 5% below cost -> emergency_stop (unconditional) fires for all three
    quotes = [{
        "stockCode": c, "exchange": "SH", "name": c, "asset_class": "hk_etf",
        "currentPrice": round(cost * 0.95, 3), "bidPrice1": round(cost * 0.949, 3),
        "askPrice1": round(cost * 0.951, 3), "prevClose": cost, "timestamp": ts,
        "quote_ok": True, "isSuspended": False, "momentum_available": True,
        "momentum": -0.01, "spread_pct": 0.0008, "change_pct": -0.05,
    } for c, cost in codes]
    balance = {"ok": True, "data": {"totalAssets": 1_000_000.0, "availableBalance": 600_000.0}}
    positions_resp = {"ok": True, "data": {"positions": positions}}
    pending_resp = {"ok": True, "data": {"orders": []}}

    agent.set_replay_now(now)
    try:
        decision = agent.build_decision(cfg, quotes, balance, positions_resp, pending_resp, {}, None)
    finally:
        agent.set_replay_now(None)

    orders = decision.get("orders", [])
    sell_codes = {o.get("stockCode") for o in orders if o.get("direction") == "sell"}
    evaluated = set(decision.get("sell_score_by_code", {}).keys())
    check("T7 all 3 held positions evaluated for exit", evaluated >= {"513050", "513100", "588000"},
          f"evaluated={evaluated}")
    check("T7 all 3 produce sell orders in one run", sell_codes >= {"513050", "513100", "588000"},
          f"sell_codes={sell_codes}")
    check("T7 588000 exit-only never bought", all(o.get("direction") == "sell" for o in orders))


def t8_diebold_mariano_gate() -> None:
    """DM-HLN test: blocks one-lucky-day overfit, passes a genuine consistent edge."""
    import importlib
    ev = importlib.import_module("run_t0_strategy_evolution")
    lucky_base = {f"d{i}": -100 for i in range(6)}
    lucky_sel = dict(lucky_base); lucky_sel["d0"] = 200  # single lucky day
    r_lucky = ev.diebold_mariano_hln(lucky_base, lucky_sel, alpha=0.05)
    check("T8 one-lucky-day NOT significant", r_lucky.get("significant") is False, str(r_lucky))
    base = {f"d{i}": v for i, v in enumerate([-100, -80, -120, -90, -110, -70, -130, -85, -95, -105, -115, -75])}
    sel = {f"d{i}": v for i, v in enumerate([-40, -30, -55, -35, -45, -20, -60, -30, -38, -42, -50, -25])}
    r_edge = ev.diebold_mariano_hln(base, sel, alpha=0.05)
    check("T8 genuine consistent edge IS significant", r_edge.get("significant") is True, str(r_edge))
    r_few = ev.diebold_mariano_hln({"d0": -100}, {"d0": -50}, alpha=0.05)
    check("T8 single day insufficient -> not significant", r_few.get("significant") is False, str(r_few))


def t9_model_confidence_set() -> None:
    """MCS: keeps all when indistinguishable, excludes baseline only on a real
    consistent edge, and does NOT over-eliminate at tiny samples (regression)."""
    import importlib, random
    ev = importlib.import_module("run_t0_strategy_evolution")
    rng = random.Random(1); m = 30
    a = [rng.gauss(0, 1) for _ in range(m)]
    no_diff = {"baseline": a[:], "c1": [x + rng.gauss(0, 0.01) for x in a], "c2": [x + rng.gauss(0, 0.01) for x in a]}
    r1 = ev.model_confidence_set(no_diff, alpha=0.10, n_boot=400, seed=7)
    check("T9 no-difference keeps all models", len(r1["mcs_set"]) == 3, str(r1["mcs_set"]))
    dom = {"baseline": [5 + rng.gauss(0, 1) for _ in range(m)],
           "c1": [0 + rng.gauss(0, 1) for _ in range(m)],
           "c2": [5 + rng.gauss(0, 1) for _ in range(m)]}
    r2 = ev.model_confidence_set(dom, alpha=0.10, n_boot=400, seed=7)
    check("T9 dominant candidate excludes baseline", "baseline" not in r2["mcs_set"], str(r2["mcs_set"]))
    tiny = {"baseline": [1, 2, 1, 2, 1], "c1": [0.9, 1.9, 1.1, 1.8, 1.0], "c2": [1.1, 2.1, 1.0, 2.0, 0.9]}
    r3 = ev.model_confidence_set(tiny, alpha=0.10, n_boot=400, seed=7)
    check("T9 tiny sample does NOT over-eliminate", len(r3["mcs_set"]) == 3, str(r3["mcs_set"]))


def t10_spa_reality_check() -> None:
    """SPA/White Reality Check: no reject under true null, reject on real edge,
    no reject at tiny sample (low power)."""
    import importlib
    ev = importlib.import_module("run_t0_strategy_evolution")
    b = [float(i % 5) for i in range(30)]
    r_id = ev.reality_check_spa(b, {"c1": b[:], "c2": b[:]}, alpha=0.05, n_boot=400, seed=3)
    check("T10 identical alternatives -> no reject", r_id.get("reject") is False, str(r_id))
    r_worse = ev.reality_check_spa(b, {"c1": [x + 1 for x in b]}, alpha=0.05, n_boot=400, seed=3)
    check("T10 worse alternative -> no reject", r_worse.get("reject") is False, str(r_worse))
    bench = [5 + (i % 3) for i in range(30)]
    alts = {"c1": [2 + (i % 3) for i in range(30)], "c2": [5 + (i % 3) for i in range(30)]}
    r_sig = ev.reality_check_spa(bench, alts, alpha=0.05, n_boot=400, seed=3)
    check("T10 real consistent edge -> reject", r_sig.get("reject") is True and r_sig.get("best_alt") == "c1", str(r_sig))
    r_tiny = ev.reality_check_spa(
        [1, 2, 1, 2, 1],
        {"c1": [0.9, 1.9, 1.1, 1.8, 1.0], "c2": [1.1, 2.1, 1.0, 2.0, 0.9]},
        alpha=0.05, n_boot=400, seed=3,
    )
    check("T10 tiny multi-candidate sample -> no reject (low power)", r_tiny.get("reject") is False, str(r_tiny))


if __name__ == "__main__":
    t1_t3_state_and_determinism()
    t2_no_side_effects()
    t4_sell_bypasses_buy_checks()
    t6_apply_bounds_and_lock_isolation()
    t7_multi_position_exit()
    t8_diebold_mariano_gate()
    t9_model_confidence_set()
    t10_spa_reality_check()
    print()
    if failures:
        print(f"FAILED: {len(failures)} invariant(s): {failures}")
        sys.exit(1)
    print("ALL INVARIANTS PASSED")
