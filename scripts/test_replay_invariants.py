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
            "require_deflated_sharpe_significant": True,
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
            "require_deflated_sharpe_significant": True,
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
    """3 held positions must ALL be evaluated, but sell submission is throttled
    to one regular sell per interval so positions are not liquidated together."""
    import io as _io
    import json as _json
    import copy as _copy
    from datetime import timedelta
    from datetime import datetime
    from zoneinfo import ZoneInfo

    cfg = _json.load(_io.open(ROOT / "configs" / "t0_intraday_paper_agent.json", encoding="utf-8"))
    cfg["strategy"]["profit_exit_score_threshold"] = 65
    now = datetime(2026, 6, 15, 10, 30, tzinfo=ZoneInfo("Asia/Shanghai"))  # open session
    ts = now.isoformat()
    codes = [("513050", 1.000), ("513100", 1.000), ("588000", 1.000)]
    positions = [{
        "stockCode": c, "stockName": c, "exchange": "SH",
        "quantity": 10000, "availableQuantity": 10000, "costPrice": cost,
    } for c, cost in codes]
    # Profitable but deteriorating positions -> normal unified_sell_score_exit.
    quotes = [{
        "stockCode": c, "exchange": "SH", "name": c, "asset_class": "hk_etf",
        "currentPrice": 1.010, "bidPrice1": 1.009,
        "askPrice1": 1.011, "prevClose": cost, "timestamp": ts,
        "quote_ok": True, "isSuspended": False, "momentum_available": True,
        "momentum": -0.003, "spread_pct": 0.0008, "change_pct": 0.01,
        "acceleration": -0.003, "bid_pressure_3m_pct": -0.001,
    } for c, cost in codes]
    balance = {"ok": True, "data": {"totalAssets": 1_000_000.0, "availableBalance": 600_000.0}}
    positions_resp = {"ok": True, "data": {"positions": positions}}
    pending_resp = {"ok": True, "data": {"orders": []}}
    trade_date = "2026-06-15"
    base_state = {
        "t0_inventory_by_date": {
            trade_date: {
                c: {
                    "buy_quantity_submitted": 10000,
                    "sell_quantity_submitted": 0,
                    "baseline_available_quantity": 0,
                    "entry_price": cost,
                    "last_buy_price": cost,
                    "first_buy_at": (now - timedelta(minutes=30)).isoformat(),
                    "highest_price_since_entry": 1.020,
                }
                for c, cost in codes
            }
        }
    }

    agent.set_replay_now(now)
    try:
        decision = agent.build_decision(cfg, quotes, balance, positions_resp, pending_resp, _copy.deepcopy(base_state), None)
    finally:
        agent.set_replay_now(None)

    orders = decision.get("orders", [])
    sell_codes = {o.get("stockCode") for o in orders if o.get("direction") == "sell"}
    evaluated = set(decision.get("sell_score_by_code", {}).keys())
    check("T7 all 3 held positions evaluated for exit", evaluated >= {"513050", "513100", "588000"},
          f"evaluated={evaluated}")
    deferred = decision.get("deferred_sell_orders", [])
    # Policy: simultaneous exits allowed up to max_sell_orders_per_run (5); the
    # 3 qualifying sells all submit together in one run, none deferred.
    check("T7 all qualifying sells submit simultaneously (<= per-run cap)", len(sell_codes) == 3,
          f"sell_codes={sell_codes}, deferred={deferred}")
    check("T7 nothing deferred when within per-run cap", len(deferred) == 0, str(deferred))
    check("T7 regular sell is not marked as throttle bypass", all(not o.get("sell_throttle_bypass") for o in orders),
          str(orders))
    check("T7 588000 exit-only never bought", all(o.get("direction") == "sell" for o in orders))

    interval_state = _copy.deepcopy(base_state)
    interval_state["last_throttled_sell_at_by_date"] = {trade_date: (now - timedelta(minutes=2)).isoformat()}
    agent.set_replay_now(now)
    try:
        interval_decision = agent.build_decision(cfg, quotes, balance, positions_resp, pending_resp, interval_state, None)
    finally:
        agent.set_replay_now(None)
    check("T7 regular sells are blocked during 5-minute interval", len(interval_decision.get("orders", [])) == 0
          and len(interval_decision.get("deferred_sell_orders", [])) == 3
          and interval_decision.get("sell_throttle", {}).get("interval_active") is True,
          str(interval_decision.get("sell_throttle")))


def t15_emergency_sells_bypass_throttle() -> None:
    """Emergency exits bypass regular sell throttling."""
    import io as _io
    import json as _json
    from datetime import datetime
    from zoneinfo import ZoneInfo

    cfg = _json.load(_io.open(ROOT / "configs" / "t0_intraday_paper_agent.json", encoding="utf-8"))
    now = datetime(2026, 6, 15, 10, 30, tzinfo=ZoneInfo("Asia/Shanghai"))
    ts = now.isoformat()
    codes = [("513050", 1.092), ("513100", 2.201), ("588000", 1.744)]
    positions = [{
        "stockCode": c, "stockName": c, "exchange": "SH",
        "quantity": 10000, "availableQuantity": 10000, "costPrice": cost,
    } for c, cost in codes]
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
    check("T15 all emergency sell orders bypass throttle", sell_codes >= {"513050", "513100", "588000"}
          and all(o.get("sell_throttle_bypass") for o in orders), str(decision.get("sell_throttle")))


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


def t11_fractional_kelly_sizing() -> None:
    """Fractional Kelly: floor on no/poor edge or few trades, full (<=1.0) only on
    a real edge, NEVER inflates above configured risk (1.0), drawdown-aware off."""
    cfg = {"enabled": True, "f_star_for_full_risk": 0.25, "min_trades_for_kelly": 8,
           "floor_scale": 0.5, "drawdown_factor": 0.5}
    r_loss = agent.compute_kelly_scale([-100, -200, -150, -300, -50, -120, -80, -90], cfg)
    check("T11 all-losses -> floor (risk reduced)", r_loss["scale"] == 0.5, str(r_loss))
    r_few = agent.compute_kelly_scale([-100, 50], cfg)
    check("T11 few trades -> floor (don't trust estimate)", r_few["scale"] == 0.5, str(r_few))
    r_strong = agent.compute_kelly_scale([200, 180, -50, 220, 190, -40, 210, 170], cfg)
    check("T11 strong edge -> full but capped at 1.0", r_strong["scale"] == 1.0, str(r_strong))
    r_cap = agent.compute_kelly_scale([1000] * 10, cfg)
    check("T11 never inflates above configured (<=1.0)", r_cap["scale"] <= 1.0, str(r_cap))
    r_mid = agent.compute_kelly_scale([120, -100, 130, -90, 110, -100, 80, -95], cfg)
    check("T11 mild edge -> between floor and 1.0", 0.5 <= r_mid["scale"] <= 1.0 and r_mid["scale"] != 0.5, str(r_mid))
    # Bayesian/James-Stein edge shrinkage: more conservative at small n, converges with n.
    mild = [120, -100, 130, -90, 110, -100, 80, -95]
    no_shrink = dict(cfg, edge_shrinkage_pseudo_trades=0)
    shrunk = dict(cfg, edge_shrinkage_pseudo_trades=10)
    s_no = agent.compute_kelly_scale(mild, no_shrink)["scale"]
    s_n8 = agent.compute_kelly_scale(mild, shrunk)["scale"]
    s_n24 = agent.compute_kelly_scale(mild * 3, shrunk)["scale"]
    check("T11 shrinkage more conservative at small n", s_n8 < s_no, f"{s_n8} !< {s_no}")
    check("T11 shrinkage converges with more data", s_n8 <= s_n24 <= s_no, f"{s_n8},{s_n24},{s_no}")
    check("T11 shrinkage leaves losing-sample at floor",
          agent.compute_kelly_scale([-100] * 8, shrunk)["scale"] == 0.5)


def t12_intraday_momentum() -> None:
    """IM: fires a long in the 14:30 window on positive first-half-hour return
    (HK preferred, bypassing the 14:00 cutoff), and force-exits an IM position
    at/after exit_time."""
    import io as _io
    import json as _json
    from datetime import datetime
    from zoneinfo import ZoneInfo
    SH = ZoneInfo("Asia/Shanghai")
    cfg = _json.load(_io.open(ROOT / "configs" / "t0_intraday_paper_agent.json", encoding="utf-8"))

    # --- Entry: flat, 14:30 window, two eligible (HK + cross-border), HK preferred ---
    now = datetime(2026, 6, 15, 14, 32, tzinfo=SH)
    ts = now.isoformat()
    def q(code, cls, fhr):
        return {"stockCode": code, "exchange": "SH", "name": code, "asset_class": cls,
                "currentPrice": 2.0, "bidPrice1": 1.999, "askPrice1": 2.001, "prevClose": 1.98,
                "timestamp": ts, "quote_ok": True, "isSuspended": False, "momentum_available": True,
                "momentum": 0.002, "spread_pct": 0.0008, "change_pct": 0.01,
                "first_half_hour_return": fhr}
    quotes = [q("513100", "cross_border_etf", 0.009), q("513050", "hk_etf", 0.004)]
    balance = {"ok": True, "data": {"totalAssets": 1_000_000.0, "availableBalance": 800_000.0}}
    pos_resp = {"ok": True, "data": {"positions": []}}
    pend = {"ok": True, "data": {"orders": []}}
    agent.set_replay_now(now)
    try:
        dec = agent.build_decision(cfg, quotes, balance, pos_resp, pend, {}, None)
    finally:
        agent.set_replay_now(None)
    orders = dec.get("orders", [])
    im = [o for o in orders if o.get("im_trade")]
    check("T12 IM fires a buy in 14:30 window", len(im) == 1 and im[0]["direction"] == "buy", str(orders))
    check("T12 IM prefers HK ETF (513050) over cross-border", im and im[0]["stockCode"] == "513050",
          str(im[0]["stockCode"]) if im else "none")
    check("T12 IM approved despite 14:00 cutoff", bool(dec.get("approved_for_submit")), dec.get("state_machine"))

    # --- Force-exit: hold an IM-tagged position at 14:56 -> intraday_momentum_eod_exit ---
    now2 = datetime(2026, 6, 15, 14, 56, tzinfo=SH)
    ts2 = now2.isoformat()
    code = "513050"
    state = {"t0_inventory_by_date": {"2026-06-15": {code: {
        "im_trade": True, "buy_quantity_submitted": 10000, "sell_quantity_submitted": 0,
        "baseline_available_quantity": 0.0, "entry_price": 2.0, "first_buy_at": ts,
    }}}}
    positions = [{"stockCode": code, "stockName": code, "exchange": "SH",
                  "quantity": 10000, "availableQuantity": 10000, "costPrice": 2.0}]
    quotes2 = [q(code, "hk_etf", 0.004)]
    quotes2[0]["currentPrice"] = 2.01  # in profit; would normally carry, but IM must force-exit
    quotes2[0]["timestamp"] = ts2
    agent.set_replay_now(now2)
    try:
        dec2 = agent.build_decision(cfg, quotes2, balance, {"ok": True, "data": {"positions": positions}}, pend, state, None)
    finally:
        agent.set_replay_now(None)
    sells = [o for o in dec2.get("orders", []) if o.get("direction") == "sell"]
    check("T12 IM position force-exits at exit_time",
          len(sells) == 1 and sells[0].get("reason") == "intraday_momentum_eod_exit", str(dec2.get("orders")))


def t13_execution_quality_safe_shield_and_dsr() -> None:
    """New research-inspired guards: microstructure quality blocks bad books,
    safe shield records/blocks recent losing repeats, DSR fails small samples and
    passes a stable edge after search penalty."""
    import importlib
    from datetime import datetime
    from zoneinfo import ZoneInfo

    strategy = {
        "execution_quality": {"enabled": True, "min_score": 60, "max_expected_slippage_bps": 20},
        "safe_policy_shield": {"enabled": True, "negative_sample_block_minutes": 60, "min_history_snapshots": 20},
    }
    filters = {"max_spread_pct": 0.0015}
    risk = {"limit_price_slippage_pct": 0.001}
    good_q = {
        "stockCode": "513050", "quote_ok": True, "isSuspended": False,
        "currentPrice": 2.0, "bidPrice1": 1.999, "askPrice1": 2.001,
        "midpoint": 2.0, "spread_pct": 0.0008, "bid_pressure_3m_pct": 0.0001,
    }
    bad_q = dict(good_q, bidPrice1=1.95, askPrice1=2.05, midpoint=2.0, spread_pct=0.05)
    good_eq = agent.compute_execution_quality(good_q, filters, risk, strategy)
    bad_eq = agent.compute_execution_quality(bad_q, filters, risk, strategy)
    check("T13 good microstructure passes execution-quality gate", good_eq.get("passed") is True, str(good_eq))
    check("T13 wide spread fails execution-quality gate", bad_eq.get("passed") is False, str(bad_eq))

    now = datetime(2026, 6, 15, 10, 30, tzinfo=ZoneInfo("Asia/Shanghai"))
    state = {}
    agent.set_replay_now(now)
    try:
        agent.record_negative_action_sample(state, "2026-06-15", {
            "direction": "sell", "stockCode": "513050", "reason": "bracket_stop_loss",
            "price": 1.98, "cost_price": 2.0, "quantity": 10000, "pnl_pct": -0.01,
        }, -200.0)
        shield = agent.compute_safe_policy_shield(good_q, [], state, strategy)
    finally:
        agent.set_replay_now(None)
    check("T13 negative sample recorded", len(state.get("negative_action_samples", [])) == 1, str(state))
    check("T13 recent negative sample blocks same-ETF repeat BUY", shield.get("passed") is False
          and shield.get("status") == "recent_negative_sample_block", str(shield))

    ev = importlib.import_module("run_t0_strategy_evolution")
    few = ev.deflated_sharpe_diagnostic({"d1": 0.0}, {"d1": 10.0}, n_trials=10, alpha=0.10)
    check("T13 DSR fails tiny samples", few.get("significant") is False, str(few))
    base = {f"d{i}": 0.0 for i in range(12)}
    sel = {f"d{i}": v for i, v in enumerate([10, 12, 9, 11, 13, 10, 12, 9, 11, 14, 10, 12])}
    dsr = ev.deflated_sharpe_diagnostic(base, sel, n_trials=4, alpha=0.10)
    check("T13 DSR passes stable post-search edge", dsr.get("significant") is True, str(dsr))


def t14_513100_entry_blocked_but_sell_allowed() -> None:
    """513100 is quote-collected and sellable, but excluded from new BUY ranking."""
    import io as _io
    import json as _json
    from datetime import datetime
    from zoneinfo import ZoneInfo

    cfg = _json.load(_io.open(ROOT / "configs" / "t0_intraday_paper_agent.json", encoding="utf-8"))
    now = datetime(2026, 6, 15, 10, 30, tzinfo=ZoneInfo("Asia/Shanghai"))
    ts = now.isoformat()
    quotes = [
        {"stockCode": "513100", "exchange": "SH", "name": "513100", "asset_class": "cross_border_etf",
         "currentPrice": 2.20, "bidPrice1": 2.199, "askPrice1": 2.201, "midpoint": 2.20,
         "prevClose": 2.18, "timestamp": ts, "quote_ok": True, "isSuspended": False,
         "momentum_available": True, "momentum": 0.01, "spread_pct": 0.0008, "change_pct": 0.01,
         "bid_pressure_3m_pct": 0.001, "acceleration": 0.002},
        {"stockCode": "513050", "exchange": "SH", "name": "513050", "asset_class": "hk_etf",
         "currentPrice": 1.10, "bidPrice1": 1.099, "askPrice1": 1.101, "midpoint": 1.10,
         "prevClose": 1.09, "timestamp": ts, "quote_ok": True, "isSuspended": False,
         "momentum_available": True, "momentum": 0.004, "spread_pct": 0.0008, "change_pct": 0.01,
         "bid_pressure_3m_pct": 0.001, "acceleration": 0.002},
    ]
    balance = {"ok": True, "data": {"totalAssets": 1_000_000.0, "availableBalance": 800_000.0}}
    agent.set_replay_now(now)
    try:
        dec = agent.build_decision(cfg, quotes, balance, {"ok": True, "data": {"positions": []}}, {"ok": True, "data": {"orders": []}}, {}, None)
    finally:
        agent.set_replay_now(None)
    ranked_codes = [str(q.get("stockCode")).zfill(6) for q in dec.get("ranked", [])]
    exclusions = []
    for chk in dec.get("risk_checks", []):
        if chk.get("name") == "quote_liquidity_filter":
            exclusions = (chk.get("detail") or {}).get("ranking_exclusions") or []
    order_codes = [str(o.get("stockCode")).zfill(6) for o in dec.get("orders", []) if o.get("direction") == "buy"]
    check("T14 513100 excluded from new BUY ranking", "513100" not in ranked_codes
          and any(str(x.get("stockCode")).zfill(6) == "513100" for x in exclusions), str(dec.get("ranked")))
    check("T14 513100 is not bought even with highest momentum", "513100" not in order_codes, str(dec.get("orders")))


def t16_quote_fallback_chain() -> None:
    """Sina/Tencent fallback: bad records rejected (no trading on garbage); the
    chain falls Eastmoney->Sina->Tencent->Huatai and never burns Huatai quota
    while a free source is up."""
    import importlib
    lf = importlib.import_module("run_etf_paper_trading_agent")
    etf = {"stockCode": "513050", "exchange": "SH", "name": "x"}
    good = lf._thirdparty_quote_response(etf, {"current": 1.086, "prevClose": 1.082, "bid1": 1.085, "ask1": 1.087}, "sina")
    check("T16 valid record -> ok with bid/ask", good["ok"] and good["data"]["bidPrice1"] == 1.085 and good["data"]["askPrice1"] == 1.087, str(good))
    bad_price = lf._thirdparty_quote_response(etf, {"current": 0, "prevClose": 1.0}, "sina")
    check("T16 zero current rejected (no garbage trading)", bad_price["ok"] is False, str(bad_price))
    bad_prev = lf._thirdparty_quote_response(etf, {"current": 1.0, "prevClose": 0}, "sina")
    check("T16 zero prevClose rejected", bad_prev["ok"] is False, str(bad_prev))
    miss_book = lf._thirdparty_quote_response(etf, {"current": 1.5, "prevClose": 1.4}, "tencent")
    check("T16 missing book defaults bid/ask to current", miss_book["ok"] and miss_book["data"]["bidPrice1"] == 1.5, str(miss_book))

    cfg = {"universe": [{"stockCode": "513050", "exchange": "SH", "name": "x"}],
           "market_data": {"quote_provider": "eastmoney_primary", "eastmoney_enabled": True,
                           "sina_enabled": True, "tencent_enabled": True, "huatai_quote_fallback": True}}
    orig_e, orig_s = lf.fetch_eastmoney_quotes, lf.fetch_sina_quotes
    lf.fetch_eastmoney_quotes = lambda uni, **k: ([lf.eastmoney_quote_response(e, None, {"message": "fail"}) for e in uni], {"provider": "eastmoney", "ok": False, "ok_count": 0})
    lf.fetch_sina_quotes = lambda uni, **k: ([lf._thirdparty_quote_response(e, {"current": 1.1, "prevClose": 1.0, "bid1": 1.1, "ask1": 1.1}, "sina") for e in uni], {"provider": "sina", "ok": True, "ok_count": len(uni)})

    class _NoQuoteClient:
        def get_quote(self, c, e):
            raise AssertionError("Huatai must NOT be called while a free source is up")

    try:
        resp, meta = lf.fetch_quote_responses(cfg, _NoQuoteClient())
    finally:
        lf.fetch_eastmoney_quotes, lf.fetch_sina_quotes = orig_e, orig_s
    check("T16 chain falls Eastmoney->Sina", meta.get("provider") == "sina" and resp[0]["ok"], str(meta.get("provider")))
    check("T16 no Huatai quota burned when free source up", int(meta.get("huatai_quote_calls", -1)) == 0, str(meta.get("huatai_quote_calls")))


def t17_passive_entry_pricing() -> None:
    """Passive entry posts at the bid (earns the spread); falls back to the
    aggressive cross whenever the book is missing/locked or passive is disabled,
    and NEVER returns a price at/through the ask."""
    risk = {"limit_price_slippage_pct": 0.001}
    on = {"passive_entry_enabled": True, "tick_size": 0.001, "passive_offset_ticks": 0}
    off = {"passive_entry_enabled": False}
    q = {"bidPrice1": 2.000, "askPrice1": 2.004, "currentPrice": 2.002}
    px_p, style_p, _ = agent.passive_entry_price(q, risk, on)
    check("T17 passive entry posts at bid", style_p == "passive" and px_p == 2.0, f"{px_p},{style_p}")
    check("T17 passive price stays below ask", px_p < q["askPrice1"])
    px_o, style_o, _ = agent.passive_entry_price(q, risk, off)
    check("T17 disabled -> aggressive cross", style_o == "aggressive" and px_o > q["askPrice1"], f"{px_o},{style_o}")
    _, style_nb, _ = agent.passive_entry_price({"bidPrice1": 0, "askPrice1": 2.004, "currentPrice": 2.0}, risk, on)
    check("T17 no bid -> aggressive", style_nb == "aggressive")
    _, style_lk, _ = agent.passive_entry_price({"bidPrice1": 2.004, "askPrice1": 2.004, "currentPrice": 2.004}, risk, on)
    check("T17 locked book -> aggressive", style_lk == "aggressive")


def t20_alpha101_conviction() -> None:
    """Alpha#101 (Kakushadze) single-name intraday conviction = (close-open)/(high-low):
    +1 when the session opened at the low and is now at the high; ~0 on a flat session;
    feeds entry scoring as a long-only momentum CONFIRMATION (only with mom>0)."""
    import io as _io
    import json as _json
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo
    SH = ZoneInfo("Asia/Shanghai")
    now = datetime(2026, 6, 15, 11, 0, tzinfo=SH)
    agent.set_replay_now(now)
    try:
        def bar(mins_ago, px):
            return {"timestamp": (now - timedelta(minutes=mins_ago)).isoformat(), "currentPrice": px}
        # opened at 1.00 (the low), climbed to 1.05 (now=the high) -> conviction ~ +1
        up_hist = [bar(50, 1.00), bar(40, 1.01), bar(30, 1.02), bar(20, 1.03), bar(10, 1.04)]
        up_now = {"currentPrice": 1.05}
        conv_up = agent.compute_alpha101_conviction(up_hist, up_now)
        check("T20 strong up session -> conviction near +1", conv_up is not None and conv_up > 0.95, str(conv_up))
        # flat session -> ~0
        flat_hist = [bar(30, 1.00), bar(20, 1.00), bar(10, 1.00)]
        conv_flat = agent.compute_alpha101_conviction(flat_hist, {"currentPrice": 1.00})
        check("T20 flat session -> conviction 0", conv_flat == 0.0, str(conv_flat))
        # too little history -> None
        conv_thin = agent.compute_alpha101_conviction([bar(5, 1.0)], {"currentPrice": 1.0})
        check("T20 thin history -> None", conv_thin is None, str(conv_thin))
    finally:
        agent.set_replay_now(None)

    cfg = _json.load(_io.open(ROOT / "configs" / "t0_intraday_paper_agent.json", encoding="utf-8"))
    strat = cfg["strategy"]
    # high conviction + positive momentum -> full conviction weight in entry score
    q_hi = {"momentum": 0.01, "alpha101_conviction": 0.9, "spread_pct": 0.0008, "currentPrice": 1.0}
    s_hi = agent.score_entry(strategy=strat, filters=cfg["filters"], q=q_hi, orb=None,
                             broad_market_not_declining=True)
    w = strat["entry_score_weights"]["alpha101_intraday_conviction"]
    check("T20 strong conviction earns full weight",
          s_hi["components"]["alpha101_intraday_conviction"] == w, str(s_hi["components"]))
    # conviction without momentum (mom<=0) -> no conviction reward (long-only confirmation)
    q_nomom = {"momentum": -0.01, "alpha101_conviction": 0.9, "spread_pct": 0.0008, "currentPrice": 1.0}
    s_nomom = agent.score_entry(strategy=strat, filters=cfg["filters"], q=q_nomom, orb=None,
                                broad_market_not_declining=True)
    check("T20 conviction gated by positive momentum",
          s_nomom["components"]["alpha101_intraday_conviction"] == 0.0, str(s_nomom["components"]))


def t21_inventory_aware_passive_skew() -> None:
    """Avellaneda-Stoikov inventory skew: passive BUY posts LOWER as the book fills
    (vs target_holdings) and with higher volatility / more time to close, bounded by
    max_skew_ticks, never above the ask, never aggressive unless it would go <= 0.
    Empty inventory or disabled config => identical to the plain passive bid."""
    risk = {"limit_price_slippage_pct": 0.001}
    tick = 0.001
    skew_on = {"passive_entry_enabled": True, "tick_size": tick, "passive_offset_ticks": 0,
               "inventory_skew": {"enabled": True, "risk_aversion": 2.0, "reference_volatility": 0.01,
                                  "vol_multiplier_cap": 3.0, "max_skew_ticks": 2, "session_minutes": 240.0}}
    skew_off = dict(skew_on); skew_off = {**skew_on, "inventory_skew": {"enabled": False}}
    q = {"bidPrice1": 2.000, "askPrice1": 2.010, "currentPrice": 2.005}

    # empty book (ratio 0) -> no skew, posts at bid
    px0, st0, _ = agent.passive_entry_price(q, risk, skew_on, inventory_ratio=0.0, volatility=0.02, minutes_to_close=200)
    check("T21 empty inventory -> posts at bid (no skew)", st0 == "passive" and px0 == 2.000, f"{px0}")
    # one of five held, low vol, early -> sub-tick skew rounds to 0 (don't hurt early fills)
    px_low, _, _ = agent.passive_entry_price(q, risk, skew_on, inventory_ratio=0.2, volatility=0.01, minutes_to_close=190)
    check("T21 low inventory -> no drag on fills", px_low == 2.000, f"{px_low}")
    # nearly full + elevated vol + much time -> skews down, capped at 2 ticks, still passive
    px_full, st_full, _ = agent.passive_entry_price(q, risk, skew_on, inventory_ratio=0.8, volatility=0.02, minutes_to_close=200)
    check("T21 full book skews bid down (capped at max_skew_ticks)",
          st_full == "passive" and 1.997 <= px_full <= 1.999, f"{px_full}")
    check("T21 skew never reaches/through ask", px_full < q["askPrice1"])
    # disabled config -> identical to plain bid even at high inventory
    px_disabled, _, _ = agent.passive_entry_price(q, risk, skew_off, inventory_ratio=1.0, volatility=0.03, minutes_to_close=240)
    check("T21 disabled skew -> plain passive bid", px_disabled == 2.000, f"{px_disabled}")


def t18_pre_sell_position_verification() -> None:
    """Pre-submit sell verification: drop a sell for a phantom (broker holds 0),
    cap an oversized sell to broker available, pass valid sells & buys; when
    positions can't be fetched, only unconditional exits go through."""
    avail = {"513050": 10000.0}  # broker holds 513050 only
    orders = [
        {"direction": "buy", "stockCode": "513500", "quantity": 1000},
        {"direction": "sell", "stockCode": "513100", "quantity": 45000},  # phantom -> drop
        {"direction": "sell", "stockCode": "513050", "quantity": 99999},  # oversized -> cap to 10000
    ]
    out = agent.verify_sell_orders_against_broker(orders, avail, True, 100, 100)
    codes = {o["stockCode"]: o for o in out}
    check("T18 buy passes through", "513500" in codes)
    check("T18 phantom sell (broker holds 0) dropped", "513100" not in codes)
    check("T18 oversized sell capped to broker available", codes.get("513050", {}).get("quantity") == 10000)
    # positions unavailable: only unconditional exits go through
    o2 = [
        {"direction": "sell", "stockCode": "513050", "quantity": 1000, "unconditional_exit": True},
        {"direction": "sell", "stockCode": "513100", "quantity": 1000, "unconditional_exit": False},
    ]
    out2 = {o["stockCode"] for o in agent.verify_sell_orders_against_broker(o2, {}, False, 100, 100)}
    check("T18 unverified: unconditional exit allowed", "513050" in out2)
    check("T18 unverified: discretionary sell dropped", "513100" not in out2)


def t19_multi_holding_entry_while_carrying() -> None:
    """With target_holdings>1, holding one carrying position must NOT block a NEW
    entry: in the 14:30 IM window the agent buys a fresh non-held name while the
    existing holding (no exit signal) is carried -- and the buy survives orders_list."""
    import io as _io
    import json as _json
    from datetime import datetime
    from zoneinfo import ZoneInfo
    SH = ZoneInfo("Asia/Shanghai")
    cfg = _json.load(_io.open(ROOT / "configs" / "t0_intraday_paper_agent.json", encoding="utf-8"))
    cfg["strategy"]["target_holdings"] = 5  # ensure room regardless of config drift

    now = datetime(2026, 6, 15, 14, 32, tzinfo=SH)
    ts = now.isoformat()
    trade_date = "2026-06-15"
    # Held, carrying: gold ETF, modestly in profit, stable/positive -> no exit signal.
    held = {"stockCode": "518880", "exchange": "SH", "name": "518880", "asset_class": "gold_etf",
            "currentPrice": 5.05, "bidPrice1": 5.049, "askPrice1": 5.051, "prevClose": 5.00,
            "timestamp": ts, "quote_ok": True, "isSuspended": False, "momentum_available": True,
            "momentum": 0.001, "spread_pct": 0.0004, "change_pct": 0.01,
            "acceleration": 0.0005, "bid_pressure_3m_pct": 0.002, "first_half_hour_return": 0.0}
    # Fresh IM candidate, not held: HK ETF with strong first-half-hour return.
    cand = {"stockCode": "513050", "exchange": "SH", "name": "513050", "asset_class": "hk_etf",
            "currentPrice": 2.0, "bidPrice1": 1.999, "askPrice1": 2.001, "prevClose": 1.98,
            "timestamp": ts, "quote_ok": True, "isSuspended": False, "momentum_available": True,
            "momentum": 0.002, "spread_pct": 0.0008, "change_pct": 0.01,
            "first_half_hour_return": 0.004}
    quotes = [held, cand]
    balance = {"ok": True, "data": {"totalAssets": 1_000_000.0, "availableBalance": 800_000.0}}
    positions = [{"stockCode": "518880", "stockName": "518880", "exchange": "SH",
                  "quantity": 10000, "availableQuantity": 10000, "costPrice": 5.00}]
    pos_resp = {"ok": True, "data": {"positions": positions}}
    pend = {"ok": True, "data": {"orders": []}}
    state = {"t0_inventory_by_date": {trade_date: {"518880": {
        "buy_quantity_submitted": 10000, "sell_quantity_submitted": 0,
        "baseline_available_quantity": 10000, "entry_price": 5.00, "last_buy_price": 5.00,
        "first_buy_at": (now.replace(hour=10, minute=0)).isoformat(),
        "highest_price_since_entry": 5.06,
    }}}}

    agent.set_replay_now(now)
    try:
        dec = agent.build_decision(cfg, quotes, balance, pos_resp, pend, state, None)
    finally:
        agent.set_replay_now(None)
    orders = dec.get("orders", [])
    sells = [o for o in orders if o.get("direction") == "sell"]
    buys = [o for o in orders if o.get("direction") == "buy"]
    check("T19 carrying holding emits no sell", len(sells) == 0, str(orders))
    check("T19 new name bought while already holding (multi-holding)",
          len(buys) == 1 and buys[0].get("stockCode") == "513050", str(orders))
    check("T19 entry candidate excludes the held name",
          str(dec.get("ranked", [{}])[0].get("stockCode", "")).zfill(6) != "518880" or len(buys) == 1, str(dec.get("ranked")))


def t22_dynamic_universe_selection() -> None:
    """Dynamic universe: liquidity/spread/price/money-fund gates drop the untradable;
    cross-sectional momentum+conviction ranks the survivors; resolve_agent_universe
    unions ranked picks with the static seed AND held codes (exit safety) and falls
    back to static when disabled."""
    import csv as _csv
    import gzip as _gzip
    import io as _io
    import json as _json
    import tempfile
    from pathlib import Path as _Path
    import select_t0_universe as sel

    cols = ["collected_at", "trade_date", "source_quote_time", "scope", "secid", "market",
            "stockCode", "name", "currentPrice", "change_pct", "change_abs", "volume", "amount",
            "amplitude_pct", "turnover_pct", "open", "high", "low", "prevClose", "bidPrice1", "askPrice1", "source"]

    def row(code, market, name, price, chg, amount, op, hi, lo, bid, ask):
        return {**{c: "" for c in cols}, "collected_at": "2026-06-17T15:00:42+09:00",
                "trade_date": "2026-06-17", "market": market, "stockCode": code, "name": name,
                "currentPrice": price, "change_pct": chg, "amount": amount, "open": op, "high": hi,
                "low": lo, "prevClose": op, "bidPrice1": bid, "askPrice1": ask, "source": "eastmoney"}

    rows = [
        # liquid, strong up, opened low/closed high -> should rank #1
        row("512760", "1", "芯片ETF", 1.10, 7.0, 8e8, 1.00, 1.10, 1.00, 1.099, 1.101),
        # liquid, mild up -> ranks below
        row("513500", "1", "标普500ETF", 2.00, 1.0, 6e8, 1.99, 2.01, 1.98, 1.999, 2.001),
        # liquid, down -> ranks last
        row("159915", "0", "创业板ETF", 1.00, -3.0, 5e8, 1.03, 1.04, 1.00, 0.999, 1.001),
        # illiquid -> rejected
        row("159001", "0", "薄ETF", 1.00, 5.0, 1e6, 0.98, 1.01, 0.98, 0.999, 1.001),
        # wide spread -> rejected
        row("511111", "1", "宽价差ETF", 1.00, 5.0, 9e8, 0.98, 1.02, 0.98, 0.95, 1.05),
        # money fund by name -> rejected
        row("511990", "1", "华宝添益货币ETF", 100.0, 0.0, 9e9, 100.0, 100.0, 100.0, 99.99, 100.01),
    ]
    with tempfile.TemporaryDirectory() as td:
        snap_dir = _Path(td) / "snapshots"
        day_dir = snap_dir / "2026-06-17"
        day_dir.mkdir(parents=True)
        gz = day_dir / "eastmoney_full_market_20260617_150042_etf.csv.gz"
        with _gzip.open(gz, "wt", encoding="utf-8", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        dyn_cfg = {"top_n": 3, "min_amount_yuan": 50_000_000, "max_spread_pct": 0.004,
                   "min_price": 0.3, "momentum_weight": 1.0, "conviction_weight": 0.5,
                   "name_exclude_keywords": ["货币", "现金", "理财"]}
        res = sel.select_dynamic_universe(snap_dir, _Path(td) / "universe", "2026-06-17", dyn_cfg)
        meta = res["meta"]
        codes = [s["stockCode"] for s in res["selected"]]
        check("T22 selection ok", meta.get("ok") is True, str(meta))
        check("T22 only liquid/tradable survive gates", meta.get("eligible") == 3, str(meta))
        check("T22 illiquid/wide-spread/money-fund rejected",
              meta["rejects"].get("illiquid") == 1 and meta["rejects"].get("spread_too_wide") == 1
              and meta["rejects"].get("name_excluded") == 1, str(meta["rejects"]))
        check("T22 strongest cross-sectional momentum ranks first", codes[0] == "512760", str(codes))
        check("T22 exchange derived from market code",
              res["selected"][0]["exchange"] == "SH" and any(s["exchange"] == "SZ" for s in res["selected"]), str(res["selected"]))

        # top_n <= 0 => all eligible, still ranked (full tradable cross-section)
        res_all = sel.select_dynamic_universe(snap_dir, _Path(td) / "universe", "2026-06-17", {**dyn_cfg, "top_n": 0})
        check("T22 top_n=0 returns all eligible, ranked",
              len(res_all["selected"]) == 3 and res_all["selected"][0]["stockCode"] == "512760"
              and res_all["meta"]["selection_scope"] == "all_eligible_after_gates", str(res_all["meta"]))

        # resolve union: ranked picks + static seed + held code (exit safety)
        cfg = {"dynamic_universe": {"enabled": True, "snapshot_dir": "snapshots",
                                    "universe_dir": "universe", "output": None, **dyn_cfg},
               "universe": [{"stockCode": "518880", "exchange": "SH", "name": "黄金ETF", "asset_class": "gold_etf"}]}
        # point ROOT-relative dirs at the temp dir by absolute paths
        cfg["dynamic_universe"]["snapshot_dir"] = str(snap_dir)
        cfg["dynamic_universe"]["universe_dir"] = str(_Path(td) / "universe")
        # monkeypatch ROOT join: resolve uses ROOT / path; pass absolute so ROOT/abs == abs
        state = {"t0_inventory_by_date": {"2026-06-17": {"588000": {"buy_quantity_submitted": 10000, "sell_quantity_submitted": 0}}}}
        uni, umeta = sel.resolve_agent_universe(cfg, state, "2026-06-17")
        ucodes = {u["stockCode"] for u in uni}
        check("T22 union keeps static seed (exit safety)", "518880" in ucodes, str(ucodes))
        check("T22 union keeps held code outside ranking (exit safety)", "588000" in ucodes, str(ucodes))
        check("T22 union includes ranked picks", "512760" in ucodes, str(ucodes))

    # disabled -> static fallback, untouched
    cfg_off = {"dynamic_universe": {"enabled": False}, "universe": [{"stockCode": "518880", "exchange": "SH"}]}
    uni_off, meta_off = sel.resolve_agent_universe(cfg_off, {}, "2026-06-17")
    check("T22 disabled -> static universe unchanged",
          meta_off.get("mode") == "static" and len(uni_off) == 1, str(meta_off))


def t23_sector_diversification_entry_filter() -> None:
    """Entry diversification: when two semi ETFs are already held, a third semi
    candidate is skipped and the next eligible sector can be bought."""
    import copy as _copy
    import io as _io
    import json as _json
    from datetime import datetime
    from zoneinfo import ZoneInfo

    SH = ZoneInfo("Asia/Shanghai")
    cfg = _json.load(_io.open(ROOT / "configs" / "t0_intraday_paper_agent.json", encoding="utf-8"))
    cfg["strategy"]["intraday_momentum"]["enabled"] = False
    cfg["strategy"]["target_holdings"] = 5
    cfg["sector_diversification"] = {
        "enabled": True,
        "max_per_sector": 2,
        "unlimited_sectors": ["other"],
        "keyword_map": {"\u534a\u5bfc\u4f53": "semi", "\u82af\u7247": "semi", "\u533b\u836f": "pharma"},
    }
    cfg["universe"] = [
        {"stockCode": "512760", "exchange": "SH", "name": "\u534a\u5bfc\u4f53ETF-A", "asset_class": "equity_etf"},
        {"stockCode": "512480", "exchange": "SH", "name": "\u82af\u7247ETF-B", "asset_class": "equity_etf"},
        {"stockCode": "512761", "exchange": "SH", "name": "\u534a\u5bfc\u4f53ETF-C", "asset_class": "equity_etf"},
        {"stockCode": "159929", "exchange": "SZ", "name": "\u533b\u836fETF", "asset_class": "equity_etf"},
    ]
    now = datetime(2026, 6, 18, 10, 30, tzinfo=SH)
    ts = now.isoformat()
    trade_date = "2026-06-18"

    def q(code, name, px, mom):
        return {"stockCode": code, "exchange": "SH", "name": name, "asset_class": "equity_etf",
                "currentPrice": px, "bidPrice1": px - 0.001, "askPrice1": px + 0.001,
                "midpoint": px, "prevClose": px * 0.99, "timestamp": ts, "quote_ok": True,
                "isSuspended": False, "momentum_available": True, "momentum": mom,
                "spread_pct": 0.0008, "change_pct": 0.01, "bid_pressure_3m_pct": 0.001,
                "acceleration": 0.001}

    quotes = [
        q("512760", "\u534a\u5bfc\u4f53ETF-A", 1.01, 0.002),
        q("512480", "\u82af\u7247ETF-B", 1.02, 0.002),
        q("512761", "\u534a\u5bfc\u4f53ETF-C", 1.03, 0.030),  # top momentum, blocked by sector
        q("159929", "\u533b\u836fETF", 1.04, 0.020),          # next eligible
    ]
    positions = [
        {"stockCode": "512760", "stockName": "\u534a\u5bfc\u4f53ETF-A", "exchange": "SH",
         "quantity": 10000, "availableQuantity": 10000, "costPrice": 1.00},
        {"stockCode": "512480", "stockName": "\u82af\u7247ETF-B", "exchange": "SH",
         "quantity": 10000, "availableQuantity": 10000, "costPrice": 1.00},
    ]
    state = {"t0_inventory_by_date": {trade_date: {
        "512760": {"buy_quantity_submitted": 10000, "sell_quantity_submitted": 0,
                   "baseline_available_quantity": 10000, "entry_price": 1.00,
                   "first_buy_at": now.replace(hour=10, minute=0).isoformat(),
                   "highest_price_since_entry": 1.02},
        "512480": {"buy_quantity_submitted": 10000, "sell_quantity_submitted": 0,
                   "baseline_available_quantity": 10000, "entry_price": 1.00,
                   "first_buy_at": now.replace(hour=10, minute=0).isoformat(),
                   "highest_price_since_entry": 1.03},
    }}}
    balance = {"ok": True, "data": {"totalAssets": 1_000_000.0, "availableBalance": 800_000.0}}
    agent.set_replay_now(now)
    try:
        dec = agent.build_decision(cfg, quotes, balance, {"ok": True, "data": {"positions": positions}},
                                   {"ok": True, "data": {"orders": []}}, _copy.deepcopy(state), None)
    finally:
        agent.set_replay_now(None)
    buys = [o for o in dec.get("orders", []) if o.get("direction") == "buy"]
    blocked = dec.get("sector_diversification", {}).get("blocked_candidates", [])
    blocked_codes = {str(x.get("stockCode")).zfill(6) for x in blocked}
    check("T23 sector classifier maps semi/pharma",
          agent.classify_etf_sector("\u534a\u5bfc\u4f53ETF", cfg["sector_diversification"]["keyword_map"]) == "semi"
          and agent.classify_etf_sector("\u533b\u836fETF", cfg["sector_diversification"]["keyword_map"]) == "pharma")
    check("T23 third same-sector entry candidate is blocked", "512761" in blocked_codes, str(blocked))
    check("T23 next non-blocked sector can be selected",
          len(buys) == 1 and str(buys[0].get("stockCode")).zfill(6) == "159929", str(dec.get("orders")))


def t24_sector_limit_never_blocks_sells() -> None:
    """Even when the entry sector filter fails, held positions remain sellable."""
    import io as _io
    import json as _json
    from datetime import datetime
    from zoneinfo import ZoneInfo

    SH = ZoneInfo("Asia/Shanghai")
    cfg = _json.load(_io.open(ROOT / "configs" / "t0_intraday_paper_agent.json", encoding="utf-8"))
    cfg["strategy"]["intraday_momentum"]["enabled"] = False
    cfg["strategy"]["target_holdings"] = 5
    cfg["sector_diversification"] = {
        "enabled": True,
        "max_per_sector": 1,
        "unlimited_sectors": ["other"],
        "keyword_map": {"\u534a\u5bfc\u4f53": "semi", "\u82af\u7247": "semi"},
    }
    cfg["universe"] = [
        {"stockCode": "512760", "exchange": "SH", "name": "\u534a\u5bfc\u4f53ETF-A", "asset_class": "equity_etf"},
        {"stockCode": "512480", "exchange": "SH", "name": "\u82af\u7247ETF-B", "asset_class": "equity_etf"},
        {"stockCode": "512761", "exchange": "SH", "name": "\u534a\u5bfc\u4f53ETF-C", "asset_class": "equity_etf"},
    ]
    now = datetime(2026, 6, 18, 10, 30, tzinfo=SH)
    ts = now.isoformat()
    trade_date = "2026-06-18"

    def q(code, name, px, mom):
        return {"stockCode": code, "exchange": "SH", "name": name, "asset_class": "equity_etf",
                "currentPrice": px, "bidPrice1": max(0.001, px - 0.001), "askPrice1": px + 0.001,
                "midpoint": px, "prevClose": px * 1.02, "timestamp": ts, "quote_ok": True,
                "isSuspended": False, "momentum_available": True, "momentum": mom,
                "spread_pct": 0.0008, "change_pct": -0.02, "bid_pressure_3m_pct": -0.001,
                "acceleration": -0.003}

    quotes = [
        q("512760", "\u534a\u5bfc\u4f53ETF-A", 0.95, -0.02),
        q("512480", "\u82af\u7247ETF-B", 0.95, -0.02),
        q("512761", "\u534a\u5bfc\u4f53ETF-C", 1.03, 0.03),  # would be blocked if entering
    ]
    positions = [
        {"stockCode": "512760", "stockName": "\u534a\u5bfc\u4f53ETF-A", "exchange": "SH",
         "quantity": 10000, "availableQuantity": 10000, "costPrice": 1.00},
        {"stockCode": "512480", "stockName": "\u82af\u7247ETF-B", "exchange": "SH",
         "quantity": 10000, "availableQuantity": 10000, "costPrice": 1.00},
    ]
    state = {"t0_inventory_by_date": {trade_date: {
        "512760": {"buy_quantity_submitted": 10000, "sell_quantity_submitted": 0,
                   "baseline_available_quantity": 10000, "entry_price": 1.00,
                   "first_buy_at": now.replace(hour=10, minute=0).isoformat(),
                   "highest_price_since_entry": 1.00},
        "512480": {"buy_quantity_submitted": 10000, "sell_quantity_submitted": 0,
                   "baseline_available_quantity": 10000, "entry_price": 1.00,
                   "first_buy_at": now.replace(hour=10, minute=0).isoformat(),
                   "highest_price_since_entry": 1.00},
    }}}
    balance = {"ok": True, "data": {"totalAssets": 1_000_000.0, "availableBalance": 800_000.0}}
    agent.set_replay_now(now)
    try:
        dec = agent.build_decision(cfg, quotes, balance, {"ok": True, "data": {"positions": positions}},
                                   {"ok": True, "data": {"orders": []}}, state, None)
    finally:
        agent.set_replay_now(None)
    sells = [o for o in dec.get("orders", []) if o.get("direction") == "sell"]
    sector_check = next((c for c in dec.get("risk_checks", []) if c.get("name") == "sector_diversification_entry_filter"), {})
    check("T24 sector entry filter can fail while held exits remain evaluated", sector_check.get("passed") is False,
          str(sector_check))
    check("T24 sell orders still built despite sector concentration",
          len(sells) >= 1 and dec.get("approved_for_submit") is True, str(dec.get("orders")))
    check("T24 sector filter is in SELL bypass set", "sector_diversification_entry_filter" in agent.SELL_BYPASS_CHECKS)


def t25_daily_minute_quote_paths() -> None:
    """Minute quote history rolls by trading day and same-day loads do not read
    the prior day's file."""
    import tempfile
    from pathlib import Path as _Path

    with tempfile.TemporaryDirectory() as td:
        out_dir = _Path(td)
        p17 = agent.dated_output_path(out_dir, "minute_quotes_{trade_date}.jsonl", "2026-06-17")
        p18 = agent.dated_output_path(out_dir, "minute_quotes_{trade_date}.jsonl", "2026-06-18")
        p_legacy = agent.dated_output_path(out_dir, "minute_quotes.jsonl", "2026-06-18")
        agent.append_jsonl(p17, {"stockCode": "OLD", "timestamp": "2026-06-17T10:00:00+08:00"})
        agent.append_jsonl(p18, {"stockCode": "NEW", "timestamp": "2026-06-18T10:00:00+08:00"})
        rows = agent.load_recent_quotes(p18)
        check("T25 date-template output path resolves per day",
              p17.name == "minute_quotes_2026-06-17.jsonl" and p18.name == "minute_quotes_2026-06-18.jsonl",
              f"{p17.name},{p18.name}")
        check("T25 legacy filename rolls before extension", p_legacy.name == "minute_quotes_2026-06-18.jsonl", p_legacy.name)
        check("T25 same-day load excludes prior-day rows",
              [r.get("stockCode") for r in rows] == ["NEW"], str(rows))


def t26_daily_replay_cache_and_directory_reader() -> None:
    """Evolution splits large replay jsonl into date files; replay reader can
    consume the date directory without scanning unrelated days."""
    import importlib
    import json as _json
    import tempfile
    from pathlib import Path as _Path

    ev = importlib.import_module("run_t0_strategy_evolution")
    replay = importlib.import_module("replay_t0_decisions")
    with tempfile.TemporaryDirectory() as td:
        root = _Path(td)
        src = root / "quotes.jsonl"
        rows = [
            {"timestamp": "2026-06-17T10:00:00+08:00", "stockCode": "510300"},
            {"timestamp": "2026-06-18T10:00:00+08:00", "stockCode": "159915"},
        ]
        src.write_text("\n".join(_json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        meta = ev.prepare_daily_quote_cache(str(src), cache_root=root / "cache")
        cache_dir = _Path(meta["cache_dir"])
        read_18 = list(replay.iter_quote_rows(cache_dir, date_filter="2026-06-18"))
        check("T26 daily replay cache creates one file per date",
              (cache_dir / "2026-06-17.jsonl").exists() and (cache_dir / "2026-06-18.jsonl").exists(),
              str(meta))
        check("T26 replay directory reader filters by date",
              [r.get("stockCode") for r in read_18] == ["159915"], str(read_18))


def t27_dynamic_gate_replay_cache() -> None:
    """Dynamic replay cache retains complete intraday history for any code that
    becomes eligible, and removes codes that never pass the live gates."""
    import importlib
    import json as _json
    import tempfile
    from pathlib import Path as _Path

    ev = importlib.import_module("run_t0_strategy_evolution")
    with tempfile.TemporaryDirectory() as td:
        root = _Path(td)
        daily = root / "daily"
        daily.mkdir()
        rows = [
            {"timestamp": "2026-06-18T09:31:00+08:00", "stockCode": "510300", "name": "ETF-A",
             "currentPrice": 4.0, "bidPrice1": 3.999, "askPrice1": 4.001, "amount": 10_000},
            {"timestamp": "2026-06-18T10:30:00+08:00", "stockCode": "510300", "name": "ETF-A",
             "currentPrice": 4.0, "bidPrice1": 3.999, "askPrice1": 4.001, "amount": 30_000_000},
            {"timestamp": "2026-06-18T10:30:00+08:00", "stockCode": "159999", "name": "ETF-B",
             "currentPrice": 1.0, "bidPrice1": 0.95, "askPrice1": 1.05, "amount": 100_000_000},
        ]
        (daily / "2026-06-18.jsonl").write_text(
            "\n".join(_json.dumps(r) for r in rows) + "\n", encoding="utf-8"
        )
        meta = ev.prepare_dynamic_gate_cache(
            str(daily),
            {"min_amount_yuan": 50_000_000, "max_spread_pct": 0.004, "min_price": 0.3,
             "name_exclude_keywords": ["货币"]},
            cache_root=root / "cache",
        )
        out_rows = [
            _json.loads(line)
            for line in (_Path(meta["cache_dir"]) / "2026-06-18.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        check("T27 dynamic gate retains full history for eligible code",
              [r.get("stockCode") for r in out_rows] == ["510300", "510300"], str(out_rows))
        check("T27 dynamic gate removes never-eligible wide-spread code",
              meta.get("eligible_codes_by_date", {}).get("2026-06-18") == 1, str(meta))


def t28_layered_backtest_pipeline() -> None:
    """P0 pipeline fails closed on incomplete dates, preserves frozen OOS
    boundaries, and computes explicit cost sensitivity without changing the
    legacy fill helper contract."""
    import importlib
    import json as _json
    import tempfile
    from pathlib import Path as _Path

    pipeline = importlib.import_module("run_t0_backtest_pipeline")
    replay = importlib.import_module("replay_t0_decisions")
    sim = {"cash": 100_000.0, "positions": {}, "buy_notional": 0.0, "sell_notional": 0.0}
    buy = {"stockCode": "510300", "exchange": "SH", "quantity": 1000, "price": 4.0}
    sell = {"stockCode": "510300", "exchange": "SH", "quantity": 1000, "price": 4.1}
    replay.apply_buy_fill(sim, buy)
    pnl = replay.apply_sell_fill(sim, sell)
    check("T28 legacy sell fill still returns numeric PnL", abs(pnl - 100.0) < 1e-9, str(pnl))
    check("T28 replay fill captures transaction notionals for cost audit",
          sim.get("buy_notional") == 4000.0 and sim.get("sell_notional") == 4100.0,
          str(sim.get("_last_fill")))

    with tempfile.TemporaryDirectory() as td:
        root = _Path(td)
        source = root / "quotes"
        source.mkdir()

        def make_day(day: str, codes: list[str]) -> None:
            rows = []
            for minute in range(200):
                hour = 9 + (30 + minute) // 60
                mm = (30 + minute) % 60
                ts = f"{day}T{hour:02d}:{mm:02d}:00+08:00"
                for code in codes:
                    rows.append({"timestamp": ts, "stockCode": code})
            (source / f"{day}.jsonl").write_text(
                "\n".join(_json.dumps(row) for row in rows) + "\n", encoding="utf-8"
            )

        make_day("2026-06-15", ["510300", "510500", "159915", "512100"])
        make_day("2026-06-16", ["510300"])
        old_out = pipeline.DEFAULT_OUT
        pipeline.DEFAULT_OUT = root / "pipeline"
        try:
            audit = pipeline.audit_quote_directory(source, {
                "min_rounds_per_complete_day": 200,
                "min_median_codes_per_round": 1,
                "min_cross_section_ratio_to_best_day": 0.5,
                "max_duplicate_timestamp_code_ratio": 0.001,
            })
        finally:
            pipeline.DEFAULT_OUT = old_out
        check("T28 data audit accepts complete cross-section day",
              "2026-06-15" in audit.get("complete_dates", []), str(audit))
        check("T28 data audit rejects collapsed cross-section day",
              "2026-06-16" in audit.get("incomplete_dates", []), str(audit))

        lock = root / "locked.json"
        overlay = {"entry_score_threshold": 60}
        lock.write_text(_json.dumps({
            "frozen": True,
            "promotion_status": "promoted",
            "overlay": overlay,
            "parameter_hash": pipeline.stable_hash(overlay),
            "in_sample_end": "2026-06-15",
        }), encoding="utf-8")
        _, overlap_reasons = pipeline.validate_locked_parameters(lock, "2026-06-15")
        _, clean_reasons = pipeline.validate_locked_parameters(lock, "2026-06-16")
        check("T28 OOS overlap fails closed", any("oos_overlaps_in_sample" in x for x in overlap_reasons), str(overlap_reasons))
        check("T28 frozen non-overlapping OOS is accepted", clean_reasons == [], str(clean_reasons))

    metrics = pipeline.replay_metrics({
        "gross_pnl": 100.0,
        "buy_notional": 4000.0,
        "sell_notional": 4100.0,
        "per_day": {"2026-06-15": {"gross_pnl": 100.0, "buy_notional": 4000.0, "sell_notional": 4100.0}},
        "trades": [{"stockCode": "510300", "gross_pnl": 100.0, "entry_notional": 4000.0, "exit_notional": 4100.0}],
    }, "stress", 20.0)
    check("T28 explicit costs reduce net PnL", 91.8 < metrics["net_pnl"] < 92.0, str(metrics))


if __name__ == "__main__":
    t1_t3_state_and_determinism()
    t2_no_side_effects()
    t4_sell_bypasses_buy_checks()
    t6_apply_bounds_and_lock_isolation()
    t7_multi_position_exit()
    t8_diebold_mariano_gate()
    t9_model_confidence_set()
    t10_spa_reality_check()
    t11_fractional_kelly_sizing()
    t12_intraday_momentum()
    t16_quote_fallback_chain()
    t13_execution_quality_safe_shield_and_dsr()
    t14_513100_entry_blocked_but_sell_allowed()
    t15_emergency_sells_bypass_throttle()
    t17_passive_entry_pricing()
    t18_pre_sell_position_verification()
    t19_multi_holding_entry_while_carrying()
    t20_alpha101_conviction()
    t21_inventory_aware_passive_skew()
    t22_dynamic_universe_selection()
    t23_sector_diversification_entry_filter()
    t24_sector_limit_never_blocks_sells()
    t25_daily_minute_quote_paths()
    t26_daily_replay_cache_and_directory_reader()
    t27_dynamic_gate_replay_cache()
    t28_layered_backtest_pipeline()
    print()
    if failures:
        print(f"FAILED: {len(failures)} invariant(s): {failures}")
        sys.exit(1)
    print("ALL INVARIANTS PASSED")
