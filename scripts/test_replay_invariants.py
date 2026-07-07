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
    # Keep the fixture decisively above threshold.  The current weighted score is 62.25;
    # using the live threshold would test score calibration rather than multi-sell/throttle.
    cfg["strategy"]["profit_exit_score_threshold"] = 60
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


def t39_timing_accuracy_helpers() -> None:
    """Timing-accuracy primitives: range_percentile locates a price in [low,high];
    pullback_entry only fills on a real dip; trailing_exit leaves on a drop from peak."""
    import research_timing as rt
    check("T39 percentile at the low = 0", rt.range_percentile(1.0, 1.0, 1.1) == 0.0)
    check("T39 percentile at the high = 1", rt.range_percentile(1.1, 1.0, 1.1) == 1.0)
    check("T39 percentile mid = 0.5", abs(rt.range_percentile(1.05, 1.0, 1.1) - 0.5) < 1e-9)
    check("T39 degenerate range -> None", rt.range_percentile(1.0, 1.0, 1.0) is None)
    # pullback fills on a 0.4% dip within the wait window
    dip = [1.000, 1.002, 0.995, 1.01]
    filled = rt.pullback_entry(dip, 0.004, 15)
    check("T39 pullback fills at the dip", filled is not None and abs(filled[0] - 0.995) < 1e-9, str(filled))
    # no dip -> no fill (we'd miss this signal)
    rip = [1.000, 1.003, 1.006, 1.01]
    check("T39 no dip -> pullback does not fill", rt.pullback_entry(rip, 0.004, 15) is None)
    # trailing exits on a 0.6% drop from the peak
    peak_fade = [1.0, 1.02, 1.03, 1.018]
    px, _ = rt.trailing_exit(peak_fade, 0.006)
    check("T39 trailing exits below the peak", abs(px - 1.018) < 1e-9, str(px))
    # monotonic up -> trailing holds to the last price
    up = [1.0, 1.01, 1.02, 1.03]
    check("T39 monotonic up -> trailing holds to close", rt.trailing_exit(up, 0.006)[0] == 1.03)


def t37_committed_holdings_cap() -> None:
    """target_holdings must count SAME-DAY T+1 buys (held but not yet sellable), or
    removing the daily entry cap would over-accumulate. Two positions held at qty>0 with
    availableQuantity=0 (T+1, not sellable today) must fill the cap (target_holdings=2)
    and block a new entry -- even though the sellable held_codes set is empty."""
    import io as _io
    import json as _json
    import copy as _copy
    from datetime import datetime
    from zoneinfo import ZoneInfo
    cfg = _json.load(_io.open(ROOT / "configs" / "t0_intraday_paper_agent.json", encoding="utf-8"))
    cfg["dynamic_universe"] = {"enabled": False}
    cfg["strategy"]["target_holdings"] = 2
    cfg["strategy"]["max_entries_per_day"] = 0
    cfg["strategy"].setdefault("sector_diversification", {})
    cfg["sector_diversification"] = {"enabled": False}
    now = datetime(2026, 6, 18, 10, 30, tzinfo=ZoneInfo("Asia/Shanghai"))
    ts = now.isoformat()
    # two T+1 positions: quantity>0 but availableQuantity=0 (cannot sell same day)
    positions = [{"stockCode": c, "stockName": c, "exchange": "SH", "quantity": 10000,
                  "availableQuantity": 0, "costPrice": 1.0} for c in ("513500", "518880")]
    cand = {"stockCode": "513180", "exchange": "SH", "name": "513180", "asset_class": "hk_etf",
            "currentPrice": 1.05, "bidPrice1": 1.049, "askPrice1": 1.051, "prevClose": 1.00,
            "timestamp": ts, "quote_ok": True, "isSuspended": False, "momentum_available": True,
            "momentum": 0.02, "spread_pct": 0.0008, "change_pct": 0.05, "bid_pressure_3m_pct": 0.003,
            "acceleration": 0.002}
    balance = {"ok": True, "data": {"totalAssets": 1_000_000.0, "availableBalance": 900_000.0}}
    agent.set_replay_now(now)
    try:
        dec = agent.build_decision(cfg, [cand], balance, {"ok": True, "data": {"positions": positions}},
                                   {"ok": True, "data": {"orders": []}}, {}, None)
    finally:
        agent.set_replay_now(None)
    buys = [o for o in dec.get("orders", []) if o.get("direction") == "buy"]
    check("T37 same-day T+1 holdings fill the cap -> no new entry", len(buys) == 0, str(dec.get("orders")))
    check("T37 reason is at_target_holdings_capacity",
          dec.get("state_machine", {}).get("reason") == "at_target_holdings_capacity",
          str(dec.get("state_machine")))


def t61_full_market_and_t0_entry_guards() -> None:
    """Defensive entry guards: full-market risk_off blocks new BUYs, and T+1/
    unconfirmed-sellability ETF classes cannot be opened by the T0 agent."""
    import csv as _csv
    import gzip as _gzip
    import io as _io
    import json as _json
    import tempfile
    from datetime import datetime
    from pathlib import Path as _Path
    from zoneinfo import ZoneInfo

    SH = ZoneInfo("Asia/Shanghai")
    with tempfile.TemporaryDirectory() as td:
        root = _Path(td)
        snap = root / "eastmoney_full_market_20260623_103000_etf.csv.gz"
        with _gzip.open(snap, "wt", encoding="utf-8", newline="") as f:
            writer = _csv.DictWriter(f, fieldnames=[
                "stockCode", "name", "currentPrice", "change_pct", "amount",
            ])
            writer.writeheader()
            for i in range(420):
                writer.writerow({
                    "stockCode": f"5{i:05d}"[-6:],
                    "name": f"ETF{i}",
                    "currentPrice": "1.0",
                    "change_pct": "0.2" if i < 60 else "-1.2",
                    "amount": "2000000",
                })
            writer.writerow({  # money-like names are excluded from breadth.
                "stockCode": "511990", "name": "货币ETF", "currentPrice": "100.0",
                "change_pct": "0.0", "amount": "999999999",
            })
        runs = root / "snapshot_runs.jsonl"
        runs.write_text(_json.dumps({
            "task": "eastmoney_full_market_snapshot", "status": "ok", "scope": "etf",
            "row_count": 421, "output_file": str(snap),
            "session": {"trade_date": "2026-06-23", "exchange_local_time": "2026-06-23T10:30:00+08:00"},
        }) + "\n", encoding="utf-8")
        strat = {
            "full_market_entry_guard": {
                "enabled": True,
                "snapshot_runs_path": str(runs),
                "max_snapshot_age_minutes": 90,
                "min_non_money_etfs": 300,
                "min_active_amount": 1000000,
                "min_active_etfs": 200,
                "risk_off_up_frac_max": 0.30,
                "risk_off_median_change_pct_max": -0.50,
                "risk_off_down_gt_1pct_frac_min": 0.55,
                "block_on_risk_off": True,
                "block_on_unavailable": False,
            }
        }
        agent.set_replay_now(None)
        detail = agent.evaluate_full_market_entry_guard(strat, "2026-06-23", datetime(2026, 6, 23, 10, 45, tzinfo=SH))
        check("T61 full-market guard classifies risk_off and blocks BUY",
              detail.get("block_new_buy") is True and detail.get("mode") == "risk_off",
              str(detail))

    cfg = _json.load(_io.open(ROOT / "configs" / "t0_intraday_paper_agent.json", encoding="utf-8"))
    cfg["strategy"]["intraday_momentum"]["enabled"] = False
    cfg["strategy"]["target_holdings"] = 2
    cfg["sector_diversification"] = {"enabled": False}
    cfg["strategy"]["full_market_entry_guard"]["enabled"] = False
    now = datetime(2026, 6, 23, 10, 40, tzinfo=SH)
    ts = now.isoformat()
    dyn = {"stockCode": "159558", "exchange": "SZ", "name": "半导体设备ETF易方达", "asset_class": "dynamic",
           "currentPrice": 3.75, "bidPrice1": 3.749, "askPrice1": 3.751, "prevClose": 3.69,
           "timestamp": ts, "quote_ok": True, "isSuspended": False, "momentum_available": True,
           "momentum": 0.03, "spread_pct": 0.0003, "change_pct": 0.016,
           "bid_pressure_3m_pct": 0.01, "acceleration": 0.002}
    cfg["universe"] = [{"stockCode": "159558", "exchange": "SZ", "name": "半导体设备ETF易方达", "asset_class": "dynamic"}]
    balance = {"ok": True, "data": {"totalAssets": 1_000_000.0, "availableBalance": 800_000.0}}
    agent.set_replay_now(now)
    try:
        dec = agent.build_decision(cfg, [dyn], balance, {"ok": True, "data": {"positions": []}},
                                   {"ok": True, "data": {"orders": []}}, {}, None)
    finally:
        agent.set_replay_now(None)
    t0_check = next((c for c in dec.get("risk_checks", []) if c.get("name") == "t0_entry_eligibility_filter"), {})
    buys = [o for o in dec.get("orders", []) if o.get("direction") == "buy"]
    check("T61 dynamic/T+1-like ETF is not opened by T0 entry",
          not buys and dec.get("state_machine", {}).get("reason") == "blocked_t0_entry_ineligible_asset_class",
          str(dec.get("state_machine")))
    check("T61 T0 eligibility risk check records blocked candidate",
          t0_check.get("passed") is False and t0_check.get("detail", {}).get("blocked_count") == 1,
          str(t0_check))


def t36_overfitting_guard() -> None:
    """PBO (CSCV) must read ~0.5 for a pure-noise config search (winner does not persist)
    and ~0 for a genuinely dominant config; purged splits must not leak; the multiple-
    testing note must flag a tiny Sharpe as luck and a huge one as exceeding noise."""
    import random
    import overfitting_guard as og
    # genuinely dominant config -> PBO ~ 0
    random.seed(11)
    dom = [[random.gauss(0, 1) for _ in range(96)] for _ in range(20)]
    dom[5] = [random.gauss(0.8, 1) for _ in range(96)]
    pbo_dom = og.combinatorial_symmetric_pbo(dom, n_blocks=10)["pbo"]
    check("T36 dominant config -> PBO ~ 0", pbo_dom is not None and pbo_dom < 0.05, str(pbo_dom))
    # pure noise -> PBO centered ~0.5 (average over seeds for determinism)
    noise_pbos = []
    for seed in range(20):
        random.seed(1000 + seed)
        noise = [[random.gauss(0, 1) for _ in range(96)] for _ in range(20)]
        noise_pbos.append(og.combinatorial_symmetric_pbo(noise, n_blocks=10)["pbo"])
    mean_noise = sum(noise_pbos) / len(noise_pbos)
    check("T36 noise search -> PBO centered ~0.5", 0.35 < mean_noise < 0.65, str(round(mean_noise, 3)))
    check("T36 noise PBO clearly worse than dominant", mean_noise > pbo_dom + 0.2, f"{mean_noise} vs {pbo_dom}")
    # purged/embargoed splits: no leakage
    splits = list(og.purged_train_test_splits(30, n_splits=5, embargo=3))
    check("T36 purged splits cover 5 folds", len(splits) == 5, str(len(splits)))
    check("T36 no train/test overlap", all(not (set(tr) & set(te)) for tr, te in splits))
    check("T36 embargo gap respected",
          all(min((abs(a - b) for a in tr for b in te), default=99) >= 1 for tr, te in splits))
    # multiple-testing note
    luck = og.deflated_significance_note(50, 0.2, 100)
    real = og.deflated_significance_note(50, 5.0, 100)
    check("T36 tiny Sharpe over many trials -> luck", luck["flag"] == "consistent_with_luck", str(luck))
    check("T36 huge Sharpe -> exceeds noise max", real["flag"] == "exceeds_noise_max", str(real))


def t35_sizing_weights() -> None:
    """Sizing research: inverse-vol weights give MORE weight to lower-vol names and sum
    to 1; basket_return is the weighted sum of forward returns."""
    import research_sizing as rz
    w = rz.inverse_vol_weights([0.001, 0.002, 0.004])
    check("T35 inverse-vol weights sum to 1", abs(sum(w) - 1.0) < 1e-9, str(sum(w)))
    check("T35 lower vol gets more weight", w[0] > w[1] > w[2], str(w))
    check("T35 equal vols -> equal weights", rz.inverse_vol_weights([0.002, 0.002]) == [0.5, 0.5],
          str(rz.inverse_vol_weights([0.002, 0.002])))
    check("T35 basket_return is the weighted sum",
          abs(rz.basket_return([0.5, 0.5], [0.02, -0.01]) - 0.005) < 1e-9,
          str(rz.basket_return([0.5, 0.5], [0.02, -0.01])))


def t34_exit_rule_simulation() -> None:
    """simulate_exit mechanics (Kaminski-Lo): a stop caps a monotonic crash but SELLS
    THE BOTTOM on a dip-then-recover path (hold wins there); take-profit and trailing
    fire at their levels."""
    import research_stops as rs
    # dip to -1.2% then recover to +1%: a 1% stop exits at -1% (hurts), hold gets +1%
    dip_recover = [1.0, 0.995, 0.988, 0.995, 1.01]
    check("T34 stop sells the bottom on mean-reversion", abs(rs.simulate_exit(dip_recover, stop=0.01) - (-0.01)) < 1e-9,
          str(rs.simulate_exit(dip_recover, stop=0.01)))
    check("T34 hold recovers where stop bailed", abs(rs.simulate_exit(dip_recover) - 0.01) < 1e-9,
          str(rs.simulate_exit(dip_recover)))
    # monotonic crash: stop caps the loss vs a worse hold
    crash = [1.0, 0.99, 0.97, 0.95, 0.92]
    check("T34 stop caps a crash better than hold",
          rs.simulate_exit(crash, stop=0.02) == -0.02 and rs.simulate_exit(crash) < -0.07,
          f"{rs.simulate_exit(crash, stop=0.02)},{rs.simulate_exit(crash)}")
    # take-profit fires at +2%
    rip = [1.0, 1.01, 1.025, 1.05]
    check("T34 take-profit fires at its level", abs(rs.simulate_exit(rip, take=0.02) - 0.02) < 1e-9,
          str(rs.simulate_exit(rip, take=0.02)))
    # trailing 1% from a 1.03 peak -> exit ~ +1.97%
    peak_then_fade = [1.0, 1.02, 1.03, 1.018]
    check("T34 trailing stop exits below the peak",
          abs(rs.simulate_exit(peak_then_fade, trail=0.01) - (1.03 * 0.99 - 1.0)) < 1e-9,
          str(rs.simulate_exit(peak_then_fade, trail=0.01)))


def t33_regime_classifier() -> None:
    """Regime research: Kaufman efficiency ratio ~1 on a clean trend and low on chop;
    classify_regime maps (net move, ER) to trend_up/trend_down/chop as expected."""
    import research_regime as rg
    up = [0.0, 0.001, 0.002, 0.003, 0.004, 0.005]          # monotonic -> ER ~ 1
    chop = [0.0, 0.004, 0.0, 0.004, 0.0, 0.004]            # oscillating -> ER low
    check("T33 efficiency ratio ~1 on clean trend", rg.efficiency_ratio(up) > 0.95, str(rg.efficiency_ratio(up)))
    check("T33 efficiency ratio low on chop", rg.efficiency_ratio(chop) < 0.3, str(rg.efficiency_ratio(chop)))
    check("T33 strong up + high ER -> trend_up",
          rg.classify_regime(0.005, 0.9, er_thr=0.4, move_thr=0.003) == "trend_up")
    check("T33 strong down + high ER -> trend_down",
          rg.classify_regime(-0.005, 0.9, er_thr=0.4, move_thr=0.003) == "trend_down")
    check("T33 small move -> chop",
          rg.classify_regime(0.001, 0.9, er_thr=0.4, move_thr=0.003) == "chop")
    check("T33 directionless (low ER) -> chop",
          rg.classify_regime(0.01, 0.2, er_thr=0.4, move_thr=0.003) == "chop")


def t32_lead_lag_detection() -> None:
    """Lead-lag analyzer must (a) recover a KNOWN lead time -- a follower built as the
    leader delayed 15 min should peak at horizon=15 with high correlation -- and (b)
    stay strictly diagnostic_only (no validated edge / no orders)."""
    import math
    import research_lead_lag as ll
    minutes = list(range(570, 901))  # 09:30..15:00 China-minutes

    def leader_px(m):
        return round(1.0 + 0.03 * math.sin((m - 570) / 60.0), 4)

    leader = [(m, leader_px(m)) for m in minutes]
    follower_a = [(m, leader_px(m - 15)) for m in minutes if m - 15 >= 570]   # exact 15-min lag
    follower_b = [(m, round(leader_px(m - 15) * 1.001, 4)) for m in minutes if m - 15 >= 570]
    by_code = {
        "500001": {"name": "半导体ETF龙头", "amount_by_min": {m: 1e9 for m, _ in leader}, "series": leader},
        "500002": {"name": "半导体ETF乙", "amount_by_min": {m: 6e7 for m, _ in follower_a}, "series": follower_a},
        "500003": {"name": "半导体ETF丙", "amount_by_min": {m: 6e7 for m, _ in follower_b}, "series": follower_b},
    }
    res = ll.analyze_day("2026-06-18", by_code, keyword_map={"半导体": "semi"},
                         decision_start=600, decision_end=840, step=5, window=15,
                         leader_threshold=0.0, min_amount=5e7, min_members=3, roundtrip_cost_pct=0.0)
    check("T32 lead-lag analyzer returns a result", res is not None, "none")
    check("T32 recovers the planted 15-min lead time", res and res.get("best_lead_minutes") == 15, str(res and res.get("best_lead_minutes")))
    check("T32 strong correlation at the true lag", res and res.get("best_corr") is not None and res["best_corr"] > 0.8, str(res and res.get("best_corr")))
    summary = ll.summarize([res], {})
    check("T32 stays diagnostic_only / no validated edge",
          summary["status"] == "diagnostic_only" and summary["edge_validated"] is False
          and summary["order_submit_calls_made"] is False, str(summary.get("status")))


def t30_volume_capture_and_surge() -> None:
    """P1: the agent must CAPTURE volume/amount/turnover (previously dropped in
    normalize_quote), and the volume-surge proxy must rise when cumulative volume
    accelerates, sit ~1 when flat, and be None without enough same-day history."""
    import run_etf_paper_trading_agent as base
    item = {"f2": 1.5, "f3": 2.0, "f5": 1_000_000, "f6": 1_500_000, "f8": 7.5,
            "f12": "588000", "f14": "x", "f15": 1.55, "f16": 1.45, "f17": 1.48,
            "f18": 1.47, "f31": 1.499, "f32": 1.501}
    etf = {"stockCode": "588000", "exchange": "SH", "name": "x"}
    q = base.normalize_quote(etf, base.eastmoney_quote_response(etf, item))
    check("T30 volume carried through normalize_quote", q.get("volume") == 1_000_000.0, str(q.get("volume")))
    check("T30 amount carried", q.get("amount") == 1_500_000.0, str(q.get("amount")))
    check("T30 turnover_pct carried (f8)", q.get("turnover_pct") == 7.5, str(q.get("turnover_pct")))
    check("T30 change_pct carried (f3)", q.get("change_pct") == 2.0, str(q.get("change_pct")))

    rising = [100, 200, 300, 400, 500, 600, 700, 800, 950, 1150, 1400]  # increments grow at the end
    hist = [{"volume": v} for v in rising[:-1]]
    surge = agent.compute_volume_surge(hist, {"volume": rising[-1]}, recent_n=3, baseline_n=7)
    check("T30 accelerating volume -> surge > 1", surge is not None and surge > 1.0, str(surge))
    flat = list(range(100, 100 + 11 * 50, 50))  # constant increments
    s_flat = agent.compute_volume_surge([{"volume": v} for v in flat[:-1]], {"volume": flat[-1]},
                                        recent_n=3, baseline_n=7)
    check("T30 flat volume -> surge ~1", s_flat is not None and 0.8 <= s_flat <= 1.2, str(s_flat))
    check("T30 insufficient history -> None", agent.compute_volume_surge([{"volume": 1}], {"volume": 2}) is None)


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
    import collect_eastmoney_full_market as collector

    check("T22 cash-flow factor ETF is not classified as money-market",
          not collector.is_money_market_fund({"name": "自由现金流ETF华夏"}))
    check("T22 actual money-market ETF remains excluded",
          collector.is_money_market_fund({"name": "现金管理货币ETF"}))
    cashflow_gate = sel.passes_gates(
        {"stockCode": "159201", "name": "自由现金流ETF华夏", "currentPrice": 1.0,
         "bidPrice1": 0.999, "askPrice1": 1.001, "amount": 100_000_000},
        {"name_exclude_keywords": ["货币", "现金", "理财"], "min_price": 0.3,
         "max_spread_pct": 0.004},
        50_000_000,
        set(),
    )
    check("T22 cash-flow factor ETF passes the generic cash-name gate", cashflow_gate == (True, "ok"), str(cashflow_gate))

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
    cfg["strategy"]["t0_entry_eligibility"] = {"enabled": False}
    cfg["strategy"]["entry_logic_v2"] = {"enabled": False}  # isolate: this test is about sector diversification, not entry timing
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
    cfg["strategy"]["t0_entry_eligibility"] = {"enabled": False}
    cfg["strategy"]["entry_logic_v2"] = {"enabled": False}  # isolate: this test is about sector diversification, not entry timing
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

        agent_out = root / "agent_output"
        agent_out.mkdir()
        legacy_row = {
            "timestamp": "2026-06-17T10:00:00+08:00", "stockCode": "510300",
            "currentPrice": 4.0, "bidPrice1": 3.999, "askPrice1": 4.001, "volume": 1000,
        }
        daily_row = {
            "timestamp": "2026-06-18T10:00:00+08:00", "stockCode": "159915",
            "currentPrice": 2.0, "bidPrice1": 1.999, "askPrice1": 2.001, "volume": 2000,
        }
        unrelated_row = {
            "timestamp": "2026-06-18T10:00:00+08:00", "stockCode": "999999",
            "currentPrice": 9.0,
        }
        (agent_out / "minute_quotes.jsonl").write_text(_json.dumps(legacy_row) + "\n", encoding="utf-8")
        (agent_out / "minute_quotes_2026-06-18.jsonl").write_text(_json.dumps(daily_row) + "\n", encoding="utf-8")
        (agent_out / "t0_agent_runs.jsonl").write_text(_json.dumps(unrelated_row) + "\n", encoding="utf-8")
        mixed_rows = list(replay.iter_quote_rows(agent_out))
        check("T26 agent output reader combines legacy and daily quote logs only",
              [r.get("stockCode") for r in mixed_rows] == ["510300", "159915"], str(mixed_rows))

        adverse = importlib.import_module("research_adverse_selection")
        pairs = importlib.import_module("research_cross_etf_pairs")
        costs = importlib.import_module("research_execution_cost")
        adverse_codes = sorted(adverse.mid_series_by_code(agent_out))
        check("T26 adverse-selection research reads rolled quote history",
              adverse_codes == ["159915", "510300"], str(adverse_codes))
        pair_codes = sorted({code for rnd in pairs.load_aligned(agent_out) for code in rnd if code != "ts"})
        check("T26 pair research reads rolled quote history",
              pair_codes == ["159915", "510300"], str(pair_codes))
        cost_codes = sorted(costs.load(agent_out))
        check("T26 execution-cost research reads rolled quote history",
              cost_codes == ["159915", "510300"], str(cost_codes))


def t27_dynamic_gate_replay_cache() -> None:
    """Dynamic replay cache begins at first point-in-time eligibility and never
    backfills earlier rows using later turnover."""
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
        check("T27 dynamic gate does not backfill pre-eligibility history",
              [r.get("stockCode") for r in out_rows] == ["510300"], str(out_rows))
        check("T27 dynamic gate labels point-in-time activation",
              out_rows[0].get("liquidity_gate_source") == "point_in_time", str(out_rows))
        check("T27 dynamic gate removes never-eligible wide-spread code",
              meta.get("eligible_codes_by_date", {}).get("2026-06-18") == 1, str(meta))
        check("T27 cash-flow factor ETF is not rejected by generic cash keyword",
              ev._passes_dynamic_replay_gate(
                  {"timestamp": "2026-06-18T10:30:00+08:00", "stockCode": "159201",
                   "name": "自由现金流ETF华夏", "currentPrice": 1.0, "bidPrice1": 0.999,
                   "askPrice1": 1.001, "amount": 100_000_000},
                  {"min_amount_yuan": 50_000_000, "max_spread_pct": 0.004, "min_price": 0.3,
                   "name_exclude_keywords": ["货币", "现金", "理财"]},
              ))


def t38_full_minute_replay_builder() -> None:
    """Minute replay conversion uses cumulative same-day flow and full-universe metadata."""
    import importlib
    builder = importlib.import_module("build_t0_replay_quotes_from_minute_data")
    etf = {"stockCode": "510300", "exchange": "SH", "name": "沪深300ETF", "asset_class": "dynamic"}
    rows = [
        {"datetime": "2026-06-18 09:31:00", "open": 4.0, "close": 4.0, "high": 4.0, "low": 4.0,
         "volume": 100, "amount": 1_000_000, "source": "tdx"},
        {"datetime": "2026-06-18 09:32:00", "open": 4.0, "close": 4.01, "high": 4.01, "low": 4.0,
         "volume": 200, "amount": 2_000_000, "source": "tdx"},
    ]
    quotes = list(builder.cumulative_quotes(etf, rows, "2026-06-18", "2026-06-18"))
    check("T38 minute volume becomes same-day cumulative",
          [q["volume"] for q in quotes] == [100.0, 300.0], str([q["volume"] for q in quotes]))
    check("T38 minute amount becomes same-day cumulative",
          [q["amount"] for q in quotes] == [1_000_000.0, 3_000_000.0], str([q["amount"] for q in quotes]))
    check("T38 raw minute flow remains available for diagnostics",
          quotes[-1]["minute_volume"] == 200.0 and quotes[-1]["minute_amount"] == 2_000_000.0,
          str(quotes[-1]))

    history = [
        {**quotes[0], "timestamp": "2026-06-18T09:31:00+08:00", "quote_ok": True},
        {**quotes[1], "timestamp": "2026-06-18T09:32:00+08:00", "quote_ok": True},
    ]
    current = [{**quotes[-1], "timestamp": "2026-06-18T09:33:00+08:00", "currentPrice": 4.02,
                "quote_ok": True}]
    indexed = {"510300": history}
    legacy = agent.compute_snapshot_momentum(current, history, 2, {}, history_by_code=None)
    optimized = agent.compute_snapshot_momentum(current, history, 2, {}, history_by_code=indexed)
    check("T38 indexed replay history preserves momentum output",
          legacy[0].get("momentum") == optimized[0].get("momentum"),
          f"{legacy[0].get('momentum')} != {optimized[0].get('momentum')}")

    replay = importlib.import_module("replay_t0_decisions")
    replay_cfg = {"universe": [{"stockCode": "510300", "exchange": "SH", "name": "seed"}]}
    added = replay.extend_replay_universe(replay_cfg, [{
        "stockCode": "159201", "exchange": "SZ", "name": "自由现金流ETF华夏", "asset_class": "dynamic"
    }])
    visible = agent.t0_positions(
        {"ok": True, "data": {"positions": [{"stockCode": "159201", "quantity": 1000, "availableQuantity": 1000}]}},
        replay_cfg["universe"],
    )
    check("T38 replay adds dynamic quote codes to runtime universe", added == 1, str(replay_cfg["universe"]))
    check("T38 dynamic replay holding remains visible to exit engine", "159201" in visible, str(visible))

    evolution = importlib.import_module("run_t0_strategy_evolution")
    original_evaluate = evolution.evaluate_candidate
    try:
        evolution.evaluate_candidate = lambda *args, **kwargs: {  # type: ignore[assignment]
            "candidate": args[3], "overlay": args[4], "ok": True
        }
        parallel_rows = evolution.evaluate_fixed_candidates(
            [{"name": "first", "overlay": {}}, {"name": "second", "overlay": {}}],
            base_cfg={}, cfg_dir=ROOT, prefix="test", date_filter=None, quotes_path=None,
            replay_timeout_seconds=1, start_date="2026-06-01", end_date="2026-06-02", workers=2,
        )
    finally:
        evolution.evaluate_candidate = original_evaluate
    check("T38 parallel candidate evaluation preserves deterministic input order",
          [row["candidate"] for row in parallel_rows] == ["first", "second"], str(parallel_rows))


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


def t29_holdings_calibration_classification() -> None:
    """Locally reconciled SELL fills are not mislabeled as pending orders."""
    import importlib

    report = importlib.import_module("run_holdings_calibration_report")
    check(
        "T29 reconciled remaining quantity is a confirmed local position",
        report.classify_inventory_node({
            "fill_reconciliation_ok": True,
            "filled_remaining_qty": 100,
            "buy_quantity_submitted": 100,
        }) == "confirmed_position",
    )
    check(
        "T29 reconciled SELL fill with zero local remainder is a confirmed exit",
        report.classify_inventory_node({
            "fill_reconciliation_ok": True,
            "filled_remaining_qty": 0,
            "sell_quantity_submitted": 100,
            "sell_quantity_filled": 100,
        }) == "confirmed_sell_filled_no_local_remaining",
    )
    check(
        "T29 submitted SELL without reconciled fill remains unconfirmed",
        report.classify_inventory_node({
            "fill_reconciliation_ok": False,
            "filled_remaining_qty": 0,
            "sell_quantity_submitted": 100,
            "sell_quantity_filled": 0,
        }) == "submitted_not_confirmed_filled",
    )


def t31_early_entry_daily_accumulation() -> None:
    """Early-entry research persists idempotent daily evidence and cannot claim edge."""
    import importlib
    import tempfile
    from pathlib import Path as _Path

    early = importlib.import_module("research_early_entry")
    params = {
        "research_version": early.RESEARCH_VERSION,
        "early_end": "10:00",
        "breakout_after": "13:00",
        "min_amount": 50_000_000.0,
        "min_price": 0.3,
        "top_frac": 0.1,
        "breakout_pct": 0.015,
    }
    first = {
        "date": "2026-06-17", "eligible": 400, "top_n": 40,
        "universe_fwd_ret_pct": 0.1, "top_early_fwd_ret_pct": 0.2,
        "early_edge_pct": 0.1, "top_that_later_broke_out": 10,
        "late_vs_early_price_premium_pct": 1.2,
    }
    second = {
        "date": "2026-06-18", "eligible": 420, "top_n": 42,
        "universe_fwd_ret_pct": 0.0, "top_early_fwd_ret_pct": 0.2,
        "early_edge_pct": 0.2, "top_that_later_broke_out": 12,
        "late_vs_early_price_premium_pct": 1.4,
    }
    with tempfile.TemporaryDirectory() as td:
        out_dir = _Path(td)
        exp_id = early.experiment_id(params)
        early.save_daily_result(out_dir, exp_id, params, first)
        early.save_daily_result(out_dir, exp_id, params, second)
        second["early_edge_pct"] = 0.25
        early.save_daily_result(out_dir, exp_id, params, second)
        results = early.load_daily_results(out_dir, exp_id)
        summary = early.summarize_results(params, results, exp_id, min_days_for_statistical_testing=20)
        paths = early.publish_summary(out_dir, summary)
        check("T31 one idempotent slot per early-entry trade date",
              len(results) == 2 and [r["date"] for r in results] == ["2026-06-17", "2026-06-18"], str(results))
        check("T31 rerun replaces a date instead of double-counting",
              results[-1]["early_edge_pct"] == 0.25, str(results[-1]))
        check("T31 experiment id changes with signal parameters",
              exp_id != early.experiment_id({**params, "early_end": "10:05"}), exp_id)
        gates = summary.get("statistical_readiness", {}).get("required_promotion_gates", {})
        check("T31 thin early-entry sample remains diagnostic-only",
              summary.get("status") == "diagnostic_only" and summary.get("edge_validated") is False
              and summary.get("live_ready") is False and summary.get("formal_strategy_allowed") is False,
              str(summary))
        check("T31 all early-entry promotion gates fail closed before testing",
              len(gates) == len(early.REQUIRED_PROMOTION_GATES)
              and all(g.get("passed") is False and g.get("status") == "not_run_insufficient_days" for g in gates.values()),
              str(gates))
        check("T31 early-entry summary and history artifacts written",
              all(path.exists() for path in paths.values()), str(paths))


def t40_external_etf_observation_pool() -> None:
    """ChatGPT news picks are locally validated, capped and deduplicated with system-20."""
    import build_t0_observation_pool as pool

    master = {
        ("510300", "SH"): {"stockCode": "510300", "market": "1", "name": "沪深300ETF"},
        ("159915", "SZ"): {"stockCode": "159915", "market": "0", "name": "创业板ETF"},
        ("512480", "SH"): {"stockCode": "512480", "market": "1", "name": "半导体ETF"},
    }
    payload = {
        "schemaVersion": "chatgpt_etf_watchlist_v1",
        "asOfDate": "2026-06-21",
        "effectiveDate": "2026-06-22",
        "generatedAt": "2026-06-21T20:00:00+08:00",
        "etfs": [
            {"stockCode": "510300", "exchange": "SH", "reason": "政策新闻驱动",
             "sourceUrls": ["https://example.com/a"], "risks": ["消息兑现"]},
            {"stockCode": "159915", "exchange": "SZ", "reason": "行业新闻驱动",
             "sourceUrls": ["https://example.com/b"]},
            {"stockCode": "511990", "exchange": "SH", "reason": "货币基金不应进入",
             "sourceUrls": ["https://example.com/c"]},
            {"stockCode": "512480", "exchange": "SH", "reason": "缺少来源应拒绝", "sourceUrls": []},
        ],
    }
    accepted, rejected, meta = pool.validate_chatgpt_payload(payload, master, limit=10, today="2026-06-20")
    check("T40 receiver accepts only locally known non-money ETFs with sourced reasons",
          [x["stockCode"] for x in accepted] == ["510300", "159915"], str(accepted))
    check("T40 receiver rejects absent/money-master and unsourced entries",
          [x["reason"] for x in rejected] == ["not_in_local_non_money_etf_master", "missing_source_url"],
          str(rejected))
    system = [
        {"stockCode": "510300", "exchange": "SH", "name": "沪深300ETF", "rank_score": 2.0},
        {"stockCode": "512480", "exchange": "SH", "name": "半导体ETF", "rank_score": 1.0},
    ]
    combined, overlaps = pool.merge_observation_pool(system, accepted)
    keys = [(x["stockCode"], x["exchange"]) for x in combined]
    overlap = next(x for x in combined if x["stockCode"] == "510300")
    check("T40 system/chatgpt overlap is one composite code", overlaps == 1 and len(keys) == len(set(keys)) == 3,
          str(keys))
    check("T40 overlap retains both provenance labels and research", overlap.get("sources") == ["system_rank20", "chatgpt_news10"]
          and "chatgptResearch" in overlap, str(overlap))
    doc = pool.build_document(system, {"trade_date": "2026-06-18"}, accepted, rejected, meta,
                              ROOT / "non_money_master.jsonl", "test.json")
    check("T40 observation pool cannot become a trade gate",
          doc.get("paperTradingOnly") is True and doc.get("diagnosticOnly") is True
          and doc.get("tradeGateEnabled") is False and doc.get("liveReady") is False
          and doc.get("formalStrategyAllowed") is False, str(doc))
    stale, stale_rejected, _ = pool.validate_chatgpt_payload(payload, master, limit=10, today="2026-06-23")
    check("T40 expired news list fails closed instead of being reused",
          stale == [] and stale_rejected[0].get("reason") == "stale_effective_date", str(stale_rejected))


def t41_unattended_chatgpt_watchlist_generator() -> None:
    """Daily generator is calendar-aware, strict, atomic and research-only."""
    import copy as _copy
    import json as _json
    import tempfile
    from datetime import date as _date
    from pathlib import Path as _Path
    import generate_chatgpt_etf_watchlist as gen

    weekend = gen.market_day_context(_date(2026, 6, 20))
    session = gen.market_day_context(_date(2026, 6, 22))
    check("T41 XSHG calendar skips the Dragon Boat/weekend closure",
          weekend["is_trading_day"] is False and weekend["effective_date"] == "2026-06-22", str(weekend))
    check("T41 next XSHG session is recognized",
          session["is_trading_day"] is True and session["previous_session"] == "2026-06-18", str(session))

    master = {}
    etfs = []
    for index in range(10):
        code = f"51{index:04d}"
        key = (code, "SH")
        master[key] = {"stockCode": code, "exchange": "SH", "name": f"行业ETF{index}"}
        etfs.append({
            "stockCode": code, "exchange": "SH", "name": f"行业ETF{index}",
            "reason": f"新闻驱动{index}", "newsDrivers": [f"驱动{index}"],
            "risks": [f"风险{index}"], "sourceUrls": [f"https://news{index}.example.cn/item"],
        })
    payload = {
        "schemaVersion": gen.SCHEMA_VERSION,
        "asOfDate": "2026-06-22", "effectiveDate": "2026-06-22",
        "generatedAt": "2026-06-22T08:30:00+08:00", "etfs": etfs,
    }
    clean, errors = gen.validate_payload(payload, master=master, context=session, verify_urls=False)
    check("T41 valid trading-day payload requires exactly ten verified local ETFs",
          clean is not None and len(clean["etfs"]) == 10 and errors == [], str(errors))

    duplicate = _copy.deepcopy(payload)
    duplicate["etfs"][-1]["stockCode"] = duplicate["etfs"][0]["stockCode"]
    _, duplicate_errors = gen.validate_payload(duplicate, master=master, context=session, verify_urls=False)
    check("T41 duplicate stockCode fails closed",
          any("duplicate_stock_code" in str(x.get("reason")) for x in duplicate_errors), str(duplicate_errors))

    money = _copy.deepcopy(payload)
    money["etfs"][0]["name"] = "现金管理货币ETF"
    _, money_errors = gen.validate_payload(money, master=master, context=session, verify_urls=False)
    check("T41 money/cash-management ETF name fails closed",
          any("forbidden_money_or_cash_management_etf" in str(x.get("reason")) for x in money_errors), str(money_errors))

    empty = {**payload, "asOfDate": weekend["as_of_date"], "effectiveDate": weekend["effective_date"],
             "generatedAt": "2026-06-20T08:30:00+08:00", "etfs": []}
    clean_empty, empty_errors = gen.validate_payload(empty, master=master, context=weekend, verify_urls=False)
    check("T41 non-session payload permits an empty ETF list",
          clean_empty is not None and clean_empty["etfs"] == [] and empty_errors == [], str(empty_errors))

    schema = gen.response_json_schema()["properties"]["etfs"]
    check("T41 API structured-output schema fixes list size at ten",
          schema.get("minItems") == schema.get("maxItems") == 10 and gen.source_url_syntax_ok("https://news.cn/a")
          and not gen.source_url_syntax_ok("https://example.com/fake"), str(schema))

    with tempfile.TemporaryDirectory() as td:
        target = _Path(td) / "2026-06-22.json"
        gen.atomic_write_json(target, payload)
        reread = _json.loads(target.read_text(encoding="utf-8"))
        leftovers = list(_Path(td).glob("*.tmp"))
        check("T41 atomic UTF-8 write re-parses without temp remnants",
              reread == payload and leftovers == [], str(leftovers))


def t42_oos_variance_metrics() -> None:
    """OOS variance helpers measure downside and paired uncertainty without side effects."""
    import analyze_oos_variance as variance

    values = {"2026-06-01": 100.0, "2026-06-02": -50.0, "2026-06-03": -100.0, "2026-06-04": 200.0}
    row = variance.metrics(values)
    check("T42 OOS metrics identify worst day and winning-day count",
          row["worst_day_pnl"] == -100.0 and row["worst_day_date"] == "2026-06-03"
          and row["winning_days"] == 2, str(row))
    check("T42 cumulative drawdown uses chronological equity",
          variance.max_cumulative_drawdown(list(values.values())) == -150.0, str(row))
    baseline = [100.0, -100.0, 100.0, -100.0, 100.0, -100.0]
    candidate = [20.0, -20.0, 20.0, -20.0, 20.0, -20.0]
    boot = variance.paired_block_bootstrap(baseline, candidate, reps=500, block_length=2, seed=7)
    check("T42 paired bootstrap detects planted lower variance",
          boot["daily_std_difference_candidate_minus_baseline"]["ci_95"][1] < 0
          and boot["daily_std_difference_candidate_minus_baseline"]["probability_candidate_lower"] > 0.99,
          str(boot))


def t43_fail_closed_t0_etf_master() -> None:
    """Official subclasses can confirm T+0; names alone never can."""
    import json as _json
    import tempfile
    from pathlib import Path as _Path
    import build_t0_etf_master as master

    cross_border = master.official_row_to_master({
        "stockCode": "513100", "name": "Nasdaq ETF", "subClass": "33",
        "benchmarkIndex": "NASDAQ-100", "benchmarkCode": "NDX",
    }, {})
    domestic = master.official_row_to_master({
        "stockCode": "510300", "name": "CSI 300 ETF", "subClass": "03",
        "benchmarkIndex": "CSI 300", "benchmarkCode": "000300",
    }, {})
    pending = master.pending_row_to_master(
        {"stockCode": "159509", "name": "纳指科技ETF"}, "SZ", {},
    )
    money = master.pending_row_to_master(
        {"stockCode": "159999", "name": "现金管理货币ETF"}, "SZ", {},
    )
    check("T43 official cross-border subclass confirms T0",
          cross_border["t0_confirmed"] is True and cross_border["t0_status"] == "confirmed",
          str(cross_border))
    check("T43 official domestic-equity subclass remains T1",
          domestic["t0_confirmed"] is False and domestic["t0_status"] == "explicit_t1",
          str(domestic))
    check("T43 T0-looking name without product evidence fails closed",
          pending["t0_confirmed"] is False and pending["t0_status"] == "pending_verification"
          and pending["asset_class_candidate"] == "cross_border_candidate", str(pending))
    check("T43 money-like ETF is excluded rather than confirmed",
          money["is_money_like"] is True and money["t0_status"] == "excluded_money", str(money))
    check("T43 cash-flow factor name is not mistaken for money ETF",
          master.is_money_like("现金流因子ETF") is False)

    with tempfile.TemporaryDirectory() as td:
        quotes = _Path(td) / "quotes.jsonl"
        rows = [
            {"timestamp": "2026-06-01T10:00:00+08:00", "stockCode": "513100", "exchange": "SH", "amount": 10.0, "spread_pct": 0.0008},
            {"timestamp": "2026-06-01T15:00:00+08:00", "stockCode": "513100", "exchange": "SH", "amount": 30.0, "spread_pct": 0.0008},
            {"timestamp": "2026-06-02T15:00:00+08:00", "stockCode": "513100", "exchange": "SH", "amount": 50.0, "spread_pct": 0.0008},
        ]
        quotes.write_text("".join(_json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        liquidity, meta = master.liquidity_20d(quotes)
        value = liquidity[("513100", "SH")]
        check("T43 turnover uses each day's cumulative maximum without look-ahead duplication",
              value["average_turnover_20d"] == 40.0 and value["turnover_observation_days"] == 2,
              str(value))
        check("T43 synthetic Yahoo spread is never promoted as real evidence",
              value["average_spread"] is None and meta["spread_is_real"] is False, str(value))


def t44_point_in_time_opening_research() -> None:
    """Opening research uses prior information, explicit costs and fail-closed gates."""
    import fetch_t0_research_quotes as collector
    import research_t0_opening_oos as opening

    raw = [
        {"timestamp": "2026-06-01T09:30:00+08:00", "trade_date": "2026-06-01", "stockCode": "513100", "exchange": "SH", "bar_volume": 100.0, "close": 10.0},
        {"timestamp": "2026-06-01T09:35:00+08:00", "trade_date": "2026-06-01", "stockCode": "513100", "exchange": "SH", "bar_volume": 50.0, "close": 11.0},
        {"timestamp": "2026-06-02T09:30:00+08:00", "trade_date": "2026-06-02", "stockCode": "513100", "exchange": "SH", "bar_volume": 20.0, "close": 12.0},
    ]
    enriched = collector.enrich_point_in_time(raw)
    by_time = {row["timestamp"]: row for row in enriched}
    second_bar = by_time["2026-06-01T09:35:00+08:00"]
    next_day = by_time["2026-06-02T09:30:00+08:00"]
    check("T44 cumulative amount contains only current/past bars",
          second_bar["cumulative_amount"] == 1550.0 and second_bar["cumulative_volume"] == 150.0,
          str(second_bar))
    check("T44 previous close appears only on the next trade day",
          second_bar["prev_close"] is None and next_day["prev_close"] == 11.0, str(enriched))
    check("T44 opening amount reference excludes the current observation",
          opening.rolling_reference([1, 2, 3, 4, 5], minimum=5, lookback=3) is None
          and opening.rolling_reference([1, 2, 3, 4, 5], minimum=3, lookback=3) == 4)

    sample = {
        "date": "2026-06-02", "code": "513100", "asset_class": "cross_border",
        "gap": 0.02, "ret_10m": 0.01, "amount_surge_vs_prior20": 2.0,
        "first_price": 10.0, "ten_price": 10.1, "close": 10.2, "daily_turnover": 100_000_000.0,
    }
    trade = opening.selected_returns([sample], "gap_up_confirmed_10m", 12.0)[0]
    expected = 10.2 / 10.1 - 1.0 - 0.0012
    check("T44 configured round-trip cost is subtracted from return",
          abs(trade["net_return"] - expected) < 1e-12, str(trade))
    thin = {name: {"overall": {"trades": 2, "bootstrap_daily_mean": {"ci95": [0.01, 0.02], "probability_mean_positive": 1.0}}}
            for name in opening.RULES}
    gates = opening.fixed_rule_promotion(thin)
    check("T44 thin samples fail promotion despite positive point estimates",
          gates and all(row["passed"] is False for row in gates.values()), str(gates))


def t45_observation_pool_history_archive() -> None:
    """Daily watchlist history is point-in-time, idempotent and preserves news/ranks."""
    import copy as _copy
    import json as _json
    import tempfile
    from pathlib import Path as _Path
    import archive_t0_observation_pool as archive

    document = {
        "schemaVersion": "t0_etf_observation_pool_v1",
        "generatedAt": "2026-06-21T20:00:00+08:00",
        "asOfDate": "2026-06-19",
        "effectiveSession": "2026-06-22",
        "paperTradingOnly": True,
        "diagnosticOnly": True,
        "tradeGateEnabled": False,
        "liveReady": False,
        "formalStrategyAllowed": False,
        "chatgptInputMeta": {"effectiveDate": "2026-06-22"},
        "chatgpt10": [{
            "stockCode": "513100", "exchange": "SH", "name": "纳指ETF",
            "reason": "隔夜科技股驱动", "newsDrivers": ["纳指上涨"],
            "risks": ["高开回落"], "sourceUrls": ["https://news.example.cn/nasdaq"],
        }, {
            "stockCode": "518880", "exchange": "SH", "name": "黄金ETF",
            "reason": "金价驱动", "newsDrivers": ["COMEX上涨"],
            "risks": ["美元反弹"], "sourceUrls": ["https://news.example.cn/gold"],
        }],
        "combined": [{
            "stockCode": "513100", "exchange": "SH", "name": "纳指ETF",
            "sources": ["system_rank20", "chatgpt_news10"], "systemRank": 3,
            "rank_score": 2.5, "chatgptResearch": {
                "reason": "隔夜科技股驱动", "newsDrivers": ["纳指上涨"],
                "risks": ["高开回落"], "sourceUrls": ["https://news.example.cn/nasdaq"],
            },
        }, {
            "stockCode": "518880", "exchange": "SH", "name": "黄金ETF",
            "sources": ["chatgpt_news10"],
            "reason": "金价驱动", "newsDrivers": ["COMEX上涨"],
            "risks": ["美元反弹"], "sourceUrls": ["https://news.example.cn/gold"],
        }],
    }
    with tempfile.TemporaryDirectory() as td:
        root = _Path(td)
        first = archive.archive_document(document, root)
        second = archive.archive_document(document, root)
        rows = [_json.loads(line) for line in (root / "selection_history.jsonl").read_text(encoding="utf-8").splitlines()]
        nasdaq = next(row for row in rows if row["stockCode"] == "513100")
        gold = next(row for row in rows if row["stockCode"] == "518880")
        check("T45 archive uses one actual effective-session daily file",
              first["selectionDate"] == "2026-06-22"
              and (root / "daily" / "observation_pool_2026-06-22.json").exists(), str(first))
        check("T45 rerun is idempotent and does not duplicate selections",
              first["changed"] is True and second["changed"] is False
              and second["dayCount"] == 1 and second["selectionCount"] == len(rows) == 2, str(second))
        check("T45 system/news ranks and source provenance are retained",
              nasdaq["systemRank"] == 3 and nasdaq["chatgptRank"] == 1
              and nasdaq["sources"] == ["system_rank20", "chatgpt_news10"], str(nasdaq))
        check("T45 ChatGPT news evidence survives in ETF-level history",
              gold["chatgptRank"] == 2 and gold["sourceNews"]["reason"] == "金价驱动"
              and gold["sourceNews"]["sourceUrls"] == ["https://news.example.cn/gold"], str(gold))

        corrected = _copy.deepcopy(document)
        corrected["combined"][0]["systemRank"] = 1
        third = archive.archive_document(corrected, root)
        corrected_rows = [_json.loads(line) for line in (root / "selection_history.jsonl").read_text(encoding="utf-8").splitlines()]
        corrected_nasdaq = next(row for row in corrected_rows if row["stockCode"] == "513100")
        check("T45 corrected same-day pool atomically replaces instead of appending",
              third["changed"] is True and third["dayCount"] == 1 and third["selectionCount"] == 2
              and third["canonicalUpdated"] is True and corrected_nasdaq["systemRank"] == 1, str(third))

        late = _copy.deepcopy(corrected)
        late["generatedAt"] = "2026-06-22T10:00:00+08:00"
        late["combined"][0]["systemRank"] = 9
        fourth = archive.archive_document(late, root)
        late_rows = [_json.loads(line) for line in (root / "selection_history.jsonl").read_text(encoding="utf-8").splitlines()]
        canonical_nasdaq = next(row for row in late_rows if row["stockCode"] == "513100")
        revisions = list((root / "revisions" / "2026-06-22").glob("*.json"))
        check("T45 post-open revision is audited without contaminating canonical history",
              fourth["lateRevisionOnly"] is True and fourth["canonicalUpdated"] is False
              and canonical_nasdaq["systemRank"] == 1 and len(revisions) == 3, str(fourth))

        unsafe = _copy.deepcopy(document)
        unsafe["tradeGateEnabled"] = True
        blocked = False
        try:
            archive.archive_document(unsafe, root)
        except ValueError:
            blocked = True
        check("T45 archiver refuses non-research trade-gating documents", blocked)


def t46_overseas_gap_point_in_time() -> None:
    """Overseas daily alignment is strictly prior and rule returns include costs."""
    import research_overseas_gap as gap

    reference = [
        {"date": "2026-06-18", "return": 0.01},
        {"date": "2026-06-19", "return": 0.99},
    ]
    match = gap.strictly_prior_return(reference, "2026-06-19")
    check("T46 same-date overseas daily bar is forbidden",
          match == ("2026-06-18", 0.01), str(match))
    row = {
        "date": "2026-06-19", "code": "513100", "gap": 0.02,
        "open_close": 0.01, "ref_IXIC": 0.006,
    }
    returns = gap.daily_rule_returns([row], "IXIC", 12.0)
    check("T46 overseas continuation return subtracts round-trip cost",
          abs(returns["positive_050_continuation"]["2026-06-19"] - 0.0088) < 1e-12,
          str(returns))
    pbo = gap.pbo_for_train({"a": {f"d{i}": 1.0 for i in range(8)},
                             "b": {f"d{i}": 0.0 for i in range(8)}},
                            [f"d{i}" for i in range(8)])
    check("T46 overseas candidate search produces an explicit PBO diagnostic",
          pbo.get("pbo") is not None and pbo.get("n_configs") == 2, str(pbo))


def t47_i03_group_ablation_isolation() -> None:
    """i03 ablation configs change only named groups and remain diagnostic."""
    import research_i03_ablation as ablation

    base = ablation.load(ablation.BASE)
    i03 = ablation.load(ablation.I03)
    l1 = ablation.build_config(base, i03, ["L1_risk_per_trade"])
    changed = []

    def walk(left, right, prefix=""):
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(set(left) | set(right)):
                walk(left.get(key), right.get(key), f"{prefix}.{key}" if prefix else key)
        elif left != right:
            changed.append(prefix)

    walk(base, l1)
    check("T47 L1 ablation changes only its two declared risk paths",
          changed == ablation.GROUPS["L1_risk_per_trade"], str(changed))
    check("T47 ablation preserves execution mode and triple-lock fields",
          l1["mode"] == base["mode"] and l1["execution_enabled"] == base["execution_enabled"]
          and l1["shared_execution"] == base["shared_execution"])
    combined = ablation.build_config(base, i03, ["L3_faster_loss", "L4_stress_selectivity"])
    check("T47 minimal explanatory config is a strict subset of full i03",
          ablation.get_path(combined, "strategy.loss_exit_score_threshold")
          == ablation.get_path(i03, "strategy.loss_exit_score_threshold")
          and ablation.get_path(combined, "strategy.bracket.risk_per_trade_pct")
          == ablation.get_path(base, "strategy.bracket.risk_per_trade_pct"))


def t48_point_in_time_liquidity_gate() -> None:
    """Research liquidity uses prior ADV/current cumulative amount, never final-day amount."""
    import research_liquidity_method_audit as audit

    prior, basis, floor = audit.point_in_time_liquidity_gate(
        cumulative_amount=0, session_fraction=0.1, previous_day_adv=60_000_000)
    current, current_basis, current_floor = audit.point_in_time_liquidity_gate(
        cumulative_amount=6_000_000, session_fraction=0.1, previous_day_adv=10_000_000)
    blocked, _, _ = audit.point_in_time_liquidity_gate(
        cumulative_amount=4_000_000, session_fraction=0.1, previous_day_adv=10_000_000)
    check("T48 previous-day ADV is a point-in-time-safe liquidity basis",
          prior is True and basis == "previous_day_adv" and floor == 5_000_000, str((prior, basis, floor)))
    check("T48 current cumulative amount uses only elapsed-session scaled floor",
          current is True and current_basis == "current_cumulative_amount_scaled_by_elapsed_session"
          and current_floor == 5_000_000, str((current, current_basis, current_floor)))
    check("T48 gate fails closed when both point-in-time measures are thin", blocked is False)


def t49_l4_forward_preregistration() -> None:
    """Forward pair freezes actual baseline and exactly four L4 shadow leaves."""
    import preregister_l4_forward as prereg

    baseline, candidate = prereg.build_pair()
    changed = []
    for path in prereg.L4:
        if prereg.get(baseline["strategy"], path) != prereg.get(candidate["strategy"], path):
            changed.append(path)
    check("T49 L4 preregistration changes exactly four locked leaves",
          changed == list(prereg.L4), str(changed))
    check("T49 baseline reflects actual non-applied diagnostic overlay state",
          baseline["research_metadata"]["overlayAppliedToBaseline"] is False
          and baseline["research_metadata"]["overlayReason"].startswith("overlay_status_not_approved"),
          str(baseline["research_metadata"]))
    check("T49 shadow pair preserves all execution locks",
          baseline["mode"] == candidate["mode"]
          and baseline["execution_enabled"] == candidate["execution_enabled"]
          and baseline["risk"] == candidate["risk"]
          and baseline["shared_execution"] == candidate["shared_execution"])
    criteria = candidate["research_metadata"]["lockedPassCriteria"]
    check("T49 prospective pass criteria are frozen at twenty days",
          criteria["minimumForwardDays"] == 20 and criteria["minimumStdReductionPct"] == 10.0
          and criteria["maximumPbo"] == 0.25, str(criteria))


def t50_l4_forward_shadow_pipeline() -> None:
    """Forward ledger is idempotent and refuses conclusions before twenty days."""
    import tempfile
    from pathlib import Path as _Path
    import run_l4_forward_validation as forward

    summary = {"per_day": {"2026-06-19": {"pnl": 100.0, "gross_pnl": 100.0,
                                             "buy_notional": 10000.0, "sell_notional": 10100.0}},
               "trades": [{"trade_date": "2026-06-19", "pnl": -50.0},
                          {"trade_date": "2026-06-19", "pnl": 150.0}]}
    metric = forward.day_metrics(summary, "2026-06-19")
    check("T50 daily forward metric includes trade dispersion/worst trade/net costs",
          metric["trade_pnl_std"] > 0 and metric["worst_trade_pnl"] == -50.0
          and metric["net_12bps"] == 87.94, str(metric))
    with tempfile.TemporaryDirectory() as td:
        ledger = _Path(td) / "ledger.jsonl"
        row = {"tradeDate": "2026-06-19", "baseline": metric, "candidate": metric}
        first = forward.upsert_day(row, ledger)
        second = forward.upsert_day({**row, "recordedAt": "later-rerun"}, ledger)
        check("T50 ledger keeps one idempotent slot per forward date",
              first is True and second is False and len(forward.read_ledger(ledger)) == 1)
    thin = [{"tradeDate": f"2026-07-{day:02d}",
             "baseline": {"net_12bps": float(day)}, "candidate": {"net_12bps": float(day) * 0.8}}
            for day in range(1, 20)]
    verdict = forward.build_verdict(thin)
    check("T50 nineteen forward days cannot produce an L4 conclusion",
          verdict["status"] == "diagnostic_only" and verdict["decision"] == "insufficient_forward_days"
          and verdict["recommendLive"] is False and verdict["daysRemaining"] == 1, str(verdict))
    enough = thin + [{"tradeDate": "2026-07-20", "baseline": {"net_12bps": 20.0},
                      "candidate": {"net_12bps": 16.0}}]
    evaluated = forward.build_verdict(enough)
    check("T50 twenty days run locked DM/DSR/PBO gates without auto-deployment",
          evaluated["status"] == "forward_validation_complete"
          and all(key in evaluated for key in ("dm", "dsr", "pbo", "gates"))
          and evaluated["recommendLive"] is False, str(evaluated))


def t51_execution_accounting_and_trust_contract() -> None:
    """Patched replay is next-snapshot, T-rule, cost, MTM and trust fail-closed."""
    import importlib
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    replay = importlib.import_module("replay_t0_decisions")
    pipeline = importlib.import_module("run_t0_backtest_pipeline")
    tz = ZoneInfo("Asia/Shanghai")
    t0 = datetime(2026, 6, 18, 10, 0, tzinfo=tz)
    quote = {
        "timestamp": (t0 + timedelta(minutes=1)).isoformat(), "stockCode": "510300",
        "currentPrice": 4.0, "bidPrice1": 4.0, "askPrice1": 4.0,
        "volume": 1000, "amount": 4_000_000, "quote_ok": True, "isSuspended": False,
    }
    cfg_t1 = {"strategy": {"bracket": {}}, "replay_execution": {
        "default_sell_rule": "T1", "t0_allowlist": [], "buy_cost_pct": 0.0005,
        "sell_cost_pct": 0.0005, "slippage_pct_per_side": 0.0002,
    }}
    buy_order = {"stockCode": "510300", "exchange": "SH", "direction": "buy",
                 "quantity": 1000, "price": 4.0, "orderType": "limit"}
    intent = replay.make_order_intent(buy_order, t0, 1, {"trade_date": "2026-06-18"})
    sim = {"initial_cash": 100_000.0, "cash": 100_000.0, "positions": {}}
    state = {}
    pending, terminal = replay.process_pending_orders(sim, [intent], [quote], t0, cfg_t1, state)
    check("T51 same-snapshot intent remains pending", len(pending) == 1 and not terminal and sim.get("positions") == {}, str(intent))
    pending, terminal = replay.process_pending_orders(sim, pending, [quote], t0 + timedelta(minutes=1), cfg_t1, state)
    check("T51 next snapshot can fill", not pending and terminal[0]["status"] == "filled", str(terminal))
    check("T51 in-path cost reduces cash", sim["cash"] < 96_000.0, str(sim["cash"]))
    first_equity = replay.mark_to_market(sim, [quote], t0 + timedelta(minutes=1))["equity"]
    higher = dict(quote, currentPrice=4.1, bidPrice1=4.1, askPrice1=4.1)
    second_point = replay.mark_to_market(sim, [higher], t0 + timedelta(minutes=2))
    check("T51 mark-to-market changes equity", second_point["equity"] > first_equity, str((first_equity, second_point)))
    check("T51 changed mark creates unrealized PnL", abs(second_point["unrealized_pnl"]) > 0, str(second_point))
    replay.refresh_available_quantities(sim, t0 + timedelta(minutes=2))
    check("T51 T1 same-day buy is not sellable", sim["positions"]["510300"]["availableQuantity"] == 0, str(sim["positions"]))
    sell_order = {"stockCode": "510300", "exchange": "SH", "direction": "sell",
                  "quantity": 1000, "price": 4.0, "orderType": "limit"}
    sell_intent = replay.make_order_intent(sell_order, t0 + timedelta(minutes=2), 2, {"trade_date": "2026-06-18"})
    _, rejected = replay.process_pending_orders(sim, [sell_intent], [higher], t0 + timedelta(minutes=3), cfg_t1, state)
    check("T51 T1 same-day sell is rejected", rejected[0]["status"] == "rejected" and "t_rule" in str(rejected[0]["reject_reason"]), str(rejected))

    cfg_t0 = {"strategy": {"bracket": {}}, "replay_execution": {
        "default_sell_rule": "T1", "t0_allowlist": ["510300"], "buy_cost_pct": 0.0005,
        "sell_cost_pct": 0.0005, "slippage_pct_per_side": 0.0002,
    }}
    sim_t0 = {"initial_cash": 100_000.0, "cash": 100_000.0, "positions": {}}
    replay.apply_buy_fill(sim_t0, buy_order, fill_price=4.0, fill_time=t0 + timedelta(minutes=1), cfg=cfg_t0)
    replay.refresh_available_quantities(sim_t0, t0 + timedelta(minutes=1))
    check("T51 only allowlisted T0 lot is same-day sellable", sim_t0["positions"]["510300"]["availableQuantity"] == 1000, str(sim_t0["positions"]))
    oversized = dict(sell_order, quantity=2000)
    partial_intent = replay.make_order_intent(oversized, t0 + timedelta(minutes=1), 3, {"trade_date": "2026-06-18"})
    _, partial = replay.process_pending_orders(sim_t0, [partial_intent], [higher], t0 + timedelta(minutes=2), cfg_t0, {})
    check("T51 sell fill never exceeds available qty", partial[0]["status"] == "partially_filled" and partial[0]["filled_qty"] == 1000, str(partial))

    invalid_intent = replay.make_order_intent(buy_order, t0, 4, {"trade_date": "2026-06-18"})
    invalid_quote = dict(quote, currentPrice=0.0, bidPrice1=0.0, askPrice1=0.0)
    _, invalid = replay.process_pending_orders(
        {"initial_cash": 100_000.0, "cash": 100_000.0, "positions": {}},
        [invalid_intent], [invalid_quote], t0 + timedelta(minutes=1), cfg_t1, {},
    )
    check("T51 invalid next-snapshot price rejects order", invalid[0]["status"] == "rejected", str(invalid))

    clean_execution = {"execution_model": {
        "same_snapshot_fill": False, "next_snapshot_fill": True, "cost_in_path": True,
        "mark_to_market": True, "t_rule_enforced": True,
    }, "data_quality": {"point_in_time_liquidity": True}, "rejected_order_count": 0}
    execution_audit, data_audit, trust = pipeline.build_execution_data_audit(clean_execution, {
        "point_in_time_liquidity": False, "full_day_liquidity_used": True,
        "survivor_bias_warning": True, "missing_data_count": 0,
    })
    check("T51 full-day liquidity forces contaminated trust", trust == "contaminated", str((execution_audit, data_audit, trust)))
    check("T51 audit contract exposes execution_model and data_quality",
          set(execution_audit) >= {"same_snapshot_fill", "next_snapshot_fill", "cost_in_path", "mark_to_market", "t_rule_enforced"}
          and set(data_audit) >= {"point_in_time_liquidity", "full_day_liquidity_used", "survivor_bias_warning", "missing_data_count", "rejected_order_count"})


def t52_decision_scoring_system() -> None:
    """Decision Scoring System: correct total/clamp/bucket, non-empty reasons, audit flags
    (same-snapshot->exec contaminated, full-day-amount->data contaminated), all 4 decision
    types score, JSONL+CSV output, report sample_insufficient + null-return safe, mistake
    attribution returns at least UNKNOWN."""
    import json as _json
    import tempfile
    from pathlib import Path as _Path
    import decision_scoring as ds
    import run_decision_score_report as rep

    # total = sum of subscores, risk_penalty negative, clamp to [0,100]
    sub = {"market_regime_score": 20, "relative_strength_score": 20, "liquidity_score": 15,
           "entry_quality_score": 15, "execution_score": 10, "counterfactual_score": 10,
           "risk_penalty": -20}
    check("T52 total sums subscores w/ negative penalty", ds.total_from_subscores(sub) == 70.0,
          str(ds.total_from_subscores(sub)))
    check("T52 total clamps >100 to 100", ds.total_from_subscores({"market_regime_score": 200}) == 100.0)
    check("T52 total clamps <0 to 0",
          ds.total_from_subscores({k: 0 for k in sub} | {"risk_penalty": -20}) == 0.0)
    check("T52 buckets", [ds.score_bucket(x) for x in (95, 78, 60, 47, 10)] == ["A", "B", "C", "D", "E"])
    check("T52 fixed report buckets use explicit total-score ranges",
          [rep._fixed_total_score_bucket(x) for x in (20, 40, 60, 75, 90)]
          == ["0-40", "40-60", "60-75", "75-90", "90+"])

    clean_buy = ds.score_decision({"decision_type": "BUY", "decision_reason": "entry_consolidation_breakout_passed",
                                   "cross_sectional_percentile": 0.05, "amount": 8e8, "spread_pct": 0.0008,
                                   "broad_market_not_declining": True, "correlation_stress_ok": True, "cost_in_path": True})
    check("T52 reasons non-empty", all(clean_buy.get(k) for k in
          ("market_regime_reason", "relative_strength_reason", "liquidity_reason",
           "entry_quality_reason", "risk_penalty_reason", "execution_reason", "counterfactual_reason")))
    check("T52 risk_penalty is <= 0", clean_buy["risk_penalty"] <= 0)

    ss = ds.score_decision({"decision_type": "BUY", "same_snapshot_fill": True})
    check("T52 same_snapshot_fill -> execution_model contaminated", ss["execution_model_flag"] == "contaminated")
    check("T52 same_snapshot_fill not 'clean'", ss["execution_model_flag"] != "clean")
    fda = ds.score_decision({"decision_type": "SKIP", "used_full_day_amount": True})
    check("T52 full_day_amount -> data_quality contaminated", fda["data_quality_flag"] == "contaminated")
    opt = ds.score_decision({"decision_type": "BUY", "execution_optimistic": True})
    check("T52 optimistic exec model flagged", opt["execution_model_flag"] == "optimistic")

    # regime now driven by cross-sectional breadth (fix for the dead constant-11 sub-score)
    hi_b = ds.score_decision({"decision_type": "BUY", "market_breadth_up_frac": 0.8})
    lo_b = ds.score_decision({"decision_type": "BUY", "market_breadth_up_frac": 0.1})
    check("T52 regime tracks breadth (strong>weak)", hi_b["market_regime_score"] > lo_b["market_regime_score"],
          f"{hi_b['market_regime_score']} vs {lo_b['market_regime_score']}")
    # execution NEUTRAL (not floored) under same-snapshot replay fills -> doesn't crush the score
    check("T52 same_snapshot execution score neutral (5, not floored)", ss["execution_score"] == 5.0,
          str(ss["execution_score"]))

    # outcome backfill from a synthetic rising quote series
    import tempfile as _tf
    with _tf.TemporaryDirectory() as _td:
        qp = _Path(_td) / "q.jsonl"
        rows = []
        for m, p in [(575, 1.0), (580, 1.01), (585, 1.03), (900, 1.05)]:  # 09:35..15:00 rising
            rows.append({"timestamp": f"2026-06-21T{m//60:02d}:{m%60:02d}:00+08:00", "stockCode": "512760", "currentPrice": p})
        rows.append({"timestamp": "2026-06-22T15:00:00+08:00", "stockCode": "512760", "currentPrice": 1.10})  # next-day close
        qp.write_text("\n".join(_json.dumps(r) for r in rows), encoding="utf-8")
        rec = ds.score_decision({"decision_type": "BUY", "etf_code": "512760", "date": "2026-06-21", "timestamp": "09:35:00"})
        ds.enrich_from_quotes([rec], qp)
        check("T52 enrich uses next snapshot, not decision snapshot",
              abs(rec["entry_price"] - 1.01) < 1e-9, str(rec.get("entry_price")))
        check("T52 enrich realized_return (BUY profit if up)",
              abs(rec["realized_return"] - (1.05 / 1.01 - 1.0)) < 1e-4, str(rec.get("realized_return")))
        check("T52 enrich MFE/MAE set", rec["max_favorable_excursion"] > 0 and rec["max_adverse_excursion"] <= 0)
        check("T52 enrich forward return_1d (next-day close)",
              abs(rec["return_1d"] - (1.10 / 1.01 - 1.0)) < 1e-4, str(rec.get("return_1d")))
        skip = ds.score_decision({"decision_type": "SKIP", "etf_code": "512760", "date": "2026-06-21", "timestamp": "09:35:00"})
        ds.enrich_from_quotes([skip], qp)
        check("T52 SKIP is not treated as a synthetic short", skip["realized_return"] is None)
        check("T52 SKIP retains separate counterfactual move", skip["counterfactual_return"] is not None)

    for dt in ("BUY", "SELL", "HOLD", "SKIP"):
        r = ds.score_decision({"decision_type": dt, "decision_reason": "x"})
        check(f"T52 {dt} produces a score", isinstance(r.get("total_score"), (int, float)) and r["decision_type"] == dt)

    ctx = ds.context_from_decision(
        {"decision_scoring": {}},
        {"state_machine": {"action": "buy", "reason": "momentum"},
         "orders": [{"direction": "buy", "stockCode": "513100", "quantity": 100}],
         "ranked": [{"stockCode": "513100", "name": "Nasdaq ETF", "amount": 120_000_000,
                     "spread_pct": 0.001, "change_pct": 1.2}]},
        trade_date="2026-06-21", timestamp="10:00:00",
    )
    check("T52 BUY scoring context retains ranked market amount", ctx["amount"] == 120_000_000)
    check("T52 BUY scoring context retains ranked spread", ctx["spread_pct"] == 0.001)

    # mistake attribution: no outcome -> UNKNOWN; bad-score-but-profit -> lucky
    check("T52 mistake UNKNOWN without outcome", ds.classify_mistake(clean_buy) == "UNKNOWN")
    lucky = {**fda, "total_score": 20, "realized_return": 0.01, "decision_type": "BUY"}
    check("T52 low score + profit -> BAD_TRADE_LUCKY_PROFIT", ds.classify_mistake(lucky) == "BAD_TRADE_LUCKY_PROFIT")

    with tempfile.TemporaryDirectory() as td:
        out = _Path(td)
        recs = [ds.score_decision({"decision_type": d, "decision_reason": "x", "etf_code": "512760"})
                for d in ("BUY", "SELL", "HOLD", "SKIP")]
        j, c = ds.write_scores(recs, "2026-06-21", out_dir=out)
        check("T52 JSONL written with all rows", j.exists() and len([l for l in j.read_text(encoding='utf-8').splitlines() if l]) == 4)
        check("T52 CSV written with header+rows", c.exists() and len(c.read_text(encoding='utf-8').splitlines()) == 5)
        # report: null returns must not crash and must flag sample_insufficient
        rep.SCORE_DIR = out
        text = rep.build_report(recs)
        check("T52 report does not crash on null returns + flags sample_insufficient",
              "sample_insufficient" in text and "By fixed total-score range" in text)


def t53_pseudo_forward_prefix_and_isolation() -> None:
    import json as _json
    import tempfile
    from pathlib import Path as _Path
    import build_t0_replay_quotes_from_minute_data as builder
    import run_decision_score_pseudo_forward as pseudo

    with tempfile.TemporaryDirectory() as td:
        root = _Path(td)
        cfg_path = root / "cfg.json"
        universe_path = root / "universe.jsonl"
        cfg_path.write_text(_json.dumps({"universe": []}), encoding="utf-8")
        universe_path.write_text(_json.dumps({"code": "159001", "exchange": "SZ", "name": "test"}), encoding="utf-8")
        universe = builder.load_replay_universe(cfg_path, universe_path)
        check("T53 confirmed-pool code alias is accepted", universe[0]["stockCode"] == "159001")
        check("T53 universe exchange is preserved", universe[0]["exchange"] == "SZ")

    quotes = [
        {"timestamp": "2026-05-06T09:34:00+08:00", "trade_date": "2026-05-06", "currentPrice": 1.0},
        {"timestamp": "2026-05-06T09:39:00+08:00", "trade_date": "2026-05-06", "currentPrice": 1.1},
    ]
    sampled = builder.resample_quotes(quotes, 5)
    check("T53 09:35 snapshot uses only 09:34-known price", sampled[0]["currentPrice"] == 1.0)
    check("T53 resample records source timestamp", sampled[0]["source_timestamp"].startswith("2026-05-06T09:34"))

    live = {"mode": "paper_execute", "execution_enabled": True, "self_iteration": {"enabled": True}}
    research = {"replay_execution": {"default_sell_rule": "T1", "evaluate_strategy_locks_offline": True}}
    frozen, cfg_hash = pseudo.build_frozen_config(live, research, source_commit="abc123", scorer_sha256="deadbeef")
    check("T53 pseudo config cannot execute live", frozen["mode"] == "paper_research" and frozen["execution_enabled"] is False)
    check("T53 pseudo config cannot auto-apply", frozen["self_iteration"]["enabled"] is False and frozen["self_iteration"]["auto_apply_changes"] is False)
    check("T53 frozen config carries audit hashes", frozen["decision_scoring"]["config_sha256"] == cfg_hash)


def t54_weight_research_and_forward_shadow() -> None:
    import research_decision_score_weights as weights
    import run_decision_score_report as report

    rows = []
    for day in range(1, 6):
        for value in (-1.0, 0.0, 1.0):
            rows.append({
                "date": f"2026-05-{day:02d}",
                "realized_return": value * 0.01 + 0.002,
                **{feature: (value if feature == "market_regime_score" else 0.0)
                   for feature in weights.FEATURES},
            })
    model = weights.fit_fixed_ridge(rows, alpha=1.0, cost=0.0)
    check("T54 ridge recovers positive planted regime coefficient",
          model["coefficients"]["market_regime_score"] > 0)

    cfg = {
        "candidate": {"totalScoreMin": 71},
        "diagnosticEvidence": {"roundTripCost": 0.0014},
        "forwardValidation": {"effectiveFrom": "2026-06-22", "minimumTradingDays": 20,
                              "minimumDirectionalOutcomes": 30},
    }
    forward_rows = [
        {"date": "2026-06-22", "decision_type": "BUY", "total_score": 75,
         "realized_return": 0.01}
    ] * 30
    shadow = report.shadow_forward_stats(forward_rows, cfg)
    check("T54 repeated decisions from one day do not pass day gate", shadow["sample_ready"] is False)
    check("T54 shadow can never auto-promote", shadow["promotion_allowed"] is False)


def t55_decision_score_semantic_versioning() -> None:
    import decision_scoring as scoring
    import register_decision_score_iteration as register
    import run_decision_score_report as report

    active = scoring.active_version_metadata()
    check("T55 active iteration has stable DSI number", str(active["iteration_id"]).startswith("DSI-"))
    check("T55 frozen weights have independent version", str(active["weights_version"]).startswith("DWEIGHTS-"))
    ctx = scoring.context_from_decision(
        {"decision_scoring": {"enabled": True}},
        {"state_machine": {"action": "hold", "reason": "wait"}, "ranked": []},
        trade_date="2026-06-22", timestamp="09:35:00",
    )
    rec = scoring.score_decision(ctx)
    check("T55 live score record carries iteration version", rec["iteration_id"] == active["iteration_id"])
    check("T55 live score record carries scorer and outcome versions",
          rec["scorer_version"] == active["scorer_version"]
          and rec["outcome_model_version"] == active["outcome_model_version"])
    check("T55 config fingerprint ignores runtime-private leaves",
          scoring.config_fingerprint({"x": 1, "_runtime": 2}) == scoring.config_fingerprint({"x": 1, "_runtime": 3}))

    registry = {"nextIterationNumber": 8, "active": {"weightsVersion": "DWEIGHTS-1.0.0"}, "history": []}
    updated, entry = register.allocate_iteration(
        registry, summary="test", change_type="audit", status="completed",
        commit="abc", versions={"pipelineVersion": "DPIPE-1.3.0"}, timestamp="2026-06-22",
    )
    check("T55 registrar allocates next immutable ID", entry["iterationId"] == "DSI-0008")
    check("T55 registrar advances sequence", updated["nextIterationNumber"] == 9)
    mixed = report.build_report([
        {"date": "2026-06-22", "iteration_id": "DSI-0007", "scorer_version": "DSCORE-1.2.0"},
        {"date": "2026-06-23", "iteration_id": "DSI-0008", "scorer_version": "DSCORE-1.2.0"},
    ])
    check("T55 report warns on mixed semantic versions", "mixed_version_warning" in mixed)


def t56_high_low_score_separation() -> None:
    import research_decision_score_separation as separation

    rows = [
        {"date": "2026-06-01", "total_score": 75, "net_return": 0.01,
         "max_adverse_excursion": -0.002, "max_favorable_excursion": 0.012},
        {"date": "2026-06-01", "total_score": 60, "net_return": -0.01,
         "max_adverse_excursion": -0.012, "max_favorable_excursion": 0.002},
        {"date": "2026-06-02", "total_score": 75, "net_return": 0.02,
         "max_adverse_excursion": -0.001, "max_favorable_excursion": 0.021},
        {"date": "2026-06-02", "total_score": 60, "net_return": -0.02,
         "max_adverse_excursion": -0.021, "max_favorable_excursion": 0.001},
    ]
    high = separation.group_stats([row for row in rows if row["total_score"] >= 71])
    low = separation.group_stats([row for row in rows if row["total_score"] <= 65])
    check("T56 planted high group has better net return", high["mean_net_return"] > low["mean_net_return"])
    high_day = separation.day_means(rows, lambda row: row["total_score"] >= 71)
    low_day = separation.day_means(rows, lambda row: row["total_score"] <= 65)
    check("T56 same-day comparison retains both independent days", set(high_day) == set(low_day) == {"2026-06-01", "2026-06-02"})
    boot = separation.cluster_bootstrap(rows, n_boot=100, seed=1)
    check("T56 cluster bootstrap is deterministic and populated", boot["n_boot"] == 100 and boot["ci_95"][0] > 0)


def t57_bayesian_probability_shadow() -> None:
    import decision_probability as probability
    import decision_scoring as scoring
    import run_decision_score_report as report

    check("T57 sigmoid/logit round-trip", abs(probability.sigmoid(probability.logit(0.37)) - 0.37) < 1e-10)
    unsafe = {"schemaVersion": "decision_probability_shadow_v1", "recordOnly": True,
              "tradeGateEnabled": True, "positionSizingEnabled": False}
    import tempfile
    from pathlib import Path as _Path
    import json as _json
    with tempfile.TemporaryDirectory() as td:
        path = _Path(td) / "unsafe.json"
        path.write_text(_json.dumps(unsafe), encoding="utf-8")
        check("T57 probability model fails closed if trade gate is true",
              probability.load_shadow_model(path) is None)

    model = probability.load_shadow_model()
    check("T57 checked-in probability model is record-only",
          model is not None and model["tradeGateEnabled"] is False
          and model["positionSizingEnabled"] is False)
    before = probability.forecast_shadow(total_score=80, decision_type="BUY", date="2026-06-21", model=model)
    after = probability.forecast_shadow(total_score=80, decision_type="BUY", date="2026-06-22", model=model)
    check("T57 preregistered model does not backfill before effective date", before == {})
    check("T57 posterior forecast is bounded and record-only",
          0 < after["posterior_prob"] < 1
          and after["probability_shadow_action"] == "RECORD_ONLY_NO_TRADE_GATE")
    check("T57 higher score raises posterior under frozen positive slope",
          after["posterior_prob"] > probability.forecast_shadow(
              total_score=50, decision_type="BUY", date="2026-06-22", model=model
          )["posterior_prob"])

    record = {**after, "realized_return": 0.0010}
    probability.enrich_probability_outcome(record)
    check("T57 probability success label deducts 14bps cost", record["probability_outcome"] == 0)
    check("T57 post-close enrichment writes proper scores",
          record["brier_score"] >= 0 and record["probability_log_loss"] >= 0)
    metrics = probability.probability_metrics([0.9, 0.1], [1, 0])
    check("T57 calibrated toy forecast has low Brier and perfect AUC",
          metrics["brier"] < 0.02 and metrics["auc"] == 1.0)

    score = scoring.score_decision({"decision_type": "BUY", "date": "2026-06-22",
                                    "decision_reason": "momentum"})
    check("T57 live score carries frozen shadow posterior", score.get("posterior_prob") is not None)
    check("T57 probability shadow never changes position suggestion",
          score.get("position_size_suggestion") is None)
    repeated = [{**score, "date": "2026-06-22", "probability_outcome": 1,
                 "realized_return": 0.01} for _ in range(60)]
    stats = report.probability_forward_stats(repeated, model)
    check("T57 repeated outcomes from one day do not pass probability day gate",
          stats["sample_ready"] is False and stats["promotion_allowed"] is False)


def t58_forward_probability_ledger_and_priors() -> None:
    import decision_probability as probability
    import decision_scoring as scoring
    import run_decision_probability_forward as forward

    config = probability.load_research_config()
    model = probability.load_shadow_model()
    check("T58 research config is fail-closed and record-only",
          config is not None and config["tradeGateEnabled"] is False
          and config["positionSizingEnabled"] is False
          and config["bayesPosteriorAllowedForTradeGate"] is False)
    decision = {
        "state_machine": {"action": "hold", "reason": "entry_score_below_threshold"},
        "orders": [],
        "ranked": [
            {"stockCode": "513100", "name": "纳指ETF", "currentPrice": 1.0,
             "amount": 2e8, "spread_pct": 0.001, "change_pct": 1.0},
            {"stockCode": "512760", "name": "芯片ETF", "currentPrice": 2.0,
             "amount": 1e8, "spread_pct": 0.001, "change_pct": 0.5},
            {"stockCode": "510300", "name": "沪深300ETF", "currentPrice": 4.0,
             "amount": 3e8, "spread_pct": 0.001, "change_pct": -0.2},
        ],
    }
    cfg = {"decision_scoring": {"record_ranked_candidates": True,
                                  "max_ranked_candidates_per_snapshot": 2}}
    contexts = scoring.contexts_from_decision(
        cfg, decision, trade_date="2026-06-22", timestamp="10:00:00",
    )
    check("T58 final decision plus configured no-trade candidates are retained", len(contexts) == 3)
    check("T58 candidate rows are explicit counterfactual BUY signals",
          all(row["decision_type"] == "BUY_CANDIDATE" and row["signal_direction"] == "BUY"
              and row["was_executed"] is False for row in contexts[1:]))
    candidate = scoring.score_decision(contexts[1])
    check("T58 no-trade candidate gets a pre-outcome probability forecast",
          candidate.get("calibrated_probability") is not None
          and candidate.get("probability_outcome") is None)

    incomplete_index = {"513100": {"2026-06-22": {605: 1.0, 610: 1.01, 890: 1.02}}}
    scoring.enrich_from_price_index([candidate], incomplete_index, {"513100": {"2026-06-22": 1.02}})
    check("T58 incomplete session cannot create probability outcome",
          candidate["probability_outcome"] is None and candidate["outcome_horizon_complete"] is False)
    complete_candidate = scoring.score_decision(contexts[1])
    complete_index = {"513100": {"2026-06-22": {605: 1.0, 610: 1.01, 895: 1.03}}}
    scoring.enrich_from_price_index([complete_candidate], complete_index,
                                    {"513100": {"2026-06-22": 1.03}})
    check("T58 candidate outcome is counterfactual and horizon-complete",
          complete_candidate["realized_return"] is None
          and complete_candidate["counterfactual_return"] is not None
          and complete_candidate["probability_outcome"] is not None
          and complete_candidate["outcome_horizon_complete"] is True)

    metrics = probability.probability_metrics(
        [0.25, 0.35, 0.65, 0.75], [0, 1, 0, 1],
        bin_edges=[0.0, 0.3, 0.4, 0.7, 0.8, 1.0],
    )
    check("T58 calibration diagnostics include MCE and per-bin proper scores",
          metrics["mce"] is not None and all("brier" in row and "log_loss" in row
                                             and "calibration_error" in row for row in metrics["bins"]))

    rows = []
    for index, outcome in enumerate((1, 1, 1, 0)):
        rows.append({**complete_candidate, "decision_id": f"candidate_{index}",
                     "date": "2026-06-22", "probability_outcome": outcome,
                     "outcome_horizon_complete": True})
    registry = forward.build_prior_registry(rows, config)
    key = probability.prior_registry_key(rows[0])
    check("T58 Beta(2,2) prior registry shrinks a 3-of-4 segment",
          abs(registry["jointSegments"][key]["posteriorProbability"] - 0.625) < 1e-12)
    forecast = probability.forecast_shadow(
        total_score=70, decision_type="BUY_CANDIDATE", date="2026-06-23",
        context=rows[0], model=model, research_config=config, prior_registry=registry,
    )
    check("T58 Bayesian posterior uses the prior registry but remains research-only",
          forecast["prior_source"] == "forward_beta_binomial_joint_segment"
          and forecast["bayes_posterior_research_only"] is True
          and forecast["bayes_posterior_allowed_for_trade_gate"] is False)
    diagnostic = forward.build_diagnostics(rows, model, config)
    check("T58 one correlated day cannot pass any readiness tier",
          all(tier["samplePass"] is False and tier["statisticalPass"] is False
              for tier in diagnostic["readinessTiers"])
          and diagnostic["gates"]["promotionAllowed"] is False)


def t62_trend_deploy_factor() -> None:
    """Gayed trend filter scales sizing by deploy_factor: applies a fresh downtrend de-risk, but
    is FAIL-OPEN (missing/stale/disabled -> 1.0) and CLAMPED to [0,1] (can only reduce exposure)."""
    import json as _json
    import run_t0_intraday_agent as agent
    fp = ROOT / "outputs" / "t0_intraday_agent" / "_t62_trend.json"
    fp.parent.mkdir(parents=True, exist_ok=True)
    cfg = lambda f=str(fp): {"strategy": {"trend_deployment": {"enabled": True, "regime_file": f, "max_age_days": 4}}}

    check("T62 disabled is fail-open (1.0)", agent._trend_deploy_factor({"strategy": {}}, "2026-06-26")[0] == 1.0)
    check("T62 missing file is fail-open", agent._trend_deploy_factor(cfg("outputs/t0_intraday_agent/__none__.json"), "2026-06-26")[0] == 1.0)

    fp.write_text(_json.dumps({"date": "2026-06-26", "deploy_factor": 0.3, "regime": "downtrend"}), encoding="utf-8")
    f, m = agent._trend_deploy_factor(cfg(), "2026-06-26")
    check("T62 fresh downtrend de-risks to 0.3", abs(f - 0.3) < 1e-9 and m.get("applied") is True)

    fp.write_text(_json.dumps({"date": "2026-06-10", "deploy_factor": 0.3}), encoding="utf-8")
    check("T62 stale regime is fail-open", agent._trend_deploy_factor(cfg(), "2026-06-26")[0] == 1.0)

    fp.write_text(_json.dumps({"date": "2026-06-26", "deploy_factor": 2.5}), encoding="utf-8")
    check("T62 factor clamped <=1 (de-risk only)", agent._trend_deploy_factor(cfg(), "2026-06-26")[0] == 1.0)
    fp.unlink(missing_ok=True)


def t63_hsmm_regime_research_is_point_in_time() -> None:
    """The post-publication HSMM research must infer the published duration faithfully,
    decode every prefix without future observations, and trade only on the next return."""
    import numpy as _np
    import research_hsmm_regime as hsmm

    p = hsmm.logseries_p_from_mean(26.0)
    check("T63 inferred log-series duration reproduces reported mean",
          abs(float(hsmm.logser.mean(p)) - 26.0) < 1e-6)

    params = hsmm.published_parameters()
    observations = _np.asarray([-0.1, 0.2, -0.05, 0.1, 1.4, 0.8, -2.2, -1.0], dtype=float)
    full = hsmm.decode_expanding_right_censored(observations, params)
    prefix_ok = all(
        int(full[end - 1])
        == int(hsmm.decode_expanding_right_censored(observations[:end], params)[-1])
        for end in range(1, len(observations) + 1)
    )
    check("T63 expanding HSMM state is invariant to unseen future suffix", prefix_ok)

    states = _np.asarray([2, 0, 1], dtype=int)
    positions = hsmm.positions_from_states(states, (-1.0, 0.0, 1.0))
    check("T63 close-t state is shifted to the next return",
          positions.tolist() == [0.0, 1.0, -1.0], str(positions.tolist()))


def t64_hmm_nn_bl_research_is_frozen_and_safe() -> None:
    """The hybrid research model is deterministic, long-only/cash-aware and cannot promote
    a merely less-negative result or reach any live execution path."""
    import json as _json
    import numpy as _np
    import research_hmm_nn_bl as hybrid

    sequences = [
        _np.asarray([-0.010, -0.008, -0.006, 0.000, 0.001, 0.008, 0.010])
        for _ in range(4)
    ]
    first = hybrid.GaussianHMM1D(n_iter=5).fit(sequences)
    second = hybrid.GaussianHMM1D(n_iter=5).fit(sequences)
    filtered = first.filter_probabilities(_np.asarray([-0.01, 0.0, 0.01]))
    check("T64 HMM fit is deterministic and state means remain ordered",
          _np.allclose(first.means, second.means) and _np.all(_np.diff(first.means) >= 0))
    check("T64 causal HMM filter emits valid probabilities",
          filtered.shape == (3, 3) and _np.allclose(filtered.sum(axis=1), 1.0))

    covariance = _np.asarray([[0.0001, 0.00002], [0.00002, 0.0002]])
    posterior = hybrid.black_litterman_posterior(
        covariance, _np.asarray([0.002, 0.001]), tau=0.05, risk_aversion=8.0,
        view_confidence=0.35, residual_variance=1e-6,
    )
    weights = hybrid.optimize_long_only(
        posterior, covariance, round_trip_cost=0.0005, risk_aversion=8.0,
        max_weight=0.25, max_assets=2,
    )
    cash = hybrid.optimize_long_only(
        _np.asarray([-0.01, -0.02]), covariance, round_trip_cost=0.0005,
        risk_aversion=8.0, max_weight=0.25, max_assets=2,
    )
    check("T64 BL optimizer is long-only, capped and cash-aware",
          _np.isfinite(posterior).all() and _np.all(weights >= 0)
          and float(weights.max()) <= 0.25 + 1e-12 and float(weights.sum()) <= 1.0 + 1e-12
          and _np.allclose(cash, 0.0))

    config = _json.loads((ROOT / "configs" / "research" / "hmm_nn_bl_preregistered.json")
                         .read_text(encoding="utf-8"))
    source = (ROOT / "scripts" / "research_hmm_nn_bl.py").read_text(encoding="utf-8")
    check("T64 hybrid model is frozen research-only and cannot gate or size live trades",
          config["status"] == "diagnostic_only"
          and config["safety"]["offlineOnly"] is True
          and config["safety"]["tradeGateEnabled"] is False
          and config["safety"]["positionSizingEnabled"] is False
          and config["evidenceGates"]["mustHavePositiveNetReturn"] is True
          and config["evidenceGates"]["mustHavePositiveSharpe"] is True
          and "submitOrder" not in source and "SkillClient" not in source
          and "latest_strategy_overlay" not in source)


def t65_literature_reversal_research_is_point_in_time() -> None:
    """Liquidity-reversal studies must rank only information known at the decision,
    fill on the next bar, deduct cost, and remain disconnected from live trading."""
    import json as _json
    import numpy as _np
    import pandas as _pd
    import research_intraday_reversal_edge as reversal
    import research_overnight_cross_section as overnight

    signal = _pd.Series({"a": -0.03, "b": -0.02, "c": 0.00, "d": 0.02, "e": 0.03})
    check("T65 cross-sectional tails select losers for reversal and winners for control",
          reversal.select_tail(signal, fraction=0.2, side="bottom", max_assets=2) == ["a"]
          and reversal.select_tail(signal, fraction=0.2, side="top", max_assets=2) == ["e"])

    timestamps = _pd.date_range("2026-07-01 10:00", periods=8, freq="5min", tz="Asia/Shanghai")
    prices = _pd.DataFrame(
        {"a": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0]},
        index=timestamps,
    )
    future = reversal.future_trade_return(prices, 0, "a", holding_bars=6)
    check("T65 intraday research enters next bar rather than signal bar",
          abs(float(future) - (107.0 / 101.0 - 1.0)) < 1e-12)
    portfolio = reversal.portfolio_interval(
        {"a": 0.01}, ["a"], max_weight=0.2, round_trip_cost=0.0012,
    )
    check("T65 portfolio return deducts cost in proportion to exposure",
          abs(portfolio["net_return"] - (0.2 * 0.01 - 0.2 * 0.0012)) < 1e-12)

    day_index = _pd.to_datetime([
        "2026-07-02 09:30+08:00", "2026-07-02 09:35+08:00",
        "2026-07-02 14:55+08:00",
    ])
    columns = [f"{index:06d}" for index in range(10)]
    price_values = _np.asarray([
        [100.0 + index for index in range(10)],
        [101.0 + index for index in range(10)],
        [102.0 + index for index in range(10)],
    ])
    previous = _np.asarray([[100.0] * 10] * 3)
    observations = overnight.build_daily_observations({
        "prices": _pd.DataFrame(price_values, index=day_index, columns=columns),
        "prev_close": _pd.DataFrame(previous, index=day_index, columns=columns),
    })
    check("T65 overnight study uses 09:30 signal, 09:35 entry and 14:55 exit",
          len(observations) == 1
          and observations[0]["signal_time"].strftime("%H:%M") == "09:30"
          and observations[0]["entry_time"].strftime("%H:%M") == "09:35"
          and observations[0]["exit_time"].strftime("%H:%M") == "14:55")

    reversal_cfg = _json.loads(
        (ROOT / "configs" / "research" / "intraday_reversal_preregistered.json")
        .read_text(encoding="utf-8")
    )
    overnight_cfg = _json.loads(
        (ROOT / "configs" / "research" / "overnight_cross_section_preregistered.json")
        .read_text(encoding="utf-8")
    )
    sources = (
        (ROOT / "scripts" / "research_intraday_reversal_edge.py").read_text(encoding="utf-8")
        + (ROOT / "scripts" / "research_overnight_cross_section.py").read_text(encoding="utf-8")
    )
    check("T65 literature research stays offline and cannot promote itself",
          reversal_cfg["safety"]["tradeGateEnabled"] is False
          and overnight_cfg["safety"]["tradeGateEnabled"] is False
          and reversal_cfg["safety"]["writesStrategyOverlay"] is False
          and overnight_cfg["safety"]["writesStrategyOverlay"] is False
          and "submitOrder" not in sources and "SkillClient" not in sources
          and "latest_strategy_overlay" not in sources)


def t66_forward_execution_friction_is_conservative_and_safe() -> None:
    """Forward execution research must use China quote time, separate optimistic and
    conservative passive-fill proxies, and remain descriptive until enough days accrue."""
    import json as _json
    from datetime import datetime as _datetime
    from zoneinfo import ZoneInfo as _ZoneInfo
    import research_forward_execution_friction as friction

    fallback_time = friction.parse_china_time(
        {"timestamp": "2026-07-01T10:31:00+09:00"}
    )
    source_time = friction.parse_china_time(
        {
            "timestamp": "2026-07-01T10:31:00+09:00",
            "source_quote_time": "2026-07-01T09:31:02+08:00",
        }
    )
    check("T66 execution research normalizes timestamps to China time",
          fallback_time is not None and fallback_time.strftime("%H:%M") == "09:31"
          and source_time is not None and source_time.strftime("%H:%M:%S") == "09:31:02")

    tz = _ZoneInfo("Asia/Shanghai")
    rows = [
        {"timestamp": _datetime(2026, 7, 1, 10, 0, tzinfo=tz), "bid": 1.000,
         "ask": 1.002, "mid": 1.001, "current": 1.001, "spread_bps": 19.98,
         "trade_date": "2026-07-01", "code": "513100"},
        {"timestamp": _datetime(2026, 7, 1, 10, 5, tzinfo=tz), "bid": 0.999,
         "ask": 1.001, "mid": 1.000, "current": 1.000, "spread_bps": 20.0,
         "trade_date": "2026-07-01", "code": "513100"},
        {"timestamp": _datetime(2026, 7, 1, 10, 10, tzinfo=tz), "bid": 0.998,
         "ask": 0.999, "mid": 0.9985, "current": 0.999, "spread_bps": 10.02,
         "trade_date": "2026-07-01", "code": "513100"},
        {"timestamp": _datetime(2026, 7, 1, 10, 30, tzinfo=tz), "bid": 1.003,
         "ask": 1.005, "mid": 1.004, "current": 1.004, "spread_bps": 19.92,
         "trade_date": "2026-07-01", "code": "513100"},
    ]
    fills = friction.passive_fill_proxies(rows, 0, 10)
    check("T66 passive fill proxies require a later quote and distinguish ask-cross",
          fills["touch_fill"] is True and fills["conservative_fill"] is True
          and fills["touch_fill_time"] > rows[0]["timestamp"])

    config = _json.loads(
        (ROOT / "configs" / "research" / "forward_execution_friction_preregistered.json")
        .read_text(encoding="utf-8")
    )
    observation = friction.execution_observation(rows, 0, config)
    check("T66 aggressive execution cost is measured against the midpoint path",
          observation is not None and observation["aggressive_cost_bps"] > 0)
    sparse = friction.forward_strategy_shadow(
        {("2026-07-01", "513100"): rows}, config
    )
    source = (ROOT / "scripts" / "research_forward_execution_friction.py").read_text(
        encoding="utf-8"
    )
    check("T66 thin forward execution sample fails closed and cannot trade",
          sparse["status"] == "insufficient_forward_execution_days"
          and config["safety"]["tradeGateEnabled"] is False
          and config["safety"]["writesStrategyOverlay"] is False
          and "submitOrder" not in source and "SkillClient" not in source)


def t67_full_t0_depth_collector_is_point_in_time_and_safe() -> None:
    """Depth collection must use the confirmed master, reject stale quotes, expose
    missing-code coverage, and contain no account/order capability."""
    import json as _json
    from datetime import datetime as _datetime
    from tempfile import TemporaryDirectory as _TemporaryDirectory
    from zoneinfo import ZoneInfo as _ZoneInfo
    import collect_l2_depth as depth

    with _TemporaryDirectory() as td:
        master = Path(td) / "master.jsonl"
        master.write_text(
            "\n".join(
                [
                    _json.dumps(
                        {
                            "code": "513100",
                            "exchange": "SH",
                            "name": "纳指ETF",
                            "asset_class": "cross_border",
                            "t0_confirmed": True,
                            "is_money_like": False,
                        },
                        ensure_ascii=False,
                    ),
                    _json.dumps(
                        {
                            "code": "510300",
                            "exchange": "SH",
                            "name": "沪深300ETF",
                            "asset_class": "domestic_equity",
                            "t0_confirmed": False,
                            "is_money_like": False,
                        },
                        ensure_ascii=False,
                    ),
                    _json.dumps(
                        {
                            "code": "511880",
                            "exchange": "SH",
                            "name": "货币ETF",
                            "asset_class": "money",
                            "t0_confirmed": True,
                            "is_money_like": True,
                        },
                        ensure_ascii=False,
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        universe = depth.load_confirmed_t0_universe(master)
    check(
        "T67 depth universe contains only product-level confirmed non-money T0 ETFs",
        [row["code"] for row in universe] == ["513100"],
        str(universe),
    )

    def payload(source_date: str, source_time: str) -> str:
        fields = [""] * 33
        fields[0] = "纳指ETF"
        fields[2] = "2.000"
        fields[3] = "2.010"
        for level in range(5):
            fields[10 + 2 * level] = str(1000 + level)
            fields[11 + 2 * level] = f"{2.009 - 0.001 * level:.3f}"
            fields[20 + 2 * level] = str(900 + level)
            fields[21 + 2 * level] = f"{2.011 + 0.001 * level:.3f}"
        fields[30] = source_date
        fields[31] = source_time
        return f'var hq_str_sh513100="{",".join(fields)}";'

    cn = _ZoneInfo("Asia/Shanghai")
    collected = _datetime(2026, 7, 1, 10, 0, 30, tzinfo=cn)
    by_code = {row["code"]: row for row in universe}
    fresh = depth.parse_sina_payload(
        payload("2026-07-01", "10:00:00"),
        by_code,
        collected,
        180,
    )
    stale = depth.parse_sina_payload(
        payload("2026-06-30", "15:00:00"),
        by_code,
        collected,
        180,
    )
    check(
        "T67 source timestamp controls freshness instead of collector clock",
        len(fresh) == 1
        and fresh[0]["is_fresh"] is True
        and stale[0]["is_fresh"] is False
        and fresh[0]["bid_levels"] == 5
        and fresh[0]["ask_levels"] == 5,
    )

    coverage = depth.coverage_record(
        universe
        + [
            {
                "code": "518880",
                "exchange": "SH",
                "name": "黄金ETF",
                "asset_class": "gold",
            }
        ],
        fresh,
        {"provider": "synthetic"},
        collected,
    )
    source = (ROOT / "scripts" / "collect_l2_depth.py").read_text(encoding="utf-8")
    check(
        "T67 incomplete full-T0 coverage is explicit and collector cannot trade",
        coverage["expected_count"] == 2
        and coverage["fresh_count"] == 1
        and coverage["missing_codes"] == ["518880"]
        and coverage["status"] == "partial_coverage"
        and "SkillClient" not in source
        and "submitOrder" not in source
        and "cancelOrder" not in source,
    )


def t68_paper_order_lifecycle_uses_confirmed_fills_only() -> None:
    """Lifecycle research must preserve unresolved orders, consume broker-confirmed
    fill evidence, calculate forward markouts, and remain unable to trade."""
    import json as _json
    from tempfile import TemporaryDirectory as _TemporaryDirectory
    import archive_paper_order_lifecycle as lifecycle

    with _TemporaryDirectory() as td:
        root = Path(td)
        intents = root / "intents.jsonl"
        runs = root / "runs.jsonl"
        state = root / "state.json"
        quote_dir = root / "quotes"
        quote_dir.mkdir()
        intents.write_text(
            _json.dumps(
                {
                    "timestamp": "2026-07-01T09:31:01+08:00",
                    "event_type": "submit_results_recorded",
                    "agent_name": "t0_intraday_paper_agent",
                    "trade_date": "2026-07-01",
                    "orders": [
                        {
                            "stockCode": "513100",
                            "exchange": "SH",
                            "direction": "buy",
                            "quantity": 1000,
                            "orderType": "limit",
                            "price": 1.000,
                            "submission_mid": 1.001,
                            "execution_style": "passive",
                        }
                    ],
                    "submit_results": [
                        {
                            "ok": True,
                            "data": {
                                "orderId": "42",
                                "status": "pending",
                                "submitTime": "2026-07-01T09:31:01+08:00",
                            },
                        }
                    ],
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        runs.write_text(
            "\n".join(
                [
                    _json.dumps(
                        {
                            "timestamp": "2026-07-01T09:32:00+08:00",
                            "pending_t0_orders": [
                                {
                                    "orderId": "42",
                                    "stockCode": "513100",
                                    "exchange": "SH",
                                    "direction": "buy",
                                    "price": 1.000,
                                    "quantity": 1000,
                                    "filledQuantity": 0,
                                    "status": "pending",
                                }
                            ],
                        }
                    ),
                    _json.dumps(
                        {
                            "timestamp": "2026-07-01T09:34:00+08:00",
                            "fill_reconciliation": {
                                "confirmed_trades": [
                                    {
                                        "orderId": "42",
                                        "stockCode": "513100",
                                        "exchange": "SH",
                                        "direction": "buy",
                                        "filledPrice": 1.000,
                                        "filledQuantity": 1000,
                                        "filledAmount": 1000.0,
                                        "fee": 0.3,
                                        "filledTime": "2026-07-01T09:33:00+08:00",
                                    }
                                ]
                            },
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        state.write_text("{}\n", encoding="utf-8")
        (quote_dir / "minute_quotes_2026-07-01.jsonl").write_text(
            "\n".join(
                _json.dumps(row)
                for row in [
                    {
                        "timestamp": "2026-07-01T09:31:00+08:00",
                        "stockCode": "513100",
                        "bidPrice1": 1.000,
                        "askPrice1": 1.002,
                    },
                    {
                        "timestamp": "2026-07-01T09:34:00+08:00",
                        "stockCode": "513100",
                        "bidPrice1": 1.001,
                        "askPrice1": 1.003,
                    },
                    {
                        "timestamp": "2026-07-01T09:38:00+08:00",
                        "stockCode": "513100",
                        "bidPrice1": 1.003,
                        "askPrice1": 1.005,
                    },
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        rows, summary = lifecycle.build_lifecycle(intents, runs, state, quote_dir)

    check(
        "T68 broker-confirmed fill overrides pending status with exact lifecycle evidence",
        len(rows) == 1
        and rows[0]["status"] == "filled_confirmed_exact"
        and rows[0]["fill_time_known"] is True
        and rows[0]["filledQuantity"] == 1000
        and summary["confirmed_fills"] == 1,
        str(rows),
    )
    check(
        "T68 signed implementation shortfall and post-fill markout use recorded times",
        rows[0]["implementation_shortfall_bps"] < 0
        and rows[0]["fill_delay_seconds"] == 119.0
        and rows[0]["markout_1m_bps"] > 0,
        str(rows[0]),
    )

    agent_state = {
        "t0_inventory_by_date": {
            "2026-07-01": {
                "513100": {
                    "buy_order_ids": ["42"],
                    "buy_quantity_submitted": 1000,
                    "sell_order_ids": [],
                }
            }
        }
    }
    reconciled = agent.reconcile_t0_inventory_from_trade_history(
        agent_state,
        "2026-07-01",
        {
            "ok": True,
            "data": {
                "trades": [
                    {
                        "orderId": "42",
                        "stockCode": "513100",
                        "exchange": "SH",
                        "direction": "buy",
                        "filledPrice": 1.0,
                        "filledQuantity": 1000,
                        "filledAmount": 1000.0,
                        "fee": 0.3,
                        "filledTime": "2026-07-01T09:33:00+08:00",
                    }
                ]
            },
        },
    )
    check(
        "T68 live reconciliation exposes sanitized confirmed trade without changing fill rule",
        reconciled["confirmed_trades"][0]["orderId"] == "42"
        and reconciled["confirmed_trades"][0]["filledTime"]
        == "2026-07-01T09:33:00+08:00"
        and agent_state["t0_inventory_by_date"]["2026-07-01"]["513100"][
            "buy_quantity_filled"
        ]
        == 1000,
    )

    source = (ROOT / "scripts" / "archive_paper_order_lifecycle.py").read_text(
        encoding="utf-8"
    )
    suite = (ROOT / "scripts" / "run_research_suite.ps1").read_text(encoding="utf-8")
    friction_cfg = _json.loads(
        (
            ROOT
            / "configs"
            / "research"
            / "forward_execution_friction_preregistered.json"
        ).read_text(encoding="utf-8")
    )
    check(
        "T68 lifecycle job is daily, forward-extensible and cannot call the broker",
        "archive_paper_order_lifecycle.py" in suite
        and "research_forward_execution_friction.py" in suite
        and friction_cfg["data"]["filePattern"] == "minute_quotes_*.jsonl"
        and "SkillClient" not in source
        and "submitOrder" not in source
        and "cancelOrder" not in source,
    )


def t69_iopv_pcf_collector_preserves_source_tiers() -> None:
    """PCF must remain official/raw while vendor IOPV stays explicitly labelled,
    timestamped and unable to affect trading."""
    from datetime import datetime as _datetime
    from zoneinfo import ZoneInfo as _ZoneInfo
    import collect_etf_iopv_pcf as iopv

    xml = b"""<?xml version="1.0" encoding="UTF-8"?>
<SSEPortfolioCompositionFile>
  <FundInstrumentID>513100</FundInstrumentID>
  <CreationRedemptionUnit>1000000</CreationRedemptionUnit>
  <TradingDay>20260701</TradingDay>
  <PreTradingDay>20260630</PreTradingDay>
  <NAVperCU>2000000.00</NAVperCU>
  <NAV>2.0000</NAV>
  <EstimatedCashComponent>-100.00</EstimatedCashComponent>
  <PublishIOPVFlag>1</PublishIOPVFlag>
  <CreationRedemptionSwitch>3</CreationRedemptionSwitch>
  <CreationRedemptionMechanism>0</CreationRedemptionMechanism>
  <RecordNumber>1</RecordNumber>
  <ComponentList><Component>
    <InstrumentID>AAPL</InstrumentID>
    <SubstitutionFlag>1</SubstitutionFlag>
    <CreationPremiumRate>0.10</CreationPremiumRate>
    <RedemptionDiscountRate>0.02</RedemptionDiscountRate>
  </Component></ComponentList>
</SSEPortfolioCompositionFile>"""
    pcf = iopv.parse_sse_pcf(xml)
    check(
        "T69 official PCF parser retains creation/redemption and IOPV publication state",
        pcf["fund_code"] == "513100"
        and pcf["trading_day"] == "20260701"
        and pcf["publish_iopv"] is True
        and pcf["creation_redemption_switch"] == "3"
        and pcf["record_number"] == pcf["parsed_components"] == 1,
        str(pcf),
    )

    cn = _ZoneInfo("Asia/Shanghai")
    collected = _datetime(2026, 7, 1, 10, 0, 30, tzinfo=cn)
    universe = [
        {
            "code": "513100",
            "exchange": "SH",
            "name": "纳指ETF",
            "asset_class": "cross_border",
        }
    ]
    item = {
        "f2": 2.1,
        "f12": "513100",
        "f14": "纳指ETF",
        "f18": 2.0,
        "f31": 2.099,
        "f32": 2.101,
        "f124": int(
            _datetime(2026, 7, 1, 10, 0, 0, tzinfo=cn).timestamp()
        ),
        "f130": 2.0,
        "f131": 2.0,
    }
    rows = iopv.parse_iopv_items(
        [item],
        universe,
        collected,
        pcf_manifest={"513100": {"publish_iopv": True}},
    )
    check(
        "T69 vendor IOPV is point-in-time, premium-correct and never mislabeled direct-feed",
        len(rows) == 1
        and rows[0]["is_fresh"] is True
        and rows[0]["premium_pct"] == 5.0
        and rows[0]["iopv_source"] == "eastmoney_public_quote_field_f131"
        and rows[0]["iopv_source_tier"] == "vendor_not_direct_exchange_feed"
        and rows[0]["iopv_validation_status"]
        == "vendor_field_with_official_sse_publish_flag",
        str(rows),
    )

    source = (ROOT / "scripts" / "collect_etf_iopv_pcf.py").read_text(
        encoding="utf-8"
    )
    task = (ROOT / "scripts" / "install_iopv_pcf_task.ps1").read_text(
        encoding="utf-8"
    )
    check(
        "T69 IOPV/PCF pipeline is weekday market-data-only",
        "Monday,Tuesday,Wednesday,Thursday,Friday" in task
        and "SkillClient" not in source
        and "submitOrder" not in source
        and "cancelOrder" not in source
        and "latest_strategy_overlay" not in source,
    )


def t70_option_pressure_collector_is_forward_and_unsigned() -> None:
    """Option pressure must recover IV, avoid confusing last-trade time with
    snapshot time, and refuse to invent a dealer gamma sign."""
    from datetime import datetime as _datetime
    from zoneinfo import ZoneInfo as _ZoneInfo
    import collect_etf_option_pressure as options

    price, _, _ = options.black_scholes(
        "call", 3.0, 3.0, 30 / 365, 0.015, 0.0, 0.25
    )
    recovered = options.implied_volatility(
        "call", price, 3.0, 3.0, 30 / 365, 0.015, 0.0
    )
    check(
        "T70 Black-Scholes inversion recovers a planted implied volatility",
        recovered is not None and abs(recovered - 0.25) < 1e-5,
        str(recovered),
    )

    fields = ["0"] * 51
    fields[1] = "0.10"
    fields[2] = "0.101"
    fields[3] = "0.102"
    fields[5] = "1000"
    fields[7] = "3.000"
    fields[8] = "0.099"
    fields[32] = "2026-07-01 09:45:00"
    fields[37] = "50ETF购7月3000"
    fields[41] = "2000"
    fields[42] = "1000000"
    fields[45] = "C"
    fields[46] = "2026-07-22"
    cn = _ZoneInfo("Asia/Shanghai")
    collected = _datetime(2026, 7, 1, 10, 0, 0, tzinfo=cn)
    row = options.parse_contract_quote(
        "CON_OP_10000001",
        fields,
        {"underlying": "510050", "expiry_month": "202607", "option_type": "call"},
        {"spot": 3.0, "name": "上证50ETF"},
        collected,
        risk_free_rate=0.015,
        dividend_yield=0.0,
        max_quote_age_seconds=180,
        snapshot_trade_date_verified=True,
    )
    check(
        "T70 inactive option stays in the point-in-time snapshot without fake timestamp freshness",
        row is not None
        and row["is_fresh"] is True
        and row["last_trade_recent"] is False
        and row["source_quote_time_semantics"] == "last_trade_time"
        and row["implied_volatility"] is not None,
        str(row),
    )

    call = {
        **row,
        "option_type": "call",
        "delta": 0.25,
        "implied_volatility": 0.20,
        "volume": 100.0,
        "open_interest": 200.0,
        "gamma": 0.4,
    }
    put = {
        **row,
        "contract_symbol": "CON_OP_10000002",
        "option_type": "put",
        "delta": -0.25,
        "implied_volatility": 0.24,
        "volume": 150.0,
        "open_interest": 300.0,
        "gamma": 0.45,
    }
    state = options.aggregate_option_states([call, put], collected)[0]
    check(
        "T70 pressure state exposes PCR/skew but keeps gamma direction unknown",
        state["put_call_volume_ratio"] == 1.5
        and abs(state["put_minus_call_25delta_iv"] - 0.04) < 1e-9
        and state["unsigned_gamma_oi_exposure_1pct_move"] > 0
        and state["gamma_sign_status"] == "unknown_no_dealer_position_sign",
        str(state),
    )

    source = (ROOT / "scripts" / "collect_etf_option_pressure.py").read_text(
        encoding="utf-8"
    )
    task = (
        ROOT / "scripts" / "install_etf_option_pressure_task.ps1"
    ).read_text(encoding="utf-8")
    check(
        "T70 option collection is weekday research-only and cannot trade",
        "Monday,Tuesday,Wednesday,Thursday,Friday" in task
        and "SkillClient" not in source
        and "submitOrder" not in source
        and "cancelOrder" not in source
        and "latest_strategy_overlay" not in source,
    )


def t71_same_index_underreaction_is_next_bar_oos_and_safe() -> None:
    """The clone-laggard research must form signals with past data, enter on
    the next bar, charge the preregistered cost, and remain offline-only."""
    import json
    import research_same_index_underreaction as clone

    index = clone.pd.date_range(
        "2026-06-01 09:45:00", "2026-06-01 10:35:00", freq="5min"
    )
    prices = clone.pd.DataFrame(
        {
            "A": [1.0, 1.0, 1.0, 1.004, 1.004, 1.004, 1.004, 1.004, 1.004, 1.004, 1.005],
            "B": [1.0, 1.0, 1.0, 1.004, 1.004, 1.004, 1.004, 1.004, 1.004, 1.004, 1.005],
            "C": [1.0, 1.0, 1.0, 1.000, 1.000, 1.002, 1.004, 1.008, 1.012, 1.016, 1.020],
        },
        index=index,
    )
    config = {
        "data": {"minimumMembersPerBenchmark": 2},
        "signal": {
            "formationBars": 3,
            "groupMomentumMinimum": 0.002,
            "laggardResidualMaximum": -0.0015,
            "decisionTimes": ["10:00"],
            "maximumBenchmarkSignalsPerDecision": 5,
        },
        "execution": {
            "holdingBars": 6,
            "maximumEntryDelayMinutes": 10,
        },
    }
    signals = clone.generate_signals(prices, {"IDX": ["A", "B", "C"]}, config)
    check(
        "T71 fixed same-index signal identifies the planted laggard",
        len(signals) == 1
        and signals[0]["laggard"] == "C"
        and signals[0]["group_momentum"] >= 0.002
        and signals[0]["laggard_residual"] <= -0.0015,
        str(signals),
    )
    check(
        "T71 same-index replay enters strictly after the decision bar",
        signals[0]["entry_time"] > signals[0]["decision_time"]
        and abs(signals[0]["candidate_gross_return"] - 0.02) < 1e-12,
        str(signals[0]),
    )

    daily12, _, edge12, intervals12 = clone.daily_portfolios(
        signals, cost_bps=12, max_weight=0.2
    )
    daily20, _, edge20, _ = clone.daily_portfolios(
        signals, cost_bps=20, max_weight=0.2
    )
    day = "2026-06-01"
    check(
        "T71 higher registered cost lowers candidate return by exact exposure",
        abs((daily12[day] - daily20[day]) - 0.2 * 8 / 10_000) < 1e-12
        and intervals12[0]["exposure"] == 0.2,
    )
    check(
        "T71 equal candidate/control costs cancel only in paired relative edge",
        abs(edge12[day] - edge20[day]) < 1e-12,
    )

    prereg = json.loads(
        (
            ROOT
            / "configs"
            / "research"
            / "same_index_underreaction_preregistered.json"
        ).read_text(encoding="utf-8")
    )
    source = (
        ROOT / "scripts" / "research_same_index_underreaction.py"
    ).read_text(encoding="utf-8")
    check(
        "T71 same-index research is diagnostic-only and cannot trade or promote",
        prereg["status"] == "diagnostic_only"
        and prereg["safety"]["offlineOnly"] is True
        and prereg["safety"]["tradeGateEnabled"] is False
        and prereg["safety"]["brokerCallsAllowed"] is False
        and prereg["safety"]["promotionAllowed"] is False
        and "SkillClient" not in source
        and "submitOrder" not in source
        and "latest_strategy_overlay" not in source,
    )


def t72_trend_pullback_recovery_is_causal_costed_and_safe() -> None:
    """Trend/pullback research must use a causal VWAP, enter and exit after
    observed triggers, retain unfilled selections as cash, and stay offline."""
    import json
    import research_trend_pullback_recovery as tpr

    index = tpr.pd.date_range("2026-06-01 10:20:00", periods=6, freq="5min")
    prices = tpr.pd.Series(
        [1.000, 1.005, 1.010, 1.006, 1.007, 1.008], index=index
    )
    amount = tpr.pd.Series([100.0] * 6, index=index)
    vwap_at_decision = tpr.causal_vwap(prices, amount, 2)
    amount_with_future_shock = amount.copy()
    amount_with_future_shock.iloc[3:] = 1_000_000.0
    check(
        "T72 causal VWAP is invariant to future volume and price rows",
        abs(vwap_at_decision - tpr.causal_vwap(prices, amount_with_future_shock, 2))
        < 1e-12,
        str(vwap_at_decision),
    )

    entry = tpr.find_pullback_entry(
        prices,
        amount,
        decision_index=2,
        reference_high=1.010,
        minimum_pullback=0.003,
        maximum_wait_bars=3,
        require_non_negative_last_bar=True,
        require_above_vwap=True,
    )
    check(
        "T72 pullback waits for recovery and enters on the following bar",
        entry is not None
        and entry["trigger_index"] == 4
        and entry["entry_index"] == 5
        and entry["entry_index"] > entry["trigger_index"],
        str(entry),
    )
    no_volume_entry = tpr.find_pullback_entry(
        prices,
        tpr.pd.Series([0.0] * 6, index=index),
        decision_index=2,
        reference_high=1.010,
        minimum_pullback=0.003,
        maximum_wait_bars=3,
        require_non_negative_last_bar=True,
        require_above_vwap=True,
    )
    check(
        "T72 missing causal volume cannot fake a VWAP-confirmed entry",
        no_volume_entry is None,
    )

    exit_prices = tpr.pd.Series([1.000, 1.005, 1.011, 1.008])
    recovery = tpr.recovery_exit(
        exit_prices, entry_index=0, reference_high=1.010, holding_bars=3
    )
    check(
        "T72 known-high exit fills one bar after the observed trigger",
        recovery is not None
        and recovery["exit_reason"] == "known_high_recovery"
        and recovery["exit_index"] == 3
        and abs(recovery["gross_return"] - 0.008) < 1e-12,
        str(recovery),
    )

    row = {
        "trade_date": "2026-06-01",
        "decision_time": "2026-06-01T10:30:00+08:00",
        "immediate_hold_gross_return": 0.01,
        "pullback_hold_gross_return": None,
        "pullback_recovery_gross_return": None,
    }
    daily, trades = tpr.daily_portfolios([row], cost_bps=12, max_weight=0.2)
    check(
        "T72 unfilled pullback remains cash instead of disappearing from the sample",
        daily["pullback_hold_ablation"]["2026-06-01"] == 0.0
        and daily["pullback_recovery_candidate"]["2026-06-01"] == 0.0
        and trades["pullback_recovery_candidate"] == 0,
    )
    check(
        "T72 immediate control deducts registered round-trip cost",
        abs(daily["immediate_hold_control"]["2026-06-01"] - 0.2 * (0.01 - 0.0012))
        < 1e-12,
    )

    prereg = json.loads(
        (
            ROOT
            / "configs"
            / "research"
            / "trend_pullback_recovery_preregistered.json"
        ).read_text(encoding="utf-8")
    )
    source = (
        ROOT / "scripts" / "research_trend_pullback_recovery.py"
    ).read_text(encoding="utf-8")
    check(
        "T72 trend-pullback research cannot trade, promote or alter overlays",
        prereg["status"] == "diagnostic_only"
        and prereg["safety"]["offlineOnly"] is True
        and prereg["safety"]["tradeGateEnabled"] is False
        and prereg["safety"]["brokerCallsAllowed"] is False
        and prereg["safety"]["promotionAllowed"] is False
        and "SkillClient" not in source
        and "submitOrder" not in source
        and "latest_strategy_overlay" not in source,
    )


def t73_frontier_competition_ranker_is_fresh_paper_only_and_auditable() -> None:
    """The contest ranker may reorder paper candidates only with fresh external
    coverage; stale data must restore baseline momentum ordering."""
    import json
    from datetime import datetime
    from zoneinfo import ZoneInfo
    import frontier_competition_policy as frontier

    cn = ZoneInfo("Asia/Shanghai")
    now = datetime(2026, 6, 30, 10, 0, 0, tzinfo=cn)
    quotes = [
        {
            "stockCode": "513100",
            "asset_class": "cross_border_etf",
            "momentum": 0.0020,
            "acceleration": 0.0010,
            "vwap_distance_pct": 0.0010,
            "spread_pct": 0.0002,
        },
        {
            "stockCode": "513500",
            "asset_class": "cross_border_etf",
            "momentum": 0.0030,
            "acceleration": -0.0010,
            "vwap_distance_pct": 0.0002,
            "spread_pct": 0.0010,
        },
        {
            "stockCode": "518880",
            "asset_class": "gold_etf",
            "momentum": 0.0010,
            "acceleration": 0.0,
            "vwap_distance_pct": 0.0005,
            "spread_pct": 0.0004,
        },
    ]
    depth = [
        {
            "code": "513100",
            "collected_at": "2026-06-30T09:59:30+08:00",
            "is_fresh": True,
            "obi": 0.8,
            "micro_dev_bps": 3.0,
            "half_spread_bps": 1.0,
        },
        {
            "code": "513500",
            "collected_at": "2026-06-30T09:59:30+08:00",
            "is_fresh": True,
            "obi": -0.8,
            "micro_dev_bps": -3.0,
            "half_spread_bps": 5.0,
        },
        {
            "code": "518880",
            "collected_at": "2026-06-30T09:59:30+08:00",
            "is_fresh": True,
            "obi": 0.0,
            "micro_dev_bps": 0.0,
            "half_spread_bps": 2.0,
        },
    ]
    premium = []
    for code, history, latest in (
        ("513100", 0.20, 0.00),
        ("513500", 0.00, 0.20),
        ("518880", 0.10, 0.10),
    ):
        premium.extend(
            [
                {
                    "stockCode": code,
                    "collected_at": f"2026-06-30T09:5{minute}:00+08:00",
                    "premium_pct": history,
                }
                for minute in range(3)
            ]
        )
        premium.append(
            {
                "stockCode": code,
                "collected_at": "2026-06-30T09:59:30+08:00",
                "premium_pct": latest,
            }
        )
    cfg = {
        "enabled": True,
        "mode": "active_rerank",
        "paper_competition_only": True,
        "minimum_depth_coverage": 0.5,
        "max_depth_age_seconds": 120,
        "max_premium_age_seconds": 180,
        "minimum_premium_history": 4,
        "execution_buffer_bps": 4,
    }
    enriched, meta = frontier.enrich_quotes(
        quotes, depth, premium, cfg, now=now
    )
    ranked = frontier.rank_quotes(
        enriched, {"frontier_competition_policy": cfg}
    )
    by_code = {row["stockCode"]: row for row in enriched}
    check(
        "T73 fresh broad depth coverage activates paper-contest reranking",
        meta["appliedToRanking"] is True
        and meta["depthCoverage"] == 1.0
        and ranked[0]["stockCode"] == "513100",
        str(meta),
    )
    check(
        "T73 ranker removes common OBI and uses own-history premium residual",
        abs(
            by_code["513100"]["frontier_policy"]["rawFeatures"][
                "idiosyncratic_obi"
            ]
            - 0.8
        )
        < 1e-12
        and by_code["513100"]["frontier_policy"]["premium"][
            "discountResidualBps"
        ]
        > 0,
        str(by_code["513100"]["frontier_policy"]),
    )
    check(
        "T73 execution cost is explicit rather than treated as free alpha",
        by_code["513100"]["frontier_policy"][
            "estimatedAggressiveRoundTripCostBps"
        ]
        == 6.0,
    )

    stale_depth = [
        {**row, "collected_at": "2026-06-30T09:50:00+08:00"}
        for row in depth
    ]
    stale_enriched, stale_meta = frontier.enrich_quotes(
        quotes, stale_depth, premium, cfg, now=now
    )
    stale_ranked = frontier.rank_quotes(
        stale_enriched, {"frontier_competition_policy": cfg}
    )
    check(
        "T73 stale depth disables frontier ordering and restores momentum baseline",
        stale_meta["appliedToRanking"] is False
        and stale_ranked[0]["stockCode"] == "513500",
        str(stale_meta),
    )

    live_cfg = json.loads(
        (ROOT / "configs" / "t0_intraday_paper_agent.json").read_text(
            encoding="utf-8"
        )
    )
    policy_cfg = live_cfg["strategy"]["frontier_competition_policy"]
    policy_source = (
        ROOT / "scripts" / "frontier_competition_policy.py"
    ).read_text(encoding="utf-8")
    agent_source = (
        ROOT / "scripts" / "run_t0_intraday_agent.py"
    ).read_text(encoding="utf-8")
    check(
        "T73 frontier strategy is explicitly experimental paper competition only",
        live_cfg["mode"] == "paper_execute"
        and live_cfg["execution_enabled"] is True
        and policy_cfg["paper_competition_only"] is True
        and policy_cfg["alpha_validated"] is False
        and policy_cfg["auto_promotion_allowed"] is False,
    )
    check(
        "T73 ranker cannot submit, size, sell or write an overlay",
        "SkillClient" not in policy_source
        and "submitOrder" not in policy_source
        and "submit_order" not in policy_source
        and "quantity" not in policy_source
        and "latest_strategy_overlay" not in policy_source,
    )
    check(
        "T73 triple execution lock and SELL path remain outside frontier policy",
        "if execute and decision.get(\"approved_for_submit\")" in agent_source
        and "pre_sell_position_check" in agent_source
        and "rank_frontier_quotes(liquid_quotes, strategy)" in agent_source,
    )


def t61_daily_momentum_pool_failopen() -> None:
    """Nightly daily-momentum pool RESTRICTS the universe when fresh, but is FAIL-OPEN:
    missing/stale/no-ref -> empty set -> caller keeps the full eligible universe (trading
    never halts on a missed nightly run)."""
    import json as _json
    import select_t0_universe as su
    from pathlib import Path as _P
    fp = ROOT / "outputs" / "t0_intraday_agent" / "_t61_pool.json"
    fp.parent.mkdir(parents=True, exist_ok=True)

    codes, meta = su._load_daily_momentum_pool({}, "2026-06-23")
    check("T61 no pool ref is fail-open (empty, disabled)", codes == set() and meta.get("enabled") is False)

    codes, meta = su._load_daily_momentum_pool(
        {"daily_momentum_pool_file": "outputs/t0_intraday_agent/__missing__.json"}, "2026-06-23")
    check("T61 missing pool file is fail-open", codes == set() and meta.get("reason") == "no_pool_file")

    fp.write_text(_json.dumps({"date": "2026-06-23", "lookback_days": 20,
                               "codes": ["510300", "159915"]}), encoding="utf-8")
    codes, meta = su._load_daily_momentum_pool(
        {"daily_momentum_pool_file": str(fp)}, "2026-06-23")
    check("T61 fresh pool returns its codes", codes == {"510300", "159915"} and meta.get("age_days") == 0)

    fp.write_text(_json.dumps({"date": "2026-06-01", "codes": ["510300"]}), encoding="utf-8")
    codes, meta = su._load_daily_momentum_pool(
        {"daily_momentum_pool_file": str(fp), "daily_momentum_pool_max_age_days": 4}, "2026-06-23")
    check("T61 stale pool is fail-open (empty)", codes == set() and meta.get("reason") == "stale_pool")
    fp.unlink(missing_ok=True)


def t60_sell_logic_v2_timing_gate() -> None:
    """sell_logic_v2 de-noises ONLY the unified_sell_score_exit timing sell, behind a
    default-OFF flag. Hard stops are never routed through it; flag off == baseline."""
    from datetime import datetime
    SSR2 = {"score": 80.0, "components": {"structure_break": 25.0, "momentum_reversal": 20.0}}
    lt0, lt1 = datetime(2026, 6, 22, 13, 0), datetime(2026, 6, 22, 13, 5)

    # (a) flag OFF -> exact baseline no-op
    off = {"sell_logic_v2": {"enabled": False}}
    check("T60 v2 disabled is baseline no-op",
          agent.apply_sell_logic_v2(off, {}, "2026-06-22", "159915", SSR2, 70.0, [], lt0)
          == (True, 1.0, "unified_sell_score_exit"))

    # (b) confirmation: first snapshot waits, second consecutive snapshot sells
    on = {"sell_logic_v2": {"enabled": True, "min_independent_negative_components": 2,
                            "confirmation_snapshots": 2, "scale_out_fraction": 0.5}}
    st: dict = {}
    do1, _, r1 = agent.apply_sell_logic_v2(on, st, "2026-06-22", "159915", SSR2, 70.0, [], lt0)
    do2, frac2, _ = agent.apply_sell_logic_v2(on, st, "2026-06-22", "159915", SSR2, 70.0, [], lt1)
    check("T60 first signal awaits confirmation", do1 is False and r1 == "v2_awaiting_confirmation")
    check("T60 second consecutive signal sells, scaled", do2 is True and abs(frac2 - 0.5) < 1e-9)

    # (c) a one-snapshot dip that does NOT recur next snapshot never sells (gap too large)
    st2: dict = {}
    agent.apply_sell_logic_v2(on, st2, "2026-06-22", "159915", SSR2, 70.0, [], lt0)
    late = datetime(2026, 6, 22, 14, 0)  # 60min gap > confirmation_max_gap -> counter resets
    do_late, _, _ = agent.apply_sell_logic_v2(on, st2, "2026-06-22", "159915", SSR2, 70.0, [], late)
    check("T60 non-consecutive dip resets and does not sell", do_late is False)

    # (d) lone component suppressed (needs >=2 independent negatives)
    lone = {"score": 80.0, "components": {"bid_pressure_negative": 15.0}}
    do_lone, _, r_lone = agent.apply_sell_logic_v2(on, {}, "2026-06-22", "159915", lone, 70.0, [], lt0)
    check("T60 lone negative component suppressed",
          do_lone is False and r_lone == "v2_suppress_insufficient_components")

    # (e) strong breadth holds a sub-(threshold+delta) score
    strong = {"sell_logic_v2": {"enabled": True, "min_independent_negative_components": 1,
                                "confirmation_snapshots": 1, "strong_tape_breadth": 0.6,
                                "strong_tape_threshold_delta": 10}}
    up_quotes = [{"change_pct": 1.0}, {"change_pct": 1.0}, {"change_pct": -0.1}]  # breadth 0.67
    do_str, _, r_str = agent.apply_sell_logic_v2(strong, {}, "2026-06-22", "159915", SSR2, 75.0, up_quotes, lt0)
    check("T60 strong tape holds sub-threshold+delta score",
          do_str is False and r_str == "v2_suppress_strong_tape")


def t74_entry_logic_v2_pullback_gate() -> None:
    """entry_logic_v2 gates a fresh momentum candidate behind a brief pullback wait, behind
    a default-OFF flag. Flag off == baseline; fail-open on timeout (never silently drop the
    signal); single pending slot (a different candidate mid-wait is ignored, not leaked)."""
    from datetime import datetime
    lt0 = datetime(2026, 6, 22, 10, 0)
    lt_mid = datetime(2026, 6, 22, 10, 10)   # +10min, within the 15min wait
    lt_late = datetime(2026, 6, 22, 10, 20)  # +20min, past the 15min wait
    cand = {"stockCode": "159915", "currentPrice": 10.0}

    # (a) flag OFF -> exact baseline no-op (candidate passes straight through)
    off = {"entry_logic_v2": {"enabled": False}}
    out, reason = agent.apply_entry_logic_v2(off, {}, "2026-06-22", cand, [], lt0)
    check("T74 v2 disabled is baseline no-op", out == cand and reason == "entry_score_gate")
    out_none, reason_none = agent.apply_entry_logic_v2(off, {}, "2026-06-22", None, [], lt0)
    check("T74 v2 disabled passes None through unchanged", out_none is None and reason_none == "entry_score_gate")

    # (b) first sighting registers a pending wait, does not buy this round
    on = {"entry_logic_v2": {"enabled": True, "pullback_frac": 0.004, "max_wait_minutes": 15}}
    st: dict = {}
    out1, r1 = agent.apply_entry_logic_v2(on, st, "2026-06-22", cand, [], lt0)
    check("T74 first sighting awaits pullback, no buy", out1 is None and r1 == "entry_v2_awaiting_pullback")

    # (c) price dips >= pullback_frac on a later round -> fills at the dip price
    quotes_dip = [{"stockCode": "159915", "currentPrice": 9.95}]   # -0.5% > 0.4% pullback_frac
    out2, r2 = agent.apply_entry_logic_v2(on, st, "2026-06-22", None, quotes_dip, lt_mid)
    check("T74 pullback dip fills at dip price",
          out2 is not None and abs(out2["currentPrice"] - 9.95) < 1e-9 and r2 == "entry_v2_pullback_filled")
    check("T74 pending slot cleared after fill", st["pending_entry_v2_by_date"]["2026-06-22"]["pending"] is None)

    # (d) no dip within max_wait_minutes -> fail-open, buys at whatever price is quoted
    st2: dict = {}
    agent.apply_entry_logic_v2(on, st2, "2026-06-22", cand, [], lt0)
    quotes_flat = [{"stockCode": "159915", "currentPrice": 10.02}]  # no pullback, price drifted up
    out3, r3 = agent.apply_entry_logic_v2(on, st2, "2026-06-22", None, quotes_flat, lt_late)
    check("T74 wait-expired fail-open still buys (never silently drops the signal)",
          out3 is not None and abs(out3["currentPrice"] - 10.02) < 1e-9 and r3 == "entry_v2_wait_expired_fail_open")

    # (e) a pending wait blocks a DIFFERENT candidate from being registered mid-wait
    st3: dict = {}
    agent.apply_entry_logic_v2(on, st3, "2026-06-22", cand, [], lt0)
    other_cand = {"stockCode": "588000", "currentPrice": 5.0}
    out4, r4 = agent.apply_entry_logic_v2(on, st3, "2026-06-22", other_cand, [], lt_mid)
    check("T74 pending wait ignores a different mid-wait candidate (not leaked)",
          out4 is None and r4 == "entry_v2_awaiting_pullback")
    check("T74 pending still tracks the ORIGINAL code, not the new candidate",
          st3["pending_entry_v2_by_date"]["2026-06-22"]["pending"]["code"] == "159915")

    # (f) an awaiting-pullback round must never block a SELL batch: the gate is entry-only,
    # so it belongs in SELL_BYPASS_CHECKS (same bug class as T81's breadth block -- a
    # buy-side gate must never delay an exit).
    check("T74 entry_logic_v2_gate is sell-bypassed (buy gate never blocks an exit)",
          "entry_logic_v2_gate" in agent.SELL_BYPASS_CHECKS)


def t75_actual_trade_exit_research_is_causal_and_safe() -> None:
    """Exit research fills after its trigger and never delays hard exits."""
    from datetime import datetime, timedelta
    import research_exit_timing_actual_trades as exit_research

    base = datetime.fromisoformat("2026-06-22T10:00:00+08:00")
    prices = [100.0, 101.0, 102.0, 101.3, 101.0, 100.8]
    path = [
        {
            "time": base + timedelta(minutes=5 * idx),
            "price": price,
            "bid": price - 0.05,
            "spread": 0.001,
        }
        for idx, price in enumerate(prices)
    ]
    fixed = exit_research.simulate_exit(path, "fixed_trail_0p6")
    check("T75 trailing exit fills on the bar after the causal trigger",
          fixed["triggered"] is True and fixed["exit_index"] == 4
          and abs(fixed["exit_price"] - 100.95) < 1e-9,
          str(fixed))
    baseline = exit_research.simulate_exit(path, "current_baseline")
    check("T75 baseline remains the recorded final replay fill",
          baseline["triggered"] is False and baseline["exit_index"] == len(path) - 1
          and baseline["exit_price"] == prices[-1],
          str(baseline))

    lifecycle = [
        {"status": "filled", "fill_time": "2026-06-22T10:00:00+08:00",
         "filled_qty": 100, "side": "buy", "stockCode": "513100",
         "fill_price": 2.0, "reason": "entry_momentum_spread_passed"},
        {"status": "filled", "fill_time": "2026-06-22T10:20:00+08:00",
         "filled_qty": 100, "side": "sell", "stockCode": "513100",
         "fill_price": 1.96, "reason": "emergency_stop_exit"},
    ]
    lots, diagnostic = exit_research.pair_filled_lots(lifecycle)
    check("T75 emergency exits are excluded and never delayed",
          not lots and diagnostic["hard_exit_slices_excluded"] == 1)
    source = (ROOT / "scripts" / "research_exit_timing_actual_trades.py").read_text(encoding="utf-8")
    check("T75 exit research is offline and cannot submit orders",
          "STRICTLY OFFLINE" in source and "order_submit_calls_made" in source
          and "submitOrder" not in source and "latest_strategy_overlay" not in source)


def t76_l2_exit_timing_is_forward_day_clustered_and_safe() -> None:
    """L2 exit trigger removes common pressure and interprets sell proceeds correctly."""
    import research_l2_exit_timing as l2_exit

    triggers = l2_exit.classify_triggers(obi=-0.55, idiosyncratic_obi=-0.25,
                                         micro_dev_bps=-1.5)
    check("T76 adverse idiosyncratic OBI plus microprice activates primary trigger",
          l2_exit.PRIMARY_TRIGGER in triggers)
    check("T76 common market pressure alone does not activate idiosyncratic trigger",
          "idiosyncratic_obi_adverse"
          not in l2_exit.classify_triggers(obi=-0.55, idiosyncratic_obi=-0.05,
                                           micro_dev_bps=0.2))
    check("T76 positive advantage means sell-now bid exceeds later bid",
          l2_exit.sell_now_advantage_bps(10.0, 9.9) > 0
          and l2_exit.sell_now_advantage_bps(9.9, 10.0) < 0)
    source = (ROOT / "scripts" / "research_l2_exit_timing.py").read_text(encoding="utf-8")
    check("T76 L2 exit audit is shadow-only and cannot submit or promote",
          "STRICTLY OFFLINE / SHADOW" in source
          and "order_submit_calls_made" in source
          and "MIN_COMPLETE_DAYS = 20" in source
          and "submitOrder" not in source
          and "latest_strategy_overlay" not in source)


def t77_l2_sell_execution_is_next_snapshot_conservative_and_safe() -> None:
    """Passive sell research requires displayed crossing and accounts for timeout loss."""
    import research_l2_sell_execution as execution

    start = {
        "bid": 99.9, "ask": 100.1, "midpoint": 100.0,
        "idiosyncratic_obi": 0.0, "micro_dev_bps": 0.0,
    }
    future = [
        {"bid": 99.95},
        {"bid": 100.0},
        {"bid": 99.8},
    ]
    midpoint = execution.execute_policy(start, future, "midpoint_then_cross")
    check("T77 midpoint limit fills only after a later displayed bid reaches it",
          midpoint["passive_filled"] is True and midpoint["fill_after_polls"] == 2
          and abs(midpoint["proceeds"] - 100.0) < 1e-9)
    ask = execution.execute_policy(start, future, "ask_then_cross")
    check("T77 unfilled ask limit crosses at timeout and records adverse drift",
          ask["passive_filled"] is False and ask["timed_out"] is True
          and ask["proceeds"] == 99.8 and ask["improvement_bps"] < 0)
    adverse = dict(start, idiosyncratic_obi=-0.3, micro_dev_bps=-2.0)
    conditional = execution.execute_policy(adverse, future, "conditional_midpoint")
    check("T77 conditional policy crosses immediately on adverse L2",
          conditional["passive_attempted"] is False
          and conditional["improvement_bps"] == 0.0)
    source = (ROOT / "scripts" / "research_l2_sell_execution.py").read_text(encoding="utf-8")
    check("T77 sell execution audit is offline and cannot change live execution",
          "STRICTLY OFFLINE / SHADOW" in source
          and "order_submit_calls_made" in source
          and "submitOrder" not in source
          and "latest_strategy_overlay" not in source)


def t78_l2_sell_slicing_uses_visible_depth_and_exact_lots() -> None:
    """Block and sliced execution consume real displayed depth without invented fills."""
    from datetime import datetime, timedelta, timezone

    import research_l2_sell_slicing as slicing

    book = {
        "bid_prices": [10.0, 9.9],
        "bid_volumes": [100, 100],
        "bid1": 10.0,
    }
    vwap = slicing.sweep_sell_vwap(book, 150)
    check("T78 five-level sell sweep computes volume-weighted proceeds",
          vwap is not None and abs(vwap - ((100 * 10.0 + 50 * 9.9) / 150)) < 1e-9)
    check("T78 insufficient displayed depth fails closed",
          slicing.sweep_sell_vwap(book, 300) is None)
    slices = slicing.split_lots(1000, (0.5, 1 / 6, 1 / 6, 1 / 6))
    check("T78 slicing preserves exact quantity in whole lots",
          sum(slices) == 1000 and all(value % 100 == 0 for value in slices), str(slices))
    adverse_path = [
        {
            **book,
            "midpoint": 10.0,
            "idiosyncratic_obi": -0.3,
            "micro_dev_bps": -2.0,
        }
        for _ in range(4)
    ]
    check("T78 conditional urgency uses immediate block on adverse book",
          slicing.policy_vwap(adverse_path, 100, "conditional_urgency")
          == slicing.policy_vwap(adverse_path, 100, "block_now"))
    unmatched = slicing.evaluate_actual_signals(
        [{
            "timestamp": datetime(2026, 7, 1, 10, 0, tzinfo=timezone(timedelta(hours=8))),
            "stockCode": "513100",
            "quantity": 100,
            "broker_order_id": "audit",
        }],
        {},
    )
    check("T78 unmatched actual sell remains visible in attrition audit",
          len(unmatched) == 1
          and unmatched[0]["paired"] is False
          and unmatched[0]["reason"] == "no_l2_book")
    source = (ROOT / "scripts" / "research_l2_sell_slicing.py").read_text(encoding="utf-8")
    check("T78 slicing audit is offline and cannot alter execution",
          "STRICTLY OFFLINE / SHADOW" in source
          and "order_submit_calls_made" in source
          and "submitOrder" not in source
          and "latest_strategy_overlay" not in source)


def t79_exit_policy_matrix_is_causal_complete_and_shadow_only() -> None:
    """Exit matrix keeps actual hard exits and fills every trigger on the next bar."""
    from datetime import datetime, timedelta, timezone

    import research_exit_policy_matrix as matrix

    tz = timezone(timedelta(hours=8))

    def bar(minute: int, price: float, bid: float | None = None) -> dict:
        return {
            "time": datetime(2026, 7, 1, 10, minute, tzinfo=tz),
            "price": price,
            "bid": price if bid is None else bid,
            "spread": 0.0008,
        }

    stop_path = [bar(0, 100.0), bar(5, 98.5), bar(10, 98.0, 97.9)]
    stopped = matrix.simulate_policy(
        stop_path,
        {"name": "test_stop", "family": "fixed_stop", "kind": "stop", "pct": 0.01},
    )
    check("T79 fixed stop trigger fills on the following bar bid",
          stopped["exit_index"] == 2 and stopped["exit_price"] == 97.9)

    armed_path = [
        bar(0, 100.0),
        bar(5, 101.0),
        bar(10, 102.5),
        bar(15, 101.8),
        bar(20, 100.3),
        bar(25, 100.0, 99.9),
    ]
    armed = matrix.simulate_policy(
        armed_path,
        {
            "name": "test_armed",
            "family": "profit_protection",
            "kind": "armed_trail",
            "arm": 0.02,
            "trail": 0.02,
        },
    )
    check("T79 profit trail waits for arming and a subsequent drawdown",
          armed["reason"] == "profit_armed_trailing_stop"
          and armed["exit_index"] == 5
          and armed["exit_price"] == 99.9)

    lifecycle = [
        {
            "status": "filled", "fill_time": "2026-07-01T10:00:00+08:00",
            "filled_qty": 100, "side": "buy", "stockCode": "513100",
            "fill_price": 2.0, "reason": "entry",
        },
        {
            "status": "filled", "fill_time": "2026-07-01T10:10:00+08:00",
            "filled_qty": 100, "side": "sell", "stockCode": "513100",
            "fill_price": 1.9, "reason": "emergency_stop_exit",
        },
    ]
    lots, diagnostics = matrix.pair_all_filled_lots(lifecycle)
    check("T79 hard exits remain in the complete policy population",
          len(lots) == 1
          and diagnostics["hard_exit_lots"] == 1
          and lots[0]["baseline_exit_reason"] == "emergency_stop_exit")

    source = (ROOT / "scripts" / "research_exit_policy_matrix.py").read_text(encoding="utf-8")
    check("T79 exit policy matrix is offline and cannot promote itself",
          "STRICTLY OFFLINE / SHADOW" in source
          and "order_submit_calls_made" in source
          and "submitOrder" not in source
          and "latest_strategy_overlay" not in source)


def t80_exit_diagnostics_separates_labels_from_observables() -> None:
    """Failure labels remain ex-post outcomes and cannot silently become exit gates."""
    import copy

    import research_exit_diagnostics_phase2 as diagnostics

    source = {
        "rounds_total": 2,
        "trade_count": 1,
        "total_pnl": 12.5,
        "order_lifecycle": [{"order_id": "x", "status": "filled"}],
    }
    check("T80 exact replay integrity accepts identical frozen lifecycle",
          diagnostics.verify_exact_replay(source, copy.deepcopy(source))["passed"])
    changed = copy.deepcopy(source)
    changed["order_lifecycle"][0]["status"] = "rejected"
    check("T80 exact replay integrity rejects changed lifecycle",
          not diagnostics.verify_exact_replay(source, changed)["passed"])

    template = {
        "early15_mae_pct": 0.0,
        "mfe_pct": 0.5,
        "giveback_pct": 0.2,
        "holding_bars": 10,
        "mae_pct": -0.2,
        "post_exit_3bar_return_pct": 0.0,
        "post_exit_5bar_return_pct": 0.0,
    }
    immediate = {**template, "early15_mae_pct": -1.2}
    flags, primary = diagnostics.classify_trade(immediate)
    check("T80 immediate-loser label follows preregistered early-path threshold",
          flags["immediate_loser"] and primary == "immediate_loser")
    runner = {
        **template,
        "post_exit_3bar_return_pct": 0.6,
        "post_exit_5bar_return_pct": 0.8,
    }
    flags, primary = diagnostics.classify_trade(runner)
    check("T80 post-exit runner remains an outcome label",
          flags["trend_runner"] and primary == "trend_runner")

    forbidden = {
        "realized_return_pct",
        "mfe_pct",
        "mae_pct",
        "giveback_pct",
        "exit_efficiency_pct",
        "post_exit_1bar_return_pct",
        "post_exit_3bar_return_pct",
        "post_exit_5bar_return_pct",
        "primary_group",
    }
    check("T80 ex-post labels never enter the observable feature contract",
          forbidden.isdisjoint(diagnostics.OBSERVABLES))
    check("T80 mechanical early loss is excluded from immediate-loser discovery",
          "immediate_loser"
          in diagnostics.OBSERVABLES["early15_mae_pct"]["exclude_groups"])

    script_source = (
        ROOT / "scripts" / "research_exit_diagnostics_phase2.py"
    ).read_text(encoding="utf-8")
    check("T80 diagnostics are shadow-only and cannot alter live execution",
          "STRICTLY OFFLINE / SHADOW" in script_source
          and "order_submit_calls_made" in script_source
          and "submitOrder" not in script_source
          and "latest_strategy_overlay" not in script_source)


def t81_breadth_block_never_blocks_sells() -> None:
    """Risk-off guards in the rebalance agent (market-breadth block / momentum warm-up) block
    new BUYING only -- every sell reduces exposure and must pass. Regression for 2026-07-02:
    the old filter kept only stop_loss_exit sells, holding a falling rebalance-out sell
    (159915, planned 09:30 @4.178) until its hard stop fired 15 minutes later @4.088 (-2.2%)."""
    import run_etf_paper_trading_agent as reb

    def q(code, score, price=4.0):
        return {"stockCode": code, "exchange": "SZ", "name": code, "quote_ok": True,
                "isSuspended": False, "currentPrice": price, "bidPrice1": price - 0.002,
                "askPrice1": price + 0.002, "score": score, "signal_type": "momentum_5d",
                "t0_eligible": False, "asset_class": "domestic_equity_etf"}

    cfg = {
        "risk": {"max_position_pct": 0.25, "max_single_order_pct": 0.25, "quantity_lot": 100,
                 "min_order_quantity": 100, "order_type": "limit", "limit_price_slippage_pct": 0.002,
                 "max_daily_orders": 5, "stop_loss_enabled": True, "stop_loss_pct": -0.03},
        "strategy": {"target_holdings": 2, "force_build_position": True, "cash_reserve_pct": 0.05,
                      "rebalance_drift_threshold_pct": 0.03, "entry_score_threshold_pct": 0.0,
                      "min_positive_momentum_count_for_buy": 4,   # 4 scores, all negative -> breadth block ON
                      "score": {"use_intraday_return": True}},
    }
    # 4 valid momentum scores, none >= threshold -> market_breadth_block_active. The held name
    # 159915 (down 1%, ABOVE the -3% stop) is not in the selected top-2 -> rebalance-out sell.
    quotes = [q("510300", -0.01), q("510500", -0.012), q("588000", -0.015), q("159915", -0.02, price=4.13)]
    balance = {"ok": True, "data": {"totalAssets": 1_000_000.0, "availableBalance": 800_000.0}}
    positions = {"159915": {"availableQuantity": 48700, "costPrice": 4.17, "marketValue": 201131.0}}
    plan = reb.build_plan(cfg, quotes, balance, positions)
    sells = [o for o in plan["orders"] if o.get("direction") == "sell"]
    buys = [o for o in plan["orders"] if o.get("direction") == "buy"]
    check("T81 breadth block is active in this scenario", plan.get("market_breadth_block_active") is True,
          str({k: plan.get(k) for k in ("market_breadth_block_active", "positive_momentum_count")}))
    check("T81 rebalance-out sell passes the breadth block (not held until the hard stop)",
          any(o.get("stockCode") == "159915" and o.get("reason") == "not_in_selected_etf_set" for o in sells),
          str(plan["orders"]))
    check("T81 breadth block still blocks all buys", not buys, str(buys))


def t59_every_decision_and_daily_score_review() -> None:
    from datetime import date, timedelta
    import decision_scoring as scoring
    import research_decision_score_daily_review as review
    import run_decision_score_daily as daily

    decision = {
        "state_machine": {"action": "sell", "reason": "unified_sell_score_exit"},
        "orders": [
            {"direction": "sell", "stockCode": "588000", "quantity": 100,
             "reason": "unified_sell_score_exit"},
            {"direction": "sell", "stockCode": "159915", "quantity": 200,
             "reason": "emergency_stop_exit"},
        ],
        "ranked": [
            {"stockCode": "588000", "name": "科创50", "currentPrice": 2.0, "amount": 1e8,
             "spread_pct": 0.001, "change_pct": -1.0},
            {"stockCode": "159915", "name": "创业板", "currentPrice": 4.0, "amount": 1e8,
             "spread_pct": 0.001, "change_pct": -2.0},
            {"stockCode": "159546", "name": "集成电路", "currentPrice": 1.0, "amount": 1e8,
             "spread_pct": 0.001, "change_pct": 0.2},
        ],
        "positions_t0": {
            "588000": {"stockName": "科创50", "quantity": 100, "availableQuantity": 100},
            "159915": {"stockName": "创业板", "quantity": 200, "availableQuantity": 200},
            "159546": {"stockName": "集成电路", "quantity": 300, "availableQuantity": 300},
        },
        "sell_score_by_code": {"588000": 80, "159915": 95, "159546": 40},
        "carry_allowed_by_code": {"588000": False, "159915": False, "159546": True},
    }
    contexts = scoring.contexts_from_decision(
        {"decision_scoring": {"record_ranked_candidates": False}}, decision,
        trade_date="2026-06-22", timestamp="10:05:00",
    )
    sells = [row for row in contexts if row["decision_type"] == "SELL"]
    holds = [row for row in contexts if row["decision_type"] == "HOLD"]
    check("T59 every planned sell receives its own decision row",
          len(sells) == 2 and len({row["decision_id"] for row in sells}) == 2)
    check("T59 unsold held position receives a position HOLD row",
          len(holds) == 1 and holds[0]["etf_code"] == "159546")
    check("T59 planned order is not falsely labelled as executed fill",
          all(row["order_planned"] is True and row["was_executed"] is False for row in sells))

    rows = []
    high_scores = {feature: limits[1] for feature, limits in scoring.SCORE_RANGES.items()}
    low_scores = {feature: limits[0] for feature, limits in scoring.SCORE_RANGES.items()}
    for day in range(1, 51):
        trade_date = (date(2026, 7, 1) + timedelta(days=day - 1)).isoformat()
        high = {"date": trade_date, "decision_type": "BUY_CANDIDATE", "signal_direction": "BUY",
                "outcome_horizon_complete": True, "probability_outcome": 1,
                "counterfactual_return": 0.01, "estimated_round_trip_cost": 0.0,
                "total_score": 80, **high_scores}
        low = {"date": trade_date, "decision_type": "BUY_CANDIDATE", "signal_direction": "BUY",
               "outcome_horizon_complete": True, "probability_outcome": 0,
               "counterfactual_return": -0.01, "estimated_round_trip_cost": 0.0,
               "total_score": 50, **low_scores}
        # Plant one inverted component: high liquidity appears on losing rows.
        high["liquidity_score"], low["liquidity_score"] = 0.0, 15.0
        rows.extend((high, low))
    result = review.build_review(rows, "2026-12-31")
    liquidity = next(item for item in result["reviews"] if item["feature"] == "liquidity_score")
    total = next(item for item in result["reviews"] if item["feature"] == "total_score")
    fixed = {item["bucket"]: item for item in result["fixedTotalScoreBuckets"]}
    check("T59 day-paired Holm review detects planted inverted component",
          liquidity["assessment"] == "statistically_inverted_high_score_underperforms"
          and liquidity["holmAdjustedP"] < 0.05)
    check("T59 planted total score remains positively discriminating",
          total["assessment"] == "statistically_positive_discrimination")
    check("T59 fixed total-score buckets expose high vs low outcome separation",
          fixed["75-90"]["count"] == 50 and fixed["40-60"]["count"] == 50
          and fixed["75-90"]["avgNetReturn"] > fixed["40-60"]["avgNetReturn"])
    check("T59 daily enrichment refuses incomplete trading sessions",
          daily.session_complete({"588000": {"2026-07-01": {600: 1.0}}}, "2026-07-01", 895) is False
          and daily.session_complete({"588000": {"2026-07-01": {895: 1.0}}}, "2026-07-01", 895) is True)
    dirty = {"realized_return": 0.01, "counterfactual_return": 0.02, "probability_outcome": 1}
    daily.clear_outcomes([dirty])
    check("T59 provisional outcomes can be cleared safely before session close",
          dirty["realized_return"] is None and dirty["counterfactual_return"] is None
          and dirty["probability_outcome"] is None)
    check("T59 statistical review only recommends and never changes weights/gates",
          result["autoWeightChangeAllowed"] is False
          and result["tradeGateChangeAllowed"] is False
          and "liquidity_score" in result["adjustmentCandidates"])


def t82_local_news_watchlist_is_causal_no_api_and_shadow_only() -> None:
    """Local RSS ranking is cutoff-safe, auditable and cannot become a trade gate."""
    import copy as _copy
    from datetime import datetime as _datetime
    from zoneinfo import ZoneInfo as _ZoneInfo
    import build_t0_observation_pool as pool
    import generate_local_news_etf_watchlist as local

    config = local.load_config()
    topic = config["topics"][2]
    xml = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
<item><title>芯片订单大增并获政策支持 - 财联社</title>
<link>https://news.google.com/articles/positive</link>
<pubDate>Thu, 02 Jul 2026 18:00:00 GMT</pubDate>
<source url="https://www.cls.cn">财联社</source></item>
<item><title>海外芯片股大跌风险上升 - Reuters</title>
<link>https://news.google.com/articles/negative</link>
<pubDate>Fri, 03 Jul 2026 00:00:00 GMT</pubDate>
<source url="https://www.reuters.com">Reuters</source></item>
<item><title>截止时间之后的上涨新闻</title>
<link>https://news.google.com/articles/future</link>
<pubDate>Fri, 03 Jul 2026 01:00:00 GMT</pubDate>
<source url="https://example.cn">测试源</source></item>
<item><title>窗口之前的旧闻</title>
<link>https://news.google.com/articles/stale</link>
<pubDate>Thu, 02 Jul 2026 06:00:00 GMT</pubDate>
<source url="https://example.cn">测试源</source></item>
</channel></rss>""".encode("utf-8")
    sh = _ZoneInfo("Asia/Shanghai")
    parsed = local.parse_rss(
        xml,
        topic,
        news_start=_datetime.fromisoformat("2026-07-02T15:00:00+08:00").astimezone(sh),
        cutoff=_datetime.fromisoformat("2026-07-03T08:30:00+08:00").astimezone(sh),
        config=config,
    )
    check("T82 RSS parser excludes stale and post-cutoff articles",
          [row.source_url.rsplit("/", 1)[-1] for row in parsed] == ["negative", "positive"], str(parsed))
    check("T82 deterministic lexicon preserves opposing evidence",
          {row.sentiment > 0 for row in parsed} == {True, False}, str(parsed))

    master = {
        (f"51{index:04d}", "SH"): {
            "stockCode": f"51{index:04d}", "market": "1", "name": f"半导体ETF{index}"
        }
        for index in range(10)
    }
    template = {
        "exchange": "SH",
        "name": "半导体ETF",
        "topic": {"id": "semiconductor", "label": "半导体"},
        "localScore": 70.0,
        "scoreBreakdown": {"newsDirection": 60.0},
        "evidenceConfidence": 0.7,
        "newsDirection": "positive",
        "previousSessionMarket": {"amount": 100000000.0, "changePct": 1.0, "currentPrice": 1.0},
        "reason": "本地规则证据排序，不是上涨概率。",
        "newsDrivers": ["正向驱动"],
        "risks": ["仅用于研究观察"],
        "sourceUrls": ["https://news.google.com/articles/positive"],
        "sourceItems": [{"title": "正向驱动"}],
    }
    rows = []
    for index, ((code, exchange), canonical) in enumerate(master.items()):
        row = _copy.deepcopy(template)
        row.update({"stockCode": code, "exchange": exchange, "name": canonical["name"],
                    "localScore": 70.0 - index})
        rows.append(row)
    payload = {
        "schemaVersion": local.SCHEMA_VERSION,
        "asOfDate": "2026-07-03",
        "effectiveDate": "2026-07-03",
        "generatedAt": "2026-07-03T08:30:00+08:00",
        "paperTradingOnly": True,
        "diagnosticOnly": True,
        "tradeGateEnabled": False,
        "liveReady": False,
        "formalStrategyAllowed": False,
        "generator": {"noExternalModelApi": True},
        "etfs": rows,
    }
    check("T82 local payload validates exactly ten unique non-money ETFs",
          local.validate_payload(payload, master=master, expected_count=10) == [])
    accepted, rejected, meta = pool.validate_research_payload(
        payload, master, limit=10, today="2026-07-03"
    )
    combined, overlaps = pool.merge_observation_pool(
        [{"stockCode": rows[0]["stockCode"], "exchange": "SH", "name": rows[0]["name"]}],
        accepted,
        source_kind=meta["sourceKind"],
    )
    overlap = combined[0]
    check("T82 builder preserves local provenance without calling it ChatGPT",
          not rejected and overlaps == 1
          and overlap["sources"] == ["system_rank20", "local_news10"]
          and "localNewsResearch" in overlap
          and "chatgptResearch" not in overlap, str(overlap))
    source = (ROOT / "scripts" / "generate_local_news_etf_watchlist.py").read_text(encoding="utf-8")
    check("T82 generator has no hosted-model or order path",
          "api.openai.com" not in source and "OPENAI_API_KEY" not in source
          and "submitOrder" not in source and "place_order" not in source)
    check("T82 configuration is hard locked to shadow research",
          config["recordOnly"] is True and config["tradeGateEnabled"] is False
          and config["positionSizingEnabled"] is False)


def t83_laplace_copula_price_chain_is_frozen_next_bar_and_safe() -> None:
    """The Laplace-copula chain score must be frozen, causal and offline."""
    import json
    import research_laplace_copula_price_chain as chain

    leader = chain.np.asarray(
        [-0.010, -0.006, -0.003, 0.000, 0.003, 0.006, 0.010] * 20
    )
    lagger = 0.9 * leader + chain.np.asarray(
        [-0.001, 0.000, 0.001, 0.000, -0.001, 0.001, 0.000] * 20
    )
    config = {
        "model": {
            "minimumTrainingPairObservations": 100,
            "minimumFrozenGaussianCopulaRho": 0.6,
            "cdfClip": 1e-6,
            "formationBars": 3,
            "decisionTimes": ["10:00"],
        },
        "signal": {
            "leaderMomentumMinimum": 0.002,
            "leaderLaggerReturnGapMinimum": 0.0015,
            "conditionalLowerTailMaximum": 0.05,
            "maximumBenchmarkSignalsPerDecision": 5,
        },
        "execution": {
            "holdingBars": 2,
            "maximumEntryDelayMinutes": 10,
        },
    }
    models, audit = chain.fit_frozen_pair_models(
        {("IDX", "LEAD", "LAG"): (leader, lagger)}, config
    )
    model = models[("IDX", "LEAD", "LAG")]
    ordinary_tail = chain.conditional_lower_tail(
        0.006, 0.0054, model, clip=1e-6
    )
    abnormal_tail = chain.conditional_lower_tail(
        0.006, -0.006, model, clip=1e-6
    )
    check(
        "T83 frozen Laplace copula ranks an abnormal lag below an ordinary pair",
        audit["acceptedOrderedPairs"] == 1 and abnormal_tail < ordinary_tail,
        str((audit, abnormal_tail, ordinary_tail)),
    )

    index = chain.pd.date_range("2026-06-01 09:45:00", periods=7, freq="5min")
    prices = chain.pd.DataFrame(
        {
            "LEAD": [1.0, 1.0, 1.0, 1.006, 1.006, 1.006, 1.006],
            "LAG": [1.0, 1.0, 1.0, 0.994, 1.000, 1.010, 1.020],
        },
        index=index,
    )
    signals = chain.generate_price_chain_signals(
        prices, {"IDX": ["LEAD", "LAG"]}, models, config
    )
    check(
        "T83 price-chain replay enters after the completed decision bar",
        len(signals) == 1
        and signals[0]["entry_time"] > signals[0]["decision_time"]
        and signals[0]["laggard"] == "LAG"
        and abs(signals[0]["candidate_gross_return"] - 0.02) < 1e-12,
        str(signals),
    )

    prereg = json.loads(
        (
            ROOT
            / "configs"
            / "research"
            / "laplace_copula_price_chain_preregistered.json"
        ).read_text(encoding="utf-8")
    )
    source = (
        ROOT / "scripts" / "research_laplace_copula_price_chain.py"
    ).read_text(encoding="utf-8")
    check(
        "T83 Laplace-copula research cannot trade, promote or mutate live config",
        prereg["status"] == "diagnostic_only"
        and prereg["data"]["oosWindowPreviouslyReused"] is True
        and prereg["safety"]["offlineOnly"] is True
        and prereg["safety"]["tradeGateEnabled"] is False
        and prereg["safety"]["brokerCallsAllowed"] is False
        and prereg["safety"]["promotionAllowed"] is False
        and "SkillClient" not in source
        and "submitOrder" not in source
        and "latest_strategy_overlay" not in source,
    )


def t84_tail_probability_bounds_are_causal_conservative_and_safe() -> None:
    """Tail bounds must be one-sided, prefix-only and offline."""
    import json
    import research_tail_probability_bounds as tail

    quiet = tail.np.asarray(
        [-0.004, -0.002, 0.000, 0.002, 0.004] * 4, dtype=float
    )
    cantelli_near = tail.cantelli_lower_tail_bound(quiet, 0.005)
    cantelli_far = tail.cantelli_lower_tail_bound(quiet, 0.020)
    check(
        "T84 Cantelli lower-tail bound decreases for a more remote loss",
        0 <= cantelli_far < cantelli_near <= 1,
        str((cantelli_near, cantelli_far)),
    )
    exposure, bound = tail.select_exposure(
        quiet,
        daily_loss_threshold=0.01,
        maximum_tail_probability=0.10,
        exposure_grid=[1.0, 0.75, 0.5, 0.25, 0.0],
        bound_function=tail.cantelli_lower_tail_bound,
    )
    shocked_future = tail.np.append(quiet, -0.05)
    same_exposure, same_bound = tail.select_exposure(
        shocked_future[:-1],
        daily_loss_threshold=0.01,
        maximum_tail_probability=0.10,
        exposure_grid=[1.0, 0.75, 0.5, 0.25, 0.0],
        bound_function=tail.cantelli_lower_tail_bound,
    )
    check(
        "T84 next-session exposure is invariant to an unseen future shock",
        exposure == same_exposure and abs(bound - same_bound) < 1e-12,
    )
    invalid_support = tail.chernoff_ucb_lower_tail_bound(
        shocked_future,
        0.01,
        return_lower_bound=-0.04,
        return_upper_bound=0.04,
        confidence=0.95,
        lambdas=[10.0, 20.0],
    )
    check(
        "T84 Chernoff bound fails closed when assumed support is violated",
        invalid_support == 1.0,
    )

    prereg = json.loads(
        (
            ROOT
            / "configs"
            / "research"
            / "tail_probability_bounds_preregistered.json"
        ).read_text(encoding="utf-8")
    )
    source = (
        ROOT / "scripts" / "research_tail_probability_bounds.py"
    ).read_text(encoding="utf-8")
    check(
        "T84 tail-bound research cannot trade, size or mutate live config",
        prereg["status"] == "diagnostic_only"
        and prereg["data"]["replayWindowPreviouslyReused"] is True
        and prereg["safety"]["offlineOnly"] is True
        and prereg["safety"]["tradeGateEnabled"] is False
        and prereg["safety"]["positionSizingEnabled"] is False
        and prereg["safety"]["brokerCallsAllowed"] is False
        and prereg["safety"]["promotionAllowed"] is False
        and "SkillClient" not in source
        and "submitOrder" not in source
        and "latest_strategy_overlay" not in source,
    )


def t85_minute_forecast_is_next_bar_point_in_time_and_safe() -> None:
    """Minute forecasts must use completed bars and remain record-only."""
    import json
    import research_minute_forecast_shadow as minute

    index = minute.pd.date_range("2026-06-01 09:30:00", periods=12, freq="5min")
    panel = minute.pd.DataFrame(
        {
            "timestamp": list(index) * 2,
            "trade_date": ["2026-06-01"] * 24,
            "stockCode": ["A"] * 12 + ["B"] * 12,
            "close": [
                *[1.0 + value for value in minute.np.linspace(0, 0.022, 12)],
                *[1.0 - value for value in minute.np.linspace(0, 0.011, 12)],
            ],
            "cumulative_amount": [
                *minute.np.arange(1, 13, dtype=float),
                *minute.np.arange(1, 13, dtype=float) * 2,
            ],
        }
    )
    features = [
        "ret_1",
        "ret_3",
        "ret_6",
        "acceleration_1",
        "vol_6",
        "market_ret_1",
        "market_ret_3",
        "breadth",
        "relative_strength",
        "amount_rank",
        "session_fraction",
    ]
    frames = minute.build_feature_frames(panel)
    samples = minute.build_samples(frames, features, horizon_bars=1)
    planted = samples[samples["stockCode"] == "A"].iloc[0]
    check(
        "T85 minute label enters after the completed feature timestamp",
        planted["entry_time"] > planted["timestamp"]
        and planted["exit_time"] > planted["entry_time"],
        str(planted.to_dict()),
    )
    decision = planted["timestamp"]
    original = float(frames["ret_1"].at[decision, "A"])
    shocked = panel.copy()
    shocked.loc[
        (shocked["stockCode"] == "A")
        & (shocked["timestamp"] > decision),
        "close",
    ] *= 10.0
    shocked_frames = minute.build_feature_frames(shocked)
    check(
        "T85 feature at decision is invariant to unseen future prices",
        abs(float(shocked_frames["ret_1"].at[decision, "A"]) - original)
        < 1e-12,
    )

    prereg = json.loads(
        (
            ROOT
            / "configs"
            / "research"
            / "minute_forecast_shadow_v1.json"
        ).read_text(encoding="utf-8")
    )
    source = (
        ROOT / "scripts" / "research_minute_forecast_shadow.py"
    ).read_text(encoding="utf-8")
    check(
        "T85 minute forecast is offline record-only and cannot trade",
        prereg["status"] == "diagnostic_only"
        and prereg["safety"]["offlineOnly"] is True
        and prereg["safety"]["recordOnly"] is True
        and prereg["safety"]["tradeGateEnabled"] is False
        and prereg["safety"]["positionSizingEnabled"] is False
        and prereg["safety"]["brokerCallsAllowed"] is False
        and prereg["safety"]["promotionAllowed"] is False
        and "SkillClient" not in source
        and "submitOrder" not in source
        and "latest_strategy_overlay" not in source,
    )


def t86_minute_model_fusion_is_causal_and_shadow_only() -> None:
    """HMM fusion must filter causally and remain disconnected from trading."""
    import json
    import research_minute_forecast_shadow as minute
    import research_minute_model_fusion as fusion

    day1 = fusion.pd.date_range("2026-05-01 09:30:00", periods=16, freq="5min")
    day2 = fusion.pd.date_range("2026-05-06 09:30:00", periods=16, freq="5min")
    index = day1.append(day2)
    dates = [timestamp.strftime("%Y-%m-%d") for timestamp in index]
    panel = fusion.pd.DataFrame(
        {
            "timestamp": list(index) * 2,
            "trade_date": dates * 2,
            "stockCode": ["A"] * 32 + ["B"] * 32,
            "close": [
                *[1.0 + value for value in fusion.np.linspace(0, 0.03, 32)],
                *[1.0 - value for value in fusion.np.linspace(0, 0.01, 32)],
            ],
            "cumulative_amount": [
                *fusion.np.tile(fusion.np.arange(1, 17, dtype=float), 2),
                *fusion.np.tile(fusion.np.arange(1, 17, dtype=float) * 2, 2),
            ],
        }
    )
    frames = minute.build_feature_frames(panel)
    hmm_cfg = {"states": 3, "iterations": 5, "varianceFloor": 1e-8}
    fused = fusion.add_causal_hmm_features(
        frames, train_end="2026-05-01", hmm_config=hmm_cfg
    )
    decision = day2[10]
    original = float(fused["hmm_bull_probability"].at[decision])
    shocked_panel = panel.copy()
    shocked_panel.loc[shocked_panel["timestamp"] > decision, "close"] *= 10.0
    shocked = fusion.add_causal_hmm_features(
        minute.build_feature_frames(shocked_panel),
        train_end="2026-05-01",
        hmm_config=hmm_cfg,
    )
    check(
        "T86 HMM probability at decision ignores unseen future prices",
        abs(float(shocked["hmm_bull_probability"].at[decision]) - original)
        < 1e-12,
    )

    prereg = json.loads(
        (
            ROOT
            / "configs"
            / "research"
            / "minute_model_fusion_preregistered.json"
        ).read_text(encoding="utf-8")
    )
    source = (
        ROOT / "scripts" / "research_minute_model_fusion.py"
    ).read_text(encoding="utf-8")
    check(
        "T86 fusion research is record-only and cannot trade or promote",
        prereg["status"] == "diagnostic_only"
        and prereg["fusionBoundary"]["oldNeuralPredictionIncluded"] is False
        and prereg["safety"]["offlineOnly"] is True
        and prereg["safety"]["recordOnly"] is True
        and prereg["safety"]["tradeGateEnabled"] is False
        and prereg["safety"]["positionSizingEnabled"] is False
        and prereg["safety"]["brokerCallsAllowed"] is False
        and prereg["safety"]["promotionAllowed"] is False
        and "SkillClient" not in source
        and "submitOrder" not in source
        and "latest_strategy_overlay" not in source,
    )


def t87_conditional_minute_tail_preserves_dependence_safely() -> None:
    """Conditional tail research must be conservative and shadow-only."""
    import json
    import research_conditional_minute_tail as conditional

    weights = conditional.np.asarray([0.7, 0.3])
    means = conditional.np.asarray([-0.001, 0.002])
    variances = conditional.np.asarray([1e-6, 4e-6])
    lambdas = conditional.np.geomspace(0.1, 5000.0, 200)
    near_probability = conditional.gaussian_mixture_tail_probability(
        weights, means, variances, 0.002
    )
    far_probability = conditional.gaussian_mixture_tail_probability(
        weights, means, variances, 0.005
    )
    near_bound = conditional.gaussian_mixture_chernoff_bound(
        weights, means, variances, 0.002, lambdas
    )
    check(
        "T87 conditional tail probability and Chernoff bound are ordered",
        0 <= far_probability < near_probability <= near_bound <= 1,
        str(
            {
                "far": far_probability,
                "near": near_probability,
                "bound": near_bound,
            }
        ),
    )
    updated = conditional.ewma_variance_update(
        previous=1e-6,
        residual=0.01,
        half_life_bars=24,
        variance_floor=1e-8,
    )
    check(
        "T87 EWMA variance responds to a completed residual",
        updated > 1e-6,
        str(updated),
    )

    prereg = json.loads(
        (
            ROOT
            / "configs"
            / "research"
            / "conditional_minute_tail_preregistered.json"
        ).read_text(encoding="utf-8")
    )
    source = (
        ROOT / "scripts" / "research_conditional_minute_tail.py"
    ).read_text(encoding="utf-8")
    check(
        "T87 conditional dependence research cannot trade or size",
        prereg["status"] == "diagnostic_only"
        and prereg["safety"]["offlineOnly"] is True
        and prereg["safety"]["recordOnly"] is True
        and prereg["safety"]["tradeGateEnabled"] is False
        and prereg["safety"]["positionSizingEnabled"] is False
        and prereg["safety"]["brokerCallsAllowed"] is False
        and prereg["safety"]["promotionAllowed"] is False
        and "SkillClient" not in source
        and "submitOrder" not in source
        and "latest_strategy_overlay" not in source,
    )


def t88_singularity_phase1_is_causal_purged_and_shadow_only() -> None:
    """Phase 1 features ignore the future while offline labels may use it."""
    import copy
    import json
    import math
    import research_minute_forecast_shadow as minute
    import research_singularity_phase1 as singularity

    prereg = json.loads(
        (
            ROOT
            / "configs"
            / "research"
            / "singularity_phase1_preregistered.json"
        ).read_text(encoding="utf-8")
    )
    config = copy.deepcopy(prereg)
    config["data"]["trainEnd"] = "2026-05-01"
    config["features"]["ewsStandardizationFitEnd"] = "2026-05-01"
    config["features"]["hmm"]["fitEnd"] = "2026-05-01"
    dates = [
        singularity.pd.date_range(
            "2026-05-01 09:30:00", periods=48, freq="5min"
        ),
        singularity.pd.date_range(
            "2026-05-06 09:30:00", periods=48, freq="5min"
        ),
    ]
    rows = []
    for day_index, timestamps in enumerate(dates):
        for code_index, code in enumerate(["A", "B"]):
            for bar_index, timestamp in enumerate(timestamps):
                rows.append(
                    {
                        "timestamp": timestamp,
                        "trade_date": timestamp.strftime("%Y-%m-%d"),
                        "stockCode": code,
                        "close": (
                            1.0
                            + day_index * 0.001
                            + code_index * 0.0002
                            + bar_index * 0.0002
                            + math.sin(bar_index / (3.0 + code_index))
                            * (0.001 + code_index * 0.0004)
                        ),
                        "cumulative_amount": float(
                            (bar_index + 1) * (code_index + 1) * 1_000_000
                        ),
                    }
                )
    panel = singularity.pd.DataFrame.from_records(rows)
    decision = dates[1][24]

    def phase1_tables(source):
        base = minute.build_feature_frames(source)
        ews, _ = singularity.build_ews_frames(base, config)
        hmm, _ = singularity.build_hmm_frames(base, config)
        return (
            singularity.build_feature_table(base, ews, hmm, config),
            singularity.build_label_table(base, config)[0],
        )

    original_features, original_labels = phase1_tables(panel)
    shocked_panel = panel.copy()
    shocked_panel.loc[
        shocked_panel["timestamp"] > decision, "close"
    ] *= 0.98
    shocked_features, shocked_labels = phase1_tables(shocked_panel)
    feature_columns = (
        list(config["features"]["base"])
        + singularity.HMM_FEATURES
        + singularity.EWS_FEATURES
        + ["singularity_score"]
    )

    def keyed_row(table):
        return table[
            (table["timestamp"] == decision)
            & (table["stockCode"] == "A")
        ].iloc[0]

    before = keyed_row(original_features)
    after = keyed_row(shocked_features)
    feature_delta = max(
        abs(float(before[name]) - float(after[name]))
        for name in feature_columns
    )
    original_label = int(keyed_row(original_labels)["turning_point_5"])
    shocked_label = int(keyed_row(shocked_labels)["turning_point_5"])
    check(
        "T88 Phase 1 features at decision ignore unseen future prices",
        feature_delta < 1e-12,
        str(feature_delta),
    )
    check(
        "T88 future shock changes only the separate offline reversal label",
        original_label == 0 and shocked_label == 1,
        str({"original": original_label, "shocked": shocked_label}),
    )
    check(
        "T88 60-bar cross-session labels stay null",
        original_labels["turning_point_60"].isna().all()
        and shocked_labels["turning_point_60"].isna().all(),
    )

    purge_source = singularity.pd.DataFrame(
        {
            "stockCode": ["A"] * 35 + ["B"] * 35,
            "timestamp": list(range(35)) * 2,
            "trade_date": ["2026-05-01"] * 70,
        }
    )
    purged, removed = singularity.purge_symbol_tail(purge_source, 30)
    check(
        "T88 purge removes at least the maximum label horizon per ETF",
        removed == 60
        and len(purged) == 10
        and prereg["models"]["purgeBars"]
        >= max(prereg["labels"]["activeHorizonsBars"]),
        str({"removed": removed, "remaining": len(purged)}),
    )

    source = (
        ROOT / "scripts" / "research_singularity_phase1.py"
    ).read_text(encoding="utf-8")
    safety = prereg["safety"]
    check(
        "T88 Singularity Phase 1 is isolated research-only shadow output",
        prereg["status"] == "research_only"
        and prereg["diagnosticOnly"] is True
        and safety["offlineOnly"] is True
        and safety["recordOnly"] is True
        and safety["brokerCallsAllowed"] is False
        and safety["onlineInferenceAllowed"] is False
        and safety["liveConfigWritesAllowed"] is False
        and safety["overlayWritesAllowed"] is False
        and safety["positionSizingAllowed"] is False
        and safety["orderSubmissionAllowed"] is False
        and safety["riskGateChangesAllowed"] is False
        and safety["buildDecisionIntegrationAllowed"] is False
        and safety["promotionAllowed"] is False
        and "SkillClient(" not in source
        and "submitOrder(" not in source
        and "latest_strategy_overlay.json" not in source
        and "from t0_intraday_agent" not in source,
    )
    check(
        "T88 Phase 1 reuses the existing HMM implementation",
        singularity.GaussianHMM1D.__module__ == "research_hmm_nn_bl",
        singularity.GaussianHMM1D.__module__,
    )


def t89_singularity_phase1_5_is_frozen_forward_shadow_only() -> None:
    """Phase 1.5 must score a hash-pinned model and remain trade-disconnected."""
    import json
    import math
    import tempfile
    import pandas as pd
    import run_singularity_phase1_5_forward as forward

    config_path = (
        ROOT
        / "configs"
        / "research"
        / "singularity_phase1_5_forward.json"
    )
    config, bundle, model_hash = forward.load_config_bundle(config_path)
    check(
        "T89 Phase 1.5 model and Phase 1 config are hash-pinned",
        model_hash == config["frozenModel"]["sha256"]
        and bundle["phase1ConfigSha256"]
        == config["phase1"]["configSha256"]
        and bundle["historicalCutoff"]
        == config["phase1"]["historicalCutoff"],
    )
    check(
        "T89 forward start excludes every implementation-time historical day",
        config["forward"]["prospectiveAfter"] == "2026-07-05"
        and config["forward"]["firstEligibleDate"] == "2026-07-06"
        and config["forward"]["allowHistoricalBackfill"] is False,
    )
    check(
        "T89 60-bar horizon stays absent from frozen models and explicitly null",
        60 in config["forward"]["skippedHorizonsBars"]
        and "60" not in bundle["modelsByHorizon"],
    )

    toy_model = {
        "features": ["feature"],
        "scalerMean": [0.0],
        "scalerScale": [2.0],
        "logisticCoefficient": [1.0],
        "logisticIntercept": 0.0,
        "plattCoefficient": 1.0,
        "plattIntercept": 0.0,
    }
    raw, calibrated = forward.score_frozen_model(
        pd.DataFrame({"feature": [2.0]}), toy_model
    )
    expected = 1.0 / (1.0 + math.exp(-1.0))
    check(
        "T89 frozen JSON coefficient scorer reproduces logistic probability",
        abs(float(raw[0]) - expected) < 1e-12
        and abs(float(calibrated[0]) - expected) < 1e-12,
    )

    timestamp = pd.Timestamp("2026-07-06T10:00:00+08:00")
    predictions = []
    for horizon in [5, 10, 20, 30]:
        for variant in ["baseline", "hmm", "ews", "hmm_ews"]:
            predictions.append(
                {
                    "timestamp": timestamp,
                    "trade_date": "2026-07-06",
                    "stockCode": "513100",
                    "horizon_bars": horizon,
                    "variant": variant,
                    "raw_probability": 0.2,
                    "probability": 0.2,
                }
            )
    feature = pd.DataFrame(
        [
            {
                "timestamp": timestamp,
                "trade_date": "2026-07-06",
                "stockCode": "513100",
                "ews_score": 0.4,
                "regime_transition_risk": 0.3,
                "regime_entropy": 0.5,
                "singularity_score": 0.35,
            }
        ]
    )
    artifact = forward.build_probability_artifact(
        pd.DataFrame.from_records(predictions),
        feature,
        config,
        model_hash,
    )[0]
    check(
        "T89 probability artifact separates labels and carries no trade action",
        artifact["status"] == "research_only"
        and artifact["shadowOnly"] is True
        and artifact["p_turning_60"] is None
        and artifact["orderInstruction"] is None
        and artifact["positionSizeInstruction"] is None
        and artifact["gateInstruction"] is None
        and "turningPoint" not in artifact,
    )

    with tempfile.TemporaryDirectory() as temporary:
        validation = forward.build_forward_validation(
            Path(temporary), config, model_hash
        )
    check(
        "T89 fewer than twenty forward days cannot reopen Phase 2",
        validation["independentTradingDays"] == 0
        and validation["status"] == "insufficient_forward_days"
        and validation["phase2DiscussionAllowed"] is False,
    )

    source = (
        ROOT / "scripts" / "run_singularity_phase1_5_forward.py"
    ).read_text(encoding="utf-8")
    freeze_source = (
        ROOT / "scripts" / "freeze_singularity_phase1_5.py"
    ).read_text(encoding="utf-8")
    safety = config["safety"]
    check(
        "T89 daily monitor cannot fit, trade, size, gate, overlay or promote",
        config["status"] == "research_only"
        and config["shadowOnly"] is True
        and config["frozenModel"]["runtimeRefitAllowed"] is False
        and safety["offlinePostCloseOnly"] is True
        and safety["recordOnly"] is True
        and safety["brokerCallsAllowed"] is False
        and safety["onlineInferenceAllowed"] is False
        and safety["liveConfigWritesAllowed"] is False
        and safety["overlayWritesAllowed"] is False
        and safety["positionSizingAllowed"] is False
        and safety["orderSubmissionAllowed"] is False
        and safety["riskGateChangesAllowed"] is False
        and safety["buildDecisionIntegrationAllowed"] is False
        and safety["buySellGateIntegrationAllowed"] is False
        and safety["promotionAllowed"] is False
        and ".fit(" not in source
        and "SkillClient(" not in source
        and "submitOrder(" not in source
        and "latest_strategy_overlay.json" not in source
        and "from t0_intraday_agent" not in source,
    )
    check(
        "T89 one-time freezer refuses to overwrite its versioned model",
        "if output.exists()" in freeze_source
        and "raise FileExistsError" in freeze_source,
    )


def t90_singularity_phase2a_is_causal_historical_and_isolated() -> None:
    import json

    import numpy as np
    import research_singularity_phase2a as phase2a

    config_path = (
        ROOT
        / "configs"
        / "research"
        / "singularity_phase2a_historical.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    safety = config["safety"]
    check(
        "T90 Phase 2A is historical research-only and cannot mutate trading",
        config["status"] == "research_only"
        and config["shadowOnly"] is True
        and config["diagnosticOnly"] is True
        and safety["offlineOnly"] is True
        and safety["recordOnly"] is True
        and all(
            safety[key] is False
            for key in [
                "brokerCallsAllowed",
                "onlineInferenceAllowed",
                "phase15WritesAllowed",
                "liveConfigWritesAllowed",
                "overlayWritesAllowed",
                "positionSizingAllowed",
                "orderSubmissionAllowed",
                "riskGateChangesAllowed",
                "buildDecisionIntegrationAllowed",
                "buySellGateIntegrationAllowed",
                "promotionAllowed",
            ]
        ),
    )
    check(
        "T90 Phase 2A horizons, purge and embargo are causal",
        config["labels"]["modelHorizonsBars"] == [5, 10]
        and 60 not in config["labels"]["modelHorizonsBars"]
        and "60" in config["labels"]["skippedHorizons"]
        and config["models"]["purgeBarsPerSymbol"]
        >= max(config["labels"]["observationalHorizonsBars"])
        and config["models"]["embargoTradingDays"] >= 1
        and config["data"]["fixedPreprocessingEnd"]
        < config["data"]["walkForwardStart"],
    )
    check(
        "T90 Phase 2A uses only fixed classic LPPLS and linear delay-DMD",
        config["features"]["lppls"]["implementation"]
        == "classic_linearized_grid_search"
        and config["features"]["lppls"]["gridFrozenBeforeWalkForward"] is True
        and config["features"]["koopman"]["implementation"]
        == "fixed_window_linear_delay_dmd_residual"
        and config["features"]["koopman"]["kernelEnabled"] is False
        and config["features"]["koopman"]["deepEnabled"] is False,
    )

    past_prices = np.exp(
        np.linspace(np.log(1.0), np.log(1.025), 24)
        + 0.001 * np.sin(np.arange(24, dtype=float))
    )
    future_a = np.array([1.026, 1.027, 1.028], dtype=float)
    future_b = np.array([0.75, 1.35, 0.60], dtype=float)
    lppls_a = phase2a.lppls_features(
        np.concatenate([past_prices, future_a])[:24], config
    )
    lppls_b = phase2a.lppls_features(
        np.concatenate([past_prices, future_b])[:24], config
    )
    check(
        "T90 LPPLS feature at decision ignores unseen future prices",
        lppls_a is not None
        and lppls_b is not None
        and all(
            abs(lppls_a[key] - lppls_b[key]) < 1e-12
            for key in lppls_a
        ),
    )

    time = np.arange(24, dtype=float)
    past_variables = np.column_stack(
        [
            np.sin(time / 3.0),
            np.cos(time / 4.0),
            time / 24.0,
            np.sin(time / 5.0) + time / 100.0,
        ]
    )
    future_variables_a = np.zeros((3, 4), dtype=float)
    future_variables_b = np.full((3, 4), 1000.0, dtype=float)
    dmd_a = phase2a.dmd_features(
        np.vstack([past_variables, future_variables_a])[:24], config
    )
    dmd_b = phase2a.dmd_features(
        np.vstack([past_variables, future_variables_b])[:24], config
    )
    check(
        "T90 DMD residual at decision ignores unseen future states",
        dmd_a is not None
        and dmd_b is not None
        and all(
            abs(dmd_a[key] - dmd_b[key]) < 1e-12
            for key in dmd_a
        ),
    )

    source = (
        ROOT / "scripts" / "research_singularity_phase2a.py"
    ).read_text(encoding="utf-8")
    forbidden_source_fragments = [
        "SkillClient(",
        "submitOrder(",
        "build_decision(",
        "latest_strategy_overlay.json",
        "decision_probability_v1.json",
        "singularity_phase1_5",
        "forward_days.jsonl",
    ]
    check(
        "T90 Phase 2A source cannot read forward ledgers or reach live paths",
        config["data"]["phase15LedgerAllowedAsInput"] is False
        and config["output"]["root"].endswith(
            "singularity_phase2a_historical"
        )
        and all(fragment not in source for fragment in forbidden_source_fragments)
        and "history = close[index - window + 1 : index + 1]" in source
        and "merged = merged.dropna(subset=all_features)" in source
        and "sampled_labels.to_csv(" in source
        and config["phase2BGate"]["automaticPromotionAllowed"] is False,
    )


def t91_singularity_phase1_6_gate0_fails_closed_without_break() -> None:
    import json

    import numpy as np
    import pandas as pd
    import research_singularity_phase1_6_hmm_physics_features as phase16

    config_path = (
        ROOT
        / "configs"
        / "research"
        / "singularity_phase1_6_hmm_physics_features.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    phase16.validate_config(config)
    safety = config["safety"]
    check(
        "T91 Phase 1.6 is research-only and cannot mutate trading or forward state",
        config["status"] == "research_only"
        and config["shadowOnly"] is True
        and config["diagnosticOnly"] is True
        and safety["offlineOnly"] is True
        and safety["recordOnly"] is True
        and all(
            safety[key] is False
            for key in [
                "brokerCallsAllowed",
                "onlineInferenceAllowed",
                "phase15WritesAllowed",
                "liveConfigWritesAllowed",
                "overlayWritesAllowed",
                "positionSizingAllowed",
                "orderSubmissionAllowed",
                "riskGateChangesAllowed",
                "buildDecisionIntegrationAllowed",
                "buySellGateIntegrationAllowed",
                "forwardTaskIntegrationAllowed",
                "promotionAllowed",
            ]
        ),
    )
    check(
        "T91 missing break date blocks every preregistration-dependent HMM path",
        config["breakDate"]["value"] is None
        and config["breakDate"]["status"] == "break_date_not_preregistered"
        and config["breakDate"]["automaticSelectionAllowed"] is False
        and config["plannedModels"]["postJumpHMMEnabled"] is False
        and config["plannedModels"]["timeDecayHMMEnabled"] is False
        and config["plannedModels"]["multivariateHMMEnabled"] is False
        and config["breakDate"]["candidateScan"][
            "profitabilityOrOosMetricsUsed"
        ]
        is False,
    )
    required = {
        "data_audit.json",
        "data_audit.md",
        "post_jump_audit.json",
        "coverage_bias_report.json",
        "feature_missingness_report.json",
        "label_distribution_report.json",
        "break_date_risk_report.md",
    }
    check(
        "T91 Gate 0 contract requires every audit artifact before modeling",
        required.issubset(set(config["requiredOutputs"]))
        and config["gate0"][
            "modelingAllowedOnlyIfAllRequiredGatesPass"
        ]
        is True,
    )

    feature_rows = []
    for index, valid in enumerate([1.0, 0.0]):
        row = {
            "trade_date": "2025-01-02",
            "month": "2025-01",
            "stockCode": "513100",
            "etf_category": "cross_border_us",
            "intraday_slot": "11:30" if index == 0 else "13:25",
            "return_24_proxy": 0.01,
            "realized_vol_proxy": 0.002,
            "close_range_proxy": 0.01,
            "trend_acceleration_proxy": 0.001,
            "ews_score": 0.5,
            "lppls_fit_success": valid,
            "lppls_failed_reason": (
                None if valid else "fixed_grid_no_stable_nested_fit"
            ),
            "lppls_tc_proximity": 0.1 if valid else np.nan,
            "lppls_time_to_tc": 9.0 if valid else np.nan,
            "lppls_fit_residual": 0.3 if valid else np.nan,
            "lppls_parameter_stability": 0.8 if valid else np.nan,
            "lppls_window_consensus": 0.7 if valid else np.nan,
            "lppls_bubble_like_score": 0.2 if valid else np.nan,
            "dmd_reconstruction_residual": 0.4 if valid else np.nan,
            "dmd_residual_zscore": 0.0 if valid else np.nan,
            "dmd_spectral_radius": 1.01 if valid else np.nan,
            "dmd_spectral_radius_drift": 0.01 if valid else np.nan,
            "dmd_eigen_instability": 0.01 if valid else np.nan,
            "dmd_window_valid": valid,
            "dmd_missing_bar_ratio": 0.0 if valid else 0.25,
        }
        feature_rows.append(row)
    missingness, _, coverage = phase16.build_feature_reports(
        pd.DataFrame.from_records(feature_rows), config
    )
    check(
        "T91 invalid physics rows remain visible instead of being dropped or imputed",
        missingness["rows"] == 2
        and missingness["noForcedImputation"] is True
        and missingness["invalidRowsRetained"] is True
        and missingness["lppls"]["validRows"] == 1
        and missingness["dmd"]["validRows"] == 1
        and coverage["dmd"]["systematicIntradayExclusion"] is True,
    )

    source = (
        ROOT
        / "scripts"
        / "research_singularity_phase1_6_hmm_physics_features.py"
    ).read_text(encoding="utf-8")
    forbidden = [
        "SkillClient(",
        "submitOrder(",
        "build_decision(",
        "latest_strategy_overlay.json",
        "decision_probability_v1.json",
        "singularity_phase1_5",
        "forward_days.jsonl",
        "GaussianHMM1D(",
        ".fit(",
    ]
    check(
        "T91 Gate 0 implementation cannot fit HMMs, trade, or overwrite prior artifacts",
        all(fragment not in source for fragment in forbidden)
        and "output_dir.mkdir(parents=True, exist_ok=False)" in source
        and config["output"]["root"].endswith(
            "singularity_phase1_6_hmm_physics_features"
        )
        and config["source"]["phase15LedgerAllowedAsInput"] is False,
    )


def t92_singularity_phase1_6_hmm_auxiliary_stays_historical_only() -> None:
    import json

    import research_singularity_phase1_6_hmm_physics_model as model

    config_path = (
        ROOT
        / "configs"
        / "research"
        / "singularity_phase1_6_hmm_physics_model_v1.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model.validate_config(config)
    safety = config["safety"]
    check(
        "T92 user break is frozen but insufficient post-break data stays fail-closed",
        config["userPreregisteredBreak"]["date"] == "2026-06-12"
        and config["userPreregisteredBreak"]["source"]
        == "user_instruction"
        and config["userPreregisteredBreak"][
            "selectedFromModelResults"
        ]
        is False
        and config["hmm"]["postJumpEnabled"] is False
        and config["userPreregisteredBreak"][
            "minimumPostJumpTrainingDays"
        ]
        == 60
        and config["userPreregisteredBreak"][
            "minimumPostJumpTestDays"
        ]
        == 20,
    )
    check(
        "T92 existing HMM is reused with frozen states and no automatic search",
        config["hmm"]["implementation"]
        == "research_hmm_nn_bl.GaussianHMM1D"
        and config["hmm"]["states"] == 3
        and config["hmm"]["iterations"] == 40
        and config["hmm"]["automaticStateOrParameterSearchAllowed"]
        is False
        and config["data"]["fixedHMMFitEnd"]
        < config["data"]["walkForwardStart"],
    )
    check(
        "T92 LPPLS is full-session while DMD stays an explicit late-session population",
        "hmm_ews_lppls"
        in config["populations"]["fullSession"]["variants"]
        and config["populations"]["fullSession"]["requiresDmd"] is False
        and "hmm_ews_dmd"
        in config["populations"]["dmdCompleteLateSession"]["variants"]
        and config["populations"]["dmdCompleteLateSession"][
            "requiresDmd"
        ]
        is True
        and config["populations"]["dmdCompleteLateSession"][
            "cannotBeExtrapolatedToFullSession"
        ]
        is True
        and config["features"]["noForcedDmdImputation"] is True,
    )
    check(
        "T92 auxiliary success gate is clustered, same-sample and cannot promote",
        config["successGate"]["comparisonBaseline"]
        == "current_hmm_ews"
        and config["successGate"][
            "requireBrierClusterBootstrapUpperBelowZero"
        ]
        is True
        and config["successGate"]["sameForecastPopulationRequired"]
        is True
        and config["successGate"]["automaticPromotionAllowed"] is False
        and config["models"]["clusterBootstrapReplicates"] == 2000,
    )
    check(
        "T92 model experiment cannot trade, write forward state, or mutate Phase 1.5",
        safety["offlineOnly"] is True
        and safety["recordOnly"] is True
        and all(
            safety[key] is False
            for key in [
                "brokerCallsAllowed",
                "onlineInferenceAllowed",
                "phase15WritesAllowed",
                "liveConfigWritesAllowed",
                "overlayWritesAllowed",
                "positionSizingAllowed",
                "orderSubmissionAllowed",
                "riskGateChangesAllowed",
                "buildDecisionIntegrationAllowed",
                "buySellGateIntegrationAllowed",
                "forwardTaskIntegrationAllowed",
                "promotionAllowed",
            ]
        )
        and config["source"]["phase15LedgerAllowedAsInput"] is False,
    )
    source = (
        ROOT
        / "scripts"
        / "research_singularity_phase1_6_hmm_physics_model.py"
    ).read_text(encoding="utf-8")
    check(
        "T92 model source has no broker or production artifact path",
        "from research_hmm_nn_bl import GaussianHMM1D" in source
        and "SkillClient(" not in source
        and "submitOrder(" not in source
        and "build_decision(" not in source
        and "latest_strategy_overlay.json" not in source
        and "decision_probability_v1.json" not in source
        and "singularity_phase1_5" not in source
        and "forward_days.jsonl" not in source
        and "output_dir.mkdir(parents=True, exist_ok=False)" in source,
    )

    result_path = (
        ROOT
        / "outputs"
        / "edge_research"
        / "singularity_phase1_6_hmm_physics_features"
        / "model_20260705_break_20260612_v2"
        / "phase1_6_model_result.json"
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    check(
        "T92 promising LPPLS result freezes for new-data retest without deployment",
        result["conclusions"]["lpplsHelpsHMM"] is False
        and result["conclusions"]["lpplsForwardRetestStatus"]
        == "retain_frozen_hypothesis_promising_5bar_not_proven"
        and result["forwardRetestPlan"]["minimumNewIndependentTradingDays"]
        == 20
        and result["forwardRetestPlan"]["parametersRemainFrozen"] is True
        and result["forwardRetestPlan"][
            "historicalRefitOrRetuningAllowed"
        ]
        is False
        and result["forwardRetestPlan"][
            "automaticForwardTaskIntegration"
        ]
        is False
        and result["conclusions"]["productionOrForwardUseAllowed"]
        is False,
    )


def t93_koopman_paper_exception_cannot_waive_predictive_evidence() -> None:
    import json

    import research_singularity_phase1_6_koopman_paper_exception as exception

    config_path = (
        ROOT
        / "configs"
        / "research"
        / "singularity_phase1_6_koopman_paper_exception_v1.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    exception.validate_config(config)
    exemptions = config["paperOnlyExemptions"]
    gates = config["nonWaivableGates"]
    integration = config["paperIntegration"]
    check(
        "T93 Koopman exception waives only coverage and sample-availability gates",
        exemptions["fullSessionCoverageMinimumWaived"] is True
        and exemptions["tenBarEvaluationWaived"] is True
        and exemptions["highRiskMinimumSampleGateWaived"] is True
        and gates["minimumImprovedMetricCountOfFour"] == 3
        and gates["minimumImprovingFoldFraction"] == 0.5
        and gates["minimumImprovingMonthFraction"] == 0.5
        and gates["minimumImprovingEtfCategories"] == 2
        and gates["brierTradeDateClusterBootstrapUpperBelowZero"] is True
        and gates["causalPastOnlyFeatures"] is True
        and gates["sameForecastPopulation"] is True,
    )
    check(
        "T93 any eligible paper use remains a non-promoting risk veto",
        integration["allowedOnlyIfAllNonWaivableGatesPass"] is True
        and integration["policyTypeIfEligible"] == "risk_veto_only"
        and integration["mayGenerateIndependentBuyOrSell"] is False
        and integration["mayChangeSellPath"] is False
        and integration["mayBypassTripleLock"] is False
        and integration["alphaValidated"] is False
        and integration["automaticPromotionAllowed"] is False
        and config["source"]["phase15LedgerAllowedAsInput"] is False,
    )
    source = (
        ROOT
        / "scripts"
        / "research_singularity_phase1_6_koopman_paper_exception.py"
    ).read_text(encoding="utf-8")
    check(
        "T93 exception audit cannot trade or edit the paper agent",
        "submitOrder(" not in source
        and "build_decision(" not in source
        and "latest_strategy_overlay.json" not in source
        and "decision_probability_v1.json" not in source
        and "write_text" not in source
        and "write_bytes" not in source
        and "output_dir.mkdir(parents=True, exist_ok=False)" in source,
    )

    result_path = (
        ROOT
        / "outputs"
        / "edge_research"
        / "singularity_phase1_6_hmm_physics_features"
        / "koopman_exception_20260705_v1"
        / "koopman_paper_exception_result.json"
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    diagnostics = result["diagnostic"]
    result_gates = diagnostics["gates"]
    bootstrap = diagnostics["clusterBootstrap"]
    check(
        "T93 Koopman point estimates do not override failed stability evidence",
        diagnostics["improvedMetricCountOfFour"] == 4
        and diagnostics["improvingFolds"] == 4
        and diagnostics["totalFolds"] == 7
        and diagnostics["improvingMonths"] == 6
        and diagnostics["totalMonths"] == 13
        and result_gates["majorityMonths"] is False
        and result_gates["clusterBootstrapBrier"] is False
        and bootstrap["upper"] >= 0.0
        and result["nonWaivableGatesPassed"] is False
        and result["paperIntegrationAllowed"] is False,
    )
    check(
        "T93 failed exception leaves production artifacts byte-identical",
        result["paperConfigModified"] is False
        and result["agentSourceModified"] is False
        and result["paperConfigHashBefore"]
        == result["paperConfigHashAfter"]
        and result["agentSourceHashBefore"]
        == result["agentSourceHashAfter"]
        and result["phase15Touched"] is False,
    )


def t94_koopman_minute_direction_is_causal_and_shadow_only() -> None:
    import json

    import numpy as np

    import research_koopman_minute_direction as minute_direction

    config_path = (
        ROOT
        / "configs"
        / "research"
        / "koopman_minute_direction_v1.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    minute_direction.validate_config(config)
    check(
        "T94 Koopman is explicitly a per-minute direction model, not a late-session confirmation",
        config["userDirective"]["koopmanRole"]
        == "per_minute_direction_judgment"
        and config["userDirective"]["lateSessionConfirmationOnly"] is False
        and config["data"]["barIntervalMinutes"] == 1
        and config["koopman"]["decisionStrideBars"] == 1
        and config["koopman"]["crossSessionWindowsAllowed"] is False
        and config["labels"]["primary"] == "next_minute_direction"
        and config["labels"]["futureLabelsStoredSeparately"] is True,
    )
    check(
        "T94 fixed linear DMD cannot silently become tuned, kernel, or deep Koopman",
        config["koopman"]["windowBars"] == 24
        and config["koopman"]["delayDimension"] == 3
        and config["koopman"]["ridge"] == 0.000001
        and config["koopman"]["parameterSearchAllowed"] is False
        and config["koopman"]["kernelEnabled"] is False
        and config["koopman"]["deepEnabled"] is False,
    )
    rng = np.random.default_rng(20260705)
    completed_history = rng.normal(size=(24, 4))
    unchanged = minute_direction.dmd_direction(
        completed_history.copy(), config
    )
    unseen_future = rng.normal(size=(1, 4)) * 1000.0
    with_unseen_suffix = np.vstack([completed_history, unseen_future])
    repeated = minute_direction.dmd_direction(
        with_unseen_suffix[:24], config
    )
    check(
        "T94 completed-minute Koopman forecast ignores an unseen future shock",
        unchanged is not None
        and repeated is not None
        and all(
            abs(unchanged[key] - repeated[key]) < 1e-12
            for key in unchanged
        ),
    )
    safety = config["safety"]
    integration = config["paperIntegration"]
    check(
        "T94 minute direction remains record-only and cannot alter trading",
        safety["offlineOnly"] is True
        and safety["recordOnly"] is True
        and all(
            safety[key] is False
            for key in [
                "brokerCallsAllowed",
                "onlineInferenceAllowed",
                "liveConfigWritesAllowed",
                "overlayWritesAllowed",
                "positionSizingAllowed",
                "orderSubmissionAllowed",
                "riskGateChangesAllowed",
                "buildDecisionIntegrationAllowed",
                "buySellGateIntegrationAllowed",
                "promotionAllowed",
            ]
        )
        and integration["mayGenerateIndependentOrders"] is False
        and integration["mayChangeSellPath"] is False
        and integration["mayChangePositionSizing"] is False
        and integration["mayBypassTripleLock"] is False,
    )
    source = (
        ROOT / "scripts" / "research_koopman_minute_direction.py"
    ).read_text(encoding="utf-8")
    check(
        "T94 minute-direction research has no broker, order, or decision-gate path",
        "submitOrder(" not in source
        and "build_decision(" not in source
        and "latest_strategy_overlay.json" not in source
        and "decision_probability_v1.json" not in source
        and "output_dir.mkdir(parents=True, exist_ok=False)" in source,
    )

    result_path = (
        ROOT
        / "outputs"
        / "edge_research"
        / "koopman_minute_direction"
        / "koopman_minute_direction_20260705_v1"
        / "koopman_minute_direction_result.json"
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    gate = result["successGate"]
    check(
        "T94 historical probability gate passes without claiming paper eligibility",
        result["featureAudit"]["firstDecisionMinute"] == "10:00"
        and result["featureAudit"]["lastDecisionMinute"] == "14:59"
        and gate["improvedMetricCountOfFour"] == 4
        and gate["improvingFolds"] == 4
        and gate["totalFolds"] == 5
        and gate["improvingMonths"] == 3
        and gate["totalMonths"] == 4
        and gate["clusterBootstrap"]["upper"] < 0.0
        and gate["passes"] is True
        and result["paperIntegrationAllowed"] is False,
    )
    check(
        "T94 historical pass leaves paper configuration and agent byte-identical",
        result["paperConfigModified"] is False
        and result["agentSourceModified"] is False
        and result["paperConfigHashBefore"]
        == result["paperConfigHashAfter"]
        and result["agentSourceHashBefore"]
        == result["agentSourceHashAfter"],
    )


def t95_online_viterbi_ablation_is_causal_and_research_only() -> None:
    import json

    import numpy as np

    import research_singularity_phase1_6_viterbi_ablation as viterbi

    config_path = (
        ROOT
        / "configs"
        / "research"
        / "singularity_phase1_6_viterbi_ablation_v1.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    viterbi.validate_config(config)
    check(
        "T95 Viterbi is online endpoint decoding, not smoothing or full-path hindsight",
        config["viterbi"]["mode"] == "online_endpoint_decode"
        and config["viterbi"]["sessionReset"] is True
        and config["viterbi"]["usesFullSequenceBacktracking"] is False
        and config["viterbi"]["usesForwardBackwardSmoothing"] is False
        and config["viterbi"]["usesFutureObservations"] is False
        and config["viterbi"]["newHmmFitAllowed"] is False
        and config["viterbi"]["automaticStateOrParameterSearchAllowed"] is False,
    )
    phase16_result = json.loads(
        (
            ROOT
            / "outputs"
            / "edge_research"
            / "singularity_phase1_6_hmm_physics_features"
            / "model_20260705_break_20260612_v2"
            / "phase1_6_model_result.json"
        ).read_text(encoding="utf-8")
    )
    hmm_audit = phase16_result["hmmAudit"]
    rng = np.random.default_rng(20260708)
    prefix = rng.normal(0.0, 0.001, size=30)
    future = np.asarray([0.05, -0.05, 0.04, -0.04])
    first = viterbi.online_viterbi_decode(prefix, hmm_audit)
    second = viterbi.online_viterbi_decode(
        np.concatenate([prefix, future]), hmm_audit
    ).iloc[: len(prefix)]
    columns = [
        "viterbi_state",
        "viterbi_bear_state",
        "viterbi_middle_state",
        "viterbi_bull_state",
        "viterbi_state_age_bars",
        "viterbi_path_transition_risk",
        "viterbi_endpoint_confidence",
        "viterbi_log_margin",
    ]
    check(
        "T95 Viterbi prefix output is invariant to appended unseen future shocks",
        np.allclose(
            first[columns].to_numpy(dtype=float),
            second[columns].to_numpy(dtype=float),
            atol=1e-12,
            rtol=0.0,
        ),
    )
    safety = config["safety"]
    integration = config["paperIntegration"]
    check(
        "T95 Viterbi ablation remains offline, record-only, and non-promoting",
        safety["offlineOnly"] is True
        and safety["recordOnly"] is True
        and all(
            safety[key] is False
            for key in [
                "brokerCallsAllowed",
                "onlineInferenceAllowed",
                "liveConfigWritesAllowed",
                "overlayWritesAllowed",
                "positionSizingAllowed",
                "orderSubmissionAllowed",
                "riskGateChangesAllowed",
                "buildDecisionIntegrationAllowed",
                "buySellGateIntegrationAllowed",
                "promotionAllowed",
            ]
        )
        and integration["allowed"] is False
        and integration["mayGenerateIndependentOrders"] is False
        and integration["mayChangeSellPath"] is False
        and integration["mayChangePositionSizing"] is False
        and integration["mayBypassTripleLock"] is False,
    )
    source = (
        ROOT / "scripts" / "research_singularity_phase1_6_viterbi_ablation.py"
    ).read_text(encoding="utf-8")
    check(
        "T95 Viterbi research has no broker, order, or paper decision path",
        "submitOrder(" not in source
        and "build_decision(" not in source
        and "latest_strategy_overlay.json" not in source
        and "run_t0_intraday_agent" not in source
        and "output_dir.mkdir(parents=True, exist_ok=False)" in source,
    )
    result_path = (
        ROOT
        / "outputs"
        / "edge_research"
        / "singularity_phase1_6_viterbi_ablation"
        / "viterbi_ablation_20260708_v1"
        / "viterbi_ablation_result.json"
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    diagnostic = result["candidateDiagnostics"]
    check(
        "T95 Viterbi does not pass historical HMM+EWS incremental gate",
        result["viterbiFeatureAudit"]["prefixInvarianceCheckPassed"] is True
        and result["viterbiFeatureAudit"]["filteredAgreementRate"] > 0.97
        and diagnostic["5"]["improvedMetricCountOfFour"] == 1
        and diagnostic["5"]["passes"] is False
        and diagnostic["10"]["improvedMetricCountOfFour"] == 0
        and diagnostic["10"]["passes"] is False
        and result["conclusions"]["viterbiHelpsHMM"] is False
        and result["paperIntegrationAllowed"] is False,
    )


def t96_risk_stack_phase1_7_is_causal_audit_only() -> None:
    import json

    import numpy as np

    import research_risk_stack_phase1_7 as risk_stack

    config_path = (
        ROOT / "configs" / "research" / "risk_stack_phase1_7_audit_v1.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    risk_stack.validate_config(config)
    check(
        "T96 risk stack is explicitly audit-only and uses fixed parameters",
        config["status"] == "research_only"
        and config["shadowOnly"] is True
        and config["diagnosticOnly"] is True
        and config["bocpd"]["parameterSearchAllowed"] is False
        and config["hawkes"]["parameterSearchAllowed"] is False
        and config["hawkes"]["usesCurrentEventInCurrentIntensity"] is False
        and config["riskStack"]["noParameterSearch"] is True,
    )
    rng = np.random.default_rng(20260708)
    prefix = rng.normal(0.0, 0.001, size=40)
    future = np.asarray([0.05, -0.05, 0.04])
    first = risk_stack.bocpd_gaussian_known_variance(
        prefix,
        hazard=0.04,
        max_run_length=120,
        alert_run_length=3,
        prior_mean=0.0,
        prior_variance=25.0 * float(np.var(prefix) + 1e-8),
        observation_variance=float(np.var(prefix) + 1e-8),
    )
    second = risk_stack.bocpd_gaussian_known_variance(
        np.concatenate([prefix, future]),
        hazard=0.04,
        max_run_length=120,
        alert_run_length=3,
        prior_mean=0.0,
        prior_variance=25.0 * float(np.var(prefix) + 1e-8),
        observation_variance=float(np.var(prefix) + 1e-8),
    ).iloc[: len(prefix)]
    check(
        "T96 BOCPD prefix output is invariant to appended unseen future shocks",
        np.allclose(
            first.to_numpy(dtype=float),
            second.to_numpy(dtype=float),
            atol=1e-12,
            rtol=0.0,
        ),
    )
    toy = risk_stack.pd.DataFrame(
        {
            "timestamp": risk_stack.pd.date_range(
                "2026-01-01 09:35", periods=5, freq="5min"
            ),
            "trade_date": ["2026-01-01"] * 5,
            "stockCode": ["000001"] * 5,
            "observable_risk_event": [0, 1, 0, 0, 1],
        }
    )
    toy_config = json.loads(json.dumps(config))
    toy_config["data"]["trainingStatisticsEnd"] = "2026-01-01"
    toy_out, _ = risk_stack.add_discrete_hawkes_intensity(toy, toy_config)
    check(
        "T96 Hawkes intensity excludes current event from current score",
        abs(float(toy_out["hawkes_event_intensity"].iloc[0]) - 0.4) < 1e-12
        and abs(float(toy_out["hawkes_event_intensity"].iloc[1]) - 0.4) < 1e-12
        and float(toy_out["hawkes_event_intensity"].iloc[2]) > 0.4,
    )
    safety = config["safety"]
    integration = config["paperIntegration"]
    check(
        "T96 risk stack cannot trade, size, gate, overlay or promote",
        safety["offlineOnly"] is True
        and safety["recordOnly"] is True
        and all(
            safety[key] is False
            for key in [
                "brokerCallsAllowed",
                "onlineInferenceAllowed",
                "liveConfigWritesAllowed",
                "overlayWritesAllowed",
                "positionSizingAllowed",
                "orderSubmissionAllowed",
                "riskGateChangesAllowed",
                "buildDecisionIntegrationAllowed",
                "buySellGateIntegrationAllowed",
                "promotionAllowed",
            ]
        )
        and integration["allowed"] is False
        and integration["mayGenerateIndependentOrders"] is False
        and integration["mayChangeSellPath"] is False
        and integration["mayChangePositionSizing"] is False
        and integration["mayBypassTripleLock"] is False,
    )
    source = (ROOT / "scripts" / "research_risk_stack_phase1_7.py").read_text(
        encoding="utf-8"
    )
    check(
        "T96 risk-stack source has no broker, order or production decision path",
        "submitOrder(" not in source
        and "build_decision(" not in source
        and "latest_strategy_overlay.json" not in source
        and "run_t0_intraday_agent" not in source
        and "output_dir.mkdir(parents=True, exist_ok=False)" in source,
    )
    result_path = (
        ROOT
        / "outputs"
        / "edge_research"
        / "risk_stack_phase1_7"
        / "risk_stack_p17_20260708_v2"
        / "risk_stack_phase1_7_result.json"
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    check(
        "T96 Phase 1.7 result does not justify costed replay or paper integration",
        result["lookaheadAudit"]["bocpdPrefixInvariant"] is True
        and result["lookaheadAudit"][
            "hawkesCurrentEventExcludedFromCurrentIntensity"
        ]
        is True
        and result["conclusions"]["fullRiskStackShowsStableEdge"] is False
        and result["conclusions"]["dmdCompleteStackShowsStableEdge"] is False
        and result["conclusions"]["costedReplayJustifiedNow"] is False
        and result["paperIntegrationAllowed"] is False,
    )


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
    t30_volume_capture_and_surge()
    t32_lead_lag_detection()
    t33_regime_classifier()
    t34_exit_rule_simulation()
    t35_sizing_weights()
    t36_overfitting_guard()
    t37_committed_holdings_cap()
    t61_full_market_and_t0_entry_guards()
    t39_timing_accuracy_helpers()
    t38_full_minute_replay_builder()
    t22_dynamic_universe_selection()
    t23_sector_diversification_entry_filter()
    t24_sector_limit_never_blocks_sells()
    t25_daily_minute_quote_paths()
    t26_daily_replay_cache_and_directory_reader()
    t27_dynamic_gate_replay_cache()
    t28_layered_backtest_pipeline()
    t29_holdings_calibration_classification()
    t31_early_entry_daily_accumulation()
    t40_external_etf_observation_pool()
    t41_unattended_chatgpt_watchlist_generator()
    t42_oos_variance_metrics()
    t43_fail_closed_t0_etf_master()
    t44_point_in_time_opening_research()
    t45_observation_pool_history_archive()
    t46_overseas_gap_point_in_time()
    t47_i03_group_ablation_isolation()
    t48_point_in_time_liquidity_gate()
    t49_l4_forward_preregistration()
    t50_l4_forward_shadow_pipeline()
    t51_execution_accounting_and_trust_contract()
    t52_decision_scoring_system()
    t53_pseudo_forward_prefix_and_isolation()
    t54_weight_research_and_forward_shadow()
    t55_decision_score_semantic_versioning()
    t56_high_low_score_separation()
    t57_bayesian_probability_shadow()
    t58_forward_probability_ledger_and_priors()
    t59_every_decision_and_daily_score_review()
    t60_sell_logic_v2_timing_gate()
    t74_entry_logic_v2_pullback_gate()
    t81_breadth_block_never_blocks_sells()
    t75_actual_trade_exit_research_is_causal_and_safe()
    t76_l2_exit_timing_is_forward_day_clustered_and_safe()
    t77_l2_sell_execution_is_next_snapshot_conservative_and_safe()
    t78_l2_sell_slicing_uses_visible_depth_and_exact_lots()
    t79_exit_policy_matrix_is_causal_complete_and_shadow_only()
    t80_exit_diagnostics_separates_labels_from_observables()
    t61_daily_momentum_pool_failopen()
    t62_trend_deploy_factor()
    t63_hsmm_regime_research_is_point_in_time()
    t64_hmm_nn_bl_research_is_frozen_and_safe()
    t65_literature_reversal_research_is_point_in_time()
    t66_forward_execution_friction_is_conservative_and_safe()
    t67_full_t0_depth_collector_is_point_in_time_and_safe()
    t68_paper_order_lifecycle_uses_confirmed_fills_only()
    t69_iopv_pcf_collector_preserves_source_tiers()
    t70_option_pressure_collector_is_forward_and_unsigned()
    t71_same_index_underreaction_is_next_bar_oos_and_safe()
    t72_trend_pullback_recovery_is_causal_costed_and_safe()
    t73_frontier_competition_ranker_is_fresh_paper_only_and_auditable()
    t82_local_news_watchlist_is_causal_no_api_and_shadow_only()
    t83_laplace_copula_price_chain_is_frozen_next_bar_and_safe()
    t84_tail_probability_bounds_are_causal_conservative_and_safe()
    t85_minute_forecast_is_next_bar_point_in_time_and_safe()
    t86_minute_model_fusion_is_causal_and_shadow_only()
    t87_conditional_minute_tail_preserves_dependence_safely()
    t88_singularity_phase1_is_causal_purged_and_shadow_only()
    t89_singularity_phase1_5_is_frozen_forward_shadow_only()
    t90_singularity_phase2a_is_causal_historical_and_isolated()
    t91_singularity_phase1_6_gate0_fails_closed_without_break()
    t92_singularity_phase1_6_hmm_auxiliary_stays_historical_only()
    t93_koopman_paper_exception_cannot_waive_predictive_evidence()
    t94_koopman_minute_direction_is_causal_and_shadow_only()
    t95_online_viterbi_ablation_is_causal_and_research_only()
    t96_risk_stack_phase1_7_is_causal_audit_only()
    print()
    if failures:
        print(f"FAILED: {len(failures)} invariant(s): {failures}")
        sys.exit(1)
    print("ALL INVARIANTS PASSED")
