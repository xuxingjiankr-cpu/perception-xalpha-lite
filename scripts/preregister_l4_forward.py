"""One-time, fail-closed preregistration of the L4 forward shadow pair."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from run_etf_paper_trading_agent import ROOT
from run_t0_intraday_agent import apply_evolution_overlay_if_enabled


LIVE_CONFIG = ROOT / "configs" / "t0_intraday_paper_agent.json"
OVERLAY = ROOT / "outputs" / "t0_strategy_evolution" / "latest_strategy_overlay.json"
SHADOW_DIR = ROOT / "configs" / "shadow"
BASELINE = SHADOW_DIR / "l4_forward_baseline.json"
CANDIDATE = SHADOW_DIR / "l4_forward_candidate.json"
CUTOFF = "2026-06-18"
EXPECTED_BASE = {
    "market_correlation_stress.avg_abs_corr_threshold": 0.75,
    "entry_score_threshold": 50,
    "entry_momentum_pct": 0.0015,
    "min_hold_minutes": 10,
}
L4 = {
    "market_correlation_stress.avg_abs_corr_threshold": 0.68,
    "entry_score_threshold": 53,
    "entry_momentum_pct": 0.0012,
    "min_hold_minutes": 14,
}


def sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def get(strategy: dict, path: str):
    node = strategy
    for part in path.split("."):
        node = node[part]
    return node


def set_value(strategy: dict, path: str, value) -> None:
    parts = path.split(".")
    node = strategy
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value


def atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temp = Path(handle.name)
    temp.replace(path)


def build_pair() -> tuple[dict, dict]:
    raw = json.loads(LIVE_CONFIG.read_text(encoding="utf-8"))
    effective = apply_evolution_overlay_if_enabled(copy.deepcopy(raw))
    overlay_meta = effective.pop("_evolution_overlay", {})
    actual = {path: get(effective["strategy"], path) for path in EXPECTED_BASE}
    if actual != EXPECTED_BASE:
        raise RuntimeError(f"live effective baseline no longer matches preregistration: {actual}")
    registered = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
    common = {
        "preregisteredAt": registered,
        "prospectiveAfter": CUTOFF,
        "hypothesisId": "l4_selectivity_forward_v1",
        "singleCandidate": True,
        "diagnosticOnly": True,
        "liveReady": False,
        "formalStrategyAllowed": False,
        "liveConfigSha256": sha(LIVE_CONFIG),
        "overlayFileSha256": sha(OVERLAY),
        "overlayAppliedToBaseline": bool(overlay_meta.get("applied")),
        "overlayReason": overlay_meta.get("reason"),
        "lockedMetrics": {
            "primary": ["daily_pnl_std_reduction_pct", "worst_day_improvement"],
            "secondary": ["net_12bps_total", "daily_sharpe"],
        },
        "lockedPassCriteria": {
            "minimumForwardDays": 20,
            "minimumStdReductionPct": 10.0,
            "minimumWorstDayImprovement": 0.0,
            "candidateSharpeMustNotBeLower": True,
            "dmOneSidedAlpha": 0.05,
            "dsrAlpha": 0.10,
            "maximumPbo": 0.25,
        },
    }
    baseline = copy.deepcopy(effective)
    baseline["research_metadata"] = {**common, "role": "frozen_effective_live_baseline", "l4Changes": {}}
    candidate = copy.deepcopy(effective)
    for path, value in L4.items():
        set_value(candidate["strategy"], path, value)
    candidate["research_metadata"] = {
        **common, "role": "shadow_l4_candidate",
        "l4Changes": {path: {"baseline": EXPECTED_BASE[path], "candidate": value} for path, value in L4.items()},
    }
    return baseline, candidate


def main() -> int:
    if BASELINE.exists() or CANDIDATE.exists():
        raise RuntimeError("preregistered files already exist; refusing to overwrite the hypothesis")
    baseline, candidate = build_pair()
    atomic(BASELINE, baseline)
    atomic(CANDIDATE, candidate)
    print(f"baseline={BASELINE}")
    print(f"candidate={CANDIDATE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
