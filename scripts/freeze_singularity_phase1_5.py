"""Create the one-time frozen model bundle for Singularity Phase 1.5.

The command reads only the sealed Phase 1 historical feature/label tables,
refuses to include data after the configured historical cutoff, and refuses to
overwrite an existing bundle.  The daily Phase 1.5 runner has no fitting path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from research_hmm_nn_bl import ROOT
from research_singularity_phase1 import (
    make_classifier,
    purge_symbol_tail,
    variant_features,
)


DEFAULT_CONFIG = (
    ROOT / "configs" / "research" / "singularity_phase1_5_forward.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_sha256(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def export_pipeline(
    model: Any, calibrator: LogisticRegression
) -> dict[str, Any]:
    scaler = model.named_steps["scale"]
    logistic = model.named_steps["model"]
    if list(logistic.classes_) != [0, 1] or list(calibrator.classes_) != [0, 1]:
        raise RuntimeError("unexpected classifier class order")
    return {
        "scalerMean": scaler.mean_.astype(float).tolist(),
        "scalerScale": scaler.scale_.astype(float).tolist(),
        "logisticCoefficient": logistic.coef_[0].astype(float).tolist(),
        "logisticIntercept": float(logistic.intercept_[0]),
        "plattCoefficient": float(calibrator.coef_[0][0]),
        "plattIntercept": float(calibrator.intercept_[0]),
        "classes": [0, 1],
    }


def fit_frozen_bundle(config_path: Path) -> tuple[Path, dict[str, Any]]:
    forward_config = json.loads(config_path.read_text(encoding="utf-8"))
    output = resolve(ROOT, forward_config["frozenModel"]["path"])
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen bundle: {output}")
    phase1_cfg_path = resolve(ROOT, forward_config["phase1"]["config"])
    if json_sha256(phase1_cfg_path) != forward_config["phase1"]["configSha256"]:
        raise RuntimeError("Phase 1 config hash mismatch")
    phase1_config = json.loads(phase1_cfg_path.read_text(encoding="utf-8"))
    feature_path = resolve(ROOT, forward_config["phase1"]["featureTable"])
    label_path = resolve(ROOT, forward_config["phase1"]["labelTable"])
    result_path = resolve(ROOT, forward_config["phase1"]["result"])
    universe_path = resolve(ROOT, forward_config["phase1"]["universeAudit"])
    for path in (feature_path, label_path, result_path, universe_path):
        if not path.exists():
            raise FileNotFoundError(path)

    features = pd.read_csv(feature_path)
    labels = pd.read_csv(label_path)
    features["timestamp"] = pd.to_datetime(features["timestamp"])
    labels["timestamp"] = pd.to_datetime(labels["timestamp"])
    cutoff = str(forward_config["phase1"]["historicalCutoff"])
    if str(features["trade_date"].max()) > cutoff:
        raise RuntimeError("feature table contains post-cutoff data")
    if str(labels["trade_date"].max()) > cutoff:
        raise RuntimeError("label table contains post-cutoff data")

    phase1_result = json.loads(result_path.read_text(encoding="utf-8"))
    universe = json.loads(universe_path.read_text(encoding="utf-8"))
    codes = [str(row["stockCode"]) for row in universe["selected"]]
    if len(codes) != int(phase1_config["data"]["universeSize"]):
        raise RuntimeError("frozen universe size mismatch")

    keys = ["timestamp", "trade_date", "stockCode"]
    merged = features.merge(labels, on=keys, how="inner", validate="one_to_one")
    variants = variant_features(phase1_config)
    purge_bars = int(phase1_config["models"]["purgeBars"])
    calibration_days = int(phase1_config["models"]["calibrationDays"])
    bundle_models: dict[str, Any] = {}
    for horizon in phase1_config["labels"]["activeHorizonsBars"]:
        horizon = int(horizon)
        target = f"turning_point_{horizon}"
        columns_needed = sorted(
            {column for values in variants.values() for column in values}
        )
        rows = merged.dropna(subset=columns_needed + [target]).copy()
        rows = rows[rows["trade_date"] <= cutoff].copy()
        rows[target] = rows[target].astype(int)
        before_test, test_purged = purge_symbol_tail(rows, purge_bars)
        dates = sorted(before_test["trade_date"].unique())
        if len(dates) <= calibration_days + int(
            phase1_config["models"]["minimumBaseFitDays"]
        ):
            raise RuntimeError(f"insufficient freeze dates for horizon {horizon}")
        calibration_dates = set(dates[-calibration_days:])
        calibration = before_test[
            before_test["trade_date"].isin(calibration_dates)
        ].copy()
        fit_raw = before_test[
            ~before_test["trade_date"].isin(calibration_dates)
        ].copy()
        fit, calibration_purged = purge_symbol_tail(fit_raw, purge_bars)
        if fit[target].nunique() != 2 or calibration[target].nunique() != 2:
            raise RuntimeError(f"class missing while freezing horizon {horizon}")
        horizon_models: dict[str, Any] = {}
        for variant, columns in variants.items():
            model = make_classifier(phase1_config)
            model.fit(
                fit[columns].to_numpy(dtype=float),
                fit[target].to_numpy(dtype=int),
            )
            raw_calibration = np.clip(
                model.predict_proba(
                    calibration[columns].to_numpy(dtype=float)
                )[:, 1],
                1e-6,
                1.0 - 1e-6,
            )
            calibration_logit = np.log(
                raw_calibration / (1.0 - raw_calibration)
            ).reshape(-1, 1)
            calibrator = LogisticRegression(
                C=float(phase1_config["models"]["calibrationC"]),
                max_iter=500,
                random_state=int(phase1_config["models"]["randomSeed"]),
            )
            calibrator.fit(
                calibration_logit,
                calibration[target].to_numpy(dtype=int),
            )
            horizon_models[variant] = {
                "features": columns,
                **export_pipeline(model, calibrator),
            }
        bundle_models[str(horizon)] = {
            "target": target,
            "baseFitStart": str(fit["trade_date"].min()),
            "baseFitEnd": str(fit["trade_date"].max()),
            "baseFitDays": int(fit["trade_date"].nunique()),
            "baseFitSamples": int(len(fit)),
            "baseFitPositiveRate": float(fit[target].mean()),
            "calibrationStart": str(calibration["trade_date"].min()),
            "calibrationEnd": str(calibration["trade_date"].max()),
            "calibrationDays": int(calibration["trade_date"].nunique()),
            "calibrationSamples": int(len(calibration)),
            "calibrationPositiveRate": float(calibration[target].mean()),
            "testBoundaryPurgedRows": test_purged,
            "calibrationBoundaryPurgedRows": calibration_purged,
            "models": horizon_models,
        }

    bundle = {
        "schemaVersion": "singularity_phase1_5_frozen_model_v1",
        "status": "research_only",
        "shadowOnly": True,
        "version": forward_config["version"],
        "createdAt": datetime.now().astimezone().isoformat(timespec="seconds"),
        "historicalCutoff": cutoff,
        "prospectiveAfter": forward_config["forward"]["prospectiveAfter"],
        "runtimeRefitAllowed": False,
        "parameterUpdatesAllowed": False,
        "hashMode": "canonical_json_sha256",
        "phase1ConfigSha256": json_sha256(phase1_cfg_path),
        "featureTableSha256": sha256(feature_path),
        "labelTableSha256": sha256(label_path),
        "phase1ResultSha256": sha256(result_path),
        "universeAuditSha256": sha256(universe_path),
        "universe": codes,
        "featureConfiguration": phase1_config["features"],
        "labelConfiguration": phase1_config["labels"],
        "modelConfiguration": phase1_config["models"],
        "frozenEwsStandardization": phase1_result["featureDefinition"]["ews"][
            "frozenStandardization"
        ],
        "frozenHmm": phase1_result["featureDefinition"]["hmm"],
        "modelsByHorizon": bundle_models,
        "safety": {
            "orderSubmissionAllowed": False,
            "positionSizingAllowed": False,
            "buildDecisionIntegrationAllowed": False,
            "buySellGateIntegrationAllowed": False,
            "promotionAllowed": False,
        },
    }
    atomic_json(output, bundle)
    return output, bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output, bundle = fit_frozen_bundle(args.config.resolve())
    print(
        json.dumps(
            {
                "status": bundle["status"],
                "version": bundle["version"],
                "historicalCutoff": bundle["historicalCutoff"],
                "modelHorizons": sorted(bundle["modelsByHorizon"]),
                "output": str(output),
                "sha256": json_sha256(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
