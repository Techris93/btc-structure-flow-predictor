from __future__ import annotations

import json
from pathlib import Path


DEFAULT_FLOW_GATE = {
    "gate_mode": "independent",
    "legacy_threshold": 0.40,
    "market_threshold": 0.40,
    "raw_threshold": 0.40,
    "price_bucket": 25.0,
    "full_credit_ratio": 1.5,
}


def load_flow_gate(path, requested_mode="independent", overrides=None):
    """Load the requested gate without mislabelling hand-set values calibrated.

    ``independent`` activates the separately configured market-flow and raw-
    footprint thresholds. ``calibrated`` still requires a passed artifact,
    while ``shadow`` retains the legacy composite for controlled comparisons.
    """
    config = {**DEFAULT_FLOW_GATE, **(overrides or {})}
    requested_mode = str(requested_mode or "independent").lower()
    artifact = None
    try:
        artifact = json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError):
        artifact = None
    if requested_mode == "independent":
        config["gate_mode"] = "independent"
        return config, artifact
    if requested_mode != "calibrated":
        config["gate_mode"] = "shadow"
        return config, artifact
    selected = (artifact or {}).get("selected_config") or {}
    if not (artifact or {}).get("promotion_passed") or not selected:
        config["gate_mode"] = "shadow"
        config["fallback_reason"] = "missing_or_unpassed_calibration"
        return config, artifact
    config.update({
        "gate_mode": "calibrated",
        "market_threshold": float(selected["market_threshold"]),
        "raw_threshold": float(selected["raw_threshold"]),
        "price_bucket": float(selected["price_bucket"]),
        "full_credit_ratio": float(selected["full_credit_ratio"]),
        "artifact_run_hash": artifact.get("run_hash"),
    })
    return config, artifact
