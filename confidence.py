"""
confidence.py -- Confidence Scoring & Dynamic MIN_EDGE (Phase 4C).

Computes a confidence score ∈ [0, 1] from four gates:
  1. Forecast freshness: how recently was the forecast fetched
  2. Observation freshness: how recent is the METAR/ASOS observation
  3. Observation alignment: |residual_ewma| -- how well obs match forecast
  4. Boundary brittleness: z = distance_to_nearest_boundary / σ

The confidence score modulates MIN_EDGE:
  - High confidence → lower MIN_EDGE (trade more aggressively)
  - Low confidence → higher MIN_EDGE (require more edge)

Pre-dawn special rules:
  - confidence ≥ 0.6 AND edge ≥ 20% AND z ≥ 1.0
"""

import math
import logging
from datetime import datetime, timezone, timedelta

import config

logger = logging.getLogger(__name__)


def compute_confidence(forecast_age_min, obs_age_min, residual_ewma,
                       boundary_z, hour_et):
    """
    Compute confidence score from four gates.

    Args:
        forecast_age_min: minutes since last forecast fetch
        obs_age_min: minutes since last observation (inf if none)
        residual_ewma: observation residual EWMA (None if no obs)
        boundary_z: min(distance_to_any_boundary / σ) -- how close μ is to a bucket edge
        hour_et: current ET hour (0-23)

    Returns:
        (confidence: float [0,1], gate_scores: dict)
    """
    gates = {}

    # Gate 1: Forecast freshness
    # Full score if < FORECAST_FRESH_MINUTES, linear decay to 0 at 3x
    max_age = config.FORECAST_FRESH_MINUTES
    if forecast_age_min <= max_age:
        gates["forecast_freshness"] = 1.0
    elif forecast_age_min <= max_age * 3:
        gates["forecast_freshness"] = 1.0 - (forecast_age_min - max_age) / (max_age * 2)
    else:
        gates["forecast_freshness"] = 0.0

    # Gate 2: Observation freshness
    # Full score if < OBS_STALE_MINUTES, linear decay to 0 at 3x
    obs_max = config.OBS_STALE_MINUTES
    if obs_age_min <= obs_max:
        gates["obs_freshness"] = 1.0
    elif obs_age_min <= obs_max * 3:
        gates["obs_freshness"] = 1.0 - (obs_age_min - obs_max) / (obs_max * 2)
    else:
        gates["obs_freshness"] = 0.0

    # Gate 3: Observation alignment
    # Full score if |residual_ewma| < 0.5, linear decay to 0 at OBS_ALIGNMENT_MAX_RESIDUAL
    if residual_ewma is None:
        gates["obs_alignment"] = 0.5  # neutral when no obs
    else:
        abs_res = abs(residual_ewma)
        max_res = config.OBS_ALIGNMENT_MAX_RESIDUAL
        if abs_res <= 0.5:
            gates["obs_alignment"] = 1.0
        elif abs_res <= max_res:
            gates["obs_alignment"] = 1.0 - (abs_res - 0.5) / (max_res - 0.5)
        else:
            gates["obs_alignment"] = 0.0

    # Gate 4: Boundary brittleness
    # Full score if z > 2.0, linear decay to 0 at z < threshold
    z_thresh = config.BOUNDARY_BRITTLENESS_Z_THRESHOLD
    if boundary_z is None or boundary_z >= 2.0:
        gates["boundary_brittleness"] = 1.0
    elif boundary_z >= z_thresh:
        gates["boundary_brittleness"] = (boundary_z - z_thresh) / (2.0 - z_thresh)
    else:
        gates["boundary_brittleness"] = 0.0

    # Weighted average
    weights = config.CONFIDENCE_GATE_WEIGHTS
    total_weight = sum(weights.values())
    confidence = sum(
        gates[gate] * weights[gate] for gate in gates
    ) / total_weight

    confidence = max(0.0, min(1.0, confidence))

    return confidence, gates


def compute_dynamic_min_edge(hour_et, confidence):
    """
    Compute dynamic MIN_EDGE based on time-of-day and confidence.

    Lower confidence → higher required edge.
    Pre-dawn always requires high edge.

    Returns min_edge (float, e.g. 0.12 for 12%).
    """
    # Base edge by time-of-day
    if hour_et < 6:
        base_edge = config.MIN_EDGE_PREDAWN
    elif hour_et < 10:
        base_edge = config.MIN_EDGE_MORNING
    elif hour_et < 16:
        base_edge = config.MIN_EDGE_AFTERNOON
    else:
        base_edge = config.MIN_EDGE_EVENING

    # Scale by confidence: low confidence → higher edge
    # At confidence=1.0, use 80% of base; at confidence=0.3, use 150% of base
    if confidence >= 0.8:
        scale = 0.8
    elif confidence >= 0.5:
        scale = 0.8 + (0.8 - confidence) * (0.7 / 0.3)  # 0.8 to 1.5
    else:
        scale = 1.5

    dynamic_edge = base_edge * scale

    # Floor: never go below the old static MIN_EDGE
    dynamic_edge = max(dynamic_edge, config.MIN_EDGE)

    return dynamic_edge


def compute_boundary_z(mu, sigma, bucket_boundaries):
    """
    Compute z = min distance from μ to any bucket boundary, divided by σ.

    Args:
        mu: model mean temperature
        sigma: model std dev
        bucket_boundaries: list of boundary values (ints) from parsed buckets

    Returns:
        z (float) -- lower means μ is closer to a boundary (more brittle)
    """
    if not bucket_boundaries or sigma <= 0:
        return None

    min_distance = min(abs(mu - b) for b in bucket_boundaries)
    z = min_distance / sigma
    return z


def extract_bucket_boundaries(buckets):
    """
    Extract all unique boundary values from a list of Bucket objects.
    These are the critical temperature thresholds where small forecast
    errors can flip the outcome.
    """
    boundaries = set()
    for b in buckets:
        if b.low is not None:
            boundaries.add(b.low)
            boundaries.add(b.low - 0.5)  # continuity correction boundary
        if b.high is not None:
            boundaries.add(b.high)
            boundaries.add(b.high + 0.5)  # continuity correction boundary
    return sorted(boundaries)


def passes_predawn_gates(confidence, edge, boundary_z):
    """
    Check pre-dawn special gates.
    Returns (passes: bool, reason: str).
    """
    if confidence < config.PREDAWN_MIN_CONFIDENCE:
        return False, "confidence %.2f < %.2f predawn min" % (
            confidence, config.PREDAWN_MIN_CONFIDENCE)

    if edge < config.PREDAWN_MIN_EDGE:
        return False, "edge %.3f < %.3f predawn min" % (
            edge, config.PREDAWN_MIN_EDGE)

    if boundary_z is not None and boundary_z < config.PREDAWN_MIN_Z:
        return False, "boundary_z %.2f < %.2f predawn min" % (
            boundary_z, config.PREDAWN_MIN_Z)

    return True, "OK"


# --- Self-test ---------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print("=" * 60)
    print("CONFIDENCE SCORING TEST")
    print("=" * 60)

    # Test scenarios
    scenarios = [
        ("Fresh forecast, fresh obs, aligned, safe z",
         30, 15, 0.2, 2.5, 14),
        ("Fresh forecast, stale obs, misaligned, risky z",
         30, 200, 1.8, 0.5, 14),
        ("Stale forecast, no obs, pre-dawn",
         400, float("inf"), None, 1.2, 4),
        ("Everything perfect, evening",
         10, 5, 0.1, 3.0, 19),
        ("Everything bad",
         600, 500, 3.0, 0.2, 3),
    ]

    for name, fc_age, obs_age, res_ewma, bz, hour in scenarios:
        conf, gates = compute_confidence(fc_age, obs_age, res_ewma, bz, hour)
        min_edge = compute_dynamic_min_edge(hour, conf)

        print("\n%s:" % name)
        print("  Inputs: fc_age=%dmin, obs_age=%s, residual_ewma=%s, boundary_z=%.1f, hour=%d ET" % (
            fc_age,
            "%dmin" % obs_age if obs_age < float("inf") else "inf",
            "%.1f" % res_ewma if res_ewma is not None else "None",
            bz, hour))
        print("  Gates: %s" % {k: "%.2f" % v for k, v in gates.items()})
        print("  Confidence: %.3f" % conf)
        print("  Dynamic MIN_EDGE: %.1f%%" % (min_edge * 100))

        if hour < 6:
            ok, reason = passes_predawn_gates(conf, min_edge, bz)
            print("  Pre-dawn gate: %s (%s)" % (ok, reason))

    # Test boundary z
    print("\n\nBoundary Z test:")
    print("  μ=35.0, σ=1.2, boundaries=[33,34,36,37]")
    z = compute_boundary_z(35.0, 1.2, [33, 34, 36, 37])
    print("  z = %.2f (min distance = %.1f)" % (z, z * 1.2))

    print("\n" + "=" * 60)
