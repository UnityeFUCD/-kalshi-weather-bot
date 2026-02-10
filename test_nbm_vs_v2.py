"""
test_nbm_vs_v2.py -- Settled-data comparison of V2 vs NBM shadow model.

Focus:
- mu MAE against settlement Tmax
- +/-1 sigma calibration coverage
- best-edge disagreement winner
- NBM bucket probability sum quality
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import config
from cli_scraper import reconcile_all_shadow_records


def _read_jsonl(path: Path):
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _metric_summary(rows):
    settled = [r for r in rows if r.get("actual_tmax") is not None]
    if not settled:
        return None

    v2_err = []
    nbm_err = []
    v2_1s_hits = 0
    nbm_1s_hits = 0
    v2_1s_n = 0
    nbm_1s_n = 0
    disagreement_n = 0
    v2_disagree_wins = 0
    nbm_disagree_wins = 0
    nbm_prob_sums = []

    for row in settled:
        actual = float(row["actual_tmax"])
        v2_mu = row.get("v2_mu")
        nbm_mu = row.get("nbm_mu")
        v2_sigma = row.get("v2_sigma")
        nbm_sigma = row.get("nbm_sigma")

        if v2_mu is not None:
            v2_err.append(abs(float(v2_mu) - actual))
        if nbm_mu is not None:
            nbm_err.append(abs(float(nbm_mu) - actual))

        if v2_mu is not None and v2_sigma is not None and float(v2_sigma) > 0:
            v2_1s_n += 1
            if abs(float(v2_mu) - actual) <= float(v2_sigma):
                v2_1s_hits += 1
        if nbm_mu is not None and nbm_sigma is not None and float(nbm_sigma) > 0:
            nbm_1s_n += 1
            if abs(float(nbm_mu) - actual) <= float(nbm_sigma):
                nbm_1s_hits += 1

        nbm_probs = row.get("nbm_bucket_probs") or {}
        if nbm_probs:
            nbm_prob_sums.append(sum(float(v) for v in nbm_probs.values()))

        v2_edge = row.get("v2_best_edge") or {}
        nbm_edge = row.get("nbm_best_edge") or {}
        if v2_edge and nbm_edge and (
            v2_edge.get("bucket") != nbm_edge.get("bucket")
            or v2_edge.get("side") != nbm_edge.get("side")
        ):
            v2_hit = row.get("v2_signal_correct")
            nbm_hit = row.get("nbm_signal_correct")
            if v2_hit is None or nbm_hit is None:
                continue
            disagreement_n += 1
            if v2_hit and not nbm_hit:
                v2_disagree_wins += 1
            elif nbm_hit and not v2_hit:
                nbm_disagree_wins += 1

    v2_mae = (sum(v2_err) / len(v2_err)) if v2_err else None
    nbm_mae = (sum(nbm_err) / len(nbm_err)) if nbm_err else None
    v2_cov = (v2_1s_hits / v2_1s_n) if v2_1s_n else None
    nbm_cov = (nbm_1s_hits / nbm_1s_n) if nbm_1s_n else None
    nbm_prob_sum_rate = None
    if nbm_prob_sums:
        nbm_prob_sum_rate = sum(1 for s in nbm_prob_sums if 0.98 <= s <= 1.02) / len(nbm_prob_sums)

    criteria = []
    if v2_mae is not None and nbm_mae is not None:
        criteria.append(("mu_accuracy", nbm_mae <= v2_mae))
    if v2_cov is not None and nbm_cov is not None:
        criteria.append(("sigma_calibration", abs(nbm_cov - 0.68) <= abs(v2_cov - 0.68)))
    if disagreement_n > 0:
        criteria.append(("edge_quality", nbm_disagree_wins >= v2_disagree_wins))
    if nbm_prob_sum_rate is not None:
        criteria.append(("probability_sum", nbm_prob_sum_rate >= 0.80))
    nbm_wins = sum(1 for _, ok in criteria if ok)

    return {
        "settled_rows": len(settled),
        "v2_mu_mae": v2_mae,
        "nbm_mu_mae": nbm_mae,
        "v2_sigma_coverage_1s": v2_cov,
        "nbm_sigma_coverage_1s": nbm_cov,
        "disagreement_rows_scored": disagreement_n,
        "v2_disagree_wins": v2_disagree_wins,
        "nbm_disagree_wins": nbm_disagree_wins,
        "nbm_prob_sum_in_range_rate": nbm_prob_sum_rate,
        "criteria": criteria,
        "nbm_wins_criteria": nbm_wins,
        "recommendation": "switch_to_nbm" if nbm_wins >= 3 else "keep_v2",
    }


def _fmt(value, digits=4):
    if value is None:
        return "N/A"
    return ("%0." + str(digits) + "f") % value


def main():
    parser = argparse.ArgumentParser(description="Compare NBM shadow vs V2 on settled outcomes")
    parser.add_argument(
        "--skip-reconcile",
        action="store_true",
        help="Do not run reconciliation before computing metrics",
    )
    args = parser.parse_args()

    if not args.skip_reconcile:
        reconcile_all_shadow_records(run_paper_reconcile=False)

    rows = _read_jsonl(config.NBM_SHADOW_PATH)
    summary = _metric_summary(rows)
    if summary is None:
        print("No settled NBM shadow rows yet.")
        return

    print("=" * 72)
    print("NBM vs V2 SETTLED COMPARISON")
    print("=" * 72)
    print("Settled rows: %d" % summary["settled_rows"])
    print("V2  mu MAE:   %s F" % _fmt(summary["v2_mu_mae"], 3))
    print("NBM mu MAE:   %s F" % _fmt(summary["nbm_mu_mae"], 3))
    print("V2  +/-1s:    %s" % _fmt(summary["v2_sigma_coverage_1s"], 3))
    print("NBM +/-1s:    %s" % _fmt(summary["nbm_sigma_coverage_1s"], 3))
    print("Disagree rows: %d (v2 wins=%d, nbm wins=%d)" % (
        summary["disagreement_rows_scored"],
        summary["v2_disagree_wins"],
        summary["nbm_disagree_wins"],
    ))
    print("NBM prob-sum in [0.98,1.02]: %s" % _fmt(summary["nbm_prob_sum_in_range_rate"], 3))
    print("")
    print("Criteria:")
    for name, ok in summary["criteria"]:
        print("  %-18s %s" % (name, "NBM_WIN" if ok else "V2_WIN"))
    print("")
    print("NBM wins criteria: %d" % summary["nbm_wins_criteria"])
    print("Recommendation: %s" % summary["recommendation"])

    out_path = config.PROJECT_ROOT / "reports" / "nbm_vs_v2_analysis.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print("Saved: %s" % out_path)


if __name__ == "__main__":
    main()
