"""
Task E4: anchor comparison. For every cell in E1-E3, add the distance to
the Task B4b random top-K anchor (overlap@K=0.096, gap=-0.39pp), so
Section 6.5 can distinguish mitigation that reduces disparity by
improving fairness (overlap_above_anchor stays well above 0) from
mitigation that reduces it by becoming uninformative (overlap_above_anchor
collapses toward 0).
"""

import sys
import time
import traceback
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS_DIR, Logger

ANCHOR_OVERLAP = 0.096
ANCHOR_GAP_PP = -0.39

OUT_CSV = RESULTS_DIR / "mitig_anchor_comparison.csv"


def run(logger: Logger):
    log = logger.log
    t0 = time.time()
    log("\n" + "=" * 70)
    log("TASK E4 - ANCHOR COMPARISON (E1-E3 vs random top-K anchor)")
    log("=" * 70)
    log(f"\nAnchor (Task B4b): overlap@K={ANCHOR_OVERLAP}, gap={ANCHOR_GAP_PP}pp")

    frames = []
    try:
        e1 = pd.read_csv(RESULTS_DIR / "mitig_reweighing.csv")
        e1["source"] = "E1_reweighing"
        e1["level"] = 0
        e1["constraint_strength"] = pd.NA
        frames.append(e1)
    except Exception:
        log("\nERROR loading mitig_reweighing.csv:")
        log(traceback.format_exc())

    try:
        e2 = pd.read_csv(RESULTS_DIR / "mitig_threshold.csv")
        e2["source"] = "E2_threshold"
        e2["level"] = 10
        e2["constraint_strength"] = 1.0
        frames.append(e2)
    except Exception:
        log("\nERROR loading mitig_threshold.csv:")
        log(traceback.format_exc())

    try:
        e3 = pd.read_csv(RESULTS_DIR / "mitig_frontier.csv")
        e3["source"] = "E3_frontier"
        frames.append(e3)
    except Exception:
        log("\nERROR loading mitig_frontier.csv:")
        log(traceback.format_exc())

    if not frames:
        log("\nNo E1-E3 outputs found; nothing to do.")
        return False

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["overlap_above_anchor"] = combined["overlap_at_k"] - ANCHOR_OVERLAP
    combined["gap_above_anchor"] = combined["gap_pp"] - ANCHOR_GAP_PP

    log(f"\nCombined {len(combined)} rows across E1 ({len(frames[0]) if frames else 0}), "
        f"E2, E3.")
    log("\nSummary (min/median/max overlap_above_anchor and gap_above_anchor):")
    for src, g in combined.groupby("source"):
        log(f"  {src:<14s} overlap_above_anchor: min={g['overlap_above_anchor'].min():.4f} "
            f"median={g['overlap_above_anchor'].median():.4f} "
            f"max={g['overlap_above_anchor'].max():.4f}   "
            f"gap_above_anchor: min={g['gap_above_anchor'].min():+.2f}pp "
            f"median={g['gap_above_anchor'].median():+.2f}pp "
            f"max={g['gap_above_anchor'].max():+.2f}pp")

    near_anchor = combined[combined["overlap_above_anchor"] < 0.05]
    if len(near_anchor):
        log(f"\n  {len(near_anchor)} rows have overlap_above_anchor < 0.05 "
            f"(fidelity close to the uninformative random anchor -- any parity "
            f"gain there is suspect, per Section 6.5's requirement):")
        for _, r in near_anchor.iterrows():
            log(f"    {r['source']} block={r.get('block')} model={r.get('model')} "
                f"level={r.get('level')}  overlap_above_anchor={r['overlap_above_anchor']:.4f}  "
                f"gap_pp={r['gap_pp']:+.2f}")
    else:
        log("\n  No cell approaches the random anchor's fidelity -- all mitigations "
            "retain meaningfully more signal than an uninformative rule.")

    combined.to_csv(OUT_CSV, index=False)
    log(f"\nSaved {OUT_CSV} ({len(combined)} rows).")
    log(f"\nE4 total elapsed: {time.time()-t0:,.0f}s")
    return True


if __name__ == "__main__":
    logger = Logger(RESULTS_DIR / "taskE_log.txt", mode="a")
    try:
        run(logger)
    finally:
        logger.close()
