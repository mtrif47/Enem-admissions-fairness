"""
Task B4: smoke test on feature block B1, logistic regression only, plus
B4b: the random top-K anchor. Primary capacity K = 10%.

Design note on the benchmark inside a split: Section 5.1 requires the
predictive rule to select "the same capacity as the benchmark." Section
4.6's S^0 is defined once over the full national ranking (N = 123,879),
which does not divide evenly across an arbitrary train/val/test subset
(a stratified-by-decile split makes each split's benchmark-positive share
very close to K, but not exactly, since deciles are coarser than the K=10%
cutoff and ties intervene). To make the Section 5.2 zero-sum identity hold
exactly on the evaluation population being reported -- which is what
"same capacity" and the zero-sum check both require -- the benchmark is
recomputed via the identical top-K rule (Section 4.6 formula, same tie
seed) WITHIN each evaluation split (validation for tuning, test for final
metrics), rather than merely subset from the global Task A column. The
global, whole-sample benchmark from Task A/B1 remains the descriptive
object reported in Section 6.1; this is a separate, per-split device
needed only for the modelling comparison.
"""

import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ANALYTIC_SAMPLE_PARQUET, RESULTS_DIR, Logger, build_topk, TIE_BREAK_SEED
import modeling as mdl

SPLIT_PARQUET = RESULTS_DIR / "split_indices.parquet"
SMOKE_OUT = RESULTS_DIR / "smoke_test.csv"
K_PRIMARY = 0.10


def load_data():
    df = pd.read_parquet(ANALYTIC_SAMPLE_PARQUET)
    split = pd.read_parquet(SPLIT_PARQUET)[["NU_INSCRICAO", "split"]]
    df = df.merge(split, on="NU_INSCRICAO", how="inner")
    assert len(df) == len(split), "split merge dropped rows"
    return df


def report_cell(log, label, results):
    log(f"\n  [{label}]")
    log(f"    overlap@K={results['overlap_at_k']:.4f}  Jaccard={results['jaccard']:.4f}  "
        f"PR-AUC={results['pr_auc']:.4f}  ROC-AUC={results['roc_auc']:.4f}")
    log(f"    White rate={results['rate_white_pct']:.2f}%  "
        f"Black/Pardo rate={results['rate_bp_pct']:.2f}%  "
        f"gap={results['gap_pp']:.2f}pp  ratio={results['ratio']:.2f}  "
        f"(benchmark gap on this split={results['bench_gap_pp']:.2f}pp)")
    for row in results["group_rows"]:
        log(f"      {row['group']:<12s} n={row['n']:,}  p_g={row['p_g']:.4f}  "
            f"q_g={row['q_g']:.4f}  net_realloc={row['net_reallocation']:+.4f}  "
            f"FPR={row['fpr']:.4f}  FNR={row['fnr']:.4f}")
    zs = results["zero_sum_value"]
    status = "OK (~0)" if abs(zs) < 0.5 else "!!! NOT ZERO - CHECK IMPLEMENTATION !!!"
    log(f"    ZERO-SUM CHECK: sum_g n_g*(q_g - p_g) = {zs:.6f}  [{status}]")


def run(logger: Logger):
    log = logger.log
    log("\n" + "=" * 70)
    log("TASK B4 - SMOKE TEST (feature block B1, logistic regression) + B4b (random anchor)")
    log("=" * 70)

    rows_out = []

    try:
        df = load_data()
        log(f"\nLoaded analytic sample with split labels: N={len(df):,}")
        train_mask_all = (df["split"] == "train").to_numpy()

        X, feature_names = mdl.build_block_features(df, "B1", train_mask_all)
        log(f"Block B1 design matrix: {X.shape[1]} columns -> {feature_names}")

        idx_train = np.flatnonzero(df["split"].to_numpy() == "train")
        idx_val = np.flatnonzero(df["split"].to_numpy() == "val")
        idx_test = np.flatnonzero(df["split"].to_numpy() == "test")

        composite = df["composite_raw_equal"].to_numpy()
        race = df["race_label"]

        y_val_bench, _ = build_topk(composite[idx_val], K=K_PRIMARY, seed=TIE_BREAK_SEED)
        y_test_bench, _ = build_topk(composite[idx_test], K=K_PRIMARY, seed=TIE_BREAK_SEED)

        # y_train for fitting: benchmark computed within TRAIN split (the
        # model must learn to predict S^0, and only training-split labels
        # are used for fitting, consistent with "predict S^0 ... excluding
        # score components" in Section 5.3).
        y_train_bench, _ = build_topk(composite[idx_train], K=K_PRIMARY, seed=TIE_BREAK_SEED)

        log("\nTuning logistic regression (grid on C, metric = overlap@K on validation):")
        best_overlap, best_C, best_clf, best_scaler = mdl.tune_logreg(
            X[idx_train], y_train_bench, X[idx_val], y_val_bench, K_PRIMARY, log=log
        )
        log(f"  Selected C={best_C} (validation overlap@K={best_overlap:.4f})")

        test_scores = best_clf.predict_proba(best_scaler.transform(X[idx_test]))[:, 1]
        results = mdl.evaluate_selection(test_scores, y_test_bench, race.iloc[idx_test],
                                          K_PRIMARY)
        report_cell(log, "B4: Block B1, Logistic Regression, K=10%, TEST", results)

        rows_out.append(dict(
            step="B4", block="B1", model="logistic_regression", K=K_PRIMARY,
            best_C=best_C, val_overlap_at_k=best_overlap,
            overlap_at_k=results["overlap_at_k"], jaccard=results["jaccard"],
            pr_auc=results["pr_auc"], roc_auc=results["roc_auc"],
            rate_white_pct=results["rate_white_pct"], rate_bp_pct=results["rate_bp_pct"],
            gap_pp=results["gap_pp"], ratio=results["ratio"],
            zero_sum_value=results["zero_sum_value"], n_test=len(idx_test),
        ))
    except Exception:
        log("\nERROR in B4 (logistic regression smoke test):")
        log(traceback.format_exc())

    # ------------------------------------------------------------------
    # B4b: random top-K anchor
    # ------------------------------------------------------------------
    try:
        df = load_data() if "df" not in dir() else df
        idx_test = np.flatnonzero(df["split"].to_numpy() == "test")
        composite = df["composite_raw_equal"].to_numpy()
        race = df["race_label"]
        y_test_bench, _ = build_topk(composite[idx_test], K=K_PRIMARY, seed=TIE_BREAK_SEED)

        rng = np.random.default_rng(TIE_BREAK_SEED)
        random_scores = rng.random(len(idx_test))
        results_rand = mdl.evaluate_selection(random_scores, y_test_bench,
                                               race.iloc[idx_test], K_PRIMARY)
        report_cell(log, "B4b: Random top-K anchor, K=10%, TEST", results_rand)
        log("  (Expected: near-zero gap and near-zero fidelity, per Section 5.5.)")

        rows_out.append(dict(
            step="B4b", block="anchor", model="random", K=K_PRIMARY,
            best_C=np.nan, val_overlap_at_k=np.nan,
            overlap_at_k=results_rand["overlap_at_k"], jaccard=results_rand["jaccard"],
            pr_auc=results_rand["pr_auc"], roc_auc=results_rand["roc_auc"],
            rate_white_pct=results_rand["rate_white_pct"],
            rate_bp_pct=results_rand["rate_bp_pct"],
            gap_pp=results_rand["gap_pp"], ratio=results_rand["ratio"],
            zero_sum_value=results_rand["zero_sum_value"], n_test=len(idx_test),
        ))
    except Exception:
        log("\nERROR in B4b (random anchor):")
        log(traceback.format_exc())

    if rows_out:
        pd.DataFrame(rows_out).to_csv(SMOKE_OUT, index=False)
        log(f"\nSaved {SMOKE_OUT} ({len(rows_out)} rows).")

    return len(rows_out) > 0


if __name__ == "__main__":
    logger = Logger(RESULTS_DIR / "taskB_log.txt", mode="a")
    try:
        run(logger)
    finally:
        logger.close()
