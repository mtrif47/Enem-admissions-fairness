"""
Task B5: full primary grid. K=10%, raw equal-weight composite, White vs
Black/Pardo contrast, feature blocks B1-B6 (Section 5.4) x logistic
regression and histogram gradient boosting (Section 5.5). Hyperparameters
tuned on validation via overlap@K; every reported metric comes from the
untouched test set.

Persists per-cell test-set model scores (results/main_grid_test_scores.parquet)
and the test-set benchmark/race labels (results/test_benchmark.parquet) so
Task B6 can bootstrap without refitting any model.
"""

import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ANALYTIC_SAMPLE_PARQUET, RESULTS_DIR, Logger, build_topk, TIE_BREAK_SEED
import modeling as mdl

SPLIT_PARQUET = RESULTS_DIR / "split_indices.parquet"
MAIN_GRID_OUT = RESULTS_DIR / "main_grid.csv"
TEST_SCORES_OUT = RESULTS_DIR / "main_grid_test_scores.parquet"
TEST_BENCHMARK_OUT = RESULTS_DIR / "test_benchmark.parquet"
K_PRIMARY = 0.10


def load_data():
    df = pd.read_parquet(ANALYTIC_SAMPLE_PARQUET)
    split = pd.read_parquet(SPLIT_PARQUET)[["NU_INSCRICAO", "split"]]
    df = df.merge(split, on="NU_INSCRICAO", how="inner")
    assert len(df) == len(split)
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
    status = "OK (~0)" if abs(zs) < 0.5 else "!!! NOT ZERO - CHECK !!!"
    log(f"    zero-sum = {zs:.6f}  [{status}]")


def run(logger: Logger):
    log = logger.log
    t0 = time.time()
    log("\n" + "=" * 70)
    log("TASK B5 - FULL PRIMARY GRID (K=10%, B1-B6 x {LogReg, HGB})")
    log("=" * 70)

    df = load_data()
    n = len(df)
    idx_train = np.flatnonzero(df["split"].to_numpy() == "train")
    idx_val = np.flatnonzero(df["split"].to_numpy() == "val")
    idx_test = np.flatnonzero(df["split"].to_numpy() == "test")
    composite = df["composite_raw_equal"].to_numpy()
    race = df["race_label"]

    y_train_bench, _ = build_topk(composite[idx_train], K=K_PRIMARY, seed=TIE_BREAK_SEED)
    y_val_bench, _ = build_topk(composite[idx_val], K=K_PRIMARY, seed=TIE_BREAK_SEED)
    y_test_bench, _ = build_topk(composite[idx_test], K=K_PRIMARY, seed=TIE_BREAK_SEED)

    log(f"\nN={n:,}  train={len(idx_train):,}  val={len(idx_val):,}  test={len(idx_test):,}")
    log(f"Benchmark positives: train={y_train_bench.sum():,}  val={y_val_bench.sum():,}  "
        f"test={y_test_bench.sum():,}")

    pd.DataFrame({
        "NU_INSCRICAO": df["NU_INSCRICAO"].to_numpy()[idx_test],
        "race_label": race.iloc[idx_test].to_numpy(),
        "benchmark_selected": y_test_bench,
    }).to_parquet(TEST_BENCHMARK_OUT, index=False)
    log(f"Saved {TEST_BENCHMARK_OUT}.")

    rows_out = []
    test_score_rows = []

    for block in mdl.ALL_BLOCKS:
        log("\n" + "-" * 70)
        log(f"Block {block}")
        log("-" * 70)
        try:
            X, feature_names = mdl.build_block_features(
                df, block, (df["split"] == "train").to_numpy(), te_target=y_train_bench
            )
            log(f"  Design matrix: {X.shape[1]} columns")

            # ---- Logistic regression ----
            t1 = time.time()
            log("  Logistic regression grid (C):")
            best_overlap, best_C, best_clf, best_scaler = mdl.tune_logreg(
                X[idx_train], y_train_bench, X[idx_val], y_val_bench, K_PRIMARY, log=log
            )
            test_scores_lr = best_clf.predict_proba(best_scaler.transform(X[idx_test]))[:, 1]
            results_lr = mdl.evaluate_selection(test_scores_lr, y_test_bench,
                                                 race.iloc[idx_test], K_PRIMARY)
            report_cell(log, f"{block} / logistic_regression (C={best_C}, "
                              f"{time.time()-t1:.0f}s)", results_lr)
            rows_out.append(dict(
                block=block, model="logistic_regression", K=K_PRIMARY,
                n_features=X.shape[1], best_params=str(dict(C=best_C)),
                val_overlap_at_k=best_overlap,
                overlap_at_k=results_lr["overlap_at_k"], jaccard=results_lr["jaccard"],
                pr_auc=results_lr["pr_auc"], roc_auc=results_lr["roc_auc"],
                rate_white_pct=results_lr["rate_white_pct"],
                rate_bp_pct=results_lr["rate_bp_pct"], gap_pp=results_lr["gap_pp"],
                ratio=results_lr["ratio"], bench_gap_pp=results_lr["bench_gap_pp"],
                zero_sum_value=results_lr["zero_sum_value"],
                white_p_g=results_lr["group_rows"][0]["p_g"],
                white_q_g=results_lr["group_rows"][0]["q_g"],
                white_net_realloc=results_lr["group_rows"][0]["net_reallocation"],
                white_fpr=results_lr["group_rows"][0]["fpr"],
                white_fnr=results_lr["group_rows"][0]["fnr"],
                bp_p_g=results_lr["group_rows"][1]["p_g"],
                bp_q_g=results_lr["group_rows"][1]["q_g"],
                bp_net_realloc=results_lr["group_rows"][1]["net_reallocation"],
                bp_fpr=results_lr["group_rows"][1]["fpr"],
                bp_fnr=results_lr["group_rows"][1]["fnr"],
            ))
            for i, ni in enumerate(df["NU_INSCRICAO"].to_numpy()[idx_test]):
                test_score_rows.append((ni, block, "logistic_regression", test_scores_lr[i]))
        except Exception:
            log(f"\nERROR in block {block} / logistic_regression:")
            log(traceback.format_exc())

        try:
            # ---- Histogram gradient boosting ----
            t1 = time.time()
            log(f"  HGB grid ({len(mdl.HGB_GRID)} combinations, parallel):")
            best_overlap_hgb, best_params, best_hgb = mdl.tune_hgb(
                X[idx_train], y_train_bench, X[idx_val], y_val_bench, K_PRIMARY, log=log
            )
            test_scores_hgb = best_hgb.predict_proba(X[idx_test])[:, 1]
            results_hgb = mdl.evaluate_selection(test_scores_hgb, y_test_bench,
                                                  race.iloc[idx_test], K_PRIMARY)
            report_cell(log, f"{block} / gradient_boosting (params={best_params}, "
                              f"{time.time()-t1:.0f}s)", results_hgb)
            rows_out.append(dict(
                block=block, model="gradient_boosting", K=K_PRIMARY,
                n_features=X.shape[1], best_params=str(best_params),
                val_overlap_at_k=best_overlap_hgb,
                overlap_at_k=results_hgb["overlap_at_k"], jaccard=results_hgb["jaccard"],
                pr_auc=results_hgb["pr_auc"], roc_auc=results_hgb["roc_auc"],
                rate_white_pct=results_hgb["rate_white_pct"],
                rate_bp_pct=results_hgb["rate_bp_pct"], gap_pp=results_hgb["gap_pp"],
                ratio=results_hgb["ratio"], bench_gap_pp=results_hgb["bench_gap_pp"],
                zero_sum_value=results_hgb["zero_sum_value"],
                white_p_g=results_hgb["group_rows"][0]["p_g"],
                white_q_g=results_hgb["group_rows"][0]["q_g"],
                white_net_realloc=results_hgb["group_rows"][0]["net_reallocation"],
                white_fpr=results_hgb["group_rows"][0]["fpr"],
                white_fnr=results_hgb["group_rows"][0]["fnr"],
                bp_p_g=results_hgb["group_rows"][1]["p_g"],
                bp_q_g=results_hgb["group_rows"][1]["q_g"],
                bp_net_realloc=results_hgb["group_rows"][1]["net_reallocation"],
                bp_fpr=results_hgb["group_rows"][1]["fpr"],
                bp_fnr=results_hgb["group_rows"][1]["fnr"],
            ))
            for i, ni in enumerate(df["NU_INSCRICAO"].to_numpy()[idx_test]):
                test_score_rows.append((ni, block, "gradient_boosting", test_scores_hgb[i]))
        except Exception:
            log(f"\nERROR in block {block} / gradient_boosting:")
            log(traceback.format_exc())

        # save incrementally after every block, in case of a later crash
        if rows_out:
            pd.DataFrame(rows_out).to_csv(MAIN_GRID_OUT, index=False)
        if test_score_rows:
            pd.DataFrame(test_score_rows,
                         columns=["NU_INSCRICAO", "block", "model", "score"]
                         ).to_parquet(TEST_SCORES_OUT, index=False)

    log(f"\nTotal B5 elapsed: {time.time()-t0:,.0f}s. Saved {MAIN_GRID_OUT} "
        f"({len(rows_out)} rows) and {TEST_SCORES_OUT}.")
    return len(rows_out) > 0


if __name__ == "__main__":
    logger = Logger(RESULTS_DIR / "taskB_log.txt", mode="a")
    try:
        run(logger)
    finally:
        logger.close()
