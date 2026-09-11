"""
Task E2: capacity-constrained equal-opportunity threshold adjustment
(post-processing, Hardt et al. 2016), blocks B3/B4, both models, K=10%.

Refits the UNMITIGATED primary model (same hyperparameters, features and
target as results/main_grid.csv) to obtain validation-set scores, which
Task B5 never persisted to disk -- test-set scores are verified to match
main_grid_test_scores.parquet exactly as a consistency check, confirming
this reproduces the primary fit exactly rather than a different model.

See mitig_common.py's module docstring for the full design rationale:
capacity is held at floor(K*N_test) (matching the primary grid exactly),
Asian/Indigenous candidates keep their unmitigated selection decision
unchanged, and the two White/Black-Pardo thresholds are fit on validation
then transferred to test via rank-based counts so the realised total is
exactly floor(K*N_test) by construction, not by chance.
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
import robust_common as rc
import mitig_common as mc

SPLIT_PARQUET = RESULTS_DIR / "split_indices.parquet"
TEST_SCORES_PARQUET = RESULTS_DIR / "main_grid_test_scores.parquet"
OUT_CSV = RESULTS_DIR / "mitig_threshold.csv"
K_PRIMARY = 0.10
BLOCKS = ["B3", "B4"]


def load_data():
    df = pd.read_parquet(ANALYTIC_SAMPLE_PARQUET)
    split = pd.read_parquet(SPLIT_PARQUET)[["NU_INSCRICAO", "split"]]
    df = df.merge(split, on="NU_INSCRICAO", how="inner")
    assert len(df) == len(split)
    return df


def refit_primary(df, block, model, train_mask_all, y_train_bench, params):
    X, _ = mdl.build_block_features(df, block, train_mask_all, te_target=y_train_bench)
    clf, scaler = rc.fit_cell(X[train_mask_all], y_train_bench, model, params)
    return X, clf, scaler


def run_cell(df, block, model, params, idx_train, idx_val, idx_test, y_train_bench,
             y_val_bench, y_test_bench, race, ni, log, primary_test_scores_check):
    train_mask_all = (df["split"] == "train").to_numpy()
    X, clf, scaler = refit_primary(df, block, model, train_mask_all, y_train_bench, params)
    val_scores = rc.score_cell(clf, scaler, X[idx_val])
    test_scores = rc.score_cell(clf, scaler, X[idx_test])

    # consistency check against the saved primary test scores
    ref = primary_test_scores_check[(primary_test_scores_check["block"] == block)
                                     & (primary_test_scores_check["model"] == model)]
    ref = ref.set_index("NU_INSCRICAO").loc[ni[idx_test], "score"].to_numpy()
    max_diff = np.max(np.abs(ref - test_scores))
    log(f"  [{block}/{model}] refit consistency vs main_grid_test_scores.parquet: "
        f"max|diff|={max_diff:.2e} {'OK' if max_diff < 1e-6 else '!!! MISMATCH !!!'}")

    is_white = (race == "White").to_numpy()
    is_bp = race.isin(["Black", "Pardo"]).to_numpy()
    is_ai = ~is_white & ~is_bp  # Asian + Indigenous

    unmit_val_sel, _ = build_topk(val_scores, K=K_PRIMARY, seed=TIE_BREAK_SEED)
    unmit_test_sel, _ = build_topk(test_scores, K=K_PRIMARY, seed=TIE_BREAK_SEED)

    n_ai_val = int(unmit_val_sel[is_ai[idx_val]].sum())
    n_ai_test = int(unmit_test_sel[is_ai[idx_test]].sum())
    capacity_val = int(np.floor(K_PRIMARY * len(idx_val)))
    capacity_test = int(np.floor(K_PRIMARY * len(idx_test)))
    capacity_remaining_val = capacity_val - n_ai_val
    capacity_remaining_test = capacity_test - n_ai_test

    scores_white_val = val_scores[is_white[idx_val]]
    bench_white_val = y_val_bench[is_white[idx_val]]
    scores_bp_val = val_scores[is_bp[idx_val]]
    bench_bp_val = y_val_bench[is_bp[idx_val]]

    split_info = mc.search_threshold_split(scores_white_val, bench_white_val,
                                            scores_bp_val, bench_bp_val,
                                            capacity_remaining_val, target_gap=0.0)

    scores_white_test = test_scores[is_white[idx_test]]
    scores_bp_test = test_scores[is_bp[idx_test]]
    sel_white_test, sel_bp_test, n_white_test, n_bp_test = mc.apply_split_to_test(
        scores_white_test, scores_bp_test, split_info["rate_white"], split_info["rate_bp"],
        capacity_remaining_test,
    )

    model_selected = np.zeros(len(idx_test), dtype=bool)
    model_selected[is_white[idx_test]] = sel_white_test
    model_selected[is_bp[idx_test]] = sel_bp_test
    model_selected[is_ai[idx_test]] = unmit_test_sel[is_ai[idx_test]]  # held fixed

    n_total = int(model_selected.sum())
    target = capacity_test
    cap_status = "OK" if n_total == target else "!!! CAPACITY MISMATCH !!!"
    log(f"    capacity: realised={n_total:,}  target=floor(K*N_test)={target:,}  [{cap_status}]")
    log(f"    val split: n_white={split_info['n_white']} n_bp={split_info['n_bp']}  "
        f"TPR_white={split_info['tpr_white']:.4f} TPR_bp={split_info['tpr_bp']:.4f}  "
        f"tau_white={split_info['tau_white']:.4f} tau_bp={split_info['tau_bp']:.4f}")
    log(f"    test: n_white_selected={n_white_test} n_bp_selected={n_bp_test} "
        f"n_ai_held_fixed={n_ai_test}")

    res = mc.evaluate_from_selected(model_selected, y_test_bench, race.iloc[idx_test],
                                     K_PRIMARY, model_scores=test_scores)
    return res, split_info, n_total, target, model_selected


def run(logger: Logger):
    log = logger.log
    t0 = time.time()
    log("\n" + "=" * 70)
    log("TASK E2 - CAPACITY-CONSTRAINED EQUAL-OPPORTUNITY THRESHOLDS (B3/B4, K=10%)")
    log("=" * 70)

    df = load_data()
    idx_train = np.flatnonzero((df["split"] == "train").to_numpy())
    idx_val = np.flatnonzero((df["split"] == "val").to_numpy())
    idx_test = np.flatnonzero((df["split"] == "test").to_numpy())
    composite = df["composite_raw_equal"].to_numpy()
    race = df["race_label"]
    ni = df["NU_INSCRICAO"].to_numpy()

    y_train_bench, _ = build_topk(composite[idx_train], K=K_PRIMARY, seed=TIE_BREAK_SEED)
    y_val_bench, _ = build_topk(composite[idx_val], K=K_PRIMARY, seed=TIE_BREAK_SEED)
    y_test_bench, _ = build_topk(composite[idx_test], K=K_PRIMARY, seed=TIE_BREAK_SEED)

    params_by_block = rc.load_primary_params()
    primary_test_scores = pd.read_parquet(TEST_SCORES_PARQUET)

    rows_out = []
    for block in BLOCKS:
        for model in ["logistic_regression", "gradient_boosting"]:
            try:
                params = params_by_block[block][model]
                res, split_info, n_total, target, model_selected = run_cell(
                    df, block, model, params, idx_train, idx_val, idx_test,
                    y_train_bench, y_val_bench, y_test_bench, race, ni, log, primary_test_scores,
                )
                row = dict(block=block, model=model, K=K_PRIMARY, mitigation="threshold_eq_opp",
                           overlap_at_k=res["overlap_at_k"], jaccard=res["jaccard"],
                           pr_auc=res["pr_auc"], roc_auc=res["roc_auc"],
                           rate_white_pct=res["rate_white_pct"], rate_bp_pct=res["rate_bp_pct"],
                           gap_pp=res["gap_pp"], ratio=res["ratio"],
                           bench_gap_pp=res["bench_gap_pp"], D=res["D"],
                           zero_sum_value=res["zero_sum_value"],
                           tpr_white=res["tpr_white"], tpr_bp=res["tpr_bp"],
                           tpr_gap=res["tpr_gap"],
                           tau_white=split_info["tau_white"], tau_bp=split_info["tau_bp"],
                           n_realised=n_total, n_target=target,
                           white_fpr=res["group_rows"][0]["fpr"],
                           white_fnr=res["group_rows"][0]["fnr"],
                           bp_fpr=res["group_rows"][1]["fpr"],
                           bp_fnr=res["group_rows"][1]["fnr"],
                           white_net_realloc=res["group_rows"][0]["net_reallocation"],
                           bp_net_realloc=res["group_rows"][1]["net_reallocation"])
                rows_out.append(row)

                mc.record_scores("E2", "threshold_eq_opp", block, model, K_PRIMARY,
                                  "White", "Black_Pardo", ni[idx_test], model_selected,
                                  y_test_bench,
                                  np.where((race.iloc[idx_test] == "White").to_numpy(), "White",
                                           np.where(race.iloc[idx_test].isin(["Black", "Pardo"]).to_numpy(),
                                                    "Black_Pardo", "other")))

                zs_status = "OK (~0)" if abs(res["zero_sum_value"]) < 0.5 else "!!! NOT ZERO !!!"
                log(f"    overlap@K={res['overlap_at_k']:.4f}  gap={res['gap_pp']:+.2f}pp  "
                    f"ratio={res['ratio']:.2f}  D={res['D']:+.2f}pp  "
                    f"zero-sum={res['zero_sum_value']:.6f} [{zs_status}]\n")
            except Exception:
                log(f"\nERROR fitting E2 {block}/{model}:")
                log(traceback.format_exc())

    if rows_out:
        pd.DataFrame(rows_out).to_csv(OUT_CSV, index=False)
        log(f"\nSaved {OUT_CSV} ({len(rows_out)} rows).")
    n_scores = mc.flush_scores()
    log(f"mitig_all_scores.parquet now has {n_scores} rows." if n_scores else "No scores flushed.")

    log(f"\nE2 total elapsed: {time.time()-t0:,.0f}s")
    return len(rows_out) > 0


if __name__ == "__main__":
    logger = Logger(RESULTS_DIR / "taskE_log.txt", mode="a")
    try:
        run(logger)
    finally:
        logger.close()
