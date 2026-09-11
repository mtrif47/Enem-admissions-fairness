"""
Task E1: reweighing (Kamiran & Calders 2012), pre-processing mitigation.
Blocks B3/B4, both models, K=10%, on top of the primary B5 fits. Compute
per-instance weights on TRAINING so group and benchmark label are
statistically independent in the reweighted distribution (see
mitig_common.py for the multi-group generalisation used), refit with
those weights (same reused hyperparameters as everywhere else in Task D),
apply the standard local top-K rule at K=10% on TEST.
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
OUT_CSV = RESULTS_DIR / "mitig_reweighing.csv"
K_PRIMARY = 0.10
BLOCKS = ["B3", "B4"]


def load_data():
    df = pd.read_parquet(ANALYTIC_SAMPLE_PARQUET)
    split = pd.read_parquet(SPLIT_PARQUET)[["NU_INSCRICAO", "split"]]
    df = df.merge(split, on="NU_INSCRICAO", how="inner")
    assert len(df) == len(split)
    return df


def run(logger: Logger):
    log = logger.log
    t0 = time.time()
    log("\n" + "=" * 70)
    log("TASK E1 - REWEIGHING (blocks B3/B4, K=10%)")
    log("=" * 70)

    df = load_data()
    train_mask_all = (df["split"] == "train").to_numpy()
    idx_train = np.flatnonzero(train_mask_all)
    idx_test = np.flatnonzero(df["split"].to_numpy() == "test")
    composite = df["composite_raw_equal"].to_numpy()
    race = df["race_label"]

    y_train_bench, _ = build_topk(composite[idx_train], K=K_PRIMARY, seed=TIE_BREAK_SEED)
    y_test_bench, _ = build_topk(composite[idx_test], K=K_PRIMARY, seed=TIE_BREAK_SEED)

    weights = mc.kamiran_calders_weights(race.iloc[idx_train].to_numpy(), y_train_bench)
    log(f"\nReweighing weights on train (N={len(idx_train):,}): "
        f"min={weights.min():.3f} max={weights.max():.3f} mean={weights.mean():.3f}")
    wtab = pd.DataFrame({"race": race.iloc[idx_train].to_numpy(), "y": y_train_bench,
                          "w": weights}).groupby(["race", "y"])["w"].first()
    for (g, y), w in wtab.items():
        log(f"  weight(group={g}, y={y}) = {w:.4f}")

    params_by_block = rc.load_primary_params()
    rows_out = []
    for block in BLOCKS:
        try:
            X, _ = mdl.build_block_features(df, block, train_mask_all, te_target=y_train_bench)
        except Exception:
            log(f"\nERROR building E1 features for {block}:")
            log(traceback.format_exc())
            continue
        for model in ["logistic_regression", "gradient_boosting"]:
            try:
                params = params_by_block[block][model]
                clf, scaler = mc.fit_cell_weighted(X[idx_train], y_train_bench, model, params,
                                                    sample_weight=weights)
                test_scores = rc.score_cell(clf, scaler, X[idx_test])
                model_selected, _ = build_topk(test_scores, K=K_PRIMARY, seed=TIE_BREAK_SEED)

                res = mc.evaluate_from_selected(model_selected, y_test_bench, race.iloc[idx_test],
                                                 K_PRIMARY, model_scores=test_scores)
                row = dict(block=block, model=model, K=K_PRIMARY, mitigation="reweighing",
                           overlap_at_k=res["overlap_at_k"], jaccard=res["jaccard"],
                           pr_auc=res["pr_auc"], roc_auc=res["roc_auc"],
                           rate_white_pct=res["rate_white_pct"], rate_bp_pct=res["rate_bp_pct"],
                           gap_pp=res["gap_pp"], ratio=res["ratio"],
                           bench_gap_pp=res["bench_gap_pp"], D=res["D"],
                           zero_sum_value=res["zero_sum_value"],
                           white_p_g=res["group_rows"][0]["p_g"],
                           white_q_g=res["group_rows"][0]["q_g"],
                           white_net_realloc=res["group_rows"][0]["net_reallocation"],
                           white_fpr=res["group_rows"][0]["fpr"],
                           white_fnr=res["group_rows"][0]["fnr"],
                           bp_p_g=res["group_rows"][1]["p_g"],
                           bp_q_g=res["group_rows"][1]["q_g"],
                           bp_net_realloc=res["group_rows"][1]["net_reallocation"],
                           bp_fpr=res["group_rows"][1]["fpr"],
                           bp_fnr=res["group_rows"][1]["fnr"])
                rows_out.append(row)

                mc.record_scores("E1", "reweighing", block, model, K_PRIMARY,
                                  "White", "Black_Pardo", df["NU_INSCRICAO"].to_numpy()[idx_test],
                                  model_selected, y_test_bench,
                                  np.where((race.iloc[idx_test] == "White").to_numpy(), "White",
                                           np.where(race.iloc[idx_test].isin(["Black", "Pardo"]).to_numpy(),
                                                    "Black_Pardo", "other")))

                zs_status = "OK (~0)" if abs(res["zero_sum_value"]) < 0.5 else "!!! NOT ZERO !!!"
                log(f"\n  [{block}/{model}] overlap@K={res['overlap_at_k']:.4f}  "
                    f"gap={res['gap_pp']:+.2f}pp  ratio={res['ratio']:.2f}  D={res['D']:+.2f}pp")
                log(f"    zero-sum={res['zero_sum_value']:.6f}  [{zs_status}]")
                for gr in res["group_rows"]:
                    log(f"    {gr['group']:<12s} FPR={gr['fpr']:.4f}  FNR={gr['fnr']:.4f}  "
                        f"net_realloc={gr['net_reallocation']:+.4f}")
            except Exception:
                log(f"\nERROR fitting E1 {block}/{model}:")
                log(traceback.format_exc())

    if rows_out:
        pd.DataFrame(rows_out).to_csv(OUT_CSV, index=False)
        log(f"\nSaved {OUT_CSV} ({len(rows_out)} rows).")
    n_scores = mc.flush_scores()
    log(f"mitig_all_scores.parquet now has {n_scores} rows." if n_scores else "No scores flushed.")

    log(f"\nE1 total elapsed: {time.time()-t0:,.0f}s")
    return len(rows_out) > 0


if __name__ == "__main__":
    logger = Logger(RESULTS_DIR / "taskE_log.txt", mode="a")
    try:
        run(logger)
    finally:
        logger.close()
