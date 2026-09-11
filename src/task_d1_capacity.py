"""
Task D1: capacity robustness. Repeat the B1-B6 x {LR, HGB} grid at
K=5% and K=20%, reusing the Section 5.5 hyperparameters tuned once at
K=10% (see robust_common.py's module docstring for why).

The model itself must still be REFIT at each K (with those fixed
hyperparameters): the prediction target S^0 is capacity-specific by
definition (Section 4.6), so "predict S^0 at K=5%" is a different label
from "predict S^0 at K=10%" even though the learner and its
hyperparameters do not change.

Per the task's note, K=5% cuts through the middle of the top composite
decile (deciles are 10%-wide bins used for the Task B3 stratified split),
so the test split's local top-5% need not coincide with which test rows
fall in the GLOBAL top-5% of the whole analytic sample. K=20% coincides
with a decile boundary (deciles 8+9) so no such divergence is expected
there. Both definitions are reported for both K's so the K=20% numbers
serve as an internal check that divergence really is a K=5%-specific
artefact of decile-vs-K granularity, not a bug.
  (a) "local"  -- rank and threshold WITHIN the test split (the
       convention used everywhere in Task B; guarantees equal, fixed
       capacity for benchmark and model on the population being reported).
  (b) "global" -- rank and threshold over the WHOLE analytic sample
       (train+val+test), then look at which test rows fall in that
       global top-K. Capacity need not come out exactly equal for the
       benchmark and the model under this definition.
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

SPLIT_PARQUET = RESULTS_DIR / "split_indices.parquet"
OUT_CSV = RESULTS_DIR / "robust_capacity.csv"
K_LIST = [0.05, 0.20]


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
    log("TASK D1 - CAPACITY ROBUSTNESS (K=5%, K=20%, hyperparameters reused from K=10%)")
    log("=" * 70)

    df = load_data()
    train_mask_all = (df["split"] == "train").to_numpy()
    idx_train = np.flatnonzero(train_mask_all)
    idx_val = np.flatnonzero(df["split"].to_numpy() == "val")
    idx_test = np.flatnonzero(df["split"].to_numpy() == "test")
    composite = df["composite_raw_equal"].to_numpy()
    race = df["race_label"]
    is_white = (race == "White").to_numpy()
    is_bp = race.isin(["Black", "Pardo"]).to_numpy()
    ni = df["NU_INSCRICAO"].to_numpy()

    params_by_block = rc.load_primary_params()
    log(f"\nLoaded primary hyperparameters for {len(params_by_block)} blocks from "
        f"{rc.MAIN_GRID_CSV}.")

    rows_out = []

    for K in K_LIST:
        y_train_bench, _ = build_topk(composite[idx_train], K=K, seed=TIE_BREAK_SEED)
        y_test_bench_local, _ = build_topk(composite[idx_test], K=K, seed=TIE_BREAK_SEED)
        bench_selected_global_full, _ = build_topk(composite, K=K, seed=TIE_BREAK_SEED)
        bench_selected_global_test = bench_selected_global_full[idx_test]
        k_local = int(np.floor(K * len(idx_test)))

        log(f"\n{'='*20} K = {int(K*100)}% {'='*20}")
        log(f"  local k (floor(K*N_test)) = {k_local:,}   "
            f"global-in-test bench count = {int(bench_selected_global_test.sum()):,}")

        for block in mdl.ALL_BLOCKS:
            try:
                X, _ = mdl.build_block_features(df, block, train_mask_all, te_target=y_train_bench)
            except Exception:
                log(f"\nERROR building features for block {block} at K={K}:")
                log(traceback.format_exc())
                continue

            for model in ["logistic_regression", "gradient_boosting"]:
                try:
                    params = params_by_block[block][model]
                    clf, scaler = rc.fit_cell(X[idx_train], y_train_bench, model, params)
                    scores_all = rc.score_cell(clf, scaler, X)

                    model_selected_local, _ = build_topk(scores_all[idx_test], K=K,
                                                          seed=TIE_BREAK_SEED)
                    model_selected_global_full, _ = build_topk(scores_all, K=K,
                                                                seed=TIE_BREAK_SEED)
                    model_selected_global_test = model_selected_global_full[idx_test]

                    for definition, m_sel, b_sel in [
                        ("local", model_selected_local, y_test_bench_local),
                        ("global", model_selected_global_test, bench_selected_global_test),
                    ]:
                        gm = rc.group_metrics(m_sel, b_sel, is_white[idx_test], is_bp[idx_test])
                        ov = rc.overlap_at_k(m_sel, b_sel, k_local)
                        row = dict(block=block, model=model, K=K, definition=definition,
                                   n_test_selected_model=int(m_sel.sum()),
                                   n_test_selected_bench=int(b_sel.sum()),
                                   k_local=k_local, overlap_at_k=ov, **gm)
                        rows_out.append(row)

                        rc.record_scores("D1", f"K{int(K*100)}_{definition}", block, model, K,
                                          "White", "Black_Pardo", ni[idx_test], m_sel,
                                          b_sel, np.where(is_white[idx_test], "White",
                                                          np.where(is_bp[idx_test], "Black_Pardo", "other")))

                    gap_local = [r for r in rows_out if r["block"] == block and r["model"] == model
                                 and r["K"] == K and r["definition"] == "local"][-1]["gap_pp"]
                    gap_global = [r for r in rows_out if r["block"] == block and r["model"] == model
                                  and r["K"] == K and r["definition"] == "global"][-1]["gap_pp"]
                    diverg = gap_global - gap_local
                    flag = "  !!! DIVERGENCE > 1pp !!!" if abs(diverg) > 1.0 else ""
                    log(f"  [{block}/{model}] gap_local={gap_local:+.2f}pp  "
                        f"gap_global={gap_global:+.2f}pp  diff={diverg:+.2f}pp{flag}")
                except Exception:
                    log(f"\nERROR fitting/evaluating {block}/{model} at K={K}:")
                    log(traceback.format_exc())

    if rows_out:
        pd.DataFrame(rows_out).to_csv(OUT_CSV, index=False)
        log(f"\nSaved {OUT_CSV} ({len(rows_out)} rows).")
    n_scores = rc.flush_scores()
    log(f"Appended score records for bootstrap; robust_all_scores.parquet now has "
        f"{n_scores} rows." if n_scores else "No score records flushed.")

    log(f"\nD1 total elapsed: {time.time()-t0:,.0f}s")
    return len(rows_out) > 0


if __name__ == "__main__":
    logger = Logger(RESULTS_DIR / "taskD_log.txt", mode="a")
    try:
        run(logger)
    finally:
        logger.close()
