"""
Task D5: ranking-population robustness. Repeat B3/B4, both models, K=10%,
with training AND ranking restricted to White and Black/Pardo candidates
only (Asian and Indigenous dropped from the population entirely, not just
from the contrast). Reuses the Task B3 split by subsetting it to White/BP
rows (their original split labels are kept exactly; Asian/Indigenous rows
are simply excluded) -- no redraw. The benchmark S^0 is recomputed on this
smaller ranking population (N=120,237), since Section 4.4 defines the
ranking population as part of what the top-K rank is taken over.
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
OUT_CSV = RESULTS_DIR / "robust_population.csv"
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
    log("TASK D5 - RANKING POPULATION: WHITE + BLACK/PARDO ONLY (B3/B4, K=10%)")
    log("=" * 70)

    df_full = load_data()
    df = df_full[df_full["race_label"].isin(["White", "Black", "Pardo"])].reset_index(drop=True)
    log(f"\nRestricted population: N={len(df):,} "
        f"(full analytic sample was {len(df_full):,}; "
        f"dropped {len(df_full)-len(df):,} Asian/Indigenous rows).")
    log(f"  Split sizes here: "
        f"{df['split'].value_counts().to_dict()}")

    train_mask_all = (df["split"] == "train").to_numpy()
    idx_train = np.flatnonzero(train_mask_all)
    idx_test = np.flatnonzero(df["split"].to_numpy() == "test")
    composite = df["composite_raw_equal"].to_numpy()
    race = df["race_label"]
    is_white = (race == "White").to_numpy()
    is_bp = race.isin(["Black", "Pardo"]).to_numpy()
    ni = df["NU_INSCRICAO"].to_numpy()

    y_train_bench, _ = build_topk(composite[idx_train], K=K_PRIMARY, seed=TIE_BREAK_SEED)
    y_test_bench, _ = build_topk(composite[idx_test], K=K_PRIMARY, seed=TIE_BREAK_SEED)
    log(f"  Benchmark recomputed on this population: train pos={int(y_train_bench.sum()):,}, "
        f"test pos={int(y_test_bench.sum()):,}")

    params_by_block = rc.load_primary_params()
    rows_out = []
    for block in BLOCKS:
        try:
            X, _ = mdl.build_block_features(df, block, train_mask_all, te_target=y_train_bench)
        except Exception:
            log(f"\nERROR building D5 features for {block}:")
            log(traceback.format_exc())
            continue
        for model in ["logistic_regression", "gradient_boosting"]:
            try:
                params = params_by_block[block][model]
                clf, scaler = rc.fit_cell(X[idx_train], y_train_bench, model, params)
                test_scores = rc.score_cell(clf, scaler, X[idx_test])
                model_selected, _ = build_topk(test_scores, K=K_PRIMARY, seed=TIE_BREAK_SEED)

                gm = rc.group_metrics(model_selected, y_test_bench, is_white[idx_test],
                                       is_bp[idx_test])
                k_local = int(np.floor(K_PRIMARY * len(idx_test)))
                ov = rc.overlap_at_k(model_selected, y_test_bench, k_local)

                row = dict(block=block, model=model, K=K_PRIMARY, overlap_at_k=ov,
                           n_population=len(df), **gm)
                rows_out.append(row)
                rc.record_scores("D5", "white_bp_only_population", block, model, K_PRIMARY,
                                  "White", "Black_Pardo", ni[idx_test], model_selected,
                                  y_test_bench, np.where(is_white[idx_test], "White",
                                                         np.where(is_bp[idx_test], "Black_Pardo", "other")))

                log(f"  [{block}/{model}] gap={gm['gap_pp']:+.2f}pp  ratio={gm['ratio']:.2f}  "
                    f"D={gm['D']:+.2f}pp  overlap@K={ov:.4f}")
            except Exception:
                log(f"\nERROR fitting D5 {block}/{model}:")
                log(traceback.format_exc())

    if rows_out:
        pd.DataFrame(rows_out).to_csv(OUT_CSV, index=False)
        log(f"\nSaved {OUT_CSV} ({len(rows_out)} rows).")
    n_scores = rc.flush_scores()
    log(f"robust_all_scores.parquet now has {n_scores} rows." if n_scores else "No scores flushed.")

    log(f"\nD5 total elapsed: {time.time()-t0:,.0f}s")
    return len(rows_out) > 0


if __name__ == "__main__":
    logger = Logger(RESULTS_DIR / "taskD_log.txt", mode="a")
    try:
        run(logger)
    finally:
        logger.close()
