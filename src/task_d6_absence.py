"""
Task D6: end-to-end absence robustness (Section 4.5). Repeat B3/B4, both
models, K=10%, treating non-completers (absent from one or more objective
exams, or missing a score) as NOT SELECTED rather than excluding them, so
reported disparities incorporate both the completion stage and the
selection stage: P(selected|G) = P(completes|G) x P(selected|completes,G).

Implementation: the "completes" population is exactly the existing
analytic sample (cascade steps 1,2,3,4,5 = completers with declared race),
so its S^0 and split assignment are reused UNCHANGED from Task A/B3. The
"non-completer" population (cascade steps 1,2,5 only, i.e. same population
minus the steps-3/4 attendance and complete-score requirements) is
appended with S^0 fixed at 0 (by construction: an absent candidate cannot
be ranked by a composite it doesn't have, so it is not selected) and a
fresh race-stratified 60/20/20 split assignment, seed 20190101 --
decile-based stratification is impossible for these rows since they have
no composite score, so this extends rather than redraws the completers'
split, same convention as Task D4(d).
"""

import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (RESULTS_DIR, DADOS_DIR, ANALYTIC_SAMPLE_PARQUET, Logger, build_topk,
                     TIE_BREAK_SEED, apply_cascade, RACE_MAP, SPLIT_SEED, SCORE_COLS)
import modeling as mdl
import robust_common as rc

SPLIT_PARQUET = RESULTS_DIR / "split_indices.parquet"
OUT_CSV = RESULTS_DIR / "robust_absence.csv"
K_PRIMARY = 0.10
BLOCKS = ["B3", "B4"]


def build_e2e_frame(logger):
    log = logger.log
    completers = pd.read_parquet(ANALYTIC_SAMPLE_PARQUET)
    split = pd.read_parquet(SPLIT_PARQUET)[["NU_INSCRICAO", "split"]]
    completers = completers.merge(split, on="NU_INSCRICAO", how="inner")
    assert len(completers) == len(split)

    composite = completers["composite_raw_equal"].to_numpy()
    idx_train_c = np.flatnonzero(completers["split"].to_numpy() == "train")
    idx_test_c = np.flatnonzero(completers["split"].to_numpy() == "test")
    y_train_bench, _ = build_topk(composite[idx_train_c], K=K_PRIMARY, seed=TIE_BREAK_SEED)
    y_test_bench, _ = build_topk(composite[idx_test_c], K=K_PRIMARY, seed=TIE_BREAK_SEED)
    completers = completers.copy()
    completers["y_e2e"] = 0
    completers.loc[completers.index[idx_train_c], "y_e2e"] = y_train_bench.astype(int)
    completers.loc[completers.index[idx_test_c], "y_e2e"] = y_test_bench.astype(int)
    # val split's y is unused (no retuning happens in Task D) but fill for completeness
    idx_val_c = np.flatnonzero(completers["split"].to_numpy() == "val")
    y_val_bench, _ = build_topk(composite[idx_val_c], K=K_PRIMARY, seed=TIE_BREAK_SEED)
    completers.loc[completers.index[idx_val_c], "y_e2e"] = y_val_bench.astype(int)
    log(f"\nCompleters (= analytic sample): N={len(completers):,}, "
        f"e2e-positive={completers['y_e2e'].sum():,} "
        f"(should equal train+val+test benchmark positives from Task B)")

    raw = pd.read_csv(DADOS_DIR / "enem_2019_sample.csv", sep=";", encoding="latin-1",
                       low_memory=False)
    pop_e2e_all = apply_cascade(raw, steps=(1, 2, 5))
    log(f"Population after cascade steps 1,2,5 (completion+race, attendance/scores NOT "
        f"required): N={len(pop_e2e_all):,}")

    non_completers = pop_e2e_all[~pop_e2e_all["NU_INSCRICAO"].isin(completers["NU_INSCRICAO"])].copy()
    log(f"Non-completers (absent from >=1 exam or missing a score): N={len(non_completers):,}")

    non_completers["race_label"] = non_completers["TP_COR_RACA"].map(RACE_MAP)
    non_completers["composite_raw_equal"] = np.nan
    non_completers["y_e2e"] = 0

    idx_nc = non_completers.index.to_numpy()
    strat = non_completers["race_label"].to_numpy()
    train_idx, temp_idx = train_test_split(idx_nc, test_size=0.40, random_state=SPLIT_SEED,
                                            stratify=strat)
    temp_strat = non_completers.loc[temp_idx, "race_label"].to_numpy()
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=SPLIT_SEED,
                                          stratify=temp_strat)
    non_completers["split"] = "train"
    non_completers.loc[val_idx, "split"] = "val"
    non_completers.loc[test_idx, "split"] = "test"
    log(f"  Non-completer split sizes: train={len(train_idx):,} val={len(val_idx):,} "
        f"test={len(test_idx):,}")

    keep_cols = list(set(completers.columns) & set(non_completers.columns))
    combined = pd.concat([completers[keep_cols], non_completers[keep_cols]], ignore_index=True)
    log(f"Combined end-to-end frame: N={len(combined):,}  "
        f"(completers {len(completers):,} + non-completers {len(non_completers):,})")
    return combined


def run(logger: Logger):
    log = logger.log
    t0 = time.time()
    log("\n" + "=" * 70)
    log("TASK D6 - END-TO-END ABSENCE ROBUSTNESS (B3/B4, K=10%)")
    log("=" * 70)

    combined = build_e2e_frame(logger)
    train_mask_all = (combined["split"] == "train").to_numpy()
    idx_train = np.flatnonzero(train_mask_all)
    idx_test = np.flatnonzero(combined["split"].to_numpy() == "test")
    race = combined["race_label"]
    is_white = (race == "White").to_numpy()
    is_bp = race.isin(["Black", "Pardo"]).to_numpy()
    ni = combined["NU_INSCRICAO"].to_numpy()
    y_all = combined["y_e2e"].to_numpy()
    y_train = y_all[idx_train]
    y_test = y_all[idx_test]

    log(f"\nEnd-to-end test set: N={len(idx_test):,}, positives={int(y_test.sum()):,} "
        f"({100*y_test.mean():.2f}% -- lower than 10% since absentees dilute the rate)")

    params_by_block = rc.load_primary_params()
    rows_out = []
    for block in BLOCKS:
        try:
            X, _ = mdl.build_block_features(combined, block, train_mask_all, te_target=y_train)
        except Exception:
            log(f"\nERROR building D6 features for {block}:")
            log(traceback.format_exc())
            continue
        for model in ["logistic_regression", "gradient_boosting"]:
            try:
                params = params_by_block[block][model]
                clf, scaler = rc.fit_cell(X[idx_train], y_train, model, params)
                test_scores = rc.score_cell(clf, scaler, X[idx_test])
                # capacity: same absolute seat count as the primary K=10%
                # analysis, i.e. floor(K * N_completers_in_test), not
                # floor(K * N_e2e_test) -- see module docstring.
                n_completers_test = int((~combined["composite_raw_equal"].isna()).to_numpy()[idx_test].sum())
                k_e2e = int(np.floor(K_PRIMARY * n_completers_test))
                # build_topk takes a K fraction of len(scores); we need an
                # exact seat COUNT (k_e2e, the same absolute capacity as the
                # primary K=10% analysis) instead, so rank/threshold directly.
                order = np.argsort(-test_scores, kind="mergesort")
                selected = np.zeros(len(test_scores), dtype=bool)
                selected[order[:k_e2e]] = True
                model_selected = selected

                gm = rc.group_metrics(model_selected, y_test.astype(bool), is_white[idx_test],
                                       is_bp[idx_test])
                ov = rc.overlap_at_k(model_selected, y_test.astype(bool), k_e2e)

                row = dict(block=block, model=model, K=K_PRIMARY, k_seats=k_e2e,
                           n_test_e2e=len(idx_test), n_completers_test=n_completers_test,
                           overlap_at_k=ov, **gm)
                rows_out.append(row)
                rc.record_scores("D6", "end_to_end_absence", block, model, K_PRIMARY,
                                  "White", "Black_Pardo", ni[idx_test], model_selected,
                                  y_test.astype(bool), np.where(is_white[idx_test], "White",
                                                                np.where(is_bp[idx_test], "Black_Pardo", "other")))

                log(f"  [{block}/{model}] seats={k_e2e:,}  gap={gm['gap_pp']:+.2f}pp  "
                    f"ratio={gm['ratio']:.2f}  D={gm['D']:+.2f}pp  overlap@K={ov:.4f}")
            except Exception:
                log(f"\nERROR fitting D6 {block}/{model}:")
                log(traceback.format_exc())

    if rows_out:
        pd.DataFrame(rows_out).to_csv(OUT_CSV, index=False)
        log(f"\nSaved {OUT_CSV} ({len(rows_out)} rows).")
    n_scores = rc.flush_scores()
    log(f"robust_all_scores.parquet now has {n_scores} rows." if n_scores else "No scores flushed.")

    log(f"\nD6 total elapsed: {time.time()-t0:,.0f}s")
    return len(rows_out) > 0


if __name__ == "__main__":
    logger = Logger(RESULTS_DIR / "taskD_log.txt", mode="a")
    try:
        run(logger)
    finally:
        logger.close()
