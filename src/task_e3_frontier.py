"""
Task E3: constraint sweep (the fidelity-parity frontier), blocks B3/B4,
both models, K=10%. Sweeps the equal-opportunity constraint strength from
"no constraint" (the unmitigated primary model) to full TPR equalisation
(exactly E2's result), at 11 levels total. Capacity is held at
floor(K*N_test) at every level via the same search-and-transfer machinery
as E2 (see mitig_common.py's module docstring), just varying the TARGET
TPR gap instead of always targeting zero.

Level 0 is the raw unmitigated top-K rule directly (not the search
machinery), so it reproduces results/main_grid.csv exactly. Levels
1-10 interpolate target_gap linearly from the unmitigated model's own
observed TPR gap (on validation) down to 0; level 10 is therefore
identical in spec to Task E2 (same target_gap=0, same search-and-transfer
protocol), included here for a complete, self-contained frontier.
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
OUT_CSV = RESULTS_DIR / "mitig_frontier.csv"
K_PRIMARY = 0.10
BLOCKS = ["B3", "B4"]
N_LEVELS = 11  # level 0 = unmitigated, level 10 = full equalisation (>=10 constrained levels)


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
    log("TASK E3 - CONSTRAINT SWEEP / FIDELITY-PARITY FRONTIER (B3/B4, K=10%, 11 levels)")
    log("=" * 70)

    df = load_data()
    train_mask_all = (df["split"] == "train").to_numpy()
    idx_train = np.flatnonzero(train_mask_all)
    idx_val = np.flatnonzero((df["split"] == "val").to_numpy())
    idx_test = np.flatnonzero((df["split"] == "test").to_numpy())
    composite = df["composite_raw_equal"].to_numpy()
    race = df["race_label"]
    ni = df["NU_INSCRICAO"].to_numpy()

    y_train_bench, _ = build_topk(composite[idx_train], K=K_PRIMARY, seed=TIE_BREAK_SEED)
    y_val_bench, _ = build_topk(composite[idx_val], K=K_PRIMARY, seed=TIE_BREAK_SEED)
    y_test_bench, _ = build_topk(composite[idx_test], K=K_PRIMARY, seed=TIE_BREAK_SEED)

    is_white = (race == "White").to_numpy()
    is_bp = race.isin(["Black", "Pardo"]).to_numpy()
    is_ai = ~is_white & ~is_bp

    params_by_block = rc.load_primary_params()
    rows_out = []

    for block in BLOCKS:
        try:
            X, _ = mdl.build_block_features(df, block, train_mask_all, te_target=y_train_bench)
        except Exception:
            log(f"\nERROR building E3 features for {block}:")
            log(traceback.format_exc())
            continue

        for model in ["logistic_regression", "gradient_boosting"]:
            try:
                params = params_by_block[block][model]
                clf, scaler = rc.fit_cell(X[idx_train], y_train_bench, model, params)
                val_scores = rc.score_cell(clf, scaler, X[idx_val])
                test_scores = rc.score_cell(clf, scaler, X[idx_test])

                unmit_val_sel, _ = build_topk(val_scores, K=K_PRIMARY, seed=TIE_BREAK_SEED)
                unmit_test_sel, _ = build_topk(test_scores, K=K_PRIMARY, seed=TIE_BREAK_SEED)

                capacity_val = int(np.floor(K_PRIMARY * len(idx_val)))
                capacity_test = int(np.floor(K_PRIMARY * len(idx_test)))
                n_ai_val = int(unmit_val_sel[is_ai[idx_val]].sum())
                n_ai_test = int(unmit_test_sel[is_ai[idx_test]].sum())
                cap_rem_val = capacity_val - n_ai_val
                cap_rem_test = capacity_test - n_ai_test

                tpr_w_unmit = ((unmit_val_sel[is_white[idx_val]] & y_val_bench[is_white[idx_val]]).sum()
                               / y_val_bench[is_white[idx_val]].sum())
                tpr_b_unmit = ((unmit_val_sel[is_bp[idx_val]] & y_val_bench[is_bp[idx_val]]).sum()
                               / y_val_bench[is_bp[idx_val]].sum())
                unmit_gap = tpr_w_unmit - tpr_b_unmit
                log(f"\n  [{block}/{model}] unmitigated val TPR gap = {unmit_gap:+.4f} "
                    f"(TPR_white={tpr_w_unmit:.4f}, TPR_bp={tpr_b_unmit:.4f})")

                target_gaps = np.linspace(unmit_gap, 0.0, N_LEVELS)

                for level, target_gap in enumerate(target_gaps):
                    if level == 0:
                        model_selected = unmit_test_sel.copy()
                        tpr_w_test = tpr_w_unmit  # reported val-side TPRs, consistent with search levels
                        tpr_b_test = tpr_b_unmit
                        n_total = int(model_selected.sum())
                    else:
                        scores_white_val = val_scores[is_white[idx_val]]
                        bench_white_val = y_val_bench[is_white[idx_val]]
                        scores_bp_val = val_scores[is_bp[idx_val]]
                        bench_bp_val = y_val_bench[is_bp[idx_val]]
                        split_info = mc.search_threshold_split(
                            scores_white_val, bench_white_val, scores_bp_val, bench_bp_val,
                            cap_rem_val, target_gap=target_gap,
                        )
                        scores_white_test = test_scores[is_white[idx_test]]
                        scores_bp_test = test_scores[is_bp[idx_test]]
                        sel_w, sel_b, _, _ = mc.apply_split_to_test(
                            scores_white_test, scores_bp_test,
                            split_info["rate_white"], split_info["rate_bp"], cap_rem_test,
                        )
                        model_selected = np.zeros(len(idx_test), dtype=bool)
                        model_selected[is_white[idx_test]] = sel_w
                        model_selected[is_bp[idx_test]] = sel_b
                        model_selected[is_ai[idx_test]] = unmit_test_sel[is_ai[idx_test]]
                        n_total = int(model_selected.sum())
                        tpr_w_test, tpr_b_test = split_info["tpr_white"], split_info["tpr_bp"]

                    res = mc.evaluate_from_selected(model_selected, y_test_bench,
                                                     race.iloc[idx_test], K_PRIMARY,
                                                     model_scores=test_scores)
                    constraint_strength = (1.0 - target_gap / unmit_gap) if unmit_gap != 0 else np.nan

                    row = dict(block=block, model=model, K=K_PRIMARY, level=level,
                               constraint_strength=constraint_strength, target_tpr_gap=target_gap,
                               overlap_at_k=res["overlap_at_k"], gap_pp=res["gap_pp"],
                               ratio=res["ratio"], D=res["D"], tpr_white=tpr_w_test,
                               tpr_bp=tpr_b_test, n_realised=n_total,
                               n_target=capacity_test, zero_sum_value=res["zero_sum_value"])
                    rows_out.append(row)

                    if level in (0, N_LEVELS - 1):
                        mc.record_scores(
                            "E3", f"level{level}_{'unmitigated' if level == 0 else 'full_eq'}",
                            block, model, K_PRIMARY, "White", "Black_Pardo", ni[idx_test],
                            model_selected, y_test_bench,
                            np.where(is_white[idx_test], "White",
                                     np.where(is_bp[idx_test], "Black_Pardo", "other")))

                log(f"    levels: " + ", ".join(
                    f"L{r['level']}(gap={r['gap_pp']:+.1f}pp)" for r in rows_out[-N_LEVELS:]))
            except Exception:
                log(f"\nERROR fitting E3 {block}/{model}:")
                log(traceback.format_exc())

    if rows_out:
        pd.DataFrame(rows_out).to_csv(OUT_CSV, index=False)
        log(f"\nSaved {OUT_CSV} ({len(rows_out)} rows).")
    n_scores = mc.flush_scores()
    log(f"mitig_all_scores.parquet now has {n_scores} rows." if n_scores else "No scores flushed.")

    log(f"\nE3 total elapsed: {time.time()-t0:,.0f}s")
    return len(rows_out) > 0


if __name__ == "__main__":
    logger = Logger(RESULTS_DIR / "taskE_log.txt", mode="a")
    try:
        run(logger)
    finally:
        logger.close()
