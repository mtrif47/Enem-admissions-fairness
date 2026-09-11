"""
Task D3: state-only geography (Section 5.5 leakage check). Refit B2-B6,
both models, K=10%, with CO_MUNICIPIO_PROVA dropped so state
(SG_UF_PROVA, already in every block) is the only geography. Hyperparameters
reused from the primary grid (see robust_common.py).

Municipality is disabled by temporarily forcing modeling.BLOCK_MUNICIPALITY
to False for every block within this process, then restoring it -- simpler
than duplicating build_block_features, and safe since this script runs
standalone.

Compares the B3->B4 gap increase (the race-block effect) under state-only
geography against the same delta in the primary grid (which target-encodes
municipality from B2 onward): if dropping municipality changes that delta
materially, municipality's target encoding was carrying some of what looks
like the "race" effect in the primary grid.
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
OUT_CSV = RESULTS_DIR / "robust_state_only.csv"
K_PRIMARY = 0.10
BLOCKS = ["B2", "B3", "B4", "B5", "B6"]


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
    log("TASK D3 - STATE-ONLY GEOGRAPHY (municipality dropped, B2-B6, K=10%)")
    log("=" * 70)

    df = load_data()
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

    params_by_block = rc.load_primary_params()
    primary_grid = pd.read_csv(rc.MAIN_GRID_CSV).set_index(["block", "model"])

    original_muni = dict(mdl.BLOCK_MUNICIPALITY)
    for k in mdl.BLOCK_MUNICIPALITY:
        mdl.BLOCK_MUNICIPALITY[k] = False
    log("\nBLOCK_MUNICIPALITY forced to False for all blocks (state-only ablation).")

    rows_out = {}  # (block, model) -> gap_pp, for the B3->B4 delta check
    try:
        for block in BLOCKS:
            try:
                X, feature_names = mdl.build_block_features(df, block, train_mask_all,
                                                              te_target=y_train_bench)
                assert not any("MUNICIPIO" in n for n in feature_names), \
                    f"municipality leaked into {block} features: {feature_names}"
            except Exception:
                log(f"\nERROR building state-only features for block {block}:")
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

                    primary_row = primary_grid.loc[(block, model)]
                    delta_vs_primary = gm["gap_pp"] - primary_row["gap_pp"]

                    row = dict(block=block, model=model, K=K_PRIMARY, overlap_at_k=ov,
                               primary_gap_pp=primary_row["gap_pp"],
                               delta_vs_primary_gap_pp=delta_vs_primary, **gm)
                    rows_out[(block, model)] = row
                    rc.record_scores("D3", "state_only", block, model, K_PRIMARY,
                                      "White", "Black_Pardo", ni[idx_test], model_selected,
                                      y_test_bench, np.where(is_white[idx_test], "White",
                                                             np.where(is_bp[idx_test], "Black_Pardo", "other")))

                    log(f"  [{block}/{model}] gap={gm['gap_pp']:+.2f}pp  "
                        f"(primary={primary_row['gap_pp']:+.2f}pp, "
                        f"delta={delta_vs_primary:+.2f}pp)")
                except Exception:
                    log(f"\nERROR fitting state-only {block}/{model}:")
                    log(traceback.format_exc())
    finally:
        mdl.BLOCK_MUNICIPALITY.update(original_muni)
        log("\nBLOCK_MUNICIPALITY restored.")

    log("\n" + "-" * 70)
    log("B3 -> B4 gap-increase comparison (state-only vs primary)")
    log("-" * 70)
    for model in ["logistic_regression", "gradient_boosting"]:
        try:
            d_state = rows_out[("B4", model)]["gap_pp"] - rows_out[("B3", model)]["gap_pp"]
            d_primary = (primary_grid.loc[("B4", model), "gap_pp"]
                         - primary_grid.loc[("B3", model), "gap_pp"])
            diff = d_state - d_primary
            flag = "  !!! MATERIALLY DIFFERENT (>1pp) -- municipality may be doing hidden work !!!" \
                if abs(diff) > 1.0 else "  (no material difference)"
            log(f"  {model}: B3->B4 delta state-only={d_state:+.2f}pp  "
                f"primary={d_primary:+.2f}pp  diff={diff:+.2f}pp{flag}")
        except KeyError:
            log(f"  {model}: could not compute (missing cell)")

    if rows_out:
        pd.DataFrame(list(rows_out.values())).to_csv(OUT_CSV, index=False)
        log(f"\nSaved {OUT_CSV} ({len(rows_out)} rows).")
    n_scores = rc.flush_scores()
    log(f"robust_all_scores.parquet now has {n_scores} rows." if n_scores else "No scores flushed.")

    log(f"\nD3 total elapsed: {time.time()-t0:,.0f}s")
    return len(rows_out) > 0


if __name__ == "__main__":
    logger = Logger(RESULTS_DIR / "taskD_log.txt", mode="a")
    try:
        run(logger)
    finally:
        logger.close()
