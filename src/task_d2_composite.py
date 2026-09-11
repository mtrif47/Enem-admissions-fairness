"""
Task D2: composite-scheme robustness. Rebuild the benchmark under each
Section 4.6 alternative weighting, refit blocks B3 and B4 only (both
models, K=10%), reusing Section-5.5 hyperparameters (see robust_common.py).

Composite construction is a fixed arithmetic transform of the five raw
score columns (institutional weights, or standardisation), not something
fit on training data -- exactly like the primary raw equal-weight
composite in Task A, which is a population-level quantity, not a
train-fold-fitted feature. "Standardised equal weights" therefore
z-scores each component using the whole analytic sample's own mean/SD
(consistent with how the primary composite itself is computed over the
whole sample, with no train/test split), not training-fold statistics.
"""

import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ANALYTIC_SAMPLE_PARQUET, RESULTS_DIR, Logger, build_topk, TIE_BREAK_SEED, SCORE_COLS
import robust_common as rc

SPLIT_PARQUET = RESULTS_DIR / "split_indices.parquet"
OUT_CSV = RESULTS_DIR / "robust_composite.csv"
K_PRIMARY = 0.10
BLOCKS = ["B3", "B4"]

WEIGHT_SCHEMES = {
    "ufs_eng_computacao": dict(NU_NOTA_CN=0.15, NU_NOTA_CH=0.10, NU_NOTA_LC=0.15,
                                NU_NOTA_MT=0.35, NU_NOTA_REDACAO=0.25),
    "ufs_medicina": dict(NU_NOTA_CN=0.35, NU_NOTA_CH=0.10, NU_NOTA_LC=0.20,
                          NU_NOTA_MT=0.15, NU_NOTA_REDACAO=0.20),
    "ufs_direito": dict(NU_NOTA_CN=0.10, NU_NOTA_CH=0.20, NU_NOTA_LC=0.25,
                         NU_NOTA_MT=0.15, NU_NOTA_REDACAO=0.30),
    "objective_only": dict(NU_NOTA_CN=0.25, NU_NOTA_CH=0.25, NU_NOTA_LC=0.25,
                            NU_NOTA_MT=0.25, NU_NOTA_REDACAO=0.0),
}


def build_composite(df, scheme):
    if scheme == "standardised_equal":
        z = (df[SCORE_COLS] - df[SCORE_COLS].mean()) / df[SCORE_COLS].std(ddof=1)
        return z.mean(axis=1).to_numpy()
    weights = WEIGHT_SCHEMES[scheme]
    return sum(df[c].to_numpy() * w for c, w in weights.items())


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
    log("TASK D2 - COMPOSITE SCHEME ROBUSTNESS (blocks B3/B4, K=10%, "
        "hyperparameters reused from primary)")
    log("=" * 70)

    df = load_data()
    train_mask_all = (df["split"] == "train").to_numpy()
    idx_train = np.flatnonzero(train_mask_all)
    idx_test = np.flatnonzero(df["split"].to_numpy() == "test")
    race = df["race_label"]
    is_white = (race == "White").to_numpy()
    is_bp = race.isin(["Black", "Pardo"]).to_numpy()
    ni = df["NU_INSCRICAO"].to_numpy()

    params_by_block = rc.load_primary_params()
    schemes = list(WEIGHT_SCHEMES.keys()) + ["standardised_equal"]

    rows_out = []
    for scheme in schemes:
        try:
            composite_scheme = build_composite(df, scheme)
        except Exception:
            log(f"\nERROR building composite for scheme {scheme}:")
            log(traceback.format_exc())
            continue

        y_train_bench, _ = build_topk(composite_scheme[idx_train], K=K_PRIMARY, seed=TIE_BREAK_SEED)
        y_test_bench, _ = build_topk(composite_scheme[idx_test], K=K_PRIMARY, seed=TIE_BREAK_SEED)
        log(f"\n{'='*15} scheme = {scheme} {'='*15}")
        log(f"  composite mean={composite_scheme.mean():.4f} sd={composite_scheme.std(ddof=1):.4f}  "
            f"train positives={int(y_train_bench.sum()):,}  test positives={int(y_test_bench.sum()):,}")

        for block in BLOCKS:
            try:
                X, _ = rc.mdl.build_block_features(df, block, train_mask_all, te_target=y_train_bench)
            except Exception:
                log(f"\nERROR building features for block {block}, scheme {scheme}:")
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

                    row = dict(scheme=scheme, block=block, model=model, K=K_PRIMARY,
                               overlap_at_k=ov, **gm)
                    rows_out.append(row)
                    rc.record_scores("D2", scheme, block, model, K_PRIMARY,
                                      "White", "Black_Pardo", ni[idx_test], model_selected,
                                      y_test_bench, np.where(is_white[idx_test], "White",
                                                             np.where(is_bp[idx_test], "Black_Pardo", "other")))

                    log(f"  [{block}/{model}] gap={gm['gap_pp']:+.2f}pp  ratio={gm['ratio']:.2f}  "
                        f"D={gm['D']:+.2f}pp  overlap@K={ov:.4f}")
                except Exception:
                    log(f"\nERROR fitting {block}/{model}, scheme {scheme}:")
                    log(traceback.format_exc())

    if rows_out:
        pd.DataFrame(rows_out).to_csv(OUT_CSV, index=False)
        log(f"\nSaved {OUT_CSV} ({len(rows_out)} rows).")
    n_scores = rc.flush_scores()
    log(f"robust_all_scores.parquet now has {n_scores} rows." if n_scores else "No scores flushed.")

    log(f"\nD2 total elapsed: {time.time()-t0:,.0f}s")
    return len(rows_out) > 0


if __name__ == "__main__":
    logger = Logger(RESULTS_DIR / "taskD_log.txt", mode="a")
    try:
        run(logger)
    finally:
        logger.close()
