"""
Task D4: protected-group variants of blocks B3/B4 (both models, K=10%).

Variants (a)-(c) change only which groups are contrasted for reporting;
the underlying fitted model, features, and prediction target (S^0 at
K=10% over the same N=123,879 ranking population) are identical to the
primary B5 grid, so they reuse results/main_grid_test_scores.parquet and
results/test_benchmark.parquet directly -- no refitting.

Variant (d) genuinely changes the ranking population (adds back the
~2,434 candidates with TP_COR_RACA=0, "not declared," per Section 4.2
step 5 being skipped), which changes S^0 itself (ranks shift) and
requires new features/target encoding/model fits. Its extra rows were
never split in Task B3, so -- per "reuse the split, don't redraw" -- the
existing 123,879 rows keep their exact original split label, and only
the ~2,434 new rows get a fresh stratified 60/20/20 assignment (by their
own composite decile, seed 20190101), extending rather than redrawing
the split.
"""

import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (RESULTS_DIR, DADOS_DIR, Logger, build_topk, TIE_BREAK_SEED,
                     apply_cascade, add_composite, RACE_MAP, SPLIT_SEED)
import modeling as mdl
import robust_common as rc

SPLIT_PARQUET = RESULTS_DIR / "split_indices.parquet"
ANALYTIC_SAMPLE_PARQUET = RESULTS_DIR / "analytic_sample.parquet"
TEST_SCORES_PARQUET = RESULTS_DIR / "main_grid_test_scores.parquet"
TEST_BENCHMARK_PARQUET = RESULTS_DIR / "test_benchmark.parquet"
OUT_CSV = RESULTS_DIR / "robust_groups.csv"
K_PRIMARY = 0.10
BLOCKS = ["B3", "B4"]


def variant_abc(logger, sex_lookup):
    log = logger.log
    log("\n" + "-" * 70)
    log("D4 (a)-(c): reusing primary B3/B4 test scores (no refit)")
    log("-" * 70)

    scores = pd.read_parquet(TEST_SCORES_PARQUET)
    bench = pd.read_parquet(TEST_BENCHMARK_PARQUET)
    bench = bench.merge(sex_lookup, on="NU_INSCRICAO", how="left")

    rows = []
    for block in BLOCKS:
        for model in ["logistic_regression", "gradient_boosting"]:
            cell = scores[(scores["block"] == block) & (scores["model"] == model)]
            cell = cell.set_index("NU_INSCRICAO").loc[bench["NU_INSCRICAO"]]
            s = cell["score"].to_numpy()
            model_selected, _ = build_topk(s, K=K_PRIMARY, seed=TIE_BREAK_SEED)
            b_sel = bench["benchmark_selected"].to_numpy().astype(bool)
            race = bench["race_label"]
            sex = bench["TP_SEXO"]

            def add_row(variant, mask_a, mask_b, label_a, label_b):
                gm = rc.group_metrics(model_selected, b_sel, mask_a, mask_b)
                rows.append(dict(variant=variant, block=block, model=model, K=K_PRIMARY,
                                  group_a=label_a, group_b=label_b, **gm))
                rc.record_scores("D4", variant, block, model, K_PRIMARY, label_a, label_b,
                                  bench["NU_INSCRICAO"].to_numpy(), model_selected, b_sel,
                                  np.where(mask_a, label_a, np.where(mask_b, label_b, "other")))
                log(f"  [{variant}] {block}/{model}: {label_a} vs {label_b}  "
                    f"n={gm['n_a']}/{gm['n_b']}  gap={gm['gap_pp']:+.2f}pp  ratio={gm['ratio']:.2f}  "
                    f"bench_pos={gm['n_bench_positive_a']}/{gm['n_bench_positive_b']}  "
                    f"model_pos={gm['n_model_positive_a']}/{gm['n_model_positive_b']}")

            # (a) Black and Pardo separately vs White
            add_row("black_vs_white", (race == "White").to_numpy(), (race == "Black").to_numpy(),
                    "White", "Black")
            add_row("pardo_vs_white", (race == "White").to_numpy(), (race == "Pardo").to_numpy(),
                    "White", "Pardo")
            # (b) Black/Pardo/Indigenous combined vs White
            add_row("bpi_vs_white", (race == "White").to_numpy(),
                    race.isin(["Black", "Pardo", "Indigenous"]).to_numpy(),
                    "White", "Black_Pardo_Indigenous")
            # (c) sex F vs M
            add_row("sex_f_vs_m", (sex == "F").to_numpy(), (sex == "M").to_numpy(), "F", "M")
    return rows


def build_extended_split(logger):
    """Section 4.2 population after steps 1-4 only (race NOT restricted to
    declared), N should be 126,313. Extends -- does not redraw -- the
    Task B3 split."""
    log = logger.log
    raw = pd.read_csv(DADOS_DIR / "enem_2019_sample.csv", sep=";", encoding="latin-1",
                       low_memory=False)
    pop = apply_cascade(raw, steps=(1, 2, 3, 4)).copy()
    pop = add_composite(pop)
    pop["race_label"] = pop["TP_COR_RACA"].map(RACE_MAP)
    log(f"\nD4(d) population after cascade steps 1-4 (race not restricted): "
        f"N={len(pop):,} (Section 4.2 reports 126,313 at this step).")

    existing_split = pd.read_parquet(SPLIT_PARQUET)[["NU_INSCRICAO", "split"]]
    pop = pop.merge(existing_split, on="NU_INSCRICAO", how="left")
    new_mask = pop["split"].isna()
    n_new = int(new_mask.sum())
    log(f"  {len(pop) - n_new:,} rows already split in Task B3 (unchanged); "
        f"{n_new:,} new 'not declared' rows need a fresh split assignment.")

    new_rows = pop[new_mask].copy()
    decile_new = pd.qcut(new_rows["composite_raw_equal"], 10, labels=False, duplicates="drop")
    idx_new = new_rows.index.to_numpy()
    train_idx, temp_idx = train_test_split(idx_new, test_size=0.40, random_state=SPLIT_SEED,
                                            stratify=decile_new.to_numpy())
    temp_decile = decile_new.loc[temp_idx].to_numpy()
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=SPLIT_SEED,
                                          stratify=temp_decile)
    pop.loc[train_idx, "split"] = "train"
    pop.loc[val_idx, "split"] = "val"
    pop.loc[test_idx, "split"] = "test"
    assert pop["split"].isna().sum() == 0
    log(f"  New-row split sizes: train={len(train_idx):,} val={len(val_idx):,} "
        f"test={len(test_idx):,}")
    return pop


def variant_d(logger):
    log = logger.log
    log("\n" + "-" * 70)
    log("D4 (d): race not declared retained (new population, refit B3/B4)")
    log("-" * 70)

    pop = build_extended_split(logger)
    train_mask_all = (pop["split"] == "train").to_numpy()
    idx_train = np.flatnonzero(train_mask_all)
    idx_test = np.flatnonzero(pop["split"].to_numpy() == "test")
    composite = pop["composite_raw_equal"].to_numpy()
    race = pop["race_label"]
    is_white = (race == "White").to_numpy()
    is_bp = race.isin(["Black", "Pardo"]).to_numpy()
    is_nd = (race == "Not declared").to_numpy()
    ni = pop["NU_INSCRICAO"].to_numpy()

    y_train_bench, _ = build_topk(composite[idx_train], K=K_PRIMARY, seed=TIE_BREAK_SEED)
    y_test_bench, _ = build_topk(composite[idx_test], K=K_PRIMARY, seed=TIE_BREAK_SEED)
    log(f"  Benchmark over extended population: train pos={int(y_train_bench.sum()):,}, "
        f"test pos={int(y_test_bench.sum()):,}")

    params_by_block = rc.load_primary_params()
    rows = []
    for block in BLOCKS:
        try:
            X, _ = mdl.build_block_features(pop, block, train_mask_all, te_target=y_train_bench)
        except Exception:
            log(f"\nERROR building D4(d) features for {block}:")
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
                n_nd_test = int(is_nd[idx_test].sum())
                n_nd_bench_pos = int(y_test_bench[is_nd[idx_test]].sum())
                n_nd_model_pos = int(model_selected[is_nd[idx_test]].sum())

                row = dict(variant="race_not_declared_retained", block=block, model=model,
                           K=K_PRIMARY, group_a="White", group_b="Black_Pardo",
                           n_not_declared=n_nd_test,
                           n_bench_positive_not_declared=n_nd_bench_pos,
                           n_model_positive_not_declared=n_nd_model_pos, **gm)
                rows.append(row)
                rc.record_scores("D4", "race_not_declared_retained", block, model, K_PRIMARY,
                                  "White", "Black_Pardo", ni[idx_test], model_selected,
                                  y_test_bench, np.where(is_white[idx_test], "White",
                                                         np.where(is_bp[idx_test], "Black_Pardo", "other")))

                log(f"  [{block}/{model}] gap={gm['gap_pp']:+.2f}pp  ratio={gm['ratio']:.2f}  "
                    f"not-declared: n={n_nd_test}, bench_pos={n_nd_bench_pos}, "
                    f"model_pos={n_nd_model_pos}")
            except Exception:
                log(f"\nERROR fitting D4(d) {block}/{model}:")
                log(traceback.format_exc())
    return rows


def run(logger: Logger):
    log = logger.log
    t0 = time.time()
    log("\n" + "=" * 70)
    log("TASK D4 - PROTECTED GROUP VARIANTS (B3/B4, K=10%)")
    log("=" * 70)

    analytic = pd.read_parquet(ANALYTIC_SAMPLE_PARQUET)
    sex_lookup = analytic[["NU_INSCRICAO", "TP_SEXO"]]

    all_rows = []
    try:
        all_rows.extend(variant_abc(logger, sex_lookup))
    except Exception:
        log("\nERROR in D4 (a)-(c):")
        log(traceback.format_exc())

    try:
        all_rows.extend(variant_d(logger))
    except Exception:
        log("\nERROR in D4 (d):")
        log(traceback.format_exc())

    if all_rows:
        pd.DataFrame(all_rows).to_csv(OUT_CSV, index=False)
        log(f"\nSaved {OUT_CSV} ({len(all_rows)} rows).")
    n_scores = rc.flush_scores()
    log(f"robust_all_scores.parquet now has {n_scores} rows." if n_scores else "No scores flushed.")

    log(f"\nD4 total elapsed: {time.time()-t0:,.0f}s")
    return len(all_rows) > 0


if __name__ == "__main__":
    logger = Logger(RESULTS_DIR / "taskD_log.txt", mode="a")
    try:
        run(logger)
    finally:
        logger.close()
