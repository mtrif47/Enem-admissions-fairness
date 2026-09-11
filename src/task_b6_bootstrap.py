"""
Task B6: bootstrap inference for every B5 cell. 1000 replicates, stratified
by protected group (White resampled within White, Black/Pardo resampled
within Black/Pardo, each at fixed group size), percentile CIs on the
signed gap difference D = gap_model - gap_benchmark (both in percentage
points, White minus Black/Pardo) and on the selection-rate ratio.

Consumes results/main_grid_test_scores.parquet and results/test_benchmark.parquet
from Task B5 -- no refitting, since the point-estimate selected/not-selected
label per test candidate is already fixed from B5's top-K-within-test-split
evaluation.
"""

import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS_DIR, Logger, BOOTSTRAP_SEED, N_BOOTSTRAP

TEST_SCORES_PARQUET = RESULTS_DIR / "main_grid_test_scores.parquet"
TEST_BENCHMARK_PARQUET = RESULTS_DIR / "test_benchmark.parquet"
BOOTSTRAP_OUT = RESULTS_DIR / "bootstrap_ci.csv"
K_PRIMARY = 0.10
EPSILON = 1.0  # percentage points, Section 5.2


def topk_mask(scores: np.ndarray, K: float, seed: int):
    from common import build_topk
    selected, _ = build_topk(scores, K=K, seed=seed)
    return selected


def bootstrap_cell(model_selected, benchmark_selected, is_white, is_bp, seed, n_boot):
    white_idx = np.flatnonzero(is_white)
    bp_idx = np.flatnonzero(is_bp)
    n_white, n_bp = len(white_idx), len(bp_idx)

    rng = np.random.default_rng(seed)
    D_vals = np.empty(n_boot)
    ratio_vals = np.empty(n_boot)

    for b in range(n_boot):
        w_res = rng.choice(white_idx, size=n_white, replace=True)
        bp_res = rng.choice(bp_idx, size=n_bp, replace=True)

        rw_model = model_selected[w_res].mean()
        rbp_model = model_selected[bp_res].mean()
        rw_bench = benchmark_selected[w_res].mean()
        rbp_bench = benchmark_selected[bp_res].mean()

        gap_model = 100.0 * (rw_model - rbp_model)
        gap_bench = 100.0 * (rw_bench - rbp_bench)
        D_vals[b] = gap_model - gap_bench
        ratio_vals[b] = rw_model / rbp_model if rbp_model > 0 else np.nan

    return D_vals, ratio_vals


def _region(x, eps):
    if x > eps:
        return 1        # amplification
    if x < -eps:
        return -1       # attenuation
    return 0             # practical reproduction


def classify(D, ci_lo, ci_hi, eps=EPSILON):
    """Section 5.2: amplification (D>eps), attenuation (D<-eps), practical
    reproduction (|D|<=eps), or inconclusive when the CI spans more than
    one of those three regions."""
    if _region(ci_lo, eps) != _region(ci_hi, eps):
        return "inconclusive"
    return {1: "amplification", -1: "attenuation", 0: "practical_reproduction"}[_region(D, eps)]


def run(logger: Logger):
    log = logger.log
    log("\n" + "=" * 70)
    log("TASK B6 - BOOTSTRAP (1000 replicates, stratified by group, "
        "percentile CIs on D and ratio)")
    log("=" * 70)

    try:
        scores_long = pd.read_parquet(TEST_SCORES_PARQUET)
        bench = pd.read_parquet(TEST_BENCHMARK_PARQUET)
        log(f"\nLoaded {len(scores_long):,} score rows across "
            f"{scores_long[['block','model']].drop_duplicates().shape[0]} cells, "
            f"and {len(bench):,} test-set benchmark rows.")
    except Exception:
        log("\nERROR loading B5 outputs (has B5 finished?):")
        log(traceback.format_exc())
        return False

    is_white_full = (bench["race_label"] == "White").to_numpy()
    is_bp_full = bench["race_label"].isin(["Black", "Pardo"]).to_numpy()
    benchmark_selected_full = bench["benchmark_selected"].to_numpy().astype(bool)
    ni_order = bench["NU_INSCRICAO"].to_numpy()

    rows_out = []
    cells = scores_long[["block", "model"]].drop_duplicates().values.tolist()

    for block, model in cells:
        try:
            cell = scores_long[(scores_long["block"] == block) & (scores_long["model"] == model)]
            cell = cell.set_index("NU_INSCRICAO").loc[ni_order]  # align to bench order
            scores = cell["score"].to_numpy()

            model_selected = topk_mask(scores, K_PRIMARY, seed=BOOTSTRAP_SEED)

            # point estimates (should match B5's main_grid.csv row for this cell)
            rw = model_selected[is_white_full].mean()
            rbp = model_selected[is_bp_full].mean()
            gap_point = 100 * (rw - rbp)
            ratio_point = rw / rbp if rbp > 0 else np.nan
            bw = benchmark_selected_full[is_white_full].mean()
            bbp = benchmark_selected_full[is_bp_full].mean()
            bench_gap_point = 100 * (bw - bbp)
            D_point = gap_point - bench_gap_point

            D_vals, ratio_vals = bootstrap_cell(
                model_selected, benchmark_selected_full, is_white_full, is_bp_full,
                seed=BOOTSTRAP_SEED, n_boot=N_BOOTSTRAP,
            )
            D_lo, D_hi = np.percentile(D_vals, [2.5, 97.5])
            r_lo, r_hi = np.percentile(ratio_vals, [2.5, 97.5])
            verdict = classify(D_point, D_lo, D_hi)
            # Section 5.2: a sign change in the gap itself is a "reversal,"
            # reported separately from the magnitude-based D verdict above.
            sign_reversal = (np.sign(gap_point) != np.sign(bench_gap_point)
                              and bench_gap_point != 0)

            log(f"\n  [{block} / {model}] D={D_point:+.2f}pp  95% CI=[{D_lo:+.2f}, {D_hi:+.2f}]  "
                f"-> {verdict}{'  [SIGN REVERSAL]' if sign_reversal else ''}")
            log(f"    ratio={ratio_point:.2f}  95% CI=[{r_lo:.2f}, {r_hi:.2f}]")

            rows_out.append(dict(
                block=block, model=model, K=K_PRIMARY, n_boot=N_BOOTSTRAP,
                D_point=D_point, D_ci_lo=D_lo, D_ci_hi=D_hi, verdict=verdict,
                sign_reversal=bool(sign_reversal),
                ratio_point=ratio_point, ratio_ci_lo=r_lo, ratio_ci_hi=r_hi,
                gap_point_pp=gap_point, bench_gap_point_pp=bench_gap_point,
            ))
        except Exception:
            log(f"\nERROR bootstrapping cell {block}/{model}:")
            log(traceback.format_exc())

    if rows_out:
        pd.DataFrame(rows_out).to_csv(BOOTSTRAP_OUT, index=False)
        log(f"\nSaved {BOOTSTRAP_OUT} ({len(rows_out)} rows).")

    return len(rows_out) > 0


if __name__ == "__main__":
    logger = Logger(RESULTS_DIR / "taskB_log.txt", mode="a")
    try:
        run(logger)
    finally:
        logger.close()
