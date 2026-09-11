"""
Task D7: bootstrap every cell produced by D1-D6. 1000 replicates,
stratified by protected group (group_a resampled within group_a, group_b
resampled within group_b, each at fixed observed size), percentile CIs on
the signed gap difference D = gap_model - gap_benchmark (percentage
points, group_a minus group_b) and on the selection-rate ratio. Verdict
via epsilon = 1pp, same rule as Task B6.

Consumes results/robust_all_scores.parquet, which already carries the
fixed model_selected/benchmark_selected boolean label for every row in
every (task, variant, block, model, K, group_a, group_b) cell -- no
refitting, no re-thresholding, just resampling rows and averaging labels
(see robust_common.py's record_scores docstring for why the stored label
must already be final, especially for Task D1's "global" definition).
"""

import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS_DIR, Logger, BOOTSTRAP_SEED, N_BOOTSTRAP
import robust_common as rc

OUT_CSV = RESULTS_DIR / "robust_bootstrap.csv"
EPSILON = 1.0


def _region(x, eps):
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0


def classify(D, ci_lo, ci_hi, eps=EPSILON):
    if _region(ci_lo, eps) != _region(ci_hi, eps):
        return "inconclusive"
    return {1: "amplification", -1: "attenuation", 0: "practical_reproduction"}[_region(D, eps)]


def bootstrap_cell(model_selected, benchmark_selected, mask_a, mask_b, seed, n_boot):
    idx_a = np.flatnonzero(mask_a)
    idx_b = np.flatnonzero(mask_b)
    n_a, n_b = len(idx_a), len(idx_b)
    rng = np.random.default_rng(seed)
    D_vals = np.empty(n_boot)
    ratio_vals = np.empty(n_boot)
    for i in range(n_boot):
        r_a = rng.choice(idx_a, size=n_a, replace=True)
        r_b = rng.choice(idx_b, size=n_b, replace=True)
        ra_m, rb_m = model_selected[r_a].mean(), model_selected[r_b].mean()
        ra_bh, rb_bh = benchmark_selected[r_a].mean(), benchmark_selected[r_b].mean()
        gap_m = 100 * (ra_m - rb_m)
        gap_bh = 100 * (ra_bh - rb_bh)
        D_vals[i] = gap_m - gap_bh
        ratio_vals[i] = ra_m / rb_m if rb_m > 0 else np.nan
    return D_vals, ratio_vals


def run(logger: Logger):
    log = logger.log
    log("\n" + "=" * 70)
    log("TASK D7 - BOOTSTRAP FOR EVERY D1-D6 CELL (1000 reps, stratified, eps=1pp)")
    log("=" * 70)

    try:
        data = pd.read_parquet(rc.ALL_SCORES_PARQUET)
    except Exception:
        log("\nERROR loading robust_all_scores.parquet (have D1-D6 run?):")
        log(traceback.format_exc())
        return False

    key_cols = ["task", "variant", "block", "model", "K", "group_a", "group_b"]
    cells = data[key_cols].drop_duplicates()
    log(f"\nLoaded {len(data):,} rows across {len(cells)} cells.")

    rows_out = []
    for _, key in cells.iterrows():
        try:
            mask_key = np.logical_and.reduce([data[c] == key[c] for c in key_cols])
            cell = data[mask_key]
            model_selected = cell["model_selected"].to_numpy()
            benchmark_selected = cell["benchmark_selected"].to_numpy()
            group_value = cell["group_value"].to_numpy()
            mask_a = group_value == key["group_a"]
            mask_b = group_value == key["group_b"]

            if mask_a.sum() < 2 or mask_b.sum() < 2:
                log(f"\nSKIP {tuple(key)}: insufficient group size for bootstrap "
                    f"(n_a={mask_a.sum()}, n_b={mask_b.sum()})")
                continue

            rate_a_m, rate_b_m = model_selected[mask_a].mean(), model_selected[mask_b].mean()
            rate_a_b, rate_b_b = benchmark_selected[mask_a].mean(), benchmark_selected[mask_b].mean()
            gap_point = 100 * (rate_a_m - rate_b_m)
            bench_gap_point = 100 * (rate_a_b - rate_b_b)
            D_point = gap_point - bench_gap_point
            ratio_point = rate_a_m / rate_b_m if rate_b_m > 0 else np.nan

            D_vals, ratio_vals = bootstrap_cell(model_selected, benchmark_selected, mask_a, mask_b,
                                                 seed=BOOTSTRAP_SEED, n_boot=N_BOOTSTRAP)
            D_lo, D_hi = np.percentile(D_vals, [2.5, 97.5])
            r_lo, r_hi = np.percentile(ratio_vals, [2.5, 97.5])
            verdict = classify(D_point, D_lo, D_hi)
            sign_reversal = bool(np.sign(gap_point) != np.sign(bench_gap_point)
                                  and bench_gap_point != 0)

            row = dict(**key.to_dict(), n_a=int(mask_a.sum()), n_b=int(mask_b.sum()),
                       D_point=D_point, D_ci_lo=D_lo, D_ci_hi=D_hi, verdict=verdict,
                       sign_reversal=sign_reversal, ratio_point=ratio_point,
                       ratio_ci_lo=r_lo, ratio_ci_hi=r_hi,
                       gap_point_pp=gap_point, bench_gap_point_pp=bench_gap_point)
            rows_out.append(row)
            log(f"  [{key['task']}/{key['variant']}/{key['block']}/{key['model']}] "
                f"D={D_point:+.2f}pp CI=[{D_lo:+.2f},{D_hi:+.2f}] -> {verdict}"
                f"{'  [SIGN REVERSAL]' if sign_reversal else ''}")
        except Exception:
            log(f"\nERROR bootstrapping cell {tuple(key)}:")
            log(traceback.format_exc())

    if rows_out:
        pd.DataFrame(rows_out).to_csv(OUT_CSV, index=False)
        log(f"\nSaved {OUT_CSV} ({len(rows_out)} rows).")

    return len(rows_out) > 0


if __name__ == "__main__":
    logger = Logger(RESULTS_DIR / "taskD_log.txt", mode="a")
    try:
        run(logger)
    finally:
        logger.close()
