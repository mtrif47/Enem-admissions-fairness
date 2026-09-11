"""
Task E5: bootstrap for every cell in E1 and E2, plus the E3 frontier
endpoints (level 0 = unmitigated, level 10 = full equalisation). 1000
replicates, stratified by protected group (White resampled within White,
Black/Pardo resampled within Black/Pardo, each at fixed observed size).
Reports D (signed gap difference vs the benchmark) with its 95% CI and
verdict (epsilon=1pp, same rule as Task B6/D7), and the fidelity cost --
overlap@K of the mitigated model minus overlap@K of the corresponding
unmitigated primary model (results/main_grid.csv) -- with its own 95% CI.

The unmitigated comparator for each (block, model) cell is reconstructed
by applying the standard top-K rule to results/main_grid_test_scores.parquet
(the exact primary B5 fit), aligned row-for-row with the mitigated
selection via NU_INSCRICAO.

Fidelity cost is computed on the SAME resampled White+Black/Pardo index
set as D (this project's bootstrap has stratified on that pair throughout;
Asian/Indigenous rows are not part of the resample here, so this is a
White+BP-conditional overlap difference, not the whole-test-set overlap@K
reported as a point estimate in E1-E3). Within each replicate, overlap for
a rule is TP/(that rule's own selected count in the resample), matching
the precision-style definition already used for Task D1's "global"
definition when two rules do not select equal counts.
"""

import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS_DIR, Logger, BOOTSTRAP_SEED, N_BOOTSTRAP, build_topk, TIE_BREAK_SEED
import mitig_common as mc

MAIN_GRID_TEST_SCORES = RESULTS_DIR / "main_grid_test_scores.parquet"
OUT_CSV = RESULTS_DIR / "mitig_bootstrap.csv"
K_PRIMARY = 0.10
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


def bootstrap_cell(mitig_selected, unmit_selected, benchmark_selected, mask_a, mask_b,
                    seed, n_boot):
    idx_a = np.flatnonzero(mask_a)
    idx_b = np.flatnonzero(mask_b)
    n_a, n_b = len(idx_a), len(idx_b)
    rng = np.random.default_rng(seed)
    D_vals = np.empty(n_boot)
    fidelity_cost_vals = np.empty(n_boot)
    for i in range(n_boot):
        r_a = rng.choice(idx_a, size=n_a, replace=True)
        r_b = rng.choice(idx_b, size=n_b, replace=True)
        r = np.concatenate([r_a, r_b])

        ra_m, rb_m = mitig_selected[r_a].mean(), mitig_selected[r_b].mean()
        ra_bh, rb_bh = benchmark_selected[r_a].mean(), benchmark_selected[r_b].mean()
        gap_m = 100 * (ra_m - rb_m)
        gap_bh = 100 * (ra_bh - rb_bh)
        D_vals[i] = gap_m - gap_bh

        mitig_r, unmit_r, bench_r = mitig_selected[r], unmit_selected[r], benchmark_selected[r]
        k_mitig = mitig_r.sum()
        k_unmit = unmit_r.sum()
        overlap_mitig = (mitig_r & bench_r).sum() / k_mitig if k_mitig > 0 else np.nan
        overlap_unmit = (unmit_r & bench_r).sum() / k_unmit if k_unmit > 0 else np.nan
        fidelity_cost_vals[i] = overlap_mitig - overlap_unmit
    return D_vals, fidelity_cost_vals


def get_unmitigated(block, model, ni_order):
    scores = pd.read_parquet(MAIN_GRID_TEST_SCORES)
    cell = scores[(scores["block"] == block) & (scores["model"] == model)]
    cell = cell.set_index("NU_INSCRICAO").loc[ni_order]
    s = cell["score"].to_numpy()
    selected, _ = build_topk(s, K=K_PRIMARY, seed=TIE_BREAK_SEED)
    return selected


def run(logger: Logger):
    log = logger.log
    log("\n" + "=" * 70)
    log("TASK E5 - BOOTSTRAP (E1, E2, E3 endpoints; 1000 reps, eps=1pp)")
    log("=" * 70)

    try:
        data = pd.read_parquet(mc.MITIG_ALL_SCORES_PARQUET)
    except Exception:
        log("\nERROR loading mitig_all_scores.parquet (have E1-E3 run?):")
        log(traceback.format_exc())
        return False

    wanted_tasks = {
        "E1": None,  # all variants
        "E2": None,
        "E3": {"level0_unmitigated", "level10_full_eq"},
    }

    key_cols = ["task", "variant", "block", "model", "K", "group_a", "group_b"]
    cells = data[key_cols].drop_duplicates()
    cells = cells[cells.apply(
        lambda r: r["task"] in wanted_tasks and (wanted_tasks[r["task"]] is None
                                                  or r["variant"] in wanted_tasks[r["task"]]),
        axis=1)]
    log(f"\nLoaded {len(data):,} rows; bootstrapping {len(cells)} cells.")

    rows_out = []
    for _, key in cells.iterrows():
        try:
            mask_key = np.logical_and.reduce([data[c] == key[c] for c in key_cols])
            cell = data[mask_key].sort_values("row_id")
            ni_order = cell["row_id"].to_numpy()
            mitig_selected = cell["model_selected"].to_numpy()
            benchmark_selected = cell["benchmark_selected"].to_numpy()
            group_value = cell["group_value"].to_numpy()
            mask_a = group_value == key["group_a"]
            mask_b = group_value == key["group_b"]

            unmit_selected = get_unmitigated(key["block"], key["model"], ni_order)

            rate_a_m, rate_b_m = mitig_selected[mask_a].mean(), mitig_selected[mask_b].mean()
            rate_a_b, rate_b_b = benchmark_selected[mask_a].mean(), benchmark_selected[mask_b].mean()
            gap_point = 100 * (rate_a_m - rate_b_m)
            bench_gap_point = 100 * (rate_a_b - rate_b_b)
            D_point = gap_point - bench_gap_point

            k_mitig_pt = mitig_selected[mask_a | mask_b].sum()
            k_unmit_pt = unmit_selected[mask_a | mask_b].sum()
            overlap_mitig_pt = ((mitig_selected & benchmark_selected)[mask_a | mask_b]).sum() / k_mitig_pt
            overlap_unmit_pt = ((unmit_selected & benchmark_selected)[mask_a | mask_b]).sum() / k_unmit_pt
            fidelity_cost_point = overlap_mitig_pt - overlap_unmit_pt

            D_vals, fc_vals = bootstrap_cell(mitig_selected, unmit_selected, benchmark_selected,
                                              mask_a, mask_b, seed=BOOTSTRAP_SEED, n_boot=N_BOOTSTRAP)
            D_lo, D_hi = np.percentile(D_vals, [2.5, 97.5])
            fc_lo, fc_hi = np.percentile(fc_vals, [2.5, 97.5])
            verdict = classify(D_point, D_lo, D_hi)

            row = dict(**key.to_dict(), D_point=D_point, D_ci_lo=D_lo, D_ci_hi=D_hi,
                       verdict=verdict, fidelity_cost_point=fidelity_cost_point,
                       fidelity_cost_ci_lo=fc_lo, fidelity_cost_ci_hi=fc_hi,
                       gap_point_pp=gap_point, bench_gap_point_pp=bench_gap_point)
            rows_out.append(row)
            log(f"  [{key['task']}/{key['variant']}/{key['block']}/{key['model']}] "
                f"D={D_point:+.2f}pp CI=[{D_lo:+.2f},{D_hi:+.2f}] -> {verdict}  |  "
                f"fidelity_cost={fidelity_cost_point:+.4f} CI=[{fc_lo:+.4f},{fc_hi:+.4f}]")
        except Exception:
            log(f"\nERROR bootstrapping cell {tuple(key)}:")
            log(traceback.format_exc())

    if rows_out:
        pd.DataFrame(rows_out).to_csv(OUT_CSV, index=False)
        log(f"\nSaved {OUT_CSV} ({len(rows_out)} rows).")

    return len(rows_out) > 0


if __name__ == "__main__":
    logger = Logger(RESULTS_DIR / "taskE_log.txt", mode="a")
    try:
        run(logger)
    finally:
        logger.close()
