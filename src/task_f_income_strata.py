"""
Task F: income-stratified robustness cell, following Rocha and Nascimento
(2019) on income-conditional racial achievement gaps. Question: does the
B3->B4 race increment vary by income tercile?

Reuses Task B's split, seeds and already-fitted primary models entirely --
no refitting, no re-tuning, no redrawing the split. All metrics computed
on the TEST split only.

F1: income terciles from Q006 (17 ordered bands, A-Q). Since bands are
discrete, exact 33/33/33 cut points don't generally exist; the cut is
placed at whichever band boundary's cumulative share is closest to 1/3
and 2/3, and the resulting (uneven) tercile sizes are reported as-is
rather than forced to sizes they can't discretely reach.

F2/F3: per the task's explicit instruction, the benchmark's global top-10%
selection (already fixed in Task A over the full N=123,879 ranking
population) and each model's global top-10%-within-test selection
(already fixed in Task B's main_grid_test_scores.parquet) are NOT
recomputed within tercile subsets -- only MEASURED there. Re-ranking
within a tercile would answer a different question (a tercile-specific
benchmark) than the one asked (how the existing benchmark/models play out
across income strata).
"""

import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ANALYTIC_SAMPLE_PARQUET, RESULTS_DIR, Logger, TIE_BREAK_SEED, BOOTSTRAP_SEED, N_BOOTSTRAP, build_topk
import features as feat

SPLIT_PARQUET = RESULTS_DIR / "split_indices.parquet"
TEST_SCORES_PARQUET = RESULTS_DIR / "main_grid_test_scores.parquet"
OUT_CSV = RESULTS_DIR / "robust_income_strata.csv"
K_PRIMARY = 0.10
BLOCKS = ["B3", "B4"]
MODELS = ["logistic_regression", "gradient_boosting"]
MIN_EVENTS = 30


def assign_terciles(df, logger):
    log = logger.log
    ordinal = df["Q006"].map(feat.LETTER_TO_ORDINAL)
    bands = sorted(df["Q006"].unique(), key=lambda c: feat.LETTER_TO_ORDINAL[c])
    counts = df["Q006"].value_counts()
    counts = counts.reindex(bands)
    cum_pct = counts.cumsum() / len(df)

    def nearest_band(target):
        diffs = (cum_pct - target).abs()
        return diffs.idxmin()

    cut1_band = nearest_band(1 / 3)
    cut2_band = nearest_band(2 / 3)
    cut1_ord = feat.LETTER_TO_ORDINAL[cut1_band]
    cut2_ord = feat.LETTER_TO_ORDINAL[cut2_band]
    log(f"\nF1: Q006 cumulative shares by band:\n{cum_pct.round(4).to_string()}")
    log(f"\nNearest-to-1/3 band: {cut1_band} (cum={cum_pct[cut1_band]:.4f}); "
        f"nearest-to-2/3 band: {cut2_band} (cum={cum_pct[cut2_band]:.4f})")

    tercile = np.where(ordinal <= cut1_ord, "Low",
                        np.where(ordinal <= cut2_ord, "Mid", "High"))
    low_bands = [b for b in bands if feat.LETTER_TO_ORDINAL[b] <= cut1_ord]
    mid_bands = [b for b in bands if cut1_ord < feat.LETTER_TO_ORDINAL[b] <= cut2_ord]
    high_bands = [b for b in bands if feat.LETTER_TO_ORDINAL[b] > cut2_ord]
    log(f"\nTercile band assignment: Low={low_bands}  Mid={mid_bands}  High={high_bands}")

    tercile_ser = pd.Series(tercile, index=df.index, name="income_tercile")
    for t in ["Low", "Mid", "High"]:
        n = (tercile_ser == t).sum()
        log(f"  {t:<5s} N={n:,} ({100*n/len(df):.2f}% of analytic sample)")
    return tercile_ser, dict(low_bands=low_bands, mid_bands=mid_bands, high_bands=high_bands)


def compute_gap(selected, benchmark_or_selected_is_rate, mask_white, mask_bp):
    """rate/gap for a fixed boolean selection array, generic helper."""
    rate_w = 100 * selected[mask_white].mean()
    rate_b = 100 * selected[mask_bp].mean()
    return rate_w, rate_b, rate_w - rate_b


def bootstrap_increment(b3_selected, b4_selected, benchmark_selected, mask_white, mask_bp,
                         seed, n_boot):
    idx_w = np.flatnonzero(mask_white)
    idx_b = np.flatnonzero(mask_bp)
    n_w, n_b = len(idx_w), len(idx_b)
    rng = np.random.default_rng(seed)
    increments = np.empty(n_boot)
    for i in range(n_boot):
        r_w = rng.choice(idx_w, size=n_w, replace=True)
        r_b = rng.choice(idx_b, size=n_b, replace=True)
        gap3 = 100 * (b3_selected[r_w].mean() - b3_selected[r_b].mean())
        gap4 = 100 * (b4_selected[r_w].mean() - b4_selected[r_b].mean())
        increments[i] = gap4 - gap3
    return increments


def run(logger: Logger):
    log = logger.log
    t0 = time.time()
    log("\n" + "=" * 70)
    log("TASK F - INCOME-STRATIFIED RACE INCREMENT (B3->B4, K=10%)")
    log("=" * 70)

    analytic = pd.read_parquet(ANALYTIC_SAMPLE_PARQUET)
    split = pd.read_parquet(SPLIT_PARQUET)[["NU_INSCRICAO", "split"]]
    analytic = analytic.merge(split, on="NU_INSCRICAO", how="inner")
    assert len(analytic) == len(split)

    tercile_ser, band_info = assign_terciles(analytic, logger)
    analytic["income_tercile"] = tercile_ser

    test_df = analytic[analytic["split"] == "test"].copy()
    log(f"\nTest-set tercile sizes:")
    for t in ["Low", "Mid", "High"]:
        n = (test_df["income_tercile"] == t).sum()
        log(f"  {t:<5s} N={n:,}")

    race = test_df["race_label"]
    is_white = (race == "White").to_numpy()
    is_bp = race.isin(["Black", "Pardo"]).to_numpy()
    benchmark_selected = test_df["benchmark_top10"].to_numpy().astype(bool)
    log(f"\nF2: global benchmark_top10 (from Task A, full N=123,879 ranking population) "
        f"restricted to test: {benchmark_selected.sum():,} of {len(test_df):,} selected "
        f"({100*benchmark_selected.mean():.2f}%, target was 10%).")

    scores = pd.read_parquet(TEST_SCORES_PARQUET)
    model_selected_by_cell = {}
    for block in BLOCKS:
        for model in MODELS:
            cell = scores[(scores["block"] == block) & (scores["model"] == model)]
            cell = cell.set_index("NU_INSCRICAO").loc[test_df["NU_INSCRICAO"]]
            s = cell["score"].to_numpy()
            sel, _ = build_topk(s, K=K_PRIMARY, seed=TIE_BREAK_SEED)
            model_selected_by_cell[(block, model)] = sel
            log(f"  Reused primary top-K-within-test selection for {block}/{model}: "
                f"{int(sel.sum()):,} selected (matches main_grid.csv's own test evaluation).")

    rows_out = []
    increments_for_comparison = {}  # model -> {tercile: (point, lo, hi)}

    for tercile in ["Low", "Mid", "High"]:
        log(f"\n{'='*20} Tercile = {tercile} {'='*20}")
        t_mask = (test_df["income_tercile"] == tercile).to_numpy()
        mask_white_t = is_white & t_mask
        mask_bp_t = is_bp & t_mask
        n_white_t, n_bp_t = int(mask_white_t.sum()), int(mask_bp_t.sum())

        bench_pos_white = int(benchmark_selected[mask_white_t].sum())
        bench_pos_bp = int(benchmark_selected[mask_bp_t].sum())
        flag = ""
        if bench_pos_white < MIN_EVENTS or bench_pos_bp < MIN_EVENTS:
            flag = "  !!! FLAG: benchmark positives < 30 in at least one group !!!"
        log(f"  N: White={n_white_t:,}  Black/Pardo={n_bp_t:,}  "
            f"benchmark_positive: White={bench_pos_white}  BP={bench_pos_bp}{flag}")

        rw, rb, bench_gap = compute_gap(benchmark_selected, None, mask_white_t, mask_bp_t)
        log(f"  F2 benchmark gap (White vs BP), measured within tercile: "
            f"White={rw:.2f}%  BP={rb:.2f}%  gap={bench_gap:+.2f}pp")

        for model in MODELS:
            b3_sel = model_selected_by_cell[("B3", model)]
            b4_sel = model_selected_by_cell[("B4", model)]

            rw3, rb3, gap3 = compute_gap(b3_sel, None, mask_white_t, mask_bp_t)
            rw4, rb4, gap4 = compute_gap(b4_sel, None, mask_white_t, mask_bp_t)
            increment = gap4 - gap3

            model_pos_white_b3 = int(b3_sel[mask_white_t].sum())
            model_pos_bp_b3 = int(b3_sel[mask_bp_t].sum())
            model_pos_white_b4 = int(b4_sel[mask_white_t].sum())
            model_pos_bp_b4 = int(b4_sel[mask_bp_t].sum())

            inc_vals = bootstrap_increment(b3_sel, b4_sel, benchmark_selected,
                                            mask_white_t, mask_bp_t,
                                            seed=BOOTSTRAP_SEED, n_boot=N_BOOTSTRAP)
            inc_lo, inc_hi = np.percentile(inc_vals, [2.5, 97.5])
            increments_for_comparison.setdefault(model, {})[tercile] = (increment, inc_lo, inc_hi)

            log(f"  [{model}] B3 gap={gap3:+.2f}pp (White={rw3:.2f}%,BP={rb3:.2f}%)  "
                f"B4 gap={gap4:+.2f}pp (White={rw4:.2f}%,BP={rb4:.2f}%)  "
                f"increment={increment:+.2f}pp  95%CI=[{inc_lo:+.2f},{inc_hi:+.2f}]")

            rows_out.append(dict(
                tercile=tercile, model=model, K=K_PRIMARY,
                n_white=n_white_t, n_bp=n_bp_t,
                bench_positive_white=bench_pos_white, bench_positive_bp=bench_pos_bp,
                low_event_flag=bool(bench_pos_white < MIN_EVENTS or bench_pos_bp < MIN_EVENTS),
                bench_rate_white_pct=rw, bench_rate_bp_pct=rb, bench_gap_pp=bench_gap,
                b3_rate_white_pct=rw3, b3_rate_bp_pct=rb3, b3_gap_pp=gap3,
                b3_model_positive_white=model_pos_white_b3, b3_model_positive_bp=model_pos_bp_b3,
                b4_rate_white_pct=rw4, b4_rate_bp_pct=rb4, b4_gap_pp=gap4,
                b4_model_positive_white=model_pos_white_b4, b4_model_positive_bp=model_pos_bp_b4,
                race_increment_pp=increment, increment_ci_lo=inc_lo, increment_ci_hi=inc_hi,
                low_bands=str(band_info["low_bands"]), mid_bands=str(band_info["mid_bands"]),
                high_bands=str(band_info["high_bands"]),
            ))

    log("\n" + "-" * 70)
    log("F5: do increments differ across terciles beyond sampling variation?")
    log("-" * 70)
    for model, tdict in increments_for_comparison.items():
        log(f"\n  {model}:")
        for t in ["Low", "Mid", "High"]:
            pt, lo, hi = tdict[t]
            log(f"    {t:<5s} increment={pt:+.2f}pp  95% CI=[{lo:+.2f}, {hi:+.2f}]")
        pairs = [("Low", "Mid"), ("Mid", "High"), ("Low", "High")]
        for a, b in pairs:
            pa, la, ha = tdict[a]
            pb, lb, hb = tdict[b]
            overlap = not (ha < lb or hb < la)
            verdict = "CIs overlap -- no evidence of a difference beyond sampling variation" \
                if overlap else "CIs DO NOT overlap -- suggestive evidence of a real difference"
            log(f"    {a} vs {b}: {verdict}")

    if rows_out:
        pd.DataFrame(rows_out).to_csv(OUT_CSV, index=False)
        log(f"\nSaved {OUT_CSV} ({len(rows_out)} rows).")

    log(f"\nTask F total elapsed: {time.time()-t0:,.0f}s")
    return len(rows_out) > 0


if __name__ == "__main__":
    logger = Logger(RESULTS_DIR / "taskF_log.txt", mode="a")
    try:
        run(logger)
    finally:
        logger.close()
