"""
Task B1 (full-file recompute of Section 6.1 + Section 4.5 completion rates)
and Task B2 (sample vs full-file marginal comparison), from the §4.2
pending item.

Reads DADOS/MICRODADOS_ENEM_2019.csv (~5.1M rows, 2.4GB) once, in 200k-row
chunks, using only the columns Section 4.3 defines. One pass accumulates:
  (a) raw (unfiltered) marginals for B2, comparable to the same marginals
      on the untouched 4% sample, and
  (b) the Section 4.2 restriction cascade, whose surviving rows (composite
      + race only) are kept in memory for the Section 6.1 recomputation.

Designed to be run standalone: `python3 src/task_b1_b2_fullfile.py`.
Writes results/fullfile_descriptives.csv (B1) and results/sample_vs_full.csv
(B2). Also returns (b1_ok, b2_ok) so an orchestrator can log partial failure.
"""

import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    ROOT, DADOS_DIR, RESULTS_DIR, SAMPLE_CSV, FULLFILE_CSV,
    NEEDED_COLS, SCORE_COLS, PRESENCE_COLS, RACE_MAP, TIE_BREAK_SEED,
    apply_cascade, add_composite, build_topk, selection_table, Logger,
)

CHUNK_SIZE = 200_000

FULLFILE_DESC_OUT = RESULTS_DIR / "fullfile_descriptives.csv"
SAMPLE_VS_FULL_OUT = RESULTS_DIR / "sample_vs_full.csv"


class RunningMoments:
    """Streaming count/sum/sumsq for a continuous variable's non-missing values."""

    def __init__(self):
        self.n = 0
        self.s = 0.0
        self.ss = 0.0

    def update(self, values: np.ndarray):
        v = values[~np.isnan(values)]
        self.n += v.size
        self.s += v.sum()
        self.ss += np.square(v).sum()

    def mean(self):
        return self.s / self.n if self.n else np.nan

    def sd(self):
        if self.n < 2:
            return np.nan
        var = (self.ss - self.n * self.mean() ** 2) / (self.n - 1)
        return np.sqrt(max(var, 0.0))


class RunningCounts:
    def __init__(self):
        self.counts = {}
        self.total = 0

    def update(self, series: pd.Series):
        vc = series.value_counts(dropna=False)
        for k, v in vc.items():
            self.counts[k] = self.counts.get(k, 0) + int(v)
        self.total += len(series)

    def as_pct(self):
        return {k: 100.0 * v / self.total for k, v in self.counts.items()}


def compute_raw_marginals(df: pd.DataFrame, moments=None, counts=None):
    """Update running marginal accumulators from one raw (unfiltered) chunk
    or the full in-memory sample."""
    if moments is None:
        moments = {c: RunningMoments() for c in SCORE_COLS}
    if counts is None:
        counts = {
            "TP_COR_RACA": RunningCounts(),
            "TP_SEXO": RunningCounts(),
            "SG_UF_PROVA": RunningCounts(),
            **{c: RunningCounts() for c in PRESENCE_COLS},
        }
    for c in SCORE_COLS:
        moments[c].update(df[c].to_numpy(dtype=float))
    counts["TP_COR_RACA"].update(df["TP_COR_RACA"])
    counts["TP_SEXO"].update(df["TP_SEXO"])
    counts["SG_UF_PROVA"].update(df["SG_UF_PROVA"])
    for c in PRESENCE_COLS:
        counts[c].update(df[c])
    return moments, counts


def run(logger: Logger):
    log = logger.log
    t0 = time.time()
    log("=" * 70)
    log("TASK B1 + B2 - FULL-FILE RECOMPUTE AND SAMPLE-VS-FULL CHECK")
    log("=" * 70)

    # ------------------------------------------------------------------
    # Single chunked pass over the full file.
    # ------------------------------------------------------------------
    log(f"\nReading {FULLFILE_CSV} in chunks of {CHUNK_SIZE:,} "
        f"(usecols = {len(NEEDED_COLS)} columns from Section 4.3) ...")

    full_moments, full_counts = None, None
    restricted_frames = []  # slim: composite inputs + race, post full cascade
    post12_group_counts = {}       # race -> N after steps 1+2 (completion denom)
    post12_present_counts = {}     # race -> N of those also present all 4 (numerator)

    n_chunks = 0
    n_rows_seen = 0

    reader = pd.read_csv(
        FULLFILE_CSV, sep=";", encoding="latin-1", usecols=NEEDED_COLS,
        chunksize=CHUNK_SIZE, low_memory=False,
    )

    for chunk in reader:
        n_chunks += 1
        n_rows_seen += len(chunk)

        # (a) raw marginals, unfiltered
        full_moments, full_counts = compute_raw_marginals(chunk, full_moments, full_counts)

        # (b) restriction cascade, steps 1-2 first (completion-rate population)
        post12 = apply_cascade(chunk, steps=(1, 2))
        present_all4 = (post12[PRESENCE_COLS] == 1).all(axis=1)
        for race_code, sub in post12.groupby("TP_COR_RACA"):
            label = RACE_MAP.get(race_code, f"code_{race_code}")
            post12_group_counts[label] = post12_group_counts.get(label, 0) + len(sub)
        for race_code, mask in present_all4.groupby(post12["TP_COR_RACA"]):
            label = RACE_MAP.get(race_code, f"code_{race_code}")
            post12_present_counts[label] = post12_present_counts.get(label, 0) + int(mask.sum())

        # continue cascade through steps 3-5 for the Section 6.1 recompute
        restricted = post12[present_all4]
        restricted = restricted[restricted[SCORE_COLS].notna().all(axis=1)]
        restricted = restricted[restricted["TP_COR_RACA"] != 0]
        restricted_frames.append(
            restricted[["TP_COR_RACA"] + SCORE_COLS].assign(
                race_label=restricted["TP_COR_RACA"].map(RACE_MAP)
            )
        )

        if n_chunks % 5 == 0 or n_chunks == 1:
            log(f"  ... chunk {n_chunks}, rows seen {n_rows_seen:,}, "
                f"elapsed {time.time()-t0:,.0f}s")

    log(f"\nFinished reading. Total rows in full file: {n_rows_seen:,} "
        f"across {n_chunks} chunks ({time.time()-t0:,.0f}s).")

    restricted_full = pd.concat(restricted_frames, ignore_index=True)
    del restricted_frames
    n_restricted = len(restricted_full)
    log(f"Full-file analytic population after cascade steps 1-5: N = {n_restricted:,}")

    # ------------------------------------------------------------------
    # B1a: completion rates by group (Section 4.5), full file
    # ------------------------------------------------------------------
    b1_rows = []
    b1_ok = False
    try:
        log("\n" + "-" * 70)
        log("B1 / Section 4.5: completion rates by group (full file)")
        log("-" * 70)
        comp_rows = []
        for label, n_post12 in sorted(post12_group_counts.items(), key=lambda kv: -kv[1]):
            n_present = post12_present_counts.get(label, 0)
            rate = 100.0 * n_present / n_post12 if n_post12 else np.nan
            comp_rows.append(dict(metric="completion_rate", group=label,
                                   n_denominator=n_post12, n_numerator=n_present, value=rate))
            log(f"  {label:<14s} N={n_post12:>10,}  completion={rate:6.2f}%")
        b1_rows.extend(comp_rows)
    except Exception:
        log("\nERROR in B1 completion-rate computation:")
        log(traceback.format_exc())

    # ------------------------------------------------------------------
    # B1b: Section 6.1 recompute on the full-file analytic population
    # ------------------------------------------------------------------
    try:
        log("\n" + "-" * 70)
        log("B1 / Section 6.1 recompute (full file, N = {:,})".format(n_restricted))
        log("-" * 70)

        restricted_full = add_composite(restricted_full)
        comp = restricted_full["composite_raw_equal"]
        overall_mean, overall_sd = comp.mean(), comp.std(ddof=1)
        log(f"  Composite overall: mean={overall_mean:.1f}  SD={overall_sd:.1f}")
        b1_rows.append(dict(metric="composite_overall", group="ALL",
                             n_denominator=n_restricted, n_numerator=np.nan, value=overall_mean))
        b1_rows.append(dict(metric="composite_overall_sd", group="ALL",
                             n_denominator=n_restricted, n_numerator=np.nan, value=overall_sd))

        # subject-level gaps, White vs Black/Pardo
        is_white = restricted_full["race_label"] == "White"
        is_bp = restricted_full["race_label"].isin(["Black", "Pardo"])
        log("\n  Subject-level White vs Black/Pardo gaps:")
        for c in SCORE_COLS:
            sd_c = restricted_full[c].std(ddof=1)
            mean_w = restricted_full.loc[is_white, c].mean()
            mean_bp = restricted_full.loc[is_bp, c].mean()
            gap = mean_w - mean_bp
            gap_sd = gap / sd_c
            log(f"    {c:<16s} SD={sd_c:6.1f}  White={mean_w:6.1f}  "
                f"BP={mean_bp:6.1f}  gap={gap:5.1f}  gap/SD={gap_sd:.3f}")
            b1_rows.append(dict(metric=f"subject_gap_{c}", group="White_minus_BP",
                                 n_denominator=np.nan, n_numerator=np.nan, value=gap))
            b1_rows.append(dict(metric=f"subject_gap_sd_{c}", group="White_minus_BP",
                                 n_denominator=np.nan, n_numerator=np.nan, value=gap_sd))

        # composite means/SDs by group
        log("\n  Composite by group:")
        for label in ["White", "Asian", "Pardo", "Black", "Indigenous"]:
            sub = restricted_full[restricted_full["race_label"] == label]
            m, s = sub["composite_raw_equal"].mean(), sub["composite_raw_equal"].std(ddof=1)
            log(f"    {label:<12s} N={len(sub):>9,}  mean={m:7.1f}  SD={s:6.1f}")
            b1_rows.append(dict(metric="composite_mean", group=label,
                                 n_denominator=len(sub), n_numerator=np.nan, value=m))
            b1_rows.append(dict(metric="composite_sd", group=label,
                                 n_denominator=len(sub), n_numerator=np.nan, value=s))

        gap_composite = (restricted_full.loc[is_white, "composite_raw_equal"].mean()
                         - restricted_full.loc[is_bp, "composite_raw_equal"].mean())
        gap_composite_pooled_sd = gap_composite / overall_sd
        log(f"\n  White vs Black/Pardo composite gap: {gap_composite:.1f} points "
            f"({gap_composite_pooled_sd:.3f} pooled SD)")
        b1_rows.append(dict(metric="composite_gap_white_bp", group="ALL",
                             n_denominator=np.nan, n_numerator=np.nan, value=gap_composite))
        b1_rows.append(dict(metric="composite_gap_white_bp_pooled_sd", group="ALL",
                             n_denominator=np.nan, n_numerator=np.nan, value=gap_composite_pooled_sd))

        # selection table + pool composition + event counts at K=5/10/20
        composite_arr = comp.to_numpy()
        race_labels = restricted_full["race_label"]
        log("\n  Selection table (White vs Black/Pardo), pool composition, "
            "and benchmark-positive event counts:")
        for K in (0.05, 0.10, 0.20):
            selected, tie_info = build_topk(composite_arr, K=K, seed=TIE_BREAK_SEED)
            table = selection_table(race_labels, selected)
            log(f"\n    K = {int(K*100)}%  (tied at boundary = {tie_info['n_tied']:,}, "
                f"selected total = {int(selected.sum()):,})")
            log(f"      White rate={table['rate_white']:.2f}%  "
                f"BP rate={table['rate_black_pardo']:.2f}%  "
                f"gap={table['gap_pp']:.2f}pp  ratio={table['ratio']:.2f}")
            for k, v in table.items():
                b1_rows.append(dict(metric=f"selection_{k}", group=f"K{int(K*100)}",
                                     n_denominator=np.nan, n_numerator=np.nan, value=v))
            b1_rows.append(dict(metric="tied_at_boundary", group=f"K{int(K*100)}",
                                 n_denominator=np.nan, n_numerator=np.nan, value=tie_info["n_tied"]))

            # pool composition
            log(f"      Selected-pool composition:")
            eligible_pct = race_labels.value_counts(normalize=True) * 100
            for label in ["White", "Black", "Pardo", "Indigenous", "Asian"]:
                pool_pct = 100.0 * (race_labels[selected] == label).sum() / selected.sum()
                elig_pct = eligible_pct.get(label, 0.0)
                log(f"        {label:<12s} eligible={elig_pct:5.2f}%  selected_pool={pool_pct:5.2f}%")
                b1_rows.append(dict(metric="pool_composition_pct", group=f"K{int(K*100)}_{label}",
                                     n_denominator=np.nan, n_numerator=np.nan, value=pool_pct))
                b1_rows.append(dict(metric="eligible_pool_pct", group=label,
                                     n_denominator=np.nan, n_numerator=np.nan, value=elig_pct))

            # benchmark-positive event counts
            log(f"      Benchmark-positive event counts:")
            for label in ["White", "Black", "Pardo", "Asian", "Indigenous"]:
                n_pos = int((selected & (race_labels == label).to_numpy()).sum())
                log(f"        {label:<12s} n_positive={n_pos:,}")
                b1_rows.append(dict(metric="benchmark_positive_n", group=f"K{int(K*100)}_{label}",
                                     n_denominator=np.nan, n_numerator=np.nan, value=n_pos))

        b1_ok = True
    except Exception:
        log("\nERROR in B1 Section 6.1 recompute:")
        log(traceback.format_exc())

    if b1_rows:
        pd.DataFrame(b1_rows).to_csv(FULLFILE_DESC_OUT, index=False)
        log(f"\nSaved {FULLFILE_DESC_OUT} ({len(b1_rows)} rows).")

    # ------------------------------------------------------------------
    # B2: sample vs full-file marginals (race, sex, state, attendance, scores)
    # ------------------------------------------------------------------
    b2_ok = False
    try:
        log("\n" + "-" * 70)
        log("B2: sample vs full-file marginal comparison (Section 4.2 pending item)")
        log("-" * 70)

        log(f"\nLoading raw sample {SAMPLE_CSV} for its own (unfiltered) marginals ...")
        sample_df = pd.read_csv(SAMPLE_CSV, sep=";", encoding="latin-1",
                                 usecols=NEEDED_COLS, low_memory=False)
        sample_moments, sample_counts = compute_raw_marginals(sample_df)

        b2_rows = []

        def compare_categorical(name, full_rc: RunningCounts, samp_rc: RunningCounts):
            full_pct = full_rc.as_pct()
            samp_pct = samp_rc.as_pct()
            keys = sorted(set(full_pct) | set(samp_pct), key=lambda k: str(k))
            for k in keys:
                fp = full_pct.get(k, 0.0)
                sp = samp_pct.get(k, 0.0)
                b2_rows.append(dict(variable=name, category=str(k),
                                     full_file_pct=fp, sample_pct=sp,
                                     abs_diff_pp=abs(fp - sp)))

        compare_categorical("TP_COR_RACA", full_counts["TP_COR_RACA"], sample_counts["TP_COR_RACA"])
        compare_categorical("TP_SEXO", full_counts["TP_SEXO"], sample_counts["TP_SEXO"])
        compare_categorical("SG_UF_PROVA", full_counts["SG_UF_PROVA"], sample_counts["SG_UF_PROVA"])
        for c in PRESENCE_COLS:
            compare_categorical(c, full_counts[c], sample_counts[c])

        log("\n  Categorical marginals (max abs diff shown per variable):")
        b2_df_cat = pd.DataFrame(b2_rows)
        for name, g in b2_df_cat.groupby("variable"):
            log(f"    {name:<18s} max |diff| = {g['abs_diff_pp'].max():.3f}pp "
                f"(n_categories={len(g)})")

        log("\n  Score marginals (mean / SD, full file vs sample):")
        score_rows = []
        for c in SCORE_COLS:
            fm, fs = full_moments[c].mean(), full_moments[c].sd()
            sm, ss = sample_moments[c].mean(), sample_moments[c].sd()
            log(f"    {c:<16s} full mean={fm:7.2f} SD={fs:6.2f}   "
                f"sample mean={sm:7.2f} SD={ss:6.2f}   "
                f"diff_mean={abs(fm-sm):.3f}  diff_sd={abs(fs-ss):.3f}")
            score_rows.append(dict(variable=c, category="mean",
                                    full_file_pct=fm, sample_pct=sm, abs_diff_pp=abs(fm - sm)))
            score_rows.append(dict(variable=c, category="sd",
                                    full_file_pct=fs, sample_pct=ss, abs_diff_pp=abs(fs - ss)))

        b2_full = pd.concat([b2_df_cat, pd.DataFrame(score_rows)], ignore_index=True)
        b2_full.to_csv(SAMPLE_VS_FULL_OUT, index=False)
        log(f"\nSaved {SAMPLE_VS_FULL_OUT} ({len(b2_full)} rows).")
        b2_ok = True
    except Exception:
        log("\nERROR in B2 sample-vs-full comparison:")
        log(traceback.format_exc())

    log(f"\nB1+B2 total elapsed: {time.time()-t0:,.0f}s")
    return b1_ok, b2_ok


if __name__ == "__main__":
    logger = Logger(RESULTS_DIR / "taskB_log.txt", mode="a")
    try:
        run(logger)
    finally:
        logger.close()
