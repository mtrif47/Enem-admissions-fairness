"""
Task C: correct the Section 5.5 reference figures (Q001-Q004 "don't know"
rates, municipality counts), which the draft quotes from the raw 4% sample
rather than the population the encoders actually fit on.

Computes the same five figures on two populations:
  - the analytic sample (results/analytic_sample.parquet, N=123,879)
  - the training split only (results/split_indices.parquet, split=='train'),
    since that is what every encoder is actually fit on (Section 5.5:
    "All encoders, scalers and target-encoding maps are fitted on training
    data only").

Saves both to results/encoding_reference_figures.csv.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ANALYTIC_SAMPLE_PARQUET, RESULTS_DIR, Logger

SPLIT_PARQUET = RESULTS_DIR / "split_indices.parquet"
OUT_CSV = RESULTS_DIR / "encoding_reference_figures.csv"

DONTKNOW = {"Q001": "H", "Q002": "H", "Q003": "F", "Q004": "F"}


def compute_figures(df: pd.DataFrame) -> dict:
    n = len(df)
    out = {"n": n}
    for col, code in DONTKNOW.items():
        rate = 100.0 * (df[col] == code).mean()
        out[f"{col}_dontknow_pct"] = rate

    muni_counts = df["CO_MUNICIPIO_PROVA"].value_counts()
    out["n_municipalities"] = int(muni_counts.shape[0])
    out["n_municipalities_below_30"] = int((muni_counts < 30).sum())
    out["min_obs_per_municipality"] = int(muni_counts.min())
    return out


def run(logger: Logger):
    log = logger.log
    log("\n" + "=" * 70)
    log("TASK C - CORRECTED SECTION 5.5 ENCODING REFERENCE FIGURES")
    log("=" * 70)

    analytic = pd.read_parquet(ANALYTIC_SAMPLE_PARQUET)
    split = pd.read_parquet(SPLIT_PARQUET)[["NU_INSCRICAO", "split"]]
    merged = analytic.merge(split, on="NU_INSCRICAO", how="inner")
    assert len(merged) == len(analytic), "split merge dropped rows"
    train = merged[merged["split"] == "train"]

    log(f"\nAnalytic sample N={len(analytic):,}; training split N={len(train):,} "
        f"({100*len(train)/len(analytic):.1f}%).")

    rows = []
    for pop_name, pop_df in [("analytic_sample", analytic), ("training_split", train)]:
        fig = compute_figures(pop_df)
        log(f"\n[{pop_name}] N={fig['n']:,}")
        for col in DONTKNOW:
            log(f"  {col} don't-know rate: {fig[f'{col}_dontknow_pct']:.2f}%")
        log(f"  Distinct municipalities: {fig['n_municipalities']:,}")
        log(f"  Municipalities with <30 observations: {fig['n_municipalities_below_30']:,}")
        log(f"  Minimum observations per municipality: {fig['min_obs_per_municipality']}")

        row = {"population": pop_name, "n": fig["n"]}
        for col in DONTKNOW:
            row[f"{col}_dontknow_pct"] = fig[f"{col}_dontknow_pct"]
        row["n_municipalities"] = fig["n_municipalities"]
        row["n_municipalities_below_30"] = fig["n_municipalities_below_30"]
        row["min_obs_per_municipality"] = fig["min_obs_per_municipality"]
        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False)
    log(f"\nSaved {OUT_CSV}:")
    log(out_df.to_string(index=False))

    log("\nFor reference, the raw 4% sample (203,807 rows, pre-cascade) figures "
        "the draft actually quotes -- Q001=8.75%, Q002=2.86%, Q003=11.35%, "
        "Q004=8.56%, 1,726 municipalities, 484 below 30, min=1 -- are NOT the "
        "analytic-sample or training-split figures above; they describe a "
        "different, larger, unfiltered population. This script deliberately "
        "does not touch final_paper_draft.md.")

    return True


if __name__ == "__main__":
    logger = Logger(RESULTS_DIR / "taskB_log.txt", mode="a")
    try:
        run(logger)
    finally:
        logger.close()
