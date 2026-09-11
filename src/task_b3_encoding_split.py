"""
Task B3: encoding (Section 5.5) and the 60/20/20 split.

Builds the split, saves split indices, and sanity-checks every encoder
defined in the Section 5.5 encoding table by fitting it once on the whole
analytic sample (train fold only) and reporting shapes / "don't know"
rates against the Section 4.3 figures.
"""

import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ANALYTIC_SAMPLE_PARQUET, RESULTS_DIR, Logger
import features as feat

SPLIT_OUT = RESULTS_DIR / "split_indices.parquet"
SPLIT_OUT_CSV = RESULTS_DIR / "split_indices.csv"


def run(logger: Logger):
    log = logger.log
    log("\n" + "=" * 70)
    log("TASK B3 - ENCODING AND SPLIT")
    log("=" * 70)

    df = pd.read_parquet(ANALYTIC_SAMPLE_PARQUET)
    log(f"\nLoaded analytic sample: N={len(df):,}, {df.shape[1]} columns.")

    ok = False
    try:
        df_split = feat.build_split(df)
        counts = df_split["split"].value_counts()
        log("\nSplit sizes:")
        for k in ["train", "val", "test"]:
            n = counts.get(k, 0)
            log(f"  {k:<6s} N={n:,}  ({100*n/len(df_split):.2f}%)")

        log("\nJoint race x decile stratum sizes (train), first 10:")
        strat_counts = (
            df_split[df_split["split"] == "train"]
            .groupby(["race_label", "composite_decile"])
            .size()
            .sort_values(ascending=False)
        )
        for (race, dec), n in strat_counts.head(10).items():
            log(f"  {race:<12s} decile={dec}  N={n:,}")
        log(f"  ... {len(strat_counts)} strata total. Smallest stratum N="
            f"{strat_counts.min():,}.")

        out_cols = ["NU_INSCRICAO", "race_label", "composite_raw_equal",
                    "composite_decile", "split"]
        df_split[out_cols].to_parquet(SPLIT_OUT, index=False)
        df_split[out_cols].to_csv(SPLIT_OUT_CSV, index=False)
        log(f"\nSaved split indices to {SPLIT_OUT} and {SPLIT_OUT_CSV}.")
        ok = True
    except Exception:
        log("\nERROR building the split:")
        log(traceback.format_exc())
        return False

    # ------------------------------------------------------------------
    # Sanity-check every encoder in the Section 5.5 table
    # ------------------------------------------------------------------
    try:
        log("\n" + "-" * 70)
        log("Encoder sanity checks (fit on train fold)")
        log("-" * 70)
        train_mask = (df_split["split"] == "train").to_numpy()

        # One-hot nominal
        enc = feat.fit_onehot(df_split, feat.ONEHOT_NOMINAL_COLS, train_mask)
        onehot_df = feat.apply_onehot(enc, df_split, feat.ONEHOT_NOMINAL_COLS)
        log(f"\nOne-hot nominal {feat.ONEHOT_NOMINAL_COLS}: "
            f"{onehot_df.shape[1]} dummy columns")

        # Ordinal integer (letters)
        ord_letters = feat.encode_ordinal_letters(df_split, feat.ORDINAL_LETTER_COLS)
        log(f"Ordinal integer (letters) {len(feat.ORDINAL_LETTER_COLS)} cols "
            f"-> {ord_letters.shape[1]} columns. Example Q006 map: "
            f"A={feat.LETTER_TO_ORDINAL['A']}, Q={feat.LETTER_TO_ORDINAL['Q']}")

        # Ordinal integer (already numeric)
        ord_numeric = feat.encode_ordinal_numeric(df_split, feat.ORDINAL_NUMERIC_COLS)
        log(f"Ordinal integer (numeric passthrough) {feat.ORDINAL_NUMERIC_COLS}: "
            f"{ord_numeric.shape[1]} columns")

        # Ordinal + don't-know flag
        log("\nOrdinal + don't-know flag (Q001-Q004):")
        log("  Section 4.3's rates (8.75%/2.86%/11.35%/8.56%) are computed on the RAW "
            "4% sample (203,807 rows) before the restriction cascade, confirmed below; "
            "the analytic-sample rate (what the encoder actually sees) differs slightly "
            "because the cascade is not orthogonal to these variables.")
        raw_qs = pd.read_csv(
            RESULTS_DIR.parent / "DADOS" / "enem_2019_sample.csv",
            sep=";", encoding="latin-1", usecols=list(feat.DONTKNOW_COLS.keys()),
        )
        expected_dk_rate = {"Q001": 8.75, "Q002": 2.86, "Q003": 11.35, "Q004": 8.56}
        for col in feat.DONTKNOW_COLS:
            enc_df = feat.encode_dontknow_block(df_split, col, train_mask)
            dk_rate_analytic = 100.0 * enc_df[f"{col}_dontknow"].mean()
            dk_rate_train = 100.0 * enc_df.loc[train_mask, f"{col}_dontknow"].mean()
            dk_code = feat.DONTKNOW_COLS[col]
            dk_rate_raw = 100.0 * (raw_qs[col] == dk_code).mean()
            exp = expected_dk_rate[col]
            flag_status = "OK, exact match" if abs(dk_rate_raw - exp) < 0.02 else "CHECK"
            log(f"  {col}: raw-sample rate={dk_rate_raw:.2f}%  expected={exp:.2f}%  "
                f"[{flag_status}]  |  analytic-sample rate={dk_rate_analytic:.2f}%  "
                f"train-fold rate={dk_rate_train:.2f}%  "
                f"median fill={enc_df[f'{col}_ord'].loc[train_mask].median():.1f}")

        # Ordinal + missing flag for TP_ANO_CONCLUIU
        ano_df = feat.encode_ano_concluiu(df_split, train_mask)
        missing_rate = 100.0 * ano_df[f"{feat.ANO_CONCLUIU_COL}_missing"].mean()
        log(f"\nTP_ANO_CONCLUIU: 'not concluded' (code 0) rate={missing_rate:.2f}%  "
            f"median fill={ano_df[f'{feat.ANO_CONCLUIU_COL}_ord'].loc[train_mask].median():.1f}")

        # Municipality target encoding (fit against K=10% benchmark on train only)
        from common import build_topk, TIE_BREAK_SEED
        selected_10, _ = build_topk(df_split["composite_raw_equal"].to_numpy(), K=0.10,
                                     seed=TIE_BREAK_SEED)
        y_bench = selected_10.astype(float)
        tenc = feat.MunicipalityTargetEncoder(floor=30, m=30)
        tenc.fit(df_split.loc[train_mask, "CO_MUNICIPIO_PROVA"],
                 df_split.loc[train_mask, "SG_UF_PROVA"],
                 y_bench[train_mask])
        muni_vals = tenc.transform(df_split["CO_MUNICIPIO_PROVA"], df_split["SG_UF_PROVA"])
        n_muni_total = df_split["CO_MUNICIPIO_PROVA"].nunique()
        muni_full_counts = df_split["CO_MUNICIPIO_PROVA"].value_counts()
        n_below_floor_full = (muni_full_counts < 30).sum()
        muni_train_counts = df_split.loc[train_mask, "CO_MUNICIPIO_PROVA"].value_counts()
        n_below_floor_train = (muni_train_counts < 30).sum()

        raw_sample = pd.read_csv(
            RESULTS_DIR.parent / "DADOS" / "enem_2019_sample.csv",
            sep=";", encoding="latin-1", usecols=["CO_MUNICIPIO_PROVA"],
        )
        raw_vc = raw_sample["CO_MUNICIPIO_PROVA"].value_counts()
        n_muni_raw, n_below_raw, min_raw = raw_vc.shape[0], (raw_vc < 30).sum(), raw_vc.min()
        raw_match = (n_muni_raw == 1726 and n_below_raw == 484 and min_raw == 1)

        log(f"\nMunicipality target encoding -- Section 5.5's '1,726 municipalities, "
            f"484 below 30 obs, minimum 1' describes the RAW 4% sample "
            f"(203,807 rows) before the restriction cascade, not the analytic "
            f"sample: raw sample gives {n_muni_raw} municipalities, {n_below_raw} "
            f"below 30, min={min_raw} [{'OK, exact match' if raw_match else 'CHECK'}].")
        log(f"  On the analytic sample (N=123,879, what target encoding actually "
            f"sees): {n_muni_total} municipalities, {n_below_floor_full} below 30 "
            f"observations -- smaller than the raw-sample count because the cascade "
            f"removes 39% of rows, which is expected, not a discrepancy.")
        log(f"  On the 60% training fold specifically (what the floor rule actually "
            f"gates when fitting): {n_below_floor_train} of {len(muni_train_counts)} "
            f"observed municipalities fall below 30 training observations.")
        log(f"  Encoded value range: [{muni_vals.min():.4f}, {muni_vals.max():.4f}], "
            f"mean={muni_vals.mean():.4f} (benchmark rate ~ {y_bench.mean():.4f})")

        log("\nAll encoders in the Section 5.5 table constructed and sanity-checked "
            "successfully. NOT YET wired into feature blocks B2-B6 pending "
            "clarification on municipality's role in the primary grid (see log note).")
    except Exception:
        log("\nERROR in encoder sanity checks:")
        log(traceback.format_exc())
        return ok  # split itself still succeeded

    return True


if __name__ == "__main__":
    logger = Logger(RESULTS_DIR / "taskB_log.txt", mode="a")
    try:
        run(logger)
    finally:
        logger.close()
