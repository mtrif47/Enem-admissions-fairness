"""
Task B3: encoding (Section 5.5 encoding table) and the 60/20/20 split
(Section 5.5, stratified jointly by race and composite decile).

Feature-block assembly (Section 5.4) lives here too, since it consumes the
same encoders. Municipality (CO_MUNICIPIO_PROVA) target encoding is
implemented per the encoding table's recipe (smoothed, training-fold only,
floor at 30 obs falling back to the state mean) but is deliberately NOT
wired into any feature block yet -- see taskB_log.txt for the open
question about whether blocks B2-B6 use it in the PRIMARY grid or only in
the separate Section 5.8 "Geography" robustness axis.
"""

import string
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from common import SPLIT_SEED, RESULTS_DIR

LETTER_TO_ORDINAL = {c: i + 1 for i, c in enumerate(string.ascii_uppercase)}

# Q001/Q002: A-G ordered, H = "don't know". Q003/Q004: A-E ordered, F = "don't know".
DONTKNOW_COLS = {
    "Q001": "H", "Q002": "H",
    "Q003": "F", "Q004": "F",
}

# Genuinely ordered integer scales (already numeric, or letter-coded ordinal
# with no "don't know" sentinel).
ORDINAL_LETTER_COLS = ["Q006"] + [f"Q{str(i).zfill(3)}" for i in range(7, 26)]
ORDINAL_NUMERIC_COLS = ["TP_FAIXA_ETARIA", "Q005"]

ONEHOT_NOMINAL_COLS = ["SG_UF_PROVA", "TP_ESTADO_CIVIL", "TP_NACIONALIDADE", "TP_ST_CONCLUSAO"]

ANO_CONCLUIU_COL = "TP_ANO_CONCLUIU"  # 0 = "not concluded", not a year


def build_split(df: pd.DataFrame, seed: int = SPLIT_SEED):
    """60/20/20 split stratified jointly by race_label and composite decile.

    Returns df with an added 'split' column in {'train','val','test'} and
    logs the stratum used.
    """
    df = df.copy()
    decile = pd.qcut(df["composite_raw_equal"], 10, labels=False, duplicates="drop")
    df["composite_decile"] = decile
    stratum = df["race_label"].astype(str) + "_" + decile.astype(str)

    idx = df.index.to_numpy()
    train_idx, temp_idx = train_test_split(
        idx, test_size=0.40, random_state=seed, stratify=stratum.to_numpy()
    )
    temp_stratum = stratum.loc[temp_idx].to_numpy()
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=seed, stratify=temp_stratum
    )

    df["split"] = "train"
    df.loc[val_idx, "split"] = "val"
    df.loc[test_idx, "split"] = "test"
    return df


def _ordinal_letter_map(series: pd.Series) -> pd.Series:
    return series.map(LETTER_TO_ORDINAL).astype(float)


def encode_dontknow_block(df: pd.DataFrame, col: str, train_mask: np.ndarray):
    """Ordinal + missingness-flag treatment for Q001-Q004."""
    dk_code = DONTKNOW_COLS[col]
    is_dk = (df[col] == dk_code).to_numpy()
    ordinal = _ordinal_letter_map(df[col]).to_numpy()
    ordinal_known = ordinal[train_mask & ~is_dk]
    median_val = np.median(ordinal_known)
    ordinal_filled = np.where(is_dk, median_val, ordinal)
    flag = is_dk.astype(float)
    return pd.DataFrame({f"{col}_ord": ordinal_filled, f"{col}_dontknow": flag}, index=df.index)


def encode_ano_concluiu(df: pd.DataFrame, train_mask: np.ndarray):
    """Ordinal + flag treatment for TP_ANO_CONCLUIU (0 = not concluded)."""
    raw = df[ANO_CONCLUIU_COL].to_numpy().astype(float)
    is_missing = raw == 0
    known = raw[train_mask & ~is_missing]
    median_val = np.median(known)
    filled = np.where(is_missing, median_val, raw)
    flag = is_missing.astype(float)
    return pd.DataFrame(
        {f"{ANO_CONCLUIU_COL}_ord": filled, f"{ANO_CONCLUIU_COL}_missing": flag}, index=df.index
    )


def encode_ordinal_letters(df: pd.DataFrame, cols):
    out = {}
    for c in cols:
        out[f"{c}_ord"] = _ordinal_letter_map(df[c]).to_numpy()
    return pd.DataFrame(out, index=df.index)


def encode_ordinal_numeric(df: pd.DataFrame, cols):
    return df[cols].astype(float).copy()


def fit_onehot(df: pd.DataFrame, cols, train_mask: np.ndarray):
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    enc.fit(df.loc[train_mask, cols])
    return enc


def apply_onehot(enc: OneHotEncoder, df: pd.DataFrame, cols):
    arr = enc.transform(df[cols])
    names = enc.get_feature_names_out(cols)
    return pd.DataFrame(arr, columns=names, index=df.index)


class MunicipalityTargetEncoder:
    """Smoothed target encoding for CO_MUNICIPIO_PROVA, fit on training rows
    only. Municipalities with fewer than `floor` training observations fall
    back to their state's mean target. Smoothing parameter `m` blends the
    municipality mean toward the state mean for municipalities at or above
    the floor, per the "smoothed ... with a floor" description in Section
    5.5 (the floor value of 30 is specified there; the smoothing constant m
    is not, so m = floor = 30 is used, giving equal-weight blending at
    exactly the floor)."""

    def __init__(self, floor: int = 30, m: int = 30):
        self.floor = floor
        self.m = m

    def fit(self, muni: pd.Series, state: pd.Series, y: np.ndarray):
        df = pd.DataFrame({"muni": muni.to_numpy(), "state": state.to_numpy(), "y": y})
        self.global_mean_ = df["y"].mean()
        self.state_mean_ = df.groupby("state")["y"].mean().to_dict()
        muni_stats = df.groupby("muni")["y"].agg(["mean", "count"])
        self.muni_mean_ = {}
        for muni_id, row in muni_stats.iterrows():
            n = row["count"]
            st = df.loc[df["muni"] == muni_id, "state"].iloc[0]
            state_mean = self.state_mean_.get(st, self.global_mean_)
            if n < self.floor:
                self.muni_mean_[muni_id] = state_mean
            else:
                lam = n / (n + self.m)
                self.muni_mean_[muni_id] = lam * row["mean"] + (1 - lam) * state_mean
        return self

    def transform(self, muni: pd.Series, state: pd.Series) -> np.ndarray:
        out = np.empty(len(muni), dtype=float)
        muni_arr = muni.to_numpy()
        state_arr = state.to_numpy()
        for i, (m_id, st) in enumerate(zip(muni_arr, state_arr)):
            if m_id in self.muni_mean_:
                out[i] = self.muni_mean_[m_id]
            else:
                out[i] = self.state_mean_.get(st, self.global_mean_)
        return out
