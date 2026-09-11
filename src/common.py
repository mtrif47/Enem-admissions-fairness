"""
Shared constants and helpers for Task B scripts.

Restriction cascade, race coding, composite construction and top-K
tie-breaking mirror src/task_a_data_prep.py exactly (Section 4.2 / 4.6 of
final_paper_draft.md). Kept separate from task_a_data_prep.py so Task A's
already-verified script and output are not touched by Task B work.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DADOS_DIR = ROOT / "DADOS"
RESULTS_DIR = ROOT / "results"

SAMPLE_CSV = DADOS_DIR / "enem_2019_sample.csv"
FULLFILE_CSV = DADOS_DIR / "MICRODADOS_ENEM_2019.csv"
ANALYTIC_SAMPLE_PARQUET = RESULTS_DIR / "analytic_sample.parquet"

TIE_BREAK_SEED = 20190101
SPLIT_SEED = 20190101
BOOTSTRAP_SEED = 20190101
N_BOOTSTRAP = 1000

SCORE_COLS = ["NU_NOTA_CN", "NU_NOTA_CH", "NU_NOTA_LC", "NU_NOTA_MT", "NU_NOTA_REDACAO"]
PRESENCE_COLS = ["TP_PRESENCA_CN", "TP_PRESENCA_CH", "TP_PRESENCA_LC", "TP_PRESENCA_MT"]

RACE_MAP = {0: "Not declared", 1: "White", 2: "Black", 3: "Pardo", 4: "Asian", 5: "Indigenous"}

# Columns needed anywhere in Task B (cascade + descriptives + B2 marginals +
# B3-B5 feature blocks). Reading only these from the 76-column full file
# keeps the chunked read memory-bounded.
NEEDED_COLS = sorted(set(
    [
        "IN_TREINEIRO",
        "TP_ST_CONCLUSAO",
        "TP_COR_RACA",
        "TP_SEXO",
        "SG_UF_PROVA",
        "CO_MUNICIPIO_PROVA",
        "TP_FAIXA_ETARIA",
        "TP_ESTADO_CIVIL",
        "TP_NACIONALIDADE",
        "TP_ANO_CONCLUIU",
        "TP_ESCOLA",
    ]
    + PRESENCE_COLS
    + SCORE_COLS
    + [f"Q{str(i).zfill(3)}" for i in range(1, 26)]
))


def apply_cascade(df, steps=(1, 2, 3, 4, 5), counts=None):
    """Apply the Section 4.2 restriction cascade steps in order.

    `steps` lets a caller stop early (e.g. only steps 1-2 for the
    completion-rate population in Section 4.5). `counts`, if given a list,
    has len(df) appended after each requested step (step 0 is whatever was
    passed in, not appended here).
    """
    if 1 in steps:
        df = df[df["IN_TREINEIRO"] == 0]
        if counts is not None:
            counts.append(len(df))
    if 2 in steps:
        df = df[df["TP_ST_CONCLUSAO"].isin([1, 2])]
        if counts is not None:
            counts.append(len(df))
    if 3 in steps:
        df = df[(df[PRESENCE_COLS] == 1).all(axis=1)]
        if counts is not None:
            counts.append(len(df))
    if 4 in steps:
        df = df[df[SCORE_COLS].notna().all(axis=1)]
        if counts is not None:
            counts.append(len(df))
    if 5 in steps:
        df = df[df["TP_COR_RACA"] != 0]
        if counts is not None:
            counts.append(len(df))
    return df


def add_composite(df, score_cols=SCORE_COLS, out_col="composite_raw_equal"):
    df = df.copy()
    df[out_col] = df[score_cols].mean(axis=1)
    return df


def build_topk(composite: np.ndarray, K: float, seed: int = TIE_BREAK_SEED):
    """Select the top floor(K*N) by composite score, tie-breaking at the
    boundary with a fixed-seed random draw. Returns (selected_bool, tie_info).
    """
    n = len(composite)
    k = int(np.floor(K * n))
    if k <= 0:
        return np.zeros(n, dtype=bool), dict(boundary_value=np.nan, n_tied=0, n_tied_selected=0)

    order = np.argsort(-composite, kind="mergesort")
    sorted_scores = composite[order]
    boundary_value = sorted_scores[k - 1]

    strictly_above = composite > boundary_value
    at_boundary = composite == boundary_value

    n_gt = int(strictly_above.sum())
    n_tied = int(at_boundary.sum())
    n_slots_from_tied = k - n_gt

    selected = strictly_above.copy()
    if n_slots_from_tied > 0:
        tied_idx = np.flatnonzero(at_boundary)
        rng = np.random.default_rng(seed)
        chosen = rng.choice(tied_idx, size=n_slots_from_tied, replace=False)
        selected[chosen] = True

    tie_info = dict(boundary_value=float(boundary_value), n_tied=n_tied,
                     n_tied_selected=n_slots_from_tied)
    return selected, tie_info


def selection_table(race_labels: pd.Series, selected: np.ndarray):
    """White vs Black/Pardo rate, gap (pp) and ratio for one K."""
    is_white = (race_labels == "White").to_numpy()
    is_bp = race_labels.isin(["Black", "Pardo"]).to_numpy()
    rate_white = 100.0 * selected[is_white].mean()
    rate_bp = 100.0 * selected[is_bp].mean()
    gap = rate_white - rate_bp
    ratio = rate_white / rate_bp if rate_bp > 0 else np.nan
    return dict(rate_white=rate_white, rate_black_pardo=rate_bp, gap_pp=gap, ratio=ratio)


class Logger:
    """Appends to a log file while also printing, and never raises."""

    def __init__(self, path: Path, mode="a"):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(path, mode)

    def log(self, msg=""):
        print(msg)
        self._fh.write(str(msg) + "\n")
        self._fh.flush()

    def close(self):
        self._fh.close()
