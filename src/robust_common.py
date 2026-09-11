"""
Shared infrastructure for Task D (Section 5.8 robustness suite).

Modelling-protocol decision, applied across D1-D6 and logged here once
rather than repeated in every script: Section 5.5 explicitly tells D1
("Capacity") to reuse the hyperparameters tuned once at the primary
specification rather than retune per K, giving the reason "retuning would
confound differences in capacity with differences in tuning." That is a
general principle about isolating the manipulated factor, not a K-specific
rule, and it is standard robustness-check practice besides (change ONE
thing, hold the modelling protocol fixed). Re-running the full 108-point
HGB grid for every one of D2-D6's ~35 additional cells would cost well
over an hour of compute for what the spec never asks to be retuned. This
module therefore reuses each block's Section-5.5-tuned hyperparameters
(from results/main_grid.csv, the primary K=10% grid) for every cell in
D1-D6: only the manipulated factor (capacity, composite, geography
encoding, protected group, ranking population, or absence handling)
changes; the learner and its hyperparameters do not.

Also centralises: fitting with fixed hyperparameters (no grid search),
recording long-format test-set scores for Task D7's bootstrap, and the
group-rate/gap/ratio/event-count computation shared by every D-script.
"""

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS_DIR, TIE_BREAK_SEED, build_topk
import modeling as mdl
import features as feat

MAIN_GRID_CSV = RESULTS_DIR / "main_grid.csv"
ALL_SCORES_PARQUET = RESULTS_DIR / "robust_all_scores.parquet"


def load_primary_params():
    """block -> model -> params dict, from the Task B5 primary K=10% grid."""
    df = pd.read_csv(MAIN_GRID_CSV)
    out = {}
    for _, row in df.iterrows():
        out.setdefault(row["block"], {})[row["model"]] = ast.literal_eval(row["best_params"])
    return out


def fit_cell(X_train, y_train, model, params):
    """Fit with FIXED hyperparameters (no grid search). Returns (fitted_model, scaler_or_None)."""
    if model == "logistic_regression":
        clf, scaler = mdl.fit_logreg(X_train, y_train, C=params["C"])
        return clf, scaler
    elif model == "gradient_boosting":
        clf = mdl.fit_hgb(X_train, y_train, params)
        return clf, None
    else:
        raise ValueError(model)


def score_cell(clf, scaler, X):
    if scaler is not None:
        X = scaler.transform(X)
    return clf.predict_proba(X)[:, 1]


def group_metrics(model_selected: np.ndarray, benchmark_selected: np.ndarray,
                   mask_a: np.ndarray, mask_b: np.ndarray):
    """Rate/gap/ratio/event-counts for a generic two-group contrast
    (a minus b), e.g. White vs Black/Pardo, or F vs M."""
    n_a, n_b = int(mask_a.sum()), int(mask_b.sum())
    rate_a_model = 100 * model_selected[mask_a].mean()
    rate_b_model = 100 * model_selected[mask_b].mean()
    rate_a_bench = 100 * benchmark_selected[mask_a].mean()
    rate_b_bench = 100 * benchmark_selected[mask_b].mean()
    gap_model = rate_a_model - rate_b_model
    gap_bench = rate_a_bench - rate_b_bench
    ratio_model = rate_a_model / rate_b_model if rate_b_model > 0 else np.nan
    n_pos_a = int(model_selected[mask_a].sum())
    n_pos_b = int(model_selected[mask_b].sum())
    n_bench_pos_a = int(benchmark_selected[mask_a].sum())
    n_bench_pos_b = int(benchmark_selected[mask_b].sum())
    return dict(
        n_a=n_a, n_b=n_b, rate_a_pct=rate_a_model, rate_b_pct=rate_b_model,
        gap_pp=gap_model, ratio=ratio_model, bench_gap_pp=gap_bench,
        bench_rate_a_pct=rate_a_bench, bench_rate_b_pct=rate_b_bench,
        n_model_positive_a=n_pos_a, n_model_positive_b=n_pos_b,
        n_bench_positive_a=n_bench_pos_a, n_bench_positive_b=n_bench_pos_b,
        D=gap_model - gap_bench,
    )


def overlap_at_k(model_selected, benchmark_selected, k):
    return (model_selected & benchmark_selected).sum() / k if k > 0 else np.nan


_score_records = []


def record_scores(task, variant, block, model, K, group_a, group_b,
                   row_ids, model_selected, benchmark_selected, group_values):
    """Append long-format rows for later bootstrap by Task D7.

    Stores the already-computed boolean `model_selected` label (not a
    continuous score) for each row -- bootstrap only ever resamples rows
    and averages their FIXED selected/not-selected labels, it never
    re-thresholds. This matters because some cells (Task D1's "global"
    definition) determine model_selected from a top-K rule applied over a
    DIFFERENT population than the one being reported here; storing a raw
    score and re-deriving the label generically in D7 would silently
    reproduce the wrong (local) definition instead.

    `group_values` is a string array aligned with row_ids/model_selected,
    taking values in {group_a, group_b, ...anything else, dropped at
    bootstrap time}.
    """
    _score_records.append(pd.DataFrame({
        "task": task, "variant": variant, "block": block, "model": model, "K": K,
        "group_a": group_a, "group_b": group_b,
        "row_id": row_ids, "model_selected": np.asarray(model_selected, dtype=bool),
        "benchmark_selected": np.asarray(benchmark_selected, dtype=bool),
        "group_value": group_values,
    }))


def flush_scores(mode="a"):
    """Write accumulated score records to the shared parquet (append)."""
    if not _score_records:
        return
    new_df = pd.concat(_score_records, ignore_index=True)
    if mode == "a" and ALL_SCORES_PARQUET.exists():
        old_df = pd.read_parquet(ALL_SCORES_PARQUET)
        # de-duplicate: drop any existing rows for (task, variant) keys we're
        # about to rewrite, so re-running a script is idempotent
        keys_new = new_df[["task", "variant"]].drop_duplicates()
        merge_key = old_df.merge(keys_new, on=["task", "variant"], how="left", indicator=True)
        old_df = old_df[merge_key["_merge"].to_numpy() == "left_only"]
        combined = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_parquet(ALL_SCORES_PARQUET, index=False)
    _score_records.clear()
    return len(combined)
