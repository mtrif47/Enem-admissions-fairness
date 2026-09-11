"""
Shared infrastructure for Task E (Section 5.7 bias mitigation).

Both mitigations are applied on top of the PRIMARY B3/B4 fits (the ones
in results/main_grid.csv, trained on the full declared-race population),
not a restricted population -- Task E5 explicitly diffs mitigated
overlap@K against "the corresponding unmitigated model from
results/main_grid.csv," which is only a fair, like-for-like comparison if
the ranking population and capacity (N_test=24,776, k=floor(K*N)=2,477)
are identical to the primary grid's.

REWEIGHING (E1): Kamiran & Calders (2012) generalises directly to a
multi-valued group variable (weight(g,y) = P(g)P(y)/P(g,y)); the training
population here already spans all five declared-race groups (matching
B3/B4), so weights are computed over all five rather than collapsing
Asian/Indigenous into an artificial "other" bucket.

THRESHOLD ADJUSTMENT (E2/E3): Section 5.7 describes exactly two
group-specific thresholds ("White and Black/Pardo") and never mentions a
third rule for Asian/Indigenous -- consistent with equal opportunity
being inherently a two-group technique (Hardt et al. 2016). To keep
capacity exact at floor(K*N_test) (the same N as the primary grid, for
E5's comparison to be meaningful) while only touching the two contrast
groups, Asian/Indigenous candidates keep EXACTLY the selection decision
the unmitigated primary model already made for them (held fixed); the
two thresholds then reallocate only the remaining "White+BP" seats
(capacity_remaining = floor(K*N) - n_selected_asian_indigenous_unmitigated)
between White and Black/Pardo to satisfy the equal-opportunity target.

Search-and-transfer protocol, since a threshold fit on validation cannot
guarantee an EXACT count on a different dataset (test) by chance alone:
  1. On validation, grid-search the integer split (n_white, n_bp) with
     n_white + n_bp = capacity_remaining_val, picking whichever split's
     resulting TPR gap (TPR_white - TPR_bp) is closest to the target gap
     for this level (0 for E2's full equalisation; a value from the E3
     sweep otherwise).
  2. Convert the winning counts to SELECTION RATES within each group
     (n_white*/N_white_val, n_bp*/N_bp_val) and to the boundary SCORE
     value at that rank in each group (reported as "the two fitted
     thresholds").
  3. On test, apply those same rates to each group's OWN size
     (n_white_test = round(rate_white* * N_white_test); n_bp_test =
     capacity_remaining_test - n_white_test, forcing the exact residual)
     and select the top-ranked candidates by score within each group.
     This guarantees the realised total equals floor(K*N_test) exactly
     by construction, while the decision rule is still "learned on
     validation, applied to test."
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import TIE_BREAK_SEED


def fit_cell_weighted(X_train, y_train, model, params, sample_weight):
    """Same as robust_common.fit_cell, but passes sample_weight through to
    .fit() -- needed for E1's reweighing, not used elsewhere in Task D/B."""
    if model == "logistic_regression":
        scaler = StandardScaler().fit(X_train)
        Xs = scaler.transform(X_train)
        clf = LogisticRegression(penalty="l2", C=params["C"], solver="lbfgs", max_iter=2000)
        clf.fit(Xs, y_train, sample_weight=sample_weight)
        return clf, scaler
    elif model == "gradient_boosting":
        clf = HistGradientBoostingClassifier(
            max_depth=params["max_depth"], learning_rate=params["learning_rate"],
            max_iter=params["max_iter"], min_samples_leaf=params["min_samples_leaf"],
            early_stopping=True, n_iter_no_change=20, validation_fraction=0.1,
            random_state=TIE_BREAK_SEED,
        )
        clf.fit(X_train, y_train, sample_weight=sample_weight)
        return clf, None
    else:
        raise ValueError(model)


def kamiran_calders_weights(group: np.ndarray, y: np.ndarray) -> np.ndarray:
    """weight(g,y) = P(g)*P(y) / P(g,y), estimated by training frequencies,
    generalised to however many distinct group values are present."""
    n = len(y)
    weights = np.empty(n, dtype=float)
    p_y = {yy: (y == yy).mean() for yy in np.unique(y)}
    for g in np.unique(group):
        mask_g = group == g
        p_g = mask_g.mean()
        for yy in np.unique(y):
            mask_gy = mask_g & (y == yy)
            p_gy = mask_gy.mean()
            if p_gy > 0:
                weights[mask_gy] = (p_g * p_y[yy]) / p_gy
    return weights


def evaluate_from_selected(model_selected: np.ndarray, benchmark_selected: np.ndarray,
                            race_labels: pd.Series, K: float,
                            model_scores: np.ndarray = None):
    """Same metric set as modeling.evaluate_selection, but takes an
    ALREADY-DECIDED selection mask (needed for E2/E3's group-specific
    thresholds, which aren't a single top-K-by-score rule)."""
    n = len(model_selected)
    k = int(model_selected.sum())  # realised capacity, should equal floor(K*n) by construction

    overlap_at_k = (model_selected & benchmark_selected).sum() / max(k, 1)
    union = (model_selected | benchmark_selected).sum()
    jaccard = (model_selected & benchmark_selected).sum() / union if union else np.nan

    if model_scores is not None:
        pr_auc = average_precision_score(benchmark_selected, model_scores)
        try:
            roc_auc = roc_auc_score(benchmark_selected, model_scores)
        except ValueError:
            roc_auc = np.nan
    else:
        pr_auc = roc_auc = np.nan

    results = dict(overlap_at_k=overlap_at_k, jaccard=jaccard, pr_auc=pr_auc, roc_auc=roc_auc,
                    n=n, k=k)

    is_white = (race_labels == "White").to_numpy()
    is_bp = race_labels.isin(["Black", "Pardo"]).to_numpy()

    group_rows = []
    for label, mask in [("White", is_white), ("Black_Pardo", is_bp)]:
        n_g = int(mask.sum())
        p_g = benchmark_selected[mask].mean()
        q_g = model_selected[mask].mean()
        pos = benchmark_selected[mask]
        pred = model_selected[mask]
        fp = int((pred & ~pos).sum())
        fn = int((~pred & pos).sum())
        neg_g = int((~pos).sum())
        pos_g = int(pos.sum())
        fpr = fp / neg_g if neg_g else np.nan
        fnr = fn / pos_g if pos_g else np.nan
        tpr = 1 - fnr if not np.isnan(fnr) else np.nan
        net_realloc = q_g - p_g
        group_rows.append(dict(group=label, n=n_g, p_g=p_g, q_g=q_g,
                                net_reallocation=net_realloc, fpr=fpr, fnr=fnr, tpr=tpr))

    results["group_rows"] = group_rows
    results["rate_white_pct"] = 100 * group_rows[0]["q_g"]
    results["rate_bp_pct"] = 100 * group_rows[1]["q_g"]
    results["gap_pp"] = results["rate_white_pct"] - results["rate_bp_pct"]
    results["ratio"] = (results["rate_white_pct"] / results["rate_bp_pct"]
                         if results["rate_bp_pct"] > 0 else np.nan)
    results["bench_gap_pp"] = 100 * (group_rows[0]["p_g"] - group_rows[1]["p_g"])
    results["D"] = results["gap_pp"] - results["bench_gap_pp"]
    results["tpr_white"] = group_rows[0]["tpr"]
    results["tpr_bp"] = group_rows[1]["tpr"]
    results["tpr_gap"] = group_rows[0]["tpr"] - group_rows[1]["tpr"]

    zero_sum_terms = []
    for label in race_labels.unique():
        mask = (race_labels == label).to_numpy()
        n_g = mask.sum()
        p_g = benchmark_selected[mask].mean()
        q_g = model_selected[mask].mean()
        zero_sum_terms.append(n_g * (q_g - p_g))
    results["zero_sum_value"] = float(np.sum(zero_sum_terms))

    return results


def _rank_select_top_n(scores: np.ndarray, n: int, seed=TIE_BREAK_SEED):
    """Select the top n by score (ties broken by fixed-seed random draw at
    the boundary, matching common.build_topk's convention)."""
    n = max(0, min(n, len(scores)))
    if n == 0:
        return np.zeros(len(scores), dtype=bool)
    if n == len(scores):
        return np.ones(len(scores), dtype=bool)
    order = np.argsort(-scores, kind="mergesort")
    boundary_value = scores[order[n - 1]]
    strictly_above = scores > boundary_value
    at_boundary = scores == boundary_value
    n_gt = int(strictly_above.sum())
    n_from_tied = n - n_gt
    selected = strictly_above.copy()
    if n_from_tied > 0:
        tied_idx = np.flatnonzero(at_boundary)
        rng = np.random.default_rng(seed)
        chosen = rng.choice(tied_idx, size=n_from_tied, replace=False)
        selected[chosen] = True
    return selected


def group_tpr(scores: np.ndarray, bench: np.ndarray, n_select: int):
    """TPR achieved by selecting the top n_select candidates (by score)
    within this group, against this group's own benchmark labels."""
    selected = _rank_select_top_n(scores, n_select)
    n_pos = int(bench.sum())
    if n_pos == 0:
        return np.nan, selected
    tpr = (selected & bench).sum() / n_pos
    return tpr, selected


def _sorted_cum_tpr(scores: np.ndarray, bench: np.ndarray):
    """Sort descending ONCE; return (sorted_scores_desc, cum_tpr) where
    cum_tpr[n-1] = TPR from taking the top n by score (cum_tpr[-1]=0 slot
    prepended for n=0). Used so the grid search below is O(n log n) total
    instead of O(capacity * n log n) from re-sorting at every candidate n."""
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_bench = bench[order]
    n_pos = sorted_bench.sum()
    cum_hits = np.concatenate([[0], np.cumsum(sorted_bench)])  # cum_hits[n] = hits in top n
    cum_tpr = cum_hits / n_pos if n_pos > 0 else np.full(len(cum_hits), np.nan)
    return sorted_scores, cum_tpr


def search_threshold_split(scores_white_val, bench_white_val, scores_bp_val, bench_bp_val,
                            capacity_val, target_gap):
    """Grid search over n_white in [0, capacity_val] for the split whose
    TPR gap (white - bp) is closest to target_gap. Returns
    (n_white*, n_bp*, tpr_white*, tpr_bp*, tau_white, tau_bp).

    Vectorised via precomputed cumulative-TPR-by-rank arrays (one sort per
    group, O(1) per candidate n_white) rather than re-deriving a selection
    mask (and re-sorting) at every one of up to ~capacity_val grid points.
    """
    n_white_max = len(scores_white_val)
    n_bp_max = len(scores_bp_val)
    sorted_white, cum_tpr_white = _sorted_cum_tpr(scores_white_val, bench_white_val)
    sorted_bp, cum_tpr_bp = _sorted_cum_tpr(scores_bp_val, bench_bp_val)

    lo = max(0, capacity_val - n_bp_max)
    hi = min(capacity_val, n_white_max)
    n_w_grid = np.arange(lo, hi + 1)
    n_b_grid = capacity_val - n_w_grid

    tpr_w_grid = cum_tpr_white[n_w_grid]
    tpr_b_grid = cum_tpr_bp[n_b_grid]
    valid = ~(np.isnan(tpr_w_grid) | np.isnan(tpr_b_grid))
    if not valid.any():
        raise RuntimeError("threshold search found no feasible split")

    gap_grid = tpr_w_grid - tpr_b_grid
    dist_grid = np.abs(gap_grid - target_gap)
    dist_grid[~valid] = np.inf
    best_i = int(np.argmin(dist_grid))
    n_w, n_b = int(n_w_grid[best_i]), int(n_b_grid[best_i])
    tpr_w, tpr_b, gap = float(tpr_w_grid[best_i]), float(tpr_b_grid[best_i]), float(gap_grid[best_i])

    def boundary_score(sorted_scores_desc, n):
        if n <= 0:
            return np.inf
        if n >= len(sorted_scores_desc):
            return -np.inf
        return sorted_scores_desc[n - 1]

    tau_w = boundary_score(sorted_white, n_w)
    tau_b = boundary_score(sorted_bp, n_b)
    return dict(n_white=n_w, n_bp=n_b, tpr_white=tpr_w, tpr_bp=tpr_b, tpr_gap=gap,
                tau_white=tau_w, tau_bp=tau_b,
                rate_white=n_w / n_white_max, rate_bp=n_b / n_bp_max)


from common import RESULTS_DIR

MITIG_ALL_SCORES_PARQUET = RESULTS_DIR / "mitig_all_scores.parquet"
_e_score_records = []


def record_scores(task, variant, block, model, K, group_a, group_b,
                   row_ids, model_selected, benchmark_selected, group_values):
    """Same convention as robust_common.record_scores, but writes to a
    separate Task-E manifest (mitig_all_scores.parquet) so E5's bootstrap
    stays scoped to Task E cells."""
    _e_score_records.append(pd.DataFrame({
        "task": task, "variant": variant, "block": block, "model": model, "K": K,
        "group_a": group_a, "group_b": group_b,
        "row_id": row_ids, "model_selected": np.asarray(model_selected, dtype=bool),
        "benchmark_selected": np.asarray(benchmark_selected, dtype=bool),
        "group_value": group_values,
    }))


def flush_scores(mode="a"):
    if not _e_score_records:
        return None
    new_df = pd.concat(_e_score_records, ignore_index=True)
    if mode == "a" and MITIG_ALL_SCORES_PARQUET.exists():
        old_df = pd.read_parquet(MITIG_ALL_SCORES_PARQUET)
        keys_new = new_df[["task", "variant"]].drop_duplicates()
        merge_key = old_df.merge(keys_new, on=["task", "variant"], how="left", indicator=True)
        old_df = old_df[merge_key["_merge"].to_numpy() == "left_only"]
        combined = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_parquet(MITIG_ALL_SCORES_PARQUET, index=False)
    _e_score_records.clear()
    return len(combined)


def apply_split_to_test(scores_white_test, scores_bp_test, rate_white, rate_bp, capacity_test):
    """Transfer validation-fitted group RATES to test, forcing the exact
    integer total (capacity_test) by construction."""
    n_white_test = int(round(rate_white * len(scores_white_test)))
    n_white_test = max(0, min(n_white_test, len(scores_white_test), capacity_test))
    n_bp_test = capacity_test - n_white_test
    n_bp_test = max(0, min(n_bp_test, len(scores_bp_test)))
    # if bp couldn't absorb the residual (group too small), push remainder back to white
    residual = capacity_test - n_white_test - n_bp_test
    if residual > 0:
        n_white_test = min(n_white_test + residual, len(scores_white_test))
    sel_white = _rank_select_top_n(scores_white_test, n_white_test)
    sel_bp = _rank_select_top_n(scores_bp_test, n_bp_test)
    return sel_white, sel_bp, n_white_test, n_bp_test
