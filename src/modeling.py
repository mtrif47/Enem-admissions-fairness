"""
Task B4/B5: feature-block assembly, model fitting/tuning, and the
Section 5.2/5.6 evaluation metrics (overlap@K, Jaccard, PR-AUC, ROC-AUC,
selection rates, and the FPR/FNR reallocation decomposition).

Top-K within an evaluation split is always computed WITHIN that split
(same capacity as the benchmark, applied to that split's own N), per
Section 5.1: "selects the top floor(KN) by that score, at the same
capacity as the benchmark." Both S^0 and the predictive rule are ranked
and thresholded within whichever split is being evaluated (see the design
note in task_b4_smoke_test.py on why this, rather than subsetting the
whole-sample benchmark, is what makes the Section 5.2 zero-sum identity
hold exactly).
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as feat
from common import TIE_BREAK_SEED

LOGREG_C_GRID = [1e-3, 1e-2, 1e-1, 1, 10, 1e2]

ALL_BLOCKS = ["B1", "B2", "B3", "B4", "B5", "B6"]

# Section 5.4 feature-block definitions:
#   B1 Administrative -- age band, completion status, grad year, marital
#       status, nationality, state.
#   B2 + Geography     -- B1 + municipality (target-encoded; see the
#       AskUserQuestion resolution recorded in taskB_log.txt: municipality
#       is included in the primary B1-B6 progression, per Section 5.4's
#       literal block definitions).
#   B3 + Socioeconomic -- B2 + Q001-Q025.
#   B4 + Race          -- B3 + TP_COR_RACA.
#   B5 + Sex           -- B3 + TP_SEXO.
#   B6 + Both          -- B3 + TP_COR_RACA + TP_SEXO.
_STATE_ONEHOT = ["SG_UF_PROVA", "TP_ESTADO_CIVIL", "TP_NACIONALIDADE", "TP_ST_CONCLUSAO"]
BLOCK_ONEHOT = {b: _STATE_ONEHOT for b in ALL_BLOCKS}

BLOCK_ORDINAL_NUMERIC = {b: ["TP_FAIXA_ETARIA"] for b in ALL_BLOCKS}

BLOCK_ANO_CONCLUIU = {b: True for b in ALL_BLOCKS}

BLOCK_MUNICIPALITY = {b: (b != "B1") for b in ALL_BLOCKS}

BLOCK_Q_BLOCK = {b: (b not in ("B1", "B2")) for b in ALL_BLOCKS}

BLOCK_RACE = {b: False for b in ALL_BLOCKS}
BLOCK_RACE["B4"] = True
BLOCK_RACE["B6"] = True

BLOCK_SEX = {b: False for b in ALL_BLOCKS}
BLOCK_SEX["B5"] = True
BLOCK_SEX["B6"] = True


def build_block_features(df_split: pd.DataFrame, block: str, train_mask: np.ndarray,
                          te_target: np.ndarray = None):
    """Assemble the design matrix for one Section 5.4 feature block.
    Fits all encoders on train_mask rows only; transforms all rows.

    `te_target` is the binary target used to fit the municipality smoothed
    target encoder for blocks B2-B6 (the primary-capacity, K=10% benchmark
    indicator on the train rows -- required whenever the block includes
    municipality, per BLOCK_MUNICIPALITY).

    Returns (X: np.ndarray, feature_names: list[str]).
    """
    pieces = []
    names = []

    onehot_cols = BLOCK_ONEHOT[block]
    enc = feat.fit_onehot(df_split, onehot_cols, train_mask)
    onehot_df = feat.apply_onehot(enc, df_split, onehot_cols)
    pieces.append(onehot_df.to_numpy())
    names.extend(onehot_df.columns.tolist())

    ord_num_cols = BLOCK_ORDINAL_NUMERIC[block]
    ord_num_df = feat.encode_ordinal_numeric(df_split, ord_num_cols)
    pieces.append(ord_num_df.to_numpy())
    names.extend(ord_num_df.columns.tolist())

    if BLOCK_MUNICIPALITY[block]:
        if te_target is None:
            raise ValueError(f"block {block} requires te_target for municipality "
                              f"target encoding")
        tenc = feat.MunicipalityTargetEncoder(floor=30, m=30)
        tenc.fit(df_split.loc[train_mask, "CO_MUNICIPIO_PROVA"],
                 df_split.loc[train_mask, "SG_UF_PROVA"], te_target)
        muni_vals = tenc.transform(df_split["CO_MUNICIPIO_PROVA"], df_split["SG_UF_PROVA"])
        pieces.append(muni_vals.reshape(-1, 1))
        names.append("CO_MUNICIPIO_PROVA_target_enc")

    if BLOCK_ANO_CONCLUIU[block]:
        ano_df = feat.encode_ano_concluiu(df_split, train_mask)
        pieces.append(ano_df.to_numpy())
        names.extend(ano_df.columns.tolist())

    if BLOCK_Q_BLOCK[block]:
        ord_letters_df = feat.encode_ordinal_letters(df_split, feat.ORDINAL_LETTER_COLS)
        pieces.append(ord_letters_df.to_numpy())
        names.extend(ord_letters_df.columns.tolist())

        q005_df = feat.encode_ordinal_numeric(df_split, ["Q005"])
        pieces.append(q005_df.to_numpy())
        names.extend(q005_df.columns.tolist())

        for col in feat.DONTKNOW_COLS:
            dk_df = feat.encode_dontknow_block(df_split, col, train_mask)
            pieces.append(dk_df.to_numpy())
            names.extend(dk_df.columns.tolist())

    if BLOCK_RACE[block]:
        race_enc = feat.fit_onehot(df_split, ["TP_COR_RACA"], train_mask)
        race_df = feat.apply_onehot(race_enc, df_split, ["TP_COR_RACA"])
        pieces.append(race_df.to_numpy())
        names.extend(race_df.columns.tolist())

    if BLOCK_SEX[block]:
        sex_enc = feat.fit_onehot(df_split, ["TP_SEXO"], train_mask)
        sex_df = feat.apply_onehot(sex_enc, df_split, ["TP_SEXO"])
        pieces.append(sex_df.to_numpy())
        names.extend(sex_df.columns.tolist())

    X = np.concatenate(pieces, axis=1).astype(float)
    return X, names


def topk_within_split(scores: np.ndarray, K: float, seed: int = TIE_BREAK_SEED):
    """Same tie-breaking convention as common.build_topk, applied to a
    scores vector already restricted to one split."""
    from common import build_topk
    return build_topk(scores, K=K, seed=seed)


def fit_logreg(X_train, y_train, C):
    scaler = StandardScaler().fit(X_train)
    Xs = scaler.transform(X_train)
    clf = LogisticRegression(penalty="l2", C=C, solver="lbfgs", max_iter=2000)
    clf.fit(Xs, y_train)
    return clf, scaler


def tune_logreg(X_train, y_train, X_val, y_val_benchmark, K, log=print):
    best = None
    for C in LOGREG_C_GRID:
        clf, scaler = fit_logreg(X_train, y_train, C)
        val_scores = clf.predict_proba(scaler.transform(X_val))[:, 1]
        selected_val, _ = topk_within_split(val_scores, K)
        n_k = int(np.floor(K * len(val_scores)))
        overlap = (selected_val & y_val_benchmark.astype(bool)).sum() / n_k
        log(f"    C={C:<8g} overlap@K(val)={overlap:.4f}")
        if best is None or overlap > best[0]:
            best = (overlap, C, clf, scaler)
    return best  # (overlap, C, clf, scaler)


HGB_GRID = [
    dict(max_depth=d, learning_rate=lr, max_iter=mi, min_samples_leaf=msl)
    for d in [3, 5, 8, None]
    for lr in [0.03, 0.1, 0.3]
    for mi in [100, 300, 600]
    for msl in [20, 50, 200]
]


def fit_hgb(X_train, y_train, params):
    from sklearn.ensemble import HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(
        max_depth=params["max_depth"], learning_rate=params["learning_rate"],
        max_iter=params["max_iter"], min_samples_leaf=params["min_samples_leaf"],
        early_stopping=True, n_iter_no_change=20, validation_fraction=0.1,
        random_state=TIE_BREAK_SEED,
    )
    clf.fit(X_train, y_train)
    return clf


def _hgb_grid_one(params, X_train, y_train, X_val, y_val_benchmark, K):
    clf = fit_hgb(X_train, y_train, params)
    val_scores = clf.predict_proba(X_val)[:, 1]
    selected_val, _ = topk_within_split(val_scores, K)
    n_k = int(np.floor(K * len(val_scores)))
    overlap = (selected_val & y_val_benchmark.astype(bool)).sum() / n_k
    return overlap, params, clf


def tune_hgb(X_train, y_train, X_val, y_val_benchmark, K, log=print, n_jobs=-1):
    from joblib import Parallel, delayed
    results = Parallel(n_jobs=n_jobs)(
        delayed(_hgb_grid_one)(params, X_train, y_train, X_val, y_val_benchmark, K)
        for params in HGB_GRID
    )
    best = max(results, key=lambda r: r[0])
    log(f"    grid search over {len(HGB_GRID)} combinations complete; "
        f"best overlap@K(val)={best[0]:.4f} at params={best[1]}")
    return best  # (overlap, params, clf) -- no separate scaler needed for HGB


def evaluate_selection(model_scores: np.ndarray, benchmark_selected: np.ndarray,
                        race_labels: pd.Series, K: float, seed: int = TIE_BREAK_SEED):
    """Everything reported for one (block, model) cell on one split."""
    n = len(model_scores)
    k = int(np.floor(K * n))
    model_selected, tie_info = topk_within_split(model_scores, K, seed=seed)

    overlap_at_k = (model_selected & benchmark_selected).sum() / k
    union = (model_selected | benchmark_selected).sum()
    jaccard = (model_selected & benchmark_selected).sum() / union if union else np.nan

    pr_auc = average_precision_score(benchmark_selected, model_scores)
    try:
        roc_auc = roc_auc_score(benchmark_selected, model_scores)
    except ValueError:
        roc_auc = np.nan

    results = dict(overlap_at_k=overlap_at_k, jaccard=jaccard, pr_auc=pr_auc, roc_auc=roc_auc,
                    n=n, k=k, n_tied=tie_info["n_tied"])

    is_white = (race_labels == "White").to_numpy()
    is_bp = race_labels.isin(["Black", "Pardo"]).to_numpy()

    group_rows = []
    zero_sum_terms = []
    for label, mask in [("White", is_white), ("Black_Pardo", is_bp)]:
        n_g = mask.sum()
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
        net_realloc = q_g - p_g
        zero_sum_terms.append(n_g * net_realloc)
        group_rows.append(dict(group=label, n=int(n_g), p_g=p_g, q_g=q_g,
                                net_reallocation=net_realloc, fpr=fpr, fnr=fnr))

    results["group_rows"] = group_rows
    results["rate_white_pct"] = 100 * group_rows[0]["q_g"]
    results["rate_bp_pct"] = 100 * group_rows[1]["q_g"]
    results["gap_pp"] = results["rate_white_pct"] - results["rate_bp_pct"]
    results["ratio"] = (results["rate_white_pct"] / results["rate_bp_pct"]
                         if results["rate_bp_pct"] > 0 else np.nan)
    results["bench_gap_pp"] = 100 * (group_rows[0]["p_g"] - group_rows[1]["p_g"])

    # Zero-sum check over ALL groups present (not just White/BP), since the
    # identity in Section 5.2 is sum_g n_g (q_g - p_g) = 0 over every group
    # sharing the ranking population, not only the contrast pair.
    all_zero_sum = []
    for label in race_labels.unique():
        mask = (race_labels == label).to_numpy()
        n_g = mask.sum()
        p_g = benchmark_selected[mask].mean()
        q_g = model_selected[mask].mean()
        all_zero_sum.append(n_g * (q_g - p_g))
    results["zero_sum_value"] = float(np.sum(all_zero_sum))

    return results
