"""
Task A - Data preparation only (no modelling).

Implements the restriction cascade (spec Section 4.2), the raw equal-weight
composite (Section 4.6), the top-K benchmark indicator with tie-breaking
(Section 4.6), and the White vs Black/Pardo selection-rate table
(Section 6.1) from final_paper_draft.md.

Run from the repository root:
    python3 src/task_a_data_prep.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "DADOS" / "enem_2019_sample.csv"
RESULTS_DIR = ROOT / "results"
PARQUET_OUT = RESULTS_DIR / "analytic_sample.parquet"
LOG_OUT = RESULTS_DIR / "taskA_log.txt"

TIE_BREAK_SEED = 20190101

SCORE_COLS = ["NU_NOTA_CN", "NU_NOTA_CH", "NU_NOTA_LC", "NU_NOTA_MT", "NU_NOTA_REDACAO"]

# Expected N after each cascade step, per the table in Section 4.2.
EXPECTED_CASCADE = [203807, 179226, 178429, 126313, 126313, 123879]

STEP_LABELS = [
    "0: Random sample (loaded file)",
    "1: Exclude treineiros (IN_TREINEIRO == 0)",
    "2: Completed or completing secondary education in 2019 (TP_ST_CONCLUSAO in {1,2})",
    "3: Present at all four objective examinations (TP_PRESENCA_CN/CH/LC/MT == 1)",
    "4: Non-missing scores on all five components",
    "5: Race declared (TP_COR_RACA != 0)",
]

log_lines = []


def log(msg=""):
    print(msg)
    log_lines.append(str(msg))


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    log("=" * 70)
    log("TASK A - DATA PREPARATION")
    log("=" * 70)

    # ------------------------------------------------------------------
    # Step 0: load
    # ------------------------------------------------------------------
    log(f"\nLoading {INPUT_PATH} (sep=';', encoding='latin-1') ...")
    df = pd.read_csv(INPUT_PATH, sep=";", encoding="latin-1", low_memory=False)
    n_values = [len(df)]
    log(f"{STEP_LABELS[0]}: N = {len(df):,}")

    # ------------------------------------------------------------------
    # Step 1: exclude treineiros
    # ------------------------------------------------------------------
    df = df[df["IN_TREINEIRO"] == 0].copy()
    n_values.append(len(df))
    log(f"{STEP_LABELS[1]}: N = {len(df):,}")

    # ------------------------------------------------------------------
    # Step 2: completed or completing secondary education in 2019
    # TP_ST_CONCLUSAO: 1 = already completed, 2 = completing in 2019
    # ------------------------------------------------------------------
    df = df[df["TP_ST_CONCLUSAO"].isin([1, 2])].copy()
    n_values.append(len(df))
    log(f"{STEP_LABELS[2]}: N = {len(df):,}")

    # ------------------------------------------------------------------
    # Step 3: present at all four objective examinations
    # TP_PRESENCA_*: 0 absent, 1 present, 2 eliminated
    # ------------------------------------------------------------------
    presence_cols = ["TP_PRESENCA_CN", "TP_PRESENCA_CH", "TP_PRESENCA_LC", "TP_PRESENCA_MT"]
    mask_present = (df[presence_cols] == 1).all(axis=1)
    df = df[mask_present].copy()
    n_values.append(len(df))
    log(f"{STEP_LABELS[3]}: N = {len(df):,}")

    # ------------------------------------------------------------------
    # Step 4: non-missing scores on all five components
    # ------------------------------------------------------------------
    df = df[df[SCORE_COLS].notna().all(axis=1)].copy()
    n_values.append(len(df))
    log(f"{STEP_LABELS[4]}: N = {len(df):,}")

    # ------------------------------------------------------------------
    # Step 5: race declared
    # TP_COR_RACA: 0 = not declared
    # ------------------------------------------------------------------
    df = df[df["TP_COR_RACA"] != 0].copy()
    n_values.append(len(df))
    log(f"{STEP_LABELS[5]}: N = {len(df):,}")

    # ------------------------------------------------------------------
    # Compare against the expected cascade
    # ------------------------------------------------------------------
    log("\n" + "-" * 70)
    log("Cascade check against Section 4.2")
    log("-" * 70)
    mismatch = False
    for i, (observed, expected) in enumerate(zip(n_values, EXPECTED_CASCADE)):
        status = "OK" if observed == expected else "MISMATCH"
        if observed != expected:
            mismatch = True
        log(f"  Step {i}: observed={observed:,}  expected={expected:,}  [{status}]")

    if mismatch:
        log("\nSTOP: cascade N does not match Section 4.2. Halting before any "
            "further computation, per task instructions.")
        write_log()
        sys.exit(1)

    log("\nAll six cascade values match Section 4.2 exactly.")

    analytic_sample = df.reset_index(drop=True)
    n_total = len(analytic_sample)
    assert n_total == 123879

    # ------------------------------------------------------------------
    # Step 4.6: raw equal-weight composite (simple mean of raw scores)
    # ------------------------------------------------------------------
    analytic_sample["composite_raw_equal"] = analytic_sample[SCORE_COLS].mean(axis=1)

    log("\n" + "-" * 70)
    log("Raw equal-weight composite (Section 4.6)")
    log("-" * 70)
    log(f"  N = {n_total:,}")
    log(f"  Mean = {analytic_sample['composite_raw_equal'].mean():.1f}  "
        f"(spec reports 522.0)")
    log(f"  SD   = {analytic_sample['composite_raw_equal'].std(ddof=1):.1f}  "
        f"(spec reports 83.7)")

    # ------------------------------------------------------------------
    # Race group labels
    # TP_COR_RACA: 1 White, 2 Black, 3 Pardo, 4 Asian, 5 Indigenous
    # (0 excluded by Step 5)
    # ------------------------------------------------------------------
    race_map = {1: "White", 2: "Black", 3: "Pardo", 4: "Asian", 5: "Indigenous"}
    analytic_sample["race_label"] = analytic_sample["TP_COR_RACA"].map(race_map)

    # ------------------------------------------------------------------
    # Top-K benchmark indicator, K = 10%, tie-break seed 20190101
    # ------------------------------------------------------------------
    log("\n" + "-" * 70)
    log("Top-K benchmark construction, K = 10% (Section 4.6, item 5)")
    log("-" * 70)

    selected_10, tie_info_10 = build_topk(
        analytic_sample["composite_raw_equal"].to_numpy(), K=0.10, seed=TIE_BREAK_SEED
    )
    analytic_sample["benchmark_top10"] = selected_10

    k_target = int(np.floor(0.10 * n_total))
    log(f"  N = {n_total:,}, K = 10% -> floor(K*N) = {k_target:,} places")
    log(f"  Boundary composite value = {tie_info_10['boundary_value']:.4f}")
    log(f"  Candidates tied at the boundary value = {tie_info_10['n_tied']:,}")
    log(f"  Of those tied, selected by random tie-break (seed {TIE_BREAK_SEED}) = "
        f"{tie_info_10['n_tied_selected']:,}")
    log(f"  Candidates selected overall = {int(selected_10.sum()):,} "
        f"(should equal {k_target:,})")
    assert int(selected_10.sum()) == k_target

    # ------------------------------------------------------------------
    # Section 6.1 selection table at K = 5%, 10%, 20%
    # ------------------------------------------------------------------
    log("\n" + "-" * 70)
    log("Section 6.1 reproduction: White vs Black/Pardo selection rates")
    log("-" * 70)

    is_white = analytic_sample["race_label"] == "White"
    is_black_pardo = analytic_sample["race_label"].isin(["Black", "Pardo"])
    n_white = int(is_white.sum())
    n_black_pardo = int(is_black_pardo.sum())

    expected_table = {
        0.05: dict(white=8.96, bp=2.68, gap=6.28, ratio=3.34),
        0.10: dict(white=16.97, bp=5.97, gap=11.00, ratio=2.84),
        0.20: dict(white=30.68, bp=13.91, gap=16.77, ratio=2.21),
    }

    composite_arr = analytic_sample["composite_raw_equal"].to_numpy()
    for K in (0.05, 0.10, 0.20):
        selected, tie_info = build_topk(composite_arr, K=K, seed=TIE_BREAK_SEED)
        rate_white = 100.0 * selected[is_white.to_numpy()].mean()
        rate_bp = 100.0 * selected[is_black_pardo.to_numpy()].mean()
        gap = rate_white - rate_bp
        ratio = rate_white / rate_bp

        exp = expected_table[K]
        log(f"\n  K = {int(K*100)}%  (tied at boundary = {tie_info['n_tied']:,})")
        log(f"    White:       observed {rate_white:.2f}%   expected {exp['white']:.2f}%")
        log(f"    Black/Pardo: observed {rate_bp:.2f}%   expected {exp['bp']:.2f}%")
        log(f"    Gap (pp):    observed {gap:.2f}      expected {exp['gap']:.2f}")
        log(f"    Ratio:       observed {ratio:.2f}      expected {exp['ratio']:.2f}")

        # Flag any mismatch beyond rounding (tolerance 0.01 on the reported scale).
        tol = 0.015
        for name, obs, exp_v in [
            ("White rate", rate_white, exp["white"]),
            ("Black/Pardo rate", rate_bp, exp["bp"]),
            ("Gap", gap, exp["gap"]),
            ("Ratio", ratio, exp["ratio"]),
        ]:
            if abs(obs - exp_v) > tol:
                log(f"    FLAG: {name} differs from spec by more than rounding "
                    f"({obs:.4f} vs {exp_v:.4f})")

        if K == 0.10:
            log(f"    (n_white={n_white:,}, n_black_pardo={n_black_pardo:,})")

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    analytic_sample.to_parquet(PARQUET_OUT, index=False)
    log(f"\nSaved analytic sample to {PARQUET_OUT} (N={len(analytic_sample):,}, "
        f"{analytic_sample.shape[1]} columns).")

    write_log()


def build_topk(composite: np.ndarray, K: float, seed: int):
    """Select the top floor(K*N) candidates by composite score, breaking ties
    at the boundary with a fixed-seed random permutation.

    Returns (selected_bool_array, tie_info_dict).
    """
    n = len(composite)
    k = int(np.floor(K * n))

    order = np.argsort(-composite, kind="mergesort")  # stable, descending
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

    tie_info = dict(
        boundary_value=boundary_value,
        n_tied=n_tied,
        n_tied_selected=n_slots_from_tied,
    )
    return selected, tie_info


def write_log():
    with open(LOG_OUT, "w") as f:
        f.write("\n".join(log_lines) + "\n")


if __name__ == "__main__":
    main()
