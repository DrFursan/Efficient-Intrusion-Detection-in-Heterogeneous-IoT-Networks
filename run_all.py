"""
run_all.py
==========
Master pipeline for the AEGIS-IoT framework.

Executes the complete experimental workflow end-to-end:

    Step 1 — Preprocess UNSW-NB15 data
    Step 2 — Feature selection (Scenario 1: IG+MI,  Scenario 2: RF importance)
    Step 3 — Train all individual classifiers (both scenarios)
    Step 4 — Train all ensemble models (both scenarios)
    Step 5 — Generate all 10 publication figures

All intermediate files go to data/, all result CSVs to results/,
all models to saved_models/, and all figures to figures/.

Usage
-----
    python run_all.py

    # Skip slow models (SVM, GB) for a quick test run:
    python run_all.py --quick

    # Use a different dataset path:
    python run_all.py --data_dir /path/to/your/data
"""

import argparse
import os
import sys
import time

import pandas as pd
from sklearn.model_selection import train_test_split

# ── Ensure src/ is on the path ────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from src.preprocessing      import preprocess
from src.feature_selection  import (select_by_mutual_information,
                                     select_by_rf_importance)
from src.models.train_individual import train_classifier, ALL_MODELS
from src.models.train_stacking   import train_stacking
from src.models.train_voting     import train_voting
from src.visualize               import generate_all

# ── Directories ───────────────────────────────────────────────────
DATA_DIR    = "data"
RESULTS_DIR = "results"
MODELS_DIR  = "saved_models"
FIGURES_DIR = "figures"

UNSW_TEST_FILE = os.path.join(DATA_DIR, "UNSW_NB15_testing-set.csv")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _banner(msg: str) -> None:
    sep = "=" * 60
    print(f"\n{sep}\n  {msg}\n{sep}")


def _save_feature_csv(X, y, label_col: str, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df = X.copy()
    df[label_col] = y.values
    df.to_csv(path, index=False)
    print(f"  Saved feature CSV → {path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(data_dir:    str = DATA_DIR,
                 results_dir: str = RESULTS_DIR,
                 models_dir:  str = MODELS_DIR,
                 figures_dir: str = FIGURES_DIR,
                 quick:       bool = False) -> None:

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir,  exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    total_start = time.time()

    # ── Step 1: Preprocess ────────────────────────────────────────
    _banner("Step 1 — Preprocessing UNSW-NB15")

    unsw_path = os.path.join(data_dir, "UNSW_NB15_testing-set.csv")
    if not os.path.exists(unsw_path):
        print(f"\n  ⚠️  Dataset not found: {unsw_path}")
        print("  Please download UNSW-NB15 and place it in data/.")
        print("  See data/README.md for instructions.")
        sys.exit(1)

    # Use 5% stratified sample to match the paper's setup
    X_raw, y_raw = preprocess(
        file_path=unsw_path,
        dataset="unsw",
        sample_frac=0.05,          # 5% of 82k ≈ 4,100 rows for quick dev
    )

    # ── Step 2a: Scenario 1 features (IG + MI) ───────────────────
    _banner("Step 2a — Feature Selection: Scenario 1 (IG + MI)")

    X_s1, feats_s1 = select_by_mutual_information(X_raw, y_raw, top_k=10)
    s1_csv = os.path.join(data_dir, "unsw_scenario1_features.csv")
    _save_feature_csv(X_s1, y_raw, "label", s1_csv)

    # ── Step 2b: Scenario 2 features (RF importance) ─────────────
    _banner("Step 2b — Feature Selection: Scenario 2 (RF importance)")

    imp_csv = os.path.join(results_dir, "important_features.csv")
    X_s2, feats_s2, _ = select_by_rf_importance(
        X_raw, y_raw, top_k=9, save_csv=imp_csv
    )
    s2_csv = os.path.join(data_dir, "unsw_scenario2_features.csv")
    _save_feature_csv(X_s2, y_raw, "label", s2_csv)

    # ── Step 3: Individual classifiers ───────────────────────────
    _banner("Step 3 — Individual Classifiers")

    models_to_run = ALL_MODELS
    if quick:
        models_to_run = ["dt", "rf", "nb", "lr"]
        print("  [quick mode] Skipping SVM, GB, XGBoost")

    all_results = {}

    # Scenario 1
    print("\n  --- Scenario 1 ---")
    df_s1 = pd.read_csv(s1_csv)
    Xs1   = df_s1.drop(columns=["label"]).values
    ys1   = df_s1["label"]
    Xs1_tr, Xs1_te, ys1_tr, ys1_te = train_test_split(
        Xs1, ys1, test_size=0.2, random_state=42, stratify=ys1
    )
    for key in models_to_run:
        res = train_classifier(
            key, Xs1_tr, ys1_tr, Xs1_te, ys1_te,
            feature_names=feats_s1,
            results_dir=os.path.join(results_dir, "scenario1"),
            models_dir=os.path.join(models_dir, "scenario1"),
        )
        all_results[f"S1_{key}"] = res

    # Scenario 2
    print("\n  --- Scenario 2 ---")
    df_s2 = pd.read_csv(s2_csv)
    Xs2   = df_s2.drop(columns=["label"]).values
    ys2   = df_s2["label"]
    Xs2_tr, Xs2_te, ys2_tr, ys2_te = train_test_split(
        Xs2, ys2, test_size=0.2, random_state=42, stratify=ys2
    )
    for key in models_to_run:
        res = train_classifier(
            key, Xs2_tr, ys2_tr, Xs2_te, ys2_te,
            feature_names=feats_s2,
            results_dir=os.path.join(results_dir, "scenario2"),
            models_dir=os.path.join(models_dir, "scenario2"),
        )
        all_results[f"S2_{key}"] = res

    # ── Step 4: Ensemble models ───────────────────────────────────
    _banner("Step 4 — Ensemble Models")

    if not quick:
        # Scenario 2 stacking (paper's optimal model)
        print("\n  --- Stacking: DT+LR (Scenario 2) ---")
        res_stk = train_stacking(
            Xs2_tr, ys2_tr, Xs2_te, ys2_te,
            results_dir=os.path.join(results_dir, "scenario2"),
            models_dir=os.path.join(models_dir, "scenario2"),
        )
        all_results["S2_stacking"] = res_stk

        # Voting ensembles
        for combo in ["rf_svm", "svm_gb"]:
            print(f"\n  --- Voting: {combo} (Scenario 2) ---")
            res_vot = train_voting(
                combo=combo,
                X_train=Xs2_tr, y_train=ys2_tr,
                X_test=Xs2_te,  y_test=ys2_te,
                results_dir=os.path.join(results_dir, "scenario2"),
                models_dir=os.path.join(models_dir, "scenario2"),
            )
            all_results[f"S2_{combo}"] = res_vot
    else:
        print("  [quick mode] Skipping ensemble models")

    # ── Step 5: Figures ───────────────────────────────────────────
    _banner("Step 5 — Generating Figures")
    generate_all(results_dir=results_dir, output_dir=figures_dir)

    # ── Final summary table ───────────────────────────────────────
    _banner("Pipeline Complete — Final Summary")
    print(f"\n  {'Experiment':<30} {'Acc':>7} {'F1':>7} {'Time(s)':>10}")
    print("  " + "-" * 58)
    for key, m in all_results.items():
        print(f"  {key:<30} {m['accuracy']:>7.4f} "
              f"{m['f1']:>7.4f} {m.get('time',0):>10.2f}")

    total = time.time() - total_start
    print(f"\n  Total wall-clock time: {total/60:.1f} minutes")
    print(f"  Results  → {results_dir}/")
    print(f"  Models   → {models_dir}/")
    print(f"  Figures  → {figures_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="AEGIS-IoT Master Pipeline")
    p.add_argument("--data_dir",    default=DATA_DIR)
    p.add_argument("--results_dir", default=RESULTS_DIR)
    p.add_argument("--models_dir",  default=MODELS_DIR)
    p.add_argument("--figures_dir", default=FIGURES_DIR)
    p.add_argument("--quick",       action="store_true",
                   help="Skip slow models (SVM, GB, XGBoost) and ensembles")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        models_dir=args.models_dir,
        figures_dir=args.figures_dir,
        quick=args.quick,
    )
