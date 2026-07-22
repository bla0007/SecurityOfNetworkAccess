"""
train_unsw.py — SONA v3: Train models on UNSW-NB15
=====================================================
Same structure as train.py (NSL-KDD version) — trains and compares
Logistic Regression, Decision Tree, Random Forest, and XGBoost —
but on the modern UNSW-NB15 dataset with 9 real attack families.

Usage:
    python src/train_unsw.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import time
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.tree            import DecisionTreeClassifier
from sklearn.metrics         import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from preprocess_unsw import prepare_pipeline


def get_models(n_classes: int) -> dict:
    xgb_params = dict(
        n_estimators=200, max_depth=8, learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="mlogloss" if n_classes > 2 else "logloss",
        random_state=42, n_jobs=-1,
    )
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
        "Decision Tree":       DecisionTreeClassifier(max_depth=15, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=150, max_depth=20, random_state=42, n_jobs=-1),
        "XGBoost":             XGBClassifier(**xgb_params),
    }


def evaluate(model, X_test, y_test, label_names: list) -> dict:
    y_pred = model.predict(X_test)
    return {
        "accuracy":     accuracy_score(y_test, y_pred),
        "precision":    precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall":       recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1":           f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_macro":     f1_score(y_test, y_pred, average="macro", zero_division=0),
        "y_pred":       y_pred,
        "report":       classification_report(y_test, y_pred, target_names=label_names, zero_division=0),
    }


def plot_confusion_matrix(y_true, y_pred, labels, model_name: str, save_path: str):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Confusion matrix — {model_name} (UNSW-NB15)", fontsize=14, fontweight="bold")
    for ax, data, title, fmt in zip(axes, [cm, cm_norm], ["Raw counts", "Normalised"], ["d", ".2f"]):
        sns.heatmap(data, annot=True, fmt=fmt, ax=ax, xticklabels=labels, yticklabels=labels,
                    cmap="Blues", linewidths=0.5, annot_kws={"size": 7})
        ax.set_title(title)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_model_comparison(results: dict, save_path: str):
    metrics = ["accuracy", "precision", "recall", "f1"]
    model_names = list(results.keys())
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    colors = ["#4F46E5", "#10B981", "#F59E0B", "#EF4444"]
    for ax, metric, color in zip(axes, metrics, colors):
        vals = [results[m][metric] for m in model_names]
        bars = ax.bar(model_names, vals, color=color, alpha=0.85)
        ax.set_ylim(0, 1.05)
        ax.set_title(metric.capitalize(), fontweight="bold")
        ax.set_xticklabels(model_names, rotation=25, ha="right", fontsize=9)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.01, f"{v:.3f}",
                     ha="center", va="bottom", fontsize=8)
    fig.suptitle("Model comparison — UNSW-NB15", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def train_all(model_dir: str = "models_unsw", plot_dir: str = "plots_unsw", apply_smote: bool = True):
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    print("=" * 60)
    print("STEP 1: Loading and preprocessing UNSW-NB15")
    print("=" * 60)
    data = prepare_pipeline(model_dir=model_dir)
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]
    label_enc  = data["label_encoder"]
    feat_cols  = data["feature_cols"]
    label_names = list(label_enc.classes_)
    n_classes = len(label_names)

    if apply_smote:
        print("\n" + "=" * 60)
        print("STEP 1b: Balancing minority classes with SMOTE")
        print("=" * 60)
        unique, counts = np.unique(y_train, return_counts=True)
        smallest_class = counts.min()
        # k_neighbors must be < smallest class size, or SMOTE errors out.
        # Worms/Analysis/Backdoors are tiny — cap neighbors conservatively.
        k = max(1, min(5, smallest_class - 1))
        print(f"  Smallest class has {smallest_class} samples → using k_neighbors={k}")

        before = dict(zip(label_enc.inverse_transform(unique), counts))
        smote = SMOTE(random_state=42, k_neighbors=k)
        X_train, y_train = smote.fit_resample(X_train, y_train)

        unique2, counts2 = np.unique(y_train, return_counts=True)
        after = dict(zip(label_enc.inverse_transform(unique2), counts2))
        print(f"\n  {'Class':15s} {'Before':>10s} {'After':>10s}")
        for cls in sorted(after, key=lambda c: -after[c]):
            print(f"  {cls:15s} {before.get(cls,0):>10,} {after[cls]:>10,}")

    print("\n" + "=" * 60)
    print("STEP 2: Training models")
    print("=" * 60)
    models  = get_models(n_classes)
    results = {}

    for name, model in models.items():
        print(f"\n  [{name}]")
        t0 = time.time()
        model.fit(X_train, y_train)
        print(f"  Training time: {time.time()-t0:.1f}s")

        metrics = evaluate(model, X_test, y_test, label_names)
        results[name] = metrics
        print(f"  Accuracy      : {metrics['accuracy']:.4f}")
        print(f"  F1 (weighted) : {metrics['f1']:.4f}")
        print(f"  F1 (macro)    : {metrics['f1_macro']:.4f}  <- minority-class performance")

        plot_confusion_matrix(y_test, metrics["y_pred"], label_names, name,
            os.path.join(plot_dir, f"cm_{name.lower().replace(' ', '_')}.png"))
        joblib.dump(model, os.path.join(model_dir, f"{name.lower().replace(' ', '_')}.pkl"))

    print("\n" + "=" * 60)
    print("STEP 3: Model comparison")
    print("=" * 60)
    plot_model_comparison(results, os.path.join(plot_dir, "model_comparison.png"))

    summary = pd.DataFrame({
        name: {k: r[k] for k in ["accuracy", "precision", "recall", "f1", "f1_macro"]}
        for name, r in results.items()
    }).T.sort_values("f1_macro", ascending=False)
    print("\nResults summary (sorted by F1-macro — rewards detecting ALL attack types, not just common ones):")
    print(summary.to_string(float_format="{:.4f}".format))

    best_name  = summary.index[0]
    best_model = models[best_name]
    joblib.dump(best_model, os.path.join(model_dir, "best_model.pkl"))
    joblib.dump(label_names, os.path.join(model_dir, "label_names.pkl"))
    print(f"\nBest model: {best_name} (F1-macro={summary.loc[best_name,'f1_macro']:.4f}, "
          f"F1-weighted={summary.loc[best_name,'f1']:.4f})")
    print(f"Saved → {model_dir}/best_model.pkl")

    print(f"\nDetailed report — {best_name}:")
    print(results[best_name]["report"])

    return results, best_model, label_names


if __name__ == "__main__":
    train_all()
    print("\nAll done! Check plots_unsw/ for visualizations.")
    print("Model artifacts saved separately in models_unsw/ — the original")
    print("NSL-KDD model in models/ is untouched, so you can compare both.")
