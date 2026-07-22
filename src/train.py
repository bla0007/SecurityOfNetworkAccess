"""
train.py — SONA (Security of Network Access)
============================================
Train and evaluate multiple ML models for network intrusion detection.
Trains Logistic Regression, Random Forest, and XGBoost.
Compares them and saves the best model.

Usage:
    python src/train.py
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
    roc_auc_score, roc_curve
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from preprocess import prepare_pipeline


# ── Model definitions ────────────────────────────────────────────────────────

def get_models(n_classes: int) -> dict:
    """Return dict of model name → unfitted model."""
    xgb_params = dict(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="mlogloss" if n_classes > 2 else "logloss",
        random_state=42,
        n_jobs=-1,
    )
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, n_jobs=-1
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=15,
            random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(**xgb_params),
    }


# ── Evaluation helpers ───────────────────────────────────────────────────────

def evaluate(model, X_test, y_test, label_names: list) -> dict:
    """Compute all metrics for a fitted model."""
    y_pred = model.predict(X_test)
    avg    = "binary" if len(label_names) == 2 else "weighted"

    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average=avg, zero_division=0),
        "recall":    recall_score(y_test, y_pred, average=avg, zero_division=0),
        "f1":        f1_score(y_test, y_pred, average=avg, zero_division=0),
        "y_pred":    y_pred,
        "report":    classification_report(
            y_test, y_pred, target_names=label_names, zero_division=0
        ),
    }
    return metrics


def plot_confusion_matrix(y_true, y_pred, labels, model_name: str, save_path: str):
    """Save a styled confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Confusion matrix — {model_name}", fontsize=14, fontweight="bold")

    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ["Raw counts", "Normalised (row %)"],
        ["d", ".2f"]
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, ax=ax,
            xticklabels=labels, yticklabels=labels,
            cmap="Blues", linewidths=0.5
        )
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_feature_importance(model, feature_cols: list, model_name: str, save_path: str, top_n: int = 20):
    """Save a horizontal bar chart of the top-N most important features."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).mean(axis=0)
    else:
        return

    idx = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(
        [feature_cols[i] for i in idx],
        importances[idx],
        color="#4F46E5", alpha=0.8
    )
    ax.set_xlabel("Importance score")
    ax.set_title(f"Top {top_n} features — {model_name}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def plot_model_comparison(results: dict, save_path: str):
    """Bar chart comparing all models across 4 metrics."""
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
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Model comparison", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


# ── Main training loop ───────────────────────────────────────────────────────

def train_all(binary: bool = False, apply_smote: bool = False):
    """Full training pipeline."""
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots",  exist_ok=True)

    # 1. Load data
    print("=" * 60)
    print("STEP 1: Loading and preprocessing data")
    print("=" * 60)
    data = prepare_pipeline(
        train_path="data/KDDTrain+.txt",
        test_path="data/KDDTest+.txt",
        binary=binary,
    )
    X_train    = data["X_train"]
    X_test     = data["X_test"]
    y_train    = data["y_train"]
    y_test     = data["y_test"]
    label_enc  = data["label_encoder"]
    feat_cols  = data["feature_cols"]

    label_names = (
        ["Normal", "Attack"] if binary
        else list(label_enc.classes_)
    )
    n_classes = len(label_names)

    # 2. Optional SMOTE for class imbalance
    if apply_smote and not binary:
        print("\nApplying SMOTE to balance classes...")
        smote = SMOTE(random_state=42, k_neighbors=3)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        unique, counts = np.unique(y_train, return_counts=True)
        for u, c in zip(unique, counts):
            print(f"  Class {label_names[u]}: {c:,}")

    # 3. Train models
    print("\n" + "=" * 60)
    print("STEP 2: Training models")
    print("=" * 60)
    models  = get_models(n_classes)
    results = {}

    for name, model in models.items():
        print(f"\n  [{name}]")
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0
        print(f"  Training time: {elapsed:.1f}s")

        metrics = evaluate(model, X_test, y_test, label_names)
        results[name] = metrics

        print(f"  Accuracy : {metrics['accuracy']:.4f}")
        print(f"  F1 score : {metrics['f1']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall   : {metrics['recall']:.4f}")

        # Save confusion matrix
        plot_confusion_matrix(
            y_test, metrics["y_pred"], label_names, name,
            f"plots/cm_{name.lower().replace(' ', '_')}.png"
        )

        # Save feature importance
        plot_feature_importance(
            model, feat_cols, name,
            f"plots/fi_{name.lower().replace(' ', '_')}.png"
        )

        # Save model
        joblib.dump(model, f"models/{name.lower().replace(' ', '_')}.pkl")

    # 4. Compare all models
    print("\n" + "=" * 60)
    print("STEP 3: Model comparison")
    print("=" * 60)
    plot_model_comparison(results, "plots/model_comparison.png")

    summary = pd.DataFrame({
        name: {
            "accuracy":  r["accuracy"],
            "precision": r["precision"],
            "recall":    r["recall"],
            "f1":        r["f1"],
        }
        for name, r in results.items()
    }).T.sort_values("f1", ascending=False)

    print("\nResults summary (sorted by F1):")
    print(summary.to_string(float_format="{:.4f}".format))

    # 5. Save best model
    best_name  = summary.index[0]
    best_model = models[best_name]
    joblib.dump(best_model, "models/best_model.pkl")
    joblib.dump(label_names, "models/label_names.pkl")
    print(f"\nBest model: {best_name} (F1={summary.loc[best_name,'f1']:.4f})")
    print("Saved → models/best_model.pkl")

    # 6. Detailed report for best model
    print(f"\nDetailed classification report — {best_name}:")
    print(results[best_name]["report"])

    return results, best_model, label_names


if __name__ == "__main__":
    results, best_model, label_names = train_all(
        binary=False,       # False = 5-class, True = binary (Normal/Attack)
        apply_smote=False,  # Set True if you want SMOTE balancing
    )
    print("\nAll done! Check plots/ for visualizations.")
