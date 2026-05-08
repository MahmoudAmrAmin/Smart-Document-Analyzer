"""
Generates confusion matrix + per-class accuracy charts.
Run: python src/ml/evaluate.py
"""

import os
import sys

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.insert(0, ROOT_DIR)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from src.ml.classifier import load_model
from src.config.settings import ROOT_DIR, TEST_CSV

REPORTS_DIR = os.path.join(
    ROOT_DIR, "data", "annotations",
    "classification_labels", "reports"
)


def run_evaluation():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    model   = load_model()
    test_df = pd.read_csv(TEST_CSV).dropna(subset=["text", "label"])
    X_test  = test_df["text"].tolist()
    y_test  = test_df["label"].tolist()
    y_pred  = model.predict(X_test)
    classes = list(model.classes_)

    # ── Print report ───────────────────────────────────
    print("\n" + "="*55)
    print("  Evaluation on Test Set")
    print("="*55)
    print(classification_report(y_test, y_pred, target_names=classes))

    # ── Confusion matrix ───────────────────────────────
    cm = confusion_matrix(y_test, y_pred, labels=classes)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=ax
    )
    ax.set_title("Confusion Matrix — Document Classifier\n"
                 "(20 Newsgroups + AG News)", pad=14)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    cm_path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"✅ Confusion matrix → {cm_path}")

    # ── Per-class accuracy bar chart ───────────────────
    per_class = cm.diagonal() / cm.sum(axis=1)
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    colors = ["#4e79a7", "#f28e2b", "#59a14f"]
    bars   = ax2.bar(classes, per_class, color=colors, width=0.5)
    ax2.set_ylim(0, 1.15)
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Per-Class Accuracy\n(20 Newsgroups + AG News)")
    for bar, val in zip(bars, per_class):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.03, f"{val:.1%}",
            ha="center", fontsize=12, fontweight="bold"
        )
    plt.tight_layout()
    bar_path = os.path.join(REPORTS_DIR, "per_class_accuracy.png")
    plt.savefig(bar_path, dpi=150)
    plt.close()
    print(f"✅ Per-class chart  → {bar_path}")

    # ── Dataset composition chart ──────────────────────
    labels_count = test_df["label"].value_counts()
    fig3, ax3    = plt.subplots(figsize=(5, 5))
    ax3.pie(
        labels_count.values,
        labels=labels_count.index,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90
    )
    ax3.set_title("Test Set Distribution")
    pie_path = os.path.join(REPORTS_DIR, "test_distribution.png")
    plt.savefig(pie_path, dpi=150)
    plt.close()
    print(f"✅ Distribution pie → {pie_path}")


if __name__ == "__main__":
    run_evaluation()