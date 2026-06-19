"""
evaluate.py
===========
Evaluation utilities for the Legal Notice Classifier.
Computes classification metrics, plots confusion matrices, and measures timing.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def train_and_time(model, X_train, y_train):
    """
    Fit a model on training data and record the wall-clock training time.

    Args:
        model: An unfitted sklearn estimator.
        X_train: Training feature matrix (sparse or dense).
        y_train (array-like): Training labels.

    Returns:
        tuple: (fitted_model, training_time_seconds)
    """
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    return model, elapsed


def predict_and_time(model, X_test):
    """
    Generate predictions and record inference time.

    Args:
        model: A fitted sklearn estimator.
        X_test: Test feature matrix.

    Returns:
        tuple: (predictions_array, inference_time_seconds)
    """
    start = time.time()
    preds = model.predict(X_test)
    elapsed = time.time() - start
    return preds, elapsed


def compute_metrics(y_true, y_pred, label: str = "") -> dict:
    """
    Compute a full set of classification metrics.

    Reports accuracy, macro-averaged and weighted-averaged precision,
    recall, and F1. Macro treats all classes equally (important for balanced
    dataset). Weighted accounts for class frequency.

    Args:
        y_true (array-like): Ground-truth labels.
        y_pred (array-like): Predicted labels.
        label (str): Identifier string for display.

    Returns:
        dict: Keys — accuracy, precision_macro, recall_macro, f1_macro,
              precision_weighted, recall_weighted, f1_weighted.
    """
    metrics = {
        "label": label,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    print(f"\n{'='*60}")
    print(f"  Evaluation: {label}")
    print(f"{'='*60}")
    print(f"  Accuracy          : {metrics['accuracy']:.4f}")
    print(f"  F1 (Macro)        : {metrics['f1_macro']:.4f}")
    print(f"  F1 (Weighted)     : {metrics['f1_weighted']:.4f}")
    print(f"\n  Full Classification Report:\n")
    print(classification_report(y_true, y_pred, zero_division=0))

    return metrics


def plot_confusion_matrix(y_true, y_pred, class_names, title: str, save_path: str = None):
    """
    Plot and optionally save a labelled confusion matrix heatmap.

    A heatmap is used instead of a raw table because it allows quick visual
    identification of which class pairs are most confused by the model.

    Args:
        y_true (array-like): Ground-truth labels.
        y_pred (array-like): Predicted labels.
        class_names (list): Ordered list of class label strings.
        title (str): Plot title.
        save_path (str, optional): File path to save the figure (e.g., "results/cm.png").
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Confusion matrix saved → {save_path}")

    plt.show()
    return fig
