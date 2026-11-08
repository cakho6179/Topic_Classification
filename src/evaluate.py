"""
Comprehensive Evaluation Suite for 13-Class Topic Classification.
Computes Macro-F1, Granular Per-Topic Metrics, Confusion Matrix, ECE, and Selective Classification.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

from topic_research.src.data_loader import TOPIC_CLASSES, TOPIC_TO_ID, ID_TO_TOPIC

def compute_topic_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Computes primary and granular metrics for the 13 topic classes.
    """
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    # Granular per-topic report
    target_names = [TOPIC_CLASSES[i] for i in range(len(TOPIC_CLASSES))]
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(TOPIC_CLASSES))),
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )

    per_topic = {}
    for topic in TOPIC_CLASSES:
        if topic in report_dict:
            per_topic[topic] = {
                "precision": round(report_dict[topic]["precision"], 4),
                "recall": round(report_dict[topic]["recall"], 4),
                "f1-score": round(report_dict[topic]["f1-score"], 4),
                "support": int(report_dict[topic]["support"])
            }

    # Normalized Confusion Matrix (by true rows)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(TOPIC_CLASSES))))
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = np.nan_to_num(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])

    results = {
        "macro_f1": round(float(macro_f1), 4),
        "micro_f1": round(float(micro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "accuracy": round(float(accuracy), 4),
        "per_topic": per_topic,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": np.round(cm_norm, 4).tolist()
    }

    if y_probs is not None:
        ece = compute_expected_calibration_error(y_true, y_probs)
        results["expected_calibration_error"] = round(float(ece), 4)

    return results

def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Computes Expected Calibration Error (ECE) across probability bins.
    """
    confidences = np.max(y_probs, axis=1)
    predictions = np.argmax(y_probs, axis=1)
    accuracies = (predictions == y_true)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece

def evaluate_selective_classification(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float = 0.55
) -> Dict[str, Any]:
    """
    Simulates a Rejection Option: only accept predictions with max_prob >= threshold.
    """
    confidences = np.max(y_probs, axis=1)
    predictions = np.argmax(y_probs, axis=1)
    
    accepted_mask = confidences >= threshold
    coverage = float(np.mean(accepted_mask))
    
    if np.sum(accepted_mask) > 0:
        selective_acc = float(accuracy_score(y_true[accepted_mask], predictions[accepted_mask]))
        selective_f1 = float(f1_score(y_true[accepted_mask], predictions[accepted_mask], average="macro", zero_division=0))
    else:
        selective_acc = 0.0
        selective_f1 = 0.0

    return {
        "threshold": threshold,
        "coverage": round(coverage, 4),
        "selective_accuracy": round(selective_acc, 4),
        "selective_macro_f1": round(selective_f1, 4),
        "total_samples": len(y_true),
        "accepted_samples": int(np.sum(accepted_mask)),
        "rejected_samples": int(np.sum(~accepted_mask))
    }

def extract_error_cases(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    text_col: str = "target"
) -> pd.DataFrame:
    """
    Extracts misclassified instances with ground truth, prediction, confidence, and text.
    """
    confidences = np.max(y_probs, axis=1)
    misclassified_mask = y_true != y_pred
    
    errors_df = df.iloc[misclassified_mask].copy()
    errors_df["true_topic"] = [ID_TO_TOPIC[y] for y in y_true[misclassified_mask]]
    errors_df["pred_topic"] = [ID_TO_TOPIC[y] for y in y_pred[misclassified_mask]]
    errors_df["confidence"] = np.round(confidences[misclassified_mask], 4)
    errors_df["sample_text"] = errors_df[text_col]
    
    return errors_df[["comment_id", "true_topic", "pred_topic", "confidence", "sample_text"]]
