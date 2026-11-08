"""
Execution Script: Train and Evaluate Dual-Gram FeatureUnion Baseline for 13-Class Topic Classification.
"""

from __future__ import annotations

import sys
import os
import json
import numpy as np

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from topic_research.src.data_loader import (
    load_and_deduplicate_topic_data,
    split_topic_dataset,
    get_class_frequencies,
    TOPIC_CLASSES
)
from topic_research.src.preprocessing import TopicPreprocessor
from topic_research.src.models.baseline_tfidf import build_topic_tfidf_pipeline
from topic_research.src.evaluate import (
    compute_topic_metrics,
    evaluate_selective_classification,
    extract_error_cases
)

def main():
    print("=" * 70)
    print("TOPIC CLASSIFICATION: DUAL-GRAM FEATUREUNION TF-IDF BASELINE")
    print("=" * 70)

    # 1. Load and Deduplicate Data
    data_path = "data/raw/Vietnamese_Social_Listening_Synthetic_1000.xlsx"
    if not os.path.exists(data_path):
        data_path = "data/processed/clean_data.csv"
        
    print(f"\n[Step 1] Loading data from {data_path}...")
    df = load_and_deduplicate_topic_data(data_path)
    print(f"-> Total unique comments deduplicated by comment_id: {len(df)}")
    
    counts = get_class_frequencies(df)
    print("-> Class distribution across 13 topics:")
    for t, c in zip(TOPIC_CLASSES, counts):
        pct = c / len(df) * 100
        print(f"   {t}: {c:4d} samples ({pct:5.1f}%)")

    # 2. Prevent Thread Leakage with Group Split
    print("\n[Step 2] Splitting dataset by thread_id (GroupShuffleSplit)...")
    train_df, val_df, test_df = split_topic_dataset(df, group_col="thread_id", test_size=0.15, val_size=0.15)
    print(f"-> Train samples: {len(train_df)} | Val samples: {len(val_df)} | Test samples: {len(test_df)}")

    # 3. Preprocess Inputs (Target + optional Context, strictly NO brand)
    print("\n[Step 3] Preprocessing text inputs (Conservative cleaning)...")
    preprocessor = TopicPreprocessor(use_word_segmentation=False)
    
    X_train = [preprocessor.process(t, c) for t, c in zip(train_df["target"], train_df["context"])]
    y_train = train_df["topic_id"].to_numpy()

    X_val = [preprocessor.process(t, c) for t, c in zip(val_df["target"], val_df["context"])]
    y_val = val_df["topic_id"].to_numpy()

    X_test = [preprocessor.process(t, c) for t, c in zip(test_df["target"], test_df["context"])]
    y_test = test_df["topic_id"].to_numpy()

    # 4. Build and Train FeatureUnion Pipeline
    print("\n[Step 4] Training Dual-Gram FeatureUnion (Word 1-2 & Char-wb 3-5) + Calibrated Linear SVM...")
    pipeline = build_topic_tfidf_pipeline(
        word_ngram_range=(1, 2),
        char_ngram_range=(3, 5),
        classifier_type="linear_svm",
        calibrate=True
    )
    pipeline.fit(X_train, y_train)
    print("-> Model training complete.")

    # 5. Evaluate on Validation Set
    print("\n[Step 5] Evaluating on Validation Set...")
    val_probs = pipeline.predict_proba(X_val)
    val_preds = np.argmax(val_probs, axis=1)
    val_metrics = compute_topic_metrics(y_val, val_preds, val_probs)
    print(f"-> Validation Macro-F1: {val_metrics['macro_f1']:.4f}")
    print(f"-> Validation Accuracy: {val_metrics['accuracy']:.4f}")

    # 6. Evaluate on Test Set (Final Independent Benchmark)
    print("\n[Step 6] Evaluating on Test Set...")
    test_probs = pipeline.predict_proba(X_test)
    test_preds = np.argmax(test_probs, axis=1)
    test_metrics = compute_topic_metrics(y_test, test_preds, test_probs)

    print(f"\n" + "=" * 50)
    print(f"FINAL TEST RESULTS (BASELINE TF-IDF)")
    print(f"=" * 50)
    print(f"• Macro-F1:  {test_metrics['macro_f1']:.4f}")
    print(f"• Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"• Weighted-F1: {test_metrics['weighted_f1']:.4f}")
    if "expected_calibration_error" in test_metrics:
        print(f"• ECE:        {test_metrics['expected_calibration_error']:.4f}")

    print("\nPer-Topic Performance Breakdown (Test Set):")
    print(f"{'Topic':<7} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 55)
    for topic, stats in test_metrics["per_topic"].items():
        print(f"{topic:<7} | {stats['precision']:<10.4f} | {stats['recall']:<10.4f} | {stats['f1-score']:<10.4f} | {stats['support']:<8}")

    # 7. Evaluate Selective Classification (Rejection Option)
    print("\nSelective Classification Analysis (Threshold Sweep):")
    print(f"{'Threshold':<10} | {'Coverage':<10} | {'Selective Acc':<14} | {'Accepted':<10} | {'Rejected':<10}")
    print("-" * 65)
    for tau in [0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8]:
        res = evaluate_selective_classification(y_test, test_probs, threshold=tau)
        print(f"{tau:<10.2f} | {res['coverage']*100:<9.1f}% | {res['selective_accuracy']*100:<13.1f}% | {res['accepted_samples']:<10} | {res['rejected_samples']:<10}")

    # 8. Save artifacts
    output_report_path = "topic_research/reports/baseline_results.json"
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": "DualGram_FeatureUnion_Calibrated_LinearSVM",
            "val_metrics": val_metrics,
            "test_metrics": test_metrics
        }, f, indent=2, ensure_ascii=False)
    print(f"\n[Done] Full benchmark results saved to {output_report_path}")

if __name__ == "__main__":
    main()
