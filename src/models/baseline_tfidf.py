"""
Dual-Gram FeatureUnion Baseline Model for 13-Class Topic Classification.
Combines Word n-grams (1,2) with Char-wb n-grams (3,5) and Calibrated Linear SVM.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any, Tuple
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

def build_topic_tfidf_pipeline(
    word_ngram_range: Tuple[int, int] = (1, 2),
    word_min_df: int = 2,
    word_max_df: float = 0.95,
    char_ngram_range: Tuple[int, int] = (3, 5),
    char_min_df: int = 3,
    char_max_df: float = 0.95,
    classifier_type: str = "linear_svm",
    svm_C: float = 1.0,
    calibrate: bool = True
) -> Pipeline:
    """
    Constructs a FeatureUnion TF-IDF pipeline specialized for Vietnamese social listening text.
    """
    # 1. Word n-grams: captures compound phrases (e.g. 'trạm sạc', 'bảo dưỡng')
    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=word_ngram_range,
        min_df=word_min_df,
        max_df=word_max_df,
        sublinear_tf=True,
        norm="l2"
    )

    # 2. Char-wb n-grams: captures subwords, abbreviations ('xdv', 'cskh'), typos, teencode
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=char_ngram_range,
        min_df=char_min_df,
        max_df=char_max_df,
        sublinear_tf=True,
        norm="l2"
    )

    union = FeatureUnion(
        transformer_list=[
            ("word_tfidf", word_vectorizer),
            ("char_tfidf", char_vectorizer),
        ],
        transformer_weights={"word_tfidf": 1.0, "char_tfidf": 0.8}
    )

    # 3. Base Classifier
    if classifier_type == "linear_svm":
        base_clf = LinearSVC(C=svm_C, class_weight="balanced", random_state=42, max_iter=2000)
        if calibrate:
            # Calibrate SVM logits to output true probabilities across 13 classes
            clf = CalibratedClassifierCV(estimator=base_clf, method="sigmoid", cv=3)
        else:
            clf = base_clf
    elif classifier_type == "logistic_regression":
        clf = LogisticRegression(C=svm_C, class_weight="balanced", max_iter=1000, random_state=42)
    else:
        raise ValueError(f"Unsupported classifier: {classifier_type}")

    return Pipeline([
        ("features", union),
        ("classifier", clf)
    ])
