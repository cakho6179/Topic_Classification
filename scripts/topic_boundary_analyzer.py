"""
Semantic Boundary & Vocabulary Overlap Analyzer for 13 Topic Classes.
Computes Pairwise Lexical Overlaps, Top Discriminant Keywords, and Boundary Vulnerabilities.
"""

from __future__ import annotations

import sys
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Set
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from topic_research.src.data_loader import (
    load_and_deduplicate_topic_data,
    TOPIC_CLASSES,
    ID_TO_TOPIC
)
from topic_research.src.preprocessing import clean_text

def analyze_topic_vocabulary(df: pd.DataFrame, top_k: int = 10) -> Dict[str, List[str]]:
    """
    Extracts top discriminant keywords per topic using class-wise TF-IDF.
    """
    topic_keywords = {}
    
    # Group texts by canonical topic
    grouped = df.groupby("canonical_topic")["target"].apply(lambda s: " ".join(clean_text(x) for x in s))
    
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.85)
    tfidf_matrix = vec.fit_transform(grouped.values)
    feature_names = np.array(vec.get_feature_names_out())
    
    for idx, topic in enumerate(grouped.index):
        row = tfidf_matrix[idx].toarray().flatten()
        top_indices = np.argsort(row)[::-1][:top_k]
        topic_keywords[topic] = feature_names[top_indices].tolist()
        
    return topic_keywords

def compute_pairwise_jaccard(topic_keywords: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Computes pairwise Jaccard similarity across the top keywords of the 13 topics.
    High Jaccard indicates semantic overlap and boundary vulnerability.
    """
    topics = sorted(list(topic_keywords.keys()))
    matrix = np.zeros((len(topics), len(topics)))
    
    for i, t1 in enumerate(topics):
        set1 = set(topic_keywords[t1])
        for j, t2 in enumerate(topics):
            set2 = set(topic_keywords[t2])
            union_len = len(set1.union(set2))
            inter_len = len(set1.intersection(set2))
            matrix[i, j] = inter_len / max(union_len, 1)
            
    return pd.DataFrame(matrix, index=topics, columns=topics)

def main():
    print("=" * 70)
    print("TOPIC SEMANTIC BOUNDARY & VOCABULARY OVERLAP ANALYZER")
    print("=" * 70)

    data_path = "data/raw/Vietnamese_Social_Listening_Synthetic_1000.xlsx"
    if not os.path.exists(data_path):
        data_path = "data/processed/clean_data.csv"
        
    df = load_and_deduplicate_topic_data(data_path)
    print(f"Loaded {len(df)} deduplicated comments across {df['canonical_topic'].nunique()} classes.")

    print("\n[1] Top 8 Discriminant Keywords per Topic (TF-IDF):")
    print("-" * 70)
    keywords = analyze_topic_vocabulary(df, top_k=8)
    for topic, kw in sorted(keywords.items()):
        print(f"• {topic:<5}: {', '.join(kw)}")

    print("\n[2] High-Vulnerability Confusion Pairs (Vocabulary Overlap):")
    print("-" * 70)
    jaccard_df = compute_pairwise_jaccard(keywords)
    
    # Find top non-diagonal overlaps
    pairs = []
    topics = jaccard_df.index.tolist()
    for i in range(len(topics)):
        for j in range(i + 1, len(topics)):
            t1, t2 = topics[i], topics[j]
            score = jaccard_df.loc[t1, t2]
            common = set(keywords[t1]).intersection(set(keywords[t2]))
            pairs.append((t1, t2, score, list(common)))
            
    pairs.sort(key=lambda x: x[2], reverse=True)
    for t1, t2, score, common in pairs[:5]:
        print(f"  [{t1} <--> {t2}] Overlap: {score:.3f} | Common terms: {common if common else 'Distinct'}")

    # Generate Markdown Report
    report_path = "topic_research/reports/boundary_analysis_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# BÁO CÁO PHÂN TÍCH RANH GIỚI TỪ VỰNG & MỨC ĐỘ CHỒNG LẤN 13 CHỦ ĐỀ\n\n")
        f.write(f"- Tổng số mẫu bình luận: {len(df)}\n")
        f.write("- Số lượng chủ đề: 13 (T01 - T13)\n\n")
        f.write("## 1. Top Từ Khóa Đặc Trưng Theo Từng Chủ Đề\n\n")
        for topic, kw in sorted(keywords.items()):
            f.write(f"* **{topic}:** `{', '.join(kw)}`\n")
        f.write("\n## 2. Các Cặp Chủ Đề Có Độ Chồng Lấn Từ Vựng Cao Nhất\n\n")
        for t1, t2, score, common in pairs[:7]:
            f.write(f"* **{t1} ⟷ {t2}:** Jaccard Overlap = `{score:.3f}` | Từ khóa chung: `{common}`\n")
            
    print(f"\n[Done] Analysis saved to {report_path}")

if __name__ == "__main__":
    main()
