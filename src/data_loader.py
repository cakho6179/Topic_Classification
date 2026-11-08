"""
Dedicated Data Loading & Split Module for Vietnamese Topic Classification.
Enforces Strict Comment Deduplication and Thread-Level Group Leakage Prevention.
Supports both Vietnamese full topic names and canonical T01-T13 codes.
"""

from __future__ import annotations

import re
import unicodedata
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import GroupShuffleSplit

# Canonical 13 Topic Codes and Mapping
TOPIC_CLASSES = [f"T{i:02d}" for i in range(1, 14)]
TOPIC_TO_ID = {topic: i for i, topic in enumerate(TOPIC_CLASSES)}
ID_TO_TOPIC = {i: topic for topic, i in TOPIC_TO_ID.items()}

# Mapping from raw Vietnamese topic names in datasets to T01-T13
RAW_VI_TOPIC_MAP = {
    "pin & sạc": "T01",
    "chất lượng hạ tầng sạc/ đổi pin": "T02",
    "chất lượng hạ tầng sạc/đổi pin": "T02",
    "chính sách hậu mãi (bảo hành, bảo dưỡng, sửa chữa)": "T03",
    "quy trình đặt hẹn dịch vụ": "T04",
    "hệ thống xưởng dịch vụ (xdv)": "T05",
    "chất lượng dịch vụ tại xdv": "T06",
    "thái độ và chuyên môn của nhân sự xdv": "T07",
    "chi phí sửa chữa, bảo dưỡng, thay thế phụ tùng": "T08",
    "chính sách cứu hộ, hỗ trợ khẩn cấp": "T09",
    "tổng đài cskh": "T10",
    "dịch vụ sửa chữa tại nhà": "T11",
    "đề cập chung": "T12",
    "chi phí & chính sách sạc": "T13",
}

def normalize_topic_code(topic_raw: str) -> Optional[str]:
    """
    Normalizes any topic string (code T01-T13 or raw Vietnamese text) into canonical code T01-T13.
    """
    if not isinstance(topic_raw, str) or not topic_raw.strip():
        return None
    val = unicodedata.normalize("NFC", topic_raw.strip())
    val_upper = val.upper()
    if val_upper in TOPIC_CLASSES:
        return val_upper
    
    val_lower = val.lower()
    return RAW_VI_TOPIC_MAP.get(val_lower, None)

def parse_info_field(info: str) -> Dict[str, str]:
    """
    Parses unstructured 'info' text into thread_info, context, and target.
    """
    if not isinstance(info, str) or not info.strip():
        return {"thread_info": "", "context": "", "target": ""}

    text = info.strip()
    thread_info = ""
    context = ""
    target = ""

    # Check for structured prefix patterns
    thread_match = re.search(r"Thread_info:\s*(.*?)(?=(?:Context:|Target:|$))", text, re.IGNORECASE | re.DOTALL)
    context_match = re.search(r"Context:\s*(.*?)(?=(?:Target:|$))", text, re.IGNORECASE | re.DOTALL)
    target_match = re.search(r"Target:\s*(.*?)$", text, re.IGNORECASE | re.DOTALL)

    if target_match:
        target = target_match.group(1).strip()
        if thread_match:
            thread_info = thread_match.group(1).strip()
        if context_match:
            context = context_match.group(1).strip()
    else:
        # Fallback if no prefixes found: entire text is treated as target
        target = text

    return {
        "thread_info": thread_info,
        "context": context,
        "target": target,
    }

def load_and_deduplicate_topic_data(
    file_path: str,
    comment_col: str = "comment_id",
    thread_col: str = "thread_id",
    info_col: str = "info",
    topic_col: str = "topic"
) -> pd.DataFrame:
    """
    Loads data and performs strict deduplication by comment_id.
    Guarantees that each comment has exactly one topic record.
    """
    if file_path.endswith(".xlsx") or file_path.endswith(".xls"):
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)

    # Validate required columns
    for col in [comment_col, thread_col, topic_col]:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' missing from {file_path}")

    # Standardize topic labels to canonical T01 - T13
    df["canonical_topic"] = df[topic_col].apply(normalize_topic_code)
    
    # Filter for valid topic labels only
    df = df[df["canonical_topic"].notna()].copy()

    # Deduplicate by comment_id
    dedup_df = df.drop_duplicates(subset=[comment_col]).copy()

    # Parse info field if target/context columns not already separated
    if "target" not in dedup_df.columns and info_col in dedup_df.columns:
        parsed_records = dedup_df[info_col].apply(parse_info_field).tolist()
        parsed_df = pd.DataFrame(parsed_records, index=dedup_df.index)
        dedup_df["target"] = parsed_df["target"]
        dedup_df["context"] = parsed_df["context"]
        dedup_df["thread_info"] = parsed_df["thread_info"]

    # Assign integer label IDs
    dedup_df["topic"] = dedup_df["canonical_topic"]
    dedup_df["topic_id"] = dedup_df["canonical_topic"].map(TOPIC_TO_ID)

    return dedup_df.reset_index(drop=True)

def split_topic_dataset(
    df: pd.DataFrame,
    group_col: str = "thread_id",
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits dataset into Train, Validation, and Test sets using GroupShuffleSplit on thread_id.
    Prevents thread context leakage between sets.
    """
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
    train_val_idx, test_idx = next(gss_test.split(df, groups=df[group_col]))
    
    train_val_df = df.iloc[train_val_idx].copy().reset_index(drop=True)
    test_df = df.iloc[test_idx].copy().reset_index(drop=True)

    adjusted_val_size = val_size / (1.0 - test_size)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=adjusted_val_size, random_state=random_seed)
    train_idx, val_idx = next(gss_val.split(train_val_df, groups=train_val_df[group_col]))

    train_df = train_val_df.iloc[train_idx].copy().reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].copy().reset_index(drop=True)

    return train_df, val_df, test_df

def get_class_frequencies(df: pd.DataFrame, topic_id_col: str = "topic_id") -> List[int]:
    """
    Computes sample count per topic class from T01 (0) to T13 (12).
    """
    counts = df[topic_id_col].value_counts().to_dict()
    return [counts.get(i, 0) for i in range(len(TOPIC_CLASSES))]
