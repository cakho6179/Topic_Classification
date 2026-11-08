"""
Specialized Preprocessing Module for Vietnamese Topic Classification.
Enforces Conservative Preprocessing: Preserves Emojis, Case, and Natural Punctuation.
Strictly isolates Topic from Brand information.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

EMPTY_INDICATORS = {
    "",
    "(không có)",
    "không có",
    "none",
    "nan",
    "null",
    "n/a",
    "khong co",
    "(khong co)",
}

def clean_text(text: Optional[str]) -> str:
    """
    Cleans and normalizes text conservatively for Vietnamese NLP.
    
    Principles:
    1. Unicode NFC Normalization (fixes Unikey/EVKey composed vs precomposed accents).
    2. Normalize non-breaking spaces and excessive whitespace.
    3. Filter missing/garbage tokens to empty string.
    4. PRESERVE 100% case, emojis, and natural punctuation.
    """
    if text is None:
        return ""
    
    text = str(text)
    # Unicode NFC normalization
    text = unicodedata.normalize("NFC", text)
    
    # Replace non-breaking space & special whitespace
    text = text.replace("\u00a0", " ").replace("\ufeff", "")
    
    # Collapse multiple spaces while preserving newlines if desired, or single line
    text = re.sub(r"[^\S\r\n]+", " ", text).strip()
    
    if text.lower() in EMPTY_INDICATORS:
        return ""
    
    return text

def format_topic_input(target: str, context: Optional[str] = None) -> str:
    """
    Builds the standardized input string for Topic Classification.
    
    Formula:
    - If context exists: "Target: {target} [SEP] Context: {context}"
    - If context is empty: "Target: {target}"
    
    NOTE: Brand is STRICTLY EXCLUDED to avoid brand bias/leakage in topic prediction.
    """
    c_target = clean_text(target)
    c_context = clean_text(context)
    
    if not c_target:
        # Fallback to context if target is somehow empty
        c_target = c_context
        c_context = ""
    
    if c_context:
        return f"Target: {c_target} [SEP] Context: {c_context}"
    return f"Target: {c_target}"

def segment_words_pyvi(text: str) -> str:
    """
    Applies PyVi word segmentation for PhoBERT-based architectures.
    E.g. 'trạm sạc' -> 'trạm_sạc', 'bảo hành' -> 'bảo_hành'.
    """
    if not text:
        return ""
    try:
        from pyvi import ViTokenizer
        return ViTokenizer.tokenize(text)
    except ImportError:
        # Fallback if pyvi is not installed in the environment
        return text

class TopicPreprocessor:
    """
    End-to-end preprocessor for topic classification records.
    Supports ViSoBERT (raw) and PhoBERT (word segmented).
    """
    def __init__(self, use_word_segmentation: bool = False):
        self.use_word_segmentation = use_word_segmentation

    def process(self, target: str, context: Optional[str] = None) -> str:
        formatted = format_topic_input(target, context)
        if self.use_word_segmentation:
            # Tokenize words while keeping [SEP] token intact
            parts = formatted.split(" [SEP] ")
            segmented_parts = [segment_words_pyvi(p) for p in parts]
            return " [SEP] ".join(segmented_parts)
        return formatted
