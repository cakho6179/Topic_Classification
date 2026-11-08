# TOPIC RESEARCH INITIATIVE
## Vietnamese Social Listening NLP Engine: 13-Class Single-Label Topic Classification
**Thư mục nghiên cứu chuyên sâu:** `topic_research/`  
**Ngày cập nhật:** 09/11/2026  

---

## 1. TỔNG QUAN DỰ ÁN NGHIÊN CỨU

Thư mục `topic_research/` là không gian nghiên cứu chuyên sâu (Deep Research & Engineering Sandbox) dành riêng cho bài toán **Phân loại đơn nhãn 13 chủ đề (T01 – T13)** trong ngành ô tô và xe điện tại Việt Nam.

Khác với các phương pháp tiếp cận chung chung, thư mục này tập trung vào 2 trụ cột kỹ thuật cốt lõi:
1. **Pipeline Tiền Xử Lý Dữ Liệu Chuyên Sâu (End-to-End Data Pipeline):** Bảo toàn tối đa ngữ nghĩa tiếng Việt, xử lý triệt để xung đột Unicode, chuẩn hóa dấu thanh, bảo tồn emoji/case và chống rò rỉ dữ liệu (Thread Leakage).
2. **Kiến Trúc Mô Hình & Nghiên Cứu Decision Models Mới Nhất:** So sánh toàn diện giữa Encoders truyền thống (ViSoBERT, PhoBERT), **các hệ Decision Models đột phá (Jev của TypeSafe AI, d1 của Liquid AI, Quyet)** và Generative SLMs (Qwen3.5-4B LoRA SFT).

---

## 2. BỘ TÀI LIỆU CHUYÊN KHẢO KỸ THUẬT (TECHNICAL WHITEPAPERS)

Bộ tài liệu chuyên khảo được biên soạn độc lập tại thư mục [`topic_research/doc/`](file:///d:/VSF/Sentiment_Topic_Classification/topic_research/doc):

| Tài liệu | Nội dung trọng tâm |
|---|---|
| **[01_Data_Pipeline_Deep_Dive.md](file:///d:/VSF/Sentiment_Topic_Classification/topic_research/doc/01_Data_Pipeline_Deep_Dive.md)** | • Quy trình 6 chặng từ Raw Data $\rightarrow$ Input Tensor.<br>• Giải quyết xung đột Unicode NFC vs. NFD, vị trí dấu thanh kiểu mới vs. cũ.<br>• Conservative Preprocessing: Giữ 100% Emoji, Case, dấu câu, cơ chế BPE cho từ kéo dài.<br>• Thuật toán regex phân rã `thread_info`, `context`, `target`, cách ly `brand`.<br>• Chiến lược Left-Truncation cho Context (bảo tồn 100% Target).<br>• Khử trùng lặp `comment_id` và GroupKFold Split theo `thread_id` chống leakage. |
| **[02_Modeling_and_Decision_Models_Deep_Dive.md](file:///d:/VSF/Sentiment_Topic_Classification/topic_research/doc/02_Modeling_and_Decision_Models_Deep_Dive.md)** | • Tiến hóa phân loại: Encoders tĩnh vs. Generative LLMs (System 2) vs. Decision Models (System 1).<br>• **Nghiên cứu chuyên sâu Decision Models:** Jev (TypeSafe AI), d1 (Liquid AI - LFM), Quyet.<br>• Bảng ma trận đối sánh 6 họ mô hình: Latency, Throughput, VRAM, Type-safety, Macro-F1.<br>• Toán học Transformer: Learned Attention Pooling, Multi-Sample Dropout (MSD).<br>• Tối ưu Loss lệch lớp đuôi dài: Weighted CE, Focal Loss, Class-Balanced Loss ($E_n$), LDAM Loss.<br>• Học tương phản có giám sát (SupCon) phân tách ranh giới.<br>• Selective Classification: Temperature Scaling, ECE, cơ chế Rejection Threshold $\tau$. |
| **[03_Taxonomy_and_Boundary_Disambiguation.md](file:///d:/VSF/Sentiment_Topic_Classification/topic_research/doc/03_Taxonomy_and_Boundary_Disambiguation.md)** | • Đặc tả chi tiết 13 topic (In-scope, Out-of-scope, Keywords đặc thù).<br>• Cấu trúc phân cấp 4 Macro-clusters: Hạ tầng, XDV, CSKH & Hậu mãi, Đề cập chung.<br>• **Ma trận 7 cặp ranh giới chồng lấn lớn nhất** (Confusion Pairs) kèm ví dụ thực tế.<br>• Bộ thuật toán giải quyết tranh chấp nhãn (Tie-breaking Algorithm 4 bước). |
| **[04_Evaluation_and_Error_Taxonomy.md](file:///d:/VSF/Sentiment_Topic_Classification/topic_research/doc/04_Evaluation_and_Error_Taxonomy.md)** | • Hệ thống chỉ số phân cấp: Macro-F1 (trọng tâm), Per-topic metrics, Rare-class Recall.<br>• Phân tích Ma trận nhầm lẫn chuẩn hóa (Row & Column Normalized Confusion Matrix).<br>• Phân loại 4 nhóm lỗi kỹ thuật (Error Taxonomy Loại A, B, C, D) có thể hành động.<br>• Bộ 3 bài kiểm thử đối kháng tự động (Stress Testing): Brand-Swap Invariance, Context Ablation, Teencode Perturbation. |

---

## 3. CẤU TRÚC MÃ NGUỒN (SOURCE CODE ARCHITECTURE)

```
topic_research/
├── configs/
│   └── topic_config.yaml               <- Cấu hình siêu tham số, data path, loss settings
├── doc/
│   ├── 01_Data_Pipeline_Deep_Dive.md
│   ├── 02_Modeling_and_Decision_Models_Deep_Dive.md
│   ├── 03_Taxonomy_and_Boundary_Disambiguation.md
│   └── 04_Evaluation_and_Error_Taxonomy.md
├── reports/
│   └── baseline_results.json           <- Kết quả benchmark chi tiết của Baseline
├── scripts/
│   ├── run_topic_baseline.py           <- Script thực thi huấn luyện & đánh giá Baseline TF-IDF
│   └── topic_boundary_analyzer.py      <- Script phân tích ranh giới từ vựng giữa 13 lớp
└── src/
    ├── __init__.py
    ├── data_loader.py                  <- Loader khử trùng lặp & GroupShuffleSplit theo thread_id
    ├── preprocessing.py               <- Text Normalizer chuẩn NFC, Task Input Formatter
    ├── losses.py                       <- Weighted CE, Focal Loss, Class-Balanced Focal Loss
    ├── evaluate.py                     <- Tính toán Macro-F1, Per-topic F1, ECE, Selective Acc
    └── models/
        ├── __init__.py
        ├── baseline_tfidf.py           <- Dual-Gram FeatureUnion + Calibrated Linear SVM
        └── transformer_classifier.py   <- Transformer Topic Classifier (MSD, Attention Pooling)
```

---

## 4. HƯỚNG DẪN THỰC THI (EXECUTION)

### Chạy Thực Nghiệm Baseline TF-IDF FeatureUnion:
```bash
python topic_research/scripts/run_topic_baseline.py
```
*Script sẽ tự động:*
1. Khử trùng lặp bình luận theo `comment_id`.
2. Phân chia tập dữ liệu Train / Val / Test theo nhóm `thread_id`.
3. Chuẩn hóa chuỗi văn bản theo nguyên tắc bảo toàn (Conservative).
4. Huấn luyện Dual-Gram FeatureUnion (Word 1-2 & Char-wb 3-5) kết hợp Calibrated Linear SVM.
5. Xuất báo cáo điểm số Macro-F1, bảng Per-Topic Breakdown và phân tích ngưỡng tin cậy Selective Classification vào `topic_research/reports/baseline_results.json`.
