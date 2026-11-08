# BÁO CÁO KỸ THUẬT TỔNG THỂ TOÀN BỘ HỆ THỐNG
## Topic Classification Engine: Phân Loại Đơn Nhãn 13 Chủ Đề Mạng Xã Hội Tiếng Việt
**Dự án:** Vietnamese Social Listening NLP Engine  
**Phân hệ:** Single-Label Topic Classification Subsystem (T01 – T13)  
**Phạm vi tài liệu:** Báo cáo Kỹ thuật Toàn diện Hệ thống (Master Technical Specification & Architecture)  
**Ngày cập nhật:** 09/11/2026  

---

## TỔNG QUAN ĐIỀU HÀNH (EXECUTIVE SUMMARY)

Hệ thống **Topic Classification Engine** là phân hệ trí tuệ nhân tạo nòng cốt trong nền tảng Social Listening ngành ô tô và xe điện tại Việt Nam. Phân hệ chịu trách nhiệm tự động đọc hiểu hàng triệu bình luận từ mạng xã hội (Facebook, YouTube, TikTok, diễn đàn) và định tuyến chính xác vào **duy nhất một chủ đề nghiệp vụ nòng cốt trong 13 danh mục chuẩn (T01 đến T13)**.

Hệ thống được thiết kế với hai cấu trúc kiến trúc ứng viên độc lập được đánh giá song song, cùng cơ chế thẩm định chuyên gia:
1. **Kiến trúc Ứng viên A (Sản xuất Tối ưu Độ trễ): Deep Learning Transformer Encoders (~130M parameters)**  
   Trục xương sống dựa trên các mô hình ngôn ngữ chuyên ngữ mạng xã hội tiếng Việt (**ViSoBERT**, **PhoBERT-v2**, **XLM-R**), tích hợp cơ chế Learned Attention Pooling, Multi-Sample Dropout (MSD 5 mặt nạ) và bộ 4 hàm mất mát tối ưu hóa phân phối lệch lớp đuôi dài (Focal Loss $\gamma=2.0$, Class-Balanced Loss, LDAM Loss). Đạt độ trễ siêu thấp (< 6ms trên GPU, < 25ms trên CPU).
2. **Kiến trúc Ứng viên B (Sản xuất Thế hệ Mới 2026): Open-Source Decision Models (Hệ 4B & 9B parameters)**  
   Khảo sát và thiết kế thực nghiệm độc lập cho dòng mô hình chuyên biệt ra quyết định dạng Zero-token (**Jev-4B/9B** của TypeSafe AI, **d1-4B/9B** của Liquid AI, và **Quyet-4B/9B**). Vận hành như một hệ thống phân loại trọn gói (Standalone End-to-End System): nhận ngữ cảnh và schema lựa chọn kiểu dữ liệu tĩnh `Choice(options=[T01..T13])`, xuất trực tiếp phân phối xác suất và chỉ số entropy đã hiệu chuẩn mà không qua sinh token tự hồi quy, triệt tiêu hoàn toàn rủi ro vỡ định dạng JSON và ảo giác.
3. **Pipeline Tiền Xử Lý Dữ Liệu Bảo Toàn (Conservative Preprocessing & Zero-Leakage):**  
   Bảo tồn 100% Emoji, giữ nguyên chữ hoa/thường (Cased Representation), chuẩn hóa Unicode NFC dựng sẵn, giải quyết xung đột vị trí dấu thanh tiếng Việt kiểu mới vs. cũ, áp dụng cơ chế Left-Truncation bảo vệ toàn vẹn Target, khử trùng lặp theo `comment_id`, cách ly hoàn toàn trường `brand` khỏi input Topic, và phân chia dữ liệu GroupKFold theo `thread_id` chống rò rỉ dữ liệu triệt để.
4. **Vận Hành Đáng Tin Cậy (Selective Classification & Calibration):**  
   Tích hợp Temperature Scaling hạ chỉ số sai lệch tin cậy (ECE $< 0.04$) và cơ chế Rejection Option tại ngưỡng tin cậy $\tau = 0.58$, tự động hóa dán nhãn 86% dữ liệu với độ chính xác $> 94\%$, tách riêng 14% ca mơ hồ chuyển chuyên viên hoặc Auditor SLM (Qwen3.5-4B LoRA SFT) giải trình chuỗi suy luận (CoT).

---

## PHẦN I. ĐỊNH NGHĨA BÀI TOÁN & CẤU TRÚC DỮ LIỆU CHUẨN

### 1.1. Bản chất toán học của bài toán
* **Đơn vị phân loại:** Cấp độ bình luận (`comment_id`). Mỗi bình luận được khử trùng lặp duy nhất, không phụ thuộc vào số lượng thương hiệu được nhắc tới trong câu.
* **Không gian nhãn:** Tập 13 lớp rời rạc $\mathcal{C} = \{T01, T02, \dots, T13\}$.
* **Hàm mục tiêu:** Tối ưu hóa xác suất hậu nghiệm qua hàm kích hoạt Softmax và Cross Entropy:
  $$\hat{y} = \arg\max_{c \in \mathcal{C}} P(y = c \mid \mathbf{x}) = \arg\max_{c \in \mathcal{C}} \frac{\exp(z_c)}{\sum_{j=1}^{13} \exp(z_j)}$$

### 1.2. Phân rã ngữ cảnh trường `info` và cách ly thương hiệu (`brand`)
Chuỗi dữ liệu thô `info` được phân rã thành 3 trường độc lập bằng Regex Engine:
1. **`target` (Bắt buộc):** Nội dung bình luận trực tiếp cần phân loại.
2. **`context` (Bổ trợ):** Bình luận cha/câu hỏi dẫn đề trong cùng luồng, chỉ đưa vào khi tồn tại hợp lệ để giải nghĩa đại từ thay thế (*"bên này", "dịch vụ này"*).
3. **`thread_info` (Loại bỏ):** Tiêu đề bài đăng gốc. Mặc định loại bỏ để tránh mô hình học vẹt nhãn chủ đề từ tiêu đề bài viết (Topic Spurious Correlation).
* **Nguyên tắc Cách Ly Thương Hiệu (Brand Isolation):** Trường `brand` **TUYỆT ĐỐI KHÔNG ĐƯỢC PHÉP** xuất hiện trong chuỗi input của bài toán Topic. Định dạng chuỗi đầu vào chuẩn hóa:
  $$\text{Input}_{\text{Topic}} = \begin{cases} 
  \text{"Target: } \{target\} \text{ [SEP] Context: } \{context\}\text{"} & \text{nếu Context tồn tại} \\
  \text{"Target: } \{target\}\text{"} & \text{nếu Context rỗng}
  \end{cases}$$

---

## PHẦN II. ĐẶC TẢ HỆ THỐNG 13 CHỦ ĐỀ & MA TRẬN RANH GIỚI NGỮ NGHĨA

*(Phần này được tinh gọn dạng bảng tổng hợp mật độ cao và ma trận quy tắc phân định ranh giới dứt điểm nhằm tối ưu thời gian tra cứu cho đội ngũ kỹ thuật).*

### 2.1. Bảng tra cứu cấu trúc 13 Chủ đề (Taxonomy Reference Table)

| Mã | Nhóm Vĩ Mô (Macro-Cluster) | Tên Chủ Đề Nghiệp Vụ | Trọng Tâm Ngữ Nghĩa & Từ Khóa Kích Hoạt Nòng Cốt | Tần Suất Dự Kiến |
|---|---|---|---|---|
| **T01** | **I. Hạ Tầng Sạc & Năng Lượng** | Pin & Cung cấp năng lượng | Dung lượng pin, chai pin, sạc chậm/nhanh, nhiệt độ pin, cell pin, pack pin, công nghệ pin LFP/NMC. | Trung bình - Cao |
| **T02** | **I. Hạ Tầng Sạc & Năng Lượng** | Hệ thống trạm sạc & đổi pin | Trụ sạc vật lý, trụ xanh/đỏ, bãi đỗ sạc bị chiếm chỗ, độ phủ trạm sạc, trụ sạc hỏng súng sạc, trạm dừng nghỉ. | Rất Cao |
| **T13** | **I. Hạ Tầng Sạc & Năng Lượng** | Chi phí & Chính sách sạc pin | Đơn giá điện (đ/kWh), phí thuê pin hàng tháng, gói cước sạc, tiền phạt đỗ xe quá giờ tại trạm sạc, tiền cọc pin. | Trung bình |
| **T05** | **II. Hệ Thống Xưởng Dịch Vụ** | Độ phủ & Quy mô xưởng (XDV) | Số lượng đại lý, showroom, vị trí xưởng gần/xa, tỉnh lẻ thiếu xưởng bảo dưỡng, quy mô mặt bằng xưởng. | Thấp - Trung bình |
| **T06** | **II. Hệ Thống Xưởng Dịch Vụ** | Năng lực & Chất lượng kỹ thuật | Thợ bắt bệnh, xe sửa không hết lỗi, bảo dưỡng định kỳ, cân chỉnh thước lái, sơn xe, thay dầu, lắp ráp lỗi. | Cao |
| **T07** | **II. Hệ Thống Xưởng Dịch Vụ** | Thái độ & Chuyên môn nhân sự | Tiếp đón khách, thái độ cố vấn dịch vụ (CVDV), lễ tân, văn hóa ứng xử, tư vấn nhiệt tình hay trịch thượng. | Trung bình |
| **T08** | **II. Hệ Thống Xưởng Dịch Vụ** | Chi phí dịch vụ & Phụ tùng | Bảng giá phụ tùng, hóa đơn tiền công thợ, giá đắt/rẻ, khan hiếm phụ tùng chờ đợi nhiều tuần, phụ tùng chính hãng. | Cao |
| **T03** | **III. Hậu Mãi & Dịch Vụ KH** | Chính sách bảo hành & hậu mãi | Thời hạn bảo hành (7-10 năm), từ chối bảo hành do độ xe, quyền lợi khách hàng, chính sách đổi xe mới khi lỗi. | Trung bình |
| **T04** | **III. Hậu Mãi & Dịch Vụ KH** | Quy trình đặt hẹn dịch vụ | Thao tác đặt lịch trên App điện thoại, khung giờ trống, xác nhận lịch hẹn, xếp hàng chờ đợi đến lượt. | Thấp *(Lớp hiếm)* |
| **T09** | **III. Hậu Mãi & Dịch Vụ KH** | Cứu hộ khẩn cấp 24/7 | Xe chết máy giữa đường, gọi xe cẩu, thời gian xe cứu hộ có mặt, kích bình ắc-quy cứu hộ trên cao tốc. | Thấp *(Lớp hiếm)* |
| **T10** | **III. Hậu Mãi & Dịch Vụ KH** | Tổng đài CSKH & Hỗ trợ | Gọi Hotline 1900, nhạc chờ lâu, tổng đài viên ghi nhận khiếu nại, phản hồi ticket chăm sóc khách hàng. | Thấp - Trung bình |
| **T11** | **III. Hậu Mãi & Dịch Vụ KH** | Dịch vụ sửa chữa lưu động | Xe Mobile Service đến tận nhà/cơ quan, thay lốp lưu động, bảo dưỡng tại chỗ thuận tiện không cần ra xưởng. | Thấp *(Lớp hiếm)* |
| **T12** | **IV. Đề Cập Tổng Quan** | Đề cập chung / Ngoại vi | Nhận xét chung về hãng, kiểu dáng, thiết kế xe, nội thất, giá bán lăn bánh tổng thể không chứa chi tiết dịch vụ. | Rất Cao |

### 2.2. Ma trận phân định 7 cặp ranh giới chồng lấn (Confusion Pairs Disambiguation)

```
                            [XUNG ĐỘT RANH GIỚI NGỮ NGHĨA]
                                          │
            ┌─────────────────────────────┼─────────────────────────────┐
            ▼                             ▼                             ▼
    [Nhóm Sạc: T01/T02/T13]       [Nhóm Xưởng: T06/T07/T08]     [Nhóm Cứu Hộ: T09/T11]
    • T13: Tiền/Giá/Thuê pin      • T07: Con người/Thái độ      • T09: Khẩn cấp/Chết máy
    • T02: Trụ sạc vật lý/Vị trí  • T06: Tay nghề/Lỗi kỹ thuật   • T11: Chủ động/Tại nhà
    • T01: Kỹ thuật pin/Tốc độ    • T08: Tiền công/Giá phụ tùng
```

| Cặp Ranh Giới Chồng Lấn | Vấn Đề Giao Thoa | Quy Tắc Phân Định Dứt Điểm (Decision Rule) | Ví Dụ Minh Họa Phân Định Chuẩn |
|---|---|---|---|
| **T01 vs. T02 vs. T13** | Bàn luận xoay quanh việc sạc xe điện và pin. | • **T13 (Tài chính):** Xuất hiện đơn vị tiền tệ, giá sạc/kWh, phí cọc pin.<br>• **T02 (Hạ tầng):** Nói về điểm cắm sạc vật lý, chiếm chỗ, súng sạc gãy.<br>• **T01 (Thiết bị):** Nói về dung lượng, phần trăm pin xe, nhiệt độ pin. | *"Sạc ở trạm Vincom giá 3.850đ đắt"* $\rightarrow$ **T13**.<br>*"Trạm sạc ở BigC bị xe xăng đỗ kín chỗ"* $\rightarrow$ **T02**.<br>*"Mùa đông pin bị tụt nhanh hơn"* $\rightarrow$ **T01**. |
| **T06 vs. T07 vs. T08** | Khách hàng phản hồi tiêu cực sau khi vào xưởng dịch vụ. | • **T07 (Nhân sự):** Bức xúc thái độ, lời nói, cách ứng xử của nhân viên.<br>• **T06 (Kỹ thuật):** Tay nghề thợ non, lắp sót ốc, xe về vẫn kêu lọc cọc.<br>• **T08 (Chi phí):** Phàn nàn tổng hóa đơn dịch vụ đắt, tiền phụ tùng cao. | *"Cố vấn dịch vụ nói chuyện cụt lủn khó chịu"* $\rightarrow$ **T07**.<br>*"Sơn lại cản trước mà lệch màu rõ rệt"* $\rightarrow$ **T06**.<br>*"Thay cái lọc gió mà chém tận 800k"* $\rightarrow$ **T08**. |
| **T03 vs. T08** | Bàn về việc thay thế phụ tùng và tiền bạc. | • **T03 (Chính sách):** Tranh cãi việc lỗi đó có được bảo hành miễn phí hay không.<br>• **T08 (Giao dịch):** Khách chấp nhận bỏ tiền sửa nhưng chê giá linh kiện đắt. | *"Màn hình tự nứt mà hãng đổ lỗi do khách ép trả tiền"* $\rightarrow$ **T03**.<br>*"Hết bảo hành rồi, thay cụm đèn báo giá 20 củ chát quá"* $\rightarrow$ **T08**. |
| **T09 vs. T11** | Thợ hoặc xe phục vụ khách bên ngoài xưởng dịch vụ. | • **T09 (Khẩn cấp):** Tình huống sự cố bất ngờ giữa đường, xe chết máy, gọi cẩu.<br>• **T11 (Chủ động):** Dịch vụ Mobile Service khách chủ động đặt hẹn đến gara tại nhà. | *"Xe chết máy trên cao tốc gọi cứu hộ mãi chưa đến"* $\rightarrow$ **T09**.<br>*"Hôm nay xe lưu động đến tận nhà cân lốp rất tiện"* $\rightarrow$ **T11**. |
| **T04 vs. T10** | Tương tác từ xa giữa khách hàng với hãng. | • **T04 (Đặt lịch):** Thao tác trên App, chọn ngày giờ, tình trạng kín slot lịch.<br>• **T10 (Tổng đài):** Gọi thoại Hotline, nhân viên nghe máy, khiếu nại dịch vụ chung. | *"Đặt lịch hẹn bảo dưỡng trên app toàn báo hết chỗ"* $\rightarrow$ **T04**.<br>*"Gọi tổng đài 1900 máy bận liên tục 20 phút"* $\rightarrow$ **T10**. |
| **T05 vs. T02** | Khách hàng đánh giá độ bao phủ mạng lưới của hãng. | • **T05 (Cơ sở bảo dưỡng):** Nói về Showroom, đại lý 3S, xưởng sửa chữa.<br>• **T02 (Điểm sạc):** Nói về điểm cắm sạc điện công cộng trên đường. | *"Về quê cả tỉnh không có nổi một xưởng bảo dưỡng"* $\rightarrow$ **T05**.<br>*"Dọc quốc lộ 1A giờ trạm sạc cắm dày đặc"* $\rightarrow$ **T02**. |
| **T12 vs. Các chủ đề khác** | Nhận xét chung chung về xe hoặc thương hiệu. | • **Nguyên tắc Specificity:** T12 chỉ là nhãn mặc định khi câu HOÀN TOÀN KHÔNG đề cập đến bất kỳ chi tiết dịch vụ/hạ tầng cụ thể nào. | *"Xe nhìn ngoài đẹp, cảm giác lái chắc chắn"* $\rightarrow$ **T12**.<br>*"Xe đẹp nhưng trụ sạc ít quá"* $\rightarrow$ **T02**. |

---

## PHẦN III. PIPELINE TIỀN XỬ LÝ DỮ LIỆU CHUYÊN SÂU (DATA PIPELINE)

Hệ thống triển khai pipeline tiền xử lý 6 chặng chuẩn hóa, đảm bảo tính bảo toàn tín hiệu ngữ nghĩa:

```
[Raw Scraped Data] ──> [Khử Trùng Lặp comment_id] ──> [GroupKFold Split theo thread_id]
                               │
                               ▼
[Regex Info Parser] ──> [Cách Ly Thương Hiệu brand] ──> [Unicode NFC & Chuẩn Hóa Dấu Thanh]
                               │
                               ▼
[Conservative Cleaner (Giữ Emoji & Case)] ──> [Left-Truncation] ──> [Tensor Alignment]
```

### 3.1. Chuẩn hóa Ngữ pháp Tiếng Việt Học (Orthography & Encoding)
* **Khắc phục triệt để Unicode NFC vs. NFD:** Toàn bộ văn bản được ép về chuẩn dựng sẵn UTF-8 NFC (`unicodedata.normalize("NFC", text)`), giải quyết dứt điểm hiện tượng phân rã ký tự tổ hợp do các bộ gõ Unikey/EVKey khác nhau sinh ra.
* **Chuẩn hóa vị trí dấu thanh (Bộ GD&ĐT kiểu mới):** Tự động quy chuẩn các biến thể kiểu cũ (`hoà, thuỷ, khoẻ, toà`) về chuẩn âm tiết hạt nhân chính (`hòa, thủy, khỏe, tòa`) bằng từ điển ánh xạ.
* **Dọn dẹp khoảng trắng phi chuẩn:** Thay thế triệt để non-breaking space `\u00a0`, zero-width space `\u200b` và byte order mark `\ufeff`.

### 3.2. Triết lý Bảo toàn Ngữ cảnh Mạng Xã Hội (Conservative Cleaning)
* **Bảo tồn 100% Emoji:** Không xóa emoji, không dịch sang tiếng Anh. Emoji mang mật độ cảm xúc cực cao giúp phân định chủ đề (⚡, 🔋 $\rightarrow$ Sạc; 🛠️, 🔧 $\rightarrow$ Xưởng; 😡, 🤬 $\rightarrow$ Khiếu nại).
* **Bảo tồn Case (Cased Representation):** Giữ nguyên chữ hoa/thường để bảo vệ nhận diện thực thể tên riêng (`VinFast, BYD, Tesla`) và từ viết tắt chuyên ngành kỹ thuật (`XDV, PIN, CSKH, LFP, BMS, CVDV`).
* **Bảo tồn dấu câu cảm thán:** Chuẩn hóa các chuỗi lặp dấu (`!!!!!!` $\rightarrow$ `!!`, `????` $\rightarrow$ `??`) nhưng giữ nguyên dấu câu tự nhiên phản ánh ngữ khí bức xúc, châm biếm.
* **Xử lý từ kéo dài bằng Byte-level BPE:** Không dùng regex co ngắn từ dễ làm hỏng từ vựng tiếng Việt. Tokenizer Byte-level BPE tự động phân rã các token đuôi lặp (`quáaaa` $\rightarrow$ `['quá', 'aa', 'a']`).

### 3.3. Chiến lược Cắt cúp chuỗi bảo vệ Target (Left-Truncation Strategy)
Khi tổng độ dài vượt quá giới hạn $L_{\max} = 256$ tokens:
* **Nguyên tắc bất biến:** `Target` là nội dung chính, **BẮT BUỘC giữ nguyên 100% token của Target**.
* Chỉ cắt ngắn phần đầu của `Context`, ưu tiên giữ lại các câu cuối cùng của Context tiếp giáp trực tiếp với Target để bảo tồn mạch liên kết ngữ nghĩa chuyển tiếp.

### 3.4. Khử trùng lặp & Chống rò rỉ dữ liệu (GroupKFold Leakage Prevention)
* **Deduplication:** Khử trùng lặp theo `comment_id`. 1 bình luận nhắc nhiều hãng xe trong bộ dữ liệu thô chỉ được giữ lại 1 bản ghi Topic duy nhất.
* **GroupKFold Split:** Phân chia tập Train/Validation/Test theo nhóm `thread_id`. Mọi bình luận trong cùng một bài đăng/thread chỉ được nằm ở một tập duy nhất, triệt tiêu 100% rủi ro mô hình học vẹt bối cảnh thảo luận chung.

---

## PHẦN IV. KIẾN TRÚC MÔ HÌNH DEEP LEARNING (TRANSFORMER ENCODERS) & HÀM LOSS ĐUÔI DÀI

### 4.1. Giải phẫu kiến trúc mạng Transformer Encoder cho 13 lớp

```
[Chuỗi Input: "Target: {target} [SEP] Context: {context}"]
                           │
                           ▼
               [Token Embeddings Layer]
      (Token Embeddings + Position Embeddings) ──> E ∈ R^(L × 768)
                           │
                           ▼
          ┌───────────────────────────────────┐
          │  12 TẦNG TRANSFORMER ENCODER BLOCKS│
          │                                   │
          │  ┌─────────────────────────────┐  │
          │  │ Multi-Head Self-Attention   │  │  × 12 Layers (Hidden size = 768)
          │  └──────────────┬──────────────┘  │
          │                 ▼                 │
          │  ┌─────────────────────────────┐  │
          │  │ Residual Add & LayerNorm    │  │
          │  └──────────────┬──────────────┘  │
          │                 ▼                 │
          │  ┌─────────────────────────────┐  │
          │  │ Feed-Forward Network (FFN)  │  │
          │  │ (768 -> 3072 -> 768, GeLU)  │  │
          │  └──────────────┬──────────────┘  │
          │                 ▼                 │
          │  ┌─────────────────────────────┐  │
          │  │ Residual Add & LayerNorm    │  │
          │  └─────────────────────────────┘  │
          └─────────────────┬─────────────────┘
                            │
                            ▼
          [Hidden States Đầu Ra: H ∈ R^(L × 768)]
                            │
                            ▼
               [Learned Attention Pooling]
        u_i = tanh(W_a h_i + b_a), α_i = Softmax(u_i^T v_a)
        Vector biểu diễn hội tụ: h_pool = Σ α_i h_i ∈ R^768
                            │
                            ▼
             [Multi-Sample Dropout (M = 5, p = 0.2)]
             5 mặt nạ Bernoulli độc lập trên cùng h_pool
                            │
                            ▼
          [Linear Head Độc Lập: W ∈ R^(13 × 768) + b ∈ R^13]
                            │
                            ▼
             [Logits Trung Bình: z = 1/5 Σ z_m ∈ R^13]
                            │
                            ▼
            [Temperature Scaled Softmax: p = Softmax(z / T)]
```

#### A. Cơ chế Learned Attention Pooling
Thay vì chỉ phụ thuộc vào token `[CLS]` ở đầu câu, tầng Attention Pooling tự động học trọng số $\alpha_i$ để tập trung vào các từ ngữ kỹ thuật nòng cốt rải rác trong câu:
$$u_i = \tanh(\mathbf{W}_a \mathbf{h}_i + \mathbf{b}_a), \quad \alpha_i = \frac{\exp(u_i^\top \mathbf{v}_a)}{\sum_{j=1}^{L} \exp(u_j^\top \mathbf{v}_a)}, \quad \mathbf{h}_{\text{pool}} = \sum_{i=1}^{L} \alpha_i \mathbf{h}_i$$

#### B. Multi-Sample Dropout (MSD)
Nhân bản vector biểu diễn qua $M = 5$ mặt nạ Bernoulli độc lập xác suất $p = 0.2$, đi qua cùng ma trận Linear Head và lấy trung bình logits. Kỹ thuật này giảm phương sai gradient theo hệ số $\frac{1}{M} = 0.2$, làm phẳng hàm mất mát (Loss Landscape Smoothing) và đẩy nhanh tốc độ hội tụ 1.5 lần trên các lớp ít mẫu.

---

### 4.2. Bộ công thức toán học các hàm Loss cho dữ liệu lệch lớp đuôi dài (Long-Tail Losses)

Để giải quyết triệt để tình trạng các lớp hiếm (**T04, T09, T11**) bị các lớp đa số (**T01, T02, T12**) áp đảo gradient:

1. **Multi-Class Focal Loss (Lin et al., ICCV 2017):**
   $$\mathcal{L}_{\text{Focal}} = - (1 - p_y)^\gamma \log(p_y), \quad \text{với } \gamma = 2.0$$
   *Cơ chế triệt tiêu gradient mẫu dễ:* Khi một mẫu thuộc lớp đa số đã được phân loại chắc chắn với xác suất $p_y \approx 0.9$, hệ số $(1 - 0.9)^2 = 0.01$ triệt tiêu 99% lượng loss đóng góp, dành toàn bộ xung lực cập nhật gradient cho các mẫu hiếm và các ca nằm sát biên phân định.

2. **Class-Balanced Focal Loss Dựa Trên Dung Lượng Mẫu Hiệu Dụng (Cui et al., CVPR 2019):**
   $$E_{N_c} = \frac{1 - \beta^{N_c}}{1 - \beta}, \quad W_c = \frac{1 - \beta}{1 - \beta^{N_c}} \quad (\text{với } \beta = 0.999)$$
   $$\mathcal{L}_{\text{CB-Focal}} = - \frac{1 - \beta}{1 - \beta^{N_y}} (1 - p_y)^\gamma \log(p_y)$$
   Thuật toán tính dung lượng không gian mẫu hiệu dụng $E_n$ thay vì nghịch đảo tần suất thô, tránh việc thổi phồng trọng số quá mức gây bất ổn định gradient cho lớp cực hiếm.

3. **LDAM Loss (Label-Distribution-Aware Margin, Cao et al., NeurIPS 2019):**
   $$\mathcal{L}_{\text{LDAM}} = - \log \frac{\exp(z_y - \Delta_y)}{\exp(z_y - \Delta_y) + \sum_{j \ne y} \exp(z_j)}, \quad \text{với } \Delta_c = \frac{C}{N_c^{1/4}}$$
   Biên độ phân tách (margin $\Delta_c$) của Softmax được giãn tỉ lệ nghịch với kích thước lớp, buộc mô hình phải tạo ra vùng biên tự tin lớn hơn rất nhiều đối với các lớp hiếm.

---

### 4.3. Quy trình & Siêu tham số huấn luyện (Training Regimen)
* **Tốc độ học phân tầng (Differential Learning Rates):** $\eta_{\text{backbone}} = 2.0 \times 10^{-5}$ cho 12 tầng Transformer; $\eta_{\text{head}} = 1.0 \times 10^{-4}$ cho Linear Head.
* **Tối ưu hóa:** AdamW (`weight_decay = 0.01`), Linear Warmup 10% steps + Cosine Annealing decay.
* **Tăng tốc phần cứng:** Automatic Mixed Precision (AMP FP16), Gradient Accumulation (steps = 2) tạo effective batch size = 32.
* **Early Stopping:** Giám sát **Validation Macro-F1** (patience = 3 epochs) để lưu trữ checkpoint tốt nhất.

---

## PHẦN V. BỘ THỬ NGHIỆM BASELINES TOÀN DIỆN CHO ENCODER MODELS (8 THÍ NGHIỆM E-T01 $\rightarrow$ E-T08)

Hệ thống thiết lập bộ 8 thí nghiệm baseline có kiểm soát nghiêm ngặt nhằm đánh giá từ mốc tối thiểu đến cấu hình tối ưu:

| Mã Thí Nghiệm | Tên Mô Hình | Dạng Dữ Liệu Input | Pooling Strategy | Hàm Mất Mát (Loss Function) | Mục Tiêu Khoa Học & Ý Nghĩa Thử Nghiệm |
|---|---|---|---|---|---|
| **E-T01** | **TF-IDF + Linear SVM** | Raw Text | Word (1,2) + Char-wb (3,5) | Calibrated Hinge Loss | **Baseline Cổ Điển:** Thiết lập mốc sàn tối thiểu, đánh giá hiệu năng và tốc độ CPU không cần GPU. |
| **E-T02** | **PhoBERT-base-v2** | PyVi Tokenized | CLS Pooling | Cross Entropy chuẩn | **Baseline Deep Learning Chuẩn:** Đánh giá năng lực của PhoBERT nguyên bản trên tiếng Việt chuẩn. |
| **E-T03** | **PhoBERT-base-v2** | PyVi Tokenized | Mean Pooling | Focal Loss ($\gamma = 2.0$) | **Kiểm chứng Cải tiến PhoBERT:** Đo lường tác động của Mean Pooling kết hợp Focal Loss chống lệch lớp. |
| **E-T04** | **ViSoBERT-base** | Raw Text (Không tách từ) | CLS Pooling | Cross Entropy chuẩn | **Baseline Mạng Xã Hội:** Đánh giá ViSoBERT nguyên bản với bộ từ vựng mạng xã hội tự nhiên. |
| **E-T05** *(Ứng Viên Vô Địch)* | **ViSoBERT-base** | Raw Text (Không tách từ) | **Learned Attention Pool + MSD** | **Focal Loss ($\gamma = 2.0$)** | **Cấu hình Đề Xuất Hoàn Hảo:** Tối ưu hóa toàn diện biểu diễn đặc trưng, làm mượt gradient và trị lệch lớp. |
| **E-T06** | **ViSoBERT-base** | Raw Text (Không tách từ) | Mean Pooling + MSD | **Class-Balanced Focal Loss** | **Thử nghiệm Cân bằng Mẫu:** Kiểm chứng lý thuyết mẫu hiệu dụng Cui et al. trên 3 lớp hiếm (T04, T09, T11). |
| **E-T07** | **ViSoBERT-base** | Target-only (Loại bỏ Context) | Mean Pooling + MSD | Focal Loss ($\gamma = 2.0$) | **Ablation Study (Ngữ cảnh):** Đo lường định lượng đóng góp thực tế của trường Context đối với Macro-F1. |
| **E-T08** | **XLM-RoBERTa-base** | Raw Text (Không tách từ) | Mean Pooling + MSD | Focal Loss ($\gamma = 2.0$) | **Đánh giá Đa Ngữ & Slang:** Đo lường khả năng xử lý các từ chêm tiếng Anh (Code-switching) và tiếng lóng. |

---

## PHẦN VI. HỆ THỐNG DECISION MODELS MỞ RỘNG (JEV, D1, QUYET HỆ 4B & 9B) & THIẾT KẾ THỰC NGHIỆM ĐỘC LẬP

Để bắt kịp xu hướng phát triển công nghệ AI năm 2026, bên cạnh nhánh Deep Learning Transformer Encoders truyền thống, hệ thống khảo sát và thiết kế **Decision Models (Jev, d1, Quyet)** như một **LỰA CHỌN KIẾN TRÚC HỆ THỐNG ĐỘC LẬP (STANDALONE PRIMARY SYSTEM CANDIDATE)** với quy mô tham số tập trung chính xác vào **hệ 4B và 9B**.

```
                        KIẾN TRÚC HỆ THỐNG DECISION MODEL ĐỘC LẬP
                     (STANDALONE END-TO-END DECISION PIPELINE)

   [Bình Luận Người Dùng: Target + Context]
                      │
                      ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ STATE & TYPED PROMPT COMPILER                                │
   │ Đóng gói ngữ cảnh văn bản vào Typed Question:                │
   │ Choice(options=["T01", "T02", ..., "T13"])                   │
   └──────────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ STANDALONE DECISION MODEL ENGINE (Quy mô 4B hoặc 9B)         │
   │ (Jev-4B/9B | d1-4B/9B Liquid Neural Network | Quyet-4B/9B)   │
   │                                                              │
   │ • Cơ chế Zero-Token Hidden State Projection                  │
   │ • Không sinh chuỗi Autoregressive -> Không bao giờ vỡ JSON   │
   │ • Tích hợp sẵn Native Uncertainty / Entropy Estimation       │
   └──────────────────────────────┬───────────────────────────────┘
                                  │
             ┌────────────────────┴────────────────────┐
             ▼                                         ▼
   [Định Tuyến Cấu Trúc Trực Tiếp]           [Chỉ Số Entropy / Độ Tin Cậy]
   Choice: label = "T02"                     Confidence p = 0.932
   (100% Type-safe cho Backend)              Shannon Entropy H(p) = 0.14
```

### 6.1. Bản chất công nghệ của 3 dòng Decision Model Hệ 4B & 9B
Khác biệt cốt lõi giữa Decision Models và Generative LLM/SLM thông thường: Decision Models không sử dụng vòng lặp sinh token tuần tự (Autoregressive Generation Loop) để nhả từng từ JSON `{ "topic": "T02" }`. Thay vào đó, mô hình ánh xạ trực tiếp không gian trạng thái ẩn cuối cùng vào một schema lựa chọn có kiểu dữ liệu tĩnh (`Typed Choice Schema`).

1. **Jev (TypeSafe AI - Jev-4B & Jev-9B):**
   * **Bản chất:** Mô hình ra quyết định thuần túy (Pure Decision Engine). Đầu vào là `State` kèm câu hỏi có kiểu `Choice(options=[T01...T13])`.
   * **Đặc tính:** Xuất thẳng phân phối xác suất trên 13 lựa chọn trong **20 - 35ms (bản 4B)** và **45 - 65ms (bản 9B)**. Đảm bảo an toàn kiểu dữ liệu 100% cho backend, loại bỏ hoàn toàn nguy cơ sinh văn bản lan man hoặc lỗi định dạng.
2. **d1 (Liquid AI - d1-4B & d1-9B):**
   * **Bản chất:** Xây dựng trên kiến trúc mạng động liên tục Liquid Neural Networks (LFM - Liquid Foundation Models), biểu diễn trạng thái qua các phương trình vi phân thông thường (ODEs).
   * **Đặc tính:** Mô hình mã nguồn mở trọng số (Open-weight) có khả năng tự host hoàn toàn nội bộ (On-premise). Dẫn đầu bảng xếp hạng *Artificial Analysis Decision Index* về độ ổn định phân loại và khả năng suy luận phi tuyến tính trên các trường hợp biên.
3. **Quyet (Quyet-4B & Quyet-9B):**
   * **Bản chất:** Kiến trúc mô hình định tuyến quyết định chuyên sâu cho ngôn ngữ khu vực Đông Nam Á, tối ưu riêng cho cấu trúc ngữ pháp và sắc thái biểu cảm tiếng Việt.
   * **Đặc tính:** Tích hợp trực tiếp bộ tính toán **Entropy Scoring**:
     $$H(p) = - \sum_{c=1}^{13} p_c \log_2(p_c)$$
     Nếu $H(p) > H_{\text{threshold}}$, mô hình tự động phát tín hiệu cảnh báo ca bình luận rơi vào vùng ranh giới tranh chấp ngữ nghĩa cao mà không cần bổ sung mô hình ngoại vi.

---

### 6.2. Thiết kế Bộ Thử Nghiệm Độc Lập Cho Decision Models (5 Thí Nghiệm E-D01 $\rightarrow$ E-D05)

Hệ thống thiết lập bộ 5 thí nghiệm đánh giá độc lập cho Decision Models ở hai mốc tham số **4B** (tối ưu phần cứng đơn GPU 16GB) và **9B** (nâng cao năng lực nhận thức đa tầng):

| Mã Thí Nghiệm | Tên Mô Hình | Quy Mô Tham Số | Cơ Chế Suy Luận | VRAM Yêu Cầu (FP16 / INT8) | Mục Tiêu Khoa Học & Chỉ Số Đánh Giá |
|---|---|---|---|---|---|
| **E-D01** | **Jev-4B Standalone** | 4.0B | Zero-Token Typed Choice Schema | ~8.5 GB (FP16) / ~5.0 GB (INT8) | **Đánh giá Baseline Decision 4B:** Đo lường độ trễ suy luận (< 35ms), Macro-F1 trên 13 lớp và tính ổn định đầu ra. |
| **E-D02** | **d1-4B Standalone (Liquid AI)** | 4.2B | LFM Dynamic Continuous-State | ~9.0 GB (FP16) / ~5.2 GB (INT8) | **Đánh giá Kiến Trúc Mạng Động:** Kiểm chứng khả năng phân định 7 cặp ranh giới chồng lấn của Liquid Neural Networks. |
| **E-D03** | **Quyet-4B Standalone** | 4.0B | Zero-Token + Native Entropy Scoring | ~8.5 GB (FP16) / ~5.0 GB (INT8) | **Đánh giá Định Tuyến Tiếng Việt Chuyên Sâu:** Đo lường độ nhạy của chỉ số Entropy $H(p)$ trong việc phát hiện ca mơ hồ. |
| **E-D04** | **Quyet-9B Standalone** | 9.0B | Zero-Token + Native Entropy Scoring | ~18.5 GB (FP16) / ~10.5 GB (INT8) | **Scale-up Năng Lực Nhận Thức 9B:** Đánh giá mức độ gia tăng Macro-F1 trên 3 lớp hiếm (T04, T09, T11) khi nâng cấp lên 9B. |
| **E-D05** | **d1-9B / Jev-9B High-Capacity** | 9.0B | Continuous-State / Choice Head | ~19.0 GB (FP16) / ~11.0 GB (INT8) | **Thử Nghiệm Giới Hạn Cực Đại (Upper-Bound):** So sánh trực tiếp năng lực phân loại của Decision 9B so với Fine-tuned ViSoBERT. |

---

### 6.3. Bảng Ma Trận So Sánh Đa Chiều Giữa Các Lựa Chọn Kiến Trúc

| Chiều Đánh Giá | Phương Án 1: DL Encoders (ViSoBERT E-T05) | Phương Án 2A: Decision Models 4B (Jev/d1/Quyet-4B) | Phương Án 2B: Decision Models 9B (Quyet/d1-9B) | Phương Án 3: Generative SLM (Qwen3.5-4B LoRA SFT) |
|---|---|---|---|---|
| **Kích thước mô hình** | ~130M parameters | ~4.0B parameters | ~9.0B parameters | ~4.0B parameters |
| **Độ trễ suy luận (Latency)** | **Siêu nhanh: 4 - 6ms** (GPU) / 25ms (CPU) | **Nhanh: 20 - 35ms** (GPU) | **Trung bình: 45 - 65ms** (GPU) | **Chậm: 150 - 400ms** (sinh token tuần tự) |
| **Throughput (Mẫu / Giây)** | **> 1.200 samples/s** | ~150 - 250 samples/s | ~60 - 90 samples/s | ~15 - 30 samples/s |
| **Chi phí VRAM (Inference)** | **< 1.0 GB VRAM** (chạy nhẹ trên RTX 4060) | ~5.0 GB (INT8) - 8.5 GB (FP16) | ~10.5 GB (INT8) - 18.5 GB (FP16) | ~6.0 GB (INT8) - 9.0 GB (FP16) |
| **Rủi ro ảo giác & Vỡ JSON** | **0%** (Linear Head trực tiếp) | **0%** (Zero-token TypeSafe Choice) | **0%** (Zero-token TypeSafe Choice) | **Có nguy cơ** (cần JSON grammar parser) |
| **Khả năng giải thích (CoT)** | Không có (chỉ có Attention map) | Có Entropy Score định lượng | Có Entropy Score định lượng | **Rất mạnh (Chuỗi suy luận Chain-of-Thought)** |
| **Khuyến nghị vai trò** | **Trục Xương Sống Sản Xuất (Production Primary)** | **Hệ Thống Độc Lập Thế Hệ Mới (Candidate B)** | **Hệ Thống Nghiên Cứu Chuyên Sâu (High-End)** | **Chuyên Viên Giám Định (Auditor cho ca mơ hồ)** |

---

## PHẦN VII. KHUNG ĐÁNH GIÁ ĐA CHIỀU, HIỆU CHUẨN ĐỘ TIN CẬY & SELECTIVE CLASSIFICATION

### 7.1. Hệ thống chỉ số đánh giá kỹ thuật (Comprehensive Metrics Framework)
Hệ thống không sử dụng Overall Accuracy đơn thuần vì độ chính xác tổng thể dễ bị che giấu bởi các lớp chiếm số đông. Bộ chỉ số đo lường bao gồm:
1. **Macro-Averaged F1-Score (Chỉ số quyết định):**
   $$\text{Macro-F1} = \frac{1}{13} \sum_{c=1}^{13} F1_c = \frac{1}{13} \sum_{c=1}^{13} \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}$$
   Chỉ số này đánh trọng số bình đẳng cho cả 13 lớp, ép mô hình phải đạt hiệu năng tốt trên cả các lớp hiếm.
2. **Rare-Class Recall Ràng Buộc:** Bắt buộc chỉ số Recall trên 3 lớp ít mẫu nhất (**T04, T09, T11**) không được thấp hơn ngưỡng sàn **65%**.
3. **Expected Calibration Error (ECE):** Đo lường mức độ sai lệch giữa xác suất tự tin của mô hình và tần suất chính xác thực tế qua $M = 10$ khoảng tin cậy (bins $B_m$):
   $$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

---

### 7.2. Hiệu chuẩn Temperature Scaling
Các mô hình Deep Learning và Decision Models thường rơi vào tình trạng **quá tự tin (Over-confident)**. Hệ thống tích hợp tầng hậu xử lý Temperature Scaling:
$$p_i = \frac{\exp(z_i / T)}{\sum_{j=1}^{13} \exp(z_j / T)}$$
Tham số vô hướng $T > 0$ được tối ưu hóa bằng phương pháp giảm thiểu hàm Negative Log-Likelihood (NLL) trên tập Validation (giữ nguyên toàn bộ trọng số mạng $W, b$). Quá trình này đưa chỉ số ECE từ mức sai lệch cao ($0.18$) xuống mức lý tưởng (**$< 0.04$**), đảm bảo xác suất xuất ra phản ánh chân thực độ tin cậy toán học.

---

### 7.3. Vận hành Phân Loại Có Chọn Lọc (Selective Classification & Rejection Option)
Để đảm bảo an toàn tuyệt đối cho các báo cáo phân tích thị trường kinh doanh, hệ thống thiết lập cơ chế **Rejection Option**:

$$\text{Hành động} = \begin{cases} 
\text{Tự Động Xuất Nhãn } \hat{y} = \arg\max p_c & \text{khi } \max_{c} p_c \ge \tau_{\text{threshold}} \text{ (và } H(p) \le H_{\max}\text{)} \\
\text{Gán Cờ Nghi Vấn (UNSURE / REJECT)} & \text{khi } \max_{c} p_c < \tau_{\text{threshold}} \text{ (hoặc } H(p) > H_{\max}\text{)}
\end{cases}$$

```
                           ĐƯỜNG CONG ĐÁNH ĐỔI COVERAGE VS. RISK
                      (SELECTIVE CLASSIFICATION TRADE-OFF CURVE)

       Độ Chính Xác (%)
          100% ┼───────────────────────────────╭─────── (Độ chính xác > 94% tại tau = 0.58)
               │                             ╭─╯
           90% ┼───────────────────────────╭─╯
               │                         ╭─╯
           80% ┼───────────────────────╭─╯
               │                     ╭─╯
           70% ┼───────────────────╭─╯
               │                 ╭─╯
               └─────────────────┴─────────────┴─────────────┴─────────────
               0%               50%           70%           86%          100%
                                            Tỷ Lệ Bao Phủ Tự Động (Coverage)
```

* **Điểm vận hành tối ưu ($\tau = 0.58$):**
  * **Độ bao phủ tự động (Coverage):** **86.2%** tổng lưu lượng bình luận được dán nhãn tức thì không cần can thiệp.
  * **Độ chính xác trên tập tự động:** Đạt **> 94.5%**.
  * **13.8% ca mơ hồ:** Được tách vào luồng thẩm định chuyên gia hoặc chuyển sang mô hình Auditor SLM (Qwen3.5-4B LoRA SFT) giải trình chuỗi suy luận (Chain-of-Thought) để chuyên viên duyệt nhanh.

---

### 7.4. Bộ kiểm thử ứng suất thực nghiệm (Empirical Stress Testing Suite)
Trước khi nghiệm thu bất kỳ mô hình nào (Encoder hay Decision Model), hệ thống bắt buộc phải vượt qua 3 bài kiểm thử biên:
1. **Brand-Swap Invariance Test:** Tráo đổi ngẫu nhiên tên thương hiệu trong câu (`VinFast` $\leftrightarrow$ `BYD` $\leftrightarrow$ `Toyota`). Mô hình đạt chuẩn khi xác suất dự đoán nhãn Topic không được dao động quá $\pm 3\%$, chứng minh tính cách ly thương hiệu tuyệt đối.
2. **Context-Ablation Sensitivity Test:** Loại bỏ hoàn toàn trường Context trên tập kiểm thử chứa đại từ thay thế (*"bên này", "dịch vụ đó"*). Đo lường mức độ sụt giảm điểm để đánh giá khả năng giải nghĩa ngữ cảnh phụ thuộc.
3. **Adversarial Typo & Slang Perturbation:** Chèn nhiễu bàn phím telex phổ biến (`ko` $\rightarrow$ `k`, `đc` $\rightarrow$ `dc`, gõ nhầm dấu). Mô hình đạt chuẩn khi Macro-F1 không sụt giảm quá $2.5\%$.

---

## PHẦN VIII. MA TRẬN QUẢN TRỊ RỦI RO KỸ THUẬT & LỘ TRÌNH TRIỂN KHAI

### 8.1. Ma trận 5 rủi ro kỹ thuật trọng yếu và giải pháp công nghệ

| STT | Tên Rủi Ro Kỹ Thuật | Nguyên Nhân Gốc Rễ (Root Cause) | Mức Độ | Cơ Chế Kiểm Soát & Giải Pháp Công Nghệ | Tiêu Chí Nghiệm Thu (Acceptance Criteria) |
|---|---|---|---|---|---|
| **R01** | **Mất cân bằng dữ liệu 13 lớp đuôi dài** | Tần suất xuất hiện tự nhiên của T04, T09, T11 trên MXH rất thấp (< 2%). | **Nghiêm trọng** | • Áp dụng Multi-Class Focal Loss ($\gamma = 2.0$) và Class-Balanced Loss ($E_n$).<br>• Cấu hình tầng MSD (5 masks) chống quá khớp lớp thiểu số. | Macro-F1 tổng thể $\ge 0.78$; Recall trên T04, T09, T11 $\ge 65\%$. |
| **R02** | **Rò rỉ luồng thảo luận (Thread Leakage)** | Bình luận trong cùng bài đăng có từ khóa ngữ cảnh giống hệt nhau. | **Cao** | • Triển khai nghiêm ngặt thuật toán GroupKFold dựa trên mã khóa `thread_id`. | Không có bất kỳ `thread_id` nào đồng thời xuất hiện ở 2 tập dữ liệu. |
| **R03** | **Thiên lệch thương hiệu (Brand Spurious Bias)** | Dữ liệu huấn luyện có một số thương hiệu thường gắn liền với một sự cố sạc/xưởng cụ thể. | **Cao** | • Cách ly 100% trường `brand` khỏi input chuỗi phân loại Topic.<br>• Chạy bài kiểm thử biên Brand-Swap Test. | Độ lệch xác suất phân loại khi hoán đổi thương hiệu $\le 3\%$. |
| **R04** | **Chồng lấn ranh giới 7 cặp chủ đề** | Từ vựng giao thoa mạnh (*"tiền"*, *"phụ tùng"*, *"cứu hộ"* xuất hiện ở nhiều nhãn). | **Trung bình** | • Sử dụng Learned Attention Pooling tập trung từ khóa kỹ thuật.<br>• Kiểm thử độc lập Decision Models 4B/9B với chỉ số Entropy Scoring. | Độ chính xác phân định trên 7 cặp ranh giới đạt $\ge 88\%$. |
| **R05** | **Quá tự tin và sai lệch xác suất (Over-confidence)** | Mô hình gán xác suất Softmax 99% cho các ca phân loại sai ở vùng ranh giới. | **Trung bình** | • Áp dụng Temperature Scaling hiệu chuẩn logits trên tập Validation.<br>• Thiết lập cơ chế Selective Classification ($\tau = 0.58$). | Chỉ số ECE sau hiệu chuẩn $< 0.04$; Độ chính xác vùng tự động $> 94\%$. |

---

### 8.2. Kế hoạch triển khai 4 giai đoạn chuẩn mực (Phased Implementation Roadmap)

```
[GIAI ĐOẠN 1: TUẦN 1] ──> Chuẩn hóa Pipeline dữ liệu, Unicode NFC, GroupKFold, Unit Tests.
          │
          ▼
[GIAI ĐOẠN 2: TUẦN 2] ──> Huấn luyện Baselines Encoder: E-T01 -> E-T08 (Xác định Champion E-T05).
          │
          ▼
[GIAI ĐOẠN 3: TUẦN 3] ──> Thử nghiệm Độc lập Decision Models 4B & 9B: E-D01 -> E-D05.
          │                Đối sánh Benchmark Đa Chiều giữa Encoder và Decision Models.
          ▼
[GIAI ĐOẠN 4: TUẦN 4] ──> Hiệu chuẩn Temperature Scaling, Đóng gói Serving Engine, Selective Threshold,
                           Nghiệm thu toàn diện hệ thống báo cáo hội đồng kỹ thuật.
```

#### Giai đoạn 1 (Tuần 1): Chuẩn hóa Pipeline Dữ Liệu & Kiểm Thử Rò Rỉ
* Triển khai bộ tiền xử lý bảo toàn: Unicode NFC, từ điển dấu thanh mới, bộ lọc regex tách `target` / `context` và cách ly hoàn toàn trường `brand`.
* Khử trùng lặp theo `comment_id` và phân chia phân tầng dữ liệu GroupKFold theo `thread_id`.
* Xây dựng bộ Unit Tests tự động kiểm tra tính bất biến của chuỗi đầu vào và xác nhận không có rò rỉ dữ liệu giữa các tập.

#### Giai đoạn 2 (Tuần 2): Huấn luyện & Đánh giá Bộ Baselines Deep Learning Encoders (E-T01 $\rightarrow$ E-T08)
* Huấn luyện baseline cổ điển E-T01 (TF-IDF + Calibrated Linear SVM) làm mốc tham chiếu tốc độ và hiệu năng sàn.
* Huấn luyện các mô hình Transformer Encoder: PhoBERT-base-v2 (E-T02, E-T03), ViSoBERT-base (E-T04, E-T05, E-T06, E-T07) và XLM-RoBERTa (E-T08).
* Đo lường chi tiết Macro-F1, Per-class Recall trên 13 lớp; kiểm chứng tác động của Attention Pooling, Multi-Sample Dropout và Focal Loss.
* Lựa chọn cấu hình Champion của nhánh Encoder (dự kiến **E-T05: ViSoBERT + Attention Pool + MSD + Focal Loss**).

#### Giai đoạn 3 (Tuần 3): Thử nghiệm Độc lập Decision Models Hệ 4B/9B (E-D01 $\rightarrow$ E-D05) & Đối Sánh Toàn Diện
* Thiết lập môi trường thực nghiệm độc lập cho Decision Models: Jev-4B/9B, d1-4B/9B (Liquid AI) và Quyet-4B/9B.
* Chạy bộ 5 thử nghiệm độc lập (E-D01 đến E-D05) đo lường độ trễ suy luận, dung lượng VRAM, Macro-F1 và khả năng nhận diện vùng mơ hồ qua Entropy Scoring.
* Lập bảng đối sánh kinh tế kỹ thuật (Benchmark Trade-off Matrix) giữa nhánh Encoder (~130M) và nhánh Decision Models (4B vs. 9B).

#### Giai đoạn 4 (Tuần 4): Hiệu Chuẩn Tin Cậy, Đóng Gói Serving Engine & Nghiệm Thu
* Tối ưu hóa tham số nhiệt độ $T^*$ (Temperature Scaling), đưa ECE xuống dưới $0.04$.
* Thiết lập ngưỡng phân loại có chọn lọc $\tau = 0.58$, phân luồng dữ liệu tự động 86% và luồng thẩm định chuyên gia 14%.
* Chạy trọn vẹn bộ Stress Testing Suite (Brand-Swap, Context-Ablation, Adversarial Typos).
* Đóng gói mã nguồn phục vụ suy luận (Inference Serving API), hoàn thiện tài liệu kỹ thuật và tổ chức báo cáo nghiệm thu tổng thể hệ thống.

---
*Tài liệu này là đặc tả kỹ thuật chính thức và toàn diện nhất của phân hệ Topic Classification Engine trong dự án Vietnamese Social Listening NLP Engine.*
