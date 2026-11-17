# KHAI BÁO SỬ DỤNG CÔNG CỤ AI
# (Theo yêu cầu mục 2.3 của "Hướng dẫn thực hiện bài tập lớn học phần FIT4013")

## Nhóm đã sử dụng AI như thế nào?

- **Tìm hiểu khung quy trình**: Dùng AI để tóm tắt yêu cầu đề tài và gợi ý cấu trúc
  pipeline học máy chuẩn (EDA → Preprocessing → Feature → Model → Evaluation →
  Error Analysis) cho bộ dữ liệu `load_digits`.
- **Gợi ý mã nguồn**: AI hỗ trợ viết khung code (nạp dữ liệu, trích đặc trưng HOG,
  huấn luyện K-NN / Decision Tree, vẽ Confusion Matrix, phân tích lỗi).
- **Kiểm tra lỗi**: AI hỗ trợ tìm lỗi chạy (missing thư viện, tham số API thay đổi,
  lỗi font matplotlib, ...).

## Cam kết của nhóm

- Toàn bộ mã nguồn, kết quả và phân tích đã được **các thành viên tự đọc lại,
  chạy lại và giải thích được** trước khi nộp.
- Không sao chép nguyên văn báo cáo/đồ án của người khác.
- AI chỉ đóng vai trò hỗ trợ, mọi quyết định kỹ thuật (chọn tham số, lý do so sánh,
  giải thích kết quả) đều do nhóm tự đảm bảo và trình bày khi bảo vệ.

## Chi tiết phiên bản / công cụ

| Công cụ | Mục đích | Cách kiểm chứng |
|---------|----------|-----------------|
| ChatGPT/DeepSeek | Soạn khung pipeline, gợi ý tham số | So sánh với tài liệu sklearn, chạy thử |
| scikit-learn docs | `load_digits`, KNN, DecisionTree | Đọc trực tiếp trên trang chính thức |
| scikit-image docs | HOG | Đọc trực tiếp trên trang chính thức |

## Minh bạch khi bảo vệ

Nếu được hỏi về bất kỳ đoạn code/kết quả nào, nhóm sẵn sàng trả lời trực tiếp;
phần nào do AI gợi ý đều được ghi chú trong file này để minh bạch theo rubric
"Liêm chính học thuật và sử dụng AI".
