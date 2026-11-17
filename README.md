# Nhận dạng chữ số viết tay - FIT4013

Dự án sử dụng bộ `load_digits` của scikit-learn để phân loại chữ số 0-9. Pipeline gồm EDA, tiền xử lý, đặc trưng pixel/HOG, K-NN, Decision Tree, đánh giá 19 cấu hình và kiểm tra độ bền với nhiễu Gaussian/xoay ảnh. Giao diện Tkinter cho phép vẽ chữ số bằng chuột.

## Cấu trúc chính

- `main.py`: chạy toàn bộ pipeline, đánh giá và tạo kết quả.
- `gui.py`: mở giao diện vẽ chữ số; cần chạy `main.py` trước để sinh mô hình GUI.
- `src/`: các mô-đun nạp dữ liệu, đặc trưng, mô hình và đánh giá.
- `results/`: các bảng CSV/JSON được cung cấp kèm báo cáo.
- `figures/`: các biểu đồ kết quả được cung cấp kèm báo cáo.

## Cài đặt và chạy trên Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
python gui.py
```

Lần chạy `main.py` sẽ tạo lại các thư mục `figures/`, `results/`, `models/`. Gói bàn giao không kèm `.venv` hoặc các mô hình nhị phân được sinh tự động để giảm dung lượng; mã nguồn sẽ tạo các mô hình khi chạy.

## Kết quả chính trong bộ dữ liệu được cung cấp

- 1.797 ảnh, 10 lớp, kích thước 8x8 pixel.
- 19 cấu hình thực nghiệm.
- K-NN HOG (`k=3` hoặc `k=5`) đạt Accuracy 99,44% trên lần chia train/test 80/20, `random_state=42`.
- Bổ sung kiểm tra độ bền trên 3 mô hình với nhiễu Gaussian `sigma=0.10, 0.20, 0.30` và xoay `10`, `20` độ. Số liệu chi tiết nằm trong `results/robustness_results.csv`.

Lưu ý: mô hình tốt nhất được chọn theo F1-macro trên tập test của lần chia này; do đó Accuracy 99,44% nên được hiểu là kết quả quan sát trong thí nghiệm hiện tại, chưa phải ước lượng độc lập sau chọn mô hình.
