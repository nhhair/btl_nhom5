# -*- coding: utf-8 -*-
"""BÀI TẬP LỚN FIT4013 - ĐỀ TÀI 5: NHẬN DẠNG CHỮ SỐ VIẾT TAY.

Chạy:  python main.py

Quy trình (theo yêu cầu của đề bài):
  1. Data Source  : sklearn.datasets.load_digits (1797 ảnh 8x8, 10 lớp)
  2. EDA          : phân phối lớp, ảnh mẫu, phân bố pixel
  3. Preprocessing: chuẩn hóa pixel [0,1]; StandardScaler fit trên TRAIN
  4. Feature      : vector pixel (baseline) + HOG (ảnh 32x32)
  5. Model        : K-NN (đổi k, metric) & Decision Tree (đổi criterion, max_depth)
  6. Evaluation   : Accuracy, Precision/Recall/F1-macro, Confusion Matrix
  7. Error Analysis: cặp chữ số hay nhầm lẫn + ảnh dự đoán sai

Toàn bộ kết quả (biểu đồ, bảng CSV, mô hình) được lưu vào figures/, results/, models/.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # chạy không cần hiển thị cửa sổ đồ thị

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Lấy thư mục gốc của project (bất kể đang chạy từ đâu) rồi thêm vào sys.path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data import (eda_summary, load_data, plot_class_distribution,
                      plot_pixel_intensity, plot_sample_digits)
from src.evaluate import (build_experiments, error_analysis,
                          export_summary, plot_confusion_matrices,
                          robustness_analysis, run_experiments,
                          save_best_model, save_gui_model)
from src.features import build_hog_features, normalize_pixels

FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"
MOD_DIR = ROOT / "models"
for d in (FIG_DIR, RES_DIR, MOD_DIR):
    d.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2


def main():
    # =====================================================================
    # 1. DATA SOURCE
    # =====================================================================
    print("\nBÀI TẬP LỚN FIT4013 - ĐỀ TÀI 5: NHẬN DẠNG CHỮ SỐ VIẾT TAY\n")
    print("Dataset: scikit-learn load_digits (không cần tải, có sẵn trong thư viện).")
    print(f"random_state = {RANDOM_STATE}, test_size = {TEST_SIZE}\n")

    X, images, y, digits = load_data()

    # =====================================================================
    # 2. EDA
    # =====================================================================
    eda_summary(X, images, y, digits)
    plot_class_distribution(y, FIG_DIR / "eda_class_distribution.png")
    plot_sample_digits(images, y, FIG_DIR / "eda_sample_digits.png")
    plot_pixel_intensity(X, FIG_DIR / "eda_pixel_intensity.png")

    # =====================================================================
    # 3. PREPROCESSING + FEATURE
    # =====================================================================
    X_norm = normalize_pixels(X)            # pixel ve [0,1]
    hog = build_hog_features(images)        # dac trung HOG (324 chieu)

    # =====================================================================
    # 4. CHIA TRAIN/TEST (stratify giu ti le lop, ghi ro random_state)
    # =====================================================================
    X_train_px, X_test_px, y_train, y_test, idx_train, idx_test = train_test_split(
        X_norm, y, np.arange(len(y)), test_size=TEST_SIZE,
        stratify=y, random_state=RANDOM_STATE)

    X_train_hog, X_test_hog = hog[idx_train], hog[idx_test]
    images_test = images[idx_test]  # ảnh ứng với tập test, phục vụ phân tích lỗi

    print("\n" + "=" * 74)
    print(f"BUOC 2-4: TIEN XU LY & CHIA DU LIEU (train/test = {1 - TEST_SIZE}/{TEST_SIZE})")
    print("=" * 74)
    print(f"- Train: {X_train_px.shape[0]} mau | Test: {X_test_px.shape[0]} mau | stratify=y")
    print(f"- Dac trung pixel: {X_train_px.shape[1]} chieu (chia /16 -> [0,1])")
    print(f"- Dac trung HOG  : {X_train_hog.shape[1]} chieu (anh 32x32, 9 orientations, block 2x2)")
    print("- StandardScaler (neu dung) chi fit tren TRAIN de tranh data leakage")

    # =====================================================================
    # 5. HUAN LUYEN & 6. DANH GIA (nhiều phương án)
    # =====================================================================
    features = {
        "pixels": (X_train_px, X_test_px),
        "hog": (X_train_hog, X_test_hog),
    }
    exps = build_experiments()
    df, fitted = run_experiments(features, y_train, y_test, exps)

    print("\n" + "=" * 74)
    print("BẢNG SO SÁNH KẾT QUẢ (sắp theo F1-macro giảm dần)")
    print("=" * 74)
    view = df.sort_values("f1_macro", ascending=False).reset_index(drop=True)
    print(view.to_string(index=False))

    export_summary(df, RES_DIR / "comparison_results.csv")

    # Chọn phương án tốt nhất của từng nhóm (KNN-pixels, DT-pixels, KNN-hog, DT-hog)
    best_ids = []
    for prefix in ("KNN-pixels", "DT-pixels", "KNN-hog", "DT-hog"):
        sub = df[df["id"].str.startswith(prefix)]
        if not sub.empty:
            bid = sub.loc[sub["f1_macro"].idxmax(), "id"]
            best_ids.append(bid)
            print(f"  Toi uu [{prefix}]: {bid}")

    plot_confusion_matrices(fitted, best_ids, FIG_DIR / "confusion_matrices.png")

    # =====================================================================
    # 7. ERROR ANALYSIS (chọn phương án tốt nhất tổng thể)
    # =====================================================================
    best_id = df.loc[df["f1_macro"].idxmax(), "id"]
    error_analysis(fitted, best_id, images_test, FIG_DIR / "error_analysis.png")

    # =====================================================================
    # 8. ROBUSTNESS: thử nhiễu Gaussian và xoay nhẹ trên cùng tập test
    # =====================================================================
    robustness_ids = ["KNN-pixels-baseline", "DT-pixels-gini-d10", "KNN-hog-k5"]
    print("\n" + "=" * 74)
    print("THỬ NGHIỆM BỔ SUNG: ĐỘ BỀN VỮNG KHI ẢNH BỊ NHIỄU / XOAY")
    print("=" * 74)
    robustness_df = robustness_analysis(
        fitted, robustness_ids, images_test, y_test, RES_DIR, FIG_DIR)
    print(robustness_df.round(4).to_string(index=False))

    # =====================================================================
    # LƯU MÔ HÌNH + BÁO CÁO PHÂN LOẠI CHI TIẾT
    # =====================================================================
    exp_saved, obj = save_best_model(fitted, df, MOD_DIR)
    save_gui_model(X_norm, y, MOD_DIR)

    print("\n" + "=" * 74)
    print(f"CHI TIẾT MÔ HÌNH TỐT NHẤT: {exp_saved}")
    print("=" * 74)
    f = fitted[exp_saved]
    y_pred = f["model"].predict(f["X_test"])
    print(classification_report(f["y_test"], y_pred, digits=4))

    # =====================================================================
    # KẾT LUẬN NGẮN (cho phần thảo luận báo cáo)
    # =====================================================================
    best_row = df.loc[df["f1_macro"].idxmax()]
    base_row = df[df["id"] == "KNN-pixels-baseline"].iloc[0]
    print("\n" + "=" * 74)
    print("KẾT LUẬN & THẢO LUẬN (tóm tắt)")
    print("=" * 74)
    print(f"- Baseline (KNN mac dinh): accuracy = {base_row['accuracy']:.4f}")
    print(f"- Toi uu ({best_row['id']}): accuracy = {best_row['accuracy']:.4f}, "
          f"F1-macro = {best_row['f1_macro']:.4f}")
    print("- Thử nghiệm nhiễu/xoay: xem results/robustness_results.csv và figures/robustness_analysis.png")
    print("- Xem figure: figures/, results: results/, models: models/")
    print("- Muon demo GUI: python gui.py")


if __name__ == "__main__":
    main()
