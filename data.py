# -*- coding: utf-8 -*-
"""Nạp dữ liệu và thực hiện EDA (khám phá & phân tích dữ liệu).

Bộ dữ liệu: sklearn.datasets.load_digits
 - 1797 ảnh chữ số viết tay kích thước 8x8 pixel, ảnh xám.
 - Giá trị pixel nằm trong khoảng [0, 16] (0 = nền đen, 16 = nét chữ đậm).
 - 10 lớp tương ứng các chữ số 0..9, mỗi lớp ~180 ảnh (dữ liệu cân bằng).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits


def load_data():
    """Tải bộ dữ liệu load_digits, trả về (X, images, y, digits)."""
    digits = load_digits()
    X = digits.data.astype(np.float64)          # (1797, 64) đã duỗi phẳng
    images = digits.images.astype(np.float64)   # (1797, 8, 8) nguyên bản
    y = digits.target                           # (1797,)
    return X, images, y, digits


def eda_summary(X, images, y, digits):
    """In các thông tin cơ bản của bộ dữ liệu ra màn hình."""
    print("=" * 74)
    print("BUOC 1 - EDA: KHAM PHA VA PHAN TICH DU LIEU (load_digits)")
    print("=" * 74)

    print(f"- So mau (anh)        : {X.shape[0]}")
    print(f"- So dac trung        : {X.shape[1]}   (anh 8x8 duoc duoi phang thanh 64 pixel)")
    print(f"- Kich thuoc anh      : {images.shape[1]}x{images.shape[2]} pixel, anh xam (1 kenh mau)")
    print(f"- So lop              : {len(np.unique(y))}  (chu so 0..9)")
    print(f"- Khoang gia tri pixel: [{X.min():.1f}, {X.max():.1f}]  (0 = nen, 16 = net chu)")
    print(f"- Du lieu thieu (NaN) : {np.isnan(X).sum()} gia tri")
    uniq, cnt = np.unique(X, axis=0, return_counts=True)
    n_dup = int(np.sum(cnt > 1))
    print(f"- Anh trung lap       : {n_dup} anh")

    counts = pd.Series(y).value_counts().sort_index()
    dist = pd.DataFrame({
        "chu_so": counts.index.astype(int),
        "so_luong": counts.values,
        "ty_le_phan_tram": np.round(100 * counts.values / len(y), 2),
    })
    print("\nPhan phoi lop (du lieu can bang - moi chu so ~180 anh):")
    print(dist.to_string(index=False))

    print("\nNhan xet EDA (cho bao cao):")
    print("  - Du lieu sach: khong co gia tri thieu, khong co anh trung lap.")
    print("  - Du lieu can bang tot: khong can ky thuat xu ly mat can bang lop.")
    print("  - Anh rat nho (8x8) va da duoc cat, xoay gan dung -> day la bai toan kha 'de',")
    print("    ket qua Accuracy ky vong cao (~95%+ voi K-NN).")
    return dist


def _save(fig, out_path: Path):
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [Da luu] {out_path}")


def plot_class_distribution(y, out_path: Path):
    """Biểu đồ cột phân phối số lượng ảnh theo từng chữ số."""
    counts = pd.Series(y).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(counts.index.astype(str), counts.values, color="#5b9bd5",
                  edgecolor="black", linewidth=0.7)
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, str(v), ha="center", fontsize=9)
    ax.set_xlabel("Chu so (class)")
    ax.set_ylabel("So luong anh")
    ax.set_title("Phan phoi so luong anh theo tung chu so")
    _save(fig, out_path)


def plot_sample_digits(images, y, out_path: Path):
    """Lưới 2x10: mỗi cột là một chữ số, hiển thị 2 ảnh mẫu."""
    fig, axes = plt.subplots(2, 10, figsize=(13, 3.4))
    for cls in range(10):
        idx = np.where(y == cls)[0]
        for row in range(2):
            ax = axes[row, cls]
            ax.imshow(images[idx[row]], cmap="gray_r", vmin=0, vmax=16)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(str(cls), fontsize=11)
    fig.suptitle("Mau anh tieu bieu cua tung chu so (2 anh/lop)")
    _save(fig, out_path)


def plot_pixel_intensity(X, out_path: Path):
    """Histogram phân bố giá trị pixel toàn bộ dữ liệu (minh họa tiền xử lý)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(X.ravel(), bins=17, color="#ed7d31", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Gia tri pixel (0-16)")
    ax.set_ylabel("Tan suat")
    ax.set_title("Phan bo gia tri pixel toan bo du lieu")
    _save(fig, out_path)
