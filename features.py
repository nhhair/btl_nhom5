# -*- coding: utf-8 -*-
"""Tiền xử lý và biểu diễn đặc trưng: vector pixel (baseline) + HOG.

Tiền xử lý:
  - Chuẩn hóa pixel về khoảng [0,1] (chia cho 16).

Biểu diễn đặc trưng:
  1. Pixel  : duỗi phẳng ảnh 8x8 -> vector 64 chiều (baseline theo đề bài).
  2. HOG    : phóng to ảnh lên 32x32 (nội suy) rồi tính Histogram of Oriented
              Gradients - mô tả phân bố hướng cạnh của nét chữ.
"""
import numpy as np
from skimage.transform import resize


def normalize_pixels(X):
    """Chuẩn hóa pixel từ [0,16] về [0,1] (bất biến thang đo)."""
    return X / 16.0


# =====================================================================
# Preprocess cho GUI: crop + resize + căn giữa
# =====================================================================
# Vấn đề: ảnh trong load_digits đã được cắt & căn giữa sẵn, nhưng chữ người
# dùng VẺ trong GUI thường lệch vị trí và khác kích thước. K-NN tính khoảng
# cách pixel nên rất nhạy với sự lệch này -> đoán sai.
# Giải pháp: crop bounding box -> resize INNER x INNER -> căn giữa trong 8x8.
# Áp dụng CÙNG bước cho cả train và test để model nhất quán.

GUI_OUT = 8      # kích thước khung cuối (khớp load_digits)
GUI_INNER = 6    # kích thước chuẩn của nét chữ sau khi resize
GUI_CANVAS = 28  # kích thước canvas vẽ trong GUI (mịn hơn 8x8 để nét vẽ tự nhiên)


def downsample_canvas(canvas, out=GUI_OUT):
    """Downsample canvas vẽ (GUI_CANVAS x GUI_CANVAS) về (out, out).

    Người dùng vẽ trên canvas lớn (28x28) -> downsample về 8x8 tạo ra giá trị
    chống răng cưa (anti-alias) GIỐNG ảnh load_digits, giúp model nhận ra tốt.
    """
    return resize(canvas, (out, out), anti_aliasing=True, preserve_range=True)


def preprocess_digit(grid, out=GUI_OUT, inner=GUI_INNER):
    """Crop vùng chữ -> resize inner x inner -> căn giữa trong out x out.

    Tham số: grid (out, out) giá trị [0,1]. Trả về (out, out) giá trị [0,1].
    """
    rows = np.any(grid > 0.03, axis=1)
    cols = np.any(grid > 0.03, axis=0)
    if not rows.any() or not cols.any():
        return np.zeros((out, out))
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    cropped = grid[rmin:rmax + 1, cmin:cmax + 1]
    resized = resize(cropped, (inner, inner), anti_aliasing=True, preserve_range=True)
    out_img = np.zeros((out, out))
    pad = (out - inner) // 2
    out_img[pad:pad + inner, pad:pad + inner] = resized
    return out_img


def build_hog_features(images, out_size=32):
    """Tính đặc trưng HOG cho toàn bộ ảnh.

    Tham số HOG: orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2),
    block_norm='L2-Hys'. Với ảnh 32x32 -> lưới 4x4 ô (cell), 3x3 block -> vector 324 chiều.
    """
    from skimage.feature import hog
    from skimage.transform import resize

    n = images.shape[0]
    feats = np.zeros((n, 324))
    for i in range(n):
        img = resize(images[i] / 16.0, (out_size, out_size), anti_aliasing=True)
        feats[i] = hog(img, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), block_norm="L2-Hys")
    return feats
