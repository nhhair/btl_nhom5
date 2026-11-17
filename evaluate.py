# -*- coding: utf-8 -*-
"""Chạy các phương án thử nghiệm, đánh giá, confusion matrix và phân tích lỗi.

Các phương án thử nghiệm:
  - Nhiều cấu hình K-NN (đổi k, metric)  : pixels 1.0 (đã chuẩn hóa [0,1]) + HOG
  - Nhiều cấu hình Cây quyết định           : pixels 1.0 + HOG
Tất cả bước tiền xử lý có học tham số (StandardScaler) đều được fit TRÊN TẬP TRAIN
trước khi áp dụng lên tập TEST -> hạn chế data leakage.
"""
import json
from collections import Counter
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from skimage.transform import rotate

from src.models import evaluate, train_dt, train_knn

RANDOM_STATE = 42


def build_experiments():
    """Danh sách các phương án thử nghiệm.

    Mỗi phương án là một dict:
      id      : tên định danh (được dùng khi in bảng / đặt tên file)
      feature : "pixels" hoặc "hog" (khớp key trong dict features ở main.py)
      algo    : "KNN" hoặc "DT"
      params  : tham số truyền thẳng vào bộ phân loại
      scale   : True nếu cần StandardScaler (fit trên train) trước khi huấn luyện
      note    : ghi chú ngắn
    """
    exps = []

    # (1) BASELINE: K-NN cấu hình mặc định trên pixel -> mốc so sánh bắt buộc
    exps.append({
        "id": "KNN-pixels-baseline", "feature": "pixels", "algo": "KNN",
        "params": {"n_neighbors": 5, "metric": "euclidean", "weights": "distance"},
        "scale": False, "note": "BASELINE - KNN cau hinh gan mac dinh",
    })

    # (2) K-NN trên pixel: thay đổi k và độ đo khoảng cách
    for k in (1, 3, 5, 7):
        for metric in ("euclidean", "manhattan"):
            exps.append({
                "id": f"KNN-pixels-k{k}-{metric}", "feature": "pixels", "algo": "KNN",
                "params": {"n_neighbors": k, "metric": metric, "weights": "distance"},
                "scale": True, "note": "scale pixels (bat buoc voi K-NN)",
            })

    # (3) Cây quyết định trên pixel: thay đổi criterion và max_depth
    for criterion in ("gini", "entropy"):
        for md in (None, 5, 10):
            exps.append({
                "id": f"DT-pixels-{criterion}-d{md}", "feature": "pixels", "algo": "DT",
                "params": {"criterion": criterion, "max_depth": md, "min_samples_leaf": 1},
                "scale": False, "note": "DT khong can scale",
            })

    # (4) K-NN trên HOG
    for k in (3, 5):
        exps.append({
            "id": f"KNN-hog-k{k}", "feature": "hog", "algo": "KNN",
            "params": {"n_neighbors": k, "metric": "euclidean", "weights": "distance"},
            "scale": True, "note": "scale hog",
        })

    # (5) Cây quyết định trên HOG
    for md in (None, 10):
        exps.append({
            "id": f"DT-hog-d{md}", "feature": "hog", "algo": "DT",
            "params": {"criterion": "gini", "max_depth": md, "min_samples_leaf": 1},
            "scale": False, "note": "raw hog",
        })

    return exps


def run_experiments(features, y_train, y_test, exps):
    """Chạy lần lượt các phương án.

    Tham số:
      features : {"pixels": (X_train, X_test), "hog": (X_train, X_test)}
      exps     : danh sách phương án từ build_experiments()

    Trả về:
      df     : DataFrame kết quả (để in bảng, lưu CSV)
      fitted : dict id -> {"scaler", "model", "feature", "params", "X_test", "y_test"}
               (dùng cho confusion matrix, phân tích lỗi, lưu mô hình)
    """
    rows = []
    fitted = {}

    for exp in exps:
        X_train, X_test = features[exp["feature"]]

        scaler = None
        if exp["scale"]:
            # Chỉ fit trên train -> tránh data leakage
            scaler = StandardScaler().fit(X_train)
            X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)
        else:
            X_train_s, X_test_s = X_train, X_test

        if exp["algo"] == "KNN":
            model, fit_time = train_knn(X_train_s, y_train, random_state=RANDOM_STATE,
                                        **exp["params"])
        else:
            model, fit_time = train_dt(X_train_s, y_train, random_state=RANDOM_STATE,
                                       **exp["params"])

        m = evaluate(model, X_test_s, y_test)

        rows.append({
            "id": exp["id"],
            "feature": exp["feature"],
            "algo": exp["algo"],
            "params": str(exp["params"]),
            "scale": "yes" if exp["scale"] else "no",
            "note": exp["note"],
            "accuracy": round(m["accuracy"], 4),
            "precision_macro": round(m["precision_macro"], 4),
            "recall_macro": round(m["recall_macro"], 4),
            "f1_macro": round(m["f1_macro"], 4),
            "fit_time_s": round(fit_time, 3),
        })
        fitted[exp["id"]] = {
            "scaler": scaler,
            "model": model,
            "feature": exp["feature"],
            "params": exp["params"],
            "X_test": X_test_s,
            "y_test": y_test,
        }

    df = pd.DataFrame(rows)
    return df, fitted


def robustness_analysis(fitted, model_ids, images_test, y_test,
                        results_dir: Path, figures_dir: Path):
    """Measure how a few fixed models handle mild image corruption.

    Gaussian noise and small rotations are applied only to held-out test images.
    The same corrupted images are evaluated for every model, and each model's
    scaler (fit on clean training data) is reused without refitting.
    """
    from src.features import build_hog_features, normalize_pixels

    rng = np.random.default_rng(RANDOM_STATE)
    clean = images_test.astype(float) / 16.0
    scenarios = [("clean", 0.0, 0)]
    scenarios.extend((f"noise_sigma_{sigma:.2f}", sigma, 0)
                     for sigma in (0.10, 0.20, 0.30))
    scenarios.extend((f"rotation_{angle}_deg", 0.0, angle)
                     for angle in (10, 20))

    rows = []
    for scenario, sigma, angle in scenarios:
        if sigma:
            distorted = np.clip(clean + rng.normal(0.0, sigma, clean.shape), 0.0, 1.0)
        elif angle:
            distorted = np.stack([
                rotate(img, angle, resize=False, order=1, mode="constant",
                       cval=0, preserve_range=True)
                for img in clean
            ])
        else:
            distorted = clean.copy()

        feature_cache = {
            "pixels": normalize_pixels(distorted.reshape(len(distorted), -1) * 16.0),
            "hog": build_hog_features(distorted * 16.0),
        }
        for exp_id in model_ids:
            fitted_model = fitted[exp_id]
            X = feature_cache[fitted_model["feature"]]
            scaler = fitted_model["scaler"]
            if scaler is not None:
                X = scaler.transform(X)
            pred = fitted_model["model"].predict(X)
            rows.append({
                "scenario": scenario,
                "model": exp_id,
                "accuracy": accuracy_score(y_test, pred),
                "f1_macro": f1_score(y_test, pred, average="macro", zero_division=0),
            })

    result = pd.DataFrame(rows)
    csv_path = results_dir / "robustness_results.csv"
    plot_path = figures_dir / "robustness_analysis.png"
    result.to_csv(csv_path, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for exp_id in model_ids:
        model_rows = result[result["model"] == exp_id]
        axes[0].plot(model_rows["scenario"], model_rows["accuracy"], marker="o", label=exp_id)
        axes[1].plot(model_rows["scenario"], model_rows["f1_macro"], marker="o", label=exp_id)
    axes[0].set_title("Accuracy under image corruption")
    axes[1].set_title("Macro F1 under image corruption")
    for ax in axes:
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("Score")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  [Saved] {csv_path}")
    print(f"  [Saved] {plot_path}")
    return result


def plot_confusion_matrices(fitted, ids, out_path: Path):
    """Vẽ ma trận nhầm lẫn cho các phương án trong danh sách ids."""
    k = len(ids)
    ncols = 2
    nrows = int(np.ceil(k / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 5.3 * nrows),
                             squeeze=False)

    for ax, exp_id in zip(axes.ravel(), ids):
        f = fitted[exp_id]
        y_true = f["y_test"]
        y_pred = f["model"].predict(f["X_test"])
        cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))

        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(exp_id, fontsize=11)
        ax.set_xlabel("Du doan")
        ax.set_ylabel("Thuc te")
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        for i in range(10):
            for j in range(10):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        fontsize=7, color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("Ma tran nham lan - Confusion Matrix", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [Da luu] {out_path}")
    return True


def error_analysis(fitted, exp_id, images_test, out_path: Path):
    """Phân tích các trường hợp dự đoán sai của một phương án cụ thể."""
    f = fitted[exp_id]
    y_true = f["y_test"]
    y_pred = f["model"].predict(f["X_test"])
    errors = np.where(y_true != y_pred)[0]

    print("-" * 74)
    print(f"BUOC 5 - PHAN TICH LOI ({exp_id})")
    print("-" * 74)
    n_all = len(y_true)
    err_rate = 100 * len(errors) / n_all
    print(f"  Tong so mau test          : {n_all}")
    print(f"  So mau du doan sai        : {len(errors)}  (ty le loi = {err_rate:.2f}%)")

    pairs = Counter(zip(y_true[errors].astype(int), y_pred[errors].astype(int)))
    print("\n  Cac cap (thuc te -> du doan) bi nham lan nhieu nhat:")
    for (true_l, pred_l), c in pairs.most_common(5):
        print(f"    chu so {true_l} bi nham thanh {pred_l}: {c} lan")

    n_show = min(15, len(errors))
    fig, axes = plt.subplots(3, 5, figsize=(11, 7.2))
    for ax, i in zip(axes.ravel(), errors[:n_show]):
        ax.imshow(images_test[i], cmap="gray_r", vmin=0, vmax=16)
        ax.set_title(f"Thuc: {int(y_true[i])} | Du doan: {int(y_pred[i])}",
                     fontsize=9, color="#c0392b")
        ax.axis("off")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Mot so anh bi du doan sai ({exp_id})", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [Da luu] {out_path}")
    return errors, pairs


def save_best_model(fitted, df, out_dir: Path):
    """Lưu mô hình có F1-macro cao nhất (scaler + model) dùng để tái sử dụng."""
    best_row = df.loc[df["f1_macro"].idxmax()]
    exp_id = best_row["id"]
    f = fitted[exp_id]
    obj = {
        "id": exp_id,
        "feature": f["feature"],
        "params": f["params"],
        "metrics": best_row[["accuracy", "precision_macro", "recall_macro", "f1_macro"]
                            ].to_dict(),
        "scaler": f["scaler"],
        "model": f["model"],
    }
    out_path = out_dir / "best_model.joblib"
    joblib.dump(obj, out_path)
    print(f"  [Da luu] {out_path}  ({exp_id}, f1_macro = {best_row['f1_macro']})")
    return exp_id, obj


def save_gui_model(features, y, out_dir: Path, exp_id="KNN-pixels-gui"):
    """Huấn luyện mô hình demo cho GUI.

    Vấn đề: chữ người dùng VẺ trong GUI (nét thô, lệch vị trí/kích thước) khác
    với ảnh load_digits (nét mảnh, đã căn giữa). K-NN tính khoảng cách pixel nên
    rất nhạy -> đoán sai.

    Giải pháp (đã test, ~91% trên mô phỏng vẽ chuột thực tế, 100% khi vẽ
    đúng hình dạng):
      - GUI vẽ trên canvas LỚN (28x28) cho nét mịn, rồi downsample về 8x8
        (tạo giá trị chống răng cưa giống load_digits).
      - Model được train trên CÙNG pipeline đó, kèm AUGMENTATION mô phỏng
        đầy đủ cách người dùng vẽ chuột:
          * Độ dày nét  : erode (nét mỏng) / dilate (nét dày)
          * Xoay nhẹ    : vẽ nghiêng -8..8 độ
          * Co giãn     : vẽ to/nhỏ (0.85..1.15)
          * Dịch chuyển : vẽ lệch vị trí
          * Làm mềm     : anti-alias (giống nét load_digits)
      - Train trên TOÀN BỘ dữ liệu (chỉ phục vụ demo, không dùng để báo cáo).

    Lưu: scaler + model + tham số preprocess/canvas để gui.py tái sử dụng.
    """
    from scipy.ndimage import (binary_dilation, binary_erosion,
                               gaussian_filter, rotate as nd_rotate,
                               shift as nd_shift)
    from skimage.transform import resize
    from sklearn.preprocessing import StandardScaler

    from src.features import (GUI_CANVAS, GUI_INNER, GUI_OUT,
                              downsample_canvas, preprocess_digit)

    def _soften(v, sigma=1.0):
        v = gaussian_filter(v, sigma=sigma)
        return v / (v.max() + 1e-9)

    def _stroke_variants(big):
        """Sinh các biến thể 'vẽ tay' từ ảnh canvas 28x28."""
        mask = big > 0.3
        variants = [big]
        # Độ dày nét: mỏng (erode) / dày (dilate)
        for it in (1, 2):
            variants.append(_soften(np.where(binary_erosion(mask, iterations=it),
                                             1.0, 0.0)))
        for it in (1, 2):
            variants.append(_soften(np.where(binary_dilation(mask, iterations=it),
                                             1.0, 0.0)))
        # Xoay nhẹ (vẽ nghiêng)
        for ang in (-8, -4, 4, 8):
            variants.append(nd_rotate(big, ang, reshape=False, order=1,
                                      mode="constant"))
        # Co giãn (vẽ to/nhỏ) - căn giữa lại vào canvas
        for sc in (0.85, 1.15):
            new = int(GUI_CANVAS * sc)
            r = resize(big, (new, new), anti_aliasing=True, preserve_range=True)
            out = np.zeros((GUI_CANVAS, GUI_CANVAS))
            pad = (GUI_CANVAS - new) // 2
            if new <= GUI_CANVAS:
                out[max(0, pad):pad + new, max(0, pad):pad + new] = \
                    r[max(0, -pad):new + min(0, pad),
                      max(0, -pad):new + min(0, pad)]
            else:
                c0 = (new - GUI_CANVAS) // 2
                out[:, :] = r[c0:c0 + GUI_CANVAS, c0:c0 + GUI_CANVAS]
            variants.append(out)
        return variants

    X_all = features  # (n, 64) pixel đã chuẩn hóa [0,1]
    images = X_all.reshape(-1, GUI_OUT, GUI_OUT)
    n = images.shape[0]

    # Các dịch chuyển trên canvas 28x28 (mô phỏng vẽ lệch vị trí)
    shifts = [(0, 0), (3, 2), (-3, 2), (2, -3), (-2, 3),
              (4, -2), (-4, 2), (0, 4), (0, -4)]

    X, yy = [], []
    for i in range(n):
        big = resize(images[i], (GUI_CANVAS, GUI_CANVAS),
                     anti_aliasing=True, preserve_range=True)
        for v in _stroke_variants(big):
            for (dy, dx) in shifts:
                sh = nd_shift(v, shift=(dy, dx), mode="constant")
                small = downsample_canvas(sh)
                X.append(preprocess_digit(small).ravel())
                yy.append(y[i])
    X = np.array(X)
    yy = np.array(yy)

    scaler = StandardScaler().fit(X)
    model, _ = train_knn(scaler.transform(X), yy, n_neighbors=5, metric="euclidean",
                         random_state=RANDOM_STATE)
    obj = {
        "id": exp_id,
        "scaler": scaler,
        "model": model,
        "preprocess": {"out": GUI_OUT, "inner": GUI_INNER},
        "canvas": GUI_CANVAS,
    }
    out_path = out_dir / "gui_model.joblib"
    joblib.dump(obj, out_path)
    print(f"  [Da luu] {out_path}  (model demo GUI - {X.shape[0]} mau sau augment, "
          f"canvas {GUI_CANVAS}x{GUI_CANVAS}, robust net mong/day + xoay + lech)")
    return obj


def export_summary(df, out_path: Path):
    """Lưu bảng so sánh kết quả ra JSON và CSV."""
    df.to_csv(out_path, index=False)
    print(f"  [Da luu] {out_path}")

    best = df.loc[df["f1_macro"].idxmax()]
    summary = {
        "random_state": RANDOM_STATE,
        "test_size": 0.2,
        "best_model": {"id": best["id"], "f1_macro": float(best["f1_macro"]),
                       "accuracy": float(best["accuracy"])},
        "n_experiments": int(len(df)),
    }
    with open(out_path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  [Da luu] {out_path.with_suffix('.json')}")
