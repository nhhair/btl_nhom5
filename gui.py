# -*- coding: utf-8 -*-
"""Giao diện tương tác nhận dạng chữ số viết tay.

Cách dùng:
  1. Chạy `python main.py` một lần để tạo models/gui_model.joblib.
  2. Chạy `python gui.py`.
  3. Chọn độ dày nét, vẽ chữ số rồi bấm "Nhận dạng" (hoặc nhấn Space).

Giao diện dùng cùng pipeline với lúc huấn luyện: canvas 28x28 -> downsample
8x8 -> crop/căn giữa -> scaler và K-NN đã lưu.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

MODEL_PATH = ROOT / "models" / "gui_model.joblib"
CANVAS_SIZE = 28
CELL_PX = 16


def build_gui():
    import joblib
    import numpy as np
    import tkinter as tk
    from tkinter import ttk

    from src.features import downsample_canvas, preprocess_digit

    if not MODEL_PATH.exists():
        raise SystemExit(
            f"Chưa có mô hình: {MODEL_PATH}\n"
            "Hãy chạy 'python main.py' trước để tạo mô hình demo."
        )
    pipe = joblib.load(MODEL_PATH)

    # Bảng màu và kiểu thành phần dùng chung trong cửa sổ.
    bg = "#edf2f7"
    ink = "#142d3a"
    muted = "#657984"
    teal = "#087f8c"
    white = "#ffffff"

    root = tk.Tk()
    root.title("Digit Studio | Nhận dạng chữ số viết tay")
    root.configure(bg=bg)
    root.geometry("1020x760")
    root.minsize(960, 720)

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    style.configure("App.TFrame", background=bg)
    style.configure("Card.TFrame", background=white)
    style.configure("Title.TLabel", background=bg, foreground=ink,
                    font=("Segoe UI", 22, "bold"))
    style.configure("SubTitle.TLabel", background=bg, foreground=muted,
                    font=("Segoe UI", 10))
    style.configure("CardTitle.TLabel", background=white, foreground=ink,
                    font=("Segoe UI", 12, "bold"))
    style.configure("Body.TLabel", background=white, foreground=muted,
                    font=("Segoe UI", 9))
    style.configure("TScale", background=white)

    class DigitCanvas:
        """Lưới 28x28 có bút ba mức và nội suy điểm khi rê chuột nhanh."""

        def __init__(self, master, brush_var):
            self.size = CANVAS_SIZE
            self.cell = CELL_PX
            self.brush_var = brush_var
            self.grid = np.zeros((self.size, self.size), dtype=float)
            self.last_cell = None
            self.widget = tk.Canvas(
                master,
                width=self.size * self.cell,
                height=self.size * self.cell,
                bg="#172733",
                highlightthickness=1,
                highlightbackground="#314957",
                cursor="tcross",
            )
            self.widget.bind("<ButtonPress-1>", self._start_stroke)
            self.widget.bind("<B1-Motion>", self._continue_stroke)
            self.widget.bind("<ButtonRelease-1>", self._end_stroke)
            self._redraw()

        def _stamp(self, row, col):
            radius = max(0, int(round(float(self.brush_var.get()))) - 1)
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    distance = (dx * dx + dy * dy) ** 0.5
                    if distance > radius + 0.25:
                        continue
                    rr, cc = row + dy, col + dx
                    if 0 <= rr < self.size and 0 <= cc < self.size:
                        value = 1.0 if distance <= max(0.5, radius * 0.55) else 0.55
                        self.grid[rr, cc] = max(self.grid[rr, cc], value)

        def _draw_to(self, event):
            col, row = event.x // self.cell, event.y // self.cell
            if not (0 <= row < self.size and 0 <= col < self.size):
                self.last_cell = None
                return
            if self.last_cell is None:
                points = [(row, col)]
            else:
                old_row, old_col = self.last_cell
                steps = max(abs(row - old_row), abs(col - old_col), 1)
                points = [
                    (round(old_row + (row - old_row) * step / steps),
                     round(old_col + (col - old_col) * step / steps))
                    for step in range(1, steps + 1)
                ]
            for rr, cc in points:
                self._stamp(rr, cc)
            self.last_cell = (row, col)
            self._redraw()

        def _start_stroke(self, event):
            self.last_cell = None
            self._draw_to(event)
            return "break"

        def _continue_stroke(self, event):
            self._draw_to(event)
            return "break"

        def _end_stroke(self, _event):
            self.last_cell = None

        def _redraw(self):
            self.widget.delete("all")
            for row in range(self.size):
                for col in range(self.size):
                    value = self.grid[row, col]
                    if value > 0.03:
                        shade = int(35 + 220 * value)
                        color = f"#{shade:02x}{shade:02x}{shade:02x}"
                        self.widget.create_rectangle(
                            col * self.cell, row * self.cell,
                            (col + 1) * self.cell, (row + 1) * self.cell,
                            fill=color, outline="",
                        )
            for index in range(0, self.size + 1, 4):
                pos = index * self.cell
                self.widget.create_line(pos, 0, pos, self.size * self.cell,
                                        fill="#2a3d49")
                self.widget.create_line(0, pos, self.size * self.cell, pos,
                                        fill="#2a3d49")

        def clear(self):
            self.grid.fill(0)
            self.last_cell = None
            self._redraw()

        def array(self):
            return self.grid.copy()

    # Phần đầu trang.
    page = ttk.Frame(root, style="App.TFrame", padding=(26, 22, 26, 20))
    page.pack(fill="both", expand=True)
    ttk.Label(page, text="Digit Studio", style="Title.TLabel").pack(anchor="w")
    ttk.Label(
        page,
        text="Vẽ một chữ số từ 0 đến 9, sau đó xem dự đoán và điểm của từng lớp.",
        style="SubTitle.TLabel",
    ).pack(anchor="w", pady=(2, 16))

    body = ttk.Frame(page, style="App.TFrame")
    body.pack(fill="both", expand=True)
    body.columnconfigure(0, weight=1)
    body.columnconfigure(1, weight=0)
    body.rowconfigure(0, weight=1)

    # Thẻ bên trái: bảng vẽ và các thao tác.
    draw_card = ttk.Frame(body, style="Card.TFrame", padding=20)
    draw_card.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
    ttk.Label(draw_card, text="Bảng vẽ", style="CardTitle.TLabel").pack(anchor="w")
    ttk.Label(draw_card, text="Giữ chuột trái và vẽ trong khung.",
              style="Body.TLabel").pack(anchor="w", pady=(3, 12))

    brush_var = tk.DoubleVar(value=2)
    canvas = DigitCanvas(draw_card, brush_var)
    canvas.widget.pack(anchor="center")

    brush_row = ttk.Frame(draw_card, style="Card.TFrame")
    brush_row.pack(fill="x", pady=(13, 0))
    brush_name = tk.StringVar(value="Vừa")
    ttk.Label(brush_row, text="Độ dày nét", style="Body.TLabel").pack(side="left")
    brush_value = ttk.Label(brush_row, textvariable=brush_name,
                            style="Body.TLabel", width=8, anchor="e")
    brush_value.pack(side="right")

    def update_brush(value):
        names = {1: "Mảnh", 2: "Vừa", 3: "Dày", 4: "Rất dày"}
        brush_name.set(names.get(int(round(float(value))), "Vừa"))

    brush_scale = ttk.Scale(brush_row, from_=1, to=4, variable=brush_var,
                            command=update_brush)
    brush_scale.pack(side="left", fill="x", expand=True, padx=12)

    actions = ttk.Frame(draw_card, style="Card.TFrame")
    actions.pack(fill="x", pady=(14, 0))

    # Thẻ bên phải: kết quả lớn và điểm dự đoán theo từng chữ số.
    result_card = ttk.Frame(body, style="Card.TFrame", padding=18)
    result_card.grid(row=0, column=1, sticky="nsew")
    result_card.configure(width=310)
    result_card.grid_propagate(False)
    ttk.Label(result_card, text="Kết quả nhận dạng",
              style="CardTitle.TLabel").pack(anchor="w")
    ttk.Label(result_card, text="Chữ số dự đoán", style="Body.TLabel").pack(
        anchor="w", pady=(12, 0))
    prediction_label = tk.Label(result_card, text="—", bg=white, fg=teal,
                                font=("Segoe UI", 70, "bold"), height=1)
    prediction_label.pack(anchor="center", pady=(0, 0))
    confidence_var = tk.StringVar(value="Điểm K-NN: —")
    tk.Label(result_card, textvariable=confidence_var, bg=white, fg=ink,
             font=("Segoe UI", 10, "bold")).pack(anchor="center", pady=(0, 8))
    tk.Frame(result_card, bg="#e5edf1", height=1).pack(fill="x", pady=(2, 12))
    ttk.Label(result_card, text="Điểm theo từng chữ số",
              style="CardTitle.TLabel").pack(anchor="w", pady=(0, 8))

    score_widgets = {}
    for digit in range(10):
        row = tk.Frame(result_card, bg=white)
        row.pack(fill="x", pady=3)
        tk.Label(row, text=str(digit), bg=white, fg=ink,
                 font=("Segoe UI", 9, "bold"), width=2, anchor="w").pack(side="left")
        bar = tk.Canvas(row, width=145, height=12, bg="#e9eff2",
                        highlightthickness=0, bd=0)
        bar.pack(side="left", padx=(5, 7))
        rect = bar.create_rectangle(0, 0, 0, 12, fill=teal, outline="")
        percent = tk.Label(row, text="0.0%", bg=white, fg=muted,
                           font=("Segoe UI", 8), width=6, anchor="e")
        percent.pack(side="left")
        score_widgets[digit] = (bar, rect, percent)

    status_var = tk.StringVar(value="Chưa có dự đoán")
    tk.Label(result_card, textvariable=status_var, bg=white, fg=muted,
             font=("Segoe UI", 8), wraplength=260, justify="left").pack(
                 anchor="w", pady=(14, 0))

    def reset_result():
        prediction_label.configure(text="—")
        confidence_var.set("Điểm K-NN: —")
        status_var.set("Chưa có dự đoán")
        for bar, rect, percent in score_widgets.values():
            bar.coords(rect, 0, 0, 0, 12)
            percent.configure(text="0.0%")

    def predict():
        grid = canvas.array()
        if np.max(grid) < 0.03:
            status_var.set("Hãy vẽ một chữ số trước khi nhận dạng.")
            return

        small = preprocess_digit(downsample_canvas(grid))
        features = small.reshape(1, -1)
        if pipe.get("scaler") is not None:
            features = pipe["scaler"].transform(features)
        model = pipe["model"]
        prediction = int(model.predict(features)[0])
        prediction_label.configure(text=str(prediction))

        try:
            probabilities = model.predict_proba(features)[0]
            classes = model.classes_.astype(int)
            score_by_digit = {int(label): float(score)
                              for label, score in zip(classes, probabilities)}
            best_score = score_by_digit.get(prediction, 0.0)
            confidence_var.set(f"Điểm K-NN: {best_score * 100:.1f}%")
            for digit, (bar, rect, percent) in score_widgets.items():
                score = score_by_digit.get(digit, 0.0)
                width = int(bar.winfo_reqwidth() * score)
                bar.coords(rect, 0, 0, width, 12)
                percent.configure(text=f"{score * 100:.1f}%")
            status_var.set("Điểm thể hiện tỷ trọng láng giềng gần ủng hộ mỗi chữ số.")
        except (AttributeError, ValueError):
            confidence_var.set("Điểm K-NN: không có")
            status_var.set("Mô hình không cung cấp phân phối điểm cho dự đoán này.")

    def clear():
        canvas.clear()
        reset_result()

    def button(master, text, command, color, hover):
        widget = tk.Button(
            master, text=text, command=command, bg=color, fg="white",
            activebackground=hover, activeforeground="white",
            relief="flat", bd=0, padx=16, pady=10,
            font=("Segoe UI", 10, "bold"), cursor="hand2",
        )
        return widget

    button(actions, "Nhận dạng", predict, teal, "#066b76").pack(
        side="left", fill="x", expand=True, padx=(0, 8))
    button(actions, "Xóa nét", clear, "#526571", "#3d505c").pack(
        side="left", fill="x", expand=True, padx=(8, 0))

    ttk.Label(page, text="Phím tắt: Space = nhận dạng    •    Esc = xóa nét",
              style="SubTitle.TLabel").pack(anchor="w", pady=(10, 0))
    root.bind("<space>", lambda _event: predict())
    root.bind("<Escape>", lambda _event: clear())
    root.mainloop()


if __name__ == "__main__":
    build_gui()
