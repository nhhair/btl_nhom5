# -*- coding: utf-8 -*-
"""Huấn luyện mô hình (K-NN, Cây quyết định) và tính các độ đo đánh giá."""
from time import perf_counter

from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def train_knn(X_train, y_train, n_neighbors=5, metric="euclidean",
              weights="distance", random_state=42):
    """Huấn luyện K-NN, trả về (model, thời gian huấn luyện)."""
    model = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric, weights=weights)
    t0 = perf_counter()
    model.fit(X_train, y_train)
    return model, perf_counter() - t0


def train_dt(X_train, y_train, criterion="gini", max_depth=None,
             min_samples_leaf=1, random_state=42):
    """Huấn luyện Cây quyết định, trả về (model, thời gian huấn luyện)."""
    model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth,
                                   min_samples_leaf=min_samples_leaf,
                                   random_state=random_state)
    t0 = perf_counter()
    model.fit(X_train, y_train)
    return model, perf_counter() - t0


def evaluate(model, X_test, y_test):
    """Tính Accuracy, Precision/Recall/F1-macro trên tập kiểm thử.

    Trả về dict chứa các độ đo và nhãn dự đoán y_pred.
    """
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "n_test": len(y_test),
        "y_pred": y_pred,
    }
