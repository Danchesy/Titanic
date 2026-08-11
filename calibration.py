from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss

from log_utils import _log

__all__ = [
    "calibrate_pipeline",
    "calibration_metrics",
    "evaluate_calibration",
    "expected_calibration_error",
]


def calibrate_pipeline(
    pipeline: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    method: str | None,
    console: bool = False,
) -> Any:
    """Калибрует обученный пайплайн с помощью CalibratedClassifierCV.

    Если ``method`` равен ``None``, пайплайн возвращается без изменений.
    Иначе оборачивает пайплайн в ``CalibratedClassifierCV(cv="prefit")``
    и обучает калибратор на ``(X_val, y_val)``.

    Args:
        pipeline: Обученный sklearn-пайплайн или классификатор.
        X_val: Валидационные признаки для калибровки.
        y_val: Валидационные метки для калибровки.
        method: Метод калибровки (``"sigmoid"`` — Platt Scaling,
            ``"isotonic"`` — Isotonic Regression) или ``None``.
        console: Флаг вывода сообщений в консоль.

    Returns:
        Калиброванный пайплайн (``CalibratedClassifierCV``) или
        исходный пайплайн, если ``method is None``.
    """
    if method is None:
        _log("Calibration skipped (method=null)", console)
        return pipeline

    _log(f"Calibrating with method='{method}'...", console)

    calibrated = CalibratedClassifierCV(
        estimator=pipeline,
        method=method,
        cv="prefit",
    )
    calibrated.fit(X_val, y_val)

    _log("Calibration done", console)
    return calibrated


def expected_calibration_error(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE): среднее |accuracy - confidence| по бинам."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i < n_bins - 1:
            mask = (y_prob >= lo) & (y_prob < hi)
        else:
            mask = (y_prob >= lo) & (y_prob <= hi)

        if not mask.any():
            continue

        ece += mask.mean() * abs(y_true[mask].mean() - y_prob[mask].mean())

    return float(ece)


def calibration_metrics(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict[str, float]:
    """Brier score и ECE по вероятностям положительного класса."""
    
    return {
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "ece": expected_calibration_error(y_true, y_prob, n_bins=n_bins),
    }


def evaluate_calibration(
    estimator: Any,
    X: pd.DataFrame,
    y: pd.Series,
    n_bins: int = 10,
) -> dict[str, float] | None:
    """Считает метрики калибровки, если у модели есть ``predict_proba``."""
    if not hasattr(estimator, "predict_proba"):
        return None

    y_prob = estimator.predict_proba(X)[:, 1]
    return calibration_metrics(y, y_prob, n_bins=n_bins)
