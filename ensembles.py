import os
from pathlib import Path
from typing import Any

import hydra
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from calibration import evaluate_calibration
from log_utils import _log, add_result
from readme_leaderboard import load_leaderboard
from utils import (
    holdout_score,
    model_filename,
    run_method,
    save_submission,
    submission_output_path,
)


class PreTrainedStackingClassifier:
    """Мета-модель стекинга, которая объединяет предсказания от заранее обученных пайплайнов.
    
    Args:
        estimators: List of tuples `(name, pipeline)` where each pipeline implements `predict_proba`.
        final_estimator: Estimator to train on meta-features produced by base estimators.
    """

    def __init__(self, estimators: list[tuple[str, Any]], final_estimator: Any) -> None:
        self.estimators: list[tuple[str, Any]] = estimators
        self.final_estimator: Any = final_estimator

    def _get_meta_features(self, X: pd.DataFrame) -> np.ndarray:
        """Create meta-features by collecting positive-class probabilities from each estimator."""
        meta_features: list[np.ndarray] = []
        for name, pipe in self.estimators:
            preds = pipe.predict_proba(X)[:, 1]
            meta_features.append(preds)
        return np.column_stack(meta_features)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PreTrainedStackingClassifier":
        """Fit the final estimator on meta features extracted from X."""
        X_meta = self._get_meta_features(X)
        self.final_estimator.fit(X_meta, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict labels using the trained final estimator on meta features."""
        X_meta = self._get_meta_features(X)
        return self.final_estimator.predict(X_meta)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return class probabilities from the meta-model."""
        X_meta = self._get_meta_features(X)
        return self.final_estimator.predict_proba(X_meta)

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Compute accuracy of the stacked model on (X, y)."""
        from sklearn.metrics import accuracy_score

        return float(accuracy_score(y, self.predict(X)))

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return simple params for logging compatibility."""
        return {
            "estimators_count": len(self.estimators),
            "estimators_names": [name for name, _ in self.estimators],
        }


class PreTrainedVotingClassifier:
    """Простой усредняющий классификатор для предобученных моделей, который усредняет предсказанные вероятности."""

    def __init__(self, estimators: list[tuple[str, Any]]) -> None:
        self.estimators: list[tuple[str, Any]] = estimators  # list of (name, pipeline)

    def fit(self, X: pd.DataFrame | None = None, y: pd.Series | None = None) -> "PreTrainedVotingClassifier":
        """No-op fit for API compatibility (models are pre-trained)."""
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Average predicted probabilities from all estimators."""
        all_probas = [pipe.predict_proba(X) for _, pipe in self.estimators]
        return np.mean(all_probas, axis=0)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return class indices with highest averaged probability."""
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        return float(accuracy_score(y, self.predict(X)))

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return simple params for logging compatibility."""
        return {
            "estimators_count": len(self.estimators),
            "estimators_names": [name for name, _ in self.estimators],
        }


def load_stacking_pipeline(leaderboard: pd.DataFrame, top_k: int | None = None) -> PreTrainedStackingClassifier:
    pipelines = load_pipelines(leaderboard, top_k)
    estimators = list(pipelines.items())
    
    return PreTrainedStackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000, random_state=42)
    )


def load_voting_pipeline(leaderboard: pd.DataFrame, top_k: int | None = None) -> PreTrainedVotingClassifier:
    pipelines = load_pipelines(leaderboard, top_k)
    estimators = list(pipelines.items())

    return PreTrainedVotingClassifier(estimators=estimators)


def load_pipelines(leaderboard: pd.DataFrame, top_k: int | None = None) -> dict[str, Any]:
    best_models = (
        leaderboard.sort_values(by="accuracy", ascending=False)
            .groupby("model", as_index=False)
            .first()
    )

    if top_k is not None:
        best_models = best_models.head(top_k)

    pipeline_paths = [Path(path) for path in best_models['path'].to_list()]
    pipeline_names = best_models['model'].to_list()

    return {name: joblib.load(path) for path, name in zip(pipeline_paths, pipeline_names)}


def ensemble_return(
    model: Any,
    metric_to_score: dict[str, float],
    path: str,
    tuning_time: float | None = None,
    predict_time: float | None = None,
    n_samples: int | None = None,
) -> dict[str, Any]:
    """Строит словарь с результатами эксперимента для ансамблей моделей, включая метрики и время выполнения.

    Args:
        model: The model object or pipeline used for the ensemble.
        metric_to_score: Dict mapping metric names to their computed floats.
        path: Path where the model will be saved.
        tuning_time: Optional tuning time in seconds.
        predict_time: Optional prediction time in seconds.
        n_samples: Optional number of samples for latency computation.

    Returns:
        Dict[str, Any]: Experiment result record.
    """
    result = {
        "model": model,
        "accuracy": metric_to_score.get("accuracy", None),
        "f1_score": metric_to_score.get("f1_score", None),
        "precision": metric_to_score.get("precision", None),
        "recall": metric_to_score.get("recall", None),
        "brier_score": metric_to_score.get("brier_score", None),
        "ece": metric_to_score.get("ece", None),
        "params": model.get_params(),
        "path": path,
    }

    if tuning_time is not None:
        result["tuning_time_sec"] = round(tuning_time, 2)

    if predict_time is not None:
        result["predict_time_sec"] = round(predict_time, 4)
        if n_samples and n_samples > 0:
            # Latency: 1 obj per ms
            latency_ms = (predict_time / n_samples) * 1000
            result["latency_ms_per_sample"] = round(latency_ms, 4)

    return result


def make_ensembles(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    X_submit: pd.DataFrame,
    methods: dict[str, Any],
    cfg: Any,
    logger: Any | None = None,
) -> None:
    """Создает и оценивает ансамбли моделей, определенные в конфигурации.

    Функция читает пайплайны из лидерборда, создает ансамбли, оценивает их на отложенной выборке и при необходимости сохраняет модели/предсказания.
    """
    console = cfg.logging.console
    log_file_path = os.path.join(cfg.data.results_dir, "experiments.jsonl")

    leaderboard_df = load_leaderboard(log_file_path)

    ensemble_configs = []
    for ens_cfg in cfg.model.ensemble.list:
        ensemble_configs.append({
            "suffix": ens_cfg.suffix,
            "factory": lambda ec=ens_cfg: hydra.utils.instantiate(ec.factory, leaderboard=leaderboard_df)
        })

    for ens in ensemble_configs:
        suffix = ens["suffix"]
        model = ens["factory"]()
        
        _log(f"\n {model.__class__.__name__}", console)

        train_output = run_method(
            obj=model,
            method_name="fit",
            stage='train',
            X=X_train,
            y=y_train,
        )

        _log("Calibration skipped for ensemble", console)

        pred_output = holdout_score(model, X_val, y_val, metric=cfg.tuning.metric)

        _log(f"Holdout {cfg.tuning.metric}: {pred_output['result']:.4f}", console)
        _log(f"Final pipeline's training: {train_output['train_time_sec']:.4f} s.", console)
        _log(
            f"Holdout predictions ({len(X_val)} lines): {pred_output['predict_time_sec']:.4f} s.",
            console,
        )

        y_pred_holdout = model.predict(X_val)
        metric_to_score = {}
        for name, metric_cfg in methods.items():
            metric_fn = hydra.utils.instantiate(metric_cfg)
            score = metric_fn(y_val, y_pred_holdout)
            metric_to_score[name] = float(score)
            _log(f"Holdout {name}: {score:.4f}", console)

        n_bins = cfg.tuning.calibration.get("n_bins", 10)
        cal_scores = evaluate_calibration(model, X_val, y_val, n_bins=n_bins) or {}
        metric_to_score.update(cal_scores)
        for name, score in cal_scores.items():
            _log(f"Holdout {name}: {score:.4f}", console)

        model_name = model.__class__.__name__
        filename = model_filename(cfg, model_name, f"ensemble_{suffix}", pred_output["result"])
        
        res = ensemble_return(
            model=model,
            metric_to_score=metric_to_score,
            path=filename,
            tuning_time=train_output["train_time_sec"],
            predict_time=pred_output["predict_time_sec"],
            n_samples=len(X_val),
        )

        latency_ms_per_sample = (pred_output["predict_time_sec"] * 1000) / X_val.shape[0]
        res.update({"latency_ms_per_sample": latency_ms_per_sample})

        experiment = add_result(res, log_file_path=log_file_path)

        if logger is not None:
            logger.log_experiment(experiment)

        if cfg.logging.save_model:
            joblib.dump(model, filename)
            if logger is not None:
                logger.log_pipeline(filename)
            _log(f"Pipeline saved as: {filename}", console)

        if cfg.logging.save_predictions and X_submit is not None:
            submit_path = submission_output_path(cfg, model_name)
            save_submission(model, X_submit, submit_path)
