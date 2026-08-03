import os
from collections.abc import Callable
from logging import _log, add_result
from typing import Any

import hydra
import joblib
import optuna
import pandas as pd
from omegaconf import DictConfig
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline

from preprocessing import build_preprocessor, pipeline_fit_params
from utils import (
    ensure_dirs,
    holdout_score,
    model_filename,
    pipeline_return,
    run_method,
    save_submission,
    submission_output_path,
)

__all__ = [
    "catboost_optuna_params",
    "dt_optuna_params",
    "grid_tuning",
    "knn_optuna_params",
    "lgbm_optuna_params",
    "logreg_optuna_params",
    "optuna_tuning",
    "rf_optuna_params",
    "xgb_optuna_params",
]


def grid_tuning(
    model: Any,
    params: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cfg: DictConfig,
    model_cfg: DictConfig,
    methods: dict[str, Any],
    is_scale: bool = True,
    is_cat: bool = True,
    cat_features: list[str] | None = None,
    X_submit: pd.DataFrame | None = None,
    logger: Any = None,
) -> GridSearchCV:
    """Обёртка GridSearchCV: строит пайплайн с предобработкой, обучает, логирует и сохраняет лучший estimator.

    Args:
        model: sklearn-совместимый классификатор.
        params: сетка гиперпараметров с префиксом ``model__``.
        X_train, y_train: обучающая выборка.
        X_test, y_test: валидационная выборка для итоговой оценки.
        is_scale: передаётся в :func:`preprocessor`.
        is_cat: передаётся в :func:`preprocessor`.
        logger: объект с методом ``log_experiment`` / ``log_pipeline``.

    Returns:
        Обученный ``GridSearchCV`` с лучшим estimator'ом в ``best_estimator_``.
    """

    ensure_dirs(cfg)
    console = cfg.logging.console
    metric = cfg.tuning.metric
    cv_folds = cfg.training.cv_folds

    pipeline = Pipeline(
        [
            ("preprocessor", build_preprocessor(cfg, model_cfg, is_scale=is_scale, is_cat=is_cat)),
            ("model", model),
        ]
    )

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=params,
        scoring=metric,
        refit=True,
        cv=cv_folds,
        n_jobs=-1,
        verbose=1 if console else 0,
        pre_dispatch="2*n_jobs",
        return_train_score=False,
    )

    train_output = run_method(obj=grid_search, method_name='fit', stage='train', X=X_train, y=y_train, **pipeline_fit_params(cat_features))

    _log(f"Best parameters: {grid_search.best_params_}", console)
    _log(f"Best CV {metric}: {grid_search.best_score_:.4f}", console)
    _log(f"GridSearch trainig time: {train_output['train_time_sec']:.2f} s.", console)

    best_pipeline = grid_search.best_estimator_

    pred_output = holdout_score(pipeline=best_pipeline, X=X_test, y=y_test, metric=metric)

    _log(f"Holdout {metric}: {pred_output['result']:.4f}", console)
    _log(f"Holdout predict ({len(X_test)} lines): {pred_output['result']:.4f} s.", console)
    _log(f"Latency: {(pred_output['predict_time_sec'] / len(X_test)) * 1000:.4f} ms", console)

    y_pred_holdout = best_pipeline.predict(X_test)
    
    metric_to_score = {}
    for name, metric_cfg in methods.items():
        metric_fn = hydra.utils.instantiate(metric_cfg)
        
        score = metric_fn(y_test, y_pred_holdout)
        metric_to_score[name] = float(score)
        
        _log(f"Holdout {name}: {score:.4f}", console)

    res = pipeline_return(
        best_pipeline,
        grid_search.best_score_,
        tuning_time=train_output["train_time_sec"],
        predict_time=pred_output["predict_time_sec"],
        n_samples=len(X_test),
    )

    res.update(metric_to_score)

    experiment = add_result(res, log_file_path=os.path.join(cfg.data.results_dir, "experiments.jsonl"))

    if logger is not None:
        logger.log_experiment(experiment)

    model_name = model.__class__.__name__
    filename = model_filename(cfg, model_name, "grid", grid_search.best_score_)

    if cfg.logging.save_model:
        joblib.dump(best_pipeline, filename)
        if logger is not None:
            logger.log_pipeline(filename)
        _log(f"Pipeline saved as: {filename}", console)

    if cfg.logging.save_predictions and X_submit is not None:
        submit_path = submission_output_path(cfg, model_name)
        save_submission(best_pipeline, X_submit, submit_path)

    return grid_search


def optuna_tuning(
    model: Any,
    params_fn: Callable[[optuna.Trial], dict[str, Any]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cfg: DictConfig,
    model_cfg: DictConfig,
    methods: dict[str, Any],
    n_trials: int = 20,
    is_scale: bool = True,
    is_cat: bool = True,
    cat_features: list[str] | None = None,
    X_submit: pd.DataFrame | None = None,
    logger: Any = None,
) -> optuna.Study:
    """Оптимизация гиперпараметров через Optuna с 5-fold CV внутри objective.

    Args:
        model: базовый sklearn-классификатор (клонируется для каждого trial).
        params_fn: функция ``(trial) -> dict`` с пространством поиска.
        X_train, y_train: обучающая выборка.
        X_test, y_test: выборка для финальной оценки лучшей модели.
        direction: ``"maximize"`` или ``"minimize"``.
        n_trials: число испытаний Optuna.
        is_scale, is_cat: флаги предобработки.
        logger: объект с методом ``log_experiment`` / ``log_pipeline``.

    Returns:
        Завершённый ``optuna.Study`` с атрибутами ``best_params`` и ``best_value``.
    """

    ensure_dirs(cfg)
    console = cfg.logging.console
    metric = cfg.tuning.metric
    cv_folds = cfg.training.cv_folds
    direction = cfg.tuning.direction
    timeout = cfg.tuning.timeout

    cv_params = pipeline_fit_params(cat_features)

    def objective(trial: optuna.Trial) -> float:
        params = params_fn(trial)

        current_model = clone(model)
        current_model.set_params(**params)

        pipeline = Pipeline(
            [
                ("preprocessor", build_preprocessor(cfg, model_cfg, is_scale=is_scale, is_cat=is_cat)),
                ("model", current_model),
            ]
        )

        score = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv_folds,
            scoring=metric,
            params=cv_params,
        ).mean()
        return score

    study = optuna.create_study(direction=direction)
    optimizer_output = run_method(obj=study, method_name='optimize', stage='optuna', func=objective, n_trials=n_trials, timeout=timeout)

    best_params = study.best_trial.user_attrs.get("sklearn_params", study.best_params)

    _log(f"Best CV {metric}: {study.best_value:.4f}", console)
    _log(f"Best parameters: {best_params}", console)
    _log(f"Optuna optimization time ({n_trials} trials): {optimizer_output['optuna_time_sec']:.2f} s.", console)
    _log(f"Mean time per trial: {optimizer_output['optuna_time_sec'] / max(n_trials, 1):.2f} s.", console)

    best_model = clone(model)
    best_model.set_params(**best_params)

    final_pipeline = Pipeline(
        [
            ("preprocessor", build_preprocessor(cfg, model_cfg, is_scale, is_cat)),
            ("model", best_model),
        ]
    )

    train_output = run_method(obj=final_pipeline, method_name='fit', stage='train', X=X_train, y=y_train, **pipeline_fit_params(cat_features))

    
    pred_output = holdout_score(final_pipeline, X_test, y_test, metric)
    

    _log(f"Holdout {metric}: {pred_output['result']:.4f}", console)
    _log(f"Final pipeline's training: {train_output['train_time_sec']:.4f} s.", console)
    _log(f"Holdout predictions ({len(X_test)} lines): {pred_output['predict_time_sec']:.4f} s.", console)

    y_pred_holdout = final_pipeline.predict(X_test)
    
    metric_to_score = {}
    for name, metric_cfg in methods.items():
        metric_fn = hydra.utils.instantiate(metric_cfg)
        
        score = metric_fn(y_test, y_pred_holdout)
        metric_to_score[name] = float(score)
        
        _log(f"Holdout {name}: {score:.4f}", console)

    res = pipeline_return(
        final_pipeline,
        study.best_value,
        tuning_time=optimizer_output['optuna_time_sec'],
        predict_time=pred_output["predict_time_sec"],
        n_samples=len(X_test),
    )

    res.update(metric_to_score)

    experiment = add_result(res, log_file_path=os.path.join(cfg.data.results_dir, "experiments.jsonl"))

    if logger is not None:
        logger.log_experiment(experiment)

    model_name = model.__class__.__name__
    filename = model_filename(cfg, model_name, "optuna", study.best_value)

    if cfg.logging.save_model:
        joblib.dump(final_pipeline, filename)
        if logger is not None:
            logger.log_pipeline(filename)
        _log(f"Pipeline saved as: {filename}", console)

    if cfg.logging.save_predictions and X_submit is not None:
        submit_path = submission_output_path(cfg, model_name)
        save_submission(final_pipeline, X_submit, submit_path)

    return study


def logreg_optuna_params(
    trial: optuna.Trial,
    solver_mapping: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Пространство поиска гиперпараметров LogisticRegression."""
    mapping = solver_mapping
    solver = trial.suggest_categorical("solver", list(mapping.keys()))
    val = mapping[solver]

    if isinstance(val, dict) and "penalty" in val:
        allowed_penalties = val["penalty"]
    else:
        allowed_penalties = val

    allowed_penalties = [None if p == "none" else p for p in allowed_penalties]
    
    # Разные пространства выбора для разных solver называем penalty_<solver>
    penalty = trial.suggest_categorical(f"penalty_{solver}", allowed_penalties)

    params: dict[str, Any] = {
        "solver": solver,
        "penalty": penalty,
        "C": trial.suggest_float("C", 1e-4, 100.0, log=True),
        "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
    }

    if penalty == "elasticnet":
        params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.0, 1.0)

    trial.set_user_attr("sklearn_params", params)
    return params


def knn_optuna_params(trial: optuna.Trial) -> dict[str, Any]:
    """Пространство поиска гиперпараметров KNN."""
    return {
        "n_neighbors": trial.suggest_int("n_neighbors", 3, 17, step=2),
        "weights": trial.suggest_categorical("weights", ["distance", "uniform"]),
        "leaf_size": trial.suggest_categorical("leaf_size", [20, 30, 50]),
        "metric": trial.suggest_categorical(
            "metric", ["cosine", "manhattan", "euclidean"]
        ),
    }


def dt_optuna_params(trial: optuna.Trial) -> dict[str, Any]:
    """Пространство поиска гиперпараметров DecisionTree."""
    return {
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "min_samples_split": trial.suggest_categorical(
            "min_samples_split", [5, 10, 20, 30]
        ),
        "min_samples_leaf": trial.suggest_categorical(
            "min_samples_leaf", [2, 5, 10, 20]
        ),
        "min_weight_fraction_leaf": trial.suggest_float(
            "min_weight_fraction_leaf", 0.0, 0.2
        ),
        "max_features": trial.suggest_categorical(
            "max_features", [None, "sqrt", "log2", 0.5, 0.8]
        ),
        "max_leaf_nodes": trial.suggest_categorical(
            "max_leaf_nodes", [None, 5, 10, 25, 50, 100]
        ),
        "min_impurity_decrease": trial.suggest_float(
            "min_impurity_decrease", 0.0, 0.01
        ),
    }


def rf_optuna_params(trial: optuna.Trial) -> dict[str, Any]:
    """Пространство поиска гиперпараметров RandomForest."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "min_samples_split": trial.suggest_categorical(
            "min_samples_split", [5, 10, 20, 30]
        ),
        "min_samples_leaf": trial.suggest_categorical(
            "min_samples_leaf", [2, 5, 10, 20]
        ),
        "min_weight_fraction_leaf": trial.suggest_float(
            "min_weight_fraction_leaf", 0.0, 0.2
        ),
        "max_features": trial.suggest_categorical(
            "max_features", [None, "sqrt", "log2", 0.5, 0.8]
        ),
        "max_leaf_nodes": trial.suggest_categorical(
            "max_leaf_nodes", [None, 5, 10, 25, 50, 100]
        ),
        "min_impurity_decrease": trial.suggest_float(
            "min_impurity_decrease", 0.0, 0.01
        ),
    }


def xgb_optuna_params(trial: optuna.Trial) -> dict[str, Any]:
    """Пространство поиска гиперпараметров XGBoost."""
    return {
        "max_depth": trial.suggest_int("max_depth", 3, 5),
        "n_estimators": trial.suggest_int("n_estimators", 100, 200),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 2.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 0.8, step=0.05),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.8, step=0.05),
        "gamma": trial.suggest_float("gamma", 0.0, 0.3, step=0.05),
    }


def lgbm_optuna_params(trial: optuna.Trial) -> dict[str, Any]:
    """Пространство поиска гиперпараметров LightGBM."""
    return {
        "max_depth": trial.suggest_int("max_depth", 3, 5),
        "n_estimators": trial.suggest_int("n_estimators", 100, 200),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 2.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 0.8, step=0.05),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.8, step=0.05),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.3, step=0.05),
    }


def catboost_optuna_params(trial: optuna.Trial) -> dict[str, Any]:
    """Пространство поиска гиперпараметров CatBoost."""
    return {
        "iterations": trial.suggest_int("iterations", 100, 200),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1),
        "depth": trial.suggest_int("depth", 3, 6),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 3.0, 5.0),
        "subsample": trial.suggest_float("subsample", 0.5, 0.8),
        "rsm": trial.suggest_float("rsm", 0.5, 0.8),
    }