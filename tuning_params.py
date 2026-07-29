import os
import time
from collections.abc import Callable
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline

from utils import add_result, pipeline_return, preprocessor

__all__ = [
    "catboost_grid_params",
    "catboost_optuna_params",
    "dt_grid_params",
    "dt_optuna_params",
    "grid_tuning",
    "knn_grid_params",
    "knn_optuna_params",
    "lgbm_grid_params",
    "lgbm_optuna_params",
    "logreg_optuna_params",
    "optuna_tuning",
    "rf_grid_params",
    "rf_optuna_params",
    "xgb_grid_params",
    "xgb_optuna_params",
]


def _pipeline_fit_params(cat_features: list[str] | None) -> dict[str, Any]:
    """Параметры fit для CatBoost: cat_features нельзя задавать в __init__ (ломает CV clone)."""
    if not cat_features:
        return {}
    return {"model__cat_features": list(cat_features)}


def grid_tuning(
    model: Any,
    params: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    is_scale: bool = True,
    is_cat: bool = True,
    cat_features: list[str] | None = None,
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

    os.makedirs("models", exist_ok=True)

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor(is_scale=is_scale, is_cat=is_cat)),
            ("model", model),
        ]
    )

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=params,
        scoring="accuracy",
        refit=True,
        cv=5,
        n_jobs=-1,
        verbose=1,
        pre_dispatch="2*n_jobs",
        return_train_score=False,
    )

    start_train = time.time()
    grid_search.fit(X_train, y_train, **_pipeline_fit_params(cat_features))
    train_time = time.time() - start_train

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV Accuracy: {grid_search.best_score_:.4f}")
    print(f"GridSearch trainig time: {train_time:.2f} s.")

    best_pipeline = grid_search.best_estimator_

    start_predict = time.time()
    test_preds = best_pipeline.predict(X_test)
    predict_time = time.time() - start_predict

    final_acc = accuracy_score(y_test, test_preds)

    print(f"X_test Accuracy: {final_acc:.4f}")
    print(f"X_test time predict ({len(X_test)} lines): {predict_time:.4f} s.")
    print(f"Latency: {(predict_time / len(X_test)) * 1000:.4f} ms")

    res = pipeline_return(
        best_pipeline,
        grid_search.best_score_,
        tuning_time=train_time,
        predict_time=predict_time,
        n_samples=len(X_test),
    )
    experiment = add_result(res)

    if logger is not None:
        logger.log_experiment(experiment)

    model_name = model.__class__.__name__
    filename = f"models/{model_name}_grid_acc_{grid_search.best_score_:.4f}.pkl"
    joblib.dump(best_pipeline, filename)

    if logger is not None:
        logger.log_pipeline(filename)

    print(f"Pipeline saved as: {filename}")

    return grid_search


def optuna_tuning(
    model: Any,
    params_fn: Callable[[optuna.Trial], dict[str, Any]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    direction: str = "maximize",
    n_trials: int = 20,
    is_scale: bool = True,
    is_cat: bool = True,
    cat_features: list[str] | None = None,
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

    os.makedirs("models", exist_ok=True)

    cv_params = _pipeline_fit_params(cat_features)

    def objective(trial: optuna.Trial) -> float:
        params = params_fn(trial)

        current_model = clone(model)
        current_model.set_params(**params)

        pipeline = Pipeline(
            [
                ("preprocessor", preprocessor(is_scale=is_scale, is_cat=is_cat)),
                ("model", current_model),
            ]
        )

        accuracy = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=5,
            scoring="accuracy",
            params=cv_params,
        ).mean()
        return accuracy

    start_optuna = time.time()
    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials)
    optuna_time = time.time() - start_optuna

    best_params = study.best_trial.user_attrs.get("sklearn_params", study.best_params)
    print(f"Best CV Accuracy: {study.best_value:.4f}")
    print(f"Best parameters: {best_params}")
    print(f"Optuna optimization time ({n_trials} trials): {optuna_time:.2f} s.")
    print(f"Mean time per trial: {optuna_time / n_trials:.2f} s.")

    best_model = clone(model)
    best_model.set_params(**best_params)

    final_pipeline = Pipeline(
        [
            ("preprocessor", preprocessor(is_scale, is_cat)),
            ("model", best_model),
        ]
    )

    start_train = time.time()
    final_pipeline.fit(X_train, y_train, **_pipeline_fit_params(cat_features))
    train_time = time.time() - start_train

    start_predict = time.time()
    pred = final_pipeline.predict(X_test)
    predict_time = time.time() - start_predict

    test_accuracy = accuracy_score(y_test, pred)

    print(f"X_test Accuracy: {test_accuracy:.4f}")
    print(f"Final pipeline's training: {train_time:.4f} s.")
    print(f"X_test predictions ({len(X_test)} lines): {predict_time:.4f} s.")

    res = pipeline_return(
        final_pipeline,
        study.best_value,
        tuning_time=optuna_time,
        predict_time=predict_time,
        n_samples=len(X_test),
    )

    experiment = add_result(res)

    if logger is not None:
        logger.log_experiment(experiment)

    model_name = model.__class__.__name__
    filename = f"models/{model_name}_optuna_acc_{study.best_value:.4f}.pkl"
    joblib.dump(final_pipeline, filename)

    if logger is not None:
        logger.log_pipeline(filename)

    print(f"Pipeline saved as: {filename}")

    return study


DEFAULT_SOLVER_MAPPING = {
    "liblinear": ["l1", "l2"],
    "saga": ["l1", "l2", "elasticnet"],
    "lbfgs": ["l2"],
}


def logreg_optuna_params(
    trial: optuna.Trial,
    solver_mapping: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Пространство поиска гиперпараметров LogisticRegression."""
    mapping = solver_mapping or DEFAULT_SOLVER_MAPPING
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


logreg_grid_params = {
    "model__C": [0.01, 0.1, 1.0, 10.0],
    "model__penalty": ["l1", "l2"],
    "model__solver": ["liblinear"]
}


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


knn_grid_params = {
    "model__n_neighbors": np.arange(3, 18, 2),
    "model__weights": ["distance", "uniform"],
    "model__leaf_size": [20, 30, 50],
    "model__metric": ["cosine", "manhattan", "euclidean"],
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


dt_grid_params = {
    "model__max_depth": [3, 5, 7],
    "model__min_samples_split": [5, 20],
    "model__min_samples_leaf": [2, 10],
    "model__max_features": [None, "sqrt", "log2"],
    "model__max_leaf_nodes": [None, 25, 50],
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


rf_grid_params = {
    "model__n_estimators": [100, 300, 500],
    "model__max_depth": [3, 5, 7],
    "model__min_samples_split": [5, 20],
    "model__min_samples_leaf": [2, 10],
    "model__max_features": ["sqrt", "log2"],
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


xgb_grid_params = {
    "model__max_depth": np.arange(3, 6),
    "model__n_estimators": np.linspace(100, 200, 3, dtype=int),
    "model__learning_rate": np.logspace(np.log10(0.01), np.log10(0.1), 3),
    "model__reg_lambda": np.logspace(np.log10(0.5), np.log10(5.0), 3),
    "model__reg_alpha": np.logspace(np.log10(0.01), np.log10(2.0), 3),
    "model__min_child_weight": np.linspace(1, 10, 3, dtype=int),
    "model__subsample": np.arange(0.6, 0.81, 0.05),
    "model__colsample_bytree": np.arange(0.6, 0.81, 0.05),
    "model__gamma": np.arange(0.0, 0.31, 0.05),
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


lgbm_grid_params = {
    "model__max_depth": np.arange(3, 6),
    "model__n_estimators": np.linspace(100, 200, 3, dtype=int),
    "model__learning_rate": np.logspace(np.log10(0.01), np.log10(0.1), 3),
    "model__reg_lambda": np.logspace(np.log10(0.5), np.log10(5.0), 3),
    "model__reg_alpha": np.logspace(np.log10(0.01), np.log10(2.0), 3),
    "model__min_child_weight": np.linspace(1, 10, 3, dtype=int),
    "model__subsample": np.arange(0.6, 0.81, 0.05),
    "model__colsample_bytree": np.arange(0.6, 0.81, 0.05),
    "model__min_split_gain": np.arange(0.0, 0.31, 0.05),
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


catboost_grid_params = {
    "model__iterations": np.linspace(100, 200, 3, dtype=int),
    "model__learning_rate": np.logspace(np.log10(0.001), np.log10(0.1), 3),
    "model__depth": np.arange(3, 7),
    "model__l2_leaf_reg": np.linspace(3.0, 5.0, 3),
    "model__subsample": np.linspace(0.5, 0.8, 3),
    "model__rsm": np.linspace(0.5, 0.8, 3),
}