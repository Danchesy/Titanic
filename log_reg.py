from typing import Any

import numpy as np
import pandas as pd
from config_file import config
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from utils import add_result, pipeline_return, preprocessor


def logistic_kfold(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    penalty: str | None = None,
    C: float | None = None,
    l1_ratio: float | None = None,
    solver: str | None = None,
    tol: float | None = None,
    fit_intercept: bool | None = None,
    random_state: int | None = None,
    max_iter: int | None = None,
    verbose: int | None = None,
    n_splits: int | None = None,
    shuffle: bool | None = None,
) -> dict[str, Any]:
    """
    Выполняет кросс-валидацию логистической регрессии с предобработкой данных.

    Args:
        X_train: Обучающие признаки
        y_train: Целевая переменная
        penalty: Тип регуляризации ('l1', 'l2', 'elasticnet', None)
        C: Параметр регуляризации (обратная сила регуляризации)
        l1_ratio: Коэффициент смешивания для elasticnet
        solver: Алгоритм оптимизации
        tol: Толерантность остановки
        fit_intercept: Добавлять ли свободный член
        random_state: Seed для воспроизводимости
        max_iter: Максимальное количество итераций
        verbose: Уровень детализации
        n_splits: Количество фолдов для кросс-валидации

    Returns:
        Dict[str, Any]: Словарь с результатами, содержащий:
            - model: Обученная модель
            - fold_scores: Массив оценок по фолдам
            - mean_score: Средняя точность
            - std_score: Стандартное отклонение точности
            - params: Параметры модели
    """
    penalty = penalty if penalty is not None else config.linear_model.penalty
    C = C if C is not None else config.linear_model.C
    l1_ratio = l1_ratio if l1_ratio is not None else config.linear_model.l1_ratio
    solver = solver if solver is not None else config.linear_model.solver
    tol = tol if tol is not None else config.linear_model.tol
    fit_intercept = (
        fit_intercept if fit_intercept is not None else config.linear_model.bias
    )
    random_state = (
        random_state if random_state is not None else config.determenism.random_state
    )
    max_iter = max_iter if max_iter is not None else config.linear_model.max_iter
    verbose = verbose if verbose is not None else config.linear_model.verbose
    n_splits = n_splits if n_splits is not None else config.cv.n_splits
    shuffle = shuffle if shuffle is not None else config.determenism.shuffle

    skf = StratifiedKFold(
        n_splits=n_splits, random_state=random_state, shuffle=shuffle
    )

    model = LogisticRegression(
        penalty=penalty,
        C=C,
        l1_ratio=l1_ratio,
        solver=solver,
        tol=tol,
        fit_intercept=fit_intercept,
        random_state=random_state,
        max_iter=max_iter,
        verbose=verbose,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor(is_scale=config.scaling.is_scale, is_cat=config.is_cat)), 
        ("model", model)])

    scores = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring=config.linear_model.scoring)

    if config.logs.console:
        for fold, acc in enumerate(scores, start=1):
            print(f"fold-{fold} accuracy: {acc}")


    pipeline.fit(X_train, y_train)

    model = pipeline.named_steps["model"]

    if config.logs.console:
        print(f"{model} mean accuracy: {np.mean(scores)}")

    res = pipeline_return(pipeline, scores)
    add_result(res)

    return {
        "model": model,
        "fold_scores": scores,
        "mean_score": np.mean(scores),
        "std_score": np.std(scores),
        "params": model.get_params(),
    }


def log_reg_cv(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    X_test: pd.DataFrame, 
    y_test: pd.Series,
    tol: float | None = None,
    fit_intercept: bool | None = None,
    random_state: int | None = None,
    max_iter: int | None = None,
    solver: str | None = None,
    cv: int | Any = None,
    refit: bool | None = None,
    Cs: int | list[float] | np.ndarray = None,
    verbose: int | None = None,
    scoring: str | None = None,
) -> dict[str, Any]:
    """
    Выполняет поиск гиперпараметров логистической регрессии с помощью встроенной кросс-валидации.

    Функция строит конвейер (Pipeline) с предобработкой данных, запускает 
    оптимизацию параметра обратной силы регуляризации `C` по заданной сетке 
    значений `Cs` с использованием встроенного класса `LogisticRegressionCV`. 
    После поиска лучшая модель дообучается на всей тренировочной выборке 
    и оценивается на тестовых данных. Результаты сохраняются в лог.

    Args:
        X_train: Матрица признаков обучающей выборки.
        y_train: Вектор целевой переменной для обучения.
        X_test: Матрица признаков тестовой выборки для финального контроля.
        y_test: Вектор целевой переменной тестовой выборки.
        tol: Толерантность (порог) остановки алгоритма оптимизации.
        fit_intercept: Флаг добавления константы (свободного члена) в модель.
        random_state: Зерно генератора случайных чисел для воспроизводимости.
        max_iter: Максимальное количество итераций сходимости алгоритма.
        solver: Алгоритм оптимизации (например, 'lbfgs', 'saga', 'liblinear').
        cv: Стратегия кросс-валидации (число фолдов или генератор разбиений).
        refit: Если True, дообучает модель на всех тренировочных данных с лучшим C.
        Cs: Описание сетки значений параметра C (число шагов или массив значений).
        verbose: Уровень детализации вывода системных логов обучения.
        scoring: Строковое имя метрики для оптимизации (например, 'accuracy').

    Returns:
        Dict[str, Any]: Словарь с ключевыми результатами эксперимента:
            - 'model': Обученный объект модели LogisticRegressionCV из пайплайна.
            - 'best_C': Оптимальное найденное значение параметра регуляризации C.
            - 'best_cv_score': Лучшая средняя точность на внутренней кросс-валидации.
            - 'test_score': Финальная точность модели на отложенной тестовой выборке.
    """
    tol = tol if tol is not None else config.linear_model.tol
    fit_intercept = fit_intercept if fit_intercept is not None else config.linear_model.bias
    random_state = random_state if random_state is not None else config.determenism.random_state
    max_iter = max_iter if max_iter is not None else config.linear_model.max_iter
    solver = solver if solver is not None else config.linear_model.solver
    cv = cv if cv is not None else config.linear_model.cv
    refit = refit if refit is not None else config.linear_model.refit
    Cs = Cs if Cs is not None else config.linear_model.Cs
    verbose = verbose if verbose is not None else config.linear_model.verbose
    scoring = scoring if scoring is not None else config.linear_model.scoring


    model_cv = LogisticRegressionCV(
        tol=tol,
        fit_intercept=fit_intercept,
        random_state=random_state,
        max_iter=max_iter,
        solver=solver,
        cv=cv,
        refit=refit, 
        Cs=Cs,
        verbose=verbose,
        scoring=scoring
    )

    pipeline_cv = Pipeline([
            ('preprocessor', preprocessor(is_scale=True, is_cat=config.is_cat)),
            ('model', model_cv)
        ])

    pipeline_cv.fit(X_train, y_train)

    best_model = pipeline_cv.named_steps['model']
    best_C = best_model.C_.item()

    scores = best_model.scores_

    test_acc = pipeline_cv.score(X_test, y_test)

    label = next(iter(scores.keys()))
    mean_cv_per_c = np.mean(scores[label], axis=0)
    best_cv_score = np.max(mean_cv_per_c).item()

    print(f"best C: {best_C}")
    print(f"mean acc for best C: {best_cv_score}")
    print(f"test accuracy: {test_acc}")

    res = pipeline_return(pipeline_cv, best_cv_score)
    add_result(res)