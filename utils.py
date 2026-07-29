import json
import os
import random
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

__all__ = [
    "FeatureEngineer",
    "WandbLogger",
    "add_result",
    "data_loading",
    "pipeline_return",
    "plot_feature_importance",
    "preprocessor",
    "set_seed"
]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Класс для инженерии признаков в датасете Titanic.

    Выполняет следующие преобразования:
    - Извлечение титулов из имен пассажиров
    - Заполнение пропусков в возрасте средними значениями по титулам
    - Создание категорий на основе стоимости билета
    - Создание бинарных признаков (Male, Alone, Child)
    - Удаление неинформативных признаков

    Attributes:
        q_num (int): Количество категорий для дискретизации стоимости билета
        embarked_mode_ (str): Модальное значение порта посадки
        ini_to_age_ (Dict[str, float]): Словарь соответствия титул -> средний возраст
        bins_ (np.ndarray): Границы категорий для Fare_cat
        cat_cols (List[str]): Список категориальных колонок
    """

    def __init__(self, q_num: int = 4):
        """
        Инициализация FeatureEngineer.

        Args:
            q_num: Количество категорий для дискретизации стоимости билета
        """
        self.q_num: int = q_num
        self.embarked_mode_: str = "S"
        self.ini_to_age_: dict[str, float] = {}
        self.bins_: np.ndarray | None = None
        self.cat_cols: list[str] = ["Pclass", "Embarked", "Fare_cat", "Initial", "Male"]

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FeatureEngineer":
        """
        Обучает трансформер на основе обучающих данных.

        Вычисляет:
        - Модальное значение порта посадки
        - Средний возраст для каждого титула
        - Границы категорий для дискретизации стоимости билета

        Args:
            X: Обучающий датафрейм с признаками
            y: Целевая переменная (не используется)

        Returns:
            self: Возвращает экземпляр класса для цепочки вызовов
        """
        X = X.copy()

        self.embarked_mode_ = X["Embarked"].mode().item()

        X["Initial"] = X["Name"].str.extract(r"([A-Za-z]+)\.")
        X["Initial"] = np.where(
            X["Initial"].isin(["Mr", "Mrs", "Miss", "Master"]), X["Initial"], "Other"
        )

        self.ini_to_age_ = X.groupby("Initial")["Age"].mean().round().to_dict()

        _, bins = pd.qcut(X["Fare"], q=self.q_num, labels=False, retbins=True)
        bins -= 0.001
        self.bins_ = bins

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Преобразует данные, применяя инженерию признаков.

        Args:
            X: Датафрейм для преобразования

        Returns:
            pd.DataFrame: Преобразованный датафрейм с новыми признаками

        Raises:
            AssertionError: Если после заполнения остались пропуски в возрасте
        """
        X = X.copy()

        # Feature engineering
        X["Male"] = np.where(X["Sex"] == "male", 1, 0)
        X["Embarked"] = X["Embarked"].fillna(self.embarked_mode_)

        X["Initial"] = X["Name"].str.extract(r"([A-Za-z]+)\.")
        X["Initial"] = np.where(
            X["Initial"].isin(["Mr", "Mrs", "Miss", "Master"]), X["Initial"], "Other"
        )

        X["Age"] = X["Age"].fillna(X["Initial"].map(self.ini_to_age_))
        assert X["Age"].isnull().sum().item() == 0

        X["Fare_cat"] = pd.cut(
            X["Fare"], bins=self.bins_, labels=False, include_lowest=True
        )
        X["Fare_cat"] = X["Fare_cat"].fillna(self.q_num - 1).astype(int)

        X["Alone"] = np.where((X["Parch"] + X["SibSp"]) == 0, 1, 0)
        X["Child"] = np.where(X["Age"] <= 5, 1, 0)

        # Drop useless features
        X.drop(
            columns=["Sex", "Name", "Ticket", "Cabin", "Fare"],
            axis=1,
            inplace=True,
            errors="ignore",
        )

        # Change features to categorical type
        X[self.cat_cols] = X[self.cat_cols].astype("category")

        return X


class WandbLogger:
    """Логирование экспериментов в Weights & Biases."""

    def __init__(self, cfg: DictConfig) -> None:
        self.enabled = cfg.logging.wandb.enabled

        if not self.enabled:
            return

        wandb.init(
            project=cfg.logging.wandb.project,
            entity=cfg.logging.wandb.entity,
            tags=cfg.logging.wandb.tags,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    def log_experiment(self, experiment: dict[str, Any]) -> None:
        """Логирует результаты одного эксперимента."""
        if not self.enabled:
            return

        log_data = {
            "accuracy": experiment["accuracy"],
            "std": experiment["std"],
            "tuning_time_sec": experiment["tuning_time_sec"],
            "predict_time_sec": experiment["predict_time_sec"],
            "latency_ms_per_sample": experiment["latency_ms_per_sample"],
        }

        for key, value in experiment["params"].items():
            log_data[f"param/{key}"] = value

        wandb.log(log_data)

    def log_pipeline(self, model_path: str) -> None:
        """Сохраняет обученную модель."""
        if not self.enabled:
            return

        artifact = wandb.Artifact(
            name="best_model",
            type="model",
        )

        artifact.add_file(model_path)

        wandb.log_artifact(artifact)

    def finish(self) -> None:
        """Завершает текущий run."""
        if self.enabled:
            wandb.finish()


def set_seed(seed: int = 42) -> None:
    """Фиксирует seed для всех используемых библиотек (Python, NumPy, PyTorch)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def data_loading(
    cfg: DictConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Загружает и разделяет данные."""
    train = pd.read_csv(cfg.data.train_path)
    test = pd.read_csv(cfg.data.test_path)

    train.set_index("PassengerId", inplace=True)
    test.set_index("PassengerId", inplace=True)

    X = train.drop(columns=cfg.target_column)
    y = train[cfg.target_column]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=cfg.training.test_size,
        random_state=cfg.training.random_state,
        shuffle=cfg.training.shuffle,
        stratify=y,
    )

    return X_train, X_val, y_train, y_val, test


def preprocessor(
    is_scale: bool = True,
    is_cat: bool = True,
    scaler_type: str = "StandardScaler",
) -> Pipeline:
    """
    Создает пайплайн предобработки данных с возможностью масштабирования и кодирования.

    Args:
        is_scale: Флаг, определяющий необходимость масштабирования числовых признаков
        is_cat: Флаг, определяющий необходимость кодирования категориальных признаков
        scaler_type: Тип скейлера ("StandardScaler" или "MinMaxScaler")

    Returns:
        Pipeline: Пайплайн предобработки данных
    """
    cat_columns = ["Embarked", "Initial", "Fare_cat", "Pclass"]
    num_columns = ["Age", "SibSp", "Parch"]
    transformers = []

    if not is_cat and not is_scale:
        return FeatureEngineer()  # type: ignore

    if is_cat:
        transformers.append(
            (
                "cat",
                OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                cat_columns,
            )
        )

    if is_scale:
        if scaler_type == "StandardScaler":
            scaler = StandardScaler()
        elif scaler_type == "MinMaxScaler":
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unsupported scaler: {scaler_type}")

        transformers.append(("num", scaler, num_columns))

    if not transformers:
        return Pipeline([("feature_engineering", FeatureEngineer())])

    cols_trans = ColumnTransformer(transformers, remainder="passthrough")

    pipeline = Pipeline(
        [("feature_engineering", FeatureEngineer()), ("cols_transformer", cols_trans)]
    )

    return pipeline


def pipeline_return(
    pipeline: Pipeline,
    cv_scores: list[float] | np.ndarray,
    tuning_time: float | None = None,
    predict_time: float | None = None,
    n_samples: int | None = None,
) -> dict[str, Any]:
    """Формирует словарь с результатами пайплайна, включая метрики времени.

    Args:
        pipeline: Обученный пайплайн cv_scores: Массив оценок кросс-валидации или
        среднее значение tuning_time: Время подбора параметров (GridSearch /
        Optuna) в секундах predict_time: Общее время предсказания на тестовой
        выборке в секундах n_samples: Количество объектов в тестовой выборке для
        расчета latency

    Returns:
        Dict[str, Any]: Словарь с результатами и временными метриками
    """
    result = {
        "pipeline": pipeline,
        "mean_score": np.mean(cv_scores),
        "std_score": np.std(cv_scores),
        "params": pipeline.named_steps["model"].get_params(),
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


def add_result(
    output: dict[str, Any],
    results: list[dict[str]] | list[dict[Any]] | None = None,
    log_file_path: str | None = None,
) -> dict[str, Any]:
    """
    Добавляет результат в список и дописывает его в файл на диске.

    Args:
        output: Словарь-результат эксперимента
        results: Глобальный список для хранения истории в текущей сессии
        log_file_path: Путь к файлу на диске, куда будут дописываться логи

    Returns:
        Dict[str, Any]: Словарь с данными эксперимента
    """
    if "model" in output:
        model_name = type(output["model"]).__name__
    elif "pipeline" in output:
        model_name = type(output["pipeline"].named_steps["model"]).__name__
    else:
        model_name = "UnknownModel"

    raw_params = output.get("params", {})

    experiment_data = {
        "model": model_name,
        "accuracy": float(output.get("mean_score")),
        "std": float(output.get("std_score")),
        "params": {str(k): str(v) for k, v in raw_params.items()},
        "tuning_time_sec": output.get("tuning_time_sec"),
        "predict_time_sec": output.get("predict_time_sec"),
        "latency_ms_per_sample": output.get("latency_ms_per_sample"),
    }

    if results is not None:
        results.append(experiment_data)

    # Дописываем в файл ('a' — append)
    # JSON Lines (один эксперимент — одна строчка в файле)
    if log_file_path:
        with open(log_file_path, mode="a", encoding="utf-8") as f:
            f.write(json.dumps(experiment_data, ensure_ascii=False) + "\n")

    return experiment_data


def plot_feature_importance(
    model: Any,
    feature_names: list[str],
    top_n: int = 20,
) -> None:
    """
    Визуализирует важность признаков для моделей с атрибутом feature_importances_.

    Args:
        model: Обученная модель с атрибутом feature_importances_
        feature_names: Список названий признаков
        top_n: Количество наиболее важных признаков для отображения
    """
    importances = model.feature_importances_

    fi_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": importances}
    ).sort_values(by="Importance", ascending=False)

    fi_df = fi_df.head(top_n)

    plt.figure(figsize=(10, max(6, top_n * 0.4)))
    sns.barplot(
        x="Importance",
        y="Feature",
        data=fi_df,
        palette="viridis",
        legend=False,
    )

    plt.title(
        f"Top {top_n} Feature Importances ({model.__class__.__name__})",
        fontsize=14,
    )
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.show()


def generate_submission(
    pipeline_path: str = "models/full_pipeline.pkl",
    output_path: str = "data/submission.csv",
) -> None:
    """Генерирует файл сабмита из сохранённого пайплайна."""
    pipeline = joblib.load(pipeline_path)

    *_, test = data_loading()

    preds = pipeline.predict(test)

    submission = pd.DataFrame({
        'PassengerId': test.index,
        'Survived': preds.astype(int)
    })

    submission.to_csv(output_path, index=False)
    print(f"Submit created successfully and saved as {output_path}")