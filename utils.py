import json
import os
import random
import time
from functools import wraps
from typing import Any

import hydra
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
from sklearn.metrics import get_scorer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

__all__ = [
    "FeatureEngineer",
    "WandbLogger",
    "add_result",
    "build_preprocessor",
    "data_loading",
    "ensure_dirs",
    "generate_submission",
    "holdout_score",
    "log",
    "model_filename",
    "pipeline_fit_params",
    "pipeline_return",
    "plot_feature_importance",
    "preprocessor",
    "run_method",
    "save_submission",
    "set_seed",
    "submission_output_path",
    "time_and_score",
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

    def __init__(self, q_num: int = 4, drop_columns: list[str] | None = None):
        """
        Инициализация FeatureEngineer.

        Args:
            q_num: Количество категорий для дискретизации стоимости билета
            drop_columns: Дополнительные колонки для удаления после FE
        """
        self.q_num: int = q_num
        self.embarked_mode_: str = "S"
        self.ini_to_age_: dict[str, float] = {}
        self.bins_: np.ndarray | None = None
        self.cat_cols: list[str] = ["Pclass", "Embarked", "Fare_cat", "Initial", "Male"]
        self.drop_columns: list[str] = (
            drop_columns
            if drop_columns is not None
            else ["PassengerId", "Name", "Ticket", "Cabin"]
        )

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
        cols_to_drop = list(
            set(self.drop_columns + ["Sex", "Fare", "Name", "Ticket", "Cabin"])
        )
        X.drop(
            columns=cols_to_drop,
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
            name=cfg.experiment_name,
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

def time_and_score(stage='train'):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):

            final_stage = kwargs.pop('stage', stage)

            start_train = time.time()
            result = func(*args, **kwargs)
            timer = time.time() - start_train

            return {
                "result": result,
                f"{final_stage}_time_sec": timer
            }
        return wrapper
    return decorator


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
    scaler: BaseEstimator | None = None,
    encoder: BaseEstimator | None = None,
    drop_columns: list[str] | None = None,
    q_num: int = 4,
) -> Pipeline:
    """
    Создает пайплайн предобработки данных с возможностью масштабирования и кодирования.

    Args:
        is_scale: Флаг, определяющий необходимость масштабирования числовых признаков
        is_cat: Флаг, определяющий необходимость кодирования категориальных признаков
        scaler: Экземпляр sklearn-скейлера (StandardScaler, MinMaxScaler и т.д.)
        encoder: Экземпляр sklearn-энкодера (OneHotEncoder, OrdinalEncoder и т.д.)
        drop_columns: Колонки для удаления в FeatureEngineer
        q_num: Число категорий для дискретизации Fare

    Returns:
        Pipeline: Пайплайн предобработки данных
    """
    cat_columns = ["Embarked", "Initial", "Fare_cat", "Pclass"]
    num_columns = ["Age", "SibSp", "Parch"]
    transformers = []

    feature_engineer = FeatureEngineer(q_num=q_num, drop_columns=drop_columns)

    if not is_cat and not is_scale:
        return feature_engineer  # type: ignore[return-value]

    if is_cat:
        if encoder is None:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        transformers.append(("cat", encoder, cat_columns))

    if is_scale:
        if scaler is None:
            scaler = StandardScaler()
        transformers.append(("num", scaler, num_columns))

    if not transformers:
        return Pipeline([("feature_engineering", feature_engineer)])

    cols_trans = ColumnTransformer(transformers, remainder="passthrough")

    pipeline = Pipeline(
        [("feature_engineering", feature_engineer), ("cols_transformer", cols_trans)]
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


def build_preprocessor(
    cfg: DictConfig,
    model_cfg: DictConfig,
    is_scale: bool,
    is_cat: bool,
) -> Pipeline:
    """Создаёт preprocessor с параметрами из конфига модели или глобального preprocessing."""
    drop_columns = OmegaConf.to_container(cfg.preprocessing.drop_columns, resolve=True)

    scaler = None
    encoder = None

    if is_scale:
        scaler_cfg = model_cfg.get("scaler") or cfg.preprocessing.get("scaler")
        if scaler_cfg is None:
            raise ValueError("scaler config is required when is_scale=True")
        scaler = hydra.utils.instantiate(scaler_cfg)

    if is_cat:
        encoder_cfg = model_cfg.get("encoder") or cfg.preprocessing.get("encoder")
        if encoder_cfg is None:
            raise ValueError("encoder config is required when is_cat=True")
        encoder = hydra.utils.instantiate(encoder_cfg)

    return preprocessor(
        is_scale=is_scale,
        is_cat=is_cat,
        scaler=scaler,
        encoder=encoder,
        drop_columns=list(drop_columns),
        q_num=cfg.preprocessing.q_num,
    )


def save_submission(
    pipeline: Pipeline,
    X_submit: pd.DataFrame,
    submission_path: str,
) -> None:
    """Сохраняет предсказания в CSV для Kaggle."""
    os.makedirs(os.path.dirname(submission_path) or ".", exist_ok=True)
    preds = pipeline.predict(X_submit)
    submission = pd.DataFrame(
        {"PassengerId": X_submit.index, "Survived": preds.astype(int)}
    )
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved: {submission_path}")


def plot_feature_importance(
    model: Any,
    feature_names: list[str],
    top_n: int = 20,
    save_path: str | None = None,
) -> None:
    """
    Визуализирует важность признаков для моделей с атрибутом feature_importances_.

    Args:
        model: Обученная модель с атрибутом feature_importances_
        feature_names: Список названий признаков
        top_n: Количество наиболее важных признаков для отображения
        save_path: Путь для сохранения графика (если None — только show)
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
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Feature importance plot saved: {save_path}")
    plt.show()


def generate_submission(
    cfg: DictConfig,
    pipeline_path: str | None = None,
    output_path: str | None = None,
) -> None:
    """Генерирует файл сабмита из сохранённого пайплайна."""
    pipeline_path = pipeline_path or f"{cfg.data.models_dir}/full_pipeline.pkl"
    output_path = output_path or os.path.join(cfg.data.results_dir, "submission.csv")

    pipeline = joblib.load(pipeline_path)
    *_, test = data_loading(cfg)
    save_submission(pipeline, test, output_path)


def log(message: str, console: bool) -> None:
    if console:
        print(message)


def pipeline_fit_params(cat_features: list[str] | None) -> dict[str, Any]:
    """Параметры fit для CatBoost: cat_features нельзя задавать в __init__ (ломает CV clone)."""
    if not cat_features:
        return {}
    return {"model__cat_features": list(cat_features)}


def ensure_dirs(cfg: DictConfig) -> None:
    os.makedirs(cfg.data.models_dir, exist_ok=True)
    os.makedirs(cfg.data.results_dir, exist_ok=True)
    # os.makedirs(cfg.data.reports_dir, exist_ok=True)


def model_filename(
    cfg: DictConfig,
    model_name: str,
    method: str,
    score: float,
    extension: str = 'pkl',
) -> str:
    prefix = f"{cfg.experiment_name}_" if cfg.get("experiment_name") else ""
    return os.path.join(
        cfg.data.models_dir,
        f"{prefix}{model_name}_{method}_{score:.4f}.{extension}",
    )


@time_and_score(stage='predict')
def holdout_score(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, metric: str) -> float:
    scorer = get_scorer(metric)
    return float(scorer(pipeline, X, y))


@time_and_score()
def run_method(obj, method_name, *args, **kwargs):
    method = getattr(obj, method_name) if obj is not None else globals()[method_name]
    res = method(*args, **kwargs)

    return res


def submission_output_path(cfg: DictConfig, model_name: str) -> str:
    submit_dir = os.path.dirname(cfg.data.submission_path) or cfg.data.results_dir
    return os.path.join(submit_dir, f"{model_name}_submission.csv")
