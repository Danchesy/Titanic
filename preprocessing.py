from typing import Any

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

__all__ = [
    "FeatureEngineer",
    "build_preprocessor",
    "pipeline_fit_params",
    "preprocessor",
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


def pipeline_fit_params(cat_features: list[str] | None) -> dict[str, Any]:
    """Параметры fit для CatBoost: cat_features нельзя задавать в __init__ (ломает CV clone)."""
    if not cat_features:
        return {}
    return {"model__cat_features": list(cat_features)}
