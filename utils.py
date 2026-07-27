import json
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from config_file import config
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler


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

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> 'FeatureEngineer':
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



def data_loading(
        train_path = config.paths.path_to_train, 
        test_path = config.paths.path_to_test):
    train_data = pd.read_csv(train_path)
    train_data.set_index('PassengerId', inplace=True)

    test_data = pd.read_csv(test_path)
    test_data.set_index('PassengerId', inplace=True)

    train, test = train_data.copy(deep=True), test_data.copy(deep=True)

    X, y = train.drop(columns = config.target_column_name, axis = 1), train[config.target_column_name]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size = config.train_test_split.test_size,
        random_state = config.determenism.random_state,
        shuffle=config.determenism.shuffle,
        stratify=y
    )

    return X_train, X_val, y_train, y_val, test


def data_scaling(
    train: pd.DataFrame,
    test: pd.DataFrame,
    scaler: str = "StandardScaler",
    val: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Выполняет масштабирование признаков для обучающей, тестовой и валидационной выборок.
    
    Args:
        train: Обучающий датафрейм с признаками
        test: Тестовый датафрейм с признаками
        scaler: Тип масштабировщика. Допустимые значения: 'StandardScaler', 'MinMaxScaler'
        val: Необязательный валидационный датафрейм с признаками

    Returns:
        Если val передан: Кортеж из трех DataFrame (train_scaled, val_scaled, test_scaled)
        Если val равен None: Кортеж из двух DataFrame (train_scaled, test_scaled)

    Raises:
        ValueError: Если передан неподдерживаемый тип масштабировщика
    """
    if config.logs.console:
        print(f"Scaling Data with {scaler}...")

    if scaler == "StandardScaler":
        s = StandardScaler()
    elif scaler == "MinMaxScaler":
        s = MinMaxScaler()
    else:
        print(f"Invalid Scaler {scaler}")
        raise ValueError

    # Сохраняем формат Pandas DataFrame на выходе
    s.set_output(transform="pandas")

    train_scaled = s.fit_transform(train)
    test_scaled = s.transform(test)

    if config.logs.console:
        print("Data Scaled Successfully!\n")

    if val is not None:
        val_scaled = s.transform(val)
        return train_scaled, val_scaled, test_scaled

    return train_scaled, test_scaled


def train_val_test_split(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Разделяет исходные данные на три изолированные выборки: обучающую, валидационную и тестовую.

    Разделение происходит в два этапа с обязательной стратификацией по целевой переменной.
    Параметры разделения берутся из config_file.py.

    Args:
        X: Матрица признаков (исходный DataFrame без целевой переменной)
        y: Вектор целевой переменной (Pandas Series)

    Returns:
        Кортеж из шести элементов: X_train, X_val, X_test, y_train, y_val, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.train_test_split.test_size,
        random_state=config.determenism.random_state,
        shuffle=config.determenism.shuffle,
        stratify=y,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=config.train_test_split.val_size,
        random_state=config.determenism.random_state,
        shuffle=config.determenism.shuffle,
        stratify=y_train,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def data_preparation(
    train: pd.DataFrame, 
    test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Выполняет подготовку данных для обучения и тестирования.

    Args:
        train: Обучающий датафрейм
        test: Тестовый датафрейм

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Подготовленные обучающий и тестовый датафреймы

    Raises:
        AssertionError: Если после заполнения остались пропуски в возрасте
    """
    train, test = train.copy(), test.copy()
    for data in [train, test]:
        data["Male"] = np.where(data["Sex"] == "male", 1, 0)

        embarked_mode = data["Embarked"].mode().item()
        data["Embarked"] = data["Embarked"].fillna(embarked_mode)

        # Создаем столбец с титулами
        data["Initial"] = data["Name"].str.extract(r"([A-Za-z]+)\.")

        # Заменяем редкие титулы на Other
        data["Initial"] = np.where(
            data["Initial"].isin(["Mr", "Mrs", "Miss", "Master"]),
            data["Initial"],
            "Other",
        )

    # Dictionary {Initial: Mean age}
    ini_to_age = train.groupby("Initial")["Age"].mean().round().to_dict()

    q_num = 4
    _, bins = pd.qcut(train["Fare"], q=q_num, labels=False, retbins=True)

    # Чтобы билет стоимостью 0.0 попал в категорию
    bins -= 0.001

    for data in [train, test]:
        data["Age"] = data["Age"].fillna(data["Initial"].map(ini_to_age))
        assert data["Age"].isnull().sum().item() == 0

        # Feature generation
        data["Fare_cat"] = pd.cut(
            data["Fare"], bins=bins, labels=False, include_lowest=True
        )

        # Если в тесте будут цены выше тех что были в трейне даем категорию q_num - 1
        data["Fare_cat"] = data["Fare_cat"].fillna(q_num - 1).astype(int)

        data["Alone"] = np.where((data["Parch"] + data["SibSp"]) == 0, 1, 0)
        data["Child"] = np.where(data["Age"] <= 5, 1, 0)

        # Dropping useless features
        data.drop(
            columns=["Sex", "Name", "Ticket", "Cabin", "Fare"], axis=1, inplace=True
        )

        cat_cols = ["Pclass", "Embarked", "Fare_cat", "Initial", "Male"]
        data[cat_cols] = data[cat_cols].astype("category")

    return train, test


def preprocessor(is_scale: bool = config.scaling.is_scale, is_cat: bool = True) -> Pipeline:
    """
    Создает пайплайн предобработки данных с возможностью масштабирования и кодирования.

    Args:
        is_scale: Флаг, определяющий необходимость масштабирования числовых признаков
        is_cat: Флаг, определяющий необходимость кодирования категориальных признаков

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
        if config.scaling.scaler == "StandardScaler":
            scaler = StandardScaler()
        elif config.scaling.scaler == "MinMaxScaler":
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unsupported scaler: {config.scaling.scaler}")

        transformers.append(("num", scaler, num_columns))

    if not transformers:
        return Pipeline([("feature_engineering", FeatureEngineer())])

    cols_trans = ColumnTransformer(transformers, remainder="passthrough")

    pipeline = Pipeline([
        ("feature_engineering", FeatureEngineer()), 
        ("cols_transformer", cols_trans)]
    )

    return pipeline

def model_return(
    model: LogisticRegression, 
    accuracies: float | list[float] | np.ndarray
) -> dict[str, Any]:
    """
    Формирует словарь с результатами модели.

    Args:
        model: Обученная модель
        accuracies: Значение точности или массив значений точности

    Returns:
        Dict[str, Any]: Словарь с результатами
    """
    if not isinstance(accuracies, (list, np.ndarray)):
        accuracies = [accuracies]

    return {
        "model": model,
        "mean_score": np.mean(accuracies),
        "std_score": np.std(accuracies),
        "params": model.get_params(),
    }


def pipeline_return(
    pipeline: Pipeline, 
    cv_scores: list[float] | np.ndarray
) -> dict[str, Any]:
    """
    Формирует словарь с результатами пайплайна.

    Args:
        pipeline: Обученный пайплайн
        cv_scores: Массив оценок кросс-валидации

    Returns:
        Dict[str, Any]: Словарь с результатами
    """
    return {
        "pipeline": pipeline,
        "mean_score": np.mean(cv_scores),
        "std_score": np.std(cv_scores),
        "params": pipeline.named_steps["model"].get_params(),
    }


def add_result(
    output: dict[str, Any], 
    results: list[dict[str]] | list[dict[Any]] | None = None, 
    log_file_path: str | None = None
) -> None:
    """
    Добавляет результат в список и дописывает его в файл на диске.

    Args:
        output: Словарь-результат эксперимента
        results: Глобальный список для хранения истории в текущей сессии
        log_file_path: Путь к файлу на диске, куда будут дописываться логи
    """
    log_file_path = (
        log_file_path if log_file_path is not None else config.paths.experiments_log
    )

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
    }

    if results is not None:
        results.append(experiment_data)

    # Дописываем в файл ('a' — append)
    # JSON Lines (один эксперимент — одна строчка в файле)
    with open(log_file_path, mode="a", encoding="utf-8") as f:
        # json.dumps превращает словарь в одну текстовую строку
        f.write(json.dumps(experiment_data, ensure_ascii=False) + "\n")


def plot_feature_importance(
    model: Any, 
    feature_names: list[str], 
    top_n: int = 20
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

