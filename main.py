# main.py
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from config_file import config
from log_reg import log_reg_cv, logistic_kfold
from tuning_params import *
from utils import *

warnings.filterwarnings('ignore')


logger = WandbLogger(config)

def run_logistic_experiments(X_train, y_train, X_test, y_test, logger):
    """Логистическая регрессия."""
    print("\nЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ")

    print("\nБазовый запуск")
    logistic_kfold(X_train, y_train, logger=logger)

    print("\nLogisticRegressionCV")
    log_reg_cv(X_train, y_train, X_test, y_test, logger=logger)

    print("\nПеребор параметров")
    l1_ratios = np.arange(0.25, 0.76, 0.25)
    Cs = np.logspace(-4, 1, 6)

    for penalty in ['l1', 'l2', 'elasticnet']:
        solver = "liblinear" if penalty == "l1" else "saga" if penalty == "elasticnet" else "lbfgs"

        for c in Cs:
            if penalty == "elasticnet":
                for l1_ratio in l1_ratios:
                    logistic_kfold(
                        X_train, y_train,
                        penalty=penalty, solver=solver,
                        l1_ratio=l1_ratio, C=c
                    )
            else:
                logistic_kfold(
                    X_train, y_train,
                    penalty=penalty, solver=solver, C=c
                )


def run_knn_experiments(X_train, y_train, X_test, y_test, logger):
    """KNN."""
    print("\nKNN")

    model = KNeighborsClassifier()

    print("\nGrid Search")
    grid_tuning(
        model, knn_grid_params,
        X_train, y_train, X_test, y_test,
        is_scale=True,
        logger=logger
    )

    print("\nOptuna")
    optuna_tuning(
        model, knn_optuna_params,
        X_train, y_train, X_test, y_test,
        is_scale=True, n_trials=50,
        logger=logger
    )


def run_tree_experiments(X_train, y_train, X_test, y_test, logger):
    """Decision Tree и Random Forest."""
    print("\nДЕРЕВЬЯ РЕШЕНИЙ")

    dt = DecisionTreeClassifier(random_state=config.determinism.random_state)
    rf = RandomForestClassifier(random_state=config.determinism.random_state)

    print("\nDecision Tree (Grid)")
    grid_tuning(
        dt, dt_grid_params,
        X_train, y_train, X_test, y_test,
        is_scale=False,
        logger=logger
    )

    print("\nDecision Tree (Optuna)")
    optuna_tuning(
        dt, dt_optuna_params,
        X_train, y_train, X_test, y_test,
        is_scale=False, n_trials=50,
        logger=logger
    )

    print("\nRandom Forest (Grid)")
    grid_tuning(
        rf, rf_grid_params,
        X_train, y_train, X_test, y_test,
        is_scale=False,
        logger=logger
    )

    print("\nRandom Forest (Optuna)")
    optuna_tuning(
        rf, rf_optuna_params,
        X_train, y_train, X_test, y_test,
        is_scale=False, n_trials=50,
        logger=logger
    )


def run_boosting_experiments(X_train, y_train, X_test, y_test, logger):
    """XGBoost, LightGBM, CatBoost."""
    print("\nБУСТИНГИ")

    xgb = XGBClassifier(
        random_state=config.determinism.random_state,
        n_jobs=-1,
        enable_categorical=True
    )

    lgbm = LGBMClassifier(
        random_state=config.determinism.random_state,
        n_jobs=-1,
        verbose=-1
    )

    cat = CatBoostClassifier(
        random_seed=config.determinism.random_state,
        verbose=False
    )

    params = {'is_scale': False, 'is_cat': False}

    print("\nXGBoost")
    grid_tuning(xgb, xgb_grid_params, X_train, y_train, X_test, y_test, **params, logger=logger)
    optuna_tuning(xgb, xgb_optuna_params, X_train, y_train, X_test, y_test, **params, n_trials=50, logger=logger)

    print("\nLightGBM")
    grid_tuning(lgbm, lgbm_grid_params, X_train, y_train, X_test, y_test, **params, logger=logger)
    optuna_tuning(lgbm, lgbm_optuna_params, X_train, y_train, X_test, y_test, **params, n_trials=50, logger=logger)

    print("\nCatBoost")
    grid_tuning(cat, catboost_grid_params, X_train, y_train, X_test, y_test, **params, logger=logger)
    optuna_tuning(cat, catboost_optuna_params, X_train, y_train, X_test, y_test, **params, n_trials=50, logger=logger)


def run_dnn_experiments(X_train, y_train, X_test, y_test, logger):
    """DNN."""
    print("\nDNN")
    print("Не реализовано")


def main():
    """Запуск всех экспериментов."""
    print("ЗАПУСК ЭКСПЕРИМЕНТОВ")

    X_train, X_val, y_train, y_val, X_test = data_loading()

    X_all = pd.concat([X_train, X_val])
    y_all = pd.concat([y_train, y_val])

    print(f"\nTrain: {X_train.shape}")
    print(f"Val: {X_val.shape}")
    print(f"Test: {X_test.shape}")

    # run_logistic_experiments(X_train, y_train, X_val, y_val, logger=logger)
    run_knn_experiments(X_train, y_train, X_val, y_val, logger=logger)
    run_tree_experiments(X_train, y_train, X_val, y_val, logger=logger)
    run_boosting_experiments(X_train, y_train, X_val, y_val, logger=logger)
    run_dnn_experiments(X_train, y_train, X_val, y_val, logger=logger)

    logger.finish()

    print("\nГотово. Результаты в log/experiments.jsonl")


if __name__ == "__main__":
    main()