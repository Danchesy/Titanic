from pathlib import Path

from omegaconf import OmegaConf

BASE_DIR = Path(__file__).resolve().parent

conf = {
    "logs": {"console": True, "file": False, "wandb": True},
    "paths": {
        "path_to_train": BASE_DIR / "data" / "train.csv",
        "path_to_test": BASE_DIR / "data" / "test.csv",
        "experiments_log": BASE_DIR / "log" / "experiments.jsonl",
    },
    "determenism": {
        "random_state": 42, 
        "shuffle": True},
    "scaling": {
        "is_scale": True,
        "scaler": "StandardScaler",  # StandardScaler, MinMaxScaler
    },

    "train_test_split": {
        "test_size": 0.2,
        "is_val_set": False,
        "val_size": 0.2,
    },
    "cv": {
        "n_splits": 5,
    },
    "solver_mapping": {
        "l1": "liblinear",
        "l2": "lbfgs",
        "elasticnet": "saga",
        "None": "lbfgs",
    },
    "linear_model": {
        "penalty": "l1",
        "C": 1.0,
        "l1_ratio": 0.0,
        "max_iter": 100,
        "solver": "${solver_mapping.${linear_model.penalty}}",
        "tol": 1e-4,
        "bias": True,
        "verbose": 0,
        "cv": 5,
        "refit": True,
        "Cs": 10,
        "scoring": "accuracy",
    },
    "knn": {
        "n_neighbors": 3,
        "weights": "uniform",
        "algorithm": "auto",
        "leaf_size": 2,
        "p": 2,
        "metric": "minkowski",
    },
    "dt": {
        "criterion": "gini",
        "splitter": "best",
        "max_depth": 3,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "min_weight_fraction_leaf": 0.0,
        "max_features": None,
        "max_leaf_nodes": None,
        "min_impurity_decrease": 0.0,
        "class_weight": None,
        "ccp_alpha": 0.0,
        "monotonic_cst": None,
    },
    "rf": {
        "n_estimators": 100,
        "criterion": "gini",
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "min_weight_fraction_leaf": 0.0,
        "max_features": "sqrt",
        "max_leaf_nodes": None,
        "min_impurity_decrease": 0.0,
        "bootstrap": True,
        "oob_score": False,
        "n_jobs": -1,
        "verbose": 0,
        "warm_start": False,
        "class_weight": None,
        "ccp_alpha": 0.0,
        "max_samples": None,
        "monotonic_cst": None,
    },
    "target_column_name": "Survived",
    "is_cat": True
}

config = OmegaConf.create(conf)
