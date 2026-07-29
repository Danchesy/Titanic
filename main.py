import warnings
from functools import partial

import hydra
from omegaconf import DictConfig, OmegaConf

# from log_reg import log_reg_cv, logistic_kfold
from tuning_params import (
    catboost_optuna_params,
    dt_optuna_params,
    grid_tuning,
    knn_optuna_params,
    lgbm_optuna_params,
    logreg_optuna_params,
    optuna_tuning,
    rf_optuna_params,
    xgb_optuna_params,
)
from utils import WandbLogger, data_loading, set_seed

warnings.filterwarnings("ignore")

OPTUNA_MAP = {
    "linear_model": logreg_optuna_params,
    "knn": knn_optuna_params,
    "dt": dt_optuna_params,
    "rf": rf_optuna_params,
    "xgboost": xgb_optuna_params,
    "lightgbm": lgbm_optuna_params,
    "catboost": catboost_optuna_params,
}


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Главная функция запуска экспериментов."""
    print("ЗАПУСК ЭКСПЕРИМЕНТОВ")

    set_seed(cfg.seed)
    logger = WandbLogger(cfg)

    X_train, X_val, y_train, y_val, X_test = data_loading(cfg)

    print(f"Train : {X_train.shape} | Val : {X_val.shape} | Test : {X_test.shape}")

    for model_name, model_cfg in cfg.model.items():
        print(f"\n{model_name.upper()}")

        meta_keys = {"grid_params", "is_cat", "is_scale", "cat_features"}
        clean_cfg = OmegaConf.create(
            {k: v for k, v in model_cfg.items() if k not in meta_keys}
        )

        cat_features = (
            list(OmegaConf.to_container(model_cfg.cat_features, resolve=True))
            if "cat_features" in model_cfg
            else None
        )

        model = hydra.utils.instantiate(clean_cfg)

        grid_p = (
            OmegaConf.to_container(model_cfg.grid_params, resolve=True) if "grid_params" in model_cfg else {}
        )

        optuna_p = OPTUNA_MAP.get(model_name)

        if model_name == "linear_model" and "solver" in cfg:
            solver_map = OmegaConf.to_container(cfg.solver, resolve=True)
            optuna_p = partial(logreg_optuna_params, solver_mapping=solver_map)

        is_scale = model_cfg.get("is_scale", False)
        is_cat = model_cfg.get("is_cat", True)

        if grid_p and cfg.tuning.use_grid_search:
            grid_tuning(
                model,
                grid_p,
                X_train,
                y_train,
                X_val,
                y_val,
                is_scale=is_scale,
                is_cat=is_cat,
                cat_features=cat_features,
                logger=logger,
            )

        if optuna_p and cfg.tuning.use_optuna:
            optuna_tuning(
                model,
                optuna_p,
                X_train,
                y_train,
                X_val,
                y_val,
                is_scale=is_scale,
                is_cat=is_cat,
                cat_features=cat_features,
                n_trials=cfg.tuning.n_trials,
                logger=logger,
            )

    logger.finish()
    print("\nГотово")


if __name__ == "__main__":
    main()
