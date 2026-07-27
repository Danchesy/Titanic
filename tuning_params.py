import numpy as np
import optuna
from config_file import config
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from utils import add_result, pipeline_return, preprocessor


def grid_tuning(model, 
                params, 
                X_train, 
                y_train, 
                X_test, 
                y_test,
                is_scale = True,
                is_cat = True):
    pipeline = Pipeline([
        ('preprocessor', preprocessor(is_scale=is_scale, is_cat=is_cat)),
        ('model', model)
    ])

    grid_search = GridSearchCV(estimator=pipeline, 
                    param_grid=params,
                    scoring=config.linear_model.scoring,  
                    refit=config.linear_model.refit, 
                    cv=config.cv.n_splits, 
                    pre_dispatch='2*n_jobs',  
                    return_train_score=False)

    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV Accuracy: {grid_search.best_score_:.4f}")

    best_pipeline = grid_search.best_estimator_
    test_preds = best_pipeline.predict(X_test)
    final_acc = accuracy_score(y_test, test_preds)

    print(f"X_test Accuracy: {final_acc:.4f}")

    res = pipeline_return(best_pipeline, grid_search.best_score_)
    add_result(res)

    return grid_search


def optuna_tuning(model, 
                  params_fn, 
                  X_train, 
                  y_train, 
                  X_test, 
                  y_test,
                  direction = 'maximize',
                  n_trials = 20,
                  is_scale = config.scaling.is_scale,
                  is_cat = config.is_cat):

    def objective(trial):
        params = params_fn(trial)

        current_model = clone(model)
        current_model.set_params(**params)

        pipeline = Pipeline([
            ('preprocessor', preprocessor(is_scale=is_scale, is_cat=is_cat)),
            ('model', current_model)
        ])

        accuracy = cross_val_score(pipeline, 
                                X_train, 
                                y_train, 
                                cv=config.cv.n_splits, 
                                scoring=config.linear_model.scoring).mean()
        return accuracy


    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    if config.logs.console:
        print(f"Best CV Accuracy: {study.best_value:.4f}")
        print(f"Best parameters: {best_params}")

    best_model = clone(model)
    best_model.set_params(**best_params)

    final_pipeline = Pipeline([
        ('preprocessor', preprocessor(is_scale, is_cat)),
        ('model', best_model)
    ])

    final_pipeline.fit(X_train, y_train)

    pred = final_pipeline.predict(X_test)
    test_accuracy = accuracy_score(y_test, pred) # Правильный порядок: y_true, y_pred

    if config.logs.console:
        print(f"X_test Accuracy: {test_accuracy:.4f}")

    res = pipeline_return(final_pipeline, study.best_value)
    add_result(res)

    return study


def knn_optuna_params(trial):
    return {
              'n_neighbors': trial.suggest_int('n_neighbors', 3, 17, step=2),
              'weights': trial.suggest_categorical('weights', ['distance', 'uniform']), 
              'leaf_size': trial.suggest_categorical('leaf_size', [20, 30, 50]), 
              'metric': trial.suggest_categorical('metric', ['cosine', 'manhattan', 'euclidean'])
            }

knn_grid_params = {
            'model__n_neighbors': np.arange(3, 18, 2),
            'model__weights': ['distance', 'uniform'], 
            'model__leaf_size': [20, 30, 50], 
            'model__metric': ['cosine', 'manhattan', 'euclidean']}


def dt_optuna_params(trial):
    return {
              'max_depth': trial.suggest_int('max_depth', 3, 7), 
              'min_samples_split': trial.suggest_categorical('min_samples_split', [5, 10, 20, 30]), 
              'min_samples_leaf': trial.suggest_categorical('min_samples_leaf', [2, 5, 10, 20]),
              'min_weight_fraction_leaf': trial.suggest_float('min_weight_fraction_leaf', 0.0, 0.2), 
              'max_features': trial.suggest_categorical('max_features', [None, 'sqrt', 'log2', 0.5, 0.8]), 
              'max_leaf_nodes': trial.suggest_categorical('max_leaf_nodes', [None, 5, 10, 25, 50, 100]), 
              'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0.0, 0.01)
            }

dt_grid_params = {
            'model__max_depth': np.arange(3, 8), 
            'model__min_samples_split': [5, 10, 20, 30], 
            'model__min_samples_leaf': [2, 5, 10, 20],
            'model__min_weight_fraction_leaf': np.linspace(0.0, 0.2, 5), 
            'model__max_features': [None, 'sqrt', 'log2', 0.5, 0.8], 
            'model__max_leaf_nodes': [None, 5, 10, 25, 50, 100], 
            'model__min_impurity_decrease': np.linspace(0.0, 0.01, 5)}


def rf_optuna_params(trial):
    return {
              'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
              'max_depth': trial.suggest_int('max_depth', 3, 7), 
              'min_samples_split': trial.suggest_categorical('min_samples_split', [5, 10, 20, 30]), 
              'min_samples_leaf': trial.suggest_categorical('min_samples_leaf', [2, 5, 10, 20]),
              'min_weight_fraction_leaf': trial.suggest_float('min_weight_fraction_leaf', 0.0, 0.2), 
              'max_features': trial.suggest_categorical('max_features', [None, 'sqrt', 'log2', 0.5, 0.8]), 
              'max_leaf_nodes': trial.suggest_categorical('max_leaf_nodes', [None, 5, 10, 25, 50, 100]), 
              'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0.0, 0.01)
            }

rf_grid_params = {
            'model__n_estimators': np.linspace(100, 1000, 5, dtype=int),
            'model__max_depth': np.arange(3, 8), 
            'model__min_samples_split': [5, 10, 20, 30], 
            'model__min_samples_leaf': [2, 5, 10, 20],
            'model__min_weight_fraction_leaf': np.linspace(0.0, 0.2, 5), 
            'model__max_features': [None, 'sqrt', 'log2', 0.5, 0.8], 
            'model__max_leaf_nodes': [None, 5, 10, 25, 50, 100], 
            'model__min_impurity_decrease': np.linspace(0.0, 0.01, 5)}


def xgb_optuna_params(trial):
    return {
        'max_depth': trial.suggest_int('max_depth', 3, 5),
        'n_estimators': trial.suggest_int('n_estimators', 100, 200),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 5.0, log=True), 
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 2.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 0.8, step=0.05),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.8, step=0.05),
        'gamma': trial.suggest_float('gamma', 0.0, 0.3, step=0.05)
    }

xgb_grid_params = {
    'model__max_depth': np.arange(3, 6),
    'model__n_estimators': np.linspace(100, 200, 3, dtype=int),
    'model__learning_rate': np.logspace(np.log10(0.01), np.log10(0.1), 3),
    'model__reg_lambda': np.logspace(np.log10(0.5), np.log10(5.0), 3),
    'model__reg_alpha': np.logspace(np.log10(0.01), np.log10(2.0), 3),
    'model__min_child_weight': np.linspace(1, 10, 3, dtype=int),
    'model__subsample': np.arange(0.6, 0.81, 0.05),
    'model__colsample_bytree': np.arange(0.6, 0.81, 0.05),
    'model__gamma': np.arange(0.0, 0.31, 0.05)
}


def lgbm_optuna_params(trial):
    return {
        'max_depth': trial.suggest_int('max_depth', 3, 5),
        'n_estimators': trial.suggest_int('n_estimators', 100, 200),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 5.0, log=True), 
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 2.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 0.8, step=0.05),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.8, step=0.05),
        'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 0.3, step=0.05)
    }

lgbm_grid_params = {
    'model__max_depth': np.arange(3, 6),
    'model__n_estimators': np.linspace(100, 200, 3, dtype=int),
    'model__learning_rate': np.logspace(np.log10(0.01), np.log10(0.1), 3),
    'model__reg_lambda': np.logspace(np.log10(0.5), np.log10(5.0), 3),
    'model__reg_alpha': np.logspace(np.log10(0.01), np.log10(2.0), 3),
    'model__min_child_weight': np.linspace(1, 10, 3, dtype=int),
    'model__subsample': np.arange(0.6, 0.81, 0.05),
    'model__colsample_bytree': np.arange(0.6, 0.81, 0.05),
    'model__min_split_gain': np.arange(0.0, 0.31, 0.05)
}


def catboost_optuna_params(trial):
    return {
        'iterations': trial.suggest_int('iterations', 100, 200),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1),
        'depth': trial.suggest_int('depth', 3, 6),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 3.0, 5.0),
        'subsample': trial.suggest_float('subsample', 0.5, 0.8),
        'rsm': trial.suggest_float('rsm', 0.5, 0.8)
    }

catboost_grid_params = {
    'model__iterations': np.linspace(100, 200, 3, dtype=int),
    'model__learning_rate': np.logspace(np.log10(0.001), np.log10(0.1), 3),
    'model__depth': np.arange(3, 7),
    'model__l2_leaf_reg': np.linspace(3.0, 5.0, 3),
    'model__subsample': np.linspace(0.5, 0.8, 3),
    'model__rsm': np.linspace(0.5, 0.8, 3)
}
