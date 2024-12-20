import logging
import os
from copy import deepcopy
from statistics import mean

import optuna
from hydra.utils import to_absolute_path
from sklearn.model_selection import StratifiedKFold

from model import get_classifier, get_regressor

# import matplotlib.pyplot as plt
# import cv2

logger = logging.getLogger(__name__)


def xgboost_config(trial: optuna.Trial, model_config, name=""):
    model_config.max_depth = trial.suggest_int("max_depth", 3, 10)
    model_config.eta = trial.suggest_float("eta", 1e-5, 1.0, log=True)
    model_config.min_child_weight = trial.suggest_float("min_child_weight", 1e-8, 1e5, log=True)
    model_config.subsample = trial.suggest_float("subsample", 0.5, 1.0)
    model_config.colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
    model_config.colsample_bylevel = trial.suggest_float("colsample_bylevel", 0.5, 1.0)
    model_config.gamma = trial.suggest_float("gamma", 1e-8, 1e2, log=True)
    model_config.alpha = trial.suggest_float("alpha", 1e-8, 1e2, log=True)
    model_config["lambda"] = trial.suggest_float("lambda", 1e-8, 1e2, log=True)
    return model_config

def lightgbm_config(trial: optuna.Trial, model_config, name=""):
    model_config.max_depth = trial.suggest_int("max_depth", 3, 10)
    model_config.lambda_l1 = trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True)
    model_config.lambda_l2 = trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True)
    model_config.learning_rate = trial.suggest_float("learning_rate", 1e-5, 1.0, log=True)
    model_config.num_leaves = trial.suggest_int("num_leaves", 2, 256, log=True)
    model_config.feature_fraction = trial.suggest_float("feature_fraction", 0.4, 1.0)
    model_config.bagging_fraction = trial.suggest_float("bagging_fraction", 0.4, 1.0)
    model_config.bagging_freq = trial.suggest_int("bagging_freq", 1, 7)
    model_config.min_child_samples = trial.suggest_int("min_child_samples", 5, 100, log=True)
    model_config.min_data_in_leaf = trial.suggest_int("min_data_in_leaf", 5, 50, log=True)
    return model_config

def catboost_config(trial: optuna.Trial, model_config, name=""):
    model_config.depth = trial.suggest_int("depth", 3, 10)
    model_config.learning_rate = trial.suggest_float("learning_rate", 1e-5, 1.0)
    model_config.random_strength = trial.suggest_int("random_strength", 0, 100)
    model_config.bagging_temperature = trial.suggest_float("bagging_temperature", 0.01, 100.00, log=True)
    model_config.iterations = trial.suggest_int("iterations",100,10000)
    model_config.subsample = trial.suggest_float("subsample",0.5,1.0)
    # model_config.l2_leaf_reg = trial.suggest_int("l2_leaf_reg",,10)
    # model_config.min_data_in_leaf = trial.suggest_int("min_data_in_leaf", 1, 20)
    # model_config.colsample_bylevel= trial.suggest_float("colsample_bylevel",0,1.0)
    return model_config

def xgblgbmcat_config(trial: optuna.Trial, model_config, name=""):
    if name == "xgboost" or name == "":
        model_config = xgboost_config(trial, model_config, name)
    
    if name == "lightgbm" or name == "":
        model_config = lightgbm_config(trial, model_config, name)
    
    if name == "catboost" or name == "":
        model_config = catboost_config(trial, model_config, name)

    return model_config

def get_model_config(model_name):
    if model_name == "xgboost":
        return xgboost_config
    elif model_name == "lightgbm":
        return lightgbm_config
    elif model_name == "catboost":
        return catboost_config
    elif model_name == "xgblr":
        return xgboost_config
    elif model_name == "xgblgbmcat":
        return xgblgbmcat_config
    else:
        raise ValueError()


def update_model_cofig(default_config, best_config):
    for _p, v in best_config.items():
        current_dict = default_config
        _p = _p.split(".")
        for p in _p[:-1]:
            if p not in current_dict:
                current_dict[p] = {}
            current_dict = current_dict[p]
        last_key = _p[-1]
        current_dict[last_key] = v


class OptimParam:
    def __init__(
        self,
        model_name,
        default_config,
        input_dim,
        output_dim,
        X,
        y,
        val_data,
        columns,
        target_column,
        n_trials,
        n_startup_trials,
        storage,
        study_name,
        cv=True,
        n_jobs=1,
        seed=42,
        alpha=1,
        task="None",
    ) -> None:
        self.model_name = model_name
        self.default_config = deepcopy(default_config)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model_config = get_model_config(model_name)
        self.X = X
        self.y = y
        self.val_data = val_data
        self.columns = columns
        self.target_column = target_column
        self.n_trials = n_trials
        self.n_startup_trials = n_startup_trials
        self.storage = to_absolute_path(storage) if storage is not None else None
        self.study_name = study_name
        self.cv = cv
        self.n_jobs = n_jobs
        self.seed = seed
        self.alpha = alpha
        self.task = task

    def fit(self, model_config, X_train, y_train, X_val=None, y_val=None):
        if X_val is None and y_val is None:
            X_val = self.val_data[self.columns]
            y_val = self.val_data[self.target_column].values.squeeze()
        
        if self.task == "classifier":
            model = get_classifier(
                self.model_name,
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                model_config=model_config,
                seed=self.seed,
            )
        elif self.task == "regressor":
            model = get_regressor(
                self.model_name,
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                model_config=model_config,
                seed=self.seed,
            )
        model.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
        )
        score = model.evaluate(
            self.val_data[self.columns],
            self.val_data[self.target_column].values.squeeze(),
        )
        return score

    def cross_validation(self, model_config):
        skf = StratifiedKFold(n_splits=10, random_state=self.seed, shuffle=True)
        ave = []
        for _, (train_idx, val_idx) in enumerate(skf.split(self.X, self.y)):
            X_train, y_train = self.X.iloc[train_idx], self.y[train_idx]
            X_val, y_val = self.X.iloc[val_idx], self.y[val_idx]
            score = self.fit(model_config, X_train, y_train, X_val, y_val)
            if self.task == "classifier":
                ave.append(score["AUC"]) 
            elif self.task == "regressor":
                ave.append(score["RMSE"])
        return mean(ave)

    def one_shot(self, model_config):
        score = self.fit(model_config, self.X, self.y)
        if self.task == "classifier":
            return score["ACC"]
        elif self.task == "regressor":
            return score["RMSE"]

    def objective(self, trial):
        _model_config = self.model_config(trial, deepcopy(self.default_config))
        if self.cv:
            value = self.cross_validation(_model_config)
        else:
            value = self.one_shot(_model_config)
        return value

    def get_n_complete(self, study: optuna.Study):
        n_complete = len([trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE])
        return n_complete

    
    
    def get_best_config(self):
        if self.storage is not None:
            os.makedirs(self.storage, exist_ok=True)
            self.storage = optuna.storages.RDBStorage(
                url=f"sqlite:///{self.storage}/optuna.db",
            )
        if self.task == "classifier":
            study = optuna.create_study(
                storage=self.storage,
                study_name=self.study_name,
                direction="maximize",
                sampler=optuna.samplers.TPESampler(
                    seed=self.seed,
                    n_startup_trials=self.n_startup_trials,
                ),
                load_if_exists=True,
            )
        elif self.task == "regressor":
            study = optuna.create_study(
                storage=self.storage,
                study_name=self.study_name,
                direction="minimize",
                sampler=optuna.samplers.TPESampler(
                    seed=self.seed,
                    n_startup_trials=self.n_startup_trials,
                ),
                load_if_exists=True,
            )
        n_complete = self.get_n_complete(study)
        n_trials = self.n_trials
        if n_complete > 0:
            n_trials -= n_complete
        study.optimize(self.objective, n_trials=n_trials, n_jobs=self.n_jobs)
        best_params = study.best_params
        logger.info("Best parameters found:")
        for param, value in best_params.items():
            logger.info(f"{param}: {value}")
        update_model_cofig(self.default_config, study.best_params)
        return self.default_config
