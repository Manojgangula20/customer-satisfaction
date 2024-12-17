import logging
import optuna
import pandas as pd
import xgboost as xgb
from lightgbm import LGBMRegressor
from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import numpy as np

class Hyperparameter_Optimization:
    """
    Class for doing hyperparameter optimization.
    """
    def __init__(self, x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, y_test: pd.Series) -> None:
        """Initialize the class with the training and test data."""
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def optimize_randomforest(self, trial: optuna.Trial) -> float:
        """Optimize Random Forest hyperparameters."""
        n_estimators = trial.suggest_int("n_estimators", 1, 200)
        max_depth = trial.suggest_int("max_depth", 1, 20)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        reg = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
        )
        reg.fit(self.x_train, self.y_train)
        return reg.score(self.x_test, self.y_test)

    def optimize_lightgbm(self, trial: optuna.Trial) -> float:
        """Optimize LightGBM hyperparameters."""
        n_estimators = trial.suggest_int("n_estimators", 1, 200)
        max_depth = trial.suggest_int("max_depth", 1, 20)
        learning_rate = trial.suggest_uniform("learning_rate", 0.01, 0.99)
        reg = LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
        )
        reg.fit(self.x_train, self.y_train)
        return reg.score(self.x_test, self.y_test)

    def optimize_xgboost_regressor(self, trial: optuna.Trial) -> float:
        """Optimize XGBoost hyperparameters."""
        param = {
            "max_depth": trial.suggest_int("max_depth", 1, 30),
            "learning_rate": trial.suggest_float("learning_rate", 1e-7, 10.0),
            "n_estimators": trial.suggest_int("n_estimators", 1, 200),
        }
        reg = xgb.XGBRegressor(**param)
        reg.fit(self.x_train, self.y_train)
        return reg.score(self.x_test, self.y_test)


class ModelTrainer:
    """
    Class for training models.
    """
    def __init__(self, x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, y_test: pd.Series) -> None:
        """Initialize the class with the training and test data."""
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def preprocess_data(self):
        """Preprocess and handle missing data."""
        imputer_x = SimpleImputer(strategy='median')
        self.x_train = imputer_x.fit_transform(self.x_train)
        self.x_test = imputer_x.transform(self.x_test)

        imputer_y = SimpleImputer(strategy='median')
        self.y_train = imputer_y.fit_transform(self.y_train.values.reshape(-1, 1)).ravel()  # Flatten after imputation
        self.y_test = imputer_y.transform(self.y_test.values.reshape(-1, 1)).ravel()  # Flatten after imputation

        # Ensure no NaN values remain (using np.isnan for numpy arrays)
        assert not np.isnan(self.x_train).any(), "x_train contains NaN values"
        assert not np.isnan(self.x_test).any(), "x_test contains NaN values"
        assert not np.isnan(self.y_train).any(), "y_train contains NaN values"
        assert not np.isnan(self.y_test).any(), "y_test contains NaN values"

    def random_forest_trainer(self, fine_tuning: bool = True) -> RegressorMixin:
        """Train the Random Forest model."""
        logging.info("Started training Random Forest model.")
        try:
            if fine_tuning:
                hyper_opt = Hyperparameter_Optimization(self.x_train, self.y_train, self.x_test, self.y_test)
                study = optuna.create_study(direction="maximize")
                study.optimize(hyper_opt.optimize_randomforest, n_trials=10)
                trial = study.best_trial
                logging.info(f"Best parameters: {trial.params}")
                reg = RandomForestRegressor(
                    n_estimators=trial.params["n_estimators"],
                    max_depth=trial.params["max_depth"],
                    min_samples_split=trial.params["min_samples_split"],
                )
                reg.fit(self.x_train, self.y_train)
                return reg
            else:
                model = RandomForestRegressor(n_estimators=152, max_depth=20, min_samples_split=17)
                model.fit(self.x_train, self.y_train)
                return model
        except Exception as e:
            logging.error("Error in training Random Forest model")
            logging.error(e)
            return None

    def lightgbm_trainer(self, fine_tuning: bool = True) -> RegressorMixin:
        """Train the LightGBM model."""
        logging.info("Started training LightGBM model.")
        try:
            if fine_tuning:
                hyper_opt = Hyperparameter_Optimization(self.x_train, self.y_train, self.x_test, self.y_test)
                study = optuna.create_study(direction="maximize")
                study.optimize(hyper_opt.optimize_lightgbm, n_trials=10)
                trial = study.best_trial
                logging.info(f"Best parameters: {trial.params}")
                reg = LGBMRegressor(
                    n_estimators=trial.params["n_estimators"],
                    learning_rate=trial.params["learning_rate"],
                    max_depth=trial.params["max_depth"],
                )
                reg.fit(self.x_train, self.y_train)
                return reg
            else:
                model = LGBMRegressor(n_estimators=200, learning_rate=0.01, max_depth=20)
                model.fit(self.x_train, self.y_train)
                return model
        except Exception as e:
            logging.error("Error in training LightGBM model.")
            logging.error(e)
            return None

    # def xgboost_trainer(self, fine_tuning: bool = True) -> RegressorMixin:
    #     """
    #     It trains the xgboost model.
        
    #     Args:
    #         fine_tuning: If True, hyperparameter optimization is performed. If False, the default
    #         parameters are used, Defaults to True (optional).
    #     """
    #     logging.info("Started training XGBoost model.")
    #     try:
    #         # Preprocess missing values in the training and test data
    #         self.x_train.fillna(self.x_train.median(), inplace=True)
    #         self.x_test.fillna(self.x_test.median(), inplace=True)
    #         self.y_train.fillna(self.y_train.median(), inplace=True)
    #         self.y_test.fillna(self.y_test.median(), inplace=True)

    #         # Ensure no NaN values remain
    #         assert not self.x_train.isnull().values.any(), "x_train contains NaN values"
    #         assert not self.x_test.isnull().values.any(), "x_test contains NaN values"
    #         assert not self.y_train.isnull().values.any(), "y_train contains NaN values"
    #         assert not self.y_test.isnull().values.any(), "y_test contains NaN values"
            
    #         if fine_tuning:
    #             logging.info("Starting hyperparameter optimization with Optuna.")
    #             # Hyperparameter optimization
    #             hy_opt = Hyperparameter_Optimization(self.x_train, self.y_train, self.x_test, self.y_test)
    #             study = optuna.create_study(direction="maximize")
    #             study.optimize(hy_opt.optimize_xgboost_regressor, n_trials=10)
                
    #             # Retrieve best trial's parameters
    #             trial = study.best_trial
    #             n_estimators = trial.params["n_estimators"]
    #             learning_rate = trial.params["learning_rate"]
    #             max_depth = trial.params["max_depth"]
                
    #             logging.info(f"Best trial: n_estimators={n_estimators}, learning_rate={learning_rate}, max_depth={max_depth}")
                
    #             # Train model with best parameters
    #             reg = xgb.XGBRegressor(
    #                 n_estimators=n_estimators,
    #                 learning_rate=learning_rate,
    #                 max_depth=max_depth,
    #             )
    #             reg.fit(self.x_train, self.y_train)
    #             logging.info("Model training completed.")
    #             return reg

    #         else:
    #             # Default parameters for model training
    #             logging.info("Using default parameters for model training.")
    #             model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.01, max_depth=20)
    #             model.fit(self.x_train, self.y_train)
    #             logging.info("Model training completed.")
    #             return model

    #     except Exception as e:
    #         logging.error(f"Error during model training: {e}")
    #         raise e

