import logging
from typing import Annotated
import mlflow
import pandas as pd
from model.model_dev import ModelTrainer
from sklearn.base import RegressorMixin
from zenml import ArtifactConfig, step
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker


@step(enable_cache=False, experiment_tracker=experiment_tracker.name)
def train_model(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_type: str = "lightgbm",
    do_fine_tuning: bool = True,
) -> Annotated[
    RegressorMixin,
    ArtifactConfig(name="sklearn_regressor", is_model_artifact=True),
]:
    """
    Args:
        x_train: pd.DataFrame
        x_test: pd.DataFrame
        y_train: pd.Series
        y_test: pd.Series
        model_type: str - available options ["lightgbm", "randomforest", "xgboost"]
        do_fine_tuning: Should full training run or only fine tuning
    Returns:
        model: RegressorMixin
    """
    try:
        logging.info(f"Starting model training with model_type: {model_type}")
        model_training = ModelTrainer(x_train, y_train, x_test, y_test)

        if model_type == "lightgbm":
            mlflow.lightgbm.autolog()
            lgm_model = model_training.lightgbm_trainer(fine_tuning=do_fine_tuning)
            if lgm_model is None:
                raise ValueError("LightGBM model training failed. Model is None.")
            logging.info("LightGBM model trained successfully.")
            return lgm_model

        elif model_type == "randomforest":
            mlflow.sklearn.autolog()
            rf_model = model_training.random_forest_trainer(fine_tuning=do_fine_tuning)
            if rf_model is None:
                raise ValueError("RandomForest model training failed. Model is None.")
            logging.info("RandomForest model trained successfully.")
            return rf_model

        # elif model_type == "xgboost":
        #     mlflow.xgboost.autolog()
        #     xgb_model = model_training.xgboost_trainer(fine_tuning=do_fine_tuning)
        #     if xgb_model is None:
        #         raise ValueError("XGBoost model training failed. Model is None.")
        #     logging.info("XGBoost model trained successfully.")
        #     return xgb_model

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    except Exception as e:
        logging.error(f"An error occurred during model training: {e}")
        raise e
