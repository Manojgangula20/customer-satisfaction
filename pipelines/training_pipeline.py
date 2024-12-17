from steps.clean_data import clean_data
from steps.evaluation import evaluation
from steps.ingest_data import ingest_data  # Import the function explicitly
from steps.model_promoter import model_promoter
from steps.train_model import train_model
from zenml import pipeline


@pipeline
def customer_satisfaction_training_pipeline(model_type: str = "lightgbm"):
    """Training Pipeline.

    Args:
        model_type: str - available options ["lightgbm", "randomforest", "xgboost"]
    """
    df = ingest_data()
    x_train, x_test, y_train, y_test = clean_data(df)
    model = train_model(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        model_type=model_type,
    )
    mse, rmse = evaluation(model, x_test, y_test)
    is_promoted = model_promoter(mse=mse)
    return model, is_promoted