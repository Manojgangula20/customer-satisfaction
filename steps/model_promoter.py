from zenml import Model, get_step_context, step
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def model_promoter(mse: float, stage: str = "production") -> bool:
    """Model promotion step

    Step that conditionally promotes a model if it has an MSE better than
    the previous production model.

    Args:
        mse: Mean-squared error of the model.
        stage: Which stage to promote the model to.

    Returns:
        Whether the model was promoted or not.
    """
    # Get the current model produced by the current pipeline
    zenml_model = get_step_context().model

    # Fetch the previous model version at the specified stage
    try:
        previous_production_model = Model(
            name=zenml_model.name, version="production"
        )
    except Exception as e:
        logger.error(f"Failed to fetch previous production model: {e}")
        previous_production_model = None

    # Initialize the threshold for comparison
    previous_production_model_mse = None

    if previous_production_model:
        try:
            # Accessing the metrics dictionary safely
            metrics = (
                previous_production_model.get_artifact("sklearn_regressor")
                .run_metadata["metrics"]
            )
            previous_production_model_mse = float(metrics.get("mse", float("inf")))
        except Exception as e:
            logger.warning(
                f"Could not retrieve MSE from the previous production model: {e}"
            )
            previous_production_model_mse = float("inf")

    # Promote the model if it has a better MSE
    if mse < previous_production_model_mse:
        logger.info(f"Model promoted to {stage} stage!")
        zenml_model.set_stage(stage, force=True)
        return True
    else:
        logger.info(
            f"Model not promoted. Current model MSE ({mse:.2f}) is not better than "
            f"the previous production model MSE ({previous_production_model_mse:.2f})."
        )
        return False
