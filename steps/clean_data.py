import logging
from typing import Tuple
import pandas as pd

from zenml import step

from src.data_cleaning import (DataCleaning,
                               DataDivideStrategy,
                               DataPreprocessStrategy,

)

from typing_extensions import Annotated



@step
def clean_df(
    data: pd.DataFrame,
) -> Tuple[
    Annotated[pd.DataFrame, "x_train"],
    Annotated[pd.DataFrame, "x_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    """Data cleaning class which preprocesses the data and divides it into train and test data.

    Args:
        data: Raw data

    Returns:
        x_train: Training features
        x_test: Testing features
        y_train: Training labels
        y_test: Testing labels
    """
    try:
        preprocess_strategy = DataPreprocessStrategy()
        data_cleaning = DataCleaning(data, preprocess_strategy)
        preprocessed_data = data_cleaning.handle_data()

        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(preprocessed_data, divide_strategy)
        x_train, x_test, y_train, y_test = data_cleaning.handle_data()
        return x_train, x_test, y_train, y_test
    except Exception as e:
        logging.error(e)
        raise e