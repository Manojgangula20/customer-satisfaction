import logging
from typing import Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DataCleaning:
    """
    Data cleaning class which preprocesses the data and divides it into train and test data.
    """

    def __init__(self, data: pd.DataFrame) -> None:
        """Initializes the DataCleaning class."""
        self.df = data

    def preprocess_data(self) -> pd.DataFrame: 
        """ Removes columns which are not required, 
        fills missing values with median average values
        and converts the data type to float. 
        """ 
        try:
            logging.info("Starting data preprocessing...") 
            # Drop unnecessary columns 
            self.df = self.df.drop( 
                [ "order_approved_at", 
                 "order_delivered_carrier_date", 
                 "order_delivered_customer_date", 
                 "order_estimated_delivery_date", 
                 "order_purchase_timestamp", 
                 ], 
                 axis=1, 
            ) 
            logging.info("Dropped unnecessary columns.") 
            # Fill missing values with median for specific columns 
            self.df["product_weight_g"] = self.df["product_weight_g"].fillna(self.df["product_weight_g"].median()) 
            self.df["product_length_cm"] = self.df["product_length_cm"].fillna(self.df["product_length_cm"].median())
            self.df["product_height_cm"] = self.df["product_height_cm"].fillna(self.df["product_height_cm"].median())
            self.df["product_width_cm"] = self.df["product_width_cm"].fillna(self.df["product_width_cm"].median()) 
            self.df["review_comment_message"] = self.df["review_comment_message"].fillna("No review") 
            logging.info("Filled missing values for specific columns.")
            
            # Convert integer columns to float to handle missing values 
            for col in self.df.select_dtypes(include=['int']).columns: 
                self.df[col] = self.df[col].astype(float) 
            logging.info("Converted integer columns to float.") 
            
            # Drop additional columns 
            cols_to_drop = [ 
                "customer_zip_code_prefix", 
                "order_item_id", 
            ] 
            self.df = self.df.drop(cols_to_drop, axis=1) 
            logging.info("Dropped additional columns.") 
            # Catch all fillna in case any were missed 
            # self.df.fillna(self.df.mean(), inplace=True) 
            # logging.info("Filled remaining missing values.") 
            # Check for any remaining null values 
            null_values = self.df.isnull().sum().sum() 
            if null_values > 0: 
                logging.warning(f"There are still {null_values} null values remaining in the dataset.") 
            else: 
                logging.info("No null values remaining in the dataset.") 
            # Ensure all columns contain appropriate data types 
            for col in self.df.columns: 
                if self.df[col].dtype == 'object': 
                    try: 
                        self.df[col] = self.df[col].astype(float) 
                    except ValueError: 
                        logging.warning(f"Column {col} contains non-numeric values and cannot be converted to float.") 
                        self.df = self.df.drop(col, axis=1) 
                        logging.info(f"Dropped column {col} due to non-numeric values.") 
            
            logging.info("Data preprocessing completed successfully.") 
            return self.df 
        except Exception as e: 
            logging.error(f"Error during data preprocessing: {e}") 
            raise e


    def divide_data(self, df: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        It divides the data into train and test data.
        """
        try:
            X = df.drop("review_score", axis=1)
            y = df["review_score"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(e)
            raise e