import logging 
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DataStrategy(ABC):
    """
    Abstract class defining strategy for handling data
    """

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass

class DataPreProcessStrategy(DataStrategy):
    """
    Strategy for pre-processing data
    """

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Pre-process Data
        """

        try:
            data = data.drop(
                [
                    "order_purchase_timestamp",
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                ],
                axis = 1
            ) #drop these columns from the dataframe as they are of no significant use in customer saatisfaction prediction

            data["product_weight_g"].fillna(data["product_weight_g"].median(), inplace = True)
            data["product_length_cm"].fillna(data["product_length_cm"].median(), inplace = True)
            data["product_height_cm"].fillna(data["product_height_cm"].median(), inplace = True)
            data["product_width_cm"].fillna(data["product_width_cm"].median(), inplace = True)
            data["review_comment_message"].fillna("No Review", inplace = True)

            data = data.select_dtypes(include=[np.number])
            cols_to_drop = ["customer_zip_code_prefix", "order_item_id"]
            data = data.drop(cols_to_drop, axis = 1)
            return data
        
        except Exception as e:
            logging.error("Error in pre-processing Data: {}".format(e))
            raise e

class DataDivideStrategy(DataStrategy):
    """
    Strategy for dividing data into train and test
    """
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        divide data into training and testing splits
        """
        try:
            X = data.drop(["review_score"], axis = 1)
            y = data["review_score"]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)
            return X_train, X_test, y_train, y_test
        
        except Exception as e:
            logging.error("Error in dividing data: {}".format(e))
            raise e

class DataCleaning:
    """
    Class for cleaning data which preprocess the data and divides it into train and test sets
    """

    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """
        Handle data
        """

        try:
            return self.strategy.handle_data(self.data)
        
        except Exception as e:
            logging.error("Error in handling data: {}".format(e))
            raise e