import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
class Evaluation(ABC):
    """
    Abstrct class defining strategy for evaluation of our models
    """
    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculates the scores for the model
        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            None
        """
        pass

class RMSE(Evaluation):
    """
    Evaluation strategy using Mean Square Error
    """

    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating RMSE")
            rmse = mean_squared_error(y_true, y_pred, squared= False)
            logging.info("RMSE: {}".format(rmse))
            return rmse
        
        except Exception as e:
            logging.error("Error in calculating RMSE: {}".format(e))
            raise e
        
class R2(Evaluation):
    """Evaluation strategy using R2 scores"""

    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating MSE")
            r2 = r2_score(y_true, y_pred)
            logging.info("R2: {}".format(r2))
            return r2
        
        except Exception as e:
            logging.error("Error in calculating R2: {}".format(e))
            raise e
        
