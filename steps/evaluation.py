import logging
from zenml import step
import pandas as pd
from src.evaluation import RMSE, R2
from sklearn.base import RegressorMixin
from typing import Tuple
from typing_extensions import Annotated
from zenml.client import Client
import mlflow

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(model: RegressorMixin,
                   X_test: pd.DataFrame,
                   y_test: pd.DataFrame
                   ) -> Tuple[
                       Annotated[float, "rmse"],
                       Annotated[float, "r2"]
                   ]:
    """
    Evaluate the model on the ingested data.
    Args:
        df: the ingested data.
    """
    try:
        prediction = model.predict(X_test)
        rmse_class = RMSE()
        rmse = rmse_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("rmse", rmse)

        r2_class = R2()
        r2 = r2_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("r2", r2)
        
    except Exception as e:
        logging.error("Error while Evaluating models: {}".format(e))
        raise e
    
    return rmse, r2