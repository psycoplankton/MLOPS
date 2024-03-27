import logging

import pandas as pd
from zenml import step

class IngestData():
    def __init__(self, data_path: str):
        """
        Ingesting data from datapath
        """
        self.data_path = data_path

    def get_data(self):
        logging.info(f"Ingesting data from {self.data_path}")
        return pd.read_csv(self.data_path)

#`@step` is a decorator that converts its function into a step that can be used within a pipeline 
@step
def ingest_df(data_path: str) -> pd.DataFrame: #ingest_data is a function and it returns a pd.DataFrame object
    """
    Ingest data from directory
    
    Args:
        data_path: (str) = path to the dataset file.
    Returns:
        A pandas DataFrame object.
    
    """
    try:
        ingest_data = IngestData(data_path)
        df = ingest_data.get_data()
        return df
    except Exception as e:
        logging.error(f"Error while ingesting data: {e}")
        raise e

