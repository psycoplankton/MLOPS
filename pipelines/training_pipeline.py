from zenml import pipeline

from steps.clean_data import clean_df
from steps.evaluation import evaluate_model
from steps.ingest_data import ingest_df
from steps.model_train import train_model

@pipeline(enable_cache=True)
def train_pipeline(data_path: str):
    df = ingest_df(data_path)
    X_train, X_test, y_train, y_test = clean_df(df)
    model = train_model(X_train, X_test, y_train, y_test)
    rms, r2 = evaluate_model(model, X_test, y_test)

#When cache is enabled, if there is no change in the step, then zenml uses
# the cached version of the code, to increase the efficiency.