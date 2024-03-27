import sys
sys.path.append(r'C:\Users\vansh\AI and ML reading material\MLOps\OLips Customer Satisfaction Project')

from pipelines.training_pipeline import train_pipeline
from zenml.client import Client

if __name__ == "__main__":
    #Run the pipeline
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    train_pipeline(data_path=r"C:\Users\vansh\AI and ML reading material\MLOps\OLips Customer Satisfaction Project\data\olist_customers_dataset.csv")