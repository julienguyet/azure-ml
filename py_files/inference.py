
import pandas as pd
from joblib import load
from py_files.preprocess import preprocess


def make_predictions(input_data: pd.DataFrame) -> dict:
    encoder_path = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/churn-model-instance/code/Users/<folder_path_here>"
    model_path = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/churn-model-instance/code/Users/<folder_path_here>"
    
    encoder = load(encoder_path)
    model = load(model_path)
    
    preprocessed_data = preprocess(input_data)
    X = preprocessed_data.drop(columns='Churn')
    
    predictions = model.predict(X)

    
    return {'predictions': predictions}