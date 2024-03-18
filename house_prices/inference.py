import pandas as pd
import numpy as np
import joblib
from .preprocess import preprocessing


def make_predictions(input_data: pd.DataFrame) -> np.ndarray:
    processed_df = preprocessing(input_data)
    # Load the model from the saved path
    model_path = "../models/model.joblib"
    loaded_model = joblib.load(model_path)
    y_pred = loaded_model.predict(processed_df)
    y_pred = pd.DataFrame(y_pred, columns=["SalePrice"])
    return y_pred
