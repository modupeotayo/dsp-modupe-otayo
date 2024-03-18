import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error
import joblib
from .preprocess import preprocessing


def compute_rmsle(y_test: np.ndarray, y_pred: np.ndarray, precision: int = 2):
    rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
    return round(rmsle, precision)


def build_model(data: pd.DataFrame) -> dict[str, str]:
    # Returns a dictionary with the
    # model performances (for example {'rmse': 0.18})

    # Split train dataset into features 'X' and label 'y'
    X = preprocessing(data)  # Feature variables
    y = data["SalePrice"]  # Label variable
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=0.25
    )

    model = LinearRegression()
    model_path = "../models/model.joblib"
    model.fit(X_train, y_train)
    # Save the model
    joblib.dump(model, model_path)

    y_pred = model.predict(X_test)
    y_pred[y_pred < 0] = 0

    rmse = compute_rmsle(y_test, y_pred)

    return {"rmse": rmse}
