import pandas as pd
import joblib


def persist_local(modeler, path: str, data: pd.DataFrame) -> None:
    modeler.fit(data)
    joblib.dump(modeler, path)


def encode_from_local(path: str, data: pd.DataFrame) -> pd.DataFrame:
    modeler = joblib.load(path)
    encoded_data = modeler.transform(data)
    df = pd.DataFrame(data=encoded_data, columns=data.columns)
    return df


def preprocessing(data: pd.DataFrame):
    cols_of_interest = [
        "OverallQual",
        "GrLivArea",
        "TotalBsmtSF",
        "FullBath",
        "TotRmsAbvGrd",
        "YearBuilt",
        "Foundation",
        "CentralAir",
    ]
    cols_of_interest_cont = [
        "OverallQual",
        "GrLivArea",
        "TotalBsmtSF",
        "FullBath",
        "TotRmsAbvGrd",
        "YearBuilt",
    ]
    cols_of_interest_cat = ["Foundation", "CentralAir"]

    cont = data[cols_of_interest_cont]
    df_cont = pd.DataFrame(cont, columns=cols_of_interest_cont)

    # Drop row with missing value
    df_cont = df_cont.dropna()

    df_cat = data[cols_of_interest_cat]
    df_cat_encoded = encode_from_local("../models/encoder.joblib", df_cat)

    df_merged = df_cont.join(df_cat_encoded)

    scaled = encode_from_local("../models/scaler.joblib", df_merged)
    df = pd.DataFrame(data=scaled, columns=cols_of_interest)

    return df
