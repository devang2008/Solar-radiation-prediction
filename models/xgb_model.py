import pandas as pd
import numpy as np
import xgboost as xgb

def train_and_predict(df: pd.DataFrame) -> pd.DataFrame:
    seq_len = 3
    data = df.radiation.values
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len])
    X, y = np.array(X), np.array(y)

    # XGBoost model
    model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100)
    model.fit(X, y)

    # Make predictions for the next hours
    preds = []
    window = data[-seq_len:].tolist()
    for _ in df.hour:
        arr = np.array(window[-seq_len:]).reshape(1, -1)
        p = model.predict(arr)[0]
        preds.append(p)
        window.append(p)

    # Return predictions as a DataFrame
    return pd.DataFrame({"hour": df.hour.values, "predicted": preds})
