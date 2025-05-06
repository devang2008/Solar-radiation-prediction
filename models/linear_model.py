from sklearn.linear_model import LinearRegression
import pandas as pd

def train_and_predict(df: pd.DataFrame) -> pd.DataFrame:
    X = df.hour.values.reshape(-1,1)
    y = df.radiation.values
    model = LinearRegression().fit(X, y)
    # next day hours 6–18
    hours = df.hour.values
    preds = model.predict(hours.reshape(-1,1))
    return pd.DataFrame({"hour": hours, "predicted": preds})
