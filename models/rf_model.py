from sklearn.ensemble import RandomForestRegressor
import pandas as pd

def train_and_predict(df: pd.DataFrame) -> pd.DataFrame:
    X = df.hour.values.reshape(-1,1)
    y = df.radiation.values
    model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
    hours = df.hour.values
    preds = model.predict(hours.reshape(-1,1))
    return pd.DataFrame({"hour": hours, "predicted": preds})
