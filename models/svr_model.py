from sklearn.svm import SVR
import pandas as pd

def train_and_predict(df: pd.DataFrame) -> pd.DataFrame:
    X = df.hour.values.reshape(-1,1)
    y = df.radiation.values
    model = SVR(kernel="rbf", C=100, gamma=0.1).fit(X, y)
    hours = df.hour.values
    preds = model.predict(hours.reshape(-1,1))
    return pd.DataFrame({"hour": hours, "predicted": preds})
