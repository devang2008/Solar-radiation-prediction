import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import pandas as pd

def train_and_predict(df: pd.DataFrame) -> pd.DataFrame:
    # prepare sliding-window sequences: use past 3 hours to predict next
    seq_len = 3
    data = df.radiation.values
    X, y = [], []
    for i in range(len(data)-seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(32, input_shape=(seq_len,1)),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=50, verbose=0)
    preds, window = [], data[-seq_len:].tolist()
    for _ in df.hour:
        arr = np.array(window[-seq_len:]).reshape((1,seq_len,1))
        p = model.predict(arr, verbose=0)[0,0]
        preds.append(p)
        window.append(p)
    return pd.DataFrame({"hour": df.hour.values, "predicted": preds})
