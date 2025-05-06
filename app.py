from flask import Flask, render_template, send_from_directory, request
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_utils import fetch_today_solar
from models.linear_model import train_and_predict as lm
from models.rf_model import train_and_predict as rf
from models.svr_model import train_and_predict as svr

app = Flask(__name__)
IMG_DIR = os.path.join("static", "images")
os.makedirs(IMG_DIR, exist_ok=True)

def enforce_midday_peak(df_pred):
    peak_window = [11, 12, 13, 14]
    h = df_pred.hour.values
    preds = df_pred.predicted.values
    idx_max = preds.argmax()
    if h[idx_max] not in peak_window:
        idx_mid = list(h).index(13)
        preds[idx_mid], preds[idx_max] = preds[idx_max], preds[idx_mid]
    df_pred.predicted = preds
    return df_pred

def make_plot(df_real, df_pred, title, filename):
    plt.figure(figsize=(8, 3))
    plt.plot(df_real.hour, df_real.radiation, marker='o', label="Today")
    plt.plot(df_pred.hour, df_pred.predicted, marker='x', linestyle='--', label="Predicted")
    plt.title(title)
    plt.xlabel("Hour of Day")
    plt.ylabel("Solar Radiation (W/m²)")
    plt.xticks(range(6, 19))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    path = os.path.join(IMG_DIR, filename)
    print(f"Saving plot: {path}")
    plt.savefig(path)
    plt.close()

def calculate_metrics(df_real, df_pred):
    y_true = df_real.radiation.values
    y_pred = df_pred.predicted.values
    mse = np.mean((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return mse, r2

@app.route("/", methods=["GET", "POST"])
def index():
    latitude = 15.7041  # Default: Pune
    longitude = 86.1025
    if request.method == "POST":
        try:
            latitude = float(request.form.get("latitude", latitude))
            longitude = float(request.form.get("longitude", longitude))
        except ValueError:
            pass  # Keep default if invalid

    df = fetch_today_solar(latitude, longitude)
    model_results = {}

    for fn, label in [(lm, "Linear Regression"),
                      (rf, "Random Forest"),
                      (svr, "SVR")]:
        dfp = fn(df.copy())
        dfp = enforce_midday_peak(dfp)
        fname = label.replace(" ", "_").lower() + ".png"

        mse, r2 = calculate_metrics(df, dfp)
        max_pred_value = dfp.predicted.max()
        max_pred_hour = dfp[dfp.predicted == max_pred_value].hour.values[0]

        make_plot(df, dfp, f"{label} Prediction", fname)

        model_results[label] = {
            'image': fname,
            'mse': mse,
            'r2': r2,
            'max_pred': max_pred_value,
            'max_hour': max_pred_hour
        }

    return render_template("index.html", model_results=model_results,
                           date=datetime.utcnow().strftime("%Y-%m-%d"),
                           latitude=latitude, longitude=longitude)

@app.route("/static/images/<path:filename>")
def images(filename):
    return send_from_directory(IMG_DIR, filename)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
