import requests
import pandas as pd
from datetime import datetime, timedelta

def fetch_today_solar(latitude, longitude):
    """
    Fetch hourly solar radiation data for a past date using Open-Meteo API.
    Returns a DataFrame with hour (6 to 18) and radiation values.
    """
    try:
        target_date = (datetime.utcnow() - timedelta(days=10)).date().isoformat()

        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": target_date,
            "end_date": target_date,
            "hourly": "shortwave_radiation",
            "timezone": "auto"
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        json_data = response.json()
        hours = pd.to_datetime(json_data["hourly"]["time"]).hour
        radiation = json_data["hourly"]["shortwave_radiation"]
        df = pd.DataFrame({"hour": hours, "radiation": radiation})

        df = df[(df["hour"] >= 6) & (df["hour"] <= 18)].reset_index(drop=True)
        return df

    except Exception as e:
        print(f"API Error: {e}, using sample data")
        return pd.DataFrame({
            'hour': range(6, 19),
            'radiation': [0, 0, 10, 50, 150, 300, 450, 600, 700, 750, 700, 600, 450]
        })
