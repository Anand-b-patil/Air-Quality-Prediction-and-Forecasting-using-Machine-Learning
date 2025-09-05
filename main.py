from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np
import os
import pandas as pd

app = FastAPI()

# Static and templates setup
if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

# -----------------------------
# Load Models
# -----------------------------
MODEL_PATH = "models/aqi_pipeline.joblib"          # current AQI model
FORECAST_MODEL_PATH = "models/aqi_forecast_xgb.joblib"  # XGBoost forecasting model

model = None
forecast_model = None

if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        print(f"Loaded AQI model from {MODEL_PATH}")
    except Exception as e:
        print("Failed loading AQI model:", e)

if os.path.exists(FORECAST_MODEL_PATH):
    try:
        forecast_model = joblib.load(FORECAST_MODEL_PATH)
        print(f"Loaded Forecast model from {FORECAST_MODEL_PATH}")
    except Exception as e:
        print("Failed loading forecast model:", e)


# -----------------------------
# Home Page
# -----------------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# -----------------------------
# Current AQI Prediction
# -----------------------------
@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request,
                  PM2_5: float = Form(...),
                  PM10: float = Form(...),
                  NO: float = Form(...),
                  NO2: float = Form(...),
                  NOx: float = Form(...),
                  NH3: float = Form(...)):

    features = np.array([[PM2_5, PM10, NO, NO2, NOx, NH3]])

    if model is not None:
        try:
            pred = model.predict(features)
            aqi_pred = float(pred[0])
        except Exception as e:
            return templates.TemplateResponse("index.html", {
                "request": request,
                "error": f"Model prediction failed: {str(e)}"
            })
    else:
        # Fallback simple formula
        aqi_pred = float(0.5*PM2_5 + 0.3*PM10 + 0.1*NO2 + 0.1*NO)

    # AQI Category
    if aqi_pred <= 50:
        category = "Good"
    elif aqi_pred <= 100:
        category = "Moderate"
    elif aqi_pred <= 150:
        category = "Unhealthy for Sensitive Groups"
    elif aqi_pred <= 200:
        category = "Unhealthy"
    elif aqi_pred <= 300:
        category = "Very Unhealthy"
    else:
        category = "Hazardous"

    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": round(aqi_pred, 2),
        "category": category,
        "inputs": {
            "PM2_5": PM2_5,
            "PM10": PM10,
            "NO": NO,
            "NO2": NO2,
            "NOx": NOx,
            "NH3": NH3
        }
    })


# -----------------------------
# API Version of Prediction
# -----------------------------
@app.post("/api/predict")
async def api_predict(payload: dict):
    required = ["PM2_5", "PM10", "NO", "NO2", "NOx", "NH3"]
    try:
        features = [float(payload[k]) for k in required]
    except Exception:
        return JSONResponse(
            {"error": "Invalid input. Required fields: " + ",".join(required)},
            status_code=400,
        )

    features = np.array([features])
    if model is not None:
        aqi_pred = float(model.predict(features)[0])
    else:
        aqi_pred = float(0.5*features[0, 0] + 0.3*features[0, 1] +
                         0.1*features[0, 3] + 0.1*features[0, 2])

    # AQI Category
    if aqi_pred <= 50:
        category = "Good"
    elif aqi_pred <= 100:
        category = "Moderate"
    elif aqi_pred <= 150:
        category = "Unhealthy for Sensitive Groups"
    elif aqi_pred <= 200:
        category = "Unhealthy"
    elif aqi_pred <= 300:
        category = "Very Unhealthy"
    else:
        category = "Hazardous"

    return {"prediction": round(aqi_pred, 2), "category": category}


# -----------------------------
# Forecasting AQI (Next 7 days)
# -----------------------------
def forecast_future(model, last_known, steps=7):
    """Generate multi-step forecasts using lag features"""
    predictions = []
    data = last_known.copy()

    for _ in range(steps):
        pred = model.predict(data.values.reshape(1, -1))[0]
        predictions.append(float(pred))

        # shift lags: move values down by 1
        data = data.shift(1)
        data.iloc[0] = pred

    return predictions


@app.get("/forecast")
async def forecast_api():
    if forecast_model is None:
        return {"error": "Forecast model not available"}

    # Load dataset to get last known lags
    df = pd.read_csv("data/raw/city_day.csv", parse_dates=["Date"]).sort_values("Date")
    df = df[["Date", "AQI"]].dropna()

    # recreate lag features
    for lag in range(1, 8):
        df[f"lag_{lag}"] = df["AQI"].shift(lag)
    df = df.dropna()

    # Get last known row
    last_known = df.drop(columns=["Date", "AQI"]).iloc[-1]

    # Forecast next 7 days
    preds = forecast_future(forecast_model, last_known, steps=7)

    return {"forecast": [round(p, 2) for p in preds]}
from datetime import datetime, timedelta

@app.post("/forecast_date")
async def forecast_by_date(payload: dict):
    """
    Forecast AQI for the next 7 days starting from a user-given date.
    payload = {"date": "2025-09-10"}
    """
    if forecast_model is None:
        return {"error": "Forecast model not available"}

    # Parse input date
    try:
        target_date = datetime.strptime(payload["date"], "%Y-%m-%d")
    except Exception:
        return {"error": "Invalid date format. Use YYYY-MM-DD"}

    # Load dataset to get last known lags
    df = pd.read_csv("data/raw/city_day.csv", parse_dates=["Date"]).sort_values("Date")
    df = df[["Date", "AQI"]].dropna()

    for lag in range(1, 8):
        df[f"lag_{lag}"] = df["AQI"].shift(lag)
    df = df.dropna()

    last_date = df["Date"].iloc[-1]
    last_known = df.drop(columns=["Date", "AQI"]).iloc[-1]

    # Calculate forecast horizon (days ahead to user date)
    horizon = (target_date - last_date).days
    if horizon < 0:
        return {"error": "Please select a future date after " + str(last_date.date())}

    # Forecast until user date + 7 days
    total_steps = horizon + 7
    preds = forecast_future(forecast_model, last_known, steps=total_steps)

    # Extract predictions for the requested 7 days
    forecast_dates = [target_date + timedelta(days=i) for i in range(7)]
    forecast_values = preds[horizon:horizon+7]

    result = [
        {"date": d.strftime("%Y-%m-%d"), "forecast_aqi": round(v, 2)}
        for d, v in zip(forecast_dates, forecast_values)
    ]

    return {"start_date": target_date.strftime("%Y-%m-%d"),
            "forecasts": result}

