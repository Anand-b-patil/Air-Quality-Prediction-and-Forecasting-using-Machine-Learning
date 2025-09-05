from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np
import os

app = FastAPI()

if not os.path.exists('static'):
    os.makedirs('static')
app.mount('/static', StaticFiles(directory='static'), name='static')

templates = Jinja2Templates(directory='templates')


MODEL_PATH = 'models/aqi_pipeline.joblib'
model = None

if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        print(f"Loaded model from {MODEL_PATH}")
    except Exception as e:
        print('Failed loading model:', e)
else:
    print(f'Model file not found at {MODEL_PATH}. The app will return dummy predictions.')


@app.get('/', response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})


@app.post('/predict', response_class=HTMLResponse)
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
            aqi_pred = None
            error = str(e)
            return templates.TemplateResponse('index.html', {
                'request': request,
                'error': 'Model prediction failed: ' + error
            })
    else:
        aqi_pred = float(0.5*PM2_5 + 0.3*PM10 + 0.1*NO2 + 0.1*NO)

    category = 'Unknown'
    try:
        if aqi_pred <= 50:
            category = 'Good'
        elif aqi_pred <= 100:
            category = 'Moderate'
        elif aqi_pred <= 150:
            category = 'Unhealthy for Sensitive Groups'
        elif aqi_pred <= 200:
            category = 'Unhealthy'
        elif aqi_pred <= 300:
            category = 'Very Unhealthy'
        else:
            category = 'Hazardous'
    except Exception:
        category = 'Unknown'

    return templates.TemplateResponse('index.html', {
        'request': request,
        'prediction': round(aqi_pred, 2),
        'category': category,
        'inputs': {
            'PM2_5': PM2_5,
            'PM10': PM10,
            'NO': NO,
            'NO2': NO2,
            'NOx': NOx,
            'NH3': NH3
        }
    })


@app.post('/api/predict')
async def api_predict(payload: dict):

    required = ['PM2_5','PM10','NO','NO2','NOx','NH3']
    try:
        features = [float(payload[k]) for k in required]
    except Exception:
        return JSONResponse({'error': 'Invalid input. Required fields: ' + ','.join(required)}, status_code=400)

    features = np.array([features])
    if model is not None:
        pred = model.predict(features)
        aqi_pred = float(pred[0])
    else:
        aqi_pred = float(0.5*features[0,0] + 0.3*features[0,1] + 0.1*features[0,3] + 0.1*features[0,2])

    if aqi_pred <= 50:
        category = 'Good'
    elif aqi_pred <= 100:
        category = 'Moderate'
    elif aqi_pred <= 150:
        category = 'Unhealthy for Sensitive Groups'
    elif aqi_pred <= 200:
        category = 'Unhealthy'
    elif aqi_pred <= 300:
        category = 'Very Unhealthy'
    else:
        category = 'Hazardous'

    return {'prediction': round(aqi_pred,2), 'category': category}

