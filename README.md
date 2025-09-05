# Air Quality Index (AQI) Prediction and Forecasting Using Machine Learning

This project provides a complete solution for both predicting the current Air Quality Index (AQI) based on pollutant concentrations and forecasting future AQI values using time-series analysis. The models are served through an interactive web interface built with **FastAPI**.

## 🌟 Features

  * **Real-time AQI Prediction**: Predicts the current day's AQI using key pollutant concentrations (`PM2.5`, `PM10`, `NO`, `NO2`, `NOx`, `NH3`).
  * **7-Day AQI Forecasting**: Forecasts AQI for the next seven days based on historical time-series data. The model uses lag features to capture temporal dependencies.
  * **Interactive Web Interface**: A user-friendly UI built with **FastAPI**, **HTML**, and **Tailwind CSS** allows for easy interaction with both the prediction and forecasting models.
  * **Dynamic Forecasting**: Users can select any future start date to generate a 7-day forecast, which is then visualized in a chart.
  * **Advanced Model Optimization**: Leverages **Optuna** for efficient hyperparameter tuning to maximize the performance of the XGBoost models.
  * **Data Visualization**: Displays forecast results dynamically in an interactive line chart using **Chart.js**.

-----

## ⚙️ Project Structure

The project is organized into several key files:

  * `main.py`: The FastAPI backend that loads the trained models and serves the API endpoints for prediction and forecasting.
  * `notebooks/Air_Quality.ipynb`: A Jupyter Notebook detailing the exploratory data analysis (EDA), preprocessing, training, and evaluation of the AQI **prediction** model.
  * `notebooks/forecasting.ipynb`: A Jupyter Notebook that covers the entire workflow for building the time-series **forecasting** model using lag features.
  * `templates/index.html`: The single-page frontend template that provides the user interface for interacting with the models.
  * `/models`: This directory stores the serialized (saved) machine learning models (`aqi_pipeline.joblib` and `aqi_forecast_xgb.joblib`).
  * `/data/raw`: Contains the raw dataset (`city_day.csv`).

-----

## 🤖 Models Overview

Two distinct machine learning models power this application.

### 1\. AQI Prediction Model

This model estimates the current AQI value based on real-time pollutant data.

  * **Model**: **XGBoost Regressor** was chosen after comparing its performance against Linear Regression and Random Forest.
  * **Features Used**: `PM2.5`, `PM10`, `NO`, `NO2`, `NOx`, `NH3`, and one-hot encoded `City`.
  * **Preprocessing**: The training process involved a robust pipeline including:
      * Handling outliers using the interquartile range (IQR) method.
      * Imputing missing values using the mean of each feature.
      * Scaling numerical features with `StandardScaler`.
  * **Tuning**: Hyperparameters were fine-tuned using **Optuna** to achieve the best possible performance, reaching an **R² score of approximately 0.86**.

### 2\. AQI Forecasting Model

This model predicts future AQI values by learning from historical patterns.

  * **Model**: An **XGBoost Regressor** is used to solve the time-series problem as a supervised learning task.
  * **Feature Engineering**: The model's primary features are **lag features**, which are the AQI values from the previous 7 days. This allows the model to predict the next day's AQI based on the most recent trend.
  * **Methodology**: It employs an iterative, multi-step forecasting strategy. To predict 7 days ahead, it predicts one day at a time and uses that prediction as input for the next day's forecast.
  * **Tuning**: **Optuna** was used to minimize the Root Mean Squared Error (RMSE) during cross-validation, resulting in a highly accurate forecasting model.

-----

## 💻 Technology Stack

  * **Backend**: FastAPI, Python
  * **Data Science & ML**: Pandas, Scikit-learn, XGBoost, Optuna, Joblib
  * **Frontend**: HTML, Tailwind CSS, Chart.js
  * **Server**: Uvicorn

-----

## 🚀 Setup and Installation

To run this project locally, follow these steps:

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Anand-b-patil/Air-Quality-Prediction-and-Forecasting-using-Machine-Learning.git
    cd Air-Quality-Prediction-and-Forecasting-using-Machine-Learning
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    # For Windows
    python -m venv myenv
    myenv\Scripts\activate

    # For macOS/Linux
    python3 -m venv myenv
    source myenv/bin/activate
    ```

3.  **Install the required dependencies:**

    ```bash
    pip install requirements.txt
    ```

4.  **Run the FastAPI server:**

    ```bash
    uvicorn main:app --reload
    ```

5.  **Access the application** by opening your web browser and navigating to `http://127.0.0.1:8000`.

-----

## 🛠️ How to Use

The web interface is divided into two sections:

### To Predict Current AQI:

1.  Navigate to the **"Predict Current AQI"** card on the left.
2.  Enter the values for the six required pollutants (`PM2.5`, `PM10`, etc.).
3.  Click the **"Predict AQI"** button.
4.  The predicted AQI value and its corresponding health category (e.g., "Moderate", "Unhealthy") will be displayed below the form.

### To Forecast Future AQI:

1.  Go to the **"Forecast AQI by Start Date"** card on the right.
2.  Click on the date input field and select a future date from which you want the 7-day forecast to begin.
3.  Click the **"Get Date-based Forecast"** button.
4.  The results will appear below as a table and an interactive line chart showing the forecasted AQI trend for the next 7 days.



