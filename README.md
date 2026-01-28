# Retail Demand Forecasting Dashboard

An end-to-end Retail Demand Forecasting project that covers data preprocessing, exploratory analysis, demand aggregation, forecasting, and an interactive Streamlit dashboard for visualization and prediction.
The project focuses on building a clean and understandable pipeline from raw retail data to actionable insights, prioritizing correctness and clarity over unnecessary complexity.

---

## Project Structure

RetailDemandForecasting/
├── retail_store_inventory.csv
├── UI.py
├── project_XGB.ipynb
├── requirements.txt
└── README.md

---

## Features

- Data cleaning and feature engineering
  - Date parsing
  - Month, year, weekday extraction
- Interactive filtering by:
  - Product
  - Store
  - Category
  - Region
  - Date range
- Demand aggregation across multiple dimensions
- KPI metrics:
  - Total Forecasted Demand
  - Total Actual Demand
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
- Time-series visualization of Forecast vs Actual demand
- Future demand prediction:
  - Next Day
  - Next Week
  - Next Month
  - Next Year
- Single-screen Streamlit dashboard layout with minimal scrolling

---

## Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- Scikit-learn
- Matplotlib (EDA)
- Seaborn (EDA)
- XGBoost (experimental modeling)

---

## Installation

### 1. Clone the repository

git clone <your-repository-url>
cd RetailDemandForecasting

### 2. Create and activate a virtual environment (recommended)

python -m venv venv
source venv/bin/activate      # Linux / Mac
venv\Scripts\activate         # Windows

### 3. Install dependencies

pip install -r requirements.txt

---

## Running the Streamlit Dashboard

streamlit run UI.py

The app will be available at:
http://localhost:8501

---

## Using the Dashboard

1. Apply filters from the sidebar (product, store, category, region, date range).
2. Select aggregation dimensions using Group By.
3. View KPI metrics, aggregated data, and Forecast vs Actual charts.
4. Use the Prediction tab to forecast future demand.
5. Refer to the Mappings tab for feature encodings.

<img width="2559" height="1439" alt="Screenshot 2026-01-28 185504" src="https://github.com/user-attachments/assets/cef2a436-3d6d-4d36-91ff-5eca7ae0be4c" />

---

## Jupyter Notebook

Retail_Demand_Forecasting.ipynb includes:

- Exploratory Data Analysis (EDA)
- Data visualization
- Baseline and experimental forecasting models
- Feature impact analysis

The notebook is intended for experimentation and analysis, while the Streamlit app focuses on interaction and presentation.
