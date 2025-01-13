
# Medical Forecasting App

This application provides insights into medical sales and stock forecasts using various models, including `UCM`, `HoltWinters`, `Prophet`, and `SARIMA`. 

It accepts data input through CSV files or SQL queries, interactive model selection, and visualization using `Streamlit`.

---

## Setup Instructions


### 1. Environment Setup
1. Ensure **Python 3.9+** is installed.
2. Create Python Environment & activate it
   ```cmd
   python -m venv myenv
   ```
   ```cmd
   .\myenv\Scripts\activate
   ```
3. Install the required dependencies from `requirements.txt`:

   ```cmd
   pip install -r requirements.txt
   ```

### 2. Configuration
The application uses `app_config.ini` for configuration. Update the file to match your setup:

- **For CSV Data Input**:
  ```ini
  [inputfile_section]
  csv_file_path = ./Data_Sample_3.csv
  ```

- **For SQL Data Input**:
  ```ini
  [database]
  use_sql = true
  connection_string = DRIVER={ODBC Driver 17 for SQL Server};SERVER=localhost,1433;DATABASE=MedicalDB;UID=sa;PWD=YourStrong@Passw0rd
  query = SELECT * FROM MedicalData
  ```

- **Forecast Parameters**:
  ```ini
  [forecast_diff_threshold_section]
  diff_threshold = 20
  ```

---

## How to Run

1. Start the app using `Streamlit`:

   ```cmd
   streamlit run main.py
   ```

2. Access the app at `http://localhost:8501` in your browser.

---

## Usage Instructions

### 1. Load Data
- Use the **CSV file** defined in the configuration or connect to your **SQL database**.

### 2. Choose Parameters
- Select the medication and model for forecasting.
- Define the historical range (`Months`) and forecast period (`Days`).

### 3. Generate Forecast
- Select a model:
  - **UCM**: Best for trend analysis with seasonal and damped components (non-repeating pattern)
  - **HoltWinters**: Suitable for data with clear seasonality and trends with repearing pattern. 
  - **Prophet**: Ideal for time-series data with special events or external regressors like seasons
  - **SARIMA**: Optimal for seasonal data with auto-tuned parameters. Use if data is strictly seasonal.
- View forecasts, download CSVs, and visualize results.

### 4. Export Results
- Forecasted data can be downloaded for individual or all medications.

---

## When to Use Which Model?

| **Model**      | **Best For**                                                                                          |
|----------------|-------------------------------------------------------------------------------------------------------|
| `UCM`          | Long-term forecasting with seasonal and trend changes.                                                |
| `HoltWinters`  | Simple seasonality and trend patterns over time.                                                      |
| `Prophet`      | Handling seasonality, holidays, or regressors for better precision.                                   |
| `SARIMA`       | Complex seasonality and higher prediction accuracy with automated tuning. data is strictly seasonal   |
---

## Additional Notes
- Ensure the data file or SQL data includes all required columns: `transactiondate`, `medication_name`, `sales_volume`, and `stock_level`.
- Update the `forecast_diff_threshold` parameter in `app_config.ini` to customize stock threshold alerts.

---

## Troubleshooting
- Check log messages in the Streamlit interface for issues with data loading or forecasting.
- For dependency issues, ensure all required Python libraries in `requirements.txt` are installed correctly.

---
