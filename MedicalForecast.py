import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import statsmodels.api as sm
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
import configparser
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Read configuration file
config = configparser.ConfigParser()
config.read('app_config.ini')

def cleandata_prepare_exog_endog_parameters(data_field, df):
    """
    Clean data & preprocesses for modeling by separating dependent (endogenous) and independent (exogenous) variables 
    """
    #remove crlf from rows of string columns
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # Define required fields for training
    required_fields = [data_field, 'flu_cases', 'diarrhea_cases', 'injury_reports', 'season']
    df = df.dropna(subset=required_fields)  # Drop rows with missing values

    # Convert fields to numeric if applicable
    for col in ['flu_cases', 'diarrhea_cases', 'injury_reports']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Convert 'season' to dummy variables and define exogenous variables
    exog = pd.get_dummies(df[['flu_cases', 'diarrhea_cases', 'injury_reports', 'season']], drop_first=True)
    exog = exog.astype(int)  # Convert types to int as required for models

    # Define dependent variable
    endog = df[data_field]

    # Set 0 to NaN values for each column in df
    df = df.apply(lambda col: col.fillna(0) if col.isnull().any() else col)

    # Set 0 to NaN values for each column in exog
    exog = exog.apply(lambda col: col.fillna(0) if col.isnull().any() else col)

    return endog, exog, df

def generate_medical_forecast_graph(historical_slider_value, forecast_slider_value, historical_file_df, 
                                    modelName, sales_field_for_forecast, stock_field_for_forecast, selected_medication, plot_historical_values):
    try:
        #contret to proper datetime format, use this DF for training model
        historical_file_df = historical_file_df[historical_file_df['medication_name'] == selected_medication]
        historical_file_df.loc[:, 'transactiondate'] = pd.to_datetime(historical_file_df['transactiondate'], errors='coerce', dayfirst=True)
        historical_file_df = historical_file_df.sort_values(by='transactiondate')
        historical_file_df.set_index('transactiondate', inplace=True) # we can access dataframe with 'date' field directly
        historical_file_df.index.freq = 'D'  # Set frequency to daily
        
        # Logic for UCM model
        if modelName == 'UCM':

            def forecast_model(data_field, df):
                # Preprocess data
                endog, exog, df = cleandata_prepare_exog_endog_parameters(data_field, df)

                # Create the UCM model
                model = UnobservedComponents(
                    endog, # dependent variables
                    level='local linear trend',  # linear trend data: slowly changing trend over time
                    seasonal=12,  # Monthly progressing data 
                    damped_trend=True,  # gradually flatten over a time, rather than continuing indefinitely in a linear direction
                    exog=exog #independent variables
                )

                # Fit the model
                results = model.fit()

                # Extract fitted values
                fitted_values = results.fittedvalues

                # Generate forecast
                forecast = results.get_forecast(steps=forecast_slider_value, exog=exog.iloc[-forecast_slider_value:])
                forecast_mean = forecast.predicted_mean

                # Generate forecast dates
                last_date = df.index[-1]
                forecast_index = [last_date + pd.DateOffset(days=i) for i in range(1, forecast_slider_value + 1)]
                forecast_mean.index = forecast_index

                return results, forecast_mean, fitted_values

        # Logic for Holt-Winters model
        elif modelName == 'HoltWinters':

            def forecast_model(data_field, df):
                # Preprocess data
                endog, exog, df = cleandata_prepare_exog_endog_parameters(data_field, df)

                # Ensure enough data is available for forecasting
                if len(endog) < forecast_slider_value:
                    raise ValueError(f"Insufficient data for forecast. Available data length: {len(endog)}")

                # Fit the Holt-Winters model on the dependent variable
                # Holt-winters do not support exog variables out of the box, hence we have a alrernate approach
                hw_model = ExponentialSmoothing(endog, trend='add', seasonal='add', seasonal_periods=12, damped_trend=True)
                hw_results = hw_model.fit()

                # Extract Holt-Winters fitted values
                hw_fitted_values = hw_results.fittedvalues

                # Residuals (observed/dependent values - fitted values) from Holt-Winters model
                residuals = endog - hw_fitted_values

                # Fit a regression model on residuals using exogenous variables
                from sklearn.linear_model import LinearRegression
                regression_model = LinearRegression()
                regression_model.fit(exog, residuals)

                # Forecast with Holt-Winters
                hw_forecast = hw_results.forecast(forecast_slider_value)

                # Forecast adjustment using exog
                # get last forecast_slider_value rows from exog 
                exog_forecast = exog.iloc[-forecast_slider_value:]
                regression_adjustment = regression_model.predict(exog_forecast)

                # Adjust Holt-Winters forecast
                adjusted_forecast = hw_forecast + regression_adjustment

                # Ensure non-negative forecasts
                adjusted_forecast[adjusted_forecast < 0] = 0

                # Generate forecast dates
                last_date = df.index[-1]
                forecast_index = [last_date + pd.DateOffset(days=i) for i in range(1, forecast_slider_value + 1)]
                adjusted_forecast.index = forecast_index

                return hw_results, adjusted_forecast, hw_fitted_values
            
        #Prophet model
        if modelName == 'Prophet':

            def forecast_model(data_field, df):
                # Preprocess data using the shared function
                endog, exog, df = cleandata_prepare_exog_endog_parameters(data_field, df)

                # Prepare the Prophet dataframe using `endog`
                prophet_df = pd.DataFrame({
                    'ds': df.index,  # Date as 'ds'
                    'y': endog       # Use the `endog` variable as 'y'
                })

                prophet_df.reset_index(drop=True, inplace=True)
                exog.reset_index(drop=True, inplace=True)

                # Check lengths to ensure they match
                if len(prophet_df) != len(exog):
                    raise ValueError(f"Mismatch in lengths: prophet_df ({len(prophet_df)}) vs exog ({len(exog)})")

                # Merge DataFrames
                prophet_df = pd.concat([prophet_df, exog], axis=1)

                # Create and configure the Prophet model
                model = Prophet()

                # Add each exogenous variable to the Prophet model
                for col in exog.columns:
                    model.add_regressor(col)

                # Train the Prophet model
                model.fit(prophet_df)

                # Create future dataframe for prediction
                future = model.make_future_dataframe(periods=forecast_slider_value, freq='D')

                # Add exogenous variables for future periods
                # get last forecast_slider_value records from exog 
                exog_for_forecast = exog.iloc[-forecast_slider_value:].reset_index(drop=True)

                future.reset_index(drop=True, inplace=True)
                exog_for_forecast.reset_index(drop=True, inplace=True)

                future = pd.concat([future, exog_for_forecast], axis=1)

                # Set 0 to NaN values for each column in future
                future = future.apply(lambda col: col.fillna(0) if col.isnull().any() else col)

                # Forecast with the model
                forecast = model.predict(future)

                # Ensure non-negative forecast values
                forecast['yhat'] = forecast['yhat'].apply(lambda x: max(0, x))

                # Extract forecasted values
                forecast_mean = forecast[['ds', 'yhat']].set_index('ds')['yhat']

                # Make sure forecast starts after the last date in the historical data
                forecast_mean = forecast_mean[forecast_mean.index > df.index[-1]]

                # Get fitted values
                fitted_values = forecast[['ds', 'yhat']].set_index('ds')['yhat'][:len(prophet_df)]

                return model, forecast_mean, fitted_values

        elif modelName == 'SARIMA':
            def forecast_model(data_field, df):
                # Preprocess data using the shared function
                endog, exog, df = cleandata_prepare_exog_endog_parameters(data_field, df)

                # Automatically determine p, d, q, and seasonal parameters P, D, Q, m using auto_arima
                auto_sarima_model = auto_arima(
                    endog,
                    exogenous=exog,        # Include exogenous (indepedent) variables
                    seasonal=True,         # Enable seasonality
                    m=12,                  # Seasonal period (e.g., 12 for monthly data)
                    trace=True,            # Print progress for diagnostics
                    error_action='ignore',  # Ignore errors to ensure robustness
                    suppress_warnings=True,
                    stepwise=True           # Use stepwise search to speed up the process
                )

                # Extract best parameters
                p, d, q = auto_sarima_model.order
                P, D, Q, m = auto_sarima_model.seasonal_order

                # Fit the SARIMAX model
                model = SARIMAX(
                    endog,
                    exog=exog,             # Include exogenous variables for training
                    order=(p, d, q),
                    seasonal_order=(P, D, Q, m),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                sarima_result = model.fit(disp=False)

                # Get fitted values for the historical data
                fitted_values = sarima_result.predict(start=0, end=len(endog) - 1, exog=exog)

                # Forecast future values
                forecast_steps = forecast_slider_value  # Number of steps to forecast

                # Prepare future exogenous variables for forecasting
                exog_forecast = exog.iloc[-forecast_steps:].reset_index(drop=True)

                forecast = sarima_result.get_forecast(steps=forecast_steps, exog=exog_forecast)

                # Extract forecast mean and ensure non-negative values
                forecast_mean = forecast.predicted_mean.apply(lambda x: max(0, x))

                # Create a DataFrame to align with the Prophet structure
                forecast_index = pd.date_range(
                    start=df.index[-1] + pd.Timedelta(days=1),
                    periods=forecast_steps,
                    freq='D'  # Frequency of the forecast (adjust based on your data)
                )
                forecast_mean = pd.Series(forecast_mean.values, index=forecast_index)

                return model, forecast_mean, fitted_values


        # Generate forecasts for sales and stock
        # here we are passing complete historical data from historical_file_df. ALERT: it will take more time for training
        results_sales, forecast_sales_mean, sales_fitted_values = forecast_model(sales_field_for_forecast, historical_file_df)
        results_stock, forecast_stock_mean, stock_fitted_values = forecast_model(stock_field_for_forecast, historical_file_df)

        # Prepare DataFrames for plotting graph
        sales_fitted_df = pd.DataFrame({'date': sales_fitted_values.index, 'sales_fitted': sales_fitted_values})
        stock_fitted_df = pd.DataFrame({'date': stock_fitted_values.index, 'stock_fitted': stock_fitted_values})

        # Prepare DataFrames for CSV export of forecast data
        sales_forecast_df = pd.DataFrame({'date': forecast_sales_mean.index, 'sales_forecast': forecast_sales_mean})
        stock_forecast_df = pd.DataFrame({'date': forecast_stock_mean.index, 'stock_forecast': forecast_stock_mean})
        sale_stock_forecast_df = pd.merge(sales_forecast_df, stock_forecast_df, on='date')

        #filter for plot duration of existing sale, stock values and fitted values
        # If available data is less than requested months, adjust historical_slider_value
        now = datetime.now()
        available_months = (now - historical_file_df.index.min()).days // 30
        historical_slider_value = min(historical_slider_value, available_months)
        # Filter data again based on adjusted historical_slider_value
        months_ago_adjusted = now - relativedelta(months=historical_slider_value)
        sales_fitted_df = sales_fitted_df[sales_fitted_df.index >= months_ago_adjusted]
        stock_fitted_df = stock_fitted_df[stock_fitted_df.index >= months_ago_adjusted]
        historical_file_df = historical_file_df[historical_file_df.index >= months_ago_adjusted]

        # Calculate points where the difference is greater than the threshold
        diff_threshold = int(config.get('forecast_diff_threshold_section', 'diff_threshold', fallback=100))
        sale_stock_forecast_df['difference'] = sale_stock_forecast_df['stock_forecast'] - sale_stock_forecast_df['sales_forecast']
        green_dot_points = sale_stock_forecast_df[sale_stock_forecast_df['difference'] > diff_threshold]

        # Add a new column 'medication_name' with the value of selected_medication
        sale_stock_forecast_df['medication_name'] = selected_medication

        # Create traces for sales and stock forecasts along with fitted values
        traces = [
            go.Scatter(
                x=forecast_sales_mean.index,
                y=forecast_sales_mean,
                mode='lines',
                name='Sales Forecast',
                line=dict(color='blue', dash='dot')
            ),
            go.Scatter(
                x=forecast_stock_mean.index,
                y=forecast_stock_mean,
                mode='lines',
                name='Stock Forecast',
                line=dict(color='orange', dash='dot')
            ),
            go.Scatter(
                x=green_dot_points['date'],
                y=green_dot_points['stock_forecast'],
                mode='markers',
                name=f'Enough Stock (>{diff_threshold})',
                marker=dict(color='green', size=4, symbol='circle')
            ),
            go.Scatter(
                x=sales_fitted_df['date'],
                y=sales_fitted_df['sales_fitted'],
                mode='lines',
                name='Fitted Sales',
                line=dict(color='aqua', dash='solid')
            ),
            go.Scatter(
                x=stock_fitted_df['date'],
                y=stock_fitted_df['stock_fitted'],
                mode='lines',
                name='Fitted Stock',
                line=dict(color='purple', dash='solid')
            )
        ]
        # Conditionally add historical values plot
        if plot_historical_values:
            # Add traces for actual values
            traces.append(
                go.Scatter(
                    x=historical_file_df[sales_field_for_forecast].index,
                    y=historical_file_df[sales_field_for_forecast],
                    mode='lines',
                    name='Actual Sales',
                    line=dict(color='white', dash='dot')  
                )
            )
            traces.append(
                go.Scatter(
                    x=historical_file_df[stock_field_for_forecast].index,
                    y=historical_file_df[stock_field_for_forecast],
                    mode='lines',
                    name='Actual Stock',
                    line=dict(color='gray', dash='dot')  
                )
            )

        # Layout
        layout = go.Layout(
            title=f'Forecast for {selected_medication} based on last {historical_slider_value} months historical data with {modelName} model',
            xaxis=dict(title='Date'),
            yaxis=dict(title='Values'),
        )

        # Create figure
        fig = go.Figure(data=traces, layout=layout)

        return sale_stock_forecast_df, fig

    except Exception as e:
        raise Exception(f"An error occurred: {e}")

def generate_medical_forecast_for_all(historical_slider_value, forecast_slider_value, historical_file_df, 
                                      modelName, sales_field_for_forecast, stock_field_for_forecast):
    all_forecasts = []  # To collect forecast data for all medications
    
    # Get all unique medications
    unique_medications = historical_file_df['medication_name'].unique()
    
    for medication in unique_medications:
        try:
            # Call generate_medical_forecast_graph for each medication
            historical_file_df_med = historical_file_df[historical_file_df['medication_name'] == medication]
            # Generate forecast data using the existing function
            forecast_df, fig = generate_medical_forecast_graph(
                historical_slider_value, forecast_slider_value, historical_file_df_med, 
                modelName, sales_field_for_forecast, stock_field_for_forecast, 
                medication, plot_historical_values=False
            )
            # Append to the combined forecasts
            all_forecasts.append(forecast_df)
        except Exception as e:
            print(f"Error forecasting for medication {medication}: {e}")
    
    # Concatenate all forecasts
    if all_forecasts:
        all_forecasts_df = pd.concat(all_forecasts, ignore_index=True)
        return all_forecasts_df
    else:
        print("No forecasts generated.")

