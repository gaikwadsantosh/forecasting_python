import streamlit as st
from MedicalForecast import generate_medical_forecast_graph
from MedicalForecast import generate_medical_forecast_for_all
import pandas as pd
from datetime import datetime
import os
import configparser
import plotly.graph_objects as go
import pyodbc

def load_data_from_csv(file_path):
    return pd.read_csv(file_path)

def load_data_from_sql(connection_string, query):
    try:
        conn = pyodbc.connect(connection_string)
        data = pd.read_sql_query(query, conn)
        conn.close()
        return data
    except Exception as e:
        st.error(f"Error loading data from SQL: {e}")
        return pd.DataFrame()  # Return empty DataFrame in case of error
    
def main():
    config = configparser.ConfigParser()
    config.read('app_config.ini')

    # Read configurations
    use_sql = config.getboolean('database', 'use_sql', fallback=False)
    connection_string = config.get('database', 'connection_string', fallback='')
    query = config.get('database', 'query', fallback='')

    # Load data conditionally
    if use_sql:
        historical_file_df = load_data_from_sql(connection_string, query)
    else:
        file_path = config.get('inputfile_section', 'csv_file_path', fallback='./SampleDataFile.csv')
        historical_file_df = load_data_from_csv(file_path)

    # Validate the loaded data
    if historical_file_df.empty:
        st.error("No data loaded. Please check your configuration and source.")
        return

    st.set_page_config(
        page_title="Forecast Medicine",
        page_icon= ":telescope:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    headerSection = st.container()
    mainSection = st.container()
    leftSection = st.sidebar

    with headerSection:
        st.header("Medical Forecast System")

    with leftSection:
        missing_fields = []
        if 'transactiondate' not in historical_file_df.columns:
            missing_fields.append('transactiondate')
        if missing_fields:
            error_message = (
                f"{', '.join(missing_fields)} field(s) are missing in the CSV file.<br>"
                "NOTE: field names are case sensitive"
            )
            raise KeyError(error_message)
        
        plot_historical_values = st.checkbox("Plot actual sale & stock values", False)

        historical_slider_value = st.slider('Plot fitted values for last (?) MONTHS of historical data', min_value=1, max_value=60, value=3)

        forecast_slider_value = st.slider('Forecast for  (?) Days', min_value=1, max_value=90, value=30)

        # Dropdown for selecting medication from unique values in 'medication_name' column
        medication_names = historical_file_df['medication_name'].unique()
        selected_medication = st.selectbox('Select Medication', medication_names)

        # Model selection dropdown
        model_options = ['UCM','HoltWinters','Prophet','SARIMA']
        selected_model = st.selectbox('Choose Model to forecast data', model_options)

        if st.button(f"Generate {selected_model} Forecast For All"):
            try:
                all_forecast_df = generate_medical_forecast_for_all(
                    historical_slider_value=historical_slider_value,
                    forecast_slider_value=forecast_slider_value,
                    historical_file_df=historical_file_df,
                    modelName=selected_model,
                    sales_field_for_forecast='sales_volume',
                    stock_field_for_forecast='stock_level',
                )

                if not all_forecast_df.empty:
                    csv_data = all_forecast_df.to_csv(index=False)
                    # Display success message and download button in the main section
                    with mainSection:
                        st.success(f"Forecast Generated for all medicines.")
                        st.download_button(
                            label=f"Download {selected_model} Forecast for All Medications",
                            data=csv_data,
                            mime="text/csv"
                        )
            except Exception as e:
                with mainSection:
                    st.error(f"An error occurred: {e}")

    with mainSection:
        st.markdown("&nbsp;", unsafe_allow_html=True)

        # Pass selected_medication to the forecasting function
        forecast_df, fig = generate_medical_forecast_graph(
            historical_slider_value=historical_slider_value,
            forecast_slider_value=forecast_slider_value,
            historical_file_df=historical_file_df,
            modelName=selected_model,
            sales_field_for_forecast='sales_volume',
            stock_field_for_forecast='stock_level',
            selected_medication=selected_medication,
            plot_historical_values=plot_historical_values
        )

        mainSection.plotly_chart(fig, use_container_width=True)

        # Convert DataFrame to CSV and create download button
        csv_data = forecast_df.to_csv(index=False)
        st.download_button(
            label=f"Download {selected_model} Forecast CSV",
            data=csv_data,
            file_name=f"forecast_{selected_model}.csv",
            mime='text/csv'
        )

# Global error handling
try:
    main()
except Exception as e:
    st.markdown(
        f"""
        <div style="background-color: red; color:white; padding: 10px; border-radius: 5px;">
            {str(e)}
        </div>
        """,
        unsafe_allow_html=True
    )
    with open(os.path.join(os.path.dirname(__file__), 'SOP.md'), "r") as file: 
        markdown_text = file.read()
    st.markdown(markdown_text)
