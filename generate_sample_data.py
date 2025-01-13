import pandas as pd
import numpy as np
from datetime import timedelta

# Load the sample file
input_file_path = "SampleDataFile.csv"  # Replace with your input file path
output_file_path = "UpdatedSampleDataWithSeason.csv"  # Path to save the updated file

# Read the CSV file
df = pd.read_csv(input_file_path)
df['transactiondate'] = pd.to_datetime(df['transactiondate'], errors='coerce', dayfirst=True)

# Ensure transactiondate is within the last 5 years and less than today
end_date = pd.Timestamp.today()
start_date = end_date - timedelta(days=5 * 365)

# Generate a date range of the last 5 years
all_dates = pd.date_range(start=start_date, end=end_date)

# Define random pharmacy locations and their states
pharmacy_data = [
    ("Mumbai", "Maharashtra"), ("Delhi", "Delhi"), ("Bangalore", "Karnataka"),
    ("Hyderabad", "Telangana"), ("Ahmedabad", "Gujarat"), ("Chennai", "Tamil Nadu"),
    ("Kolkata", "West Bengal"), ("Pune", "Maharashtra"), ("Jaipur", "Rajasthan"),
    ("Lucknow", "Uttar Pradesh")
]

# Create a DataFrame for each medication and date combination
medication_names = df['medication_name'].unique()
new_data = []
for medication in medication_names:
    for date in all_dates:
        location, state = pharmacy_data[np.random.randint(0, len(pharmacy_data))]
        new_data.append({
            'transactiondate': date,
            'medication_name': medication,
            'pharmacy_location': location,
            'state': state
        })

expanded_df = pd.DataFrame(new_data)

# Determine the season based on the date
def get_season(date):
    month = date.month
    if month in [12, 1, 2]:  # Winter
        return 'Winter'
    elif month in [3, 4, 5]:  # Summer
        return 'Summer'
    else:  # Monsoon (June-November)
        return 'Monsoon'

expanded_df['season'] = expanded_df['transactiondate'].apply(get_season)

# Generate sales and stock with trends, seasonality, and noise
np.random.seed(42)
days = (expanded_df['transactiondate'] - expanded_df['transactiondate'].min()).dt.days
seasonal_effect = 50 * np.sin(2 * np.pi * days / 365)  # Annual seasonality
noise = np.random.normal(0, 10, len(expanded_df))

# Generate sales_volume
expanded_df['sales_volume'] = np.clip(
    100 + 0.1 * days + seasonal_effect + noise, 0, None
).astype(int)

# Generate stock_levels ensuring it is always greater than sales_volume
expanded_df['stock_levels'] = expanded_df['sales_volume'] + np.clip(
    50 + np.random.normal(10, 5, len(expanded_df)), 0, None
).astype(int)

# Additional columns
expanded_df['price_per_unit'] = np.clip(
    50 + np.random.normal(0, 5, len(expanded_df)) + 0.2 * expanded_df['sales_volume'], 5, None
).round(2)
expanded_df['flu_cases'] = np.clip(
    10 + 5 * np.sin(2 * np.pi * days / 365) + np.random.normal(0, 2, len(expanded_df)), 0, None
).astype(int)
expanded_df['diarrhea_cases'] = np.clip(
    8 + 3 * np.cos(2 * np.pi * days / 365) + np.random.normal(0, 2, len(expanded_df)), 0, None
).astype(int)
expanded_df['injury_reports'] = np.clip(
    5 + 2 * np.sin(2 * np.pi * days / 180) + np.random.normal(0, 1, len(expanded_df)), 0, None
).astype(int)

# Save the updated data
expanded_df.to_csv(output_file_path, index=False)
print(f"Updated data with additional columns and seasons saved to {output_file_path}")
