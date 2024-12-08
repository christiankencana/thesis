import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.title("Yearly Quantity Prediction Report")

# File upload form
uploaded_file = st.file_uploader("Upload Excel File", type="xlsx")

if uploaded_file:
    # Load data from the uploaded Excel file
    data = pd.read_excel(uploaded_file, sheet_name="Rekapan")
    
    # Filter relevant columns
    data_filtered = data[['Tanggal', 'SO', 'TERKIRIM', 'Harga Komoditas Bijih Besi', 'Indeks Produksi Dalam Negeri', 'Data Inflasi', 'Kurs']]

    # Rename columns for easier handling
    data_filtered.columns = ['Tanggal', 'SO', 'Terkirim', 'Harga Komoditas', 'Indeks Produksi', 'Data Inflasi', 'Kurs']

    # Convert 'Tanggal' column to datetime
    data_filtered['Tanggal'] = pd.to_datetime(data_filtered['Tanggal'])

    # Convert 'Indeks Produksi' to numeric (coercing errors)
    data_filtered['Indeks Produksi'] = pd.to_numeric(data_filtered['Indeks Produksi'], errors='coerce')

    # Create 'Year' column to group by year
    data_filtered['Year'] = data_filtered['Tanggal'].dt.year

    # Group by Year to get yearly summary (sum of SO per year)
    yearly_summary = data_filtered.groupby('Year').agg({
        'SO': 'sum'
    }).reset_index()

    # Display yearly summary
    st.write("Yearly Summary (Total SO per Year):")
    st.write(yearly_summary)

    # Set alpha value for Triple Exponential Smoothing (TES)
    alpha = 0.1

    # Initialize columns for TES and error metrics
    yearly_summary['Single'] = np.nan
    yearly_summary['Double'] = np.nan
    yearly_summary['Triple'] = np.nan
    yearly_summary['At'] = np.nan
    yearly_summary['Bt'] = np.nan
    yearly_summary['Ct'] = np.nan
    yearly_summary['TES Forecast'] = np.nan
    yearly_summary['Error'] = np.nan
    yearly_summary['MAD'] = np.nan
    yearly_summary['MSE'] = np.nan
    yearly_summary['MAPE'] = np.nan

    # Initialize first row with initial values
    yearly_summary.loc[0, 'Single'] = yearly_summary.loc[0, 'SO']
    yearly_summary.loc[0, 'Double'] = yearly_summary.loc[0, 'SO']
    yearly_summary.loc[0, 'Triple'] = yearly_summary.loc[0, 'SO']
    yearly_summary.loc[0, 'At'] = (2 * yearly_summary.loc[0, 'Single']) - yearly_summary.loc[0, 'Double']
    yearly_summary.loc[0, 'Bt'] = 0  # Initial trend (Bt) set to 0
    yearly_summary.loc[0, 'Ct'] = 0  # Initial trend (Ct) set to 0

    # Apply TES formula and calculate error metrics for each subsequent row
    for i in range(1, len(yearly_summary)):
        # Single Exponential smoothing (S't)
        yearly_summary.loc[i, 'Single'] = (alpha * yearly_summary.loc[i, 'SO']) + ((1 - alpha) * yearly_summary.loc[i-1, 'Single'])
        
        # Double Exponential smoothing (S''t)
        yearly_summary.loc[i, 'Double'] = (alpha * yearly_summary.loc[i, 'Single']) + ((1 - alpha) * yearly_summary.loc[i-1, 'Double'])
        
        # Triple Exponential smoothing (S'''t)
        yearly_summary.loc[i, 'Triple'] = (alpha * yearly_summary.loc[i, 'Double']) + ((1 - alpha) * yearly_summary.loc[i-1, 'Triple'])
        
        # At (level)
        yearly_summary.loc[i, 'At'] = (3 * yearly_summary.loc[i, 'Single']) - (3 * yearly_summary.loc[i, 'Double']) + yearly_summary.loc[i, 'Triple']
        
        # Bt (trend)
        yearly_summary.loc[i, 'Bt'] = ((alpha) / (2*(1 - alpha)**2)) * ((6-5*alpha)*yearly_summary.loc[i, 'Single'] - (10-8*alpha)*yearly_summary.loc[i, 'Double'] + (4-3*alpha)*yearly_summary.loc[i, 'Triple']) 
        
        # Ct (seasonality)
        yearly_summary.loc[i, 'Ct'] = (alpha**2)/((1 - alpha)**2) * (yearly_summary.loc[i, 'Single'] - (2*yearly_summary.loc[i, 'Double']) + yearly_summary.loc[i, 'Triple'])
        
        # TES Forecast for the next period
        if i > 0:  # Forecast from the second row onward
            yearly_summary.loc[i, 'TES Forecast'] = yearly_summary.loc[i-1, 'At'] + yearly_summary.loc[i-1, 'Bt']
        
        # Error (actual - forecast)
        yearly_summary.loc[i, 'Error'] = yearly_summary.loc[i, 'SO'] - yearly_summary.loc[i, 'TES Forecast']
        
        # MAD (Mean Absolute Deviation)
        yearly_summary.loc[i, 'MAD'] = abs(yearly_summary.loc[i, 'Error'])
        
        # MSE (Mean Squared Error)
        yearly_summary.loc[i, 'MSE'] = yearly_summary.loc[i, 'Error'] ** 2
        
        # MAPE (Mean Absolute Percentage Error)
        if yearly_summary.loc[i, 'SO'] != 0:
            yearly_summary.loc[i, 'MAPE'] = (abs(yearly_summary.loc[i, 'Error']) / yearly_summary.loc[i, 'SO']) * 100
        else:
            yearly_summary.loc[i, 'MAPE'] = np.nan

    # Show the result with TES forecast and error metrics
    st.write("Yearly TES Forecast and Error Metrics:")
    st.write(yearly_summary)

    # Forecast for future years (6 years ahead)
    years_to_forecast = 6

    # Get the last year from the original data
    last_year = yearly_summary['Year'].max()

    # Generate dummy years starting from the next year after the last year in the original data
    dummy_years = pd.DataFrame({
        'Year': range(last_year + 1, last_year + 1 + years_to_forecast),
        'SO': np.zeros(years_to_forecast)  # Set SO to 0 for future years
    })

    # Combine original data with the dummy future data
    extended_data = pd.concat([yearly_summary[['Year', 'SO']], dummy_years], ignore_index=True)

    # Initialize columns for TES forecast in the extended data
    extended_data['Single'] = np.nan
    extended_data['Double'] = np.nan
    extended_data['Triple'] = np.nan
    extended_data['At'] = np.nan
    extended_data['Bt'] = np.nan
    extended_data['Ct'] = np.nan
    extended_data['TES Forecast'] = np.nan
    extended_data['Error'] = np.nan
    extended_data['MAD'] = np.nan
    extended_data['MSE'] = np.nan
    extended_data['MAPE'] = np.nan

    # Initialize first row with initial values for forecasting
    extended_data.loc[0, 'Single'] = extended_data.loc[0, 'SO']
    extended_data.loc[0, 'Double'] = extended_data.loc[0, 'SO']
    extended_data.loc[0, 'Triple'] = extended_data.loc[0, 'SO']
    extended_data.loc[0, 'At'] = (2 * extended_data.loc[0, 'Single']) - extended_data.loc[0, 'Double']
    extended_data.loc[0, 'Bt'] = 0  # Assume no initial trend
    extended_data.loc[0, 'Ct'] = 0  # Assume no initial trend

    # Apply TES formula to historical data and future periods
    for i in range(1, len(extended_data)):
        # Apply TES for historical and future periods
        extended_data.loc[i, 'Single'] = (alpha * extended_data.loc[i, 'SO']) + ((1 - alpha) * extended_data.loc[i-1, 'Single'])
        extended_data.loc[i, 'Double'] = (alpha * extended_data.loc[i, 'Single']) + ((1 - alpha) * extended_data.loc[i-1, 'Double'])
        extended_data.loc[i, 'Triple'] = (alpha * extended_data.loc[i, 'Double']) + ((1 - alpha) * extended_data.loc[i-1, 'Triple'])
        extended_data.loc[i, 'At'] = (3 * extended_data.loc[i, 'Single']) - (3 * extended_data.loc[i, 'Double']) + extended_data.loc[i, 'Triple']
        extended_data.loc[i, 'Bt'] = ((alpha) / (2*(1 - alpha)**2)) * ((6-5*alpha)*extended_data.loc[i, 'Single'] - (10-8*alpha)*extended_data.loc[i, 'Double'] + (4-3*alpha)*extended_data.loc[i, 'Triple']) 
        extended_data.loc[i, 'Ct'] = (alpha**2)/((1 - alpha)**2) * (extended_data.loc[i, 'Single'] - (2*extended_data.loc[i, 'Double']) + extended_data.loc[i, 'Triple'])
        extended_data.loc[i, 'TES Forecast'] = extended_data.loc[i-1, 'At'] + extended_data.loc[i-1, 'Bt']
        
        # Error metrics
        extended_data.loc[i, 'Error'] = extended_data.loc[i, 'SO'] - extended_data.loc[i, 'TES Forecast']
        extended_data.loc[i, 'MAD'] = abs(extended_data.loc[i, 'Error'])
        extended_data.loc[i, 'MSE'] = extended_data.loc[i, 'Error'] ** 2
        if extended_data.loc[i, 'SO'] != 0:
            extended_data.loc[i, 'MAPE'] = (abs(extended_data.loc[i, 'Error']) / extended_data.loc[i, 'SO']) * 100
        else:
            extended_data.loc[i, 'MAPE'] = np.nan

    # Display the extended data with future forecasts
    st.write("Forecasted Data with TES and Error Metrics:")
    st.write(extended_data)

    # Plot: Actual SO vs TES Forecast
    st.write("Plot: Actual SO vs TES Forecast")
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(extended_data['Year'], extended_data['SO'], label='Actual SO', marker='o')
    ax.plot(extended_data['Year'], extended_data['TES Forecast'], label='TES Forecast', marker='x')
    ax.set_xlabel('Year')
    ax.set_ylabel('SO')
    ax.set_title('Actual SO vs TES Forecast')
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Plot: Error Evaluation
    st.write("Plot: Error and Evaluation Metrics")
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(extended_data['Year'].astype(str), extended_data['Error'], label='Error', marker='o', color='red')
    ax.plot(extended_data['Year'].astype(str), extended_data['MAD'], label='MAD', marker='x', color='blue')
    ax.plot(extended_data['Year'].astype(str), extended_data['MSE'], label='MSE', marker='s', color='green')
    ax.plot(extended_data['Year'].astype(str), extended_data['MAPE'], label='MAPE', marker='^', color='purple')
    ax.set_xlabel('Year')
    ax.set_ylabel('Error Metrics')
    ax.set_title('Error and Evaluation Metrics (Error, MAD, MSE, MAPE)')
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)
