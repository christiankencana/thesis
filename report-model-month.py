import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.title("Monthly Quantity Prediction Report")

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

    # Create 'bulan_tahun' column to group by month and year
    data_filtered['bulan_tahun'] = data_filtered['Tanggal'].dt.to_period('M')

    # Monthly Summary by 'bulan_tahun'
    monthly_summary = data_filtered.groupby('bulan_tahun').agg({
        'SO': 'sum'
    }).reset_index()

    # Display monthly summary
    st.write("Monthly Summary (Total SO per Month):")
    st.write(monthly_summary)

    # Set alpha value for Double Exponential Smoothing (DES)
    alpha = 0.5

    # Initialize columns for DES and error metrics
    monthly_summary['Single'] = np.nan
    monthly_summary['Double'] = np.nan
    monthly_summary['At'] = np.nan
    monthly_summary['Bt'] = np.nan
    monthly_summary['DES Forecast'] = np.nan
    monthly_summary['Error'] = np.nan
    monthly_summary['MAD'] = np.nan
    monthly_summary['MSE'] = np.nan
    monthly_summary['MAPE'] = np.nan

    # Set initial Single, Double, At, and Bt values based on the first observation
    monthly_summary.loc[0, 'Single'] = monthly_summary.loc[0, 'SO']
    monthly_summary.loc[0, 'Double'] = monthly_summary.loc[0, 'SO']
    monthly_summary.loc[0, 'At'] = (2 * monthly_summary.loc[0, 'Single']) - monthly_summary.loc[0, 'Double']
    monthly_summary.loc[0, 'Bt'] = 0  # Set initial trend (Bt) to 0

    # Apply DES formula and calculate error metrics for each subsequent row
    for i in range(1, len(monthly_summary)):
        # Single Exponential smoothing (S't)
        monthly_summary.loc[i, 'Single'] = (alpha * monthly_summary.loc[i, 'SO']) + ((1 - alpha) * monthly_summary.loc[i-1, 'Single'])
        
        # Double Exponential smoothing (S''t)
        monthly_summary.loc[i, 'Double'] = (alpha * monthly_summary.loc[i, 'Single']) + ((1 - alpha) * monthly_summary.loc[i-1, 'Double'])
        
        # At (level)
        monthly_summary.loc[i, 'At'] = (2 * monthly_summary.loc[i, 'Single']) - monthly_summary.loc[i, 'Double']
        
        # Bt (trend)
        monthly_summary.loc[i, 'Bt'] = (alpha / (1 - alpha)) * (monthly_summary.loc[i, 'Single'] - monthly_summary.loc[i, 'Double'])
        
        # DES Forecast for the next period
        monthly_summary.loc[i, 'DES Forecast'] = monthly_summary.loc[i-1, 'At'] + monthly_summary.loc[i-1, 'Bt']
        
        # Error (actual - forecast)
        monthly_summary.loc[i, 'Error'] = monthly_summary.loc[i, 'SO'] - monthly_summary.loc[i, 'DES Forecast']
        
        # MAD (Mean Absolute Deviation)
        monthly_summary.loc[i, 'MAD'] = abs(monthly_summary.loc[i, 'Error'])
        
        # MSE (Mean Squared Error)
        monthly_summary.loc[i, 'MSE'] = monthly_summary.loc[i, 'Error'] ** 2
        
        # MAPE (Mean Absolute Percentage Error)
        if monthly_summary.loc[i, 'SO'] != 0:
            monthly_summary.loc[i, 'MAPE'] = (abs(monthly_summary.loc[i, 'Error']) / monthly_summary.loc[i, 'SO']) * 100
        else:
            monthly_summary.loc[i, 'MAPE'] = np.nan

    # Show the result with DES forecast and error metrics
    st.write("Monthly DES Forecast and Error Metrics:")
    st.write(monthly_summary)

    # Forecast for future months (12 months ahead)
    months_to_forecast = 12

    # Get the last month from the original data
    last_month = monthly_summary['bulan_tahun'].max().to_timestamp()

    # Generate dummy months starting from the next month after the last month in the original data
    dummy_data = pd.DataFrame({
        'bulan_tahun': pd.date_range(start=last_month + pd.offsets.MonthBegin(1), periods=months_to_forecast, freq='M').to_period('M'),
        'SO': np.zeros(months_to_forecast)  # Set SO to 0 for future months
    })

    # Combine original data with the dummy future data
    extended_data = pd.concat([monthly_summary[['bulan_tahun', 'SO']], dummy_data], ignore_index=True)

    # Initialize columns for DES forecast in the extended data
    extended_data['Single'] = np.nan
    extended_data['Double'] = np.nan
    extended_data['At'] = np.nan
    extended_data['Bt'] = np.nan
    extended_data['DES Forecast'] = np.nan
    extended_data['Error'] = np.nan
    extended_data['MAD'] = np.nan
    extended_data['MSE'] = np.nan
    extended_data['MAPE'] = np.nan

    # Set initial values for the DES components based on the first row of the data
    extended_data.loc[0, 'Single'] = extended_data.loc[0, 'SO']
    extended_data.loc[0, 'Double'] = extended_data.loc[0, 'SO']
    extended_data.loc[0, 'At'] = (2 * extended_data.loc[0, 'Single']) - extended_data.loc[0, 'Double']
    extended_data.loc[0, 'Bt'] = 0  # Assume no initial trend

    # Apply DES formula to the entire dataset (historical and future periods)
    for i in range(1, len(extended_data)):
        # Apply DES formula for all periods (historical and future)
        extended_data.loc[i, 'Single'] = (alpha * extended_data.loc[i, 'SO']) + ((1 - alpha) * extended_data.loc[i-1, 'Single'])
        extended_data.loc[i, 'Double'] = (alpha * extended_data.loc[i, 'Single']) + ((1 - alpha) * extended_data.loc[i-1, 'Double'])
        extended_data.loc[i, 'At'] = (2 * extended_data.loc[i, 'Single']) - extended_data.loc[i, 'Double']
        extended_data.loc[i, 'Bt'] = (alpha / (1 - alpha)) * (extended_data.loc[i, 'Single'] - extended_data.loc[i, 'Double'])
        extended_data.loc[i, 'DES Forecast'] = extended_data.loc[i-1, 'At'] + extended_data.loc[i-1, 'Bt']
        
        # Error
        extended_data.loc[i, 'Error'] = extended_data.loc[i, 'SO'] - extended_data.loc[i, 'DES Forecast']
        
        # MAD (Mean Absolute Deviation)
        extended_data.loc[i, 'MAD'] = abs(extended_data.loc[i, 'Error'])
        
        # MSE (Mean Squared Error)
        extended_data.loc[i, 'MSE'] = extended_data.loc[i, 'Error'] ** 2
        
        # MAPE (Mean Absolute Percentage Error)
        if extended_data.loc[i, 'SO'] != 0:
            extended_data.loc[i, 'MAPE'] = (abs(extended_data.loc[i, 'Error']) / extended_data.loc[i, 'SO']) * 100
        else:
            extended_data.loc[i, 'MAPE'] = np.nan

    # Display the extended data with DES forecast results
    st.write("Forecasted Data with DES and Error Metrics:")
    st.write(extended_data)

    # Plot: Actual SO vs DES Forecast
    st.write("Plot: Actual SO vs DES Forecast")
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(extended_data['bulan_tahun'].astype(str), extended_data['SO'], label='Actual SO', marker='o')
    ax.plot(extended_data['bulan_tahun'].astype(str), extended_data['DES Forecast'], label='DES Forecast', marker='x')
    ax.set_xlabel('Month-Year')
    ax.set_ylabel('SO')
    ax.set_title('Actual SO vs DES Forecast')
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Plot: Error Evaluation
    st.write("Plot: Error and Evaluation Metrics")
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(extended_data['bulan_tahun'].astype(str), extended_data['Error'], label='Error', marker='o', color='red')
    ax.plot(extended_data['bulan_tahun'].astype(str), extended_data['MAD'], label='MAD', marker='x', color='blue')
    ax.plot(extended_data['bulan_tahun'].astype(str), extended_data['MSE'], label='MSE', marker='s', color='green')
    ax.plot(extended_data['bulan_tahun'].astype(str), extended_data['MAPE'], label='MAPE', marker='^', color='purple')
    ax.set_xlabel('Month-Year')
    ax.set_ylabel('Error Metrics')
    ax.set_title('Error and Evaluation Metrics (Error, MAD, MSE, MAPE)')
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)