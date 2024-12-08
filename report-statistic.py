import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Title for the Streamlit app
st.title("Statistical Report")

# Form for file upload
uploaded_file = st.file_uploader("Upload Excel File", type="xlsx")

if uploaded_file:
    # Load data from the uploaded Excel file
    data = pd.read_excel(uploaded_file, sheet_name="Rekapan")

    # Display the first and last few rows of the data
    st.write("First rows of data:", data.head())
    st.write("Last rows of data:", data.tail())

    # Display descriptive statistics for the dataset
    st.write("Descriptive Statistics for the Data:")
    data_description = data.describe().transpose()
    st.write(data_description)

    # Check for missing values
    st.write("Missing Values in the Data:")
    missing_values = data.isnull().sum()
    missing_percentage = (missing_values / len(data)) * 100
    missing_df = pd.DataFrame({
        'Number_of_NaN': missing_values,
        'Percent(%)': missing_percentage
    }).reset_index()
    missing_df.columns = ['Column', 'Number_of_NaN', 'Percent(%)']
    st.write(missing_df)

    # Filtering relevant columns for analysis
    data_filtered = data[['SO', 'TERKIRIM', 'Harga Komoditas Bijih Besi', 
                          'Indeks Produksi Dalam Negeri', 'Data Inflasi', 'Kurs']]

    # Converting relevant columns to numeric, coercing errors to NaN
    data_filtered = data_filtered.apply(pd.to_numeric, errors='coerce')

    # Calculating and displaying descriptive statistics for filtered data
    descriptive_stats = data_filtered.describe().transpose()
    descriptive_stats['count'] = data_filtered.count()
    descriptive_stats = descriptive_stats[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
    st.write("Descriptive Statistics for Filtered Data:")
    st.write(descriptive_stats)

    # Visualize the descriptive statistics using a heatmap
    st.write("Heatmap of Descriptive Statistics:")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(descriptive_stats, annot=True, fmt=".2f", cmap="Reds", cbar=False, ax=ax, 
                linewidths=0.5, linecolor='gray', xticklabels=True, yticklabels=True)
    ax.xaxis.tick_top()
    ax.set_title('Statistik Deskriptif Data', fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='left')
    st.pyplot(fig)

    # Plot histogram distribution of the original data
    st.write("Histogram Distribution of Data:")
    fig, ax = plt.subplots(figsize=(20, 15))
    data.hist(bins=50, ax=ax)
    st.pyplot(fig)

    # Calculate the correlation matrix for numeric columns
    st.write("Correlation Matrix for Numeric Data:")
    numeric_df = data.select_dtypes(include=['float64', 'int64'])
    correlation_matrix = numeric_df.corr()
    
    # Plot correlation matrix heatmap
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='inferno', vmin=0.1, vmax=1, 
                linewidths=0.5, linecolor='gray')
    ax.set_title('Correlation Matrix', fontsize=16, fontweight='bold')
    st.pyplot(fig)
