import streamlit as st
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import altair as alt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

st.title("Report")

# Form Upload File
uploaded_file = st.file_uploader("Upload Excel File", type="xlsx")

if uploaded_file:
    # Load data from Form Upload
    data = pd.read_excel(uploaded_file, sheet_name="Rekapan")
    
    # Form Select Customer
    customer_list = data["Customer"].unique()
    selected_customer = st.selectbox("Select Customer", customer_list)
    
    if selected_customer:
        # Filter data by selected customer
        customer_data = data[data["Customer"] == selected_customer]
        
        # Form Select Product
        item_list = customer_data["Product"].unique()
        selected_item = st.selectbox("Select Product", item_list)
        
        if selected_item:
            # Filter data by selected item
            item_data = customer_data[customer_data["Product"] == selected_item]
            # Sort Data
            item_data = item_data.sort_values("Month Year")
            
            # Form Prediction
            prediction_period = st.slider("Select Prediction Period (months)", 1, 12)
            
            if prediction_period:
                # Data Preparation
                item_data.set_index("Month Year", inplace=True)
                item_data = item_data.resample("M").sum()  # Ensure monthly frequency
                quantity_data = item_data["Quantity"]
                
                # Model : Double Exponential Smoothing Prediction
                try:
                    model = ExponentialSmoothing(quantity_data, trend="add", seasonal=None)
                    fit_model = model.fit(optimized=True, use_brute=True)
                    prediction = fit_model.forecast(prediction_period)
                    
                    # Concatenate/ Merging Data Historical and Prediction
                    prediction_dates = pd.date_range(start=quantity_data.index[-1] + pd.DateOffset(months=1), periods=prediction_period, freq='M')
                    prediction_series = pd.Series(prediction, index=prediction_dates)
                    combined_series = pd.concat([quantity_data, prediction_series])
                    
                    # Dataframe
                    combined_df = combined_series.reset_index()
                    combined_df.columns = ['Month Year', 'Quantity']
                    combined_df['Type'] = ['Historical'] * len(quantity_data) + ['Prediction'] * prediction_period
                    
                    # Table
                    st.write(f"Data Table - {selected_customer} - {selected_item}")
                    st.dataframe(combined_df, width=900, height=400)
                    
                    # Chart
                    chart_title = f"Predicted Quantity for next {prediction_period} months: {selected_customer} - {selected_item}"
                    chart = alt.Chart(combined_df).mark_line().encode(
                        x=alt.X('Month Year:T', title='Month Year', axis=alt.Axis(format='%Y-%m', tickCount='month', labelAngle=-45)),
                        y=alt.Y('Quantity', title='Quantity'),
                        color=alt.Color('Type:N', legend=alt.Legend(orient='bottom'))
                    ).properties(
                        width=800,
                        height=400,
                        title=chart_title
                    )
                    st.altair_chart(chart)

                    # Calculate Forecasting Errors
                    actual = quantity_data[-prediction_period:]
                    predicted = prediction[:len(actual)]
                    
                    mad = mean_absolute_error(actual, predicted)
                    mse = mean_squared_error(actual, predicted)
                    mape = np.mean(np.abs((actual - predicted) / actual)) * 100

                    st.write(f"Forecasting Errors for {selected_customer} - {selected_item}")
                    st.write(f"Mean Absolute Deviation (MAD): {mad:.2f}")
                    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
                    st.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
