import streamlit as st
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import altair as alt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

st.title("Yearly Quantity Prediction Report")

# Form Upload File
uploaded_file = st.file_uploader("Upload Excel File", type="xlsx")

if uploaded_file:
    # Load data from Form Upload
    data = pd.read_excel(uploaded_file, sheet_name="Rekapan")
    
    # Data Preparation for yearly summary
    data['Month Year'] = pd.to_datetime(data['Month Year'])
    data.set_index('Month Year', inplace=True)
    yearly_data = data.resample('Y').sum()
    
    # Display yearly summary
    # st.write("Yearly Summary Table")
    # st.dataframe(yearly_data[['Quantity']], width=900, height=400)
    
    # Form Prediction
    prediction_years = st.slider("Select Prediction Period (years)", 1, 5)
    
    if prediction_years:
        # Model: Double Exponential Smoothing Prediction
        try:
            quantity_data = yearly_data['Quantity']
            model = ExponentialSmoothing(quantity_data, trend="add", seasonal=None)
            fit_model = model.fit(optimized=True, use_brute=True)
            prediction = fit_model.forecast(prediction_years)
            
            # Concatenate/Merging Data Historical and Prediction
            prediction_years_idx = pd.date_range(start=quantity_data.index[-1] + pd.DateOffset(years=1), periods=prediction_years, freq='Y')
            prediction_series = pd.Series(prediction, index=prediction_years_idx)
            combined_series = pd.concat([quantity_data, prediction_series])
            
            # Dataframe
            combined_df = combined_series.reset_index()
            combined_df.columns = ['Year', 'Quantity']
            combined_df['Type'] = ['Historical'] * len(quantity_data) + ['Prediction'] * prediction_years
            
            # Format the 'Year' column to show only the year part
            combined_df['Year'] = combined_df['Year'].dt.strftime('%Y')
            
            # Table
            st.write(f"Data Table - Yearly Quantity Prediction")
            st.dataframe(combined_df, width=900, height=400)
            
            # Chart
            chart_title = f"Predicted Quantity for next {prediction_years} years"
            chart = alt.Chart(combined_df).mark_line().encode(
                x=alt.X('Year:T', title='Year', axis=alt.Axis(format='%Y', tickCount='year', labelAngle=-45)),
                y=alt.Y('Quantity', title='Quantity'),
                color=alt.Color('Type:N', legend=alt.Legend(orient='bottom'))
            ).properties(
                width=800,
                height=400,
                title=chart_title
            )
            st.altair_chart(chart)

            # Calculate Forecasting Errors
            if len(quantity_data) >= prediction_years:
                actual = quantity_data[-prediction_years:]
                predicted = prediction[:len(actual)]
                
                mad = mean_absolute_error(actual, predicted)
                mse = mean_squared_error(actual, predicted)
                if not (actual == 0).any():
                    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
                else:
                    mape = np.nan
                    st.warning("MAPE calculation skipped due to zero values in actual data.")

                st.write(f"Forecasting Errors for Yearly Prediction")
                st.write(f"Mean Absolute Deviation (MAD): {mad:.2f}")
                st.write(f"Mean Squared Error (MSE): {mse:.2f}")
                if not np.isnan(mape):
                    st.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
                else:
                    st.write("Mean Absolute Percentage Error (MAPE): Cannot be calculated due to zero values in actual data.")
            else:
                st.write("Not enough historical data to calculate forecasting errors.")
                
        except Exception as e:
            st.error(f"An error occurred: {e}")
