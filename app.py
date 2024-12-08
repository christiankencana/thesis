import streamlit as st
from streamlit_navigation_bar import st_navbar


st.set_page_config(
        page_title="ACJ Forecasting App",
        page_icon="chart_with_upwards_trend",
        # page_icon=":material/edit:",
        layout="wide",
    )
st.write("# ACJ Forecasting App")

about = st.Page("about.py", title="About", icon=":material/delete:")
report_statistic = st.Page("report-statistic.py", title="Report Statistic", icon=":material/add_circle:")
report_model_month = st.Page("report-model-month.py", title="Report Month", icon=":material/add_circle:")
report_model_year = st.Page("report-model-year.py", title="Report Year", icon=":material/add_circle:")

pg = st.navigation([about, report_statistic, report_model_month, report_model_year ])
pg.run()