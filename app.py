import streamlit as st
from streamlit_navigation_bar import st_navbar


st.set_page_config(
        page_title="ACJ Forecasting App",
        page_icon="chart_with_upwards_trend",
        # page_icon=":material/edit:",
        layout="wide",
    )
st.write("# ACJ Forecasting App")

report_year = st.Page("report-year.py", title="Report Year", icon=":material/add_circle:")
# report_month = st.Page("report-month.py", title="Report", icon=":material/add_circle:")
report_detail = st.Page("report-detail.py", title="Report Detail", icon=":material/add_circle:")
about = st.Page("about.py", title="About", icon=":material/delete:")

pg = st.navigation([report_detail, report_year, about])
pg.run()