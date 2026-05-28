import streamlit as st
import plotly.express as px
import polars as pl

st.set_page_config(
    page_title="Agentic Behavior Dashboard",
    layout="wide"
)

st.title("Agent Behavior Dashboard")

# Dummy Data
df = pl.DataFrame({
    "Agent": ["Order", "Barista", "Inventory", "Customer Service"],
    "Count": [120, 95, 80, 75]
})

# KPI Row
col1, col2, col3 = st.columns(3)

col1.metric("Cases", "1,245", "+12%")
col2.metric("Avg Throughput", "4.2 days", "-8%")
col3.metric("SLA Compliance", "91%", "+3%")

# Chart
fig = px.bar(df, x="Agent", y="Count", title="Activity Frequency")

st.plotly_chart(fig, width="stretch")