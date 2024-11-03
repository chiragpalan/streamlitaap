import os
import sqlite3
import pandas as pd
import streamlit as st
import plotly.express as px

# Constants
DATA_DB = 'predictions_v1.db'

# Function to load data from a specified table
def load_data_from_table(table_name):
    conn = sqlite3.connect(DATA_DB)
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df

# Function to get the table names in the database
def get_table_names():
    conn = sqlite3.connect(DATA_DB)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()
    return tables

# Streamlit App
st.title("Database Visualization App")

# Load table names
tables = get_table_names()
selected_table = st.selectbox("Select Table", tables)

if selected_table:
    df = load_data_from_table(selected_table)
    st.write(f"Displaying data for table: {selected_table}")
    st.dataframe(df)

    # Check if 'Date' column exists and convert to datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])

        # Filtered date range for the entire dataset
        if not df['Date'].empty:
            min_date, max_date = df['Date'].min(), df['Date'].max()
        else:
            st.warning("No valid dates found in 'Date' column.")
    
    # Select columns to visualize
    selected_columns = st.multiselect("Select Columns to Visualize", options=df.columns)

    # If columns are selected, show charts
    if selected_columns:
        # Interactive line chart
        if st.checkbox("Show Line Chart"):
            fig = px.line(df, x='Date', y=selected_columns, title='Line Chart')
            fig.update_layout(
                height=600,
                xaxis=dict(
                    rangeslider=dict(visible=True),  # Add range slider below the chart
                    rangeselector=dict(  # Add predefined range selectors
                        buttons=list([
                            dict(count=1, label="1m", step="month", stepmode="backward"),
                            dict(count=6, label="6m", step="month", stepmode="backward"),
                            dict(count=1, label="YTD", step="year", stepmode="todate"),
                            dict(count=1, label="1y", step="year", stepmode="backward"),
                            dict(step="all")
                        ])
                    )
                )
            )
            st.plotly_chart(fig, use_container_width=True)

        # Interactive bar chart
        if st.checkbox("Show Bar Chart"):
            fig = px.bar(df, x='Date', y=selected_columns, title='Bar Chart')
            fig.update_layout(
                height=600,
                xaxis=dict(
                    rangeslider=dict(visible=True),
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1, label="1m", step="month", stepmode="backward"),
                            dict(count=6, label="6m", step="month", stepmode="backward"),
                            dict(count=1, label="YTD", step="year", stepmode="todate"),
                            dict(count=1, label="1y", step="year", stepmode="backward"),
                            dict(step="all")
                        ])
                    )
                )
            )
            st.plotly_chart(fig, use_container_width=True)
