import os
import sqlite3
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Constants
DATA_DB = 'predictions_v1.db'

# Function to download the database from GitHub
def download_database():
    url = 'https://raw.githubusercontent.com/chiragpalan/train_and_predict/main/database/predictions_v1.db'
    response = requests.get(url)
    if response.status_code == 200:
        with open(DATA_DB, 'wb') as f:
            f.write(response.content)
        st.success("Database downloaded successfully.")
    else:
        st.error("Failed to download the database.")

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

# Button to refresh and download the latest database
if st.button("Refresh Database"):
    download_database()

# Load table names
tables = get_table_names()
selected_table = st.selectbox("Select Table", tables)

if selected_table:
    df = load_data_from_table(selected_table)
    st.write(f"Displaying data for table: {selected_table}")
    st.dataframe(df)

    # Ensure the Date column is in datetime format and drop rows with null dates
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])  # Drop rows where Date is NaT (null)

    # Select columns to visualize
    selected_columns = st.multiselect("Select Columns to Visualize", options=df.columns)

    if selected_columns and 'Date' in df.columns:
        # Interactive line chart with range slider
        if st.checkbox("Show Line Chart"):
            fig = px.line(df, x='Date', y=selected_columns, title='Line Chart')
            fig.update_layout(
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1, label="1m", step="month", stepmode="backward"),
                            dict(count=6, label="6m", step="month", stepmode="backward"),
                            dict(count=1, label="YTD", step="year", stepmode="todate"),
                            dict(count=1, label="1y", step="year", stepmode="backward"),
                            dict(step="all")
                        ])
                    ),
                    rangeslider=dict(visible=True),
                    type="date"
                ),
                width=1000,
                height=600
            )
            st.plotly_chart(fig)

        # Interactive bar chart with range slider
        if st.checkbox("Show Bar Chart"):
            fig = px.bar(df, x='Date', y=selected_columns, title='Bar Chart')
            fig.update_layout(
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1, label="1m", step="month", stepmode="backward"),
                            dict(count=6, label="6m", step="month", stepmode="backward"),
                            dict(count=1, label="YTD", step="year", stepmode="todate"),
                            dict(count=1, label="1y", step="year", stepmode="backward"),
                            dict(step="all")
                        ])
                    ),
                    rangeslider=dict(visible=True),
                    type="date"
                ),
                width=1000,
                height=600
            )
            st.plotly_chart(fig)
    else:
        st.warning("Please select at least one column and ensure 'Date' column is available for visualization.")
