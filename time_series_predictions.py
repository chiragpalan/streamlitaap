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
        # Convert Date column to datetime, handling errors if any non-date values exist
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])  # Drop rows where Date conversion failed

        # Verify if Date column contains valid datetime values
        if not df['Date'].empty:
            min_date, max_date = df['Date'].min(), df['Date'].max()
            
            # Ensure min_date and max_date are of type datetime
            if isinstance(min_date, pd.Timestamp) and isinstance(max_date, pd.Timestamp):
                selected_date_range = st.slider(
                    "Select Date Range:",
                    min_value=min_date.to_pydatetime(),
                    max_value=max_date.to_pydatetime(),
                    value=(min_date.to_pydatetime(), max_date.to_pydatetime())
                )
                
                # Filter data based on selected date range
                df = df[(df['Date'] >= selected_date_range[0]) & (df['Date'] <= selected_date_range[1])]
            else:
                st.error("Invalid date range detected in 'Date' column.")
        else:
            st.warning("No valid dates found in 'Date' column.")
    
    # Select columns to visualize
    selected_columns = st.multiselect("Select Columns to Visualize", options=df.columns)

    # If columns are selected, show charts
    if selected_columns:
        # Interactive line chart
        if st.checkbox("Show Line Chart"):
            fig = px.line(df, x='Date', y=selected_columns, title='Line Chart')
            fig.update_layout(height=600)  # Increase chart area height
            st.plotly_chart(fig, use_container_width=True)

        # Interactive bar chart
        if st.checkbox("Show Bar Chart"):
            fig = px.bar(df, x='Date', y=selected_columns, title='Bar Chart')
            fig.update_layout(height=600)  # Increase chart area height
            st.plotly_chart(fig, use_container_width=True)
