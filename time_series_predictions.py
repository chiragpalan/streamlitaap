import os
import sqlite3
import pandas as pd
import streamlit as st
import plotly.express as px

# Constants
DATA_DB = 'predictions_v1.db'  # Adjust the path if necessary

# Verify if database file exists
if os.path.exists(DATA_DB):
    st.success(f"Database '{DATA_DB}' found.")
else:
    st.error(f"Database '{DATA_DB}' not found. Please check the file path.")

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
st.write("Available Tables:", tables)  # Debugging: Display table names
selected_table = st.selectbox("Select Table", tables)

if selected_table:
    df = load_data_from_table(selected_table)
    st.write(f"Displaying data for table: {selected_table}")
    st.dataframe(df)

    # Select columns to visualize
    selected_columns = st.multiselect("Select Columns to Visualize", options=df.columns)

    # Date slider functionality (assuming 'Date' column exists)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])  # Ensure 'Date' is in datetime format
        min_date, max_date = df['Date'].min(), df['Date'].max()
        selected_date_range = st.slider(
            "Select Date Range:",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date)
        )
        
        # Filter data based on selected date range
        df = df[(df['Date'] >= selected_date_range[0]) & (df['Date'] <= selected_date_range[1])]
    
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
