import pandas as pd
import streamlit as st

# Example DataFrame
data = {
    'col1': [10, 20, 30, 40, 50],
    'col2': ['A', 'B', 'A', 'C', 'B'],
    'col3': [100, 200, 300, 400, 500],
    'n_cust': [1, 2, 3, 4, 5],
    'target': [10, 20, 30, 40, 50],
}
df = pd.DataFrame(data)

st.title("Interactive Dashboard with Multiple Filters")

# User Inputs for Multiple Filters
st.sidebar.title("Filters")

# List of columns for filtering
columns = df.columns.tolist()

# Initialize filters
filters = []

# Dynamic filter creation
num_filters = st.sidebar.number_input("Number of Filters", min_value=1, max_value=10, value=1)

for i in range(num_filters):
    st.sidebar.subheader(f"Filter {i+1}")
    col_name = st.sidebar.selectbox(f"Column {i+1}", columns, key=f"col_{i}")
    condition = st.sidebar.selectbox(
        f"Condition {i+1}",
        ['=', '!=', '>', '<', '>=', '<='],
        key=f"cond_{i}",
    )
    value = st.sidebar.text_input(f"Value {i+1}", key=f"value_{i}")
    if col_name and condition and value:
        filters.append((col_name, condition, value))

# Apply Filters
filtered_df = df.copy()

for col_name, condition, value in filters:
    if col_name in df.columns:
        # Convert value to appropriate type
        if df[col_name].dtype in ['int64', 'float64']:
            value = float(value)
        # Apply the condition
        if condition == '=':
            filtered_df = filtered_df[filtered_df[col_name] == value]
        elif condition == '!=':
            filtered_df = filtered_df[filtered_df[col_name] != value]
        elif condition == '>':
            filtered_df = filtered_df[filtered_df[col_name] > value]
        elif condition == '<':
            filtered_df = filtered_df[filtered_df[col_name] < value]
        elif condition == '>=':
            filtered_df = filtered_df[filtered_df[col_name] >= value]
        elif condition == '<=':
            filtered_df = filtered_df[filtered_df[col_name] <= value]

# Display Filtered Data
st.subheader("Filtered Data")
st.write(filtered_df)

# Count and Sum of Columns
st.subheader("Aggregated Results")
st.write(f"Count of n_cust: {filtered_df['n_cust'].sum()}")
st.write(f"Sum of target: {filtered_df['target'].sum()}")
