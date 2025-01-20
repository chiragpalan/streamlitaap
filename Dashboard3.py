

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

st.title("Interactive Dashboard with Waterfall Aggregation")

# User Inputs for Multiple Filters
st.sidebar.title("Filters")

# List of columns for filtering
columns = df.columns.tolist()

# Initialize filters
filters = []
intermediate_results = []

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

# Initial state (before filters)
current_df = df.copy()
initial_count = current_df['n_cust'].sum()
initial_sum = current_df['target'].sum()
intermediate_results.append({
    'filter': 'Initial',
    'count': initial_count,
    'sum': initial_sum
})

# Apply Filters
for col_name, condition, value in filters:
    if col_name in df.columns:
        # Convert value to appropriate type
        if df[col_name].dtype in ['int64', 'float64']:
            value = float(value)
        
        # Store previous values before applying the filter
        previous_count = current_df['n_cust'].sum()
        previous_sum = current_df['target'].sum()
        
        # Apply the condition
        if condition == '=':
            current_df = current_df[current_df[col_name] == value]
        elif condition == '!=':
            current_df = current_df[current_df[col_name] != value]
        elif condition == '>':
            current_df = current_df[current_df[col_name] > value]
        elif condition == '<':
            current_df = current_df[current_df[col_name] < value]
        elif condition == '>=':
            current_df = current_df[current_df[col_name] >= value]
        elif condition == '<=':
            current_df = current_df[current_df[col_name] <= value]
        
        # Track the change in values
        current_count = current_df['n_cust'].sum()
        current_sum = current_df['target'].sum()
        
        intermediate_results.append({
            'filter': f"{col_name} {condition} {value}",
            'count': current_count,
            'sum': current_sum,
            'count_diff': current_count - previous_count,
            'sum_diff': current_sum - previous_sum
        })

# Display Filtered Data
st.subheader("Filtered Data")
st.write(current_df)

# Display Waterfall Aggregation (showing the changes after each filter)
st.subheader("Waterfall Aggregation")

# Create a DataFrame for waterfall visualization
waterfall_df = pd.DataFrame(intermediate_results)

# Display each step of the aggregation
st.write(waterfall_df)

# Optional: Visualizing Waterfall (Bar Chart)
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(waterfall_df['filter'], waterfall_df['count_diff'], label="Count Change", color='blue', alpha=0.6, width=0.5)
ax.bar(waterfall_df['filter'], waterfall_df['sum_diff'], label="Sum Change", color='orange', alpha=0.6, width=0.5)

plt.xticks(rotation=45, ha='right')
plt.xlabel("Filters")
plt.ylabel("Change in Values")
plt.title("Waterfall Chart for Count and Sum Changes")
plt.legend()
st.pyplot(fig)

# Final Aggregated Results
st.subheader("Final Aggregated Results")
st.write(f"Final count of n_cust: {current_df['n_cust'].sum()}")
st.write(f"Final sum of target: {current_df['target'].sum()}")

# Add Calculated Columns
st.sidebar.title("Calculated Columns")

# User inputs for calculated columns
calc_col_expr = st.sidebar.text_input("Enter calculation (e.g., col1 * col3)", key="calc_col_expr")
calc_col_name = st.sidebar.text_input("Enter new column name", key="calc_col_name")

if calc_col_expr and calc_col_name:
    try:
        current_df[calc_col_name] = current_df.eval(calc_col_expr)
        st.sidebar.success(f"Added column '{calc_col_name}'")
    except Exception as e:
        st.sidebar.error(f"Error adding column: {e}")

# Display Data with Calculated Columns
st.subheader("Data with Calculated Columns")
st.write(current_df)

# Create Pivot Table
st.sidebar.title("Pivot Table")

# User inputs for pivot table
pivot_index = st.sidebar.selectbox("Select index column", columns, key="pivot_index")
pivot_columns = st.sidebar.multiselect("Select columns", columns, key="pivot_columns")
pivot_values = st.sidebar.selectbox("Select values column", columns, key="pivot_values")
pivot_aggfunc = st.sidebar.selectbox("Select aggregation function", ['sum', 'mean', 'count'], key="pivot_aggfunc")

if pivot_index and pivot_columns and pivot_values and pivot_aggfunc:
    try:
        pivot_table = pd.pivot_table(
            current_df,
            index=pivot_index,
            columns=pivot_columns,
            values=pivot_values,
            aggfunc=pivot_aggfunc
        )
        st.subheader("Pivot Table")
        st.write(pivot_table)
    except Exception as e:
        st.sidebar.error(f"Error creating pivot table: {e}")
