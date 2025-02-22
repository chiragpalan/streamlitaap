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
