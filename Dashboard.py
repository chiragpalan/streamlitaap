import streamlit as st
import pandas as pd

# Load the dataset
@st.cache
def load_data():
    data = pd.read_csv("data.csv")
    return data

data = load_data()

# Title of the dashboard
st.title("Interactive Dashboard")

# Show the raw data
if st.checkbox("Show Raw Data"):
    st.write(data)

# User inputs for filtering
st.sidebar.header("Filter Conditions")

# Select column
columns = data.columns.tolist()
selected_col = st.sidebar.selectbox("Select Column to Filter", columns)

# Select condition
conditions = ["=", ">", ">=", "<", "<=", "!="]
selected_condition = st.sidebar.selectbox("Select Condition", conditions)

# Enter value
selected_value = st.sidebar.text_input("Enter Value", "")

# Filter the data
if selected_value:
    try:
        if data[selected_col].dtype in ['int64', 'float64']:
            selected_value = float(selected_value)  # Convert input to numeric if the column is numeric
        query = f"{selected_col} {selected_condition} @selected_value"
        filtered_data = data.query(query)
        st.write(f"Filtered Data ({query}):")
        st.write(filtered_data)

        # Metrics: Count of n_cust and Sum of target
        st.metric("Count of n_cust", filtered_data['n_cust'].count())
        st.metric("Sum of target", filtered_data['target'].sum())
    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.warning("Enter a value to filter.")

# Download filtered data
if st.button("Download Filtered Data"):
    filtered_data.to_csv("filtered_data.csv", index=False)
    st.success("Filtered data saved as 'filtered_data.csv'")
