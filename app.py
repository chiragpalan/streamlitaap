import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode, DataReturnMode
import plotly.graph_objs as go

st.cache_resource.clear()

# Load Data for Top 100 Nifty Stocks
@st.cache_resource
def load_data():
    tickers = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS",
        "ICICIBANK.NS", "KOTAKBANK.NS", "SBIN.NS", "BAJFINANCE.NS", "BHARTIARTL.NS",
        "ITC.NS", "ASIANPAINT.NS", "HCLTECH.NS", "MARUTI.NS", "AXISBANK.NS",
        "LT.NS", "HDFCLIFE.NS", "SUNPHARMA.NS", "ULTRACEMCO.NS", "NESTLEIND.NS",
        "TITAN.NS", "WIPRO.NS", "BAJAJFINSV.NS", "ADANIGREEN.NS", "ADANIPORTS.NS",
        "DIVISLAB.NS", "JSWSTEEL.NS", "POWERGRID.NS", "NTPC.NS", "TATAMOTORS.NS",
        "GRASIM.NS", "TECHM.NS", "HINDALCO.NS", "BPCL.NS", "SHREECEM.NS",
        "ONGC.NS", "COALINDIA.NS", "BRITANNIA.NS", "HEROMOTOCO.NS", "DRREDDY.NS",
        "CIPLA.NS", "BAJAJ-AUTO.NS", "M&M.NS", "SBILIFE.NS", "EICHERMOT.NS",
        "UPL.NS", "TATASTEEL.NS", "IOC.NS", "INDUSINDBK.NS", "VEDL.NS"  # Add all 100 tickers
    ]
    stock_data = {ticker: yf.download(ticker, start="2020-01-01", end=datetime.today().strftime('%Y-%m-%d')) for ticker in tickers}
    return stock_data

# Refresh button to clear cache and rerun
if st.button("Refresh Data"):
    st.cache_data.clear()  # Clear the cached data
    st.rerun()  # Rerun the app to fetch new data

data = load_data()

# Calculate Pivot Points
def calculate_pivot_points(df):
    df['Pivot'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['R1'] = 2 * df['Pivot'] - df['Low']
    df['S1'] = 2 * df['Pivot'] - df['High']
    df['R2'] = df['Pivot'] + (df['High'] - df['Low'])
    df['S2'] = df['Pivot'] - (df['High'] - df['Low'])
    df['R3'] = df['High'] + 2 * (df['Pivot'] - df['Low'])
    df['S3'] = df['Low'] - 2 * (df['High'] - df['Pivot'])
    return df

# Filter and Add Indicators
def filter_and_add_indicators(data, cmp_column='Close'):
    filtered_data = []
    for ticker, df in data.items():
        df = calculate_pivot_points(df)
        df['50d_MA'] = df[cmp_column].rolling(window=50).mean()
        df['200d_MA'] = df[cmp_column].rolling(window=200).mean()
        df['50d_EMA'] = df[cmp_column].ewm(span=50, adjust=False).mean()
        df['200d_EMA'] = df[cmp_column].ewm(span=200, adjust=False).mean()

        latest_data = df.iloc[-1].copy()

        latest_data['CMP > 50d MA'] = 1 if latest_data[cmp_column] > latest_data['50d_MA'] else 0
        latest_data['CMP > 200d MA'] = 1 if latest_data[cmp_column] > latest_data['200d_MA'] else 0
        latest_data['CMP > 50d EMA'] = 1 if latest_data[cmp_column] > latest_data['50d_EMA'] else 0
        latest_data['CMP > 200d EMA'] = 1 if latest_data[cmp_column] > latest_data['200d_EMA'] else 0
        #adding support and resistance comparsion
        latest_data['CMP > S1'] = 1 if latest_data[cmp_column] > latest_data['S1'] else 0
        latest_data['CMP > S2'] = 1 if latest_data[cmp_column] > latest_data['S2'] else 0
        latest_data['CMP > S3'] = 1 if latest_data[cmp_column] > latest_data['S3'] else 0
        latest_data['CMP > R1'] = 1 if latest_data[cmp_column] > latest_data['R1'] else 0
        latest_data['CMP > R2'] = 1 if latest_data[cmp_column] > latest_data['R2'] else 0
        latest_data['CMP > R3'] = 1 if latest_data[cmp_column] > latest_data['R3'] else 0

        latest_data['CMP < S1'] = 1 if latest_data[cmp_column] < latest_data['S1'] else 0
        latest_data['CMP < S2'] = 1 if latest_data[cmp_column] < latest_data['S2'] else 0
        latest_data['CMP < S3'] = 1 if latest_data[cmp_column] < latest_data['S3'] else 0
        latest_data['CMP < R1'] = 1 if latest_data[cmp_column] < latest_data['R1'] else 0
        latest_data['CMP < R2'] = 1 if latest_data[cmp_column] < latest_data['R2'] else 0
        latest_data['CMP < R3'] = 1 if latest_data[cmp_column] < latest_data['R3'] else 0
        

        
        latest_data['Ticker'] = ticker  # Add ticker as a column
        filtered_data.append(latest_data)

    return pd.DataFrame(filtered_data)

# Combine Data
filtered_df = filter_and_add_indicators(data)
filtered_df['Market Cap'] = filtered_df['Ticker'].map(lambda ticker: yf.Ticker(ticker).info['marketCap'])
filtered_df.reset_index(drop=True, inplace=True)  # Reset index so that Ticker becomes a column

# AgGrid options with filters enabled and sorting
gb = GridOptionsBuilder.from_dataframe(filtered_df)
gb.configure_pagination(enabled=False)  # Disable pagination to show all rows
gb.configure_default_column(filter=True, sortable=True)  # Enable filtering and sorting on all columns
gb.configure_column("Ticker", pinned='left')  # Pin Ticker column to the left
grid_options = gb.build()

# Display using AgGrid with filters, sorting enabled
response = AgGrid(
    filtered_df,
    gridOptions=grid_options,
    update_mode=GridUpdateMode.SELECTION_CHANGED,
    data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
    enable_enterprise_modules=True
)

# Get the filtered data from AgGrid
filtered_df_aggrid = pd.DataFrame(response['data'])  # Filtered data

# Visualization for Selected Stock from Filtered Data
st.write("Select a stock to visualize")
filtered_tickers = filtered_df_aggrid['Ticker'].unique().tolist()  # Update dropdown with filtered tickers
selected_stock = st.selectbox("Select Stock", options=filtered_tickers)

if selected_stock:
    df = data[selected_stock]
    st.write(f"Showing data for {selected_stock}")

    date_buttons = {
        "1 Week": datetime.today() - timedelta(weeks=1),
        "1 Month": datetime.today() - timedelta(days=30),
        "3 Months": datetime.today() - timedelta(days=90),
        "6 Months": datetime.today() - timedelta(days=180),
        "1 Year": datetime.today() - timedelta(days=365),
        "3 Years": datetime.today() - timedelta(days=1095)
    }

    selected_range = st.selectbox("Select Time Range", list(date_buttons.keys()))

    df_filtered = df[df.index >= date_buttons[selected_range]]

    # Moving Averages Checkboxes
    show_50d_ma = st.checkbox('Show 50-Day MA', value=True)
    show_200d_ma = st.checkbox('Show 200-Day MA', value=True)
    show_50d_ema = st.checkbox('Show 50-Day EMA', value=False)
    show_200d_ema = st.checkbox('Show 200-Day EMA', value=False)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['Close'], name="CMP", yaxis="y1"))
    fig.add_trace(go.Bar(x=df_filtered.index, y=df_filtered['Volume'], name="Volume", yaxis="y2"))

    # Add selected moving averages to the chart
    if show_50d_ma:
        fig.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['50d_MA'], name="50-Day MA", line=dict(dash='dash')))
    if show_200d_ma:
        fig.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['200d_MA'], name="200-Day MA", line=dict(dash='dash')))
    if show_50d_ema:
        fig.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['50d_EMA'], name="50-Day EMA", line=dict(dash='dot')))
    if show_200d_ema:
        fig.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['200d_EMA'], name="200-Day EMA", line=dict(dash='dot')))

    # Add Pivot Points to the chart
    fig.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['Pivot'], name="Pivot", line=dict(color='blue', dash='dash')))
    fig.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['R1'], name="R1", line=dict(color='green', dash='dash')))
    fig.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['S1'], name="S1", line=dict(color='red', dash='dash')))
    fig.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['R2'], name="R2", line=dict(color='green', dash='dot')))
    fig.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['S2'], name="S2", line=dict(color='red', dash='dot')))
    fig.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['R3'], name="R3", line=dict(color='green', dash='dashdot')))
    fig.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['S3'], name="S3", line=dict(color='red', dash='dashdot')))

    fig.update_layout(
        yaxis=dict(title="CMP"),
        yaxis2=dict(title="Volume", overlaying="y", side="right"),
        xaxis=dict(title="Date"),
        title=f"{selected_stock} - CMP & Volume with Moving Averages"
    )

    st.plotly_chart(fig)
