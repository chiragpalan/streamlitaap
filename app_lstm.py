import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode, DataReturnMode
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Clear cache
st.cache_resource.clear()

# Load Data for Top 100 Nifty Stocks
@st.cache_resource
def load_data():
    tickers = [
        "RELIANCE.NS", "TCS.NS"
        # ... add remaining tickers
    ]
    stock_data = {ticker: yf.download(ticker, start="2020-01-01", end=datetime.today().strftime('%Y-%m-%d')) for ticker in tickers}
    return stock_data

data = load_data()

# Function to create and train LSTM model
def train_lstm_model(df):
    # Preprocess data
    close_prices = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    # Prepare training data
    def create_sequences(data, time_steps=60):
        x, y = [], []
        for i in range(time_steps, len(data)):
            x.append(data[i - time_steps:i, 0])
            y.append(data[i, 0])
        return np.array(x), np.array(y)

    x_train, y_train = create_sequences(scaled_data)

    # Reshape input data for LSTM [samples, time steps, features]
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    # Compile and fit the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=0)

    return model, scaler

# Corrected function to predict the next 5 days using the trained LSTM model
def predict_next_5_days(df, model, scaler):
    # Extract the last 60 days of the closing prices
    last_60_days = df['Close'].values[-60:].reshape(-1, 1)
    scaled_data = scaler.transform(last_60_days)
    
    # Reshape for LSTM input format [samples, time steps, features]
    x_input = scaled_data.reshape((1, scaled_data.shape[0], 1))
    
    predictions = []
    for _ in range(5):
        # Predict the next value
        predicted_price = model.predict(x_input)[0][0]
        predictions.append(predicted_price)
        
        # Update x_input with the predicted price for the next iteration
        # Remove the first time step and add the new prediction
        x_input = np.append(x_input[:, 1:, :], [[[predicted_price]]], axis=1)
    
    # Transform predictions back to original scale
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    return predictions


# Fit LSTM models and predict for each stock
predictions = {}
for ticker, df in data.items():
    if len(df) >= 60:  # Ensure there is enough data
        model, scaler = train_lstm_model(df)
        predictions[ticker] = predict_next_5_days(df, model, scaler)

#part1
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

#part2

def calculate_pivot_points(df):
    df['Pivot'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['R1'] = 2 * df['Pivot'] - df['Low']
    df['S1'] = 2 * df['Pivot'] - df['High']
    df['R2'] = df['Pivot'] + (df['High'] - df['Low'])
    df['S2'] = df['Pivot'] - (df['High'] - df['Low'])
    df['R3'] = df['High'] + 2 * (df['Pivot'] - df['Low'])
    df['S3'] = df['Low'] - 2 * (df['High'] - df['Pivot'])
    return df

#part3
# Display and plot the data
filtered_df = filter_and_add_indicators(data)
filtered_df['Market Cap'] = filtered_df['Ticker'].map(lambda ticker: yf.Ticker(ticker).info['marketCap'])
filtered_df.reset_index(drop=True, inplace=True)

# AgGrid options with filters enabled and sorting
gb = GridOptionsBuilder.from_dataframe(filtered_df)
gb.configure_pagination(enabled=False)
gb.configure_default_column(filter=True, sortable=True)
gb.configure_column("Ticker", pinned='left')
grid_options = gb.build()

# Display using AgGrid
response = AgGrid(
    filtered_df,
    gridOptions=grid_options,
    update_mode=GridUpdateMode.SELECTION_CHANGED,
    data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
    enable_enterprise_modules=True
)

filtered_df_aggrid = pd.DataFrame(response['data'])

# Visualization for Selected Stock
st.write("Select a stock to visualize")
filtered_tickers = filtered_df_aggrid['Ticker'].unique().tolist()
selected_stock = st.selectbox("Select Stock", options=filtered_tickers)

if selected_stock:
    df = data.get(selected_stock)
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

    # Plot LSTM predictions for the selected stock
    if selected_stock in predictions:
        future_dates = pd.date_range(df.index[-1] + timedelta(days=1), periods=5)
        fig.add_trace(go.Scatter(x=future_dates, y=predictions[selected_stock], name="Predicted Close", line=dict(color='purple', dash='dash')))

    fig.update_layout(
        yaxis=dict(title="CMP"),
        yaxis2=dict(title="Volume", overlaying="y", side="right"),
        xaxis=dict(title="Date"),
        title=f"{selected_stock} - CMP & Volume with Moving Averages and LSTM Predictions"
    )

    st.plotly_chart(fig, use_container_width=True)
