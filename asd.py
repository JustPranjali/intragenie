import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
import os
import matplotlib.pyplot as plt

# Set page configuration for a wide layout
st.set_page_config(layout="wide")

# Define available stocks
available_stocks = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]

# Fetch data from yfinance and preprocess
def fetch_data(stock_symbol, interval="5m", period="1mo"):
    data = yf.download(tickers=stock_symbol, interval=interval, period=period)
    data_30min = data['Close'].resample('30T').mean().dropna()
    if data_30min.empty:
        raise ValueError(f"Insufficient data fetched for {stock_symbol}. Try a longer period.")
    return data_30min.values

def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_normalized = scaler.fit_transform(data.reshape(-1, 1))

    X, y = [], []
    sequence_length = 13  # Past 13 intervals
    forecast_length = 13  # Predict next 13 intervals
    for i in range(len(data_normalized) - sequence_length - forecast_length):
        X.append(data_normalized[i:i + sequence_length, 0])
        y.append(data_normalized[i + sequence_length:i + sequence_length + forecast_length, 0])

    X, y = np.array(X), np.array(y)
    if X.size == 0 or y.size == 0:
        raise ValueError("Insufficient data for sequence length and forecast length.")
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y, scaler

def build_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50),
        Dense(13)  # Predicting 13 values for the 13 half-hour intervals
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def get_or_train_model(X, y, stock_symbol):
    model_path = f"{stock_symbol}_30min_predictor.h5"
    if os.path.exists(model_path):
        model = load_model(model_path, compile=False)
        model.compile(optimizer='adam', loss='mse')
    else:
        model = build_model((X.shape[1], X.shape[2]))
        model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2, verbose=2)
        model.save(model_path)
    return model

def predict_next_13_intervals(model, recent_data, scaler):
    recent_data = recent_data.reshape((1, recent_data.shape[0], 1))
    predictions = model.predict(recent_data)
    predictions = scaler.inverse_transform(predictions)
    return predictions[0]

def find_peak_and_lowest_time(predicted_prices, time_slots):
    """
    Find the best time to buy and sell to maximize profit.
    Ensure the sell time comes strictly after the buy time.
    """
    max_profit = 0  # Maximum profit seen so far
    buy_time = None  # Best time to buy
    sell_time = None  # Best time to sell

    # Iterate through each price as the potential buy price
    for buy_index in range(len(predicted_prices)):
        for sell_index in range(buy_index + 1, len(predicted_prices)):
            # Calculate profit for the current buy-sell pair
            profit = predicted_prices[sell_index] - predicted_prices[buy_index]
            if profit > max_profit:
                max_profit = profit
                buy_time = time_slots[buy_index]
                sell_time = time_slots[sell_index]

    # If no valid buy-sell pair exists, return None
    if max_profit <= 0:
        return None, None, None, None

    return buy_time, predicted_prices[time_slots.index(buy_time)], sell_time, predicted_prices[time_slots.index(sell_time)]

st.title("Stock Price Prediction App")
st.sidebar.header("Select Options")

selected_stock = st.sidebar.selectbox("Select a stock:", available_stocks)

if st.sidebar.button("Predict Stock Prices"):
    try:
        time_slots = [
            "9:15 AM", "9:45 AM", "10:15 AM", "10:45 AM", "11:15 AM", "11:45 AM", "12:15 PM", "12:45 PM",
            "1:15 PM", "1:45 PM", "2:15 PM", "2:45 PM", "3:15 PM"
        ]

        # Fetch and preprocess data
        data = fetch_data(selected_stock)
        X, y, scaler = preprocess_data(data)

        # Train or load model
        model = get_or_train_model(X, y, selected_stock)

        # Predict the next 13 intervals
        recent_data = X[-1]
        predicted_prices = predict_next_13_intervals(model, recent_data, scaler)

        # Find peak and lowest time
        result = find_peak_and_lowest_time(predicted_prices, time_slots)
        if result[0] is None:
            st.markdown("**No valid buy-sell pair found.**")
        else:
            buy_time, min_price, sell_time, max_price = result

            # Display predictions as a table
            predictions_df = pd.DataFrame({
                "Sr. No": range(1, len(time_slots) + 1),
                "Time Interval": time_slots,
                "Predicted Price": [f"${price:.2f}" for price in predicted_prices]
            })
            st.subheader(f"Predicted Prices for {selected_stock}")
            st.table(predictions_df)

            # Display best times to buy and sell
            st.markdown(f"""
                <div style="font-size:20px; margin-top:20px;">
                    <b>Best time to BUY!!!</b> at <span style="color:blue;">{buy_time}</span>: <b>${min_price:.2f}</b>
                    <br>
                    <b>Best time to SELL!!!</b> at <span style="color:green;">{sell_time}</span>: <b>${max_price:.2f}</b>
                </div>
            """, unsafe_allow_html=True)

            # Plot predictions
            st.subheader("Price Prediction Plot")
            plt.figure(figsize=(10, 6))
            plt.plot(time_slots, predicted_prices, marker='o', label=f"{selected_stock} - Predicted Prices")
            plt.xlabel("Time")
            plt.ylabel("Predicted Price")
            plt.title(f"Predicted Prices for Next 13 Half-Hour Intervals - {selected_stock}")
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.legend()
            st.pyplot(plt)

    except ValueError as e:
        st.error(f"Error: {e}")