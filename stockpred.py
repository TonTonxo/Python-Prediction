import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import ta
import altair as alt
import datetime

currentDate = datetime.datetime.now()
# currentMonthDate = currentDate.strftime("%B")
# if currentMonthDate = currentDate
nextMonthDate = currentDate.replace(day=1)+datetime.timedelta(days=32)
nextMonthDate = nextMonthDate.replace(day=1).strftime("%B")

# Set page title and icon
st.set_page_config(page_title="Stock Market App",
                   page_icon=":chart_with_upwards_trend:")

# Set page layout
st.set_option("deprecation.showPyplotGlobalUse", False)
st.markdown(
    """
    <style>
        .stApp {
            max-width: 800px;
            margin: 0 auto;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .stHeader {
            padding: 1rem;
            background-color: #f0f0f0;
            border-radius: 1rem;
            margin-bottom: 2rem;
        }
        .stButton {
            margin-top: 1rem;
            margin-bottom: 1rem;
        }
        .ticker-info {
            margin-top: 2rem;
            margin-bottom: 1rem;
            padding: 1rem;
            border: 1px solid #d0d0d0;
            border-radius: 1rem;
        }
        .ticker-info h2 {
            margin-top: 0;
            margin-bottom: 1rem;
            font-size: 2rem;
        }
        .ticker-info p {
            margin-top: 0;
            margin-bottom: 0.5rem;
        }
        .ticker-info hr {
            margin-top: 2rem;
            margin-bottom: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Define a list of tickers to choose from
tickers = ["JBFCF", "MCD", "COKE", "AAPL", "NKE"]

# Use a dropdown menu to select a ticker
ticker = st.sidebar.selectbox("Select a Ticker", tickers)

# Get company data


def get_ticker(name):
    company = yf.Ticker(name)
    return company


company = get_ticker(ticker)
data = company.history(period="24mo")

# Set header
st.markdown("<h1 style='text-align: center;'>Stock Market App</h1>",
            unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# Display company information and stock data
st.markdown("<hr>", unsafe_allow_html=True)

st.subheader("Stock Data")

# Display data in a table format
st.write(data[['Open', 'Close', 'High', 'Low']])

# Set RSI period
rsi_period = 14

# Get close prices
close_prices = data["Close"]

# Calculate RSI
rsi = ta.momentum.RSIIndicator(close=close_prices, window=rsi_period)

# Add RSI values to the DataFrame
data["RSI"] = rsi.rsi()

# Plot RSI chart
st.subheader(f"RSI ({rsi_period} period)")
st.line_chart(data["RSI"])


# Display line chart
st.line_chart(data.Close)


# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data["Close"].values.reshape(-1, 1))

# Prepare data for training
train_data = scaled_data[:int(len(scaled_data) * 0.8)]
x_train, y_train = [], []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True,
          input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))

# Compile model
model.compile(optimizer="adam", loss="mean_squared_error")

# Train model
model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

# Prepare test data
test_data = scaled_data[int(len(scaled_data) * 0.8) - 60:]
x_test, y_test = [], data[int(len(scaled_data) * 0.8):].Close.tolist()

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Predict stock prices
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Get the date of the last entry in the data
last_date = data.index[-1]

# Get the date of the next day
next_date = last_date + pd.DateOffset(months=1)

# Prepare test data for the preceding month
test_data = scaled_data[-60:]
x_test = np.array([test_data])
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Predict stock prices for the preceding month
prediction = model.predict(x_test)
prediction = scaler.inverse_transform(prediction)

# Display predicted stock price for the preceding month

st.subheader(f"Predicted stock price for the month of {nextMonthDate} 2023")
st.write(prediction[0][0])

# Predict stock prices for current month
curr_pred = model.predict(x_test)
curr_pred = scaler.inverse_transform(curr_pred)

# Get last known price
last_price = data.iloc[-1]["Close"]

# Estimate preceding month's price based on current prediction and last known price
prev_price = (curr_pred[0][0] / last_price) * data.iloc[-2]["Close"]

# Display estimated price for preceding month
st.write(
    f"Estimated stock price for the month of {nextMonthDate} 2023: {prev_price:.2f}")


# Plot actual vs predicted stock prices
st.subheader("Predictions vs Actual")
df = pd.DataFrame(data=y_test, columns=["Actual"])
df["Predicted"] = predictions

# Reset the index of the DataFrame
df.reset_index(inplace=True)

# Clear previous chart
st.empty()

# Create a line chart using vega-lite
chart = alt.Chart(df).mark_line().encode(
    x=alt.X('index', title='Index'),
    y=alt.Y('Actual', title='Actual'),
    color=alt.value("blue"),
) + alt.Chart(df).mark_line().encode(
    x=alt.X('index', title='Index'),
    y=alt.Y('Predicted', title='Predicted'),
    color=alt.value("orange"),
)

# Display the line chart using Streamlit
st.altair_chart(chart, use_container_width=True)


# Compare stock prices
st.subheader("Compare Stocks")
compare_tickers = st.multiselect("Select tickers to compare", tickers)

if len(compare_tickers) > 0:
    compare_data = pd.DataFrame()
    for t in compare_tickers:
        compare_company = get_ticker(t)
        compare_data[t] = compare_company.history(period="24mo").Close

    compare_data[ticker] = data.Close
    st.line_chart(compare_data)

# # Display model summary
# st.subheader("Model Summary")
# summary_table = tf.keras.utils.model_to_dot(
#     model, show_shapes=True, dpi=70).create(prog='dot', format='svg')
# st.components.v1.html(summary_table, width=800, height=600, scrolling=True)


# Display RSI summary statistics
rsi_summary = data.RSI.describe()
rsi_summary_table = pd.DataFrame(
    data={"Summary Statistics": rsi_summary.index, "Value": rsi_summary.values})
st.subheader("RSI Summary Statistics")
st.write(rsi_summary_table)

# Display model performance metrics
st.subheader("Model Performance Metrics")
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

metrics_table = pd.DataFrame(data={"Metric": ["Mean Squared Error", "Mean Absolute Error", "Root Mean Squared Error", "R-Squared"],
                                   "Value": [mse, mae, rmse, r2]})
st.write(metrics_table)
