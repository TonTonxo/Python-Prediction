import streamlit as st
import yfinance as yf
from prophet import Prophet

st.write("Stock Price Prediction")
st.title("Stock Market App")

# Define a list of tickers to choose from
tickers = ["JBFCF", "MCD", "COKE"]

# Use a dropdown menu to select a ticker
ticker = st.selectbox("Select a Ticker", tickers)

def get_ticker(name):
    company = yf.Ticker(name)
    return company

company = get_ticker(ticker)

data = company.history(period="24mo")

# Display company information and stock data
st.write(f"## {company.info.get('longName')} ##")
st.write(company.info.get('longBusinessSummary'))
st.write("### Stock Data")
st.write(data)
st.line_chart(data.values)

# Prophet prediction
df = data.reset_index()[["Date", "Close"]].rename({"Date": "ds", "Close": "y"}, axis=1)
df["ds"] = df["ds"].dt.tz_localize(None)
model = Prophet()
model.fit(df)
future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)
st.write("### Stock Price Prediction")
st.line_chart(forecast.set_index("ds")[["yhat", "yhat_lower", "yhat_upper"]])
