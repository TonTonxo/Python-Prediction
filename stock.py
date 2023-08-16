import streamlit as st
import yfinance as yf
from prophet import Prophet

st.write("Stock Price Prediction")
st.title("Stock Market App")


def get_ticker(name):
    company = yf.Ticker(name)
    return company


c1 = get_ticker("JBFCF")
c2 = get_ticker("MCD")
c3 = get_ticker("COKE")

jollibee = yf.download("JBFCF", start="2015-11-11", end="2022-11-11")
mcdo = yf.download("MCD", start="2015-11-11", end="2022-11-11")
coke = yf.download("COKE", start="2015-11-11", end="2022-11-11")

data1 = c1.history(period="24mo")
data2 = c2.history(period="24mo")
data3 = c3.history(period="24mo")


# Jollibee
st.write("## Jollibee Food Corporation ##")
st.write("### Stock Data")
st.write(data1)
st.line_chart(data1.values)

# Prophet prediction
df = data1.reset_index()[["Date", "Close"]].rename(
    {"Date": "ds", "Close": "y"}, axis=1)
df["ds"] = df["ds"].dt.tz_localize(None)
model = Prophet()
model.fit(df)
future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)
st.write("### Stock Price Prediction")
st.line_chart(forecast.set_index("ds")[["yhat", "yhat_lower", "yhat_upper"]])

# McDonald's
st.write("## McDonald's Corporation ##")
st.write(c2.info.get('longBusinessSummary'))
st.write("### Stock Data")
st.write(data2)
st.line_chart(data2.values)

# Prophet prediction
df = data2.reset_index()[["Date", "Close"]].rename(
    {"Date": "ds", "Close": "y"}, axis=1)
df["ds"] = df["ds"].dt.tz_localize(None)
model = Prophet()
model.fit(df)
future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)
st.write("### Stock Price Prediction")
st.line_chart(forecast.set_index("ds")[["yhat", "yhat_lower", "yhat_upper"]])

# Coca-Cola
st.write("## COCA-COLA CONSOLIDATED, INC ##")
st.write(c3.info.get('longBusinessSummary'))
st.write("### Stock Data")
st.write(data3)
st.line_chart(data3.values)

# Prophet prediction
df = data3.reset_index()[["Date", "Close"]].rename(
    {"Date": "ds", "Close": "y"}, axis=1)
df["ds"] = df["ds"].dt.tz_localize(None)
model = Prophet()
model.fit(df)
future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)
st.write("### Stock Price Prediction")
st.line_chart(forecast.set_index("ds")[["yhat", "yhat_lower", "yhat_upper"]])
