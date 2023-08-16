import streamlit as st
import yfinance as yf
from prophet import Prophet

# Set page title and icon
st.set_page_config(page_title="Stock Market App", page_icon=":chart_with_upwards_trend:")

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
tickers = ["JBFCF", "MCD", "COKE"]

# Use a dropdown menu to select a ticker
ticker = st.sidebar.selectbox("Select a Ticker", tickers)

# Get company data
def get_ticker(name):
    company = yf.Ticker(name)
    return company

company = get_ticker(ticker)
data = company.history(period="24mo")

# Set header
st.markdown("<h1 style='text-align: center;'>Stock Market App</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# Display company information and stock data
st.markdown(f"<h2>{company.info.get('longName')}</h2>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

st.subheader("Stock Data")
st.line_chart(data.values)

# Prophet prediction
df = data.reset_index()[["Date", "Close"]].rename({"Date": "ds", "Close": "y"}, axis=1)
df["ds"] = df["ds"].dt.tz_localize(None)
model = Prophet()
model.fit(df)
future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)

st.subheader("Stock Price Prediction")
st.line_chart(forecast.set_index("ds")[["yhat", "yhat_lower", "yhat_upper"]])

# Display ticker info
st.markdown("<div class='ticker-info'>", unsafe_allow_html=True)
st.markdown(f"<h2>{company.info.get('symbol')}</h2>", unsafe_allow_html=True)
st.markdown(f"<p><strong>Name:</strong> {company.info.get('longName')}</p>", unsafe_allow_html=True)
st.markdown(f"<p><strong>Market Cap:</strong> {company.info.get('marketCap')}</p>", unsafe_allow_html=True)
st.markdown(f"<p><strong>Forward P/E:</strong> {company.info.get('forwardPE')}</p>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

