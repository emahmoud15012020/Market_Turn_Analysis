import os
import streamlit as st
import pandas as pd
from polygon import RESTClient
import yfinance as yf
from datetime import date, timedelta
import numpy as np
import plotly.graph_objects as go
from selenium import webdriver
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from webdriver_manager.firefox import GeckoDriverManager
import time

st.set_page_config(layout="wide")
st.title("📊 Market Turn Analysis Dashboard")

# --- Default inputs ---
symbol = st.text_input("Enter a stock symbol", "TQQQ")

with st.sidebar:
    polygon_api_key = st.text_input("Polygon API Key", type="password")
    polygon_api_key = polygon_api_key.strip() or "YOUR_DEFAULT_POLYGON_KEY"

    start_date = st.date_input("Start Date", pd.to_datetime("2025-01-01"))
    end_date = st.date_input("End Date", date.today())

# Authenticate Polygon
client = RESTClient(polygon_api_key)

# --- Stock Quote ---
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("### Stock Quote")
    try:
        aggs = client.get_previous_close_agg(symbol)
        for agg in aggs:
            st.success(f"Ticker: {agg.ticker}\n\n"
                       f"Close: {agg.close}\n\n"
                       f"High: {agg.high}\n\n"
                       f"Low: {agg.low}\n\n"
                       f"Open: {agg.open}\n\n"
                       f"Volume: {agg.volume}")
    except Exception as e:
        st.error(f"Error fetching quote: {e}")

# --- VIX Analysis ---
st.markdown("---")
st.subheader("📈 VIX Analysis")
try:
    vix_df = yf.download("^VIX", start=str(start_date), end=str(end_date))

    if not vix_df.empty:
        if isinstance(vix_df.columns, pd.MultiIndex):
            vix_df.columns = vix_df.columns.get_level_values(0)

        vix_df = vix_df.reset_index()
        col_ma1, col_ma2 = st.columns(2)
        with col_ma1:
            ma_short = st.slider("Short-term MA (days)", min_value=1, max_value=50, value=3)
        with col_ma2:
            ma_long = st.slider("Long-term MA (days)", min_value=5, max_value=200, value=9)

        vix_df["MA_short"] = vix_df["Close"].rolling(ma_short).mean()
        vix_df["MA_long"] = vix_df["Close"].rolling(ma_long).mean()

        latest_vix = vix_df.iloc[-1]
        ma_short_val = float(latest_vix["MA_short"])
        ma_long_val = float(latest_vix["MA_long"])

        if ma_short_val > ma_long_val:
            vix_signal = f"Bearish Signal (Volatility Rising: {ma_short}-day > {ma_long}-day)"
        else:
            vix_signal = f"Bullish Signal (Volatility Falling: {ma_short}-day < {ma_long}-day)"

        st.line_chart(vix_df.set_index("Date")["Close"].to_frame().assign(
            MA_short=vix_df.set_index("Date")["MA_short"],
            MA_long=vix_df.set_index("Date")["MA_long"]
        ))
        st.info(f"Latest VIX: {latest_vix['Close']:.2f} → {vix_signal}")

        with st.expander("ℹ️ What is VIX Analysis?"):
            st.markdown("""
            - The **VIX (Volatility Index)** measures expected volatility in the S&P500.
            - Derived from S&P500 options prices (30-day implied volatility).
            - **Short-term MA > Long-term MA** → rising volatility → bearish.
            - **Short-term MA < Long-term MA** → falling volatility → bullish.
            """)
except Exception as e:
    st.error(f"Error fetching VIX: {e}")

# --- PCR Scraper ---
@st.cache_data(ttl=3600)
def scrape_cboe_daily_pcr_batch(n=10):
    """Scrape last n days of TOTAL PUT/CALL RATIO using headless Firefox"""
    options = FirefoxOptions()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    service = Service(GeckoDriverManager().install())
    driver = webdriver.Firefox(options=options, service=service)

    results = []
    try:
        for i in range(n):
            dt = date.today() - timedelta(days=i)
            dt_str = dt.strftime("%Y-%m-%d")
            url = f"https://www.cboe.com/us/options/market_statistics/daily/?dt={dt_str}"
            driver.get(url)

            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located(
                        (By.XPATH, "//table[contains(@class,'TableComponents__StyledTable')]"))
                )
            except:
                continue

            soup = BeautifulSoup(driver.page_source, "html.parser")
            rows = soup.find_all("tr")
            for row in rows:
                cells = row.find_all("td")
                if len(cells) >= 2:
                    label = cells[0].get_text(strip=True).upper()
                    if "TOTAL PUT/CALL RATIO" in label:
                        try:
                            value = float(cells[1].get_text(strip=True))
                            results.append({"Date": dt, "Total_PCR": value})
                        except:
                            pass
    finally:
        driver.quit()

    df = pd.DataFrame(results).sort_values("Date", ascending=False).reset_index(drop=True)
    return df

# --- Display PCR Table ---
st.markdown("---")
st.subheader("⚖️ CBOE Total Put/Call Ratio")
num_days = 15
df_pcr = scrape_cboe_daily_pcr_batch(num_days)

if df_pcr.empty:
    st.error("No PCR data available. Try refreshing.")
else:
    st.table(df_pcr.style.hide(axis="index"))

# --- Technical Analysis Section ---
st.markdown("---")
st.subheader("📊 Technical Analysis")

@st.cache_data(ttl=300)
def fetch_polygon_aggs(symbol, start_date, end_date, _client):
    aggs = []
    try:
        for a in _client.list_aggs(symbol, 1, "day", str(start_date), str(end_date),
                                   adjusted="true", sort="asc", limit=500):
            aggs.append(a)
        return aggs, None
    except Exception as e:
        return None, str(e)

# --- Fetch stock data ---
stock_df = None
aggs, fetch_err = fetch_polygon_aggs(symbol, start_date, end_date, client)
if aggs:
    data = [{"timestamp": a.timestamp, "open": a.open, "high": a.high,
             "low": a.low, "close": a.close, "volume": a.volume} for a in aggs]
    stock_df = pd.DataFrame(data)
    stock_df['timestamp'] = pd.to_datetime(stock_df['timestamp'], unit='ms')
    stock_df.set_index('timestamp', inplace=True)
    stock_df = stock_df.sort_index()
elif fetch_err:
    st.info(f"Polygon fetch failed: {fetch_err}")

if stock_df is None or stock_df.empty:
    try:
        yf_df = yf.download(symbol, start=str(start_date), end=str(end_date))
        if not yf_df.empty and "Volume" in yf_df.columns:
            yf_df = yf_df.rename(columns={
                "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"
            })
            stock_df = yf_df[["open", "high", "low", "close", "volume"]].copy()
            stock_df.index = pd.to_datetime(stock_df.index)
            stock_df = stock_df.sort_index()
            st.info("Using Yahoo Finance data.")
    except Exception as e:
        st.error(f"Failed to fetch Yahoo Finance data: {e}")

# --- Indicators ---
if stock_df is not None and not stock_df.empty:
    delta = stock_df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(9).mean() / loss.rolling(9).mean()
    stock_df["RSI_9"] = 100 - (100 / (1 + rs))

    # OBV
    direction = np.sign(stock_df["close"].diff().fillna(0))
    stock_df["OBV"] = (direction * stock_df["volume"]).cumsum()
    obv_short_ma = st.slider("OBV Short MA (days)", 2, 50, 10)
    obv_long_ma = st.slider("OBV Long MA (days)", obv_short_ma+1, 200, 30)
    stock_df["OBV_MA_Short"] = stock_df["OBV"].rolling(obv_short_ma).mean()
    stock_df["OBV_MA_Long"] = stock_df["OBV"].rolling(obv_long_ma).mean()

    # SMA/EMA
    stock_df["SMA_20"] = stock_df["close"].rolling(20).mean()
    stock_df["EMA_20"] = stock_df["close"].ewm(span=20, adjust=False).mean()

    # MACD
    ema12 = stock_df["close"].ewm(span=12, adjust=False).mean()
    ema26 = stock_df["close"].ewm(span=26, adjust=False).mean()
    stock_df["MACD"] = ema12 - ema26
    stock_df["MACD_signal"] = stock_df["MACD"].ewm(span=9, adjust=False).mean()

    # --- OBV chart ---
    fig_obv = go.Figure()
    fig_obv.add_trace(go.Scatter(x=stock_df.index, y=stock_df["OBV"], mode="lines", name="OBV"))
    fig_obv.add_trace(go.Scatter(x=stock_df.index, y=stock_df["OBV_MA_Short"], mode="lines", name=f"OBV MA {obv_short_ma}"))
    fig_obv.add_trace(go.Scatter(x=stock_df.index, y=stock_df["OBV_MA_Long"], mode="lines", name=f"OBV MA {obv_long_ma}"))
    fig_obv.update_layout(title="OBV with Short & Long MA", template="plotly_white", height=450)
    st.plotly_chart(fig_obv, use_container_width=True)

    # --- Price chart ---
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=stock_df.index, y=stock_df["close"], name="Close"))
    fig_price.add_trace(go.Scatter(x=stock_df.index, y=stock_df["SMA_20"], name="SMA 20"))
    fig_price.add_trace(go.Scatter(x=stock_df.index, y=stock_df["EMA_20"], name="EMA 20"))
    fig_price.update_layout(title=f"{symbol} Price with SMA & EMA", template="plotly_white", height=350)
    st.plotly_chart(fig_price, use_container_width=True)

    # RSI & MACD
    col_rsi, col_macd = st.columns(2)
    col_rsi.line_chart(stock_df[["RSI_9"]].dropna(), height=200)
    col_macd.line_chart(stock_df[["MACD","MACD_signal"]].dropna(), height=200)

    # Z-score turn detection
    stock_df["zscore"] = (stock_df["close"] - stock_df["close"].rolling(20).mean()) / stock_df["close"].rolling(20).std()
    bullish_turns = stock_df[stock_df["zscore"] < -2]
    bearish_turns = stock_df[stock_df["zscore"] > 2]

    fig_turns = go.Figure()
    fig_turns.add_trace(go.Scatter(x=stock_df.index, y=stock_df["close"], mode="lines", name="Close"))
    fig_turns.add_trace(go.Scatter(x=bullish_turns.index, y=bullish_turns["close"], mode="markers", name="Bullish", marker=dict(color="green", size=10, symbol="star")))
    fig_turns.add_trace(go.Scatter(x=bearish_turns.index, y=bearish_turns["close"], mode="markers", name="Bearish", marker=dict(color="red", size=10, symbol="star")))
    fig_turns.update_layout(title="Potential Market Turns (Z-score extremes)")
    st.plotly_chart(fig_turns, use_container_width=True)

# --- Market Sentiment Analysis ---
st.markdown("---")
st.subheader("📰 Market Sentiment Analysis")
try:
    news = client.list_ticker_news(symbol, params={
        "published_utc.gte": str(start_date),
        "published_utc.lte": str(end_date),
        "limit": 50
    })
    sentiments = []
    for article in news:
        if hasattr(article, "insights") and article.insights:
            for ins in article.insights:
                if ins.ticker == symbol and hasattr(ins, "sentiment"):
                    sentiments.append(ins.sentiment)
    if sentiments:
        pos = sum(1 for s in sentiments if s=="positive")
        neg = sum(1 for s in sentiments if s=="negative")
        neu = sum(1 for s in sentiments if s=="neutral")
        total = pos+neg+neu
        raw_score = (pos-neg)/total
        st.metric("Positive", pos)
        st.metric("Neutral", neu)
        st.metric("Negative", neg)
    else:
        st.info("No sentiment insights available.")
except Exception as e:
    st.error(f"Error fetching sentiment insights: {e}")
