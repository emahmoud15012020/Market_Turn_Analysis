import os, streamlit as st
import pandas as pd
from polygon import RESTClient
import yfinance as yf
from datetime import date
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("📊 Market Turn Analysis Dashboard")

# --- Default inputs ---
symbol = st.text_input("Enter a stock symbol", "TQQQ")

with st.sidebar:
    polygon_api_key = st.text_input("Polygon API Key", type="password")
    polygon_api_key = polygon_api_key.strip() or "hZGF1b86QLsKsAh7HHCHFGUxLcYwh3qp"

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
        # --- Fix MultiIndex columns ---
        if isinstance(vix_df.columns, pd.MultiIndex):
            vix_df.columns = vix_df.columns.get_level_values(0)

        vix_df = vix_df.reset_index()
        vix_df["MA_3"] = vix_df["Close"].rolling(3).mean()
        vix_df["MA_9"] = vix_df["Close"].rolling(9).mean()

        latest_vix = vix_df.iloc[-1]

        ma3 = float(latest_vix["MA_3"])
        ma9 = float(latest_vix["MA_9"])

        if ma3 > ma9:
            vix_signal = "Bearish Signal (Volatility Rising)"
        else:
            vix_signal = "Bullish Signal (Volatility Falling)"

        st.line_chart(vix_df.set_index("Date")[["Close", "MA_3", "MA_9"]])
        st.info(f"Latest VIX: {latest_vix['Close']:.2f} → {vix_signal}")

except Exception as e:
    st.error(f"Error fetching VIX: {e}")


# --- PCR Analysis ---
st.markdown("---")
st.subheader("⚖️ Put/Call Ratio (PCR)")
pcr_placeholder = st.empty()
pcr_gauge = st.empty()

try:
    total_puts = total_calls = 0
    cursor = None
    while True:
        response = client.list_snapshot_options_chain(
            symbol,
            params={"order":"asc", "limit":250, "sort":"ticker", "cursor":cursor}
        )
        for o in response:
            contract_type = o.details.contract_type
            oi = getattr(o, "open_interest", 0) or 0
            if contract_type == "put": total_puts += oi
            elif contract_type == "call": total_calls += oi
        cursor = getattr(response, "next_url", None)
        if not cursor: break

    if total_calls > 0:
        pcr = total_puts / total_calls
        pcr_placeholder.success(f"PCR (Open Interest): {pcr:.2f}")
        pcr_gauge.progress(min(max(pcr/2, 0.0), 1.0))
    else:
        pcr_placeholder.warning("No call option data to compute PCR.")
except Exception as e:
    st.error(f"Error fetching PCR: {e}")

# --- Technical Indicators ---
st.markdown("---")
st.subheader("📊 Technical Indicators")

stock_df = None
try:
    aggs = []
    for a in client.list_aggs(
        symbol, 1, "day", str(start_date), str(end_date),
        adjusted="true", sort="asc", limit=500
    ):
        aggs.append(a)

    if aggs:
        data = [{"timestamp": a.timestamp, "open": a.open, "high": a.high,
                 "low": a.low, "close": a.close, "volume": a.volume} for a in aggs]
        stock_df = pd.DataFrame(data)
        stock_df['timestamp'] = pd.to_datetime(stock_df['timestamp'], unit='ms')
        stock_df.set_index('timestamp', inplace=True)

        # RSI(9)
        delta = stock_df["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        rs = gain.rolling(9).mean() / loss.rolling(9).mean()
        stock_df["RSI_9"] = 100 - (100 / (1 + rs))

        # OBV
        obv = [0]
        for i in range(1, len(stock_df)):
            if stock_df["close"].iloc[i] > stock_df["close"].iloc[i-1]:
                obv.append(obv[-1] + stock_df["volume"].iloc[i])
            elif stock_df["close"].iloc[i] < stock_df["close"].iloc[i-1]:
                obv.append(obv[-1] - stock_df["volume"].iloc[i])
            else:
                obv.append(obv[-1])
        stock_df["OBV"] = obv

        # SMA / EMA
        stock_df["SMA_20"] = stock_df["close"].rolling(20).mean()
        stock_df["EMA_20"] = stock_df["close"].ewm(span=20, adjust=False).mean()

        # MACD
        ema12 = stock_df["close"].ewm(span=12, adjust=False).mean()
        ema26 = stock_df["close"].ewm(span=26, adjust=False).mean()
        stock_df["MACD"] = ema12 - ema26
        stock_df["MACD_signal"] = stock_df["MACD"].ewm(span=9, adjust=False).mean()

        st.line_chart(stock_df[["close", "SMA_20", "EMA_20"]])
        st.line_chart(stock_df[["RSI_9", "OBV", "MACD", "MACD_signal"]])
    else:
        st.warning("No OHLCV data available.")
except Exception as e:
    st.error(f"Error fetching technical indicators: {e}")

# --- Market Sentiment Analysis ---
st.markdown("---")
st.subheader("📰 Market Sentiment Analysis")

sentiment_prob = 0.5
try:
    news = client.list_ticker_news(
        symbol,
        params={"published_utc.gte": str(start_date),
                "published_utc.lte": str(end_date),
                "limit": 50}
    )
    sentiments = []
    for article in news:
        if hasattr(article, "insights") and article.insights:
            for ins in article.insights:
                if ins.ticker == symbol and hasattr(ins, "sentiment"):
                    sentiments.append(ins.sentiment)

    if sentiments:
        pos = sum(1 for s in sentiments if s == "positive")
        neg = sum(1 for s in sentiments if s == "negative")
        neu = sum(1 for s in sentiments if s == "neutral")
        total = pos + neg + neu

        raw_score = (pos - neg) / total
        sentiment_prob = (raw_score + 1) / 2

        st.metric("Positive", pos)
        st.metric("Neutral", neu)
        st.metric("Negative", neg)
    else:
        st.info("No sentiment insights available.")
except Exception as e:
    st.error(f"Error fetching sentiment insights: {e}")

# --- Fear & Greed Gauge ---
st.markdown("---")
st.subheader("🕹️ Fear & Greed Gauge")

try:
    gauge_value = sentiment_prob * 100
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = gauge_value,
        title = {'text': f"Market Sentiment for {symbol}", 'font': {'size': 24}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 2},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 25], 'color': '#800000'},
                {'range': [25, 50], 'color': '#FF4500'},
                {'range': [50, 75], 'color': '#FFD700'},
                {'range': [75, 100], 'color': '#00FF00'}
            ],
            'threshold': {'line': {'color': "black", 'width': 4},
                          'thickness': 0.75, 'value': gauge_value}
        }
    ))
    st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.error(f"Error rendering sentiment gauge: {e}")

# --- Market Turn Treemap ---
st.markdown("---")
st.subheader("🌳 Market Turn Treemap")

try:
    rsi_score = stock_df["RSI_9"].iloc[-1]/100 if stock_df is not None else 0.5
    obv_score = 0.7 if stock_df is not None and stock_df["OBV"].iloc[-1] > stock_df["OBV"].iloc[-2] else 0.3
    vix_score = 0.3 if "MA_3" in vix_df and vix_df["MA_3"].iloc[-1] > vix_df["MA_9"].iloc[-1] else 0.7
    pcr_score = 0.5
    if 'pcr' in locals():
        if pcr < 0.7: pcr_score = 0.7
        elif pcr > 1.2: pcr_score = 0.3

    factors = {
        "RSI": rsi_score, "OBV": obv_score, "VIX": vix_score,
        "PCR": pcr_score, "Sentiment": sentiment_prob
    }
    df_factors = pd.DataFrame({"Factor": factors.keys(), "Score": factors.values()})
    fig = px.treemap(df_factors, path=["Factor"], values="Score", color="Score",
                     color_continuous_scale=["red", "orange", "yellow", "green"])
    st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.error(f"Error building Market Turn Treemap: {e}")
