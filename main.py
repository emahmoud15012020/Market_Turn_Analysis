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

        # Let user choose MA windows
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

        st.line_chart(vix_df.set_index("Date")[["Close", "MA_short", "MA_long"]])
        st.info(f"Latest VIX: {latest_vix['Close']:.2f} → {vix_signal}")

except Exception as e:
    st.error(f"Error fetching VIX: {e}")



# --- PCR Analysis ---
st.markdown("---")
st.subheader("⚖️ Put/Call Ratio (PCR)")

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

        # Gauge thresholds: <0.7 = bullish, 0.7-1.2 = neutral, >1.2 = bearish
        fig_pcr = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pcr,
            number={'valueformat': ".2f"},
            title={'text': "Put/Call Ratio", 'font': {'size': 20}},
            gauge={
                'axis': {'range': [0, 2], 'tickwidth': 1, 'tickcolor': "darkgrey"},
                'bar': {'color': "blue"},
                'steps': [
                    {'range': [0, 0.7], 'color': "green"},
                    {'range': [0.7, 1.2], 'color': "white"},
                    {'range': [1.2, 2], 'color': "red"},
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': pcr
                }
            }
        ))

        st.plotly_chart(fig_pcr, use_container_width=True)

    else:
        st.warning("No call option data to compute PCR.")
except Exception as e:
    st.error(f"Error fetching PCR: {e}")



# --- Technical Analysis Section ---
st.markdown("---")
st.subheader("📊 Technical Analysis")

# ----------------------
# Data Fetch Helper
# ----------------------
@st.cache_data(ttl=300)
def fetch_polygon_aggs(symbol, start_date, end_date, _client):  # _client avoids hashing error
    aggs = []
    try:
        for a in _client.list_aggs(
            symbol, 1, "day", str(start_date), str(end_date),
            adjusted="true", sort="asc", limit=500
        ):
            aggs.append(a)
        return aggs, None
    except Exception as e:
        return None, str(e)

stock_df = None

# 1) Try Polygon data
aggs, fetch_err = fetch_polygon_aggs(symbol, start_date, end_date, client)
if aggs:
    try:
        data = [{"timestamp": a.timestamp, "open": a.open, "high": a.high,
                 "low": a.low, "close": a.close, "volume": a.volume} for a in aggs]
        stock_df = pd.DataFrame(data)
        stock_df['timestamp'] = pd.to_datetime(stock_df['timestamp'], unit='ms')
        stock_df.set_index('timestamp', inplace=True)
        stock_df = stock_df.sort_index()
    except Exception as e:
        st.warning(f"Error parsing Polygon data → falling back to Yahoo Finance: {e}")
        stock_df = None
elif fetch_err:
    st.info(f"Polygon fetch failed, using Yahoo Finance: {fetch_err}")

# 2) Fallback to yfinance
if stock_df is None:
    try:
        yf_df = yf.download(symbol, start=str(start_date), end=str(end_date))
        if not yf_df.empty and "Volume" in yf_df.columns:
            yf_df = yf_df.rename(columns={
                "Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"
            })
            stock_df = yf_df[["open","high","low","close","volume"]].copy()
            stock_df.index = pd.to_datetime(stock_df.index)
            stock_df = stock_df.sort_index()
            st.info("Using Yahoo Finance data.")
        else:
            st.warning("Yahoo Finance returned no volume data (OBV requires volume).")
    except Exception as e:
        st.error(f"Failed to fetch Yahoo Finance data: {e}")

# ----------------------
# Indicators
# ----------------------
if stock_df is None or stock_df.empty:
    st.warning("No data available for technical indicators.")
else:
    # RSI(9)
    delta = stock_df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(9).mean() / loss.rolling(9).mean()
    stock_df["RSI_9"] = 100 - (100 / (1 + rs))

    # OBV
    price_diff = stock_df["close"].diff().fillna(0)
    direction = np.sign(price_diff)  # +1, 0, -1
    stock_df["OBV"] = (direction * stock_df["volume"]).cumsum()

    # Sliders for OBV Short & Long SMA
    col1, col2 = st.columns(2)
    with col1:
        obv_short_ma = st.slider("OBV Short MA (days)", min_value=2, max_value=50, value=10)
    with col2:
        obv_long_ma = st.slider("OBV Long MA (days)", min_value=obv_short_ma+1, max_value=200, value=30)

    stock_df["OBV_MA_Short"] = stock_df["OBV"].rolling(window=obv_short_ma).mean()
    stock_df["OBV_MA_Long"] = stock_df["OBV"].rolling(window=obv_long_ma).mean()

    # OBV Signal
    latest_rows = stock_df[["OBV_MA_Short","OBV_MA_Long"]].dropna()
    if not latest_rows.empty:
        latest = latest_rows.iloc[-1]
        if latest["OBV_MA_Short"] > latest["OBV_MA_Long"]:
            obv_signal = "📈 Bullish OBV Signal (Short > Long)"
        else:
            obv_signal = "📉 Bearish OBV Signal (Short < Long)"
    else:
        obv_signal = "Not enough data for OBV MAs."

    # Price SMA / EMA
    stock_df["SMA_20"] = stock_df["close"].rolling(20).mean()
    stock_df["EMA_20"] = stock_df["close"].ewm(span=20, adjust=False).mean()

    # MACD
    ema12 = stock_df["close"].ewm(span=12, adjust=False).mean()
    ema26 = stock_df["close"].ewm(span=26, adjust=False).mean()
    stock_df["MACD"] = ema12 - ema26
    stock_df["MACD_signal"] = stock_df["MACD"].ewm(span=9, adjust=False).mean()

    # ----------------------
    # Charts
    # ----------------------

    # OBV with Short & Long SMA
    fig_obv = go.Figure()
    fig_obv.add_trace(go.Scatter(x=stock_df.index, y=stock_df["OBV"],
                                 mode="lines", name="OBV", line=dict(color="blue")))
    fig_obv.add_trace(go.Scatter(x=stock_df.index, y=stock_df["OBV_MA_Short"],
                                 mode="lines", name=f"OBV SMA ({obv_short_ma})", line=dict(color="green")))
    fig_obv.add_trace(go.Scatter(x=stock_df.index, y=stock_df["OBV_MA_Long"],
                                 mode="lines", name=f"OBV SMA ({obv_long_ma})", line=dict(color="red", dash="dot")))
    fig_obv.update_layout(title="OBV with Short & Long SMA", template="plotly_white", height=450)
    st.plotly_chart(fig_obv, use_container_width=True)
    st.info(obv_signal)

    # Price with SMA/EMA
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=stock_df.index, y=stock_df["close"], name="Close"))
    fig_price.add_trace(go.Scatter(x=stock_df.index, y=stock_df["SMA_20"], name="SMA 20"))
    fig_price.add_trace(go.Scatter(x=stock_df.index, y=stock_df["EMA_20"], name="EMA 20"))
    fig_price.update_layout(title=f"{symbol} Price with SMA & EMA", template="plotly_white", height=350)
    st.plotly_chart(fig_price, use_container_width=True)

    # RSI and MACD
    col_rsi, col_macd = st.columns(2)
    with col_rsi:
        st.line_chart(stock_df[["RSI_9"]].dropna(), height=200)
    with col_macd:
        st.line_chart(stock_df[["MACD", "MACD_signal"]].dropna(), height=200)




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
