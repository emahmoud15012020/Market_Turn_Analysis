import os, streamlit as st
import pandas as pd
from polygon import RESTClient
import yfinance as yf
from datetime import date
import numpy as np

st.set_page_config(layout="wide")
st.title("Market Turn Analysis Dashboard")

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
st.subheader("VIX Analysis")
try:
    vix_df = yf.download("^VIX", start=str(start_date), end=str(end_date))
    if not vix_df.empty:
        vix_df = vix_df.reset_index()
        if isinstance(vix_df.columns, pd.MultiIndex):
            vix_df.columns = [col[0] for col in vix_df.columns]
        vix_df["MA_3"] = vix_df["Close"].rolling(3).mean()
        vix_df["MA_9"] = vix_df["Close"].rolling(9).mean()

        latest_vix = vix_df.iloc[-1]
        vix_signal = ""
        if latest_vix["MA_3"] > latest_vix["MA_9"]:
            vix_signal = "Bearish Signal (Volatility Rising)"
        else:
            vix_signal = "Bullish Signal (Volatility Falling)"

        st.line_chart(vix_df.set_index("Date")[["Close", "MA_3", "MA_9"]])
        st.info(f"Latest VIX: {latest_vix['Close']:.2f} → {vix_signal}")
except Exception as e:
    st.error(f"Error fetching VIX: {e}")

# --- PCR Analysis ---
st.markdown("---")
st.subheader("Put/Call Ratio (PCR)")
pcr_placeholder = st.empty()
pcr_gauge = st.empty()

try:
    with st.spinner("Fetching PCR data..."):
        total_puts = 0
        total_calls = 0
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
            # Gauge visualization using st.progress
            pcr_gauge.progress(min(max(pcr/2, 0.0), 1.0))
        else:
            pcr_placeholder.warning("No call option data to compute PCR.")

except Exception as e:
    st.error(f"Error fetching PCR: {e}")

# --- Technical Indicators ---
st.markdown("---")
st.subheader("Technical Indicators")

try:
    # Fetch OHLCV from Polygon
    aggs = []
    for a in client.list_aggs(
        symbol,
        1,
        "day",
        str(start_date),
        str(end_date),
        adjusted="true",
        sort="asc",
        limit=500,
    ):
        aggs.append(a)

    if not aggs:
        st.warning("No Polygon OHLCV data available for this symbol/date range.")
    else:
        # Convert Polygon Agg objects to DataFrame
        data = []
        for a in aggs:
            data.append({
                "timestamp": a.timestamp,
                "open": a.open,
                "high": a.high,
                "low": a.low,
                "close": a.close,
                "volume": a.volume
            })

        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        # --- RSI(9) ---
        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(9).mean()
        avg_loss = loss.rolling(9).mean()
        rs = avg_gain / avg_loss
        df["RSI_9"] = 100 - (100 / (1 + rs))
        rsi_signal = "Neutral"
        rsi_color = "gray"
        if len(df) >= 2 and not np.isnan(df["RSI_9"].iloc[-1]):
            if df["RSI_9"].iloc[-1] > df["RSI_9"].iloc[-2]:
                rsi_signal = "RSI Rising → Bullish 📈"
                rsi_color = "green"
            else:
                rsi_signal = "RSI Falling → Bearish 📉"
                rsi_color = "red"

        # --- OBV ---
        obv = [0]
        for i in range(1, len(df)):
            if df["close"].iloc[i] > df["close"].iloc[i-1]:
                obv.append(obv[-1] + df["volume"].iloc[i])
            elif df["close"].iloc[i] < df["close"].iloc[i-1]:
                obv.append(obv[-1] - df["volume"].iloc[i])
            else:
                obv.append(obv[-1])
        df["OBV"] = obv
        obv_signal = "OBV not available"
        obv_color = "gray"
        if len(df) >= 2:
            if df["OBV"].iloc[-1] > df["OBV"].iloc[-2]:
                obv_signal = "OBV Rising → Bullish 📈"
                obv_color = "green"
            else:
                obv_signal = "OBV Falling → Bearish 📉"
                obv_color = "red"

        # --- SMA / EMA ---
        df["SMA_20"] = df["close"].rolling(20).mean()
        df["EMA_20"] = df["close"].ewm(span=20, adjust=False).mean()

        # --- MACD ---
        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = ema12 - ema26
        df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["MACD_diff"] = df["MACD"] - df["MACD_signal"]

        # --- Display Signals with Colors ---
        st.markdown(
            f'<h4>Signals:</h4>'
            f'<span style="color:{rsi_color}; font-weight:bold;">{rsi_signal}</span> | '
            f'<span style="color:{obv_color}; font-weight:bold;">{obv_signal}</span>',
            unsafe_allow_html=True
        )

        # --- Plot Charts ---
        st.markdown("### Price with SMA & EMA")
        st.line_chart(df[["close", "SMA_20", "EMA_20"]])

        st.markdown("### RSI, OBV & MACD")
        st.line_chart(df[["RSI_9", "OBV", "MACD", "MACD_signal"]])

except Exception as e:
    st.error(f"Error fetching technical indicators from Polygon: {e}")