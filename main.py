# app.py
import os
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from bs4 import BeautifulSoup
from polygon import RESTClient
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.core.os_manager import ChromeType

# ---------------------------
# Streamlit Page Setup
# ---------------------------
st.set_page_config(layout="wide")
st.title("📊 Market Turn Analysis Dashboard")

# ---------------------------
# User Inputs
# ---------------------------
symbol = st.text_input("Enter a stock symbol", "TQQQ")

with st.sidebar:
    polygon_api_key = st.text_input("Polygon API Key", type="password")
    polygon_api_key = polygon_api_key.strip() or "hZGF1b86QLsKsAh7HHCHFGUxLcYwh3qp"

    start_date = st.date_input("Start Date", pd.to_datetime("2025-01-01"))
    end_date = st.date_input("End Date", date.today())

# Authenticate Polygon
client = RESTClient(polygon_api_key)

# ---------------------------
# Scraper Function for PCR
# ---------------------------
def scrape_cboe_daily_pcr_batch(n=10):
    """Scrape last n days of TOTAL PUT/CALL RATIO using headless Chromium"""
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    service = Service(ChromeDriverManager(chrome_type=ChromeType.CHROMIUM).install())
    driver = webdriver.Chrome(service=service, options=options)

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
                        (By.XPATH, "//table[contains(@class,'TableComponents__StyledTable')]")
                    )
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


@st.cache_data
def fetch_last_n_pcr(n=10):
    return scrape_cboe_daily_pcr_batch(n)

# ---------------------------
# Stock Quote Section
# ---------------------------
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

# ---------------------------
# VIX Analysis
# ---------------------------
st.markdown("---")
st.subheader("📈 VIX Analysis")
try:
    vix_df = yf.download("^VIX", start=str(start_date), end=str(end_date))

    if not vix_df.empty:
        if isinstance(vix_df.columns, pd.MultiIndex):
            vix_df.columns = vix_df.columns.get_level_values(0)

        vix_df = vix_df.reset_index()
        ma_short = st.slider("Short-term MA (days)", 1, 50, 3)
        ma_long = st.slider("Long-term MA (days)", 5, 200, 9)

        vix_df["MA_short"] = vix_df["Close"].rolling(ma_short).mean()
        vix_df["MA_long"] = vix_df["Close"].rolling(ma_long).mean()

        latest_vix = vix_df.iloc[-1]
        ma_short_val, ma_long_val = float(latest_vix["MA_short"]), float(latest_vix["MA_long"])

        vix_signal = (
            f"Bearish Signal (Volatility Rising: {ma_short}-day > {ma_long}-day)"
            if ma_short_val > ma_long_val
            else f"Bullish Signal (Volatility Falling: {ma_short}-day < {ma_long}-day)"
        )

        st.line_chart(vix_df.set_index("Date")[["Close", "MA_short", "MA_long"]])
        st.info(f"Latest VIX: {latest_vix['Close']:.2f} → {vix_signal}")
except Exception as e:
    st.error(f"Error fetching VIX: {e}")

# ---------------------------
# PCR Gauge
# ---------------------------
st.markdown("---")
st.subheader("⚖️ Put/Call Ratio (PCR) Gauge")

try:
    total_puts, total_calls = 0, 0
    cursor = None
    while True:
        response = client.list_snapshot_options_chain(
            symbol,
            params={"order": "asc", "limit": 250, "sort": "ticker", "cursor": cursor}
        )
        for o in response:
            contract_type = o.details.contract_type
            oi = getattr(o, "open_interest", 0) or 0
            if contract_type == "put":
                total_puts += oi
            elif contract_type == "call":
                total_calls += oi
        cursor = getattr(response, "next_url", None)
        if not cursor:
            break

    if total_calls > 0:
        pcr = total_puts / total_calls
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
except Exception as e:
    st.error(f"Error fetching PCR gauge: {e}")

# ---------------------------
# PCR Table
# ---------------------------
st.markdown("---")
st.subheader("📊 Put/Call Ratio Table")

df_pcr = fetch_last_n_pcr(15)

if df_pcr.empty:
    st.error("No data available. Try refreshing.")
else:
    st.table(df_pcr.style.hide(axis="index"))
