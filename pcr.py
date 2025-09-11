from selenium import webdriver
from selenium.webdriver.common.by import By
import pandas as pd
import time

# URL de la page CBOE
url = "https://www.cboe.com/us/options/market_statistics/daily/?dt=2025-09-08"

# 1. Lancer Selenium avec Chrome (headless)
options = webdriver.ChromeOptions()
options.add_argument("--headless")  # pas de fenêtre
driver = webdriver.Chrome(options=options)

driver.get(url)
time.sleep(5)  # attendre que le JS charge les données

# 2. Trouver le tableau (exemple : Put/Call Ratios)
rows = driver.find_elements(By.CSS_SELECTOR, "table tbody tr")

data = []
for row in rows:
    cols = row.find_elements(By.TAG_NAME, "td")
    cols = [col.text.strip() for col in cols]
    if cols:
        data.append(cols)

driver.quit()

# 3. Transformer en DataFrame
df = pd.DataFrame(data, columns=["Category", "Put Volume", "Call Volume", "Put/Call Ratio"])

print(df)

# 4. Récupérer le PCR Equity ou Total
pcr_equity = df[df["Category"].str.contains("Equity", case=False)]["TOTAL PUT/CALL RATIO"].values[0]
print("PCR Equity:", pcr_equity)
