import pandas as pd
import yfinance as yf
from tqdm import tqdm
import time

def load_sp500_constituents():
    url = 'https://datahub.io/core/s-and-p-500-companies/r/constituents.csv'
    df = pd.read_csv(url)
    return df[['Symbol', 'Security']]

def get_market_cap(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return info.get('marketCap', None)
    except:
        return None

def get_sp500_sorted_by_market_cap():
    df = load_sp500_constituents()
    data = []

    print("Fetching market caps (may take a few minutes)...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        ticker = row['Symbol']
        name = row['Security']

        # Adjust for special tickers like BRK.B → BRK-B
        ticker = ticker.replace('.', '-')

        mcap = get_market_cap(ticker)
        time.sleep(0.2)  # avoid rate limiting
        if mcap:
            data.append({
                'Ticker': ticker,
                'Name': name,
                'MarketCap': mcap
            })

    df_out = pd.DataFrame(data)
    df_out = df_out.sort_values(by='MarketCap', ascending=False).reset_index(drop=True)
    return df_out

if __name__ == "__main__":
    sorted_df = get_sp500_sorted_by_market_cap()
    sorted_df.to_csv("sp500_full_sorted.csv", index=False)

    top100_df = sorted_df.head(100)
    top100_df.to_csv("sp500_top100.csv", index=False)

    print("\n✅ Done! Files saved:")
    print("- sp500_full_sorted.csv (all 500 sorted by MarketCap)")
    print("- sp500_top100.csv (top 100 companies)")

