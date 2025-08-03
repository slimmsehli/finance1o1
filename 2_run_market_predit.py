import yfinance as yf
import pandas as pd
import numpy as np
import random
from tqdm import tqdm

def load_top_100_from_csv(filepath):
    df = pd.read_csv(filepath)
    return df['Ticker'].tolist()

def get_annual_return(tickers, start_date, end_date, weights=None):
    data = yf.download(tickers, start=start_date, end=end_date)['Close']
    data = data.dropna(axis=1)

    if len(data.columns) == 0:
        raise ValueError("No valid tickers returned.")

    if weights is None:
        weights = np.ones(len(data.columns)) / len(data.columns)
    else:
        weights = np.array(weights)

    returns = data.pct_change().dropna()
    portfolio_return = (returns @ weights).add(1).prod() - 1
    return portfolio_return

def evaluate_combinations(tickers, sp500_return, start_date, end_date, min_assets=5, max_assets=10, trials=100):
    outperformers = []

    for _ in tqdm(range(trials), desc="Testing combinations"):
        n_assets = random.randint(min_assets, max_assets)
        sample = random.sample(tickers, n_assets)
        weights = np.random.dirichlet(np.ones(n_assets), size=1)[0]  # weights sum to 1

        try:
            portfolio_return = get_annual_return(sample, start_date, end_date, weights)
            if portfolio_return > sp500_return:
                outperformers.append({
                    "tickers": sample,
                    "weights": weights.tolist(),
                    "portfolio_return": portfolio_return,
                    "sp500_return": sp500_return,
                    "beat_by": portfolio_return - sp500_return
                })
        except Exception:
            continue

    return pd.DataFrame(outperformers)

def get_sp500_return(start_date, end_date):
    return get_annual_return(['^GSPC'], start_date, end_date)

if __name__ == "__main__":
    # === Set Parameters ===
    top100_csv_path = 'top_100_sp500.csv' #'sp500_top100.csv'
    year = 2024
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    trials = 100

    print("📥 Loading top 100 tickers...")
    top_100_tickers = load_top_100_from_csv(top100_csv_path)

    print("📈 Fetching S&P 500 return...")
    sp500_return = get_sp500_return(start_date, end_date)
    print(f"✅ S&P 500 return for {year}: {sp500_return:.2%}")

    print("🔎 Evaluating combinations...")
    results_df = evaluate_combinations(
        top_100_tickers,
        sp500_return,
        start_date,
        end_date,
        min_assets=5,
        max_assets=10,
        trials=trials
    )

    output_file = f"outperforming_combinations_{year}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n✅ Done! Saved {len(results_df)} outperforming portfolios to '{output_file}'.")

