import yfinance as yf
import pandas as pd
import numpy as np
import random
from tqdm import tqdm

## just remive the warning from the yfinance library
import warnings
warnings.filterwarnings("ignore")

## output directory for the files 
import os
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# global variables
DEBUG = 0
start_year = 2024
end_year = 2025
trials = 10
min_assets = 5
max_assets = 5

###########################################################################
##### step 1 : get the list of the top 100 companies of the sp500 index
###########################################################################


top100_csv_path = 'top_100_sp500.csv' #'sp500_top100.csv'


###########################################################################
##### step 2 : perform the market analysis of a year o n years to calculate the best performance portfolio
###########################################################################

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

################################################
############ performe calculation

print("📥 Loading top 100 tickers...")
top_100_tickers = load_top_100_from_csv(top100_csv_path)


 # === Set Parameters ===
for i in range(start_year, end_year+1):
	year = i
	start_date = f"{year}-01-01"
	end_date = f"{year}-12-31"

	print(f"📈 Fetching S&P 500 return for the year {year}...")
	sp500_return = get_sp500_return(start_date, end_date)
	print(f"✅ S&P 500 return for {year}: {sp500_return:.2%}")

	print("🔎 Evaluating combinations...")
	results_df = evaluate_combinations(
			top_100_tickers,
			sp500_return,
			start_date,
			end_date,
			min_assets=min_assets,
			max_assets=max_assets,
			trials=trials
	)

	output_file = f"outperforming_combinations_{year}.csv"
	### replaced the file name with the output directory
	results_df.to_csv(os.path.join(output_dir, output_file), index=False)
	print(f"\n✅ Done! Saved {len(results_df)} outperforming portfolios to '{output_file}'.")


###########################################################################
##### step 3 : show data 
###########################################################################

def simplify_outperforming_data(input_csv, output_csv):
	df = pd.read_csv(os.path.join(output_dir, input_csv))

	# Keep only the desired columns
	simplified_df = df[['tickers', 'portfolio_return', 'sp500_return', 'beat_by']].copy()

	# Convert returns to percentages with 2 decimal digits
	for col in ['portfolio_return', 'sp500_return', 'beat_by']:
		  simplified_df[col] = (simplified_df[col] * 100).round(2)

	# Show in terminal
	print("📊 Simplified Outperforming Combinations (Returns in %):")
	print(simplified_df.to_string(index=False))

	# Save to CSV
	### replaced the file name with the output directory
	simplified_df.to_csv(os.path.join(output_dir, output_csv), index=False)
	print(f"\n✅ Saved simplified data to '{output_csv}'")

################################################
##### perform calculation

for i in range(start_year, end_year+1):
 input_file  = f"outperforming_combinations_{i}.csv"
 output_file = f"outperforming_combinations_{i}_cleaned.csv"
 simplify_outperforming_data(input_file, output_file)


###########################################################################
##### step 4 : extract th top contributors to the pervious lists 
###########################################################################

import pandas as pd
from collections import defaultdict
import glob

def analyze_contributors(files):
    stock_stats = defaultdict(lambda: {"count": 0, "total_beat": 0})

    for file in files:
        print(f"Processing {file} ...")
        df = pd.read_csv(file)

        for _, row in df.iterrows():
            tickers = eval(row['tickers'])  # Convert string list to actual list
            beat_by = row['beat_by']

            for ticker in tickers:
                stock_stats[ticker]["count"] += 1
                stock_stats[ticker]["total_beat"] += beat_by

    results = []
    for ticker, stats in stock_stats.items():
        avg_beat = stats["total_beat"] / stats["count"]
        results.append({
            "Ticker": ticker,
            "Appearances": stats["count"],
            "Average_Beat_By": avg_beat
        })

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values(by=["Appearances", "Average_Beat_By"], ascending=False).reset_index(drop=True)
    return result_df


################################################
##### perform calculation

files = glob.glob(os.path.join(output_dir, "outperforming_combinations_*_cleaned.csv"))  # Adjust the path/pattern if needed

summary_df = analyze_contributors(files)

print("\nTop contributing stocks across all years:\n")
print(summary_df.to_string(index=False))

summary_df.to_csv(os.path.join(output_dir, "stocks_sorted.csv"), index=False)
print("\n✅ Saved summary to 'stocks_sorted.csv'")


###########################################################################
##### step 5 : get teh top 10 best Contributors stocks and check the new portfolio performance   
###########################################################################
import pandas as pd
import yfinance as yf
from datetime import datetime

def get_annual_return(ticker, year):
    start = f"{year}-01-01"
    end = f"{year}-12-31"
    
    data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if data.empty or 'Close' not in data.columns:
        return None

    # Drop any rows with missing prices
    adj_close = data['Close'].dropna()
    
    if adj_close.empty or len(adj_close) < 2:
        return None

    start_price = float(adj_close.iloc[0])
    end_price = float(adj_close.iloc[-1])

    return float((end_price - start_price) / start_price)

def evaluate_top10_portfolio(contributors_csv, year):
  # Load top contributors
  df = pd.read_csv(contributors_csv)
  top10 = df.sort_values(by=["Appearances", "Average_Beat_By"], ascending=False).head(10)
  tickers = top10["Ticker"].tolist()

  print(f"\n📊 Evaluating portfolio for year {year}")
  print(f"Top 10 tickers: {tickers}\n")

  portfolio_return = 0
  valid_tickers = 0
  weights = 1 / len(tickers)

  for ticker in tickers:
      ret = get_annual_return(ticker, year)
      if ret is not None:
          portfolio_return += ret * weights
          if (DEBUG):
          	print(f"{ticker}: {ret*100:.2f}%")
          valid_tickers += 1
      else:
          print(f"{ticker}: ❌ No data")

  if valid_tickers == 0:
      print("❌ No valid tickers. Exiting.")
      return

  # Get S&P 500 return
  sp500_return = get_annual_return("^GSPC", year)
  print(f"\n📊 Evaluating portfolio for year {year}")
  print(f"\n📈 Portfolio Return: {portfolio_return*100:.2f}%")
  print(f"📉 S&P 500 Return: {sp500_return*100:.2f}%" if sp500_return else "❌ Couldn't fetch S&P 500 data")

  if sp500_return:
      delta = portfolio_return - sp500_return
      print(f"🏆 Beat by: {delta*100:.2f}%")

################################################
##### perform calculation

for i in range(start_year, end_year+1):
 contributors_file = "stocks_sorted.csv"
 test_year = i  # Change this to any year you'd like to test
 evaluate_top10_portfolio(os.path.join(output_dir, contributors_file), test_year)















