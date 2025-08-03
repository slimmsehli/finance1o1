import pandas as pd
import yfinance as yf
import os
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

DEBUG = 0

def get_annual_return(ticker, year):
    start = f"{year}-01-01"
    end = f"{year}-12-31"
    
    data = yf.download(ticker, start=start, end=end, progress=False)
    if data.empty or 'Close' not in data.columns:
        return None

    # Drop any rows with missing prices
    adj_close = data['Close'].dropna()
    
    if adj_close.empty or len(adj_close) < 2:
        return None

    start_price = adj_close.iloc[0]
    end_price = adj_close.iloc[-1]

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
            print(f"{ticker}: {ret*100:.2f}%")
            valid_tickers += 1
        else:
            print(f"{ticker}: ❌ No data")

    if valid_tickers == 0:
        print("❌ No valid tickers. Exiting.")
        return
	 
    # Get S&P 500 return
    sp500_return = get_annual_return("^GSPC", year)
    
    beat_by = None
    if sp500_return is not None:
        beat_by = portfolio_return - sp500_return
    
    result = {
        "Year": year,
        "Tickers": ','.join(tickers),
        "Portfolio_Return": round(portfolio_return * 100, 2),
        "SP500_Return": round(sp500_return * 100, 2) if sp500_return else None,
        "Beat_By": round(beat_by * 100, 2) if beat_by else None
    }
    
    if DEBUG:
        print(f"\n📈 Portfolio Return: {result['Portfolio_Return']}%")
        print(f"📉 S&P 500 Return: {result['SP500_Return']}%")
        print(f"🏆 Beat by: {result['Beat_By']}%")
    return result
    #print(f"\n📈 Portfolio Return: {portfolio_return*100:.2f}%")
    #print(f"📉 S&P 500 Return: {sp500_return*100:.2f}%" if sp500_return else "❌ Couldn't fetch S&P 500 data")

    #if sp500_return:
    #    delta = portfolio_return - sp500_return
    #    print(f"🏆 Beat by: {delta*100:.2f}%")

allresults = []
contributors_file = "outputs/stocks_sorted.csv"
test_year = 2023  # Change this to any year you'd like to test

result = evaluate_top10_portfolio(contributors_file, test_year)
if result:
	allresults.append(result)
results_df = pd.DataFrame(allresults)
results_df.to_csv(f"top10_portfolio_performance_{test_year}.csv", index=False)

#results_df.to_csv(f"top10_portfolio_performance_{start_year}_{end_year}.csv", index=False)

































