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

if __name__ == "__main__":
    # List your outperforming CSV files here, or use glob to find them automatically:
    # e.g., files = ['outperforming_simple_2023.csv', 'outperforming_simple_2024.csv', 'outperforming_simple_2025.csv']
    files = glob.glob("outperforming_combinations_*_cleaned.csv")  # Adjust the path/pattern if needed

    summary_df = analyze_contributors(files)

    print("\nTop contributing stocks across all years:\n")
    print(summary_df.to_string(index=False))

    summary_df.to_csv("stocks_sorted.csv", index=False)
    print("\n✅ Saved summary to 'stocks_sorted.csv'")
























