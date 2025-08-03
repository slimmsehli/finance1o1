import pandas as pd

def simplify_outperforming_data(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    # Keep only the desired columns
    simplified_df = df[['tickers', 'portfolio_return', 'sp500_return', 'beat_by']].copy()

    # Convert returns to percentages with 2 decimal digits
    for col in ['portfolio_return', 'sp500_return', 'beat_by']:
        simplified_df[col] = (simplified_df[col] * 100).round(2)

    # Show in terminal
    print("📊 Simplified Outperforming Combinations (Returns in %):")
    print(simplified_df.to_string(index=False))

    # Save to CSV
    simplified_df.to_csv(output_csv, index=False)
    print(f"\n✅ Saved simplified data to '{output_csv}'")

if __name__ == "__main__":
    input_file = "outperforming_combinations_2024.csv"
    output_file = "outperforming_simple_2024.csv"
    simplify_outperforming_data(input_file, output_file)

