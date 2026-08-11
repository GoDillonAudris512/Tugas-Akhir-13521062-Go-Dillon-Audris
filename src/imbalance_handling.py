import numpy as np
import pandas as pd
from pathlib import Path

root_dir = Path.cwd()
data_dir = root_dir / 'data' / 'balanced'

np.random.seed(42)
all_df = pd.read_csv(data_dir / 'mda_manual_v20251224_with_fold.csv')
balanced_df_ver_2 = pd.read_csv(data_dir / 'balanced_mda_manual_with_fold_ver_2.csv')

# Create df per class
negative_df = balanced_df_ver_2[balanced_df_ver_2['YoY_Close_Category'] == 'negative']
neutral_df = balanced_df_ver_2[balanced_df_ver_2['YoY_Close_Category'] == 'neutral']
positive_df = balanced_df_ver_2[balanced_df_ver_2['YoY_Close_Category'] == 'positive']    

# # Get all neutral data
all_neutral_df = all_df[all_df['YoY_Close_Category'] == 'neutral']

# current_neutral_per_fold = neutral_df['Ticker_Fold'].value_counts().sort_index().to_list()
used_ticker = set(balanced_df_ver_2['Ticker'].unique())
all_neutral_ticker = set(all_neutral_df['Ticker'].unique())
unassigned_neutral_ticker = list(all_neutral_ticker - used_ticker)

neutral_count_per_fold = neutral_df['Ticker_Fold'].value_counts().sort_index().to_list()

for ticker in unassigned_neutral_ticker:
    # Ignore remaining ticker if neutral sum >= 317
    if sum(neutral_count_per_fold) >= 317:
        break

    # Find all rows in all_neutral_df with this ticker
    ticker_rows = all_neutral_df[all_neutral_df['Ticker'] == ticker]

    # Assign new fold number to these rows
    new_fold_number = neutral_count_per_fold.index(min(neutral_count_per_fold))
    
    for ticker, year, label in zip(ticker_rows['Ticker'], ticker_rows['Year'], ticker_rows['YoY_Close_Category']):
        neutral_df = neutral_df.append({
            'Ticker': ticker,
            'Year': year,
            'YoY_Close_Category': label,
            'Ticker_Fold': new_fold_number,
        }, ignore_index=True)

    neutral_count_per_fold[new_fold_number] += len(ticker_rows)

balanced_df_ver_3 = pd.concat([negative_df, neutral_df, positive_df], ignore_index=True)
balanced_df_ver_3.sort_values(by=["Ticker_Fold", "Ticker", "Year"], inplace=True)    
balanced_df_ver_3.to_csv(data_dir / 'balanced_mda_manual_with_fold_ver_3.csv', index=False)

balanced_df_ver_4 = balanced_df_ver_3.groupby('Ticker_Fold').apply(lambda x: x.sample(frac=1, random_state=42)).reset_index(drop=True)
balanced_df_ver_4.to_csv(data_dir / 'balanced_mda_manual_with_fold_ver_4.csv', index=False)