import os
import pandas as pd
import numpy as np
from kaggle.api.kaggle_api_extended import KaggleApi
from pandas.tseries.offsets import BDay

# take a look at potentially including UMCSENT, data to feb 2024
# review code to handle quartely data noted on non business days

# Initialize Kaggle API
api = KaggleApi()
api.authenticate()

def download_dataset(dataset_name, path='datasets/'):
    """
    Downloads a dataset from Kaggle and saves it to the specified path.
    """
    api.dataset_download_files(dataset_name, path=path, unzip=True)

def load_and_clean_data(file_path, date_col_index, value_col_index, percent_change=False):
    """
    Loads a CSV file, cleans the data, performs interpolation only across quarters
    where changes occur, and optionally converts values to percent change from the previous
    business day data point.
    """
    try:
        df = pd.read_csv(file_path)
        df = df.iloc[:, [date_col_index, value_col_index]]
        df.columns = ['Date', 'Value']
        df['Date'] = pd.to_datetime(df['Date'])
        df.dropna(inplace=True)

        # Reindex to include all calendar days
        all_days = pd.date_range(start=df['Date'].min(), end=df['Date'].max(), freq='D')
        df = df.set_index('Date').reindex(all_days)
        df['Value'] = df['Value'].ffill()  # Forward fill to maintain data continuity

        # Detect where actual changes occur
        changes = df['Value'].diff().fillna(0) != 0
        change_dates = changes[changes].index

        # Include the first row explicitly as a change point
        if change_dates[0] != df.index.min():
            change_dates = change_dates.insert(0, df.index.min())

        # Perform linear interpolation between change points
        for start, end in zip(change_dates, change_dates[1:]):
            if end > start:  # Ensure there is more than one day to interpolate
                df.loc[start:end, 'Value'] = np.linspace(df.at[start, 'Value'], df.at[end, 'Value'], (end - start).days + 1)
        # Filter to only include business days
        business_days = all_days.to_series().dt.dayofweek < 5
        df = df[business_days]

        # Calculate percent change on business days if requested
        if percent_change:
            df['Value'] = df['Value'].pct_change() * 100
            df.dropna(inplace=True)  # Removes NaNs created by pct_change

        return df.reset_index().rename(columns={'index': 'Date'})
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None
    
def main():
    # List of your datasets on Kaggle
    datasets = {
        'simonwildsmith/us-uncertainty-index': \
            ['United States uncertainty index', 0, 1, False],
        'simonwildsmith/10y-2y-tbill-constant-maturity': \
            ['T10Y2Y', 0, 1, False],
        'simonwildsmith/us-dollar-index': \
            ['US Dollar Index (DXY)', 0, 4, False],
        'simonwildsmith/gdp-growth-quarterly-from-preceding-period': \
            ['GDP growth', 0, 1, False],
        'simonwildsmith/us-unemployment-rate': \
            ['Unemployment Rate', 0, 1, False],
        'simonwildsmith/consumer-price-index-monthly-seasonally-adjusted': \
            ['Consumer Price Index', 0, 1, False],
        'simonwildsmith/federal-funds-effective-rate-1979-march-2024': \
            ['Effective Funds Rate DFF', 0, 1, False],
        'simonwildsmith/historical-gold-prices-march-2024': \
            ['Historical Gold Prices', 0, 1, False],
    }

    cleaned_data_dir = 'datasets/cleaned'
    if not os.path.exists(cleaned_data_dir):
        os.makedirs(cleaned_data_dir)

    for dataset_name, (download_name, date_col_index, value_col_index, percent_change) in datasets.items():
        print(f"Downloading {dataset_name}...")
        download_dataset(dataset_name)

        file_path = os.path.join('datasets', download_name + '.csv')
        
        print(f"Processing {file_path}...")
        df = load_and_clean_data(file_path, date_col_index, value_col_index, percent_change)
        
        if df is not None:
            clean_file_path = os.path.join('datasets', 'cleaned', download_name + '_cleaned.csv')
            df.to_csv(clean_file_path, index=False)
            print(f"Saved cleaned data to {clean_file_path}")

if __name__ == "__main__":
    main()
