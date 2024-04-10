import os
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

# Initialize Kaggle API
api = KaggleApi()
api.authenticate()

def download_dataset(dataset_name, path='datasets/'):
    """
    Downloads a dataset from Kaggle and saves it to the specified path.
    """
    api.dataset_download_files(dataset_name, path=path, unzip=True)

def load_and_clean_data(file_path):
    """
    Loads a CSV file, cleans the data, and returns a pandas DataFrame.
    """
    try:
        # Assuming the first column is 'Date' and the second is 'Value'
        df = pd.read_csv(file_path, parse_dates=['Date'], dayfirst=True)
        df.dropna(inplace=True)  # Remove missing values
        # Additional cleaning steps can be added here if necessary
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def main():
    # List of your datasets on Kaggle
    datasets = [
        'simonwildsmith/US Uncertainty index',
        'simonwildsmith/10Y2Y constant maturity',
        'simonwildsmith/US Dollar Index',
        'simonwildsmith/GDP Growth',
        'simonwildsmith/US Unemployment Rate',
        'simonwildsmith/Consumer Price Index',
        'simonwildsmith/Federal Funds Effective Rate',
        'simonwildsmith/Historical Gold Prices',
        # Add all your datasets here
    ]

    for dataset in datasets:
        print(f"Downloading {dataset}...")
        download_dataset(dataset)

        # Assuming file names are consistent with dataset names
        file_name = dataset.split('/')[-1] + '.csv'
        file_path = os.path.join('datasets', file_name)
        
        print(f"Processing {file_name}...")
        df = load_and_clean_data(file_path)
        
        # Save the cleaned dataframe
        if df is not None:
            clean_file_path = os.path.join('datasets/cleaned', file_name)
            df.to_csv(clean_file_path, index=False)
            print(f"Saved cleaned data to {clean_file_path}")

if __name__ == "__main__":
    main()
