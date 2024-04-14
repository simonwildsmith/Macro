# eda.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(directory='datasets/cleaned'):
    """
    Load all cleaned data from the specified directory.
    Merge them on the 'Date' column.
    """
    files = os.listdir(directory)
    data_frames = []
    for file in files:
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(directory, file))
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            df.rename(columns={'Value': file.replace('_cleaned.csv', '')}, inplace=True)
            data_frames.append(df)
    
    # Merge all dataframes on 'Date' index
    merged_df = pd.concat(data_frames, axis=1)
    return merged_df

def plot_correlation_matrix(df):
    """
    Calculate and plot correlation matrix using seaborn, and save it to a CSV file.
    """
    correlation_matrix = df.corr()
    # Save the correlation matrix to a CSV file
    correlation_matrix.to_csv('datasets/correlation_matrix.csv')
    print("Correlation matrix saved to datasets/correlation_matrix.csv.")
    
    # Plotting the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.title('Correlation Matrix')
    plt.show()

def plot_pair_plots(df, vars):
    """
    Create pair plots for selected variables to visualize individual and pairwise relationships.
    """
    sns.pairplot(df[vars])
    plt.show()

def main():
    # Load the data
    print("Loading data...")
    df = load_data()
    
    # Plot correlation matrix
    print("Plotting correlation matrix...")
    plot_correlation_matrix(df)

    # Optionally, select a few variables for pair plots if the dataset is very large
    variables_to_plot = ['US Uncertainty Index', '10y-2y Tbill constant maturity', 'US Dollar Index', 'GDP growth', 'Unemployment Rate', 'Consumer Price Index', 'Effective Funds Rate DFF', 'Historical Gold Prices']
    print("Generating pair plots...")
    plot_pair_plots(df, variables_to_plot)

if __name__ == "__main__":
    main()
