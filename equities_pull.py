import json
import yfinance as yf
import pandas as pd
from datetime import datetime
from sqlalchemy.orm import sessionmaker
from db import Equity, engine
from pandas.tseries.offsets import BDay

# Database session setup
Session = sessionmaker(bind=engine)
session = Session()

# Read NASDAQ tickers and sort by market cap
with open('nasdaq_full_tickers.json', 'r') as file:
    nasdaq_tickers = [ticker for ticker in json.load(file) if ticker['country'] == 'United States']
    # Convert marketCap to float and sort
    for ticker in nasdaq_tickers:  
        # Removing commas and converting to float
        market_cap = ticker['marketCap']
        if market_cap == '':
            ticker['marketCap'] = 0.0
        else:
            ticker['marketCap'] = float(market_cap.replace(',', '').replace('$', ''))
    nasdaq_tickers.sort(key=lambda x: x['marketCap'], reverse=True)

# Function to generate business days between two dates
def business_days(start_date, end_date):
    return pd.date_range(start=start_date, end=end_date, freq=BDay())

def fetch_stock_data(ticker_symbol, start_date, end_date):
    stock = yf.Ticker(ticker_symbol)
    data = stock.history(start=start_date, end=end_date)

    # Normalize the date index to date-only format
    data.index = data.index.normalize().date
    
    # Filter for business days
    business_days = pd.date_range(start=start_date, end=end_date, freq='B')
    business_days = business_days.date  # Convert to date-only format
    data = data[data.index.isin(business_days)]

    return data

# Function to insert data into the database
def insert_to_db(ticker_symbol, data, sector, industry):
    for index, row in data.iterrows():
        if row.isnull().values.any():
            print(f"Null values found for {ticker_symbol} on {index}")
            continue
        equity = Equity(
            date=index,
            ticker=ticker_symbol,
            open=row['Open'],
            high=row['High'],
            low=row['Low'],
            close=row['Close'],
            volume=row['Volume'],
            gics_sector=sector,
            gics_sub_industry=industry
        )
        session.add(equity)
    session.commit()

# Main loop to fetch and insert data
start_date = datetime.now() - pd.DateOffset(years=30)
end_date = datetime.now()

ticker_index = 0

while ticker_index < len(nasdaq_tickers):
    ticker = nasdaq_tickers[ticker_index]
    symbol = ticker['symbol']
    sector = ticker['sector']
    industry = ticker['industry']

    print(sector)
    print(industry)

    try:
        stock_data = fetch_stock_data(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        insert_to_db(symbol, stock_data, sector, industry)
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")

    ticker_index += 1

session.close()
