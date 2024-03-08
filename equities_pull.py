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
start_date = datetime.now() - pd.DateOffset(years=20)
end_date = datetime.now()

#Adjust the start_date to the next business day if it's a weekend or holiday
adjusted_start_date = business_days(start_date, start_date + BDay(5))[0].date()

ticker_index = 0
companies_with_start_date_data = 0
counter = 0

while ticker_index < 0:#len(nasdaq_tickers):
    ticker = nasdaq_tickers[ticker_index]
    symbol = ticker['symbol']
    sector = ticker['sector']
    industry = ticker['industry']

    try:
        stock_data = fetch_stock_data(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        if adjusted_start_date in stock_data.index:
            companies_with_start_date_data += 1
        #insert_to_db(symbol, stock_data, sector, industry)
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")

    ticker_index += 1
    counter += 1
    print(f"percentage complete: {counter/len(nasdaq_tickers) * 100:.2f}%")

print(f"Companies with start date data: {companies_with_start_date_data}")
print(f"Total companies: {len(nasdaq_tickers)}")


session.close()
