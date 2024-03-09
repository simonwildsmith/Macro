import json
import yfinance as yf
import pandas as pd
from datetime import datetime
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from db import Commodity
from pandas.tseries.offsets import BDay
from tqdm import tqdm

# Database session setup
DATABASE_URI = 'postgresql+psycopg2://postgres:postgres@localhost/db'

engine = create_engine(DATABASE_URI)
Session = sessionmaker(bind=engine)
session = Session()

# Commodity symbols
commodity_symbols = ['GC=F', 'HG=F', 'CL=F', 'NI=F', 'PA=F', 'PL=F', 'IRON=F']  # Replace with actual commodity symbols

def business_days(start_date, end_date):
    return pd.date_range(start=start_date, end=end_date, freq=BDay())

def fetch_commodity_data(ticker_symbol, start_date, end_date):
    commodity = yf.Ticker(ticker_symbol)
    data = commodity.history(start=start_date, end=end_date)
    data.index = data.index.normalize().date
    business_days = pd.date_range(start=start_date, end=end_date, freq='B')
    business_days = business_days.date
    data = data[data.index.isin(business_days)]
    return data

def insert_to_db(ticker_symbol, data):
    progress_bar = tqdm(total=len(data), desc=f"Inserting data for {ticker_symbol}")
    for index, row in data.iterrows():
        if row.isnull().values.any():
            print(f"Null values found for {ticker_symbol} on {index}")
            continue
        existing_commodity = session.query(Commodity).filter_by(date=index, ticker=ticker_symbol).first()
        if existing_commodity:
            existing_commodity.open = row['Open']
            existing_commodity.high = row['High']
            existing_commodity.low = row['Low']
            existing_commodity.close = row['Close']
            existing_commodity.volume = row['Volume']
        else:
            commodity = Commodity(
                date=index,
                ticker=ticker_symbol,
                open=row['Open'],
                high=row['High'],
                low=row['Low'],
                close=row['Close'],
                volume=row['Volume']
            )
            session.add(commodity)
        progress_bar.update(1)
    session.commit()
    progress_bar.close()
    

# Main loop to fetch and insert data
start_date = datetime.now() - pd.DateOffset(years=20)
end_date = datetime.now()
adjusted_start_date = business_days(start_date, start_date + BDay(5))[0].date()

for symbol in commodity_symbols:
    try:
        commodity_data = fetch_commodity_data(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        insert_to_db(symbol, commodity_data)
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")

session.close()
