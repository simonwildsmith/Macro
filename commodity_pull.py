import yfinance as yf
import pandas as pd
from datetime import datetime
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from db import Commodity, engine
from pandas.tseries.offsets import BDay
from tqdm import tqdm

# Database session setup
Session = sessionmaker(bind=engine)
session = Session()

# Commodity symbols as a dictionary
commodity_symbols = {
    'Gold': 'GC=F',
    'Copper': 'HG=F',
    'Crude Oil': 'CL=F',
    'Palladium': 'PA=F',
    'Platinum': 'PL=F',
}

existing_records = set(
    session.query(Commodity.date, Commodity.metal).all()
)

# Function to generate business days between two dates
def business_days(start_date, end_date):
    return pd.date_range(start=start_date, end=end_date, freq=BDay())

# Returns commodity data from Yahoo Finance
def fetch_commodity_data(ticker_symbol, start_date, end_date):
    commodity = yf.Ticker(ticker_symbol)
    data = commodity.history(start=start_date, end=end_date)
    
    # Normalize the date index to date-only format
    data.index = data.index.normalize().date
    
    # Filter for business days
    business_days = pd.date_range(start=start_date, end=end_date, freq='B')
    business_days = business_days.date
    data = data[data.index.isin(business_days)]
    
    return data

# Function to insert data into the database
def insert_to_db(metal, ticker_symbol, data):
    progress_bar = tqdm(total=len(data), desc=f"Inserting data for {metal}")
    commit_frequency = 1  # Set the desired commit frequency
    
    for index, row in data.iterrows():
        if row.isnull().values.any():
            print(f"Null values found for {ticker_symbol} on {index}")
            continue
        
        if (index, ticker_symbol) in existing_records:
            existing_commodity = session.query(Commodity).filter_by(date=index, metal=metal).first()
            existing_commodity.open = row['Open']
            existing_commodity.high = row['High']
            existing_commodity.low = row['Low']
            existing_commodity.close = row['Close']
            existing_commodity.volume = row['Volume']
        else:
            commodity = Commodity(
                date=index,
                metal=metal,
                open=row['Open'],
                high=row['High'],
                low=row['Low'],
                close=row['Close'],
                volume=row['Volume']
            )
            session.add(commodity)
        
        progress_bar.update(1)
        try:        
            if progress_bar.n % commit_frequency == 0:
                session.commit()
        except Exception as e:
            print(f"Error inserting data for {metal} on {index}: {e}")
            session.rollback()
    session.commit()
    progress_bar.close()
    

# Main loop to fetch and insert data
start_date = datetime.now() - pd.DateOffset(years=20)
end_date = datetime.now()
adjusted_start_date = business_days(start_date, start_date + BDay(5))[0].date()

for metal, symbol in commodity_symbols.items():
    try:
        commodity_data = fetch_commodity_data(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        insert_to_db(metal, symbol, commodity_data)
    except Exception as e:
        print(f"Error fetching data for {metal}: {e}")

session.close()
