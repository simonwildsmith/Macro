from sqlalchemy import create_engine, sessionmaker
from sqlalchemy.sql import func
from db import Equity, Equity_Stats, Base
from datetime import datetime
from tqdm import tqdm

# Database configuration
DATABASE_URI = 'postgresql+psycopg2://postgres:postgres@localhost/db'

# Database setup
engine = create_engine(DATABASE_URI)
Session = sessionmaker(bind=engine)
session = Session()

def calculate_true_range(high, low, previous_close):
    range1 = high - low
    range2 = abs(high - previous_close)
    range3 = abs(low - previous_close)
    true_range = max(range1, range2, range3)
    return true_range

def calculate_atr(equity):
    previous_close = equity.open  # Assuming the open price is the previous close for demonstration
    true_range = calculate_true_range(equity.high, equity.low, previous_close)
    
    # Normalize the true range by the stock's price to get the relative ATR
    relative_atr = (true_range / equity.close) * 100  # Convert to percentage
    return relative_atr

def process_data():
    # Query all the data from Equities table
    equities = session.query(Equity).all()

    # Data structure to store aggregated data
    stats = {}

    for equity in equities:
        key = (equity.date, equity.gics_sector, equity.gics_sub_industry)

        if key not in stats:
            stats[key] = {
                'change_sum': 0, 'change_count': 0, 'atr_sum': 0, 'atr_count': 0
            }

        change = ((equity.close - equity.open) / equity.open) * 100
        atr = calculate_atr(equity)

        stats[key]['change_sum'] += change
        stats[key]['change_count'] += 1
        stats[key]['atr_sum'] += atr
        stats[key]['atr_count'] += 1

        # Also aggregate data for the entire sector ('all' sub-industries)
        sector_key = (equity.date, equity.gics_sector, 'all')
        if sector_key not in stats:
            stats[sector_key] = {
                'change_sum': 0, 'change_count': 0, 'atr_sum': 0, 'atr_count': 0
            }

        stats[sector_key]['change_sum'] += change
        stats[sector_key]['change_count'] += 1
        stats[sector_key]['atr_sum'] += atr
        stats[sector_key]['atr_count'] += 1

    # Insert or update the data into Equity_Stats
    for key, data in stats.items():
        date, sector, sub_industry = key
        change_avg = data['change_sum'] / data['change_count']
        atr_avg = data['atr_sum'] / data['atr_count']

        existing_stat = session.query(Equity_Stats).filter_by(
            date=date, sector=sector, gics_sub_industry=sub_industry
        ).first()

        if existing_stat:
            # Update existing record
            existing_stat.change_day = change_avg
            existing_stat.atr = atr_avg
        else:
            # Insert new record
            new_stat = Equity_Stats(
                sector=sector,
                gics_sub_industry=sub_industry,
                date=date,
                change_day=change_avg,
                atr=atr_avg
            )
            session.add(new_stat)

    session.commit()

if __name__ == "__main__":
    process_data()
