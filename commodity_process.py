from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import func
from db import Commodity, Commodity_Stats, Base, engine
from datetime import datetime
from tqdm import tqdm

# Database configuration
DATABASE_URI = 'postgresql+psycopg2://postgres:postgres@localhost/db'

# Database setup
Session = sessionmaker(bind=engine)
session = Session()

existing_record = set(
    session.query(Commodity_Stats.date, Commodity_Stats.metal).all()
)

def calculate_true_range(high, low, previous_close):
    range1 = high - low
    range2 = abs(high - previous_close)
    range3 = abs(low - previous_close)
    true_range = max(range1, range2, range3)
    return true_range

def calculate_atr(commodity):
    previous_close = commodity.open  # Assuming the open price is the previous close for demonstration
    true_range = calculate_true_range(commodity.high, commodity.low, previous_close)
    relative_atr = (true_range / commodity.close) * 100  # Convert to percentage
    return relative_atr

def process_data():
    progress_bar = tqdm(total=session.query(Commodity).count(), desc="Processing data")
    commodities = session.query(Commodity).all()
    for commodity in commodities:
        change = ((commodity.close - commodity.open) / commodity.open) * 100
        atr = calculate_atr(commodity)

        if (commodity.date, commodity.metal) in existing_record:
            existing_stat = session.query(Commodity_Stats).filter_by(date=commodity.date, metal=commodity.metal).first()
            existing_stat.change_day = change
            existing_stat.atr = atr
        else:
            new_stat = Commodity_Stats(
                metal=commodity.metal,
                date=commodity.date,
                change_day=change,
                atr=atr
            )
            session.add(new_stat)
        progress_bar.update(1)
    session.commit()
    progress_bar.close()

if __name__ == "__main__":
    process_data()
