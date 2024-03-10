from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import func
from db import Equity, Equity_Stats, Base, engine
from datetime import datetime
from tqdm import tqdm

# Database setup
Session = sessionmaker(bind=engine)
session = Session()

existing_record = set(
    session.query(Equity_Stats.date, Equity_Stats.sector, Equity_Stats.industry).all()
)

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
    print("entered process_data")

    # Set the batch size
    batch_size = 10000  # Adjust this number based on your system's capability
    total_records = session.query(Equity).count()
    print(f"Total records to process: {total_records}")

    stats = {}
    progress_bar = tqdm(total=total_records, desc="Processing data")

    # Process in batches
    for offset in range(0, total_records, batch_size):
        equities_batch = session.query(Equity).limit(batch_size).offset(offset).all()

        for equity in equities_batch:

            if equity.open == 0:
                continue

            key = (equity.date, equity.gics_sector, equity.gics_sub_industry)

            if key not in stats:
                stats[key] = {'change_weighted_sum': 0, 'volume_sum': 0, 
                              'atr_weighted_sum': 0, 'atr_volume_sum': 0}

            change = ((equity.close - equity.open) / equity.open) * 100
            atr = calculate_atr(equity)

            # Update sums for weighted average calculations
            stats[key]['change_weighted_sum'] += change * equity.volume
            stats[key]['volume_sum'] += equity.volume

            stats[key]['atr_weighted_sum'] += atr * equity.volume
            stats[key]['atr_volume_sum'] += equity.volume

            # Repeat for sector-wide data
            sector_key = (equity.date, equity.gics_sector, 'all')
            if sector_key not in stats:
                stats[sector_key] = {'change_weighted_sum': 0, 'volume_sum': 0, 
                                     'atr_weighted_sum': 0, 'atr_volume_sum': 0}

            stats[sector_key]['change_weighted_sum'] += change * equity.volume
            stats[sector_key]['volume_sum'] += equity.volume

            stats[sector_key]['atr_weighted_sum'] += atr * equity.volume
            stats[sector_key]['atr_volume_sum'] += equity.volume

            progress_bar.update(1)

        # Free up memory
        session.expunge_all()

    progress_bar.close()

    # Insert or update the data into Equity_Stats
    for key, data in stats.items():
        date, sector, sub_industry = key
        if data['volume_sum'] > 0:
            change_avg_weighted = data['change_weighted_sum'] / data['volume_sum']
            atr_avg_weighted = data['atr_weighted_sum'] / data['atr_volume_sum']
        else:
            continue  # Avoid division by zero

        if (date, sector, sub_industry) in existing_record:
            existing_stat = session.query(Equity_Stats).filter_by(date=date, sector=sector, gics_sub_industry=sub_industry).first()
            existing_stat.change_day = change_avg_weighted
            existing_stat.atr = atr_avg_weighted
        else:
            new_stat = Equity_Stats(
                sector=sector,
                industry=sub_industry,
                date=date,
                change_day=change_avg_weighted,
                atr=atr_avg_weighted
            )
            session.add(new_stat)

    session.commit()

if __name__ == "__main__":
    process_data()
