import datetime
from fredapi import Fred
from sqlalchemy.orm import sessionmaker
from db import engine, MacroeconomicData
import pandas as pd
from tqdm import tqdm

# FRED API Key (Replace 'your_api_key_here' with your actual API key)
API_KEY = 'edc826709575a6193fe374b8a8b2df77'
fred = Fred(api_key=API_KEY)

# Database Session
Session = sessionmaker(bind=engine)
session = Session()

# List of metrics to fetch
metrics = {
    'UNRATENSA': 'Unemployment Rate',
    'A191RP1Q027SBEA': 'GDP Growth',
    'A191RL1Q225SBEA': 'Real GDP Growth',
    'DFF': 'Fed Funds Rate',
    'CPIAUCSL': 'Consumer Price Index',
    'T10Y2Y': '10Y-2Y Treasury Spread',
    'M2SL': 'M2 Money Supply'
}

def fetch_and_insert(metric_id, metric_name):
    # Calculate 20 years ago
    twenty_years_ago = datetime.datetime.now() - datetime.timedelta(days=365 * 20)

    # Fetch data
    data = fred.get_series(metric_id, observation_start=twenty_years_ago)

    # Create a date range from the start date to today
    date_range = pd.date_range(start=twenty_years_ago, end=datetime.datetime.now())

    # Reindex the fetched data to the date range with forward fill
    data = data.reindex(date_range, method='ffill')

    # Fetch existing records for this metric within the date range
    existing_records = session.query(MacroeconomicData).filter(
        MacroeconomicData.metric == metric_name,
        MacroeconomicData.date.between(twenty_years_ago, datetime.datetime.now())
    ).all()

    # Convert existing records to a dictionary for easy lookup
    existing_data = {record.date: record for record in existing_records}

    # Prepare lists for new and updated records
    new_records = []
    updated_records = []

    # Check each date in the data
    for date, value in data.items():
        progress_bar = tqdm(total=len(data), desc=f"Inserting data for {metric_name}")
        if pd.notna(value):
            if date in existing_data:
                # Update if value is different
                if existing_data[date].value != value:
                    existing_data[date].value = value
                    updated_records.append(existing_data[date])
            else:
                # Add new record
                new_record = MacroeconomicData(
                    date=date.to_pydatetime(),
                    metric=metric_name,
                    value=value
                )
                new_records.append(new_record)
        progress_bar.update(1)
    progress_bar.close()

    # Bulk insert new records and bulk update changed records
    if new_records:
        session.bulk_save_objects(new_records)
    if updated_records:
        session.bulk_save_objects(updated_records)

    # Commit the session
    session.commit()

# Fetch and insert data for each metric
for series_id, name in metrics.items():
    fetch_and_insert(series_id, name)

# Commit the session
session.commit()

# Close the session
session.close()
