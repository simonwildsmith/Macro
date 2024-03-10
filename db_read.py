import sqlalchemy
from db import Equity, Commodity_Stats, Equity_Stats, MacroeconomicData

# Create a SQLAlchemy engine to connect to the database
engine = sqlalchemy.create_engine('postgresql+psycopg2://postgres:postgres@localhost/db')

# Create a SQLAlchemy session
Session = sqlalchemy.orm.sessionmaker(bind=engine)
session = Session()

# Query the Equity table for records where the open and close values are the same
results = session.query(Equity).filter(Equity.open == Equity.close).limit(50).all()

for result in results:
    print(result.date, result.ticker, result.open, result.close, result.volume)

# Close the session
session.close()
