import sqlalchemy
from db import Equity, Commodity_Stats, Equity_Stats, MacroeconomicData

# Create a SQLAlchemy engine to connect to the database
engine = sqlalchemy.create_engine('postgresql+psycopg2://postgres:postgres@localhost/db')

# Create a SQLAlchemy session
Session = sqlalchemy.orm.sessionmaker(bind=engine)
session = Session()

# Query the Unemployment Rate for each day between September 1 and October 1, 2004
results = session.query(MacroeconomicData).filter(MacroeconomicData.date >= '2004-08-01', MacroeconomicData.date <= '2004-10-01', MacroeconomicData.metric == 'Unemployment Rate').all()

for result in results:
    print(result.date, result.metric, result.value)


# Close the session
session.close()
