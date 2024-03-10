import sqlalchemy
from db import Equity, Commodity_Stats

# Create a SQLAlchemy engine to connect to the database
engine = sqlalchemy.create_engine('postgresql+psycopg2://postgres:postgres@localhost/db')

# Create a SQLAlchemy session
Session = sqlalchemy.orm.sessionmaker(bind=engine)
session = Session()

# Query the first 50 entries in the Commodity_Stats table
results = session.query(Commodity_Stats).limit(50).all()

# Print each row
for row in results:
    print(row.date, row.metal, row.change_day, row.atr)

# Close the session
session.close()
