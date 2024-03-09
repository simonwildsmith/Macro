import sqlalchemy
from db import Equity

# Create a SQLAlchemy engine to connect to the database
engine = sqlalchemy.create_engine('postgresql+psycopg2://postgres:postgres@localhost/db')

# Create a SQLAlchemy session
Session = sqlalchemy.orm.sessionmaker(bind=engine)
session = Session()

# Query the Equities table and count the number of rows
row_count = session.query(Equity).count()

# Print the number of rows
print("Number of rows:", row_count)

# Close the session
session.close()
