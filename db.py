from sqlalchemy import create_engine, Column, Integer, Float, String, Date
from sqlalchemy.orm import sessionmaker, declarative_base

# Database Configuration
DATABASE_URI = 'postgresql+psycopg2://postgres:postgres@localhost/db'

engine = create_engine(DATABASE_URI)
Session = sessionmaker(bind=engine)
Base = declarative_base()

class Equity(Base):
    __tablename__ = 'equities'

    id = Column(Integer, primary_key=True)
    date = Column(Date)
    ticker = Column(String)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Integer)
    gics_sector = Column(String)
    gics_sub_industry = Column(String)

class Equity_Stats(Base):
    __tablename__ = 'equity_stats'

    id = Column(Integer, primary_key=True)
    sector = Column(String)
    industry = Column(String)
    date = Column(Date)
    change_day = Column(Float)
    atr = Column(Float)

class Commodity(Base):
    __tablename__ = 'commodities'

    id = Column(Integer, primary_key=True)
    date = Column(Date)
    metal = Column(String)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Integer)

class Commodity_Stats(Base):
    __tablename__ = 'commodity_stats'

    id = Column(Integer, primary_key=True)
    metal = Column(String)
    date = Column(Date)
    change_day = Column(Float)
    atr = Column(Float)

class MacroeconomicData(Base):
    __tablename__ = 'macroeconomic_data'

    id = Column(Integer, primary_key=True)
    date = Column(Date)
    metric = Column(String)
    value = Column(Float)

    #unemployment_rate = Column(Float) #series_id: UNRATENSA
    #gdp_growth = Column(Float) #series_id: A191RP1Q027SBEA
    #real_gdp_growth = Column(Float) #series_id: A191RL1Q225SBEA
    #fed_funds_rate = Column(Float) #series_id: DFF
    #consumer_price_index = Column(Float) #series_id: CPIAUCSL
    #ten_year_minus_two_year_treasury = Column(Float) #series_id: T10Y2Y
    #m2_money_supply = Column(Float) #series_id: M2SL

# Create tables
if __name__ == '__main__':
    Base.metadata.create_all(engine)
