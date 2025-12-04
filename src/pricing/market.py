import datetime as dt
from dataclasses import dataclass


@dataclass
class MarketData:
    """Class used to represent market data."""

    start_date: dt.date
    spot_price: float
    volatility: float
    interest_rate: float
    discount_rate: float
    dividend_ex_date: dt.date
    dividend_amount: float
