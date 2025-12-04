import datetime as dt
from dataclasses import dataclass

from src.pricing.barrier import Barrier


@dataclass
class Option:
    """Class used to represent an option and its parameters."""

    maturity: dt.date
    strike_price: float
    barrier: Barrier | None = None
    is_american: bool = False
    is_call: bool = True
    pricing_date: dt.date = dt.date.today()
