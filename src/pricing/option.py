import datetime as dt
from dataclasses import dataclass

from src.pricing.barrier import Barrier
from src.pricing.enums import CalendarBaseConvention, OptionPayoffType


@dataclass
class Option:
    """Class used to represent an option and its parameters."""

    maturity: dt.date
    strike_price: float
    barrier: Barrier | None = None
    is_american: bool = False
    is_call: bool = True
    pricing_date: dt.date = dt.date.today()
    calendar_base_convention: int = CalendarBaseConvention._365.value
    payoff_type: OptionPayoffType = OptionPayoffType.vanilla
