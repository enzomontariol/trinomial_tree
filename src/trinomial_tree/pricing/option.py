import datetime as dt
from dataclasses import dataclass, field

from .barrier import Barrier
from .enums import CalendarBaseConvention, OptionPayoffType


@dataclass
class Option:
    """Class used to represent an option and its parameters."""

    maturity: dt.date
    strike_price: float
    barrier: Barrier | None = None
    is_american: bool = False
    is_call: bool = True
    # default_factory, not dt.date.today(): a bare call is evaluated once at
    # import time, so every Option would share the date the module was loaded.
    pricing_date: dt.date = field(default_factory=dt.date.today)
    calendar_base_convention: int = CalendarBaseConvention._365.value
    payoff_type: OptionPayoffType = OptionPayoffType.vanilla

