from enum import Enum


class CalendarBaseConvention(Enum):
    _365 = 365
    _360 = 360
    _252 = 252
    _257 = 257
    _366 = 366


class BarrierType(Enum):
    knock_in = "Knock-in"
    knock_out = "Knock-out"


class BarrierDirection(Enum):
    up = "Up"
    down = "Down"
