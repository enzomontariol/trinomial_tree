from dataclasses import dataclass

from src.pricing.enums import BarrierType, BarrierDirection


@dataclass
class Barrier:
    """Class used to represent a barrier for a considered option"""

    def __init__(
        self,
        barrier_level: float,
        barrier_type: BarrierType | None,
        barrier_direction: BarrierDirection | None,
    ) -> None:
        self.barrier_level = barrier_level
        self.barrier_type = barrier_type
        self.barrier_direction = barrier_direction
