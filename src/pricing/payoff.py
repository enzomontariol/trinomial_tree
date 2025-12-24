from abc import ABC, abstractmethod
import numpy as np
from src.pricing.option import Option
from src.pricing.enums import BarrierType, BarrierDirection


class PayoffStrategy(ABC):
    """
    Abstract base class for payoff strategies.
    """

    @abstractmethod
    def calculate_intrinsic_value(
        self, spots: np.ndarray, option: Option
    ) -> np.ndarray:
        """Calculates the intrinsic value of the option."""
        pass

    @abstractmethod
    def apply_conditions(
        self, option_values: np.ndarray, spots: np.ndarray, option: Option
    ) -> np.ndarray:
        """Applies specific conditions (e.g., barriers) to the option values."""
        pass


class VanillaPayoff(PayoffStrategy):
    """
    Standard Vanilla Option Payoff (Call/Put).
    """

    def calculate_intrinsic_value(
        self, spots: np.ndarray, option: Option
    ) -> np.ndarray:
        if option.is_call:
            return np.maximum(spots - option.strike_price, 0.0)
        else:
            return np.maximum(option.strike_price - spots, 0.0)

    def apply_conditions(
        self, option_values: np.ndarray, spots: np.ndarray, option: Option
    ) -> np.ndarray:
        # No extra conditions for vanilla options
        return option_values


class BarrierPayoff(PayoffStrategy):
    """
    Barrier Option Payoff (Knock-Out).
    Knock-In is handled via decomposition (Vanilla - Knock-Out) in the pricer or a separate strategy.
    """

    def calculate_intrinsic_value(
        self, spots: np.ndarray, option: Option
    ) -> np.ndarray:
        # Base intrinsic value is the same as vanilla
        if option.is_call:
            return np.maximum(spots - option.strike_price, 0.0)
        else:
            return np.maximum(option.strike_price - spots, 0.0)

    def apply_conditions(
        self, option_values: np.ndarray, spots: np.ndarray, option: Option
    ) -> np.ndarray:
        if option.barrier is None:
            return option_values

        barrier = option.barrier

        # Determine breach condition
        if barrier.barrier_direction == BarrierDirection.up:
            breached = spots >= barrier.barrier_level
        elif barrier.barrier_direction == BarrierDirection.down:
            breached = spots <= barrier.barrier_level
        else:
            return option_values

        # Apply Knock-Out
        if barrier.barrier_type == BarrierType.knock_out:
            option_values[breached] = 0.0

        return option_values


class DigitalPayoff(PayoffStrategy):
    """
    Digital (Binary) Option Payoff (Cash-or-Nothing).
    Pays 1.0 unit of currency if the option ends In-The-Money.
    """

    def calculate_intrinsic_value(
        self, spots: np.ndarray, option: Option
    ) -> np.ndarray:
        if option.is_call:
            # Pays 1.0 if S > K
            return np.where(spots > option.strike_price, 1.0, 0.0)
        else:
            # Pays 1.0 if S < K
            return np.where(spots < option.strike_price, 1.0, 0.0)

    def apply_conditions(
        self, option_values: np.ndarray, spots: np.ndarray, option: Option
    ) -> np.ndarray:
        # Digital options can also have barriers, but for this basic strategy
        # we assume standard Digital.
        return option_values


class AssetOrNothingPayoff(PayoffStrategy):
    """
    Asset-or-Nothing Option Payoff.
    Pays the asset price S if the option ends In-The-Money.
    """

    def calculate_intrinsic_value(
        self, spots: np.ndarray, option: Option
    ) -> np.ndarray:
        if option.is_call:
            # Pays S if S > K
            return np.where(spots > option.strike_price, spots, 0.0)
        else:
            # Pays S if S < K
            return np.where(spots < option.strike_price, spots, 0.0)

    def apply_conditions(
        self, option_values: np.ndarray, spots: np.ndarray, option: Option
    ) -> np.ndarray:
        return option_values
