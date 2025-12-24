import numpy as np
import pandas as pd

from src.pricing.pricer import Pricer
from src.pricing.market import MarketData
from src.pricing.option import Option
from src.pricing.config import PricingConfig
from src.pricing.enums import BarrierType, BarrierDirection, OptionPayoffType
from src.pricing.barrier import Barrier
from src.pricing.payoff import (
    PayoffStrategy,
    VanillaPayoff,
    BarrierPayoff,
    DigitalPayoff,
    AssetOrNothingPayoff,
)


class InductiveTree(Pricer):
    """
    A Trinomial Tree option pricer based on the Kamrad-Ritchken model with an Escrowed Dividend adjustment.

    Methodology:
    1.  **Kamrad-Ritchken Model**: This model extends the standard binomial tree to a trinomial tree,
        allowing for three possible moves (up, horizontal, down) at each time step. This generally
        provides faster convergence and better stability than binomial trees.

    2.  **Escrowed Dividend Model**: To handle discrete dividends, we use the "Escrowed Dividend" approach.
        - The spot price process S_t is decomposed into a risk-free component (S*_t) and the present value
          of future dividends (PV(D)).
        - S_t = S*_t + PV(D)
        - The tree is built on the "clean" spot price S*_t, which follows standard log-normal dynamics
          unaffected by the jumps caused by discrete dividends.
        - At each node, the actual "dirty" spot price is reconstructed by adding back the PV of future dividends.

    3.  **Iterative Backward Induction**:
        - The tree structure is built iteratively layer by layer.
        - Pricing is performed using backward induction, leveraging vectorized numpy operations for efficiency.
        - This allows for flexible handling of path-dependent features and barriers.
    """

    def __init__(
        self,
        num_steps: int,
        market_data: MarketData,
        option: Option,
        config: PricingConfig = PricingConfig(),
    ) -> None:
        """Initialization of the class

        Args:
            num_steps (int): The number of steps in our model
            market_data (MarketData): Class used to represen market data.
            option (Option): Class used to represent an option and its parameters.
            config (PricingConfig, optional): Configuration object. Defaults to PricingConfig().
        """
        self.num_steps = num_steps
        self.market_data = market_data
        self.option = option
        self.config = config

        self._validate_inputs()

        self.delta_t = self._calculate_delta_t()
        self.capitalization_factor = self._calculate_capitalization_factor()
        self.discount_factor = self._calculate_discount_factor()
        self.div_position = self._calculate_div_position()
        self.alpha = self._calculate_alpha()

        self._validate_probabilities()

        self.payoff_strategy = self._get_payoff_strategy()

    def _validate_inputs(self) -> None:
        """Validates the input parameters for the tree."""
        if self.num_steps <= 10:
            raise ValueError("Number of steps must be greater than 10 for accuracy.")
        if self.market_data.volatility <= 0:
            raise ValueError("Volatility must be positive.")
        if self.market_data.spot_price <= 0:
            raise ValueError("Spot price must be positive.")
        if self.option.strike_price <= 0:
            raise ValueError("Strike price must be positive.")
        if self.option.maturity <= self.option.pricing_date:
            raise ValueError("Maturity date must be after pricing date.")

    def _validate_probabilities(self) -> None:
        """Validates that the transition probabilities are within [0, 1]."""
        probs = self._build_probabilities_matrix()
        if np.any(probs < 0) or np.any(probs > 1):
            raise ValueError(
                f"Transition probabilities out of bounds: {probs}. "
                "Try increasing num_steps or adjusting volatility."
            )
        if not np.isclose(np.sum(probs), 1.0):
            raise ValueError(
                f"Transition probabilities do not sum to 1: {np.sum(probs)}"
            )

    def get_time_to_maturity(self) -> float:
        """Returns the time to maturity expressed in number of years.

        Returns:
            float: time to maturity in number of years
        """
        # Use 365.25 for ACT/365.25 or similar if needed, but here we stick to the convention
        # defined in the option object, usually 360 or 365.
        # We use (date - date).days which is ACT/Convention.
        days_diff = (self.option.maturity - self.option.pricing_date).days
        return days_diff / self.option.calendar_base_convention

    def _calculate_delta_t(self) -> float:
        """Calculates the reference time interval that will be used in our model.

        Returns:
            float: the time interval delta_t
        """
        return self.get_time_to_maturity() / self.num_steps

    def _calculate_capitalization_factor(self) -> float:
        """Calculates the capitalization factor that we will use later

        Returns:
            float: a capitalization factor to apply at each dt.
        """
        return np.exp(self.market_data.interest_rate * self.delta_t)

    def _calculate_discount_factor(self) -> float:
        """Calculates the discount factor that we will use later

        Returns:
            float: a discount factor to apply at each dt.
        """
        return np.exp(-self.market_data.discount_rate * self.delta_t)

    def _calculate_div_position(self) -> float:
        """Allows us to calculate the position of the dividend in the Tree

        Returns:
            float: returns the ex-date position of the div, expressed in number of steps in the Tree.
        """
        nb_days_detachment = (
            self.market_data.dividend_ex_date - self.option.pricing_date
        ).days
        div_position = (
            nb_days_detachment / self.option.calendar_base_convention / self.delta_t
        )

        return div_position

    def _calculate_alpha(self) -> float:
        """Function allowing us to calculate alpha, which we will use in the Tree.

        Returns:
            float: Returns the alpha coefficient
        """
        alpha = np.exp(
            self.market_data.volatility
            * np.sqrt(self.config.alpha_parameter)
            * np.sqrt(self.delta_t)
        )
        return alpha

    def _get_payoff_strategy(self) -> PayoffStrategy:
        """Factory method to select the correct payoff strategy."""
        if self.option.barrier is not None:
            return BarrierPayoff()

        if self.option.payoff_type == OptionPayoffType.digital:
            return DigitalPayoff()
        elif self.option.payoff_type == OptionPayoffType.asset_or_nothing:
            return AssetOrNothingPayoff()

        return VanillaPayoff()

    def _calculate_dividend_pv(self, step: int) -> float:
        """Calculates the present value of the future dividend at a given step.

        This method is part of the Escrowed Dividend model. We assume the spot price
        process S_t can be decomposed into a risk-free component (S*_t) and the
        present value of future dividends (PV(D)).
        S_t = S*_t + PV(D)

        Args:
            step (int): The current step in the tree.

        Returns:
            float: The present value of the future dividend.
        """
        # If step is before the ex-date, we include the PV of the dividend.
        if step < self.div_position:
            time_to_ex = (self.div_position - step) * self.delta_t
            return self.market_data.dividend_amount * np.exp(
                -self.market_data.discount_rate * time_to_ex
            )
        return 0.0

    def _get_spots_at_step(self, step: int) -> np.ndarray:
        """Calculates the spot prices at a given step using the analytical formula.

        S_{i, j} = (S_0 - PV(D_0)) * M^i * alpha^(i-j) + PV(D_i)
        where j is the array index from 0 to 2*i.

        Args:
            step (int): The step number.

        Returns:
            np.ndarray: The array of spot prices at the given step.
        """
        S_clean_0 = self.market_data.spot_price - self._calculate_dividend_pv(0)

        # Indices j go from 0 to 2*step
        j = np.arange(2 * step + 1)

        # Powers of alpha: i - j
        powers = step - j

        # Calculate clean spots
        growth_factor = self.capitalization_factor**step
        spots_clean = S_clean_0 * growth_factor * (self.alpha**powers)

        # Add PV of dividends at current step
        spots_dirty = spots_clean + self._calculate_dividend_pv(step)

        return spots_dirty

    def _build_probabilities_matrix(self) -> np.ndarray:
        """Calculates the transition probabilities for the trinomial tree.

        The probabilities are calculated to match the first two moments of the log-normal distribution,
        accounting for the fact that the middle branch includes the risk-free drift (exp(r*dt)).

        Returns:
            np.ndarray: An array containing [p_up, p_mid, p_down].
        """
        sigma = self.market_data.volatility
        dt = self.delta_t
        lam = np.sqrt(self.config.alpha_parameter)

        # Epsilon is the drift correction term derived from moment matching
        # The middle node moves by r*dt in log space, so we adjust for the remaining drift
        epsilon = (np.sqrt(dt) / (2 * sigma * lam)) * (-(sigma**2 / 2))

        base_prob = 1.0 / (2 * self.config.alpha_parameter)

        p_u = base_prob + epsilon
        p_d = base_prob - epsilon
        p_m = 1.0 - p_u - p_d

        return np.array([p_u, p_m, p_d])

    def price(self) -> float:
        """
        Orchestrates the pricing process using iterative backward induction.
        Optimized for O(N) memory usage.

        Returns:
            float: The estimated price of the option.
        """
        if (
            self.option.barrier is not None
            and self.option.barrier.barrier_type == BarrierType.knock_in
        ):
            original_barrier = self.option.barrier

            self.option.barrier = None
            vanilla_price = self.price()

            ko_barrier = Barrier(
                original_barrier.barrier_level,
                BarrierType.knock_out,
                original_barrier.barrier_direction,
            )
            self.option.barrier = ko_barrier
            ko_price = self.price()

            self.option.barrier = original_barrier

            return vanilla_price - ko_price

        probs = self._build_probabilities_matrix()
        p_u, p_m, p_d = probs[0], probs[1], probs[2]

        # Initialize at maturity
        current_spots = self._get_spots_at_step(self.num_steps)
        current_values = self.payoff_strategy.calculate_intrinsic_value(
            current_spots, self.option
        )
        current_values = self.payoff_strategy.apply_conditions(
            current_values, current_spots, self.option
        )

        # Backward induction
        for step in range(self.num_steps - 1, -1, -1):
            next_values = current_values
            current_spots = self._get_spots_at_step(step)

            V_up = next_values[:-2]
            V_mid = next_values[1:-1]
            V_down = next_values[2:]

            continuation_value = self.discount_factor * (
                p_u * V_up + p_m * V_mid + p_d * V_down
            )

            if self.option.is_american:
                intrinsic_value = self.payoff_strategy.calculate_intrinsic_value(
                    current_spots, self.option
                )
                current_values = np.maximum(continuation_value, intrinsic_value)
            else:
                current_values = continuation_value

            current_values = self.payoff_strategy.apply_conditions(
                current_values, current_spots, self.option
            )

        return current_values[0]

    def get_terminal_distribution(self) -> pd.DataFrame:
        """
        Calculates the probability distribution of spot prices at maturity.
        Performs a forward induction to compute probabilities.
        """
        # 1. Build Transition Probabilities
        probs = self._build_probabilities_matrix()
        p_u, p_m, p_d = probs[0], probs[1], probs[2]

        # 2. Initialize Probabilities at t=0
        # We start with probability 1.0 at the single root node
        current_probs = np.array([1.0])

        # 3. Forward Induction
        for step in range(self.num_steps):
            # The next layer has 2 more nodes than the current layer
            next_probs = np.zeros(len(current_probs) + 2)

            # Vectorized probability propagation
            # Up moves: index i -> i
            next_probs[:-2] += current_probs * p_u
            # Mid moves: index i -> i+1
            next_probs[1:-1] += current_probs * p_m
            # Down moves: index i -> i+2
            next_probs[2:] += current_probs * p_d

            current_probs = next_probs

        # 4. Get Spot Prices at Maturity
        spots = self._get_spots_at_step(self.num_steps)

        # 5. Return DataFrame
        return pd.DataFrame({"Spot": spots, "Probability": current_probs})

    def get_exercise_boundary(self) -> pd.DataFrame:
        """
        Extracts the early exercise boundary for American options.
        Returns the spot price threshold where exercise becomes optimal at each step.
        """
        if not self.option.is_american:
            return pd.DataFrame()

        probs = self._build_probabilities_matrix()
        p_u, p_m, p_d = probs[0], probs[1], probs[2]

        # Initialize at maturity
        current_spots = self._get_spots_at_step(self.num_steps)
        current_values = self.payoff_strategy.calculate_intrinsic_value(
            current_spots, self.option
        )
        current_values = self.payoff_strategy.apply_conditions(
            current_values, current_spots, self.option
        )

        boundary_data = []

        # Backward induction
        for step in range(self.num_steps - 1, -1, -1):
            next_values = current_values
            current_spots = self._get_spots_at_step(step)

            V_up = next_values[:-2]
            V_mid = next_values[1:-1]
            V_down = next_values[2:]

            continuation_value = self.discount_factor * (
                p_u * V_up + p_m * V_mid + p_d * V_down
            )

            intrinsic_value = self.payoff_strategy.calculate_intrinsic_value(
                current_spots, self.option
            )

            # Determine Exercise Decision
            # Exercise if Intrinsic > Continuation + epsilon
            exercise_mask = intrinsic_value > continuation_value + 1e-9

            # Find the boundary
            boundary_spot = None
            if np.any(exercise_mask):
                exercised_spots = current_spots[exercise_mask]
                if not self.option.is_call:  # Put
                    boundary_spot = np.max(exercised_spots)
                else:  # Call (usually dividend related)
                    boundary_spot = np.min(exercised_spots)

            if boundary_spot is not None:
                # Calculate time in years
                t = step * self.delta_t
                boundary_data.append(
                    {"Step": step, "Time": t, "Boundary_Spot": boundary_spot}
                )

            # Update values for next iteration
            current_values = np.maximum(continuation_value, intrinsic_value)
            current_values = self.payoff_strategy.apply_conditions(
                current_values, current_spots, self.option
            )

        return pd.DataFrame(boundary_data).sort_values("Step")
