import numpy as np

from src.pricing.pricer import Pricer
from src.pricing.market import MarketData
from src.pricing.option import Option
from src.pricing.config import PricingConfig
from src.pricing.enums import BarrierType, BarrierDirection
from src.pricing.barrier import Barrier


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
        self.delta_t = self._calculate_delta_t()
        self.capitalization_factor = self._calculate_capitalization_factor()
        self.discount_factor = self._calculate_discount_factor()
        self.div_position = self._calculate_div_position()
        self.alpha = self._calculate_alpha()

    def get_time_to_maturity(self) -> float:
        """Returns the time to maturity expressed in number of years.

        Returns:
            float: time to maturity in number of years
        """
        return (
            self.option.maturity - self.option.pricing_date
        ).days / self.option.calendar_base_convention

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

    def _compute_forward_prices(self, spot_price: np.ndarray) -> np.ndarray:
        """Function used to compute the forward price at each node of the tree.

        Args:
            spot_price (np.ndarray): The spot prices at a given layer of the tree.

        Returns:
            np.ndarray: The forward prices at each node of the tree.
        """
        return spot_price * self.capitalization_factor

    def _build_spots_matrix(self) -> np.ndarray:
        """Function used to build the layers of the tree-like structure.

        We use the Escrowed Dividend method:
        1. At each step, we strip the PV of future dividends from the spot price to get a 'clean' spot price.
        2. We evolve this 'clean' spot price using the standard recombining tree logic.
        3. We add back the PV of future dividends to the evolved nodes to get the actual market spot prices.

        Returns:
            np.ndarray: The matrix of spot prices at each node.
        """
        width = 2 * self.num_steps + 1
        center = self.num_steps

        spots = np.empty(
            (self.num_steps + 1, width), dtype=np.float64
        )  # pre-allocate memory for spots matrix
        spots.fill(np.nan)  # fill with NaN values

        spots[0, center] = (
            self.market_data.spot_price
        )  # set initial spot price at the root of the tree

        for step in range(1, self.num_steps + 1):
            previous_layer = spots[step - 1]
            current_layer = spots[step]

            prev_div_pv = self._calculate_dividend_pv(step - 1)
            curr_div_pv = self._calculate_dividend_pv(step)

            previous_slice = slice(center - (step - 1), center + (step))
            current_slice = slice(center - step, center + step + 1)

            previous_act = previous_layer[previous_slice]
            current_act = current_layer[current_slice]

            previous_act_clean = previous_act - prev_div_pv

            current_act[0] = (
                previous_act_clean[0] * self.capitalization_factor * self.alpha
            ) + curr_div_pv
            current_act[-1] = (
                previous_act_clean[-1] * self.capitalization_factor / self.alpha
            ) + curr_div_pv
            current_act[1:-1] = (
                self._compute_forward_prices(previous_act_clean) + curr_div_pv
            )

        return spots

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

    def _build_tree(self) -> None:
        """Function used to build the tree-like structure"""
        return None

    def _calculate_intrinsic_value(self, spots: np.ndarray) -> np.ndarray:
        """Calculates the intrinsic value of the option for a given set of spot prices.

        Args:
            spots (np.ndarray): Array of spot prices.

        Returns:
            np.ndarray: Array of intrinsic values (max(S-K, 0) or max(K-S, 0)).
        """
        if self.option.is_call:
            return np.maximum(spots - self.option.strike_price, 0.0)
        else:
            return np.maximum(self.option.strike_price - spots, 0.0)

    def _apply_barrier_conditions(
        self, option_values: np.ndarray, spots: np.ndarray, step: int, center: int
    ) -> np.ndarray:
        """Applies barrier conditions to the option values at a given step.

        Args:
            option_values (np.ndarray): The current option values at this step.
            spots (np.ndarray): The spot prices at this step.
            step (int): The current time step index.
            center (int): The center index of the arrays.

        Returns:
            np.ndarray: The option values adjusted for barrier conditions.
        """
        if self.option.barrier is None:
            return option_values

        barrier = self.option.barrier
        current_slice = slice(center - step, center + step + 1)
        current_spots = spots[step, current_slice]
        current_values = option_values[step, current_slice]

        if barrier.barrier_direction == BarrierDirection.up:
            breached = current_spots >= barrier.barrier_level
        elif barrier.barrier_direction == BarrierDirection.down:
            breached = current_spots <= barrier.barrier_level
        else:
            return option_values

        if barrier.barrier_type == BarrierType.knock_out:
            current_values[breached] = 0.0

        option_values[step, current_slice] = current_values
        return option_values

    def price(self) -> float:
        """
        Orchestrates the pricing process using iterative backward induction.

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

        spots = self._build_spots_matrix()
        probs = self._build_probabilities_matrix()
        p_u, p_m, p_d = probs[0], probs[1], probs[2]

        option_values = np.full_like(spots, np.nan)
        center = self.num_steps

        option_values[-1, :] = self._calculate_intrinsic_value(spots[-1, :])
        self._apply_barrier_conditions(option_values, spots, self.num_steps, center)

        for step in range(self.num_steps - 1, -1, -1):
            current_slice = slice(center - step, center + step + 1)
            valid_next_indices = slice(center - step - 1, center + step + 2)
            V_next = option_values[step + 1, valid_next_indices]

            V_up = V_next[:-2]
            V_mid = V_next[1:-1]
            V_down = V_next[2:]

            continuation_value = self.discount_factor * (
                p_u * V_up + p_m * V_mid + p_d * V_down
            )

            if self.option.is_american:
                current_spots = spots[step, current_slice]
                intrinsic_value = self._calculate_intrinsic_value(current_spots)
                temp_values = np.maximum(continuation_value, intrinsic_value)
            else:
                temp_values = continuation_value

            option_values[step, current_slice] = temp_values
            self._apply_barrier_conditions(option_values, spots, step, center)

        return option_values[0, center]
