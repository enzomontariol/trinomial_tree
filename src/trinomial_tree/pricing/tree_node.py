from __future__ import annotations

import copy
import numpy as np

from .market import MarketData
from .option import Option
from .enums import BarrierType, BarrierDirection
from .pricer import Pricer
from .config import PricingConfig


class Tree(Pricer):
    def __init__(
        self,
        num_steps: int,
        market_data: MarketData,
        option: Option,
        config: PricingConfig = PricingConfig(),  # if none is provided, use defaults
    ) -> None:
        """Initialization of the class

        Args:
            num_steps (float): the number of steps in our model
            market_data (MarketData): Class used to represent market data.
            option (Option): Class used to represent an option and its parameters.
            config (PricingConfig, optional): Configuration object.

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
        self.root: Node | None = None
        self.option_price: float | None = None

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

    def _build_tree(self) -> None:
        """Procedure allowing us to build our Tree"""

        def _create_next_block_up(current_center: Node, next_node: Node) -> None:
            """Procedure allowing us to build a complete block upwards from a reference Node and a future Node

            Args:
                current_center (Node): our reference Node
                next_node (Node): the Node around which we will create the block
            """
            temp_center = current_center
            temp_future_center = next_node

            # We iterate starting from the trunk and moving towards the upper extremity of a column in order to create Nodes on the next column
            while temp_center.up is not None:
                temp_center = temp_center.up
                if temp_future_center is None:
                    raise ValueError(
                        "Tree structure inconsistency: future node is None"
                    )
                temp_center._create_next_block(temp_future_center)
                temp_future_center = temp_future_center.up

        def _create_next_block_down(current_center: Node, next_node: Node) -> None:
            """Procedure allowing us to build a complete block downwards from a reference Node and a future Node

            Args:
                current_center (Node): our reference Node
                next_node (Node): the Node around which we will create the block
            """
            temp_center = current_center
            temp_future_center = next_node

            # We iterate starting from the trunk and moving towards the lower extremity of a column in order to create Nodes on the next column
            while temp_center.down is not None:
                temp_center = temp_center.down
                if temp_future_center is None:
                    raise ValueError(
                        "Tree structure inconsistency: future node is None"
                    )
                temp_center._create_next_block(temp_future_center)
                temp_future_center = temp_future_center.down

        def _create_new_col(self, current_center: Node) -> Node:
            """Procedure allowing us to fully create a column of our Tree.

            Args:
                current_center (Node): the Node on the current trunk, which we take as reference and from which we will create the next column.

            Returns:
                Node: we return the future Node on the center in order to iterate this function on it
            """
            next_node = Node(
                current_center._calculate_forward(),
                self,
                current_center.tree_position + 1,
            )

            current_center._create_next_block(next_node)
            _create_next_block_up(current_center, next_node)
            _create_next_block_down(current_center, next_node)

            return next_node

        # We create the root of our Tree here, not being able to do it at the __init__ level to avoid recursive import
        self.root = Node(
            spot_price=self.market_data.spot_price, tree=self, tree_position=0
        )

        # Our first reference is the root
        current_center = self.root

        # We create the first block here. We will then iterate over as many steps as necessary to create the following columns.
        for step in range(self.num_steps):
            current_center = _create_new_col(self, current_center)

    def price(self) -> float:
        """Function that will allow us to build the Tree then value it to finally give the value to the "option_price" attribute.

        Returns:
            float: The calculated option price.
        """
        if (
            self.option.barrier is not None
            and self.option.barrier.barrier_type == BarrierType.knock_in
        ):
            # Use Put-Call Parity for Barrier Options: In + Out = Vanilla
            # Price = Vanilla - Out

            # 1. Price Vanilla
            vanilla_option = copy.deepcopy(self.option)
            vanilla_option.barrier = None
            vanilla_tree = Tree(
                self.num_steps, self.market_data, vanilla_option, self.config
            )
            vanilla_price = vanilla_tree.price()

            # 2. Price Knock-Out
            out_option = copy.deepcopy(self.option)
            if out_option.barrier is None:
                raise ValueError("Barrier should not be None for Knock-In option")
            out_option.barrier.barrier_type = BarrierType.knock_out
            out_tree = Tree(self.num_steps, self.market_data, out_option, self.config)
            out_price = out_tree.price()

            self.option_price = vanilla_price - out_price
            return self.option_price

        self._build_tree()

        if self.root is None:
            raise ValueError("Tree root not initialized after build.")

        self.root._calculate_intrinsic_value()
        self.option_price = self.root.intrinsic_value

        if self.option_price is None:
            raise ValueError("Option price calculation failed.")

        return self.option_price


class Node:
    def __init__(self, spot_price: float, tree: Tree, tree_position: int) -> None:
        """Initialization of the class

        Args:
            spot_price (float): the underlying price of this Node
            tree (Tree): the Tree to which our Node is attached
            tree_position (int): describes the position of the Node in the Tree on the horizontal axis
        """
        self.epsilon = tree.config.epsilon

        self.spot_price = spot_price
        self.tree = tree
        self.tree_position = tree_position

        self.down: Node | None = None
        self.up: Node | None = None
        self.previous_center: Node | None = None
        self.future_down: Node | None = None
        self.future_center: Node | None = None
        self.future_up: Node | None = None
        self.p_down: float | None = None
        self.p_mid: float | None = None
        self.p_up: float | None = None
        self.cumulative_p: float = 1.0 if self.tree_position == 0 else 0.0

        self.intrinsic_value: float | None = None

    def _calculate_forward(self) -> float:
        """Allows calculating the forward price value on the next dt

        Returns:
            float : forward price
        """
        if (
            self.tree_position < self.tree.div_position
            and self.tree_position + 1 > self.tree.div_position
        ):
            div = self.tree.market_data.dividend_amount
        else:
            div = 0

        return self.spot_price * self.tree.capitalization_factor - div

    def _calculate_variance(self) -> float:
        """Allows us to calculate the variance

        Returns:
            float: variance
        """
        return (
            (self.spot_price**2)
            * np.exp(2 * self.tree.market_data.interest_rate * self.tree.delta_t)
            * (np.exp((self.tree.market_data.volatility**2) * self.tree.delta_t) - 1)
        )

    def _calculate_proba(self) -> None:
        """Allows us to calculate the up, center, down probabilities."""
        fw = self._calculate_forward()

        if self.future_center is None:
            raise ValueError("Future center node is not defined")

        p_down = (
            (self.future_center.spot_price ** (-2))
            * (self._calculate_variance() + fw**2)
            - 1
            - (self.tree.alpha + 1) * ((self.future_center.spot_price ** (-1)) * fw - 1)
        ) / ((1 - self.tree.alpha) * (self.tree.alpha ** (-2) - 1))

        p_up = (
            (1 / self.future_center.spot_price * fw - 1)
            - (1 / self.tree.alpha - 1) * p_down
        ) / (self.tree.alpha - 1)

        p_mid = 1 - p_up - p_down

        if not (p_down > 0 and p_up > 0 and p_mid > 0):
            raise ValueError("Negative probability")

        if not np.isclose(p_down + p_up + p_mid, self.tree.config.sum_proba, atol=1e-2):
            raise ValueError(
                f"The sum of probabilities must be equal to {self.tree.config.sum_proba}"
            )
        else:
            self.p_down = p_down
            self.p_up = p_up
            self.p_mid = p_mid

    def _test_close_node(self, forward: float) -> bool:
        """This function allows us to test if the Node is between an up Node price or a down Node price.

        Args:
            forward (float): the forward price of our Node that we will have calculated beforehand.

        Returns:
            bool: pass the test or not
        """
        condition_1 = self.spot_price * (1 + 1 / self.tree.alpha) / 2 <= forward
        condition_2 = forward <= self.spot_price * (1 + self.tree.alpha) / 2
        if condition_1 and condition_2:
            return True
        else:
            return False

    def next_down(self) -> Node:
        """Allows us to create the next down Node if it does not already exist.

        Returns:
            Node: the down Node
        """
        if self.down is None:
            self.down = Node(
                self.spot_price / self.tree.alpha, self.tree, self.tree_position
            )
            self.down.up = self
        return self.down

    def next_up(self) -> Node:
        """Allows us to create the next up Node if it does not already exist.

        Returns:
            Node: the up Node
        """
        if self.up is None:
            self.up = Node(
                self.spot_price * self.tree.alpha, self.tree, self.tree_position
            )
            self.up.down = self
        return self.up

    def find_center(self, next_node: Node) -> Node:
        """Function allowing us to find the next center Node.

        Args:
            next_node (Node): candidate Node

        Returns:
            Node: the center of our reference Node.
        """
        fw = self._calculate_forward()

        if next_node._test_close_node(fw):
            next_node = next_node

        elif fw > next_node.spot_price:
            while not next_node._test_close_node(fw):
                next_node = next_node.next_up()

        else:
            while not next_node._test_close_node(fw):
                next_node = next_node.next_down()

        return next_node

    def _create_next_block(self, next_node: Node) -> None:
        """Allows us to create a complete Node block.

        Args:
            next_node (Node): _description_
        """
        self.future_center = self.find_center(next_node=next_node)
        self._calculate_proba()

        if self.p_mid is None or self.p_up is None or self.p_down is None:
            raise ValueError("Probabilities not calculated")

        if self.future_center is None:
            raise ValueError("Future center not found")

        self.future_center.cumulative_p += self.cumulative_p * self.p_mid
        self.future_center.previous_center = self

        if self.tree.config.pruning:
            if self.up is None:
                if self.cumulative_p * self.p_up >= self.epsilon:
                    self.future_up = self.future_center.next_up()
                    self.future_up.cumulative_p += self.cumulative_p * self.p_up
                else:
                    # self.p_mid += self.p_up
                    self.p_up = 0
            elif self.up is not None:
                self.future_up = self.future_center.next_up()
                self.future_up.cumulative_p += self.cumulative_p * self.p_up

            if self.down is None:
                if self.cumulative_p * self.p_down >= self.epsilon:
                    self.future_down = self.future_center.next_down()
                    self.future_down.cumulative_p += self.cumulative_p * self.p_down
                else:
                    # self.p_mid += self.p_down
                    self.p_down = 0
            elif self.down is not None:
                self.future_down = self.future_center.next_down()
                self.future_down.cumulative_p += self.cumulative_p * self.p_down

        if not self.tree.config.pruning:
            self.future_up = self.future_center.next_up()
            self.future_up.cumulative_p += self.cumulative_p * self.p_up
            self.future_down = self.future_center.next_down()
            self.future_down.cumulative_p += self.cumulative_p * self.p_down

    def _calculate_payoff(self) -> float:
        """Calculation of the payoff according to the type of contract

        Returns:
            float: the payoff
        """
        option = self.tree.option

        def call_put_payoff():
            return (
                self.spot_price - option.strike_price
                if option.is_call
                else option.strike_price - self.spot_price
            )

        if option.barrier is not None:
            if option.barrier.barrier_type is BarrierType.knock_in:
                if (
                    option.barrier.barrier_direction is BarrierDirection.up
                    and self.spot_price >= option.barrier.barrier_level
                ) or (
                    option.barrier.barrier_direction is BarrierDirection.down
                    and self.spot_price <= option.barrier.barrier_level
                ):
                    payoff = max(self.tree.config.min_payoff, call_put_payoff())
                else:
                    payoff = self.tree.config.min_payoff

            elif option.barrier.barrier_type is BarrierType.knock_out:
                if (
                    option.barrier.barrier_direction is BarrierDirection.up
                    and self.spot_price >= option.barrier.barrier_level
                ) or (
                    option.barrier.barrier_direction is BarrierDirection.down
                    and self.spot_price <= option.barrier.barrier_level
                ):
                    payoff = self.tree.config.min_payoff
                else:
                    payoff = max(self.tree.config.min_payoff, call_put_payoff())
            else:
                payoff = self.tree.config.min_payoff

        else:
            payoff = max(call_put_payoff(), self.tree.config.min_payoff)

        return payoff

    def _calculate_intrinsic_value(self) -> None:
        """Allows us to calculate the intrinsic value of the Node, taking into account the type of option considered"""
        if self.future_center is None:
            self.intrinsic_value = self._calculate_payoff()

        elif self.intrinsic_value is None:
            for future_node_name in ["future_up", "future_center", "future_down"]:
                if getattr(self, future_node_name) is None:
                    setattr(
                        self,
                        future_node_name,
                        Node(0, self.tree, self.tree_position + 1),
                    )
                    child_node = getattr(self, future_node_name)
                    child_node.intrinsic_value = 0
                else:
                    child_node = getattr(self, future_node_name)
                    if getattr(child_node, "intrinsic_value") is None:
                        child_node._calculate_intrinsic_value()

            if (
                self.future_up is None
                or self.future_center is None
                or self.future_down is None
            ):
                raise ValueError("Future nodes not defined")

            if self.p_up is None or self.p_mid is None or self.p_down is None:
                raise ValueError("Probabilities not defined")

            proba_vector = np.array(
                [self.p_up, self.p_mid, self.p_down]
            )  # vector composed of the probabilities of the future Nodes of the current Node
            price_vector = np.array(
                [
                    self.future_up.intrinsic_value,
                    self.future_center.intrinsic_value,
                    self.future_down.intrinsic_value,
                ]
            )  #
            intrinsic_value = self.tree.discount_factor * price_vector.dot(
                proba_vector
            )  # here, scalar product of prices by their probabilities

            if self.tree.option.is_american:
                payoff_dt = self._calculate_payoff()
                intrinsic_value = max(payoff_dt, intrinsic_value)

            # Handle Knock-Out Barrier (Path Dependent)
            if (
                self.tree.option.barrier is not None
                and self.tree.option.barrier.barrier_type == BarrierType.knock_out
            ):
                # Check if barrier is breached at this node
                is_breached = False
                if (
                    self.tree.option.barrier.barrier_direction == BarrierDirection.up
                    and self.spot_price >= self.tree.option.barrier.barrier_level
                ):
                    is_breached = True
                elif (
                    self.tree.option.barrier.barrier_direction == BarrierDirection.down
                    and self.spot_price <= self.tree.option.barrier.barrier_level
                ):
                    is_breached = True

                if is_breached:
                    intrinsic_value = 0  # Assuming 0 rebate

            self.intrinsic_value = intrinsic_value
