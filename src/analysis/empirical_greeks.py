from typing import Union
import datetime as dt
import copy

from src.pricing.inductive_tree import InductiveTree


class EmpiricalGreeks:
    def __init__(
        self,
        tree: InductiveTree,
        var_s: float = 0.01,
        var_v: float = 0.01,
        var_t: int = 1,
        var_r: float = 0.01,
    ):
        """Initialization of the class

        Args:
            tree (InductiveTree): The Tree for which the greeks calculation will be performed
            var_s (float, optional): The variation applied to the underlying price (in percentage of the price). Defaults to 0.01.
            var_v (float, optional): The variation of the volatility level (in percentage points here). Defaults to 0.01.
            var_t (int, optional): The variation of the pricing date (in number of days). Defaults to 1.
            var_r (float, optional): The variation of the interest rate level (in percentage points). Defaults to 0.01.
        """

        self.tree = tree
        self.var_s = var_s
        self.var_v = var_v
        self.var_t = var_t
        self.var_r = var_r

        self.base_price = self.tree.price()

    def _price_tree_shock(
        self, attribute_to_modify: str, d: Union[float, dt.date]
    ) -> float:
        """Method allowing us to value a new option by varying a given parameter.

        Args:
            attribute_to_modify (str): Which parameter we are going to move.
            d (Union[float, dt.date]): Level of variation

        Returns:
            float: The price of the new Tree that has been priced all else equal.
        """

        market_data_modified = copy.copy(self.tree.market_data)
        option_modified = copy.copy(self.tree.option)

        # Theta case
        if attribute_to_modify == "pricing_date":
            setattr(option_modified, attribute_to_modify, d)
            setattr(market_data_modified, "start_date", d)

        # Rho case
        elif (
            attribute_to_modify == "interest_rate"
        ):  # we assume that the shock on interest rates applies to both capitalization and discount factors
            attribute_to_modify_1 = "interest_rate"
            attribute_to_modify_2 = "discount_rate"
            d1 = getattr(market_data_modified, attribute_to_modify_1) + d
            d2 = getattr(market_data_modified, attribute_to_modify_2) + d
            setattr(market_data_modified, attribute_to_modify_1, d1)
            setattr(market_data_modified, attribute_to_modify_2, d2)

        # General case
        else:
            d1 = getattr(market_data_modified, attribute_to_modify) + d
            setattr(market_data_modified, attribute_to_modify, d1)

        # Initialization of the new Tree and valuation

        new_tree = InductiveTree(
            num_steps=self.tree.num_steps,
            market_data=market_data_modified,
            option=option_modified,
            config=self.tree.config,
        )

        return new_tree.price()

    def approximate_delta(self) -> float:
        """Calculation of the delta of our option, the partial derivative of the option price with respect to the underlying price.

        Returns:
            float: the delta
        """

        # calculation of the new spot that we will use in the pricing of the new Tree
        ds = self.var_s * self.tree.market_data.spot_price
        neg_ds = -self.var_s * self.tree.market_data.spot_price

        # Valuation with the new parameters
        price_1 = self._price_tree_shock("spot_price", ds)
        price_2 = self._price_tree_shock("spot_price", neg_ds)

        # Here, we calculate the delta from a centered finite difference
        delta = (price_1 - price_2) / (2 * ds)

        # we store in the class the value of the shocked Tree so as not to have to recalculate if we calculate a second order derivative
        if not hasattr(self, "price_new_tree_ds_1"):
            self.price_new_tree_ds_1 = price_1

        # same here
        if not hasattr(self, "price_new_tree_ds_2"):
            self.price_new_tree_ds_2 = price_2

        return delta

    def approximate_gamma(self) -> float:
        """Calculation of the gamma of our option, the second partial derivative of our option price with respect to the underlying price.

        Returns:
            float: the gamma
        """

        # calculation of the new spot that we will use in the pricing of the new Tree
        ds = self.var_s * self.tree.market_data.spot_price
        neg_ds = -self.var_s * self.tree.market_data.spot_price

        # In case we have not previously calculated the delta of the option.
        if not hasattr(self, "price_new_tree_ds_1"):
            self.price_new_tree_ds_1 = self._price_tree_shock("spot_price", ds)

        if not hasattr(self, "price_new_tree_ds_2"):
            self.price_new_tree_ds_2 = self._price_tree_shock("spot_price", neg_ds)

        # Calculation of gamma according to the formula of a second-order centered finite difference
        gamma = (
            self.price_new_tree_ds_1 - 2 * self.base_price + self.price_new_tree_ds_2
        ) / (ds**2)

        return gamma

    def approximate_vega(self) -> float:
        """Calculation of the vega of our option, the partial derivative of our option price with respect to the volatility level of the underlying

        Returns:
            float: the vega
        """

        dv = self.var_v
        neg_dv = -self.var_v

        # calculation of the Trees that we will use
        price_1 = self._price_tree_shock("volatility", dv)
        price_2 = self._price_tree_shock("volatility", neg_dv)

        # Centered finite difference
        vega = (price_1 - price_2) / 2 * dv * 100

        # we store in the class the value of the shocked Tree so as not to have to recalculate if we calculate a second order derivative
        if not hasattr(self, "vol_new_tree_dv_1"):
            self.vol_new_tree_ds_1 = price_1

        # same here
        if not hasattr(self, "vol_new_tree_dv_2"):
            self.vol_new_tree_ds_2 = price_2

        return vega

    def approximate_theta(self) -> float:
        """Calculation of the theta of our option, the partial derivative of our option price with respect to time.

        Returns:
            float: the theta
        """

        # Here, the difference is made on a number of days
        d_t = self.tree.option.pricing_date + dt.timedelta(days=self.var_t)

        # New Tree priced at the date determined above
        price_1 = self._price_tree_shock("pricing_date", d_t)

        # calculation of theta via forward finite difference.
        theta = price_1 - self.base_price

        # we store in the class the value of the shocked Tree so as not to have to recalculate if we calculate a second order derivative
        if not hasattr(self, "time_new_tree_dt_1"):
            self.theta_new_tree_ds_1 = price_1

        return theta

    def approximate_rho(self) -> float:
        """Calculation of rho, the partial derivative of our option price with respect to the risk-free interest rate

        Returns:
            float: The rho
        """

        dr = self.var_r
        neg_dr = -self.var_r

        # The new Trees that we will use in the finite difference
        price_1 = self._price_tree_shock("interest_rate", dr)
        price_2 = self._price_tree_shock("interest_rate", neg_dr)

        # Calculation of rho via centered finite difference
        rho = (price_1 - price_2) / 2 * dr * 100

        # we store in the class the value of the shocked Tree so as not to have to recalculate if we calculate a second order derivative
        if not hasattr(self, "vol_new_tree_dr_1"):
            self.rho_new_tree_ds_1 = price_1

        # same here
        if not hasattr(self, "vol_new_tree_dr_2"):
            self.rho_new_tree_ds_2 = price_2

        return rho
