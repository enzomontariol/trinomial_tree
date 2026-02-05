import time
from typing import Any, Dict, List
from .framework import ParallelSweepExperiment
from ...pricing.market import MarketData
from ...pricing.option import Option
from ...pricing.black_scholes import BlackScholes
from ...pricing.inductive_tree import InductiveTree


class SpotSensitivityExperiment(ParallelSweepExperiment):
    """
    Analyzes sensitivity of Price and Greeks to Spot Price (S0).
    Covers:
    - 2.1 Price vs S0
    - 2.2 Gamma vs S0 (Finite Difference)
    - 3.1 Delta vs S0
    """

    def __init__(
        self,
        S0_values: List[float],
        market_data: MarketData,
        option: Option,
        N: int,
        max_workers: int = 4,
    ):
        super().__init__(
            name="Spot Sensitivity Analysis",
            param_name="S0",
            param_values=S0_values,
            max_workers=max_workers,
        )
        self.base_market_data = market_data
        self.option = option
        self.N = N

    def _run_single_iteration(self, value: float) -> Dict[str, Any]:
        S0 = value
        # Create market data with new spot
        market_data = MarketData(
            spot_price=S0,
            start_date=self.base_market_data.start_date,
            volatility=self.base_market_data.volatility,
            interest_rate=self.base_market_data.interest_rate,
            discount_rate=self.base_market_data.discount_rate,
            dividend_ex_date=self.base_market_data.dividend_ex_date,
            dividend_amount=self.base_market_data.dividend_amount,
        )

        # BS Price & Greeks
        bs = BlackScholes(market_data, self.option)
        bs_price = bs.price()
        bs_delta = bs.delta()
        bs_gamma = bs.gamma()

        # Tree Price
        tree = InductiveTree(
            num_steps=self.N, market_data=market_data, option=self.option
        )
        tree_price = tree.price()

        tree_delta = self._calculate_tree_delta_fd(market_data, self.option, self.N)
        tree_gamma_fd = self._calculate_tree_gamma_fd(market_data, self.option, self.N)

        # Internal Gamma not supported with InductiveTree (no node access)
        tree_gamma_internal = 0.0

        return {
            "S0": S0,
            "Tree_Price": tree_price,
            "BS_Price": bs_price,
            "Tree_Delta": tree_delta,
            "BS_Delta": bs_delta,
            "Tree_Gamma": tree_gamma_fd,
            "Tree_Gamma_Internal": tree_gamma_internal,
            "BS_Gamma": bs_gamma,
        }

    def _calculate_tree_delta_fd(self, market_data, option, N) -> float:
        dS = market_data.spot_price * 0.01
        md_up = self._clone_market_data(market_data, market_data.spot_price + dS)
        md_down = self._clone_market_data(market_data, market_data.spot_price - dS)
        p_up = InductiveTree(N, md_up, option).price()
        p_down = InductiveTree(N, md_down, option).price()
        return (p_up - p_down) / (2 * dS)

    def _calculate_tree_gamma_fd(self, market_data, option, N) -> float:
        # Finite difference
        dS = market_data.spot_price * 0.01

        md_up = self._clone_market_data(market_data, market_data.spot_price + dS)
        md_down = self._clone_market_data(market_data, market_data.spot_price - dS)

        p_up = InductiveTree(N, md_up, option).price()
        p_mid = InductiveTree(N, market_data, option).price()
        p_down = InductiveTree(N, md_down, option).price()

        return (p_up - 2 * p_mid + p_down) / (dS**2)

    def _clone_market_data(self, md, spot):
        return MarketData(
            spot_price=spot,
            start_date=md.start_date,
            volatility=md.volatility,
            interest_rate=md.interest_rate,
            discount_rate=md.discount_rate,
            dividend_ex_date=md.dividend_ex_date,
            dividend_amount=md.dividend_amount,
        )


class VolatilitySensitivityExperiment(ParallelSweepExperiment):
    """
    Analyzes sensitivity of Price and Error to Volatility.
    Covers:
    - 5.1 Price vs Volatility
    - 5.2 Error vs Volatility
    """

    def __init__(
        self,
        vol_values: List[float],
        market_data: MarketData,
        option: Option,
        N: int,
        max_workers: int = 4,
    ):
        super().__init__(
            name="Volatility Sensitivity Analysis",
            param_name="Volatility",
            param_values=vol_values,
            max_workers=max_workers,
        )
        self.base_market_data = market_data
        self.option = option
        self.N = N

    def _run_single_iteration(self, value: float) -> Dict[str, Any]:
        vol = value
        market_data = MarketData(
            spot_price=self.base_market_data.spot_price,
            start_date=self.base_market_data.start_date,
            volatility=vol,
            interest_rate=self.base_market_data.interest_rate,
            discount_rate=self.base_market_data.discount_rate,
            dividend_ex_date=self.base_market_data.dividend_ex_date,
            dividend_amount=self.base_market_data.dividend_amount,
        )

        bs = BlackScholes(market_data, self.option)
        bs_price = bs.price()

        tree = InductiveTree(
            num_steps=self.N, market_data=market_data, option=self.option
        )
        tree_price = tree.price()

        error = abs(tree_price - bs_price)

        return {
            "Volatility": vol,
            "Tree_Price": tree_price,
            "BS_Price": bs_price,
            "Error": error,
        }
