import numpy as np
from typing import Any, Dict, List
from src.analysis.experiments.framework import ParallelSweepExperiment
from src.pricing.market import MarketData
from src.pricing.option import Option
from src.pricing.inductive_tree import InductiveTree


class PutCallParityExperiment(ParallelSweepExperiment):
    """
    Analyzes Put-Call Parity consistency.
    Residual = C_tree - P_tree - (S0 - K * exp(-rT))
    Note: Strictly valid for European options.
    """

    def __init__(
        self,
        sweep_param: str,  # "S0" or "N"
        param_values: List[Any],
        market_data: MarketData,
        base_option: Option,
        fixed_N: int = 100,
        max_workers: int = 4,
    ):
        super().__init__(
            name=f"Put-Call Parity Analysis (Sweeping {sweep_param})",
            param_name=sweep_param,
            param_values=param_values,
            max_workers=max_workers,
        )
        self.market_data = market_data
        self.base_option = base_option
        self.fixed_N = fixed_N
        self.sweep_param = sweep_param

    def _run_single_iteration(self, value: Any) -> Dict[str, Any]:
        # Configure Market and Option based on sweep parameter
        if self.sweep_param == "S0":
            S0 = value
            N = self.fixed_N
            market_data = self._clone_market_data(self.market_data, S0)
        elif self.sweep_param == "N":
            S0 = self.market_data.spot_price
            N = value
            market_data = self.market_data
        else:
            raise ValueError(f"Unknown sweep parameter: {self.sweep_param}")

        # Ensure we have both Call and Put options (European)
        call_option = self._create_european_option(self.base_option, is_call=True)
        put_option = self._create_european_option(self.base_option, is_call=False)

        # Price Call
        tree_call = InductiveTree(
            num_steps=N, market_data=market_data, option=call_option
        )
        C_tree = tree_call.price()

        # Price Put
        tree_put = InductiveTree(
            num_steps=N, market_data=market_data, option=put_option
        )
        P_tree = tree_put.price()

        # Theoretical Parity
        T = (
            call_option.maturity - call_option.pricing_date
        ).days / call_option.calendar_base_convention
        df = np.exp(-market_data.interest_rate * T)

        # PV of dividends
        pv_div = 0.0
        if (
            market_data.dividend_ex_date > call_option.pricing_date
            and market_data.dividend_ex_date <= call_option.maturity
        ):
            t_div = (
                market_data.dividend_ex_date - call_option.pricing_date
            ).days / call_option.calendar_base_convention
            pv_div = market_data.dividend_amount * np.exp(
                -market_data.interest_rate * t_div
            )

        parity_val = S0 - pv_div - call_option.strike_price * df
        residual = C_tree - P_tree - parity_val

        return {
            self.sweep_param: value,
            "Call_Price": C_tree,
            "Put_Price": P_tree,
            "Parity_Target": parity_val,
            "Residual": residual,
        }

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

    def _create_european_option(self, opt, is_call):
        return Option(
            maturity=opt.maturity,
            strike_price=opt.strike_price,
            is_call=is_call,
            is_american=False,  # Force European for Parity Check
            barrier=opt.barrier,
            pricing_date=opt.pricing_date,
            calendar_base_convention=opt.calendar_base_convention,
        )
