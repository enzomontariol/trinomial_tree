from src.analysis.experiments.framework import Experiment
from src.pricing.market import MarketData
from src.pricing.option import Option
from src.pricing.inductive_tree import InductiveTree

import pandas as pd
import numpy as np


class ExerciseBoundaryExperiment(Experiment):
    """
    Extracts the Early Exercise Boundary for an American Option.
    """

    def __init__(self, market_data: MarketData, option: Option, N: int):
        super().__init__("Exercise Boundary Analysis")
        self.market_data = market_data
        self.option = option
        self.N = N

    def run(self) -> pd.DataFrame:
        tree = InductiveTree(
            num_steps=self.N, market_data=self.market_data, option=self.option
        )

        if hasattr(tree, "get_exercise_boundary"):
            boundary_df = tree.get_exercise_boundary()
            boundary_df["N"] = self.N
            return boundary_df
        else:
            print("Error: InductiveTree does not support exercise boundary extraction.")
            return pd.DataFrame()


class TerminalDistributionExperiment(Experiment):
    """
    Extracts the terminal distribution of the underlying asset from the Tree.
    """

    def __init__(self, market_data: MarketData, option: Option, N: int):
        super().__init__("Terminal Distribution Analysis")
        self.market_data = market_data
        self.option = option
        self.N = N

    def run(self) -> pd.DataFrame:
        tree = InductiveTree(
            num_steps=self.N, market_data=self.market_data, option=self.option
        )

        if hasattr(tree, "get_terminal_distribution"):
            dist_df = tree.get_terminal_distribution()

            # Compute Theoretical PDF for Black-Scholes
            S_vals = dist_df["Spot"].values
            # Filter S > 0 just in case
            r = self.market_data.interest_rate
            q = 0.0
            sigma = self.market_data.volatility
            days = (self.option.maturity - self.option.pricing_date).days
            T = days / self.option.calendar_base_convention
            S0 = self.market_data.spot_price

            valid_mask = S_vals > 0
            pdf_values = np.zeros_like(S_vals, dtype=float)

            if np.any(valid_mask):
                S_valid = S_vals[valid_mask]
                # ln(S_T) ~ N(mu_log, sigma^2 * T)
                # mu_log = ln(S_0) + (r - q - 0.5 * sigma^2) * T
                mu_log = np.log(S0) + (r - q - 0.5 * sigma**2) * T
                sigma_log = sigma * np.sqrt(T)

                # PDF of lognormal
                denom = S_valid * sigma_log * np.sqrt(2 * np.pi)
                num = np.exp(-((np.log(S_valid) - mu_log) ** 2) / (2 * sigma_log**2))
                pdf_values[valid_mask] = num / denom

            dist_df["Theoretical_PDF"] = pdf_values
            dist_df["N"] = self.N
            return dist_df
        else:
            print("Error: InductiveTree does not support terminal distribution.")
            return pd.DataFrame()
