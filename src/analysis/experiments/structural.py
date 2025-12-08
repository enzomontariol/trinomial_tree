import numpy as np
from typing import Any, Dict, List
from src.analysis.experiments.framework import Experiment
from src.pricing.market import MarketData
from src.pricing.option import Option
from src.pricing.tree_node import Tree
from src.analysis.utils.tree_inspector import TreeInspector
import pandas as pd


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
        tree = Tree(num_steps=self.N, market_data=self.market_data, option=self.option)
        tree.price()

        inspector = TreeInspector(tree)
        boundary_df = inspector.get_early_exercise_boundary()
        boundary_df["N"] = self.N
        return boundary_df


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
        tree = Tree(num_steps=self.N, market_data=self.market_data, option=self.option)
        tree.price()

        inspector = TreeInspector(tree)
        dist_df = inspector.get_terminal_distribution()
        dist_df["N"] = self.N

        # Calculate Theoretical Lognormal PDF
        # PDF(S) = 1/(S*sigma*sqrt(T)*sqrt(2pi)) * exp( - (ln S - mu)^2 / (2 sigma^2 T) )
        # mu = ln S0 + (r - q - sigma^2/2)T

        S0 = self.market_data.spot_price
        r = self.market_data.interest_rate
        sigma = self.market_data.volatility
        T = (
            self.option.maturity - self.option.pricing_date
        ).days / self.option.calendar_base_convention

        # Assuming no dividends for simplicity or continuous yield if q was available
        # MarketData has dividend_amount and ex_date, which is discrete.
        # For distribution approximation, we can ignore discrete dividends or approximate.
        # Let's assume standard BS dynamics for the curve comparison.

        mu = np.log(S0) + (r - 0.5 * sigma**2) * T
        denom = sigma * np.sqrt(T)

        def lognormal_pdf(s):
            if s <= 0:
                return 0.0
            return (1.0 / (s * denom * np.sqrt(2 * np.pi))) * np.exp(
                -((np.log(s) - mu) ** 2) / (2 * denom**2)
            )

        dist_df["Theoretical_PDF"] = dist_df["Spot"].apply(lognormal_pdf)

        # Scale PDF to match probability mass roughly (PDF * bin_width approx)
        # But since tree is discrete, we can just plot PDF on a secondary axis or just overlay shape.
        # Better: Normalize both to sum to 1 or area 1?
        # Tree probs sum to 1.
        # PDF integrates to 1.
        # To compare, we can multiply PDF by the average spacing?
        # Or just return the PDF values and let visualizer handle it.

        return dist_df
