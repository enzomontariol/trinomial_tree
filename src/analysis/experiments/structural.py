import numpy as np
from typing import Any, Dict, List
from src.analysis.experiments.framework import Experiment
from src.pricing.market import MarketData
from src.pricing.option import Option
from src.pricing.inductive_tree import InductiveTree

# from src.analysis.utils.tree_inspector import TreeInspector
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
            dist_df["N"] = self.N
            return dist_df
        else:
            print("Error: InductiveTree does not support terminal distribution.")
            return pd.DataFrame()

        # Calculate Theoretical Lognormal PDF
        # ... (omitted)
