import time
from typing import Any, Dict, List
from src.analysis.experiments.framework import ParallelSweepExperiment
from src.pricing.market import MarketData
from src.pricing.option import Option
from src.pricing.black_scholes import BlackScholes
from src.pricing.tree_node import Tree


class ConvergenceExperiment(ParallelSweepExperiment):
    """
    Analyzes convergence of Tree price to Black-Scholes price as N increases.
    Covers:
    - 1.1 Price vs N
    - 1.2 Error vs N
    - 8.1 Runtime vs N
    """

    def __init__(
        self,
        N_values: List[int],
        market_data: MarketData,
        option: Option,
        max_workers: int = 4,
    ):
        super().__init__(
            name="Convergence Analysis",
            param_name="N",
            param_values=N_values,
            max_workers=max_workers,
        )
        self.market_data = market_data
        self.option = option

        # Pre-calculate BS price as benchmark
        bs_pricer = BlackScholes(market_data, option)
        self.bs_price = bs_pricer.price()

    def _run_single_iteration(self, N: int) -> Dict[str, Any]:
        start_time = time.time()

        # Instantiate Tree with N steps
        tree_pricer = Tree(
            num_steps=N, market_data=self.market_data, option=self.option
        )
        tree_price = tree_pricer.price()

        end_time = time.time()
        runtime = end_time - start_time

        error = abs(tree_price - self.bs_price)

        return {
            "N": N,
            "Tree_Price": tree_price,
            "BS_Price": self.bs_price,
            "Error": error,
            "Runtime": runtime,
        }
