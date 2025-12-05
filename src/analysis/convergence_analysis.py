from abc import ABC, abstractmethod
import concurrent.futures
import time
import pandas as pd
import warnings
from typing import Type, Any, Tuple

from src.pricing.pricer import Pricer
from src.pricing.market import MarketData
from src.pricing.option import Option

warnings.filterwarnings("ignore")


class ConvergenceAnalysis(ABC):
    """
    Abstract base class for analyzing convergence of a pricing model against a benchmark.

    This class allows sweeping a parameter (e.g. Strike, Volatility) and for each value,
    analyzing the convergence of a numerical model (e.g. Tree, MC) as a function of
    a convergence parameter (e.g. number of steps, number of simulations).
    """

    def __init__(
        self,
        model_class: Type[Pricer],
        benchmark_class: Type[Pricer],
        convergence_param_name: str,
        max_cpu: int,
        convergence_values: list,
        market_data: MarketData,
        option: Option,
        x_axis_name: str = "Convergence Parameter",
        model_price_name: str = "Model Price",
        pricing_time_name: str = "Pricing Time",
        diff_benchmark_name: str = "Diff Benchmark",
    ):
        """
        Initializes the convergence analysis.

        Args:
            model_class (Type[Pricer]): The class of the model to test (e.g. Tree).
            benchmark_class (Type[Pricer]): The class of the benchmark model (e.g. BlackScholes).
            convergence_param_name (str): The name of the argument in model_class constructor
                                          that controls convergence (e.g. "num_steps").
            max_cpu (int): Maximum number of CPU cores to use.
            convergence_values (list): List of values for the convergence parameter.
            market_data (MarketData): Base market data.
            option (Option): Base option data.
            x_axis_name (str): Label for the convergence parameter in results.
            model_price_name (str): Label for the model price in results.
            pricing_time_name (str): Label for the pricing time in results.
            diff_benchmark_name (str): Label for the difference with benchmark in results.
        """
        self.model_class = model_class
        self.benchmark_class = benchmark_class
        self.convergence_param_name = convergence_param_name
        self.max_cpu = max_cpu
        self.convergence_values = convergence_values
        self.market_data = market_data
        self.option = option
        self.x_axis_name = x_axis_name
        self.model_price_name = model_price_name
        self.pricing_time_name = pricing_time_name
        self.diff_benchmark_name = diff_benchmark_name
        self.results_df = pd.DataFrame()

    def _run_analysis(self, values: list, param_name: str):
        """Runs the analysis for a list of parameter values in parallel."""
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self.max_cpu
        ) as outer_executor:
            futures = {
                outer_executor.submit(self._process_value, val): val for val in values
            }

            for future in concurrent.futures.as_completed(futures):
                val = futures[future]
                try:
                    result_df = future.result()
                    result_df[param_name] = val
                    self.results_df = pd.concat(
                        [self.results_df, result_df], ignore_index=True
                    )
                except Exception as exc:
                    print(f"Error at {param_name} {val} : {exc}")

    def _process_value(self, value):
        """Processes a single parameter value, calculating prices for all convergence steps."""
        market_data, option, model_kwargs = self._get_config(value)

        # Calculate benchmark price
        # We assume benchmark_class takes market_data and option in constructor
        benchmark = self.benchmark_class(market_data=market_data, option=option)
        benchmark_price = benchmark.price()

        results = []
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self.max_cpu
        ) as inner_executor:
            futures = {
                inner_executor.submit(
                    self._calculate_point,
                    c_val,
                    benchmark_price,
                    market_data,
                    option,
                    model_kwargs,
                    value,
                    self.model_class,
                    self.convergence_param_name,
                ): c_val
                for c_val in self.convergence_values
            }

            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())

        return pd.DataFrame(
            results,
            columns=[
                self.x_axis_name,
                self.model_price_name,
                self.pricing_time_name,
                self.diff_benchmark_name,
                f"{self.diff_benchmark_name} * {self.x_axis_name}",
            ],
        )

    @abstractmethod
    def _get_config(self, value) -> Tuple[MarketData, Option, dict]:
        """
        Returns the configuration for a specific parameter value.

        Returns:
            Tuple[MarketData, Option, dict]: (MarketData, Option, model_kwargs)
        """
        pass

    @staticmethod
    def _calculate_point(
        c_val,
        benchmark_price,
        market_data,
        option,
        model_kwargs,
        value,
        model_class,
        convergence_param_name,
    ):
        """Calculates the price for a single convergence point."""
        now = time.time()

        # Prepare kwargs for the model
        kwargs = model_kwargs.copy()
        kwargs[convergence_param_name] = c_val

        # Instantiate and price
        model = model_class(market_data=market_data, option=option, **kwargs)
        price = model.price()

        then = time.time()
        pricing_time = then - now

        # print(f"{convergence_param_name}: {c_val}, Value: {value}, Time: {pricing_time:.4f}s")

        if price is None:
            raise ValueError("Model price not calculated")

        return (
            c_val,
            price,
            pricing_time,
            price - benchmark_price,
            price * c_val,
        )
