from abc import ABC, abstractmethod
import pandas as pd
import concurrent.futures
import time
from typing import List, Dict, Any, Callable, Union, Optional


class Experiment(ABC):
    """Base class for experiments."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def run(self) -> pd.DataFrame:
        """Runs the experiment and returns a DataFrame with results."""
        pass


class ParallelSweepExperiment(Experiment):
    """
    Helper to run a function over a range of parameters in parallel.
    """

    def __init__(
        self, name: str, param_name: str, param_values: List[Any], max_workers: int = 4
    ):
        super().__init__(name)
        self.param_name = param_name
        self.param_values = param_values
        self.max_workers = max_workers

    @abstractmethod
    def _run_single_iteration(self, value: Any) -> Dict[str, Any]:
        """
        Run a single iteration for a given parameter value.
        Returns a dictionary of results (metrics).
        """
        pass

    def run(self) -> pd.DataFrame:
        results = []
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            # Map futures to values
            future_to_val = {
                executor.submit(self._run_single_iteration, val): val
                for val in self.param_values
            }

            for future in concurrent.futures.as_completed(future_to_val):
                val = future_to_val[future]
                try:
                    res = future.result()
                    # Ensure the sweep parameter is in the result
                    if self.param_name not in res:
                        res[self.param_name] = val
                    results.append(res)
                except Exception as exc:
                    print(
                        f"Experiment {self.name} failed for {self.param_name}={val}: {exc}"
                    )

        df = pd.DataFrame(results)
        if not df.empty and self.param_name in df.columns:
            df = df.sort_values(by=self.param_name)
        return df
