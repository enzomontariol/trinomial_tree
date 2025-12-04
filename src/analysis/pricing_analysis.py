# %% Imports
import concurrent.futures
import time
import pandas as pd
import datetime as dt
import sys
import plotly.graph_objects as go
import warnings

warnings.filterwarnings("ignore")

sys.setrecursionlimit(1000000000)

from ..pricing.market import MarketData
from ..pricing.option import Option
from ..pricing.barrier import Barrier
from ..pricing.tree_node import Tree

# import Black Scholes functions
from ..pricing.black_scholes import BlackScholes

# %% Classes


class BsComparison:
    def __init__(self, max_cpu: int, step_list: list, epsilon_values: list):
        self.max_cpu = max_cpu

        # Define the two lists
        self.step_list = step_list
        self.epsilon_values = epsilon_values

        # Instantiation of required objects
        self.barrier = Barrier(0, None, None)
        self.market_data = MarketData(
            dt.date(2024, 1, 13), 100, 0.20, 0.02, 0.02, dt.date.today(), 0
        )
        self.option = Option(
            dt.date(2024, 10, 23), 101, self.barrier, False, True, dt.date(2024, 1, 13)
        )

        # Calculation of B&S price
        self.tree_bs = Tree(100, self.market_data, self.option)
        self.bs_price = BlackScholes(self.tree_bs).price()

        # DataFrame to store results
        self.results_df = pd.DataFrame()

        # Execution of calculations in parallel for each epsilon level
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self.max_cpu
        ) as outer_executor:
            futures = {
                outer_executor.submit(self.calculate_for_epsilon, epsilon): epsilon
                for epsilon in self.epsilon_values
            }

            # Results
            for future in concurrent.futures.as_completed(futures):
                epsilon = futures[future]
                try:
                    result_df = future.result()
                    result_df["Epsilon (1e-)"] = epsilon
                    self.results_df = pd.concat(
                        [self.results_df, result_df], ignore_index=True
                    )
                except Exception as exc:
                    print(f"Error at epsilon {epsilon} : {exc}")

    def calculate_for_epsilon(self, epsilon):
        list_price_step_comparison = []
        list_time_step_comparison = []
        list_diff_bs = []
        list_time_gap_bs = []

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self.max_cpu
        ) as inner_executor:
            futures = {
                inner_executor.submit(
                    self.calculate_step,
                    step,
                    self.bs_price,
                    self.market_data,
                    self.option,
                    epsilon,
                ): step
                for step in self.step_list
            }

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                list_price_step_comparison.append(result[1])
                list_time_step_comparison.append(result[2])
                list_diff_bs.append(result[3])
                list_time_gap_bs.append(result[4])

        return pd.DataFrame(
            {
                "Nombre de pas": self.step_list,
                "Prix Tree trinomial": list_price_step_comparison,
                "Temps pricing": list_time_step_comparison,
                "Différence B&S": list_diff_bs,
                "Différence * nb pas": list_time_gap_bs,
            }
        )

    @staticmethod
    def calculate_step(step, bs_price, market_data, option, epsilon):
        now = time.time()
        tree_step_comparison = Tree(step, market_data, option, epsilon=epsilon)
        tree_step_comparison.price_option()
        price_tree_step_comparison = tree_step_comparison.option_price
        then = time.time()
        pricing_time = then - now
        print(f"Step: {step}, Epsilon: {epsilon}, Pricing Time: {pricing_time:.4f}s")

        if price_tree_step_comparison is None:
            raise ValueError("Option price not calculated")

        return (
            step,
            price_tree_step_comparison,
            pricing_time,
            price_tree_step_comparison - bs_price,
            price_tree_step_comparison * step,
        )

    def bs_graph_temps_pas(self):
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=self.results_df["Nombre de pas"],
                y=self.results_df["Temps pricing"],
                mode="lines+markers",
                name="Temps de pricing",
                line=dict(color="blue", width=2),
                marker=dict(size=6),
            )
        )

        fig.update_layout(
            title="Temps de pricing en fonction du nombre de pas",
            xaxis_title="Nombre de pas",
            yaxis_title="Temps de pricing (secondes)",
        )

        return fig

    def bs_graph_prix_pas(self):
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=self.results_df["Nombre de pas"],
                y=self.results_df["Prix Tree trinomial"],
                mode="lines+markers",
                name="Prix Tree trinomial",
                line=dict(color="blue", width=2),
                marker=dict(size=6),
            )
        )

        fig.add_hline(
            y=self.bs_price,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Prix B&S: {round(self.bs_price, 4)}",
            annotation_position="bottom right",
            annotation_font=dict(size=12, color="red"),
        )

        fig.update_layout(
            title="""Prix renvoyé par l'Tree en fonction du nombre de pas""",
            xaxis_title="Nombre de pas",
            yaxis_title="Prix Tree",
        )

        return fig

    def epsilon_graph_prix_pas_bas_epsilon(self):
        fig = go.Figure()

        mask = [
            (value[0] < 1e-7)
            if isinstance(value, list) and len(value) > 0
            else (value < 1e-7)
            for value in self.results_df["Epsilon (1e-)"]
        ]

        filtered_df = self.results_df[mask]

        unique_epsilons = filtered_df["Epsilon (1e-)"].unique()

        for epsilon in unique_epsilons:
            filtered_data = self.results_df[self.results_df["Epsilon (1e-)"] == epsilon]
            fig.add_trace(
                go.Scatter(
                    x=filtered_data["Nombre de pas"],
                    y=filtered_data["Prix Tree trinomial"],
                    mode="lines+markers",
                    name=f"Epsilon = {epsilon}",
                    line=dict(width=2),
                    marker=dict(size=6),
                )
            )

        fig.add_hline(
            y=self.bs_price,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Prix B&S: {round(self.bs_price, 4)}",
            annotation_position="bottom right",
            annotation_font=dict(size=12, color="red"),
        )

        fig.update_layout(
            title="""Prix renvoyé par l'Tree en fonction du nombre de pas""",
            xaxis_title="Nombre de pas",
            yaxis_title="Prix Tree",
            legend_title="Epsilon Values",
        )

        return fig

    def epsilon_graph_prix_pas_haut_epsilon(self):
        fig = go.Figure()

        mask = [
            (value[0] > 1e-7)
            if isinstance(value, list) and len(value) > 0
            else (value > 1e-7)
            for value in self.results_df["Epsilon (1e-)"]
        ]

        filtered_df = self.results_df[mask]

        unique_epsilons = filtered_df["Epsilon (1e-)"].unique()

        for epsilon in unique_epsilons:
            filtered_data = self.results_df[self.results_df["Epsilon (1e-)"] == epsilon]
            fig.add_trace(
                go.Scatter(
                    x=filtered_data["Nombre de pas"],
                    y=filtered_data["Prix Tree trinomial"],
                    mode="lines+markers",
                    name=f"Epsilon = {epsilon}",
                    line=dict(width=2),
                    marker=dict(size=6),
                )
            )

        fig.add_hline(
            y=self.bs_price,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Prix B&S: {round(self.bs_price, 4)}",
            annotation_position="bottom right",
            annotation_font=dict(size=12, color="red"),
        )

        fig.update_layout(
            title="""Prix renvoyé par l'Tree en fonction du nombre de pas""",
            xaxis_title="Nombre de pas",
            yaxis_title="Prix Tree",
            legend_title="Epsilon Values",
        )

        return fig

    def epsilon_vs_temps_pricing_graph(self):
        fig = go.Figure()

        mask = [
            (value[0] == 5000)
            if isinstance(value, list) and len(value) > 0
            else (value == 5000)
            for value in self.results_df["Nombre de pas"]
        ]

        filtered_df = self.results_df[mask]

        fig.add_trace(
            go.Scatter(
                x=filtered_df["Epsilon (1e-)"],
                y=filtered_df["Temps pricing"],
                mode="lines+markers",
                line=dict(width=2),
                marker=dict(size=6),
            )
        )

        fig.update_xaxes(
            type="log", title="Epsilon", tickformat="1e", autorange="reversed"
        )

        fig.update_layout(
            title="Temps de valorisation pour un Tree de 5000 pas",
            xaxis_title="Epsilon",
            yaxis_title="Temps de valorisation (secondes)",
        )

        return fig


class StrikeComparison:
    def __init__(self, max_cpu: int, step_list: list, strike_values: list):
        self.max_cpu = max_cpu

        self.step_list = step_list
        self.strike_values = strike_values

        self.barrier = Barrier(0, None, None)
        self.market_data = MarketData(
            dt.date(2024, 1, 13), 100, 0.20, 0.02, 0.02, dt.date.today(), 0
        )

        self.results_df = pd.DataFrame()

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self.max_cpu
        ) as outer_executor:
            futures = {
                outer_executor.submit(self.calculate_for_strike, strike): strike
                for strike in self.strike_values
            }

            for future in concurrent.futures.as_completed(futures):
                strike = futures[future]
                try:
                    result_df = future.result()
                    result_df["Strike"] = strike
                    self.results_df = pd.concat(
                        [self.results_df, result_df], ignore_index=True
                    )
                except Exception as exc:
                    print(f"Error at strike {strike}: {exc}")

    def calculate_for_strike(self, strike):
        option = Option(
            dt.date(2024, 10, 23),
            strike_price=strike,
            barrier=self.barrier,
            is_american=False,
            is_call=True,
            pricing_date=dt.date(2024, 1, 13),
        )

        self.tree_bs = Tree(100, self.market_data, option)
        self.bs_price = BlackScholes(self.tree_bs).price()

        list_price_step_comparison = []
        list_time_step_comparison = []
        list_diff_bs = []
        list_time_gap_bs = []

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self.max_cpu
        ) as inner_executor:
            futures = {
                inner_executor.submit(
                    self.calculate_step, step, self.bs_price, self.market_data, option
                ): step
                for step in self.step_list
            }

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                list_price_step_comparison.append(result[1])
                list_time_step_comparison.append(result[2])
                list_diff_bs.append(result[3])
                list_time_gap_bs.append(result[4])

        return pd.DataFrame(
            {
                "Nombre de pas": self.step_list,
                "Prix Tree trinomial": list_price_step_comparison,
                "Temps pricing": list_time_step_comparison,
                "Différence B&S": list_diff_bs,
                "Différence * nb pas": list_time_gap_bs,
            }
        )

    @staticmethod
    def calculate_step(step, bs_price, market_data, option):
        now = time.time()
        tree_step_comparison = Tree(step, market_data, option)
        tree_step_comparison.price_option()
        price_tree_step_comparison = tree_step_comparison.option_price
        then = time.time()
        pricing_time = then - now
        print(
            f"Strike: {option.strike_price}, Prix option : {price_tree_step_comparison}, Pricing Time: {pricing_time:.4f}s"
        )

        if price_tree_step_comparison is None:
            raise ValueError("Option price not calculated")

        return (
            step,
            price_tree_step_comparison,
            pricing_time,
            price_tree_step_comparison - bs_price,
            price_tree_step_comparison * step,
        )

    def graph_strike_temps_pas(self):
        fig = go.Figure()

        sorted_df = self.results_df.sort_values("Strike", ascending=True)

        fig.add_trace(
            go.Scatter(
                x=sorted_df["Strike"],
                y=sorted_df["Différence B&S"],
                mode="lines+markers",
                line=dict(color="blue", width=2),
                marker=dict(size=6),
            )
        )

        fig.add_hline(y=0, line_dash="dash", line_color="red")

        fig.update_layout(
            title="Différence par rapport au pricing B&S en fonction du strike",
            xaxis_title="Strike",
            yaxis_title="Différence B&S",
        )

        return fig


class VolComparison:
    def __init__(self, max_cpu: int, step_list: list, vol_values: list):
        self.max_cpu = max_cpu

        self.step_list = step_list
        self.vol_values = vol_values

        self.barrier = Barrier(0, None, None)
        self.option = Option(
            dt.date(2024, 10, 23),
            strike_price=101,
            barrier=self.barrier,
            is_american=False,
            is_call=True,
            pricing_date=dt.date(2024, 1, 13),
        )

        self.results_df = pd.DataFrame()

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self.max_cpu
        ) as outer_executor:
            futures = {
                outer_executor.submit(self.calculate_for_vol, vol): vol
                for vol in self.vol_values
            }

            for future in concurrent.futures.as_completed(futures):
                vol = futures[future]
                try:
                    result_df = future.result()
                    result_df["Volatilité"] = vol
                    self.results_df = pd.concat(
                        [self.results_df, result_df], ignore_index=True
                    )
                except Exception as exc:
                    print(f"Error at vol {vol}: {exc}")

    def calculate_for_vol(self, vol):
        market_data = MarketData(
            dt.date(2024, 1, 13), 100, vol, 0.02, 0.02, dt.date.today(), 0
        )

        self.tree_bs = Tree(100, market_data, self.option)
        self.bs_price = BlackScholes(self.tree_bs).price()

        list_price_step_comparison = []
        list_time_step_comparison = []
        list_diff_bs = []
        list_time_gap_bs = []

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self.max_cpu
        ) as inner_executor:
            futures = {
                inner_executor.submit(
                    self.calculate_step, step, self.bs_price, market_data, self.option
                ): step
                for step in self.step_list
            }

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                list_price_step_comparison.append(result[1])
                list_time_step_comparison.append(result[2])
                list_diff_bs.append(result[3])
                list_time_gap_bs.append(result[4])

        return pd.DataFrame(
            {
                "Nombre de pas": self.step_list,
                "Prix Tree trinomial": list_price_step_comparison,
                "Temps pricing": list_time_step_comparison,
                "Différence B&S": list_diff_bs,
                "Différence * nb pas": list_time_gap_bs,
            }
        )

    @staticmethod
    def calculate_step(step, bs_price, market_data, option):
        now = time.time()
        tree_step_comparison = Tree(step, market_data, option)
        tree_step_comparison.price_option()
        price_tree_step_comparison = tree_step_comparison.option_price
        then = time.time()
        pricing_time = then - now
        print(
            f"Vol: {market_data.volatility}, Prix option : {price_tree_step_comparison}, Pricing Time: {pricing_time:.4f}s"
        )

        if price_tree_step_comparison is None:
            raise ValueError("Option price not calculated")

        return (
            step,
            price_tree_step_comparison,
            pricing_time,
            price_tree_step_comparison - bs_price,
            price_tree_step_comparison * step,
        )

    def graph_vol_temps_pas(self):
        fig = go.Figure()

        sorted_df = self.results_df.sort_values("Volatilité", ascending=True)

        fig.add_trace(
            go.Scatter(
                x=sorted_df["Volatilité"],
                y=sorted_df["Différence B&S"],
                mode="lines+markers",
                line=dict(color="blue", width=2),
                marker=dict(size=6),
            )
        )

        fig.add_hline(y=0, line_dash="dash", line_color="red")

        fig.update_layout(
            title="Différence par rapport au pricing B&S en fonction de la volatilité",
            xaxis_title="Volatilité",
            yaxis_title="Différence B&S",
        )

        return fig


class RateComparison:
    def __init__(self, max_cpu: int, step_list: list, rate_values: list):
        self.max_cpu = max_cpu

        self.step_list = step_list
        self.rate_values = rate_values

        self.barrier = Barrier(0, None, None)
        self.option = Option(
            dt.date(2024, 10, 23),
            strike_price=101,
            barrier=self.barrier,
            is_american=False,
            is_call=True,
            pricing_date=dt.date(2024, 1, 13),
        )

        self.results_df = pd.DataFrame()

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self.max_cpu
        ) as outer_executor:
            futures = {
                outer_executor.submit(self.calculate_for_rate, rate): rate
                for rate in self.rate_values
            }

            for future in concurrent.futures.as_completed(futures):
                rate = futures[future]
                try:
                    result_df = future.result()
                    result_df["""Taux d'intérêt"""] = rate
                    self.results_df = pd.concat(
                        [self.results_df, result_df], ignore_index=True
                    )
                except Exception as exc:
                    print(f"Error at rate {rate}: {exc}")

    def calculate_for_rate(self, rate):
        market_data = MarketData(
            dt.date(2024, 1, 13), 100, 0.2, rate, rate, dt.date.today(), 0
        )

        self.tree_bs = Tree(100, market_data, self.option)
        self.bs_price = BlackScholes(self.tree_bs).price()

        list_price_step_comparison = []
        list_time_step_comparison = []
        list_diff_bs = []
        list_time_gap_bs = []

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self.max_cpu
        ) as inner_executor:
            futures = {
                inner_executor.submit(
                    self.calculate_step, step, self.bs_price, market_data, self.option
                ): step
                for step in self.step_list
            }

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                list_price_step_comparison.append(result[1])
                list_time_step_comparison.append(result[2])
                list_diff_bs.append(result[3])
                list_time_gap_bs.append(result[4])

        return pd.DataFrame(
            {
                "Nombre de pas": self.step_list,
                "Prix Tree trinomial": list_price_step_comparison,
                "Temps pricing": list_time_step_comparison,
                "Différence B&S": list_diff_bs,
                "Différence * nb pas": list_time_gap_bs,
            }
        )

    @staticmethod
    def calculate_step(step, bs_price, market_data, option):
        now = time.time()
        tree_step_comparison = Tree(step, market_data, option)
        tree_step_comparison.price_option()
        price_tree_step_comparison = tree_step_comparison.option_price
        then = time.time()
        pricing_time = then - now
        print(
            f"Taux d'intérêt: {market_data.interest_rate}, Prix option : {price_tree_step_comparison}, Pricing Time: {pricing_time:.4f}s"
        )

        if price_tree_step_comparison is None:
            raise ValueError("Option price not calculated")

        return (
            step,
            price_tree_step_comparison,
            pricing_time,
            price_tree_step_comparison - bs_price,
            price_tree_step_comparison * step,
        )

    def graph_rate_temps_pas(self):
        fig = go.Figure()

        sorted_df = self.results_df.sort_values("""Taux d'intérêt""", ascending=True)

        fig.add_trace(
            go.Scatter(
                x=sorted_df["Taux d'intérêt"],
                y=sorted_df["Différence B&S"],
                mode="lines+markers",
                line=dict(color="blue", width=2),
                marker=dict(size=6),
            )
        )

        fig.add_hline(y=0, line_dash="dash", line_color="red")

        fig.update_layout(
            title="""Différence par rapport au pricing B&S en fonction du taux d'intérêt""",
            xaxis_title="""Taux d'intérêt""",
            yaxis_title="Différence B&S",
        )

        return fig
