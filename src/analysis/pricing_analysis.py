import sys
import plotly.graph_objects as go
import warnings

from ..pricing.tree_node import Tree
from ..pricing.market import MarketData
from ..pricing.option import Option
from ..pricing.black_scholes import BlackScholes
from .convergence_analysis import ConvergenceAnalysis

warnings.filterwarnings("ignore")

sys.setrecursionlimit(1000000000)


class TreeVsBsAnalysis(ConvergenceAnalysis):
    """Base class for Tree vs Black-Scholes analysis."""

    def __init__(self, max_cpu, step_list, market_data, option):
        super().__init__(
            model_class=Tree,
            benchmark_class=BlackScholes,
            convergence_param_name="num_steps",
            max_cpu=max_cpu,
            convergence_values=step_list,
            market_data=market_data,
            option=option,
            x_axis_name="Nombre de pas",
            model_price_name="Prix Tree trinomial",
            pricing_time_name="Temps pricing",
            diff_benchmark_name="Différence B&S",
        )


class BsComparison(TreeVsBsAnalysis):
    """Analyzes pricing convergence for a single configuration."""

    def __init__(
        self,
        max_cpu: int,
        step_list: list,
        epsilon_values: list,
        market_data: MarketData,
        option: Option,
    ):
        """Initializes the BsComparison analysis.

        Args:
            max_cpu (int): Maximum number of CPU cores to use.
            step_list (list): List of step counts to analyze.
            epsilon_values (list): List of epsilon values to test.
            market_data (MarketData): Market data object.
            option (Option): Option object.
        """
        super().__init__(max_cpu, step_list, market_data, option)
        self.epsilon_values = epsilon_values
        self._run_analysis(self.epsilon_values, "Epsilon (1e-)")

    def _get_config(self, value):
        """Returns configuration with modified epsilon."""
        return self.market_data, self.option, {"epsilon": value}

    def bs_graph_temps_pas(self):
        """Generates a graph of pricing time vs number of steps.

        Returns:
            go.Figure: Plotly figure object.
        """
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
        """Generates a graph of tree price vs number of steps.

        Returns:
            go.Figure: Plotly figure object.
        """
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

        bs_price = BlackScholes(self.market_data, self.option).price()

        fig.add_hline(
            y=bs_price,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Prix B&S: {round(bs_price, 4)}",
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
        """Generates a graph of tree price vs number of steps for low epsilon values.

        Returns:
            go.Figure: Plotly figure object.
        """
        fig = go.Figure()

        mask = [
            (value[0] < 1e-7)
            if isinstance(value, list) and len(value) > 0
            else (value < 1e-7)
            for value in self.results_df["Epsilon (1e-)"]
        ]

        filtered_df = self.results_df[mask]

        unique_epsilons = filtered_df["Epsilon (1e-)"].unique()
        bs_price = BlackScholes(self.market_data, self.option).price()

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
            y=bs_price,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Prix B&S: {round(bs_price, 4)}",
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
        """Generates a graph of tree price vs number of steps for high epsilon values.

        Returns:
            go.Figure: Plotly figure object.
        """
        fig = go.Figure()

        mask = [
            (value[0] > 1e-7)
            if isinstance(value, list) and len(value) > 0
            else (value > 1e-7)
            for value in self.results_df["Epsilon (1e-)"]
        ]

        filtered_df = self.results_df[mask]

        unique_epsilons = filtered_df["Epsilon (1e-)"].unique()
        bs_price = BlackScholes(self.market_data, self.option).price()

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
            y=bs_price,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Prix B&S: {round(bs_price, 4)}",
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
        """Generates a graph of pricing time vs epsilon for a fixed number of steps (5000).

        Returns:
            go.Figure: Plotly figure object.
        """
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


class StrikeComparison(TreeVsBsAnalysis):
    """Analyzes pricing convergence across different strike prices."""

    def __init__(
        self,
        max_cpu: int,
        step_list: list,
        strike_values: list,
        market_data: MarketData,
        base_option: Option,
    ):
        """Initializes the StrikeComparison analysis.

        Args:
            max_cpu (int): Maximum number of CPU cores to use.
            step_list (list): List of step counts to analyze.
            strike_values (list): List of strike prices to test.
            market_data (MarketData): Market data object.
            base_option (Option): Base option object (strike will be varied).
        """
        super().__init__(max_cpu, step_list, market_data, base_option)
        self.strike_values = strike_values
        self._run_analysis(self.strike_values, "Strike")

    def _get_config(self, value):
        """Returns configuration with modified strike price."""
        option = Option(
            maturity=self.option.maturity,
            strike_price=value,
            barrier=self.option.barrier,
            is_american=self.option.is_american,
            is_call=self.option.is_call,
            pricing_date=self.option.pricing_date,
            calendar_base_convention=self.option.calendar_base_convention,
        )
        return self.market_data, option, {}

    def graph_strike_temps_pas(self):
        """Generates a graph of price difference vs strike.

        Returns:
            go.Figure: Plotly figure object.
        """
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


class VolComparison(TreeVsBsAnalysis):
    """Analyzes pricing convergence across different volatilities."""

    def __init__(
        self,
        max_cpu: int,
        step_list: list,
        vol_values: list,
        base_market_data: MarketData,
        option: Option,
    ):
        """Initializes the VolComparison analysis.

        Args:
            max_cpu (int): Maximum number of CPU cores to use.
            step_list (list): List of step counts to analyze.
            vol_values (list): List of volatility values to test.
            base_market_data (MarketData): Base market data object (volatility will be varied).
            option (Option): Option object.
        """
        super().__init__(max_cpu, step_list, base_market_data, option)
        self.vol_values = vol_values
        self._run_analysis(self.vol_values, "Volatilité")

    def _get_config(self, value):
        """Returns configuration with modified volatility."""
        market_data = MarketData(
            start_date=self.market_data.start_date,
            spot_price=self.market_data.spot_price,
            volatility=value,
            interest_rate=self.market_data.interest_rate,
            discount_rate=self.market_data.discount_rate,
            dividend_ex_date=self.market_data.dividend_ex_date,
            dividend_amount=self.market_data.dividend_amount,
        )
        return market_data, self.option, {}

    def graph_vol_temps_pas(self):
        """Generates a graph of price difference vs volatility.

        Returns:
            go.Figure: Plotly figure object.
        """
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


class RateComparison(TreeVsBsAnalysis):
    """Analyzes pricing convergence across different interest rates."""

    def __init__(
        self,
        max_cpu: int,
        step_list: list,
        rate_values: list,
        base_market_data: MarketData,
        option: Option,
    ):
        """Initializes the RateComparison analysis.

        Args:
            max_cpu (int): Maximum number of CPU cores to use.
            step_list (list): List of step counts to analyze.
            rate_values (list): List of interest rate values to test.
            base_market_data (MarketData): Base market data object (rate will be varied).
            option (Option): Option object.
        """
        super().__init__(max_cpu, step_list, base_market_data, option)
        self.rate_values = rate_values
        self._run_analysis(self.rate_values, """Taux d'intérêt""")

    def _get_config(self, value):
        """Returns configuration with modified interest rate."""
        market_data = MarketData(
            start_date=self.market_data.start_date,
            spot_price=self.market_data.spot_price,
            volatility=self.market_data.volatility,
            interest_rate=value,
            discount_rate=value,
            dividend_ex_date=self.market_data.dividend_ex_date,
            dividend_amount=self.market_data.dividend_amount,
        )
        return market_data, self.option, {}

    def graph_rate_temps_pas(self):
        """Generates a graph of price difference vs interest rate.

        Returns:
            go.Figure: Plotly figure object.
        """
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
