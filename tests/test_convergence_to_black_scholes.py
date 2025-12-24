import datetime as dt
import pytest

from src.pricing import BlackScholes, MarketData, Option, InductiveTree
from src.analysis import EmpiricalGreeks


class TestPriceConvergenceToBlackScholes:
    @pytest.mark.parametrize("strike", [80, 100, 120])
    @pytest.mark.parametrize("volatility", [0.1, 0.2, 0.4])
    @pytest.mark.parametrize("maturity_years", [0.01, 0.25, 0.5, 1.25, 2.0])
    @pytest.mark.parametrize("risk_free_rate", [0.0, 0.05, 0.1])
    @pytest.mark.parametrize("is_call", [True, False])
    def test_price_convergence(
        self,
        strike: float,
        volatility: float,
        maturity_years: float,
        risk_free_rate: float,
        is_call: bool,
    ) -> None:
        """Test the convergence of our Trinomial Tree pricer's price to Black-Scholes for a large number of input parameters."""

        start_date = dt.date(2025, 1, 1)
        days = int(maturity_years * 365)
        maturity_date = start_date + dt.timedelta(days=days)

        market_data = MarketData(
            start_date=start_date,
            spot_price=100,
            volatility=volatility,
            interest_rate=risk_free_rate,  # assuming constant rates
            discount_rate=risk_free_rate,  # assuming discount rate equals interest rate
            dividend_ex_date=dt.date(2025, 1, 2),
            dividend_amount=0,
        )

        option = Option(
            maturity=maturity_date,
            strike_price=strike,
            barrier=None,
            is_american=False,
            is_call=is_call,
            pricing_date=start_date,
        )

        bs = BlackScholes(market_data=market_data, option=option)
        tree = InductiveTree(num_steps=500, market_data=market_data, option=option)

        bs_price = bs.price()
        tree_price = tree.price()

        assert abs(bs_price - tree_price) < 5e-2, (
            f"Prices do not converge: BS={bs_price:.4f}, Tree={tree_price:.4f} "
            f"(S=100, K={strike}, vol={volatility}, T={maturity_years}, Call={is_call})"
        )


class TestGreeksConvergenceToBlackScholes:
    def setup_method(self) -> None:
        self.market_data = MarketData(
            start_date=dt.date(2025, 12, 1),
            spot_price=100,
            volatility=0.2,
            interest_rate=0.05,
            discount_rate=0.05,
            dividend_ex_date=dt.date(2025, 12, 15),
            dividend_amount=0,  # No dividend as per Black-Scholes assumptions
        )
        self.option = Option(
            maturity=dt.date(2026, 12, 1),
            strike_price=100,
            barrier=None,  # Black and Scholes constraint
            is_american=False,  # Black and Scholes constraint
            is_call=True,
            pricing_date=dt.date(2025, 12, 1),
        )

        self.bs = BlackScholes(market_data=self.market_data, option=self.option)
        self.tree = InductiveTree(
            num_steps=500, market_data=self.market_data, option=self.option
        )
        self.empirical_greeks = EmpiricalGreeks(tree=self.tree)

    def test_convergence_to_black_scholes_delta(self) -> None:
        """Test the convergence of our Trinomial Tree pricer's delta to Black-Scholes for a large number of steps."""
        bs_delta = self.bs.delta()
        tree_delta = self.empirical_greeks.approximate_delta()
        assert abs(bs_delta - tree_delta) < 1e-2, (
            f"Deltas do not converge: BS={bs_delta}, Tree={tree_delta}"
        )

    def test_convergence_to_black_scholes_gamma(self) -> None:
        """Test the convergence of our Trinomial Tree pricer's gamma to Black-Scholes for a large number of steps."""
        bs_gamma = self.bs.gamma()
        tree_gamma = self.empirical_greeks.approximate_gamma()
        assert abs(bs_gamma - tree_gamma) < 1e-2, (
            f"Gammas do not converge: BS={bs_gamma}, Tree={tree_gamma}"
        )

    def test_convergence_to_black_scholes_theta(self) -> None:
        """Test the convergence of our Trinomial Tree pricer's theta to Black-Scholes for a large number of steps."""
        bs_theta = self.bs.theta()
        tree_theta = self.empirical_greeks.approximate_theta()
        assert abs(bs_theta - tree_theta) < 1e-2, (
            f"Thetas do not converge: BS={bs_theta}, Tree={tree_theta}"
        )

    def test_convergence_to_black_scholes_rho(self) -> None:
        """Test the convergence of our Trinomial Tree pricer's rho to Black-Scholes for a large number of steps."""
        bs_rho = self.bs.rho()
        tree_rho = self.empirical_greeks.approximate_rho()
        assert abs(bs_rho - tree_rho) < 1e-2, (
            f"Rhos do not converge: BS={bs_rho}, Tree={tree_rho}"
        )
