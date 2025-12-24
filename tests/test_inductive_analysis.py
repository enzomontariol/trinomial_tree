import pytest
import numpy as np
import datetime as dt
from src.pricing.market import MarketData
from src.pricing.option import Option
from src.pricing.inductive_tree import InductiveTree


class TestInductiveAnalysis:
    @pytest.fixture
    def market_data(self):
        return MarketData(
            start_date=dt.date(2024, 1, 1),
            spot_price=100,
            volatility=0.2,
            interest_rate=0.05,
            discount_rate=0.05,
            dividend_ex_date=dt.date(2025, 1, 1),
            dividend_amount=0,
        )

    @pytest.fixture
    def american_put(self):
        return Option(
            maturity=dt.date(2025, 1, 1),
            strike_price=100,
            is_call=False,
            is_american=True,
            pricing_date=dt.date(2024, 1, 1),
        )

    def test_terminal_distribution_integrity(self, market_data, american_put):
        """Test that the terminal probability distribution sums to 1 and matches forward price."""
        N = 50
        tree = InductiveTree(num_steps=N, market_data=market_data, option=american_put)
        df = tree.get_terminal_distribution()

        # 1. Sum of probabilities must be 1
        total_prob = df["Probability"].sum()
        assert np.isclose(total_prob, 1.0, atol=1e-9)

        # 2. Expected value should match Forward Price (approx)
        # E[S_T] = S_0 * exp(rT)
        expected_spot_tree = (df["Spot"] * df["Probability"]).sum()

        T = tree.get_time_to_maturity()
        expected_spot_analytical = market_data.spot_price * np.exp(
            market_data.interest_rate * T
        )

        # Trinomial tree converges to this, allow some error for N=50
        assert np.isclose(expected_spot_tree, expected_spot_analytical, rtol=1e-2)

    def test_exercise_boundary_shape(self, market_data, american_put):
        """Test that the exercise boundary for an American Put exists and behaves rationally."""
        N = 100
        tree = InductiveTree(num_steps=N, market_data=market_data, option=american_put)
        boundary = tree.get_exercise_boundary()

        assert not boundary.empty
        assert "Boundary_Spot" in boundary.columns

        # For an American Put, the exercise boundary should generally increase as we approach maturity
        # (Option value decays, so we are willing to exercise at higher spots closer to expiry)
        # However, near T=0 (Step 0), it starts low.

        # Check that we have data points
        assert len(boundary) > 0

        # Check that boundary is always below Strike (for Put)
        assert (boundary["Boundary_Spot"] < american_put.strike_price).all()

    def test_european_no_boundary(self, market_data, american_put):
        """European options should have no early exercise boundary."""
        european_put = american_put
        european_put.is_american = False

        tree = InductiveTree(num_steps=50, market_data=market_data, option=european_put)
        boundary = tree.get_exercise_boundary()

        assert boundary.empty
