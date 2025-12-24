import pytest
import datetime as dt
import numpy as np
from src.pricing import MarketData, Option, InductiveTree


class TestEdgeCases:
    @pytest.fixture
    def base_market_data(self):
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
    def base_option(self):
        return Option(
            maturity=dt.date(2025, 1, 1),
            strike_price=100,
            is_call=True,
            is_american=False,
            pricing_date=dt.date(2024, 1, 1),
        )

    def test_deep_in_the_money_call(self, base_market_data, base_option):
        """Deep ITM Call should be worth approx Spot - Strike * exp(-rT)."""
        base_market_data.spot_price = 1000  # Deep ITM
        base_option.strike_price = 10

        tree = InductiveTree(
            num_steps=50, market_data=base_market_data, option=base_option
        )
        price = tree.price()

        T = tree.get_time_to_maturity()
        expected_price = (
            base_market_data.spot_price
            - base_option.strike_price * np.exp(-base_market_data.interest_rate * T)
        )

        assert np.isclose(price, expected_price, rtol=1e-2)

    def test_deep_out_of_the_money_call(self, base_market_data, base_option):
        """Deep OTM Call should be worth approx 0."""
        base_market_data.spot_price = 10  # Deep OTM
        base_option.strike_price = 1000

        tree = InductiveTree(
            num_steps=50, market_data=base_market_data, option=base_option
        )
        price = tree.price()

        assert np.isclose(price, 0.0, atol=1e-4)

    def test_high_volatility(self, base_market_data, base_option):
        """High volatility should increase option value significantly but remain bounded by Spot."""
        base_market_data.volatility = 2.0  # 200% vol

        tree = InductiveTree(
            num_steps=50, market_data=base_market_data, option=base_option
        )
        price = tree.price()

        assert price > 0
        assert price < base_market_data.spot_price  # Call price < Spot

    def test_short_maturity(self, base_market_data, base_option):
        """Very short maturity should converge to intrinsic value."""
        base_option.maturity = base_option.pricing_date + dt.timedelta(days=1)

        tree = InductiveTree(
            num_steps=20, market_data=base_market_data, option=base_option
        )
        price = tree.price()

        intrinsic = max(base_market_data.spot_price - base_option.strike_price, 0)
        # With 1 day, time value is negligible
        assert np.isclose(price, intrinsic, atol=0.5)  # Allow some small time value
