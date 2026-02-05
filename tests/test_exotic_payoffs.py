import datetime as dt
from typing import Any
import numpy as np

from trinomial_tree.pricing import MarketData, Option, InductiveTree
from trinomial_tree.pricing.enums import OptionPayoffType


class TestExoticPayoffs:
    def _calculate_price(self, **kwargs: Any) -> float:
        # Default parameters
        params: dict[str, Any] = {
            "start_date": dt.date(2025, 1, 1),
            "maturity_date": dt.date(2026, 1, 1),
            "spot_price": 100,
            "volatility": 0.2,
            "strike_price": 100,
            "interest_rate": 0.05,
            "dividend_amount": 0,
            "payoff_type": OptionPayoffType.vanilla,
            "is_call": True,
        }
        params.update(kwargs)

        market_data = MarketData(
            start_date=params["start_date"],
            spot_price=params["spot_price"],
            volatility=params["volatility"],
            interest_rate=params["interest_rate"],
            discount_rate=params["interest_rate"],
            dividend_ex_date=params["start_date"] + dt.timedelta(days=1),
            dividend_amount=params["dividend_amount"],
        )
        option = Option(
            maturity=params["maturity_date"],
            strike_price=params["strike_price"],
            barrier=None,
            is_american=False,
            is_call=params["is_call"],
            pricing_date=params["start_date"],
            payoff_type=params["payoff_type"],
        )

        tree = InductiveTree(num_steps=200, market_data=market_data, option=option)
        return tree.price()

    def test_digital_call_price_bounds(self) -> None:
        """Test that digital call price is between 0 and discount factor."""
        price = self._calculate_price(
            payoff_type=OptionPayoffType.digital,
            is_call=True,
            interest_rate=0.05,
            maturity_date=dt.date(2026, 1, 1),
            start_date=dt.date(2025, 1, 1),
        )
        discount_factor = np.exp(-0.05 * 1.0)
        assert 0.0 <= price <= discount_factor + 1e-6

    def test_digital_put_price_bounds(self) -> None:
        """Test that digital put price is between 0 and discount factor."""
        price = self._calculate_price(
            payoff_type=OptionPayoffType.digital, is_call=False, interest_rate=0.05
        )
        discount_factor = np.exp(-0.05 * 1.0)
        assert 0.0 <= price <= discount_factor + 1e-6

    def test_digital_call_deep_itm(self) -> None:
        """Test that deep ITM digital call approaches discount factor."""
        price = self._calculate_price(
            payoff_type=OptionPayoffType.digital,
            is_call=True,
            spot_price=200,
            strike_price=100,
            interest_rate=0.05,
        )
        discount_factor = np.exp(-0.05 * 1.0)
        assert np.isclose(price, discount_factor, atol=0.05)

    def test_digital_call_deep_otm(self) -> None:
        """Test that deep OTM digital call approaches 0."""
        price = self._calculate_price(
            payoff_type=OptionPayoffType.digital,
            is_call=True,
            spot_price=50,
            strike_price=100,
        )
        assert np.isclose(price, 0.0, atol=0.01)

    def test_asset_or_nothing_call_deep_itm(self) -> None:
        """Test that deep ITM asset-or-nothing call approaches Spot * exp(-qT) (here q=0 so Spot)."""
        price = self._calculate_price(
            payoff_type=OptionPayoffType.asset_or_nothing,
            is_call=True,
            spot_price=100,
            strike_price=0.01,  # Deep ITM
            interest_rate=0.05,
        )
        assert np.isclose(price, 100.0, atol=1.0)

    def test_asset_or_nothing_put_deep_itm(self) -> None:
        """Test that deep ITM asset-or-nothing put approaches Spot Price."""
        price = self._calculate_price(
            payoff_type=OptionPayoffType.asset_or_nothing,
            is_call=False,
            spot_price=100,
            strike_price=10000,  # Deep ITM
            interest_rate=0.05,
        )
        assert np.isclose(price, 100.0, atol=1.0)

    def test_parity_relationship(self) -> None:
        """
        Test Call/Put Parity for Digital Options.
        Digital Call + Digital Put = Pays 1 always = Discount Factor.
        """
        call_price = self._calculate_price(
            payoff_type=OptionPayoffType.digital,
            is_call=True,
            spot_price=100,
            strike_price=100,
            interest_rate=0.05,
        )
        put_price = self._calculate_price(
            payoff_type=OptionPayoffType.digital,
            is_call=False,
            spot_price=100,
            strike_price=100,
            interest_rate=0.05,
        )
        discount_factor = np.exp(-0.05 * 1.0)
        assert np.isclose(call_price + put_price, discount_factor, atol=0.01)

    def test_asset_or_nothing_parity(self) -> None:
        """
        Asset-or-Nothing Call + Asset-or-Nothing Put = Pays S_T always = S_0 * exp(-qT).
        """
        call_price = self._calculate_price(
            payoff_type=OptionPayoffType.asset_or_nothing,
            is_call=True,
            spot_price=100,
            strike_price=100,
            interest_rate=0.05,
        )
        put_price = self._calculate_price(
            payoff_type=OptionPayoffType.asset_or_nothing,
            is_call=False,
            spot_price=100,
            strike_price=100,
            interest_rate=0.05,
        )
        # q=0
        assert np.isclose(call_price + put_price, 100.0, atol=0.5)
