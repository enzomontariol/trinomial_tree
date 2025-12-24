import datetime as dt
from typing import Any
import pytest
import numpy as np

from src.pricing import MarketData, Option, InductiveTree, Barrier
from src.pricing.enums import BarrierType, BarrierDirection


class TestSensitivityMonotony:
    def _calculate_price(self, **kwargs: float | dt.date) -> float:
        # Default parameters
        params: dict[str, Any] = {
            "start_date": dt.date(2025, 1, 1),
            "maturity_date": dt.date(2026, 1, 1),
            "spot_price": 100,
            "volatility": 0.2,
            "strike_price": 100,
            "interest_rate": 0.05,
            "dividend_amount": 0,
        }
        params.update(kwargs)

        market_data = MarketData(
            start_date=params["start_date"],
            spot_price=params["spot_price"],
            volatility=params["volatility"],
            interest_rate=params["interest_rate"],
            discount_rate=params[
                "interest_rate"
            ],  # assuming discount rate equals interest rate
            dividend_ex_date=params["start_date"] + dt.timedelta(days=1),
            dividend_amount=params["dividend_amount"],
        )
        option = Option(
            maturity=params["maturity_date"],
            strike_price=params["strike_price"],
            barrier=None,
            is_american=False,
            is_call=True,
            pricing_date=params["start_date"],
        )

        tree = InductiveTree(num_steps=500, market_data=market_data, option=option)
        return tree.price()

    @pytest.mark.parametrize(
        "strike1, strike2", [(80 + 20 * i, 80 + 20 * (i + 1)) for i in range(9)]
    )
    def test_monotonicity_with_respect_to_strike(
        self, strike1: float, strike2: float
    ) -> None:
        """Test that european call option prices decrease as strike prices increase."""
        price1 = self._calculate_price(strike_price=strike1)
        price2 = self._calculate_price(strike_price=strike2)
        assert price2 <= price1, (
            f"Price did not decrease with increasing strike: {strike1}->{strike2}, {price1}->{price2}"
        )

    @pytest.mark.parametrize(
        "vol1, vol2", [(0.1 + 0.05 * i, 0.1 + 0.05 * (i + 1)) for i in range(9)]
    )
    def test_monotonicity_with_respect_to_volatility(
        self, vol1: float, vol2: float
    ) -> None:
        """Test that european call option prices increase as volatility increases."""
        price1 = self._calculate_price(volatility=vol1)
        price2 = self._calculate_price(volatility=vol2)
        assert price2 >= price1, (
            f"Price did not increase with increasing volatility: {vol1}->{vol2}, {price1}->{price2}"
        )

    @pytest.mark.parametrize(
        "maturity1, maturity2",
        [
            (
                dt.date(2025, 1, 1) + dt.timedelta(days=30 * i),
                dt.date(2025, 1, 1) + dt.timedelta(days=30 * (i - 1)),
            )
            for i in range(12, 1, -1)
        ],
    )
    def test_monotonicity_with_respect_to_time(
        self, maturity1: dt.date, maturity2: dt.date
    ) -> None:
        """Test that european call option prices decrease as time to maturity decreases."""
        # maturity1 is larger (e.g. 12 months), maturity2 is smaller (e.g. 11 months)
        price1 = self._calculate_price(maturity_date=maturity1)
        price2 = self._calculate_price(maturity_date=maturity2)
        assert price2 <= price1, (
            f"Price did not decrease with decreasing maturity: {maturity1}->{maturity2}, {price1}->{price2}"
        )

    @pytest.mark.parametrize(
        "rate1, rate2", [(0.0 + 0.01 * i, 0.0 + 0.01 * (i + 1)) for i in range(10)]
    )
    def test_monotonicity_with_respect_to_interest_rate(
        self, rate1: float, rate2: float
    ) -> None:
        """Test that european call option prices increase as interest rates increase."""

        price1 = self._calculate_price(interest_rate=rate1)
        price2 = self._calculate_price(interest_rate=rate2)
        assert price2 >= price1, (
            f"Price did not increase with increasing interest rate: {rate1}->{rate2}, {price1}->{price2}"
        )

    @pytest.mark.parametrize(
        "div1, div2", [(0 + 1 * i, 0 + 1 * (i + 1)) for i in range(10)]
    )
    def test_monotonicity_with_respect_to_dividend(
        self, div1: float, div2: float
    ) -> None:
        """Test that european call option prices decrease as dividends increase."""

        price1 = self._calculate_price(dividend_amount=div1)
        price2 = self._calculate_price(dividend_amount=div2)
        assert price2 <= price1, (
            f"Price did not decrease with increasing dividend: {div1}->{div2}, {price1}->{price2}"
        )


@pytest.mark.parametrize("is_call", [True, False])
@pytest.mark.parametrize("strike_price", [80, 100, 120])
@pytest.mark.parametrize("volatility", [0.1, 0.2, 0.4])
@pytest.mark.parametrize("interest_rate", [0.0, 0.05, 0.1])
@pytest.mark.parametrize("dividend_amount", [0, 1, 2])
class TestFinancialSanitaryCheck:
    def test_put_call_parity(
        self,
        is_call: bool,
        strike_price: float,
        volatility: float,
        interest_rate: float,
        dividend_amount: float,
    ) -> None:
        """Test that European put-call parity holds for our Trinomial Tree pricer."""

        start_date = dt.date(2025, 1, 1)
        maturity_date = dt.date(2026, 1, 1)
        spot_price = 100

        market_data = MarketData(
            start_date=start_date,
            spot_price=spot_price,
            volatility=volatility,
            interest_rate=interest_rate,
            discount_rate=interest_rate,  # assuming discount rate equals interest rate
            dividend_ex_date=start_date + dt.timedelta(days=1),
            dividend_amount=dividend_amount,
        )

        call_option = Option(
            maturity=maturity_date,
            strike_price=strike_price,
            barrier=None,
            is_american=False,
            is_call=True,
            pricing_date=start_date,
        )

        put_option = Option(
            maturity=maturity_date,
            strike_price=strike_price,
            barrier=None,
            is_american=False,
            is_call=False,
            pricing_date=start_date,
        )

        tree_call = InductiveTree(
            num_steps=100, market_data=market_data, option=call_option
        )
        tree_put = InductiveTree(
            num_steps=100, market_data=market_data, option=put_option
        )

        call_price = tree_call.price()
        put_price = tree_put.price()

        # Present value of strike price
        days_to_maturity = (maturity_date - start_date).days
        pv_strike = strike_price * np.exp(
            -interest_rate * days_to_maturity / call_option.calendar_base_convention
        )

        # Present value of dividends
        days_to_dividend = (market_data.dividend_ex_date - start_date).days
        pv_dividends = dividend_amount * np.exp(
            -interest_rate * days_to_dividend / call_option.calendar_base_convention
        )

        # Put-Call Parity: C - P = S - PV(K) - PV(Dividends)
        lhs = call_price - put_price
        rhs = spot_price - pv_strike - pv_dividends

        assert abs(lhs - rhs) < 1e-2, (
            f"Put-Call Parity does not hold: LHS={lhs:.4f}, RHS={rhs:.4f}"
        )

    def test_american_european_relationship(
        self,
        is_call: bool,
        strike_price: float,
        volatility: float,
        interest_rate: float,
        dividend_amount: float,
    ) -> None:
        """Test that American option prices are at least as high as European option prices."""

        start_date = dt.date(2025, 1, 1)
        maturity_date = dt.date(2026, 1, 1)
        spot_price = 100

        market_data = MarketData(
            start_date=start_date,
            spot_price=spot_price,
            volatility=volatility,
            interest_rate=interest_rate,
            discount_rate=interest_rate,  # assuming discount rate equals interest rate
            dividend_ex_date=start_date + dt.timedelta(days=1),
            dividend_amount=dividend_amount,
        )

        european_option = Option(
            maturity=maturity_date,
            strike_price=strike_price,
            barrier=None,
            is_american=False,
            is_call=is_call,
            pricing_date=start_date,
        )

        american_option = Option(
            maturity=maturity_date,
            strike_price=strike_price,
            barrier=None,
            is_american=True,
            is_call=is_call,
            pricing_date=start_date,
        )

        tree_european = InductiveTree(
            num_steps=100, market_data=market_data, option=european_option
        )
        tree_american = InductiveTree(
            num_steps=100, market_data=market_data, option=american_option
        )

        european_price = tree_european.price()
        american_price = tree_american.price()

        assert american_price >= european_price, (
            f"American option price {american_price:.4f} is less than European option price {european_price:.4f}"
        )

        # Specific case: American Call on non-dividend paying stock should equal European Call
        if is_call and dividend_amount == 0 and interest_rate >= 0:
            assert abs(american_price - european_price) < 1e-3, (
                f"American Call (No Div) should equal European Call: Amer={american_price:.4f}, Euro={european_price:.4f}"
            )

    def test_no_arbitrage_bounds(
        self,
        is_call: bool,
        strike_price: float,
        volatility: float,
        interest_rate: float,
        dividend_amount: float,
    ) -> None:
        """Test that option prices respect no-arbitrage bounds."""
        start_date = dt.date(2025, 1, 1)
        maturity_date = dt.date(2026, 1, 1)
        spot_price = 100

        market_data = MarketData(
            start_date=start_date,
            spot_price=spot_price,
            volatility=volatility,
            interest_rate=interest_rate,
            discount_rate=interest_rate,
            dividend_ex_date=start_date + dt.timedelta(days=1),
            dividend_amount=dividend_amount,
        )

        option = Option(
            maturity=maturity_date,
            strike_price=strike_price,
            barrier=None,
            is_american=False,
            is_call=is_call,
            pricing_date=start_date,
        )

        tree = InductiveTree(num_steps=100, market_data=market_data, option=option)
        price = tree.price()

        # Calculate Present Values
        days_to_maturity = (maturity_date - start_date).days
        pv_strike = strike_price * np.exp(
            -interest_rate * days_to_maturity / option.calendar_base_convention
        )

        days_to_dividend = (market_data.dividend_ex_date - start_date).days
        pv_dividends = dividend_amount * np.exp(
            -interest_rate * days_to_dividend / option.calendar_base_convention
        )

        if is_call:
            # Call Bounds: max(0, S - PV(D) - PV(K)) <= C <= S
            lower_bound = max(0, spot_price - pv_dividends - pv_strike)
            upper_bound = spot_price
            assert lower_bound <= price <= upper_bound, (
                f"Call price {price:.4f} out of bounds [{lower_bound:.4f}, {upper_bound:.4f}]"
            )
        else:
            # Put Bounds: max(0, PV(K) - (S - PV(D))) <= P <= PV(K)
            lower_bound = max(0, pv_strike - (spot_price - pv_dividends))
            upper_bound = pv_strike
            assert lower_bound <= price <= upper_bound, (
                f"Put price {price:.4f} out of bounds [{lower_bound:.4f}, {upper_bound:.4f}]"
            )

    def test_zero_volatility(
        self,
        is_call: bool,
        strike_price: float,
        volatility: float,
        interest_rate: float,
        dividend_amount: float,
    ) -> None:
        """Test that with zero volatility, the option price corresponds to the intrinsic value of the forward.
        S_T = (S_0 - PV(Divs)) * exp(rT)
        """
        # We use a very small volatility because the trinomial tree has a singularity at vol=0
        test_vol = 1e-5

        start_date = dt.date(2025, 1, 1)
        maturity_date = dt.date(2026, 1, 1)
        spot_price = 100

        market_data = MarketData(
            start_date=start_date,
            spot_price=spot_price,
            volatility=test_vol,
            interest_rate=interest_rate,
            discount_rate=interest_rate,
            dividend_ex_date=start_date + dt.timedelta(days=1),
            dividend_amount=dividend_amount,
        )

        option = Option(
            maturity=maturity_date,
            strike_price=strike_price,
            barrier=None,
            is_american=False,
            is_call=is_call,
            pricing_date=start_date,
        )

        tree = InductiveTree(num_steps=100, market_data=market_data, option=option)
        price = tree.price()

        days_to_maturity = (maturity_date - start_date).days
        T = days_to_maturity / option.calendar_base_convention

        # PV of dividends
        days_to_dividend = (market_data.dividend_ex_date - start_date).days
        t_div = days_to_dividend / option.calendar_base_convention
        pv_dividends = dividend_amount * np.exp(-interest_rate * t_div)

        # Forward price: (S0 - PV(D)) * e^(rT)
        forward_price = (spot_price - pv_dividends) * np.exp(interest_rate * T)

        if is_call:
            intrinsic_at_maturity = max(forward_price - strike_price, 0)
        else:
            intrinsic_at_maturity = max(strike_price - forward_price, 0)

        expected_price = intrinsic_at_maturity * np.exp(-interest_rate * T)

        assert abs(price - expected_price) < 1e-2, (
            f"Price {price:.4f} does not match deterministic price {expected_price:.4f} with vol={test_vol}"
        )


class TestBarrierImpact:
    @pytest.mark.parametrize(
        "barrier_type, barrier_direction, barrier_level",
        [
            (BarrierType.knock_out, BarrierDirection.up, 120),
            (BarrierType.knock_in, BarrierDirection.up, 120),
            (BarrierType.knock_out, BarrierDirection.down, 80),
            (BarrierType.knock_in, BarrierDirection.down, 80),
        ],
    )
    def test_barrier_impact(
        self,
        barrier_type: BarrierType,
        barrier_direction: BarrierDirection,
        barrier_level: float,
    ) -> None:
        """Test that adding a barrier affects the option price.
        Everything being equal the barrier option price should be lower than the vanilla option price."""
        start_date = dt.date(2025, 1, 1)
        maturity_date = dt.date(2026, 1, 1)
        spot_price = 100
        strike_price = 100
        volatility = 0.2
        interest_rate = 0.05

        market_data = MarketData(
            start_date=start_date,
            spot_price=spot_price,
            volatility=volatility,
            interest_rate=interest_rate,
            discount_rate=interest_rate,
            dividend_ex_date=start_date + dt.timedelta(days=1),
            dividend_amount=0,
        )

        # Vanilla Option
        vanilla_option = Option(
            maturity=maturity_date,
            strike_price=strike_price,
            barrier=None,
            is_american=False,
            is_call=True,
            pricing_date=start_date,
        )

        tree_vanilla = InductiveTree(
            num_steps=100, market_data=market_data, option=vanilla_option
        )
        vanilla_price = tree_vanilla.price()

        # Barrier Option
        barrier = Barrier(
            barrier_level=barrier_level,
            barrier_type=barrier_type,
            barrier_direction=barrier_direction,
        )

        barrier_option = Option(
            maturity=maturity_date,
            strike_price=strike_price,
            barrier=barrier,
            is_american=False,
            is_call=True,
            pricing_date=start_date,
        )

        tree_barrier = InductiveTree(
            num_steps=100, market_data=market_data, option=barrier_option
        )
        barrier_price = tree_barrier.price()

        assert barrier_price < vanilla_price, (
            f"Barrier option price ({barrier_price:.4f}) should be lower than vanilla option price ({vanilla_price:.4f}) "
            f"for {barrier_type.value} {barrier_direction.value} barrier at {barrier_level}"
        )
