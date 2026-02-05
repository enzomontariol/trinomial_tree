import numpy as np
import datetime as dt

from trinomial_tree.pricing.market import MarketData
from trinomial_tree.pricing.option import Option
from trinomial_tree.pricing.barrier import Barrier
from trinomial_tree.pricing.enums import BarrierType, BarrierDirection
from trinomial_tree.pricing.inductive_tree import InductiveTree
from trinomial_tree.pricing.black_scholes import BlackScholes


def get_bs_price(S, K, T, r, sigma):
    """
    Wrapper to use the project's BlackScholes class with float parameters.
    Assumes 365-day year convention consistent with the tests.
    """
    today = dt.date.today()
    days_to_maturity = int(round(T * 365))
    maturity = today + dt.timedelta(days=days_to_maturity)

    market_data = MarketData(
        start_date=today,
        spot_price=S,
        volatility=sigma,
        interest_rate=r,
        discount_rate=r,
        dividend_ex_date=today,  # Irrelevant for 0 dividend
        dividend_amount=0.0,
    )

    option = Option(
        maturity=maturity,
        strike_price=K,
        is_american=False,
        is_call=True,
        pricing_date=today,
        calendar_base_convention=365,
    )

    return BlackScholes(market_data, option).price()


def analytical_down_out_call(S, K, H, T, r, sigma):
    """
    Price of a Down-and-Out Call option.
    Assumes H <= K (Barrier is below Strike).
    """
    if S <= H:
        return 0.0

    lambda_ = (r + 0.5 * sigma**2) / sigma**2

    # Vanilla Price
    c_bs = get_bs_price(S, K, T, r, sigma)

    # Barrier Adjustment
    # Formula H <= K: c_do = c_bs(S,K) - (H/S)^(2*lambda) * c_bs(H^2/S, K)

    adjustment = (H / S) ** (2 * lambda_) * get_bs_price(H**2 / S, K, T, r, sigma)
    return c_bs - adjustment


def test_down_and_out_call_analytical_convergence():
    """
    Validates the InductiveTree price against the analytical formula for a Down-and-Out Call.
    Case: H < K.
    """
    # Parameters
    S0 = 100.0
    K = 100.0
    H = 95.0  # Barrier < Strike
    r = 0.05
    sigma = 0.2
    T_days = 365
    maturity_date = dt.date.today() + dt.timedelta(days=T_days)

    # Analytical Price
    T = T_days / 365.0
    analytic_price = analytical_down_out_call(S0, K, H, T, r, sigma)

    # Updated MarketData initialization
    market_data = MarketData(
        start_date=dt.date.today(),
        spot_price=S0,
        volatility=sigma,
        interest_rate=r,
        discount_rate=r,
        dividend_ex_date=dt.date.today() + dt.timedelta(days=T_days + 10),
        dividend_amount=0.0,
    )
    barrier = Barrier(
        barrier_level=H,
        barrier_type=BarrierType.knock_out,
        barrier_direction=BarrierDirection.down,
    )
    option = Option(
        maturity=maturity_date,
        strike_price=K,
        barrier=barrier,
        is_american=False,  # Analytical formula is for European
        is_call=True,
        pricing_date=dt.date.today(),
    )

    # Using a large number of steps for convergence check
    # Note: Tree is discrete monitoring, analytical is continuous.
    # Discrete barrier options are worth more than continuous ones (less likely to hit).
    # We need large N to approach continuous limit. A correction factor method would be better.
    N = 800
    tree = InductiveTree(num_steps=N, market_data=market_data, option=option)
    tree_price = tree.price()

    print(f"\nAnalytical DOC Price: {analytic_price:.4f}")
    print(f"Tree DOC Price (N={N}): {tree_price:.4f}")

    # Tolerance - Trees can oscillate, but should be close.
    # 0.5% relative error or absolute error check
    assert np.isclose(tree_price, analytic_price, rtol=0.01, atol=0.05), (
        f"Tree price {tree_price} differs from analytical {analytic_price}"
    )


def test_barrier_far_away_converges_to_vanilla():
    """
    If the barrier is very far away (e.g., Down-and-Out with H very low),
    the price should match the Vanilla Black-Scholes price.
    """
    S0 = 100.0
    K = 100.0
    H = 10.0  # Very low barrier
    r = 0.05
    sigma = 0.2
    maturity_date = dt.date.today() + dt.timedelta(days=365)

    market_data = MarketData(
        start_date=dt.date.today(),
        spot_price=S0,
        volatility=sigma,
        interest_rate=r,
        discount_rate=r,
        dividend_ex_date=dt.date.today() + dt.timedelta(days=400),
        dividend_amount=0.0,
    )
    barrier = Barrier(
        barrier_level=H,
        barrier_type=BarrierType.knock_out,
        barrier_direction=BarrierDirection.down,
    )
    option = Option(
        maturity=maturity_date,
        strike_price=K,
        barrier=barrier,  # Effectively no barrier
        is_american=False,
        is_call=True,
    )

    N = 100
    tree = InductiveTree(num_steps=N, market_data=market_data, option=option)
    tree_price = tree.price()

    # Use the library's BlackScholes pricer for consistency check
    # Note: BlackScholes pricer calculates Vanilla price, ignoring the barrier attribute
    bs_pricer = BlackScholes(market_data, option)
    bs_price = bs_pricer.price()

    print(f"\nVanilla BS Price (Internal): {bs_price:.4f}")
    print(f"Tree DOC (Far Barrier) Price: {tree_price:.4f}")

    assert np.isclose(tree_price, bs_price, rtol=0.01, atol=0.05)


def test_barrier_breached_at_start():
    """
    If S0 is already beyond the barrier (Knock-Out), price should be 0.
    """
    S0 = 90.0
    H = 95.0  # S0 < H (Down-and-Out would mean S0 is already out? No wait.)
    # Down-and-Out: Knock out if S <= H.
    # Here S0=90, H=95. So S0 <= H. Should be knocked out immediately.

    match_barrier = Barrier(
        barrier_level=H,
        barrier_type=BarrierType.knock_out,
        barrier_direction=BarrierDirection.down,
    )
    option = Option(
        maturity=dt.date.today() + dt.timedelta(days=365),
        strike_price=100.0,
        barrier=match_barrier,
        is_call=True,
    )
    market_data = MarketData(
        start_date=dt.date.today(),
        spot_price=S0,
        volatility=0.2,
        interest_rate=0.05,
        discount_rate=0.05,
        dividend_ex_date=dt.date.today() + dt.timedelta(days=400),
        dividend_amount=0.0,
    )

    tree = InductiveTree(num_steps=50, market_data=market_data, option=option)
    price = tree.price()

    assert price == 0.0, f"Expected 0.0 for already breached barrier, got {price}"
