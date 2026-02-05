import pytest
import time
import datetime as dt
from trinomial_tree.pricing import MarketData, Option, InductiveTree, BlackScholes


class TestPerformance:
    @pytest.fixture
    def standard_setup(self):
        market_data = MarketData(
            start_date=dt.date(2024, 1, 1),
            spot_price=100,
            volatility=0.2,
            interest_rate=0.05,
            discount_rate=0.05,
            dividend_ex_date=dt.date(2025, 1, 1),
            dividend_amount=0,
        )
        option = Option(
            maturity=dt.date(2025, 1, 1),
            strike_price=100,
            is_call=True,
            is_american=False,
            pricing_date=dt.date(2024, 1, 1),
        )
        return market_data, option

    def test_performance_benchmark(self, standard_setup):
        """
        Benchmark Tree vs Black-Scholes.
        Tree will be slower, but we want to ensure it's within reasonable limits for N=500.
        """
        market_data, option = standard_setup

        # Black-Scholes Timing
        bs = BlackScholes(market_data, option)
        start_bs = time.time()
        for _ in range(100):
            bs.price()
        end_bs = time.time()
        avg_bs = (end_bs - start_bs) / 100

        # Tree Timing (N=500)
        tree = InductiveTree(num_steps=500, market_data=market_data, option=option)
        start_tree = time.time()
        for _ in range(10):  # Run fewer times as it's slower
            tree.price()
        end_tree = time.time()
        avg_tree = (end_tree - start_tree) / 10

        print(f"\nAvg BS Time: {avg_bs:.6f}s")
        print(f"Avg Tree(N=500) Time: {avg_tree:.6f}s")

        # Assert Tree is not ridiculously slow
        assert avg_tree < 0.2, f"Tree pricing is too slow: {avg_tree:.4f}s"

    def test_scalability_large_N(self, standard_setup):
        """
        Ensure the tree can handle N=2000 without crashing or taking forever.
        """
        market_data, option = standard_setup

        tree = InductiveTree(num_steps=2000, market_data=market_data, option=option)
        start = time.time()
        price = tree.price()
        duration = time.time() - start

        print(f"Tree(N=2000) Time: {duration:.6f}s")
        assert duration < 2.0, f"Large N tree is too slow: {duration:.4f}s"
        assert price > 0
