import numpy as np
from scipy.stats import norm

from src.pricing.pricer import Pricer
from src.pricing.market import MarketData
from src.pricing.option import Option


class BlackScholes(Pricer):
    def __init__(
        self,
        market_data: MarketData,
        option: Option,
    ) -> None:
        self.market_data = market_data
        self.option = option
        self.is_european = not self.option.is_american
        self.option_type = "Call" if self.option.is_call else "Put"
        self.spot_price = self.market_data.spot_price
        self.strike = self.option.strike_price
        self.risk_free_rate = self.market_data.interest_rate
        self.maturity = (
            self.option.maturity - self.option.pricing_date
        ).days / self.option.calendar_base_convention
        self.volatility = self.market_data.volatility

        if not self.is_european:
            raise ValueError(
                "Black and Scholes is only applicable for European options."
            )

        self.d1 = (
            np.log(self.spot_price / self.strike)
            + (self.risk_free_rate + 0.5 * (self.volatility**2)) * self.maturity
        ) / (self.volatility * np.sqrt(self.maturity))
        self.d2 = self.d1 - self.volatility * np.sqrt(self.maturity)

    def price(self) -> float:
        if self.option_type == "Call":
            bsprice = self.spot_price * norm.cdf(self.d1, 0, 1) - self.strike * np.exp(
                -self.risk_free_rate * self.maturity
            ) * norm.cdf(self.d2, 0, 1)
        else:  # we consider here that we are in the case of the put
            bsprice = self.strike * np.exp(
                -self.risk_free_rate * self.maturity
            ) * norm.cdf(-self.d2, 0, 1) - self.spot_price * norm.cdf(-self.d1, 0, 1)

        return bsprice

    def delta(self):
        if self.option_type == "Call":
            delta = norm.cdf(self.d1, 0, 1)
        else:
            delta = norm.cdf(self.d1, 0, 1) - 1

        return delta

    def theta(self):
        if self.option_type == "Call":
            theta = -(
                (self.spot_price * norm.pdf(self.d1, 0, 1) * self.volatility)
                / (2 * np.sqrt(self.maturity))
            ) - self.risk_free_rate * self.strike * np.exp(
                -self.risk_free_rate * self.maturity
            ) * norm.cdf(self.d2, 0, 1)
        else:
            theta = -(
                (self.spot_price * norm.pdf(self.d1, 0, 1) * self.volatility)
                / (2 * np.sqrt(self.maturity))
            ) + self.risk_free_rate * self.strike * np.exp(
                -self.risk_free_rate * self.maturity
            ) * norm.cdf(-self.d2, 0, 1)

        return theta / self.option.calendar_base_convention

    def gamma(self):
        return norm.pdf(self.d1, 0, 1) / (
            self.spot_price * self.volatility * np.sqrt(self.maturity)
        )

    def vega(self):
        return self.spot_price * np.sqrt(self.maturity) * norm.pdf(self.d1) / 100

    def rho(self):
        if self.option_type == "Call":
            rho = (
                self.strike
                * self.maturity
                * np.exp(-self.risk_free_rate * self.maturity)
                * norm.cdf(self.d2, 0, 1)
            )
        else:
            rho = (
                -self.strike
                * self.maturity
                * np.exp(-self.risk_free_rate * self.maturity)
                * norm.cdf(-self.d2, 0, 1)
            )

        return rho / 100
