# %% Imports

import numpy as np
from scipy.stats import norm

from src.pricing.tree_node import Tree

# %% Classes


class BlackAndScholes:
    def __init__(self, Tree: Tree):
        self.option = Tree.option
        self.is_european = not self.option.americaine
        self.type_option = "Call" if self.option.call else "Put"
        self.prix_sj = Tree.donnee_marche.prix_spot
        self.strike = self.option.prix_exercice
        self.risk_free = Tree.donnee_marche.taux_interet
        self.maturite = Tree.get_temps()
        self.volatilite = Tree.donnee_marche.volatilite

        if not self.is_european:
            raise ValueError(
                "Black and Scholes n'est applicable que dans le cas d'options européennes."
            )

        self.d1 = (
            np.log(self.prix_sj / self.strike)
            + (self.risk_free + 0.5 * (self.volatilite**2)) * self.maturite
        ) / (self.volatilite * np.sqrt(self.maturite))
        self.d2 = self.d1 - self.volatilite * np.sqrt(self.maturite)

    def bs_pricer(self):
        if self.type_option == "Call":
            bsprice = self.prix_sj * norm.cdf(self.d1, 0, 1) - self.strike * np.exp(
                -self.risk_free * self.maturite
            ) * norm.cdf(self.d2, 0, 1)
        else:  # on considère ici que nous sommes dans le cas du put
            bsprice = self.strike * np.exp(-self.risk_free * self.maturite) * norm.cdf(
                -self.d2, 0, 1
            ) - self.prix_sj * norm.cdf(-self.d1, 0, 1)

        return bsprice

    def delta(self):
        if self.type_option == "Call":
            delta = norm.cdf(self.d1, 0, 1)
        else:
            delta = norm.cdf(self.d1, 0, 1) - 1

        return delta

    def theta(self):
        if self.type_option == "Call":
            theta = -(
                self.prix_sj
                * norm.pdf(self.d1, 0, 1)
                * self.volatilite
                / 2
                * np.sqrt(self.maturite)
            ) - self.risk_free * self.strike * np.exp(
                -self.risk_free * self.maturite
            ) * norm.cdf(self.d2, 0, 1)
        else:
            theta = -(
                self.prix_sj
                * norm.pdf(self.d1, 0, 1)
                * self.volatilite
                / 2
                * np.sqrt(self.maturite)
            ) + self.risk_free * self.strike * np.exp(
                -self.risk_free * self.maturite
            ) * norm.cdf(-self.d2, 0, 1)

        return theta / 100

    def gamma(self):
        return (
            norm.pdf(self.d1, 0, 1)
            / self.prix_sj
            * self.volatilite
            * np.sqrt(self.maturite)
            * 1000
        )

    def vega(self):
        return self.prix_sj * np.sqrt(self.maturite) * norm.pdf(self.d1) / 100

    def rho(self):
        if self.type_option == "Call":
            rho = (
                self.strike
                * self.maturite
                * np.exp(-self.risk_free * self.maturite)
                * norm.cdf(self.d2, 0, 1)
            )
        else:
            rho = (
                -self.strike
                * self.maturite
                * np.exp(-self.risk_free * self.maturite)
                * norm.cdf(-self.d2, 0, 1)
            )

        return rho / 100
