from dataclasses import dataclass


@dataclass
class PricingConfig:
    """Configuration for the pricing models."""

    epsilon: float = 1e-15
    sum_proba: float = 1.0
    min_payoff: float = 0.0
    min_spot_price: float = 0.0
    alpha_parameter: float = 3.0
    pruning: bool = True
