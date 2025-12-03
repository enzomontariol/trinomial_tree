# %% Imports

import datetime as dt
from dataclasses import dataclass

# %% Classes


@dataclass
class MarketData:
    """Classe utilisée pour représenter les données de marché."""

    date_debut: dt.date
    prix_spot: float
    volatilite: float
    taux_interet: float
    taux_actualisation: float
    dividende_ex_date: dt.date
    dividende_montant: float
