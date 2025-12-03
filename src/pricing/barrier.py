#Imports
from typing import Union
from dataclasses import dataclass

from src.pricing.enums import TypeBarriere, DirectionBarriere

#Classes

@dataclass
class Barriere:
    """Classe utilisée pour représenter une barrière pour une option considérée
    """
    
    def __init__(self, niveau_barriere : float, type_barriere : TypeBarriere | None, direction_barriere : DirectionBarriere | None) -> None: 
        self.niveau_barriere = niveau_barriere
        self.type_barriere = type_barriere
        self.direction_barriere = direction_barriere