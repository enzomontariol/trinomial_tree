import datetime as dt

import src.pricing as pricing

market_data = pricing.MarketData(
    date_debut=dt.date(2024, 1, 13),
    prix_spot=100,
    volatilite=0.20,
    taux_interet=0.02,
    taux_actualisation=0.02,
    dividende_ex_date=dt.date.today(),
    dividende_montant=0,
)

option = pricing.Option(
    maturite=dt.date(2025, 1, 13),
    prix_exercice=100,
    barriere=None,
    americaine=False,
    call=True,
    date_pricing=dt.date(2024, 1, 13),
)

tree = pricing.Tree(nb_pas=100, donnee_marche=market_data, option=option)

if __name__ == "__main__":
    tree.pricer_Tree()
    print(f"Le prix de l'option est : {tree.prix_option:.2f}")
