# %% Imports

import streamlit as st
import datetime as dt
from datetime import timedelta
import numpy as np
import time
import sys
import os
import warnings


from src.pricing.enums import TypeBarriere, DirectionBarriere, ConventionBaseCalendaire
from src.pricing.market import MarketData
from src.pricing.option import Option
from src.pricing.barrier import Barriere
from src.pricing.tree_node import Tree
from src.pricing.black_scholes import BlackAndScholes


from src.analysis import (
    TreeGraph,
    GrecquesEmpiriques,
    BsComparison,
    StrikeComparison,
    VolComparison,
    RateComparison,
)

warnings.filterwarnings("ignore")
sys.setrecursionlimit(1000000000)

# %% Constantes

today = dt.date.today()

# %% Streamlit

st.set_page_config(layout="wide")


# Titre de l'application
st.title(
    "Trinomial Tree - [LIN Cécile](%s), [MONTARIOL Enzo](%s)"
    % (
        "https://www.linkedin.com/in/c%C3%A9cile-lin-196b751b5/",
        "https://www.linkedin.com/in/enzomontariol/",
    )
)

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(
    [
        "Pricing",
        "Plus d'options",
        "Graphique",
        "Greeks",
        "Comparaison avec Black-Scholes",
        "Comparaison seuil de pruning",
        "Comparaison strike",
        "Comparaison volatilité",
        "Comparaison taux d'intérêt",
    ]
)

###########################################################################
###################### Onglet 1 : Inputs Utilisateur ######################
###########################################################################


with tab1:
    activer_pricing = st.button("Pricing")

    col11, col12, col13 = st.columns(3)

    with col11:
        date_pricing = st.date_input("Entrez une date de pricing :", value=today)

    with col13:
        nb_pas = st.number_input(
            "Entrez le nombre de pas utilisé pour le pricing :",
            1,
            5000,
            value=100,
            step=1,
        )

    st.divider()

    # Données de marché

    st.subheader("Données de marché :")

    dividende_check = st.toggle("Dividende", value=False)

    col21, col22, col23 = st.columns(3)

    with col21:
        spot = st.number_input(
            "Entrez le prix de départ du sous-jacent (en €):",
            format="%.2f",
            value=100.0,
            step=0.01,
        )

    with col22:
        volatite = (
            st.number_input(
                "Entrez le niveau de volatilité (en %):",
                format="%.2f",
                value=20.0,
                step=1.00,
            )
            / 100
        )

    with col23:
        risk_free_rate = (
            st.number_input(
                "Entrez le niveau de taux d'intérêt (en %):",
                format="%.2f",
                value=4.0,
                step=1.00,
            )
            / 100
        )

    if dividende_check:
        with col21:
            dividende_ex_date = st.date_input("Entrez la date de dividende :")
        with col22:
            dividende_montant = st.number_input(
                "Entrez le montant du dividende (en €):",
                format="%.2f",
                value=0.0,
                step=1.00,
            )
    else:
        dividende_ex_date = today
        dividende_montant = 0

    # Option

    st.divider()

    st.subheader("Caractéristique de l'option :")

    col31, col32, col33 = st.columns(3)

    with col31:
        maturite = st.date_input(
            "Entrez une date de maturité :", value=today + timedelta(days=10)
        )

    with col33:
        barriere_check = st.checkbox("Option à barrière ?", value=False)

    col41, col42, col43 = st.columns(3)

    with col41:
        strike = st.number_input(
            "Entrez le strike (en €):", format="%.2f", value=100.0, step=0.01
        )

    with col42:
        option_type = st.selectbox("Choisissez le type de l'option :", ["Call", "Put"])

    with col43:
        option_exercice = st.selectbox(
            "Choisissez le type de l'exercice :", ["Européenne", "Américaine"]
        )

    # Barrière

    if barriere_check:
        st.divider()
        st.subheader("Barrière :")

        col51, col52, col53 = st.columns(3)

        with col51:
            niveau_barriere = st.number_input(
                "Entrez le niveau de la barrière (en €):",
                format="%.2f",
                value=spot * 1.1,
                step=0.01,
            )

        with col52:
            type_barriere_select = st.selectbox(
                "Choisissez le type de barrière :",
                [type.value for type in TypeBarriere],
            )
            type_barriere = TypeBarriere(type_barriere_select)

        with col53:
            direction_barriere_select = st.selectbox(
                "Choisissez le sens de la barrière :",
                [direction.value for direction in DirectionBarriere],
            )
            direction_barriere = DirectionBarriere(direction_barriere_select)
    else:
        niveau_barriere = 0
        type_barriere = None
        direction_barriere = None

    # Ici, on feed les objets

    barriere = Barriere(
        niveau_barriere=niveau_barriere,
        type_barriere=type_barriere,
        direction_barriere=direction_barriere,
    )

    donnee_marche = MarketData(
        date_pricing,
        spot,
        volatite,
        risk_free_rate,
        risk_free_rate,
        dividende_ex_date,
        dividende_montant,
    )
    option = Option(
        maturite,
        strike,
        barriere=barriere,
        americaine=False if option_exercice == "Européenne" else True,
        call=True if option_type == "Call" else False,
        date_pricing=date_pricing,
    )

    bs_check = (
        option.americaine == False
        and donnee_marche.dividende_montant == 0
        and option.barriere.direction_barriere == None
    )

###########################################################################
############# Onglet 2 : Inputs additionnels Utilisateur ##################
###########################################################################

with tab2:
    st.subheader("Plus de paramètre modulable")
    col11, col2, col3 = st.columns(3)

    with col11:
        # on garde le format float, pour garder la possibilité de mettre 365.25
        parametre_alpha = st.number_input(
            "Entrez le paramètre alpha :",
            min_value=2.0,
            max_value=4.0,
            value=3.0,
            step=0.1,
        )
        convention_base_calendaire = st.selectbox(
            "Choisissez la base annuelle :",
            [nombre.value for nombre in ConventionBaseCalendaire],
        )

    with col3:
        pruning = st.toggle("Elagage de l'Tree", value=True)
        if pruning:
            epsilon_Tree = float(
                "1e-"
                + str(
                    st.number_input(
                        "Seuil de pruning (1e-)", min_value=1, max_value=100, value=15
                    )
                )
            )
            st.markdown(epsilon_Tree)
            Tree = Tree(
                nb_pas=nb_pas,
                donnee_marche=donnee_marche,
                option=option,
                convention_base_calendaire=convention_base_calendaire,
                parametre_alpha=parametre_alpha,
                pruning=pruning,
                epsilon=epsilon_Tree,
            )
        else:
            Tree = Tree(
                nb_pas=nb_pas,
                donnee_marche=donnee_marche,
                option=option,
                convention_base_calendaire=convention_base_calendaire,
                parametre_alpha=parametre_alpha,
                pruning=pruning,
            )

with tab1:
    if activer_pricing:
        if bs_check:
            bns = BlackAndScholes(Tree=Tree)
            pricing_bns = f"{round(bns.bs_pricer(), 2)}€"
            st.divider()
            st.subheader("Pricing Black and Scholes : ")
            st.metric("", value=pricing_bns)

        start = time.time()
        st.divider()
        st.subheader("Pricing avec Tree : ")
        with st.spinner("""Valorisation de l'option en cours..."""):
            Tree.pricer_Tree()
        end = time.time()
        time_difference = round(end - start, 1)
        prix_option = f"{round(Tree.prix_option, 2)}€"

        Tree_st = Tree

        st.metric("""Valeur de l'option :""", value=prix_option, delta=None)
        st.metric("Temps de pricing (secondes) :", value=time_difference, delta=None)


# ###########################################################################
# ################## Onglet 3 : Tree Trinomial Display #####################
# ###########################################################################


with tab3:
    st.subheader("Tree Trinomiale")
    graph_Tree_boutton = st.button("Graph Tree")
    if graph_Tree_boutton:
        if nb_pas <= 50:
            with st.spinner("Graphique en cours de réalisation..."):
                graph = TreeGraph(Tree=Tree).afficher_Tree()
                st.plotly_chart(graph, use_container_width=True)
        else:
            st.markdown(
                "Les Trees avec un trop gros nombre de pas deviennent illisibles. Veuillez saisir un plus petit nombre de pas."
            )

# ###########################################################################
# ########################### Onglet 4 : Grecques ###########################
# ###########################################################################

with tab4:
    with st.expander("Paramètres grecques empiriques"):
        st.markdown("Valeurs utilisées lors du calcul de la différence centrée finie :")
        cols = st.columns(4)
        with cols[0]:
            var_s = st.number_input(
                "Variation prix sous-jacent (% spot):", 0.001, 1.0, 0.01, 0.01
            )
        with cols[1]:
            var_v = st.number_input(
                "Variation de la volatilité (pts %)", 0.001, 1.0, 0.01, 0.01
            )
        with cols[2]:
            var_t = st.number_input(
                "Variation du temps à maturité (nb jour)", 1, 10, 1, 1
            )
        with cols[3]:
            var_r = st.number_input(
                """Varitation du taux d'intérêt (pts %)""", 0.001, 1.0, 0.01, 0.01
            )

    if Tree.prix_option is not None:
        st.subheader("Grecques empiriques : ")

        grecques_empiriques = GrecquesEmpiriques(
            Tree, var_s=var_s, var_v=var_v, var_t=var_t, var_r=var_r
        )

        with st.spinner("""Calcul des grecques en cours..."""):
            delta = round(grecques_empiriques.approxime_delta(), 2)
            gamma = round(grecques_empiriques.approxime_gamma(), 2)
            vega = round(grecques_empiriques.approxime_vega(), 2)
            theta = round(grecques_empiriques.approxime_theta(), 2)
            rho = round(grecques_empiriques.approxime_rho(), 2)

        col11, col12, col13, col14, col15 = st.columns(5)

        with col11:
            st.metric(label="Delta", value=delta, delta=None)
        with col12:
            st.metric(label="Gamma", value=gamma, delta=None)
        with col13:
            st.metric(label="Vega", value=vega, delta=None)
        with col14:
            st.metric(label="Theta", value=theta, delta=None)
        with col15:
            st.metric(label="Rho", value=rho, delta=None)

    else:
        st.markdown(
            "Veuillez valoriser l'option via son Tree avant de pouvoir accéder à ses grecques calculée via différence finie centrée."
        )

    if bs_check:
        st.divider()
        st.subheader("Grecques Black and Scholes : ")

        bs = BlackAndScholes(Tree)

        bs_delta = round(bs.delta(), 2)
        bs_gamma = round(bs.gamma(), 2)
        bs_vega = round(bs.vega(), 2)
        bs_theta = round(bs.theta(), 2)
        bs_rho = round(bs.rho(), 2)

        col21, col22, col23, col24, col25 = st.columns(5)

        with col21:
            st.metric(label="Delta", value=bs_delta, delta=None)
        with col22:
            st.metric(label="Gamma", value=bs_gamma, delta=None)
        with col23:
            st.metric(label="Vega", value=bs_vega, delta=None)
        with col24:
            st.metric(label="Theta", value=bs_theta, delta=None)
        with col25:
            st.metric(label="Rho", value=bs_rho, delta=None)


# ###########################################################################
# ################## Onglet 4 : Black-Scholes Comparaison ###################
# ###########################################################################

with tab5:
    max_cpu = st.number_input(
        """Veuillez choisir le nombre de coeur qui sera mis à contribution pour le multiprocessing (choisir 1 revient à ne pas en faire, monter trop haut peut induire de l'instabilité.):""",
        1,
        os.cpu_count(),
        4,
        1,
        key="bs",
    )
    bs_comparison_button = st.button(
        """Lancer l'analyse comparative (l'opération prend environ 3min)"""
    )

    if bs_comparison_button:
        now = time.time()

        @st.cache_resource
        def call_bs_comparison():
            bs_step_list = [
                10,
                20,
                30,
                40,
                50,
                60,
                70,
                80,
                90,
                100,
                120,
                140,
                160,
                180,
                200,
                220,
                240,
                260,
                280,
                300,
                320,
                340,
                360,
                380,
                400,
                450,
                500,
                550,
                600,
                700,
                800,
                900,
                1000,
                1100,
                1200,
                1300,
                1400,
                1500,
                1750,
                2000,
                2500,
                3000,
                4000,
                5000,
            ]
            bs_epsilon_values = [1e-10]
            return BsComparison(max_cpu, bs_step_list, bs_epsilon_values)

        bs_comparison = call_bs_comparison()

        st.markdown(f"Résultats après {round(time.time() - now, 2)} secondes.")

        st.plotly_chart(bs_comparison.bs_graph_temps_pas())
        st.plotly_chart(bs_comparison.bs_graph_prix_pas())

        with st.expander(label="Données"):
            st.markdown(
                """Pour un Call Européen: avec un prix de départ du sous jacent à 100, un strike à 101, une volatilité à 20% et un taux d'intérêt à 2%, une date de pricing au 13/01/2024 et une maturité au 23/10/2024, un seuil d'élagage fixé à 1e-10, on obtient le tableau suivant en fonction du nombre de pas."""
            )
            st.metric("Prix B&S", round(bs_comparison.bs_price, 4))
            st.dataframe(bs_comparison.results_df)

# ###########################################################################
# ################## Onglet 5 : Epsilon Comparaison #########################
# ###########################################################################

with tab6:
    max_cpu = st.number_input(
        """Veuillez choisir le nombre de coeur qui sera mis à contribution pour le multiprocessing (choisir 1 revient à ne pas en faire, monter trop haut peut induire de l'instabilité.):""",
        1,
        os.cpu_count(),
        4,
        1,
        key="epsilon",
    )
    epsilon_comparison_button = st.button(
        """Lancer l'analyse comparative (l'opération prend environ 5min)"""
    )

    if epsilon_comparison_button:
        now = time.time()

        @st.cache_resource
        def call_epsilon_comparison():
            step_list = [
                50,
                60,
                70,
                80,
                90,
                100,
                120,
                140,
                160,
                180,
                200,
                220,
                240,
                260,
                280,
                300,
                320,
                340,
                360,
                380,
                400,
                450,
                500,
                550,
                600,
                700,
                800,
                900,
                1000,
                1200,
                1500,
                2000,
                3000,
                5000,
            ]
            epsilon_values = []
            for i in range(3, 13, 1):
                epsilon_values.append(10 ** (-i))
            return BsComparison(max_cpu, step_list, epsilon_values)

        st.subheader(
            "Différences de convergence selon le niveau choisi pour le pruning"
        )

        epsilon_comparison = call_epsilon_comparison()

        st.markdown(f"Résultats après {round(time.time() - now, 2)} secondes.")

        st.markdown(
            """Nous atteignons une convergence à partir d'un seuil de pruning inférieur à 1e-8. En le diminuant, on atteind un prix plus précis au détriment de temps de calcul."""
        )
        st.plotly_chart(epsilon_comparison.epsilon_graph_prix_pas_bas_epsilon())

        st.markdown(
            "Dans le cas d'un seuil de pruning qui ne serait pas assez permissif, i.e. d'une valeur trop grande, on voit bien que l'on perd la propriété de convergence vers B&S."
        )
        st.plotly_chart(epsilon_comparison.epsilon_graph_prix_pas_haut_epsilon())

        with st.expander(label="Données graphiques 1 et 2"):
            st.markdown(
                """Pour un Call Européen: avec un prix de départ du sous jacent à 100, un strike à 101, une volatilité à 20% et un taux d'intérêt à 2%, une date de pricing au 13/01/2024 et une maturité au 23/10/2024, on obtient le tableau suivant en fonction du nombre de pas et ce pour différents seuil d'élagage."""
            )
            st.metric("Prix B&S", round(epsilon_comparison.bs_price, 4))
            st.dataframe(epsilon_comparison.results_df)

        st.divider()

        st.subheader(
            """Temps nécessaire au pricing d'un Tree de 5000 pas selon le niveau de pruning choisi"""
        )

        st.markdown(
            "Ce graphique nous permet de nous rendre compte de l'effet de la variation d'epsilon sur le temps de pricing d'un Tree à 5000 pas. (L'effet en cloche semble dû à notre choix d'effectuer ces calculs en parallèle, les dernières valeur d'epsilon sont celles qui arrivent à la fin de la queue de calcul et donc pour lesquelles on a davantage de puissance de calcul.)"
        )
        st.plotly_chart(epsilon_comparison.epsilon_vs_temps_pricing_graph())

# ###########################################################################
# ################## Onglet 6 : Strike Comparaison #########################
# ###########################################################################

with tab7:
    max_cpu = st.number_input(
        """Veuillez choisir le nombre de coeur qui sera mis à contribution pour le multiprocessing (choisir 1 revient à ne pas en faire, monter trop haut peut induire de l'instabilité.):""",
        1,
        os.cpu_count(),
        4,
        1,
        key="strike",
    )
    strike_comparison_button = st.button(
        """Lancer l'analyse comparative (l'opération prend environ 2min)"""
    )

    if strike_comparison_button:
        now = time.time()

        @st.cache_resource
        def call_strike_comparison():
            step_list = [300]
            strike_list = np.arange(100, 110.1, 0.1)
            return StrikeComparison(max_cpu, step_list, strike_list)

        st.subheader("""Différence d'écart selon le niveau de strike""")

        strike_comparison = call_strike_comparison()

        st.markdown(f"Résultats après {round(time.time() - now, 2)} secondes.")
        st.plotly_chart(strike_comparison.graph_strike_temps_pas())

        with st.expander(label="Données"):
            st.markdown(
                """Pour un Call Européen: avec un prix de départ du sous jacent à 100, une volatilité à 20% et un taux d'intérêt à 2%, une date de pricing au 13/01/2024, une maturité au 23/10/2024, un niveau d'élagage à 1e-10 et des Trees de 300 pas, on obtient le tableau suivant en fonction du strike."""
            )
            st.dataframe(
                strike_comparison.results_df.sort_values(by="Strike", ascending=True)
            )

# ###########################################################################
# ################## Onglet 7 : Volatilité Comparaison ######################
# ###########################################################################

with tab8:
    max_cpu = st.number_input(
        """Veuillez choisir le nombre de coeur qui sera mis à contribution pour le multiprocessing (choisir 1 revient à ne pas en faire, monter trop haut peut induire de l'instabilité.):""",
        1,
        os.cpu_count(),
        4,
        1,
        key="vol",
    )
    vol_comparison_button = st.button(
        """Lancer l'analyse comparative (l'opération prend environ 1.5min)"""
    )

    if vol_comparison_button:
        now = time.time()

        @st.cache_resource
        def call_vol_comparison():
            step_list = [300]
            vol_list = np.arange(0.01, 1.0, 0.01)
            return VolComparison(max_cpu, step_list, vol_list)

        vol_comparison = call_vol_comparison()

        st.markdown(f"Résultats après {round(time.time() - now, 2)} secondes.")
        st.plotly_chart(vol_comparison.graph_vol_temps_pas())

        with st.expander(label="Données"):
            st.markdown(
                """Pour un Call Européen: avec un prix de départ du sous jacent à 100, un strike à 101, un taux d'intérêt à 2%, une date de pricing au 13/01/2024, une maturité au 23/10/2024, un niveau d'élagage à 1e-10 et des Trees de 300 pas, on obtient le tableau suivant en fonction de la voltatilité."""
            )
            st.dataframe(
                vol_comparison.results_df.sort_values(by="Volatilité", ascending=True)
            )

# ###########################################################################
# ################## Onglet 7 : Intérêt Comparaison ######################
# ###########################################################################

with tab9:
    max_cpu = st.number_input(
        """Veuillez choisir le nombre de coeur qui sera mis à contribution pour le multiprocessing (choisir 1 revient à ne pas en faire, monter trop haut peut induire de l'instabilité.):""",
        1,
        os.cpu_count(),
        4,
        1,
        key="rate",
    )
    rate_comparison_button = st.button(
        """Lancer l'analyse comparative (l'opération prend environ 2 min)"""
    )

    if rate_comparison_button:
        now = time.time()

        @st.cache_resource
        def call_rate_comparison():
            step_list = [100]
            rate_list = np.arange(-0.5, 0.5, 0.005)
            return RateComparison(max_cpu, step_list, rate_list)

        rate_comparison = call_rate_comparison()

        st.markdown(f"Résultats après {round(time.time() - now, 2)} secondes.")
        st.plotly_chart(rate_comparison.graph_rate_temps_pas())

        with st.expander(label="Données"):
            st.markdown(
                """Pour un Call Européen: avec un prix de départ du sous jacent à 100, un strike à 101, une volatilité à 20%, une date de pricing au 13/01/2024, une maturité au 23/10/2024, un niveau d'élagage à 1e-10 et des Trees de 300 pas, on obtient le tableau suivant en fonction du niveau de taux d'intérêt."""
            )
            st.dataframe(
                rate_comparison.results_df.sort_values(
                    by="""Taux d'intérêt""", ascending=True
                )
            )
