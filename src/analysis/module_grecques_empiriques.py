from typing import Union
import datetime as dt
import copy

from src.pricing.tree_node import Tree


class GrecquesEmpiriques:
    def __init__(
        self,
        Tree: Tree,
        var_s: float = 0.01,
        var_v: float = 0.01,
        var_t: int = 1,
        var_r: float = 0.01,
    ):
        """Initialisation de la classe

        Args:
            Tree (Tree): L'Tree pour lequel portera le calcul des grecques
            var_s (float, optional): La variation que l'on fait subir au prix du sous-jacent (en pourcentage du prix). Defaults to 0.01.
            var_v (float, optional): La variation du niveau de volatilité (en point de pourcentage ici). Defaults to 0.01.
            var_t (int, optional): La variation de la date de pricing (en nombre de jour). Defaults to 1.
            var_r (float, optional): La variation du niveau du taux d'intérêt (en point de pourcentage). Defaults to 0.01.
        """

        self.Tree = Tree
        self.var_s = var_s
        self.var_v = var_v
        self.var_t = var_t
        self.var_r = var_r

        if not hasattr(self.Tree, "prix_option"):
            # Dans le cas où l'utilisateur donnerait en input un Tree qui n'a pour le moment pas été pricé.
            self.Tree.pricer_Tree()

    def __pricer_Tree_choc(
        self, attribut_a_modifier: str, d: Union[float, dt.date]
    ) -> Tree:
        """Méthode nous permettant de valoriser une nouvelle option en faisant varier un paramètre donné.

        Args:
            attribut_a_modifier (str): Quel est le paramètre que nous allons faire bouger.
            d (Union[float, dt.date]): Niveau de la variation

        Returns:
            Tree: Le nouvel Tree qui a été pricé tout chose égale par ailleurs.
        """

        nouvel_Tree = copy.copy(self.Tree)
        donnee_marche = copy.copy(self.Tree.donnee_marche)
        option = copy.copy(self.Tree.option)

        # Cas theta
        if attribut_a_modifier == "date_pricing":
            setattr(option, attribut_a_modifier, d)
            setattr(donnee_marche, "date_debut", d)

        # Cas rho
        elif (
            attribut_a_modifier == "taux_interet"
        ):  # on part du principe que le choc sur les taux d'intérêt s'applique à la fois au facteur de capitalisation et d'actualisaition
            attribut_a_modifier_1 = "taux_interet"
            attribut_a_modifier_2 = "taux_actualisation"
            d1 = getattr(donnee_marche, attribut_a_modifier_1) + d
            d2 = getattr(donnee_marche, attribut_a_modifier_2) + d
            setattr(donnee_marche, attribut_a_modifier_1, d1)
            setattr(donnee_marche, attribut_a_modifier_2, d2)

        # Cas général
        else:
            d1 = getattr(donnee_marche, attribut_a_modifier) + d
            setattr(donnee_marche, attribut_a_modifier, d1)

        # Initialisation du nouvel Tree et valorisation

        nouvel_Tree.__init__(
            nb_pas=self.Tree.nb_pas, donnee_marche=donnee_marche, option=option
        )
        nouvel_Tree.pricer_Tree()

        return nouvel_Tree

    def approxime_delta(self) -> float:
        """Calcul du delta de notre option, la dérivée partielle du prix de l'option par rapport au prix du sous-jacent.

        Returns:
            float: le delta
        """

        # calcul du nouveau spot que l'on utilisera dans le pricing du nouvel Tree
        ds = self.var_s * self.Tree.donnee_marche.prix_spot
        neg_ds = -self.var_s * self.Tree.donnee_marche.prix_spot

        # Valorisation avec les nouveaux paramètres
        nouvel_Tree_1 = self.__pricer_Tree_choc("prix_spot", ds)
        nouvel_Tree_2 = self.__pricer_Tree_choc("prix_spot", neg_ds)

        # Ici, nous calculons le delta à partir d'une différence finie centrée
        delta = (nouvel_Tree_1.prix_option - nouvel_Tree_2.prix_option) / 2 * ds

        # on stocke dans la classe la valeur de l'Tree choqué pour ne pas à avoir à recalculer si on calcule une dérivée de second ordre
        if not hasattr(self, "prix_nouvel_Tree_ds_1"):
            self.prix_nouvel_Tree_ds_1 = nouvel_Tree_1.prix_option

        # idem ici
        if not hasattr(self, "prix_nouvel_Tree_ds_2"):
            self.prix_nouvel_Tree_ds_2 = nouvel_Tree_2.prix_option

        return delta

    def approxime_gamma(self) -> float:
        """Calcul du gamma de notre option, la dérivée partielle seconde du prix de notre option par rapport au prix du sous-jacent.

        Returns:
            float: le gamma
        """

        # calcul du nouveau spot que l'on utilisera dans le pricing du nouvel Tree
        ds = self.var_s * self.Tree.donnee_marche.prix_spot
        neg_ds = -self.var_s * self.Tree.donnee_marche.prix_spot

        # Dans le cas où nous n'aurions préalablement pas calculé de delta de l'option.
        if not hasattr(self, "prix_nouvel_Tree_ds_1"):
            nouvel_Tree_1 = self.__pricer_Tree_choc("prix_spot", ds)
            self.prix_nouvel_Tree_ds_1 = nouvel_Tree_1.prix_option

        if not hasattr(self, "prix_nouvel_Tree_ds_2"):
            nouvel_Tree_2 = self.__pricer_Tree_choc("prix_spot", neg_ds)
            self.prix_nouvel_Tree_ds_2 = nouvel_Tree_2.prix_option

        # Calcul du gamma d'après la formule d'une différence finie centrée
        gamma = (
            self.prix_nouvel_Tree_ds_1
            - 2 * self.Tree.prix_option
            + self.prix_nouvel_Tree_ds_2
        ) / (ds**2)

        return gamma

    def approxime_vega(self) -> float:
        """Calcul du vega de notre option, la dérivée partielle du prix de notre option par rapport au niveau de volatilité du sous-jacent

        Returns:
            float: le vega
        """

        dv = self.var_v
        neg_dv = -self.var_v

        # calcul des Trees que nous utiliserons
        nouvel_Tree_1 = self.__pricer_Tree_choc("volatilite", dv)
        nouvel_Tree_2 = self.__pricer_Tree_choc("volatilite", neg_dv)

        # Différence finie centrée
        vega = (nouvel_Tree_1.prix_option - nouvel_Tree_2.prix_option) / 2 * dv * 100

        # on stocke dans la classe la valeur de l'Tree choqué pour ne pas à avoir à recalculer si on calcule une dérivée de second ordre
        if not hasattr(self, "vol_nouvel_Tree_dv_1"):
            self.vol_nouvel_Tree_ds_1 = nouvel_Tree_1.prix_option

        # idem ici
        if not hasattr(self, "vol_nouvel_Tree_dv_2"):
            self.vol_nouvel_Tree_ds_2 = nouvel_Tree_2.prix_option

        return vega

    def approxime_theta(self) -> float:
        """Calcul du theta de notre option, la dérivée partielle du prix de notre option par rapport au temps.

        Returns:
            float: le theta
        """

        # Ici, la différence se fait sur un nombre de jour
        d_t = self.Tree.option.date_pricing + dt.timedelta(days=self.var_t)

        # Nouvel Tree avec pricé à la date déterminée plus haut
        nouvel_Tree_1 = self.__pricer_Tree_choc("date_pricing", d_t)

        # calcul du theta via différence finie avant.
        theta = nouvel_Tree_1.prix_option - self.Tree.prix_option

        # on stocke dans la classe la valeur de l'Tree choqué pour ne pas à avoir à recalculer si on calcule une dérivée de second ordre
        if not hasattr(self, "temps_nouvel_Tree_dt_1"):
            self.theta_nouvel_Tree_ds_1 = nouvel_Tree_1.prix_option

        return theta

    def approxime_rho(self) -> float:
        """Calcul de rho, la dérivée partielle du prix de notre option par rapport au taux d'intérêt sans risque

        Returns:
            float: Le rho
        """

        dr = self.var_r
        neg_dr = -self.var_r

        # Les nouveaux Trees que nous utiliserans dans la diférence finie
        nouvel_Tree_1 = self.__pricer_Tree_choc("taux_interet", dr)
        nouvel_Tree_2 = self.__pricer_Tree_choc("taux_interet", neg_dr)

        # Calcul de rho via différence finie centrée
        rho = (nouvel_Tree_1.prix_option - nouvel_Tree_2.prix_option) / 2 * dr * 100

        # on stocke dans la classe la valeur de l'Tree choqué pour ne pas à avoir à recalculer si on calcule une dérivée de second ordre
        if not hasattr(self, "vol_nouvel_Tree_dr_1"):
            self.rho_nouvel_Tree_ds_1 = nouvel_Tree_1.prix_option

        # idem ici
        if not hasattr(self, "vol_nouvel_Tree_dr_2"):
            self.rho_nouvel_Tree_ds_2 = nouvel_Tree_2.prix_option

        return rho
