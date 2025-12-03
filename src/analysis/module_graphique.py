# %% Imports

import networkx as nx
import plotly.graph_objects as go


from ..pricing import Tree


# %% Classes


class TreeGraph:
    def __init__(self, Tree: Tree, dimension_chart: tuple = (2000, 1500)):
        """Initialisation de la classe Tree_Graph

        Args:
            Tree (Tree): l'Tree pour lequel nous souhaitons réaliser le graphique
            dimension_chart (tuple, optional): la dimension de la fen^tre de sortie de notre graphique. Défaut à (16, 12).
        """
        self.Tree = Tree
        self.dimension_chart = dimension_chart

        # Dans le cas où l'Tree fournit en input n'aurait pas été pricé
        if self.Tree.prix_option is None:
            self.Tree.pricer_Tree()

        if self.Tree.nb_pas > 100:
            raise ValueError(
                "Nombre de pas dans l'Tree trop important pour être affiché. Veuillez choisir un nombre de pas inférieur à 10."
            )

    def afficher_Tree(self) -> None:
        """Fonction nous permettant de réaliser le graphique en positionnant les nœuds selon le prix sous-jacent."""

        # Initialisation de l'objet issu de la librairie networkx
        G = nx.DiGraph()
        labels = {}
        positions = {}
        queue = [self.Tree.racine]

        # le niveau de la barrière
        niveau_barriere = self.Tree.option.barriere.niveau_barriere

        # Déterminer le prix max et min des sous-jacents pour ajuster l'échelle
        prix_min, prix_max = float("inf"), float("-inf")

        while queue:
            Node = queue.pop(0)

            # Utilisation du prix sous-jacent pour la coordonnée y
            y = Node.prix_sj

            # Mise à jour des limites des prix
            prix_min = min(prix_min, y)
            prix_max = max(prix_max, y)

            # Création d'un label pour chaque nœud
            if not (
                Node.valeur_intrinseque == 0
                and Node.prix_sj == 0
                and Node.p_cumule == 0
            ):
                Node_label = f"Valeur intrinsèque : {Node.valeur_intrinseque:.2f}<br>Prix sous-jacent : {Node.prix_sj:.2f}<br>Probabilité cumulée : {Node.p_cumule:.6f}"
                labels[Node] = Node_label
                positions[Node] = (Node.position_Tree, y)
                G.add_node(Node)

            # Itération sur chaque futur nœud
            for direction, futur in zip(
                ["bas", "centre", "haut"],
                [Node.futur_bas, Node.futur_centre, Node.futur_haut],
            ):
                if futur is not None:
                    # Probabilité correspondante à chaque direction
                    if direction == "bas":
                        prob = Node.p_bas
                    elif direction == "centre":
                        prob = Node.p_mid
                    elif direction == "haut":
                        prob = Node.p_haut

                    # Formatage de la probabilité
                    prob_label = f"{prob:.4f}"

                    # Ajout de la ligne avec la probabilité en label
                    if not (
                        futur.valeur_intrinseque == 0
                        and futur.prix_sj == 0
                        and futur.p_cumule == 0
                    ):
                        G.add_edge(Node, futur, label=prob_label)

                    # Ajout du nœud futur à la queue pour itérer dessus
                    if futur not in queue:
                        queue.append(futur)

        # Ajuster les limites de l'axe des y selon les prix sous-jacents
        y_margin = (
            prix_max - prix_min
        ) * 0.05  # Ajout d'une marge de 10% en haut et en bas

        # Dessiner les nœuds et leurs labels
        nx.draw_networkx_nodes(G, pos=positions, node_size=2500, node_color="lightblue")
        nx.draw_networkx_labels(G, pos=positions, labels=labels, font_size=10)

        # Extraction des labels des lignes
        liaison_labels = nx.get_edge_attributes(G, "label")
        nx.draw_networkx_edges(
            G, pos=positions, arrows=True, arrowstyle="-|>", arrowsize=20
        )
        nx.draw_networkx_edge_labels(
            G,
            pos=positions,
            edge_labels=liaison_labels,
            font_color="red",
            font_size=8,
            label_pos=0.3,
        )

        # Preparation des liaison pour plotly
        liaison_x = []
        liaison_y = []
        for edge in G.edges():
            x0, y0 = positions[edge[0]]
            x1, y1 = positions[edge[1]]
            liaison_x += [x0, x1, None]
            liaison_y += [y0, y1, None]

        liaison_trace = go.Scatter(
            x=liaison_x,
            y=liaison_y,
            mode="lines",
            line=dict(width=1, color="#888"),
        )

        Node_x = [positions[node][0] for node in G.nodes()]
        Node_y = [positions[node][1] for node in G.nodes()]

        labels_Nodes = [labels[node] for node in G.nodes()]

        Nodes_trace = go.Scatter(
            x=Node_x,
            y=Node_y,
            mode="markers",
            text=labels_Nodes,
            hoverinfo="text",
            marker=dict(colorscale="YlGnBu", color=[], size=15),
        )

        # Ajout d'un titre au graphique, on fait s'adapter le titre à ce qu'on graphe

        if self.Tree.option.call:
            type_option = "call"
        else:
            type_option = "put"

        if self.Tree.option.americaine:
            exercice_option = "américaine"
        else:
            exercice_option = "européenne"

        strike = self.Tree.option.prix_exercice

        barriere_titre = ""

        if self.Tree.option.barriere.direction_barriere:
            barriere_titre = f", barrière {self.Tree.option.barriere.type_barriere.value} {self.Tree.option.barriere.direction_barriere.value} {round(self.Tree.option.barriere.niveau_barriere, 2)}"

        graphique_titre = f"Tree Trinomial, option {type_option} {exercice_option}, strike {strike}{barriere_titre}"

        fig = go.Figure(
            data=[liaison_trace, Nodes_trace],
            layout=go.Layout(
                title=f"{graphique_titre}",
                titlefont_size=30,
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            ),
        )

        if (
            self.Tree.option.barriere.direction_barriere != None
            and self.Tree.option.barriere.type_barriere != None
        ):
            fig.add_shape(
                type="line",
                x0=min(Node_x),  # Position X de départ
                y0=niveau_barriere,  # position en Y de la barriere
                x1=max(Node_x),  # Fin en X
                y1=niveau_barriere,  # position en Y de la barriere
                line=dict(
                    color="red",
                    width=2,
                    dash="dash",  # ligne pointillée
                ),
            )

        fig.update_layout(width=self.dimension_chart[0], height=self.dimension_chart[1])

        return fig
