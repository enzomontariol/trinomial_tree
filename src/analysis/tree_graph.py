# %% Imports

import networkx as nx
import plotly.graph_objects as go


from ..pricing import Tree


# %% Classes


class TreeGraph:
    def __init__(self, tree: Tree, dimension_chart: tuple = (2000, 1500)):
        """Initialization of the TreeGraph class

        Args:
            tree (Tree): the Tree for which we want to create the graph
            dimension_chart (tuple, optional): the dimension of the output window of our graph. Default to (16, 12).
        """
        self.tree = tree
        self.dimension_chart = dimension_chart

        # In case the Tree provided as input has not been priced
        if self.tree.option_price is None:
            self.tree.price_option()

        if self.tree.num_steps > 100:
            raise ValueError(
                "Number of steps in the Tree too large to be displayed. Please choose a number of steps less than 10."
            )

    def display_tree(self) -> None:
        """Function allowing us to create the graph by positioning the nodes according to the underlying price."""

        if self.tree.root is None:
            raise ValueError("Tree root is not defined")

        # Initialization of the object from the networkx library
        G = nx.DiGraph()
        labels = {}
        positions = {}
        queue = [self.tree.root]

        # the barrier level
        barrier_level = self.tree.option.barrier.barrier_level if self.tree.option.barrier else 0

        # Determine the max and min prices of the underlying to adjust the scale
        min_price, max_price = float("inf"), float("-inf")

        while queue:
            node = queue.pop(0)
            if node is None:
                continue

            # Use of the underlying price for the y coordinate
            y = node.spot_price

            # Update price limits
            min_price = min(min_price, y)
            max_price = max(max_price, y)

            # Creation of a label for each node
            if not (
                node.intrinsic_value == 0
                and node.spot_price == 0
                and node.cumulative_p == 0
            ):
                intrinsic_value = node.intrinsic_value if node.intrinsic_value is not None else 0.0
                node_label = f"Intrinsic Value : {intrinsic_value:.2f}<br>Spot Price : {node.spot_price:.2f}<br>Cumulative Probability : {node.cumulative_p:.6f}"
                labels[node] = node_label
                positions[node] = (node.tree_position, y)
                G.add_node(node)

            # Iteration on each future node
            for direction, future in zip(
                ["down", "center", "up"],
                [node.future_down, node.future_center, node.future_up],
            ):
                if future is not None:
                    # Probability corresponding to each direction
                    prob = 0.0
                    if direction == "down":
                        prob = node.p_down if node.p_down is not None else 0.0
                    elif direction == "center":
                        prob = node.p_mid if node.p_mid is not None else 0.0
                    elif direction == "up":
                        prob = node.p_up if node.p_up is not None else 0.0

                    # Formatting the probability
                    prob_label = f"{prob:.4f}"

                    # Adding the line with the probability as label
                    if not (
                        future.intrinsic_value == 0
                        and future.spot_price == 0
                        and future.cumulative_p == 0
                    ):
                        G.add_edge(node, future, label=prob_label)

                    # Adding the future node to the queue to iterate on it
                    if future not in queue:
                        queue.append(future)

        # Adjust the y-axis limits according to the underlying prices
        y_margin = (
            max_price - min_price
        ) * 0.05  # Adding a 10% margin at the top and bottom

        # Draw nodes and their labels
        nx.draw_networkx_nodes(G, pos=positions, node_size=2500, node_color="lightblue")
        nx.draw_networkx_labels(G, pos=positions, labels=labels, font_size=10)

        # Extraction of edge labels
        edge_labels = nx.get_edge_attributes(G, "label")
        nx.draw_networkx_edges(
            G, pos=positions, arrows=True, arrowstyle="-|>", arrowsize=20
        )
        nx.draw_networkx_edge_labels(
            G,
            pos=positions,
            edge_labels=edge_labels,
            font_color="red",
            font_size=8,
            label_pos=0.3,
        )

        # Preparation of edges for plotly
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = positions[edge[0]]
            x1, y1 = positions[edge[1]]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=1, color="#888"),
        )

        node_x = [positions[node][0] for node in G.nodes()]
        node_y = [positions[node][1] for node in G.nodes()]

        node_labels = [labels[node] for node in G.nodes()]

        nodes_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            text=node_labels,
            hoverinfo="text",
            marker=dict(colorscale="YlGnBu", color=[], size=15),
        )

        # Adding a title to the graph, adapting the title to what we graph

        if self.tree.option.is_call:
            option_type = "call"
        else:
            option_type = "put"

        if self.tree.option.is_american:
            option_exercise = "American"
        else:
            option_exercise = "European"

        strike = self.tree.option.strike_price

        barrier_title = ""

        if self.tree.option.barrier and self.tree.option.barrier.barrier_direction:
            barrier_title = f", barrier {self.tree.option.barrier.barrier_type.value} {self.tree.option.barrier.barrier_direction.value} {round(self.tree.option.barrier.barrier_level, 2)}"

        graph_title = f"Trinomial Tree, {option_type} {option_exercise} option, strike {strike}{barrier_title}"

        fig = go.Figure(
            data=[edge_trace, nodes_trace],
            layout=go.Layout(
                title=f"{graph_title}",
                titlefont_size=30,
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            ),
        )

        if (
            self.tree.option.barrier is not None
            and self.tree.option.barrier.barrier_direction is not None
            and self.tree.option.barrier.barrier_type is not None
        ):
            fig.add_shape(
                type="line",
                x0=min(node_x),  # Start X position
                y0=barrier_level,  # Y position of the barrier
                x1=max(node_x),  # End X
                y1=barrier_level,  # Y position of the barrier
                line=dict(
                    color="red",
                    width=2,
                    dash="dash",  # dashed line
                ),
            )

        fig.update_layout(width=self.dimension_chart[0], height=self.dimension_chart[1])

        return fig
