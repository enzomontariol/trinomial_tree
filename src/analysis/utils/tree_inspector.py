import pandas as pd
import numpy as np
from collections import defaultdict
from typing import List, Dict, Set
from src.pricing.tree_node import Tree, Node


class TreeInspector:
    def __init__(self, tree: Tree):
        self.tree = tree
        if not self.tree.root:
            raise ValueError("Tree has not been built. Run price() first.")
        self.root = self.tree.root

    def get_terminal_distribution(self) -> pd.DataFrame:
        """
        Returns DataFrame with columns ['Spot', 'Probability'] for the terminal nodes.
        """
        # Level-by-level traversal to propagate probabilities
        current_level = {self.root: 1.0}

        for step in range(self.tree.num_steps):
            next_level = defaultdict(float)
            for node, prob in current_level.items():
                # Check if children exist
                if node.future_up and node.p_up is not None:
                    next_level[node.future_up] += prob * node.p_up
                if node.future_center and node.p_mid is not None:
                    next_level[node.future_center] += prob * node.p_mid
                if node.future_down and node.p_down is not None:
                    next_level[node.future_down] += prob * node.p_down
            current_level = next_level

        # Now current_level contains leaves
        data = []
        for node, prob in current_level.items():
            data.append({"Spot": node.spot_price, "Probability": prob})

        return pd.DataFrame(data).sort_values("Spot")

    def get_early_exercise_boundary(self) -> pd.DataFrame:
        """
        Returns DataFrame with columns ['Time', 'Step', 'Boundary_Spot']
        """
        boundary_points = []

        # Start with root
        current_level_nodes = {self.root}

        for step in range(self.tree.num_steps + 1):
            exercised_spots = []

            # Sort nodes by spot price for consistent analysis
            sorted_nodes = sorted(list(current_level_nodes), key=lambda n: n.spot_price)

            for node in sorted_nodes:
                payoff = self._calculate_payoff(node)
                # Check if exercised: intrinsic_value == payoff > 0
                # Note: intrinsic_value might be None if not calculated, but price() should have calculated it.
                if node.intrinsic_value is not None:
                    # Use a small epsilon for float comparison
                    if np.isclose(node.intrinsic_value, payoff) and payoff > 1e-9:
                        exercised_spots.append(node.spot_price)

            if exercised_spots:
                # For Put, boundary is usually the upper limit of exercise region (low spots).
                # For Call, boundary is usually the lower limit of exercise region (high spots).
                if self.tree.option.is_call:
                    boundary = min(exercised_spots)
                else:
                    boundary = max(exercised_spots)

                time_val = step * self.tree.delta_t
                boundary_points.append(
                    {"Time": time_val, "Step": step, "Boundary_Spot": boundary}
                )

            # Prepare next level
            if step < self.tree.num_steps:
                next_level_nodes = set()
                for node in current_level_nodes:
                    if node.future_up:
                        next_level_nodes.add(node.future_up)
                    if node.future_center:
                        next_level_nodes.add(node.future_center)
                    if node.future_down:
                        next_level_nodes.add(node.future_down)
                current_level_nodes = next_level_nodes

        return pd.DataFrame(boundary_points)

    def _calculate_payoff(self, node: Node) -> float:
        # Accessing private method of Node as it encapsulates the logic
        return node._calculate_payoff()
