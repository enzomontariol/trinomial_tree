import plotly.graph_objects as go
import pandas as pd


class Visualizer:
    @staticmethod
    def plot_convergence_price(
        df: pd.DataFrame, title: str = "Convergence of Price vs N"
    ) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df["N"], y=df["Tree_Price"], mode="lines+markers", name="Tree Price"
            )
        )
        if "BS_Price" in df.columns:
            # Use the first BS price as it should be constant for convergence test
            bs_price = df["BS_Price"].iloc[0]
            fig.add_hline(
                y=bs_price,
                line_dash="dash",
                line_color="red",
                annotation_text="BS Price",
            )
        fig.update_layout(
            title=title, xaxis_title="Number of Steps (N)", yaxis_title="Price"
        )
        return fig

    @staticmethod
    def plot_convergence_error(
        df: pd.DataFrame, title: str = "Convergence Error vs N (Log-Log)"
    ) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=df["N"], y=df["Error"], mode="lines+markers", name="Error")
        )
        fig.update_layout(
            title=title,
            xaxis_title="Number of Steps (N)",
            yaxis_title="Absolute Error",
            xaxis_type="log",
            yaxis_type="log",
        )
        return fig

    @staticmethod
    def plot_runtime(df: pd.DataFrame, title: str = "Runtime vs N") -> go.Figure:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=df["N"], y=df["Runtime"], mode="lines+markers", name="Runtime")
        )
        fig.update_layout(
            title=title, xaxis_title="Number of Steps (N)", yaxis_title="Runtime (s)"
        )
        return fig

    @staticmethod
    def plot_price_vs_spot(
        df: pd.DataFrame, title: str = "Price vs Spot Price"
    ) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=df["S0"], y=df["Tree_Price"], mode="lines", name="Tree Price")
        )
        if "BS_Price" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["S0"],
                    y=df["BS_Price"],
                    mode="lines",
                    name="BS Price",
                    line=dict(dash="dash"),
                )
            )
        fig.update_layout(
            title=title, xaxis_title="Spot Price (S0)", yaxis_title="Price"
        )
        return fig

    @staticmethod
    def plot_greeks_vs_spot(
        df: pd.DataFrame, greek: str, title: str = None
    ) -> go.Figure:
        if title is None:
            title = f"{greek} vs Spot Price"
        fig = go.Figure()
        tree_col = f"Tree_{greek}"
        bs_col = f"BS_{greek}"

        if tree_col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["S0"], y=df[tree_col], mode="lines", name=f"Tree {greek} (FD)"
                )
            )

        # Add Internal Gamma if available and requested
        if greek == "Gamma" and "Tree_Gamma_Internal" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["S0"],
                    y=df["Tree_Gamma_Internal"],
                    mode="lines",
                    name="Tree Gamma (Internal)",
                    line=dict(dash="dot"),
                )
            )

        if bs_col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["S0"],
                    y=df[bs_col],
                    mode="lines",
                    name=f"BS {greek}",
                    line=dict(dash="dash"),
                )
            )

        fig.update_layout(title=title, xaxis_title="Spot Price (S0)", yaxis_title=greek)
        return fig

    @staticmethod
    def plot_price_vs_vol(
        df: pd.DataFrame, title: str = "Price vs Volatility"
    ) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df["Volatility"], y=df["Tree_Price"], mode="lines", name="Tree Price"
            )
        )
        if "BS_Price" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["Volatility"],
                    y=df["BS_Price"],
                    mode="lines",
                    name="BS Price",
                    line=dict(dash="dash"),
                )
            )
        fig.update_layout(title=title, xaxis_title="Volatility", yaxis_title="Price")
        return fig

    @staticmethod
    def plot_error_vs_vol(
        df: pd.DataFrame, title: str = "Pricing Error vs Volatility"
    ) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df["Volatility"], y=df["Error"], mode="lines+markers", name="Error"
            )
        )
        fig.update_layout(
            title=title, xaxis_title="Volatility", yaxis_title="Absolute Error"
        )
        return fig

    @staticmethod
    def plot_parity_residual(
        df: pd.DataFrame, x_col: str, title: str = "Put-Call Parity Residual"
    ) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df[x_col], y=df["Residual"], mode="lines+markers", name="Residual"
            )
        )
        fig.add_hline(y=0, line_color="black", line_width=1)
        fig.update_layout(
            title=title + " (Note: Scale is ~1e-12)",
            xaxis_title=x_col,
            yaxis_title="Residual (C - P - Parity)",
        )
        return fig

    @staticmethod
    def plot_exercise_boundary(
        df: pd.DataFrame, title: str = "Early Exercise Boundary"
    ) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df["Time"],
                y=df["Boundary_Spot"],
                mode="lines+markers",
                name="Exercise Boundary",
            )
        )
        fig.update_layout(
            title=title, xaxis_title="Time (Years)", yaxis_title="Critical Spot Price"
        )
        return fig

    @staticmethod
    def plot_terminal_distribution(
        df: pd.DataFrame, title: str = "Terminal Distribution"
    ) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(
            go.Bar(x=df["Spot"], y=df["Probability"], name="Tree Probability")
        )

        if "Theoretical_PDF" in df.columns:
            # Scale PDF to match probability mass roughly for visualization
            # Area of bar = prob. Area of PDF slice ~ PDF * width.
            # We can just plot PDF on secondary axis or just overlay.
            # Let's overlay on secondary axis to avoid scaling issues.
            fig.add_trace(
                go.Scatter(
                    x=df["Spot"],
                    y=df["Theoretical_PDF"],
                    mode="lines",
                    name="Theoretical PDF",
                    yaxis="y2",
                )
            )

            fig.update_layout(
                yaxis2=dict(title="PDF Density", overlaying="y", side="right")
            )

        fig.update_layout(
            title=title,
            xaxis_title="Spot Price at Maturity",
            yaxis_title="Probability Mass",
        )
        return fig
