import streamlit as st
from datetime import date, timedelta

from src.pricing import MarketData, Option
from src.analysis.experiments import (
    ConvergenceExperiment,
    SpotSensitivityExperiment,
    ExerciseBoundaryExperiment,
    TerminalDistributionExperiment,
)
from src.analysis import Visualizer

# Page Config
st.set_page_config(page_title="Trinomial Tree Explorer", layout="wide")

st.title("🌳 Trinomial Tree Option Pricer")
st.markdown("""
This dashboard validates the properties of a Trinomial Tree implementation.
It covers convergence, sensitivity (Greeks), and structural analysis (Exercise Boundary).
""")

# --- Sidebar: Global Parameters ---
st.sidebar.header("Option Parameters")
S0 = st.sidebar.number_input("Spot Price (S0)", value=100.0)
K = st.sidebar.number_input("Strike Price (K)", value=100.0)
T_years = st.sidebar.number_input("Time to Maturity (Years)", value=1.0)
r = st.sidebar.number_input("Risk-free Rate (r)", value=0.05)
sigma = st.sidebar.number_input("Volatility (σ)", value=0.2)
option_type = st.sidebar.selectbox(
    "Option Type", ["call", "put", "american_call", "american_put"], index=1
)


# Helper to create option and market data
def get_config():
    today = date.today()
    maturity = today + timedelta(days=int(T_years * 365))

    market_data = MarketData(
        spot_price=S0,
        start_date=today,
        volatility=sigma,
        interest_rate=r,
        discount_rate=r,
        dividend_ex_date=maturity + timedelta(days=1),  # No div
        dividend_amount=0.0,
    )

    is_am = "american" in option_type
    kind = "call" if "call" in option_type else "put"

    option = Option(
        maturity=maturity,
        strike_price=K,
        is_call=(kind == "call"),
        is_american=is_am,
        barrier=None,
        pricing_date=today,
        calendar_base_convention=365,
    )
    return market_data, option


# --- Tabs for Analysis ---
tab1, tab2, tab3 = st.tabs(
    ["1. Convergence", "2. Sensitivity & Greeks", "3. Structure & Boundary"]
)

# --- Tab 1: Convergence ---
with tab1:
    st.header("Convergence to Black-Scholes")
    st.markdown("Analyze how the price converges as the number of steps $N$ increases.")

    if st.button("Run Convergence Analysis"):
        with st.spinner("Running simulations..."):
            market_data, option = get_config()
            # Reuse your existing experiment logic
            N_values = [10, 20, 40, 80, 160, 320]
            try:
                exp = ConvergenceExperiment(N_values, market_data, option)
                results = exp.run()

                col1, col2 = st.columns(2)
                with col1:
                    fig_price = Visualizer.plot_convergence_price(results)
                    st.plotly_chart(fig_price, width="stretch")
                with col2:
                    fig_error = Visualizer.plot_convergence_error(results)
                    st.plotly_chart(fig_error, width="stretch")
            except ValueError as e:
                st.error(f"Unable to run convergence analysis: {e}")

# --- Tab 2: Sensitivity ---
with tab2:
    st.header("Sensitivity Analysis")
    st.markdown("Analyze Price, Delta, and Gamma vs Spot Price.")

    n_steps = st.slider("Number of Tree Steps (N)", 10, 500, 100)

    if st.button("Run Sensitivity Analysis"):
        with st.spinner("Calculating Greeks..."):
            market_data, option = get_config()

            # Generate spot range +/- 50%
            s_min = int(S0 * 0.5)
            s_max = int(S0 * 1.5)
            S0_values = [float(x) for x in range(s_min, s_max, 2)]

            exp = SpotSensitivityExperiment(S0_values, market_data, option, N=n_steps)
            results = exp.run()

            col1, col2 = st.columns(2)
            with col1:
                fig_price = Visualizer.plot_price_vs_spot(results)
                st.plotly_chart(fig_price, width="stretch")

                fig_gamma = Visualizer.plot_greeks_vs_spot(results, "Gamma")
                st.plotly_chart(fig_gamma, width="stretch")

            with col2:
                fig_delta = Visualizer.plot_greeks_vs_spot(results, "Delta")
                st.plotly_chart(fig_delta, width="stretch")

# --- Tab 3: Structure ---
with tab3:
    st.header("Tree Structure & Exercise Boundary")
    st.markdown(
        "Visualize the Early Exercise Boundary (American options) and Terminal Distribution."
    )

    n_steps_struct = st.slider("Tree Steps for Boundary", 50, 1000, 200, key="struct_n")

    if st.button("Inspect Tree Structure"):
        with st.spinner("Traversing Tree..."):
            market_data, option = get_config()

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Early Exercise Boundary")
                if option.is_american:
                    exp = ExerciseBoundaryExperiment(
                        market_data, option, N=n_steps_struct
                    )
                    res_boundary = exp.run()
                    fig_bound = Visualizer.plot_exercise_boundary(res_boundary)
                    st.plotly_chart(fig_bound, width="stretch")
                else:
                    st.info("Exercise Boundary is only available for American Options.")

            with col2:
                st.subheader("Terminal Distribution")
                # Use fewer steps for distribution to make bars visible
                exp_dist = TerminalDistributionExperiment(market_data, option, N=50)
                res_dist = exp_dist.run()
                fig_dist = Visualizer.plot_terminal_distribution(res_dist)
                st.plotly_chart(fig_dist, width="stretch")
