import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Import the core logic from your src folder
from src.game_theory import evaluate_voting_strategy, compute_sweep
from src.plotting import plot_sweep

# 1. Setup the Page
st.set_page_config(page_title="Blue vs. Red Optimizer", page_icon="🔵", layout="wide")
# Suppress Streamlit's warning about Matplotlib global figures
st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("Blue vs. Red: Expected Utility Optimizer")
st.markdown("""
This application models the optimal strategic choice in a population-scale survival scenario. 
Adjust the parameters in the sidebar to compute the utility difference $dU = U(R) - U(B)$ and view the parameter sweep.
""")

# 2. Sidebar for Inputs (Strictly Bounded Security)
with st.sidebar:
    st.header("Model Parameters")
    
    # N is strictly bounded to an integer >= 2
    N = st.number_input("Total Population (N)", min_value=2, max_value=100000, value=100, step=1)
    
    # Gamma is strictly bounded as a probability [0.0, 1.0]
    gamma = st.slider("Belief of Red Voters % (γ)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    
    # Weights for phi1 and phi2
    phi1 = st.slider("Weight: Your Own Life (φ1)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    phi2 = st.slider("Weight: Other Lives Saved (φ2)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

# 3. Execution Layer 
try:
    # --- Single-Point Strategic Evaluation ---
    st.subheader("1. Strategic Decision Output")
    
    # Calculate expected utilities using your game_theory logic
    result = evaluate_voting_strategy(N=N, phi1=phi1, phi2=phi2, gamma=gamma)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Expected Utility (Red)", f"{result['Expected_Utility_Red']:.4f}")
    col2.metric("Expected Utility (Blue)", f"{result['Expected_Utility_Blue']:.4f}")
    
    decision = result["Optimal_Choice"]
    delta_e = result["Delta_E"]
    
    # Color-code the output based on the analytical result
    if decision == "Vote RED":
        col3.error(f"Optimal Choice: **{decision}** (ΔE = {delta_e:.4f})")
    elif decision == "Vote BLUE":
        col3.info(f"Optimal Choice: **{decision}** (ΔE = {delta_e:.4f})")
    else:
        col3.warning(f"Optimal Choice: **{decision}** (ΔE = {delta_e:.4f})")

    # --- Parameter Space Sweep Visualization ---
    st.divider()
    st.subheader("2. Parameter Space Sweep")
    st.write("Visualizing the expected utility difference across a grid of beliefs (γ) and weight ratios (φ1/φ2).")
    
    # Compute the 2D grid data
    sweep_data = compute_sweep(N=N, resolution=100, ratio_min=0.1, ratio_max=5.0, ratio_scale="linear")
    
    # Call your plotting logic. Because plot_sweep generates a figure and calls plt.show(), 
    # st.pyplot() captures that active global figure and renders it safely to the web.
    plot_sweep(sweep_data)
    st.pyplot()

except ValueError as e:
    st.error(f"Input Error: {e}")
except Exception as e:
    st.error(f"An unexpected error occurred during execution.")