import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Import the core logic from your src folder
from src.game_theory import evaluate_voting_strategy, compute_sweep
from src.plotting import plot_sweep

# 1. Setup the Page
st.set_page_config(page_title="Blue vs. Red Optimizer", page_icon="🔵", layout="wide")

st.title("Blue vs. Red: Expected Utility Optimizer")

# --- THE LORE & PROBLEM STATEMENT ---
with st.expander("📖 The Origin Story: A Bachelor Party Thought Experiment"):
    st.write("""
    **It started exactly how you'd expect: at a bachelor party.** Amidst the tabletop games and late-night conversations, someone posed a hypothetical survival scenario. What started as a casual ethical debate quickly escalated into a fierce argument about game theory, trust, and optimal strategy. 
    
    The debate was so engaging—and the answers so highly dependent on individual assumptions about human nature—that I decided to formalize it mathematically. This dashboard is the result: a vectorized computational model of a late-night philosophical argument.
    """)

st.markdown("""
### The Problem Statement
Imagine a scenario where a global population of $N$ people must secretly choose to press one of two buttons: **Red** or **Blue**. 
* If the fraction of people who vote Red stays at or below a certain threshold ($\\tau$), **nobody dies**.
* If the fraction of Red voters exceeds $\\tau$, **everyone who voted Blue dies**.

Assuming you assign a specific weight to your own survival ($\phi_1$) and a specific weight to not getting others killed ($\phi_2$), what is the optimal choice based on your belief of what everyone else will do?
""")
st.divider()

# 2. Sidebar for Inputs (Strictly Bounded Security with Layman's Terms)
with st.sidebar:
    st.header("Model Parameters")
    
    # Layman's Terms Guide
    with st.expander("ℹ️ What do these mean?"):
        st.write("""
        * **N**: The total number of people voting.
        * **$\\tau$ (Threshold)**: The tipping point. If the crowd voting Red passes this percentage, Blue voters are eliminated.
        * **$\gamma$ (Belief)**: Your personal paranoia level. What percentage of the population do you think will vote Red?
        * **$\phi_1$ (Self)**: How much do you value your own life?
        * **$\phi_2$ (Others)**: How much do you value saving others?
        """)
    
    # Inputs with `help` tooltips attached directly to them
    N = st.number_input(
        "Total Population (N)", 
        min_value=2, max_value=100000, value=100, step=1,
        help="The total number of people participating in the scenario."
    )

    tau = st.slider(
        "Threshold of Voters % (τ)", 
        min_value=0.0, max_value=1.0, value=0.5, step=0.01,
        help="The percentage of Red votes required to trigger the elimination of Blue voters."
    )
    
    gamma = st.slider(
        "Belief of Red Voters % (γ)", 
        min_value=0.0, max_value=1.0, value=0.5, step=0.01,
        help="What percentage of the population do you expect will vote Red?"
    )
    
    st.markdown("### Utility Weights")
    phi1 = st.slider(
        "Weight: Your Own Life (φ1)", 
        min_value=0.0, max_value=10.0, value=1.0, step=0.1,
        help="Multiplier for the value of your own survival."
    )
    
    phi2 = st.slider(
        "Weight: Other Lives Saved (φ2)", 
        min_value=0.0, max_value=10.0, value=1.0, step=0.1,
        help="Multiplier for the value of saving the rest of the population."
    )

# 3. Execution Layer 
try:
    # --- Single-Point Strategic Evaluation ---
    st.subheader("Strategic Decision Output")
    
    # Calculate expected utilities using your game_theory logic
    result = evaluate_voting_strategy(N=N, phi1=phi1, phi2=phi2, gamma=gamma, threshold=tau)
    
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
    st.subheader("Parameter Space Sweep")
    st.write("Visualizing the expected utility difference $\Delta E = U(R) - U(B)$ across a grid of beliefs ($\gamma$) and weight ratios ($\phi_1/\phi_2$).")
    
    # Compute the 2D grid data
    sweep_data = compute_sweep(N=N, resolution=100, ratio_min=0.1, ratio_max=5.0, ratio_scale="linear", threshold=tau)
    
    # Call your plotting logic
    plot_sweep(sweep_data)
    
    # THREAD-SAFE FIX: Grab the current global figure and pass it explicitly to Streamlit
    fig = plt.gcf()
    st.pyplot(fig)

except ValueError as e:
    st.error(f"Input Error: {e}")
except Exception as e:
    st.error("An unexpected error occurred during execution.")