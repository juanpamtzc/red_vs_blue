"""Streamlit dashboard — Red vs Blue: Prisoner's Dilemma Optimizer.

Security measures applied:
  - All user inputs are bounded numeric sliders (no free-text injection).
  - No dynamic code execution (no eval/exec).
  - No file upload surface.
  - Inputs are re-validated server-side before any computation.
  - Error messages never expose internal state.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the src package is importable when run via `streamlit run app/app.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from red_vs_blue.model import (
    PayoffMatrix,
    compute_critical_delta,
    compute_decision_boundary,
    get_optimal_action,
)
from red_vs_blue.strategies import (
    AlwaysCooperate,
    AlwaysDefect,
    GrimTrigger,
    TitForTat,
    simulate,
)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Red vs Blue · Prisoner's Dilemma",
    page_icon="⚔️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Optimal strategy solver for the iterated Prisoner's Dilemma.",
        "Report a bug": None,
        "Get help": None,
    },
)

# ---------------------------------------------------------------------------
# Sidebar — game parameters
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ Game Parameters")
    st.caption(
        "Adjust the payoff values and your discount factor to explore the "
        "decision landscape."
    )
    st.divider()

    st.subheader("Payoff Matrix")
    st.markdown(
        "Standard ordering required: **T > R > P > S** and **2R > T + S**",
        help="These inequalities define a Prisoner's Dilemma. "
        "If violated, the game changes character.",
    )

    T = st.slider(
        "T — Temptation (defect while other cooperates)",
        min_value=1.0,
        max_value=10.0,
        value=5.0,
        step=0.1,
        format="%.1f",
    )
    R = st.slider(
        "R — Reward (mutual cooperation)",
        min_value=0.1,
        max_value=9.9,
        value=3.0,
        step=0.1,
        format="%.1f",
    )
    P = st.slider(
        "P — Punishment (mutual defection)",
        min_value=0.0,
        max_value=9.8,
        value=1.0,
        step=0.1,
        format="%.1f",
    )
    S = st.slider(
        "S — Sucker (cooperate while other defects)",
        min_value=-5.0,
        max_value=9.7,
        value=0.0,
        step=0.1,
        format="%.1f",
    )

    st.divider()
    st.subheader("Your State")
    delta = st.slider(
        "δ — Discount factor (how much you value the future)",
        min_value=0.01,
        max_value=0.99,
        value=0.80,
        step=0.01,
        format="%.2f",
        help="δ close to 1 means you highly value future payoffs; "
        "δ close to 0 means you are nearly myopic.",
    )

    st.divider()
    st.subheader("Simulation")
    sim_rounds = st.slider("Simulation rounds", min_value=10, max_value=500, value=100, step=10)

# ---------------------------------------------------------------------------
# Input validation (server-side)
# ---------------------------------------------------------------------------

# Clamp inputs to safe numeric ranges — belt-and-suspenders after slider bounds
T = float(np.clip(T, 0.001, 1000.0))
R = float(np.clip(R, 0.001, 1000.0))
P = float(np.clip(P, -1000.0, 1000.0))
S = float(np.clip(S, -1000.0, 1000.0))
delta = float(np.clip(delta, 1e-6, 1 - 1e-6))
sim_rounds = int(np.clip(sim_rounds, 1, 10_000))

try:
    matrix = PayoffMatrix(T=T, R=R, P=P, S=S)
    matrix_valid = True
except (ValueError, TypeError):
    matrix_valid = False

# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

st.title("⚔️ Red vs Blue: Prisoner's Dilemma")
st.markdown(
    "Discover the **optimal strategy** for the infinitely-repeated Prisoner's "
    "Dilemma — a model for conflict, cooperation, and cybersecurity decisions."
)

if not matrix_valid:
    st.error(
        "⚠️ The current payoff values do not form a valid Prisoner's Dilemma.  "
        "Please ensure **T > R > P > S** and **2R > T + S**."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Optimal-action banner
# ---------------------------------------------------------------------------

critical_delta = compute_critical_delta(matrix)
action = get_optimal_action(matrix, delta)
margin = delta - critical_delta

col_action, col_metric1, col_metric2 = st.columns([2, 1, 1])

with col_action:
    if action == "Cooperate":
        st.success("## ✅ Optimal Action: **COOPERATE**")
    else:
        st.error("## 🚫 Optimal Action: **DEFECT**")

with col_metric1:
    st.metric(
        label="Critical δ*",
        value=f"{critical_delta:.3f}",
        help="Minimum discount factor required to sustain cooperation.",
    )

with col_metric2:
    st.metric(
        label="δ − δ* (margin)",
        value=f"{margin:+.3f}",
        delta=f"{margin:+.3f}",
        help="Positive = cooperation is sustainable; negative = defection optimal.",
    )

st.divider()

# ---------------------------------------------------------------------------
# Decision boundary plot
# ---------------------------------------------------------------------------

tab_boundary, tab_strategies, tab_theory = st.tabs(
    ["📊 Decision Boundary", "🎮 Strategy Comparison", "📖 Theory"]
)

with tab_boundary:
    st.subheader("Decision Boundary in (T, δ) Space")
    st.markdown(
        "The blue region shows where **cooperation is the optimal strategy** "
        "(δ ≥ δ*). The red region shows where **defection dominates** (δ < δ*). "
        "The ⭐ marks your current game state."
    )

    T_grid, delta_grid, coop_mask = compute_decision_boundary(
        matrix,
        t_range=(max(matrix.R + 0.05, 1.0), 10.0),
        delta_range=(0.01, 0.99),
        resolution=150,
    )

    # Analytical boundary curve: δ* = (T - R) / (T - P)
    t_curve = np.linspace(max(matrix.R + 0.01, 1.0), 10.0, 400)
    with np.errstate(divide="ignore", invalid="ignore"):
        delta_curve = (t_curve - matrix.R) / (t_curve - matrix.P)
    valid_mask = (delta_curve > 0) & (delta_curve < 1)

    fig = go.Figure()

    # Heatmap — cooperation mask
    fig.add_trace(
        go.Heatmap(
            x=T_grid[0, :],
            y=delta_grid[:, 0],
            z=coop_mask.astype(int),
            colorscale=[[0, "#c0392b"], [1, "#2980b9"]],
            showscale=False,
            opacity=0.35,
            hoverinfo="skip",
        )
    )

    # Boundary curve
    fig.add_trace(
        go.Scatter(
            x=t_curve[valid_mask],
            y=delta_curve[valid_mask],
            mode="lines",
            name="Decision Boundary δ*",
            line={"color": "white", "width": 2.5, "dash": "dash"},
        )
    )

    # Current state marker
    fig.add_trace(
        go.Scatter(
            x=[T],
            y=[delta],
            mode="markers+text",
            name="Your State",
            marker={"symbol": "star", "size": 18, "color": "gold", "line": {"width": 1.5, "color": "black"}},
            text=["You"],
            textposition="top center",
            textfont={"color": "gold", "size": 13},
        )
    )

    fig.update_layout(
        xaxis_title="T — Temptation",
        yaxis_title="δ — Discount Factor",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
        margin={"l": 60, "r": 20, "t": 40, "b": 60},
        height=480,
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font={"color": "white"},
        xaxis={"gridcolor": "#333"},
        yaxis={"gridcolor": "#333"},
    )

    # Annotation labels
    fig.add_annotation(x=9.0, y=0.9, text="COOPERATE", font={"color": "#74b9ff", "size": 14}, showarrow=False)
    fig.add_annotation(x=9.0, y=0.1, text="DEFECT", font={"color": "#ff7675", "size": 14}, showarrow=False)

    st.plotly_chart(fig, use_container_width=True)

    # Payoff table
    with st.expander("📋 Current Payoff Matrix"):
        st.table(
            {
                "": ["Opponent Cooperates", "Opponent Defects"],
                "You Cooperate": [f"R = {matrix.R}", f"S = {matrix.S}"],
                "You Defect": [f"T = {matrix.T}", f"P = {matrix.P}"],
            }
        )

# ---------------------------------------------------------------------------
# Strategy comparison tab
# ---------------------------------------------------------------------------

with tab_strategies:
    st.subheader("Strategy Payoff Comparison")
    st.markdown(
        f"Simulating **{sim_rounds} rounds** (δ = {delta:.2f}) for each strategy "
        f"pair.  Higher discounted payoff = better outcome."
    )

    strategies = [AlwaysCooperate(), AlwaysDefect(), TitForTat(), GrimTrigger()]
    strategy_names = [s.name for s in strategies]

    # Build payoff matrix (row = strategy i, col = strategy j → payoff for i)
    payoff_data: list[list[float]] = []
    for s1 in strategies:
        row = []
        for s2 in strategies:
            p1, _ = simulate(s1, s2, matrix, rounds=sim_rounds, delta=delta)
            row.append(round(p1, 2))
        payoff_data.append(row)

    fig2 = go.Figure(
        data=go.Heatmap(
            z=payoff_data,
            x=strategy_names,
            y=strategy_names,
            colorscale="RdBu",
            text=[[f"{v:.1f}" for v in row] for row in payoff_data],
            texttemplate="%{text}",
            textfont={"size": 12},
            hovertemplate="Row plays %{y}<br>Col plays %{x}<br>Row payoff: %{z:.2f}<extra></extra>",
        )
    )
    fig2.update_layout(
        xaxis_title="Opponent Strategy",
        yaxis_title="Your Strategy",
        height=400,
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font={"color": "white"},
        margin={"l": 140, "r": 20, "t": 40, "b": 100},
        xaxis={"tickangle": -20},
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(
        "Cell value = **row player's discounted payoff** when row strategy "
        "faces column strategy."
    )

# ---------------------------------------------------------------------------
# Theory tab
# ---------------------------------------------------------------------------

with tab_theory:
    st.subheader("📖 The Folk Theorem & Decision Boundary")
    st.markdown(
        r"""
### Prisoner's Dilemma Setup

Two players simultaneously choose to **Cooperate (C)** or **Defect (D)**.
Stage-game payoffs:

| | Opponent C | Opponent D |
|---|---|---|
| **You C** | R (Reward) | S (Sucker) |
| **You D** | T (Temptation) | P (Punishment) |

with the constraint **T > R > P > S** and **2R > T + S**.

### Infinitely Repeated Game

When the game is repeated infinitely, players discount future payoffs by
factor δ ∈ (0, 1) per period. The discounted payoff of always cooperating
(when the opponent also cooperates) is:

$$V_C = \frac{R}{1 - \delta}$$

If you deviate (defect) and the opponent punishes forever (Grim Trigger):

$$V_D = T + \frac{\delta P}{1 - \delta}$$

### Decision Boundary

Cooperation is optimal if $V_C \geq V_D$, which simplifies to:

$$\delta \geq \delta^* = \frac{T - R}{T - P}$$

The dashboard plots this boundary in (T, δ) space with your current state
marked as ⭐.
"""
    )
