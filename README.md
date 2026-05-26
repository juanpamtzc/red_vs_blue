# Blue vs. Red: Game-Theoretic Voting Strategy Optimization

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://<YOUR-APP-NAME>.streamlit.app/)
[![Python CI](https://github.com/<YOUR-GITHUB-USERNAME>/<YOUR-REPO-NAME>/actions/workflows/python-app.yml/badge.svg)](https://github.com/<YOUR-GITHUB-USERNAME>/<YOUR-REPO-NAME>/actions)

An interactive computational framework and expected utility optimizer for population-scale choice dynamics under existential risk. This project bridges rigorous game-theoretic modeling with production-ready software architecture.

---

## 📋 Problem Statement

Given a global population $N$, an individual must choose between two discrete strategies: **Red (R)** or **Blue (B)**. Let $n$ represent the total number of individuals who select Red. The outcome function determines survival based on a majority threshold:

* **Sub-critical Regime ($\frac{n}{N} \le 0.5$):** Stable state. Nobody dies ($N$ people live).
* **Super-critical Regime ($\frac{n}{N} > 0.5$):** Catastrophic state. All individuals who voted Blue perish ($N-n$ people die).

### Objective Function
An individual optimizes their strategy based on a subjective belief $\gamma$ (the perceived probability that any other independent actor chooses Red) and two preference weights:
1.  $\phi_1$: Valuing personal self-preservation (living).
2.  $\phi_2$: Valuing collective preservation (minimizing casualties inflicted on others).

This application computes the expected utility differential $\Delta E = E[\text{Red}] - E[\text{Blue}]$ across the parameter space to identify the optimal strategic choice.

---

## 🔬 Mathematical Formulation

The system evaluates expected utilities by modeling the actions of the remaining $N-1$ agents as a binomial distribution $X \sim \text{Binomial}(N-1, \gamma)$. 

Let $k$ be the realized number of other agents voting Red. The expected utility equations are integrated numerically over the probability mass function:

$$E[\text{Red}] = \sum_{k=0}^{N-1} \binom{N-1}{k} \gamma^k (1-\gamma)^{N-1-k} \cdot U_R(k)$$

$$E[\text{Blue}] = \sum_{k=0}^{N-1} \binom{N-1}{k} \gamma^k (1-\gamma)^{N-1-k} \cdot U_B(k)$$

The codebase utilizes vectorized `NumPy` broadcasting and exact binomial probability mass functions from `SciPy` to ensure numerical stability and $O(N)$ computational complexity. Analytical limits and grid convergence validations are detailed in the accompanying research notebook.

---

## 🗂️ Repository Structure

```text
├── .github/workflows/
│   └── python-app.yml    # CI/CD pipeline (automated testing via GitHub Actions)
├── notebooks/
│   └── blue_vs_red.ipynb # Original research scratchpad & grid convergence analysis
├── src/
│   ├── __init__.py
│   └── game_theory.py    # Core deterministic math engine (pure Python/NumPy)
├── tests/
│   ├── __init__.py
│   └── test_game_theory.py # Unit tests for numerical invariant verification
├── app.py                # Streamlit reactive presentation UI
└── requirements.txt      # Bounded production dependencies
