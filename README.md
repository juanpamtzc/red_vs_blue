# Red vs Blue — Prisoner's Dilemma Optimizer

[![CI](https://github.com/juanpamtzc/red_vs_blue/actions/workflows/ci.yml/badge.svg)](https://github.com/juanpamtzc/red_vs_blue/actions/workflows/ci.yml)

Optimal strategy solver for the **infinitely-repeated Prisoner's Dilemma**, with an
interactive [Streamlit](https://streamlit.io) dashboard that displays the decision
boundary and tells you whether to **Cooperate** or **Defect** for any game state you choose.

## Project Structure

```
red_vs_blue/
├── .github/workflows/ci.yml   # GitHub Actions CI pipeline
├── app/
│   └── app.py                 # Streamlit dashboard
├── notebooks/
│   └── analysis.ipynb         # Step-by-step mathematical analysis
├── src/
│   └── red_vs_blue/
│       ├── __init__.py
│       ├── model.py            # PayoffMatrix, decision boundary, optimal action
│       └── strategies.py       # AlwaysCooperate, AlwaysDefect, TitForTat, GrimTrigger
├── tests/
│   ├── test_model.py
│   └── test_strategies.py
├── pyproject.toml
└── requirements.txt
```

## Quick Start

```bash
# 1. Install
pip install -e ".[dev]"

# 2. Run tests
pytest

# 3. Launch dashboard
streamlit run app/app.py
```

## Core Concept

In the infinitely-repeated Prisoner's Dilemma, mutual cooperation is a Nash
Equilibrium if and only if players are sufficiently patient:

```
δ  ≥  δ*  =  (T − R) / (T − P)
```

The dashboard visualises this boundary in **(T, δ) space** and marks your
current game state with ⭐.

### Payoff Matrix

|             | Opponent **C** | Opponent **D** |
|-------------|---------------|---------------|
| **You C**   | R (Reward)    | S (Sucker)    |
| **You D**   | T (Temptation)| P (Punishment)|

Standard ordering required: **T > R > P > S** and **2R > T + S**.

## Running Tests

```bash
pytest tests/ -v --cov=red_vs_blue
```

## Linting

```bash
ruff check src/ tests/ app/
```