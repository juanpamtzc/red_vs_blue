"""Core model for the iterated Prisoner's Dilemma optimal-strategy analysis.

In the infinitely-repeated Prisoner's Dilemma the Folk Theorem guarantees that
mutual cooperation can be sustained as a Nash Equilibrium if and only if the
players are sufficiently patient, i.e. their discount factor δ satisfies:

    δ ≥ δ*  where  δ* = (T - R) / (T - P)

This module exposes:
  - ``PayoffMatrix``              — validated game parameters
  - ``compute_critical_delta``   — analytical threshold δ*
  - ``is_cooperation_sustainable`` — boolean test for a given δ
  - ``get_optimal_action``       — "Cooperate" / "Defect" recommendation
  - ``compute_decision_boundary`` — NumPy grid for visualisation
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Payoff matrix
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PayoffMatrix:
    """Payoff values for a single-stage Prisoner's Dilemma game.

    Standard ordering (required):
        T > R > P > S   and   2R > T + S

    Attributes:
        T: Temptation — unilateral defection payoff for the defector.
        R: Reward     — payoff when both players cooperate.
        P: Punishment — payoff when both players defect.
        S: Sucker     — payoff for cooperating against a defector.
    """

    T: float
    R: float
    P: float
    S: float

    def __post_init__(self) -> None:
        for name, val in (("T", self.T), ("R", self.R), ("P", self.P), ("S", self.S)):
            if not isinstance(val, (int, float)):
                raise TypeError(f"Payoff '{name}' must be a number, got {type(val).__name__}")
        if not (self.T > self.R > self.P > self.S):
            raise ValueError(
                f"Payoff matrix must satisfy T > R > P > S; "
                f"got T={self.T}, R={self.R}, P={self.P}, S={self.S}"
            )
        if not (2 * self.R > self.T + self.S):
            raise ValueError(
                f"Social optimum condition 2R > T + S is violated; "
                f"got 2R={2 * self.R:.3g}, T+S={self.T + self.S:.3g}"
            )


# ---------------------------------------------------------------------------
# Analytical results
# ---------------------------------------------------------------------------


def compute_critical_delta(matrix: PayoffMatrix) -> float:
    """Return the critical discount factor δ* = (T − R) / (T − P).

    Cooperation is sustainable for any δ ≥ δ*.

    Args:
        matrix: Validated payoff matrix.

    Returns:
        δ* in the open interval (0, 1).
    """
    return (matrix.T - matrix.R) / (matrix.T - matrix.P)


def is_cooperation_sustainable(matrix: PayoffMatrix, delta: float) -> bool:
    """Return ``True`` when cooperation is a Nash Equilibrium strategy.

    Args:
        matrix: Validated payoff matrix.
        delta:  Discount factor, must be in the open interval (0, 1).

    Returns:
        ``True`` if δ ≥ δ*, ``False`` otherwise.

    Raises:
        ValueError: If *delta* is outside (0, 1).
    """
    if not (0.0 < delta < 1.0):
        raise ValueError(f"Discount factor δ must be in (0, 1); got {delta}")
    return delta >= compute_critical_delta(matrix)


def get_optimal_action(matrix: PayoffMatrix, delta: float) -> str:
    """Return the optimal action label for the given game state.

    Args:
        matrix: Validated payoff matrix.
        delta:  Discount factor in (0, 1).

    Returns:
        ``"Cooperate"`` if δ ≥ δ*, ``"Defect"`` otherwise.
    """
    return "Cooperate" if is_cooperation_sustainable(matrix, delta) else "Defect"


# ---------------------------------------------------------------------------
# Decision-boundary grid
# ---------------------------------------------------------------------------


def compute_decision_boundary(
    matrix: PayoffMatrix,
    t_range: tuple[float, float] = (1.0, 10.0),
    delta_range: tuple[float, float] = (0.01, 0.99),
    resolution: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a 2-D grid showing the cooperation/defection regions.

    The grid varies T (temptation) on the x-axis and δ on the y-axis while
    keeping R, P, S fixed at the values stored in *matrix*.  Each cell is
    ``True`` where cooperation is the optimal strategy.

    Args:
        matrix:      Base payoff matrix providing R, P, S.
        t_range:     (min_T, max_T) for the x-axis.  Both values must be
                     strictly greater than ``matrix.R``.
        delta_range: (min_δ, max_δ) for the y-axis.
        resolution:  Number of grid points along each axis.

    Returns:
        A 3-tuple ``(T_grid, delta_grid, cooperation_mask)`` where the first
        two are ``(resolution, resolution)`` float arrays and the last is a
        boolean array of the same shape.
    """
    t_values = np.linspace(t_range[0], t_range[1], resolution)
    delta_values = np.linspace(delta_range[0], delta_range[1], resolution)
    T_grid, delta_grid = np.meshgrid(t_values, delta_values)

    # Analytical boundary: δ* = (T - R) / (T - P)
    # Valid only where T > R (otherwise not a PD)
    with np.errstate(divide="ignore", invalid="ignore"):
        critical_delta = np.where(
            T_grid > matrix.R,
            (T_grid - matrix.R) / (T_grid - matrix.P),
            np.nan,
        )

    cooperation_mask = delta_grid >= critical_delta

    return T_grid, delta_grid, cooperation_mask
