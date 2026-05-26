"""Strategy implementations for the iterated Prisoner's Dilemma.

Each strategy is a callable object that maps a game history to the next
action.  The history is a list of ``(own_action, opponent_action)`` tuples.

Available strategies
--------------------
- ``AlwaysCooperate``  — unconditional cooperation.
- ``AlwaysDefect``     — unconditional defection.
- ``TitForTat``        — starts C, then mirrors opponent's last move.
- ``GrimTrigger``      — starts C, defects forever after first opponent D.

The module also exposes ``simulate`` which runs two strategies against each
other for a finite number of rounds and returns each player's discounted
cumulative payoff.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Protocol

from red_vs_blue.model import PayoffMatrix

# ---------------------------------------------------------------------------
# Action enum
# ---------------------------------------------------------------------------


class Action(Enum):
    """Possible actions in a single round of the Prisoner's Dilemma."""

    COOPERATE = auto()
    DEFECT = auto()

    def __str__(self) -> str:  # pragma: no cover
        return "C" if self is Action.COOPERATE else "D"


# ---------------------------------------------------------------------------
# Strategy protocol & base class
# ---------------------------------------------------------------------------

History = list[tuple[Action, Action]]  # [(own, opponent), ...]


class Strategy(Protocol):
    """Interface that every strategy must implement."""

    @property
    def name(self) -> str:
        """Human-readable strategy name."""
        ...

    def __call__(self, history: History) -> Action:
        """Return the action for the current round given the game history."""
        ...

    def reset(self) -> None:
        """Reset any internal state for a fresh game."""
        ...


# ---------------------------------------------------------------------------
# Concrete strategies
# ---------------------------------------------------------------------------


class AlwaysCooperate:
    """Always play Cooperate, regardless of history."""

    name: str = "Always Cooperate"

    def __call__(self, history: History) -> Action:  # noqa: ARG002
        return Action.COOPERATE

    def reset(self) -> None:
        pass


class AlwaysDefect:
    """Always play Defect, regardless of history."""

    name: str = "Always Defect"

    def __call__(self, history: History) -> Action:  # noqa: ARG002
        return Action.DEFECT

    def reset(self) -> None:
        pass


class TitForTat:
    """Cooperate on round 1, then copy the opponent's previous action."""

    name: str = "Tit-for-Tat"

    def __call__(self, history: History) -> Action:
        if not history:
            return Action.COOPERATE
        _, opponent_last = history[-1]
        return opponent_last

    def reset(self) -> None:
        pass


class GrimTrigger:
    """Cooperate until the opponent defects once, then defect forever."""

    name: str = "Grim Trigger"

    def __init__(self) -> None:
        self._triggered = False

    def __call__(self, history: History) -> Action:
        if self._triggered:
            return Action.DEFECT
        for _, opp_action in history:
            if opp_action is Action.DEFECT:
                self._triggered = True
                return Action.DEFECT
        return Action.COOPERATE

    def reset(self) -> None:
        self._triggered = False


# ---------------------------------------------------------------------------
# Simulation helper
# ---------------------------------------------------------------------------


def _stage_payoff(
    own: Action, opponent: Action, matrix: PayoffMatrix
) -> tuple[float, float]:
    """Return (own_payoff, opponent_payoff) for a single stage game."""
    if own is Action.COOPERATE and opponent is Action.COOPERATE:
        return matrix.R, matrix.R
    if own is Action.DEFECT and opponent is Action.COOPERATE:
        return matrix.T, matrix.S
    if own is Action.COOPERATE and opponent is Action.DEFECT:
        return matrix.S, matrix.T
    # Both defect
    return matrix.P, matrix.P


def simulate(
    strategy1: Strategy,
    strategy2: Strategy,
    matrix: PayoffMatrix,
    rounds: int = 100,
    delta: float = 0.9,
) -> tuple[float, float]:
    """Run a finite-horizon game between two strategies.

    Payoffs are discounted: round *t* (0-indexed) is multiplied by δᵗ.

    Args:
        strategy1: First player's strategy.
        strategy2: Second player's strategy.
        matrix:    Payoff matrix for stage-game outcomes.
        rounds:    Number of rounds to simulate.
        delta:     Discount factor applied to future payoffs (0 < δ < 1).

    Returns:
        ``(payoff1, payoff2)`` — discounted cumulative payoffs.

    Raises:
        ValueError: If *rounds* < 1 or *delta* is outside (0, 1).
    """
    if rounds < 1:
        raise ValueError(f"rounds must be ≥ 1; got {rounds}")
    if not (0.0 < delta < 1.0):
        raise ValueError(f"delta must be in (0, 1); got {delta}")

    strategy1.reset()
    strategy2.reset()

    history1: History = []  # (s1_action, s2_action) from s1's perspective
    history2: History = []  # (s2_action, s1_action) from s2's perspective

    total1 = total2 = 0.0
    discount = 1.0

    for _ in range(rounds):
        a1 = strategy1(history1)
        a2 = strategy2(history2)

        p1, p2 = _stage_payoff(a1, a2, matrix)
        total1 += discount * p1
        total2 += discount * p2

        history1.append((a1, a2))
        history2.append((a2, a1))
        discount *= delta

    return total1, total2
