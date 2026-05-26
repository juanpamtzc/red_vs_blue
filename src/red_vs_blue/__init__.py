"""Red vs Blue — Prisoner's Dilemma optimal strategy package."""

from red_vs_blue.model import (
    PayoffMatrix,
    compute_critical_delta,
    compute_decision_boundary,
    get_optimal_action,
    is_cooperation_sustainable,
)
from red_vs_blue.strategies import (
    Action,
    AlwaysCooperate,
    AlwaysDefect,
    GrimTrigger,
    Strategy,
    TitForTat,
    simulate,
)

__all__ = [
    # model
    "PayoffMatrix",
    "compute_critical_delta",
    "compute_decision_boundary",
    "get_optimal_action",
    "is_cooperation_sustainable",
    # strategies
    "Action",
    "AlwaysCooperate",
    "AlwaysDefect",
    "GrimTrigger",
    "Strategy",
    "TitForTat",
    "simulate",
]
