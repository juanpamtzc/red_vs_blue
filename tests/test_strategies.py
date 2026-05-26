"""Unit tests for red_vs_blue.strategies."""

import pytest

from red_vs_blue.model import PayoffMatrix
from red_vs_blue.strategies import (
    Action,
    AlwaysCooperate,
    AlwaysDefect,
    GrimTrigger,
    TitForTat,
    simulate,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def matrix() -> PayoffMatrix:
    return PayoffMatrix(T=5.0, R=3.0, P=1.0, S=0.0)


# ---------------------------------------------------------------------------
# Action enum
# ---------------------------------------------------------------------------

class TestAction:
    def test_two_actions_exist(self):
        assert Action.COOPERATE is not Action.DEFECT

    def test_membership(self):
        actions = list(Action)
        assert Action.COOPERATE in actions
        assert Action.DEFECT in actions


# ---------------------------------------------------------------------------
# AlwaysCooperate
# ---------------------------------------------------------------------------

class TestAlwaysCooperate:
    def test_cooperates_with_empty_history(self):
        s = AlwaysCooperate()
        assert s([]) is Action.COOPERATE

    def test_cooperates_regardless_of_history(self):
        s = AlwaysCooperate()
        history = [(Action.DEFECT, Action.DEFECT)] * 5
        assert s(history) is Action.COOPERATE

    def test_reset_is_noop(self):
        s = AlwaysCooperate()
        s.reset()
        assert s([]) is Action.COOPERATE

    def test_has_name(self):
        assert isinstance(AlwaysCooperate.name, str)
        assert len(AlwaysCooperate.name) > 0


# ---------------------------------------------------------------------------
# AlwaysDefect
# ---------------------------------------------------------------------------

class TestAlwaysDefect:
    def test_defects_with_empty_history(self):
        s = AlwaysDefect()
        assert s([]) is Action.DEFECT

    def test_defects_regardless_of_history(self):
        s = AlwaysDefect()
        history = [(Action.COOPERATE, Action.COOPERATE)] * 5
        assert s(history) is Action.DEFECT

    def test_has_name(self):
        assert isinstance(AlwaysDefect.name, str)


# ---------------------------------------------------------------------------
# TitForTat
# ---------------------------------------------------------------------------

class TestTitForTat:
    def test_cooperates_on_first_move(self):
        s = TitForTat()
        assert s([]) is Action.COOPERATE

    def test_mirrors_cooperation(self):
        s = TitForTat()
        history = [(Action.COOPERATE, Action.COOPERATE)]
        assert s(history) is Action.COOPERATE

    def test_mirrors_defection(self):
        s = TitForTat()
        history = [(Action.COOPERATE, Action.DEFECT)]
        assert s(history) is Action.DEFECT

    def test_forgives_after_cooperation(self):
        s = TitForTat()
        history = [
            (Action.COOPERATE, Action.DEFECT),  # opponent defected
            (Action.DEFECT, Action.COOPERATE),  # opponent cooperated again
        ]
        assert s(history) is Action.COOPERATE

    def test_reset_does_not_break_strategy(self):
        s = TitForTat()
        s.reset()
        assert s([]) is Action.COOPERATE


# ---------------------------------------------------------------------------
# GrimTrigger
# ---------------------------------------------------------------------------

class TestGrimTrigger:
    def test_cooperates_on_first_move(self):
        s = GrimTrigger()
        assert s([]) is Action.COOPERATE

    def test_cooperates_while_opponent_cooperates(self):
        s = GrimTrigger()
        history = [(Action.COOPERATE, Action.COOPERATE)] * 10
        assert s(history) is Action.COOPERATE

    def test_triggers_on_first_defection(self):
        s = GrimTrigger()
        history = [
            (Action.COOPERATE, Action.COOPERATE),
            (Action.COOPERATE, Action.DEFECT),  # trigger here
        ]
        assert s(history) is Action.DEFECT

    def test_remains_defecting_after_trigger(self):
        s = GrimTrigger()
        history = [
            (Action.COOPERATE, Action.DEFECT),
            (Action.DEFECT, Action.COOPERATE),  # opponent tries to cooperate
            (Action.DEFECT, Action.COOPERATE),
        ]
        assert s(history) is Action.DEFECT

    def test_reset_clears_trigger(self):
        s = GrimTrigger()
        history = [(Action.COOPERATE, Action.DEFECT)]
        s(history)  # trigger
        s.reset()
        assert s([]) is Action.COOPERATE


# ---------------------------------------------------------------------------
# simulate
# ---------------------------------------------------------------------------

class TestSimulate:
    def test_mutual_cooperation_payoff(self, matrix):
        """AlwaysCooperate vs AlwaysCooperate: discounted sum of R."""
        rounds, delta = 100, 0.9
        p1, p2 = simulate(AlwaysCooperate(), AlwaysCooperate(), matrix, rounds, delta)
        expected = sum(matrix.R * delta**t for t in range(rounds))
        assert abs(p1 - expected) < 1e-9
        assert abs(p2 - expected) < 1e-9

    def test_mutual_defection_payoff(self, matrix):
        """AlwaysDefect vs AlwaysDefect: discounted sum of P."""
        rounds, delta = 100, 0.9
        p1, p2 = simulate(AlwaysDefect(), AlwaysDefect(), matrix, rounds, delta)
        expected = sum(matrix.P * delta**t for t in range(rounds))
        assert abs(p1 - expected) < 1e-9
        assert abs(p2 - expected) < 1e-9

    def test_defector_beats_cooperator(self, matrix):
        """A defector exploiting a cooperator earns more than the cooperator."""
        p_defect, p_coop = simulate(AlwaysDefect(), AlwaysCooperate(), matrix, rounds=100, delta=0.9)
        assert p_defect > p_coop

    def test_tit_for_tat_vs_always_cooperate(self, matrix):
        """TfT vs AlwaysCooperate — both cooperate every round."""
        p_tft, p_ac = simulate(TitForTat(), AlwaysCooperate(), matrix, rounds=50, delta=0.9)
        expected = sum(matrix.R * 0.9**t for t in range(50))
        assert abs(p_tft - expected) < 1e-9

    def test_tit_for_tat_vs_always_defect(self, matrix):
        """TfT cooperates first then mirrors — first round S, rest P."""
        p_tft, _ = simulate(TitForTat(), AlwaysDefect(), matrix, rounds=50, delta=0.9)
        # Round 0: TfT cooperates, opponent defects → S
        # Rounds 1+: TfT defects, opponent defects → P
        expected = matrix.S + sum(matrix.P * 0.9**t for t in range(1, 50))
        assert abs(p_tft - expected) < 1e-9

    def test_invalid_rounds_raises(self, matrix):
        with pytest.raises(ValueError, match="rounds"):
            simulate(TitForTat(), TitForTat(), matrix, rounds=0, delta=0.9)

    def test_invalid_delta_raises(self, matrix):
        with pytest.raises(ValueError, match="delta"):
            simulate(TitForTat(), TitForTat(), matrix, rounds=10, delta=1.0)

    def test_symmetry(self, matrix):
        """Swapping strategies swaps payoffs."""
        p1, p2 = simulate(TitForTat(), GrimTrigger(), matrix, rounds=50, delta=0.9)
        q1, q2 = simulate(GrimTrigger(), TitForTat(), matrix, rounds=50, delta=0.9)
        assert abs(p1 - q2) < 1e-9
        assert abs(p2 - q1) < 1e-9

    def test_reset_called_between_games(self, matrix):
        """Running simulate twice should give identical results (reset works)."""
        s1, s2 = GrimTrigger(), AlwaysDefect()
        r1 = simulate(s1, s2, matrix, rounds=30, delta=0.8)
        r2 = simulate(s1, s2, matrix, rounds=30, delta=0.8)
        assert abs(r1[0] - r2[0]) < 1e-9
