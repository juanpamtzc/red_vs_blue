"""Unit tests for red_vs_blue.model."""

import math

import pytest

from red_vs_blue.model import (
    PayoffMatrix,
    compute_critical_delta,
    compute_decision_boundary,
    get_optimal_action,
    is_cooperation_sustainable,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def canonical_matrix() -> PayoffMatrix:
    """Standard Prisoner's Dilemma: T=5, R=3, P=1, S=0."""
    return PayoffMatrix(T=5.0, R=3.0, P=1.0, S=0.0)


# ---------------------------------------------------------------------------
# PayoffMatrix construction
# ---------------------------------------------------------------------------

class TestPayoffMatrix:
    def test_valid_construction(self, canonical_matrix):
        assert canonical_matrix.T == 5.0
        assert canonical_matrix.R == 3.0
        assert canonical_matrix.P == 1.0
        assert canonical_matrix.S == 0.0

    def test_frozen(self, canonical_matrix):
        with pytest.raises((AttributeError, TypeError)):
            canonical_matrix.T = 99  # type: ignore[misc]

    def test_ordering_violation_raises(self):
        with pytest.raises(ValueError, match="T > R > P > S"):
            PayoffMatrix(T=3.0, R=5.0, P=1.0, S=0.0)  # R > T

    def test_ordering_equality_raises(self):
        with pytest.raises(ValueError, match="T > R > P > S"):
            PayoffMatrix(T=5.0, R=3.0, P=3.0, S=0.0)  # R == P

    def test_social_optimum_violation_raises(self):
        # T=5, R=3, P=2, S=1.5: ordering holds (5>3>2>1.5) but 2R=6 < T+S=6.5
        with pytest.raises(ValueError, match="2R > T"):
            PayoffMatrix(T=5.0, R=3.0, P=2.0, S=1.5)

    def test_type_error_for_non_numeric(self):
        with pytest.raises(TypeError):
            PayoffMatrix(T="high", R=3.0, P=1.0, S=0.0)  # type: ignore[arg-type]

    def test_negative_values_allowed(self):
        # Negative S is valid as long as ordering holds
        pm = PayoffMatrix(T=5.0, R=3.0, P=1.0, S=-2.0)
        assert pm.S == -2.0


# ---------------------------------------------------------------------------
# compute_critical_delta
# ---------------------------------------------------------------------------

class TestComputeCriticalDelta:
    def test_canonical(self, canonical_matrix):
        # δ* = (5-3)/(5-1) = 2/4 = 0.5
        assert math.isclose(compute_critical_delta(canonical_matrix), 0.5)

    def test_result_in_unit_interval(self, canonical_matrix):
        delta_star = compute_critical_delta(canonical_matrix)
        assert 0 < delta_star < 1

    def test_high_temptation_increases_threshold(self):
        """Higher T should raise δ* (harder to sustain cooperation)."""
        # Both satisfy 2R > T+S: 2*3=6 > 4+0=4 and 6 > 5.5+0=5.5
        m_low = PayoffMatrix(T=4.0, R=3.0, P=1.0, S=0.0)
        m_high = PayoffMatrix(T=5.5, R=3.0, P=1.0, S=0.0)
        assert compute_critical_delta(m_high) > compute_critical_delta(m_low)

    def test_formula(self):
        m = PayoffMatrix(T=7.0, R=4.0, P=2.0, S=0.0)
        expected = (7.0 - 4.0) / (7.0 - 2.0)  # 3/5 = 0.6
        assert math.isclose(compute_critical_delta(m), expected)


# ---------------------------------------------------------------------------
# is_cooperation_sustainable
# ---------------------------------------------------------------------------

class TestIsCooperationSustainable:
    def test_above_threshold_is_true(self, canonical_matrix):
        assert is_cooperation_sustainable(canonical_matrix, delta=0.9) is True

    def test_at_threshold_is_true(self, canonical_matrix):
        # δ* = 0.5 exactly; δ >= δ* should be True
        assert is_cooperation_sustainable(canonical_matrix, delta=0.5) is True

    def test_below_threshold_is_false(self, canonical_matrix):
        assert is_cooperation_sustainable(canonical_matrix, delta=0.3) is False

    def test_invalid_delta_zero_raises(self, canonical_matrix):
        with pytest.raises(ValueError, match="Discount factor"):
            is_cooperation_sustainable(canonical_matrix, delta=0.0)

    def test_invalid_delta_one_raises(self, canonical_matrix):
        with pytest.raises(ValueError, match="Discount factor"):
            is_cooperation_sustainable(canonical_matrix, delta=1.0)

    def test_invalid_delta_negative_raises(self, canonical_matrix):
        with pytest.raises(ValueError):
            is_cooperation_sustainable(canonical_matrix, delta=-0.1)


# ---------------------------------------------------------------------------
# get_optimal_action
# ---------------------------------------------------------------------------

class TestGetOptimalAction:
    def test_cooperate_when_patient(self, canonical_matrix):
        assert get_optimal_action(canonical_matrix, delta=0.9) == "Cooperate"

    def test_defect_when_impatient(self, canonical_matrix):
        assert get_optimal_action(canonical_matrix, delta=0.1) == "Defect"

    def test_returns_string(self, canonical_matrix):
        result = get_optimal_action(canonical_matrix, delta=0.7)
        assert isinstance(result, str)
        assert result in {"Cooperate", "Defect"}


# ---------------------------------------------------------------------------
# compute_decision_boundary
# ---------------------------------------------------------------------------

class TestComputeDecisionBoundary:
    def test_output_shapes(self, canonical_matrix):
        resolution = 50
        T_grid, delta_grid, mask = compute_decision_boundary(
            canonical_matrix, resolution=resolution
        )
        assert T_grid.shape == (resolution, resolution)
        assert delta_grid.shape == (resolution, resolution)
        assert mask.shape == (resolution, resolution)

    def test_cooperation_in_high_delta_region(self, canonical_matrix):
        """Top rows (high δ) should be mostly cooperation."""
        _, delta_grid, mask = compute_decision_boundary(
            canonical_matrix, resolution=50
        )
        # Rows with δ > 0.9 should have many True cells
        high_delta_rows = delta_grid[:, 0] > 0.9
        assert mask[high_delta_rows].mean() > 0.5

    def test_defection_in_low_delta_region(self, canonical_matrix):
        """Bottom rows (low δ) should be mostly defection."""
        _, delta_grid, mask = compute_decision_boundary(
            canonical_matrix, resolution=50
        )
        low_delta_rows = delta_grid[:, 0] < 0.1
        assert mask[low_delta_rows].mean() < 0.5

    def test_custom_resolution(self, canonical_matrix):
        T_grid, _, _ = compute_decision_boundary(canonical_matrix, resolution=20)
        assert T_grid.shape == (20, 20)
