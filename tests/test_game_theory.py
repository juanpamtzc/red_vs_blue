import pytest
import numpy as np
from red_vs_blue.src.game_theory import evaluate_voting_strategy, compute_sweep

def test_population_validation():
    """Test that invalid population sizes are rejected."""
    with pytest.raises(ValueError, match="Population N must be at least 2"):
        evaluate_voting_strategy(N=1, phi1=1.0, phi2=1.0, gamma=0.5)

def test_gamma_validation():
    """Test that invalid probability bounds are rejected."""
    with pytest.raises(ValueError, match="Gamma must be a valid probability"):
        evaluate_voting_strategy(N=10, phi1=1.0, phi2=1.0, gamma=-0.1)
        
    with pytest.raises(ValueError, match="Gamma must be a valid probability"):
        evaluate_voting_strategy(N=10, phi1=1.0, phi2=1.0, gamma=1.1)

def test_extreme_gamma_zero():
    """
    Test the boundary condition where NO ONE else votes Red (gamma = 0.0).
    If N=100 and k=0, neither voting Red (1 vote) nor Blue (0 votes) 
    will trigger the super-critical threshold (>50).
    Therefore, nobody dies regardless of what you do, and you should be Indifferent.
    """
    result = evaluate_voting_strategy(N=100, phi1=1.0, phi2=1.0, gamma=0.0)
    
    assert result["Expected_Utility_Red"] == result["Expected_Utility_Blue"]
    assert result["Delta_E"] == 0.0
    assert result["Optimal_Choice"] == "Indifferent"

def test_extreme_gamma_one():
    """
    Test the boundary condition where EVERYONE else votes Red (gamma = 1.0).
    If N=100 and k=99, Red will win (>50) no matter what you do.
    If you vote Blue, you die. If you vote Red, you live. 
    Therefore, the expected utility difference should exactly equal phi1 (your life).
    """
    phi1_val = 5.0
    result = evaluate_voting_strategy(N=100, phi1=phi1_val, phi2=1.0, gamma=1.0)
    
    # Delta E should be exactly phi1
    assert result["Delta_E"] == pytest.approx(phi1_val, rel=1e-9)
    assert result["Optimal_Choice"] == "Vote RED"

def test_sweep_grid_shapes():
    """
    Test that the vectorized compute_sweep function correctly broadcasts 
    and returns arrays of the requested dimensions.
    """
    resolution = 15
    result = compute_sweep(N=10, resolution=resolution, ratio_min=0.1, ratio_max=5.0, ratio_scale="linear")
    
    # Check that it returns the expected keys
    assert "gamma_grid" in result
    assert "delta_e" in result
    
    # Check that the 2D meshgrids have the shape (resolution, resolution)
    expected_shape = (resolution, resolution)
    assert result["gamma_grid"].shape == expected_shape
    assert result["ratio_grid"].shape == expected_shape
    assert result["delta_e"].shape == expected_shape

def test_log_scale_validation():
    """Test that a log sweep strictly requires a positive ratio_min."""
    with pytest.raises(ValueError, match="ratio_min must be > 0"):
        compute_sweep(resolution=10, ratio_min=0.0, ratio_max=5.0, ratio_scale="log")