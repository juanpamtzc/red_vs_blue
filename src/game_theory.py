import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
from typing import Optional

def compute_expected_utilities(N: int, phi1: float, phi2: float, gamma: float, threshold: Optional[float] = 0.5):
    """
    Core vectorized computation of expected utilities for Red and Blue.

    phi1, phi2, gamma may each be a scalar or a numpy array of any shape.
    The k-summation (over the N possible vote counts) is handled internally
    via broadcasting on the last axis, so the returned arrays have the same
    shape as the broadcast of phi1 / gamma.

    Returns
    -------
    E_red, E_blue : ndarray (or float when inputs are scalar)
    """
    phi1_b  = np.asarray(phi1,  dtype=float)[..., np.newaxis]
    phi2_b  = np.asarray(phi2,  dtype=float)[..., np.newaxis]
    gamma_b = np.asarray(gamma, dtype=float)[..., np.newaxis]

    k = np.arange(N)                                  # shape (N,)
    probabilities = binom.pmf(k, N - 1, gamma_b)      # shape (..., N)

    majority = N * threshold

    u_red = np.where(
        (k + 1) > majority,
        phi1_b + phi2_b * k,
        phi1_b + phi2_b * (N - 1),
    )

    u_blue = np.where(
        k > majority,
        phi2_b * k,
        phi1_b + phi2_b * (N - 1),
    )

    E_red  = np.sum(probabilities * u_red,  axis=-1)
    E_blue = np.sum(probabilities * u_blue, axis=-1)
    return E_red, E_blue

def evaluate_voting_strategy(N: int, phi1: float, phi2: float, gamma: float, threshold: Optional[float] = 0.5):
    """
    Evaluates the expected utility of voting Red vs. Blue for a single parameter point.

    Parameters
    ----------
    N     : int   – total population size (>= 2)
    phi1  : float – utility weight of your own life
    phi2  : float – utility weight per other person's life saved
    gamma : float – probability in [0, 1] that any other person votes Red

    Returns
    -------
    dict with keys: Expected_Utility_Red, Expected_Utility_Blue, Delta_E, Optimal_Choice
    """
    if N < 2:
        raise ValueError("Population N must be at least 2.")
    if not (0.0 <= gamma <= 1.0):
        raise ValueError("Gamma must be a valid probability between 0.0 and 1.0.")

    E_red, E_blue = compute_expected_utilities(N, phi1, phi2, gamma, threshold)
    delta_e = float(E_red) - float(E_blue)

    if delta_e > 1e-9:
        decision = "Vote RED"
    elif delta_e < -1e-9:
        decision = "Vote BLUE"
    else:
        decision = "Indifferent"

    return {
        "Expected_Utility_Red":  float(E_red),
        "Expected_Utility_Blue": float(E_blue),
        "Delta_E":               delta_e,
        "Optimal_Choice":        decision,
    }

def compute_sweep(N: Optional[int] = 100, resolution: Optional[int] = 300, ratio_min: Optional[float] = 1e-3, ratio_max: Optional[float] = 10.0, ratio_scale: Optional[str] = "linear", threshold: Optional[float] = 0.5):
    """
    Computes the delta_E grid over (gamma, phi1/phi2 ratio) parameter space.

    phi2 is fixed at 1.0, so the ratio axis equals phi1 directly.

    Parameters
    ----------
    N            : int    – total population size
    resolution   : int    – number of sample points along each axis
    ratio_min    : float  – lower bound for the ratio axis; must be > 0 for log scale
    ratio_max    : float  – upper bound for the ratio axis
    ratio_scale  : str    – "linear" or "log"

    Returns
    -------
    dict with keys: gamma_grid, ratio_grid, delta_e, N, ratio_scale, ratio_min, ratio_max
    """
    phi2 = 1.0
    gamma_vals = np.linspace(0.0, 1.0, resolution)

    if ratio_scale == "log":
        if ratio_min <= 0:
            raise ValueError("ratio_min must be > 0 when ratio_scale='log'.")
        ratio_vals = np.logspace(np.log10(ratio_min), np.log10(ratio_max), resolution)
    elif ratio_scale == "linear":
        ratio_vals = np.linspace(ratio_min, ratio_max, resolution)
    else:
        raise ValueError("ratio_scale must be either 'linear' or 'log'.")

    gamma_grid, ratio_grid = np.meshgrid(gamma_vals, ratio_vals)

    E_red, E_blue = compute_expected_utilities(N, ratio_grid, phi2, gamma_grid, threshold)
    delta_e = E_red - E_blue

    return {
        "gamma_grid": gamma_grid,
        "ratio_grid": ratio_grid,
        "delta_e": delta_e,
        "N": N,
        "ratio_scale": ratio_scale,
        "ratio_min": ratio_min,
        "ratio_max": ratio_max,
    }