import matplotlib.pyplot as plt
from typing import Optional

def plot_sweep(sweep_result: dict, 
               vmax_clip: Optional[float] = 5.0, 
               yscale: Optional[str] = None) -> None:
    """
    Renders the parameter sweep heat-map from the dict returned by compute_sweep().

    Parameters
    ----------
    sweep_result : dict  – output of compute_sweep()
    vmax_clip    : float – color-scale cap (highlights the decision boundary)
    yscale       : str | None – "linear", "log", or None to follow sweep_result metadata
    """
    gamma_grid = sweep_result["gamma_grid"]
    ratio_grid = sweep_result["ratio_grid"]
    delta_e    = sweep_result["delta_e"]
    N          = sweep_result["N"]

    resolved_yscale = yscale or sweep_result.get("ratio_scale", "linear")

    fig, ax = plt.subplots(figsize=(10, 4))

    cf = ax.contourf(
        gamma_grid, ratio_grid, delta_e,
        levels=100, cmap="coolwarm",
        vmin=-vmax_clip, vmax=vmax_clip, extend="both",
    )
    fig.colorbar(cf, ax=ax, label="Expected Utility Difference (ΔE)")

    ax.contour(
        gamma_grid, ratio_grid, delta_e,
        levels=[0], colors="black", linewidths=2.5,
    )

    if resolved_yscale == "log":
        ax.set_yscale("log")
        ax.set_ylim(sweep_result.get("ratio_min", ratio_grid.min()), sweep_result.get("ratio_max", ratio_grid.max()))

    ax.set_title(f"Optimal Voting Strategy (N={N})", fontsize=12)
    ax.set_xlabel("Gamma: Expected % voting Red (γ)", fontsize=12)
    ax.set_ylabel("(φ1 / φ2)", fontsize=12)

    plt.tight_layout()
    plt.savefig("voting_parameter_sweep.png", dpi=150)
    plt.show()