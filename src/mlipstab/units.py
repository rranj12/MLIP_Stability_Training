"""Units. StABlE/MDsim store energies in eV and forces in eV/Å internally
(data/md17.py divides the kcal/mol labels by EV_TO_KCAL_MOL), so every
energy-valued hyperparameter in this repo (e.g. the margin m) is in eV.
"""

KB_EV = 8.617333262e-5  # Boltzmann constant, eV/K
EV_TO_KCAL_MOL = 23.06052  # same constant as mdsim.common.utils


def kT(temperature_k: float) -> float:
    """Thermal energy k_B T in eV (0.0431 eV ~= 0.994 kcal/mol at 500 K)."""
    return KB_EV * temperature_k
