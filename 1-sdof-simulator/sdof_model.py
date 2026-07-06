"""Core SDOF (single-degree-of-freedom) physics: m x'' + c x' + k x = F(t)"""

import numpy as np
from scipy.integrate import solve_ivp


def derived_quantities(m, k, c):
    wn = np.sqrt(k / m)
    zeta = c / (2 * np.sqrt(k * m))
    wd = wn * np.sqrt(1 - zeta**2) if zeta < 1 else float("nan")
    if zeta == 0:
        regime = "Undamped (oscillates forever)"
    elif zeta < 1:
        regime = "Underdamped (decaying oscillation)"
    elif zeta == 1:
        regime = "Critically damped (fastest non-oscillating decay)"
    else:
        regime = "Overdamped (no oscillation)"
    return wn, zeta, wd, regime


def _eom(t, y, m, k, c, F):
    x, v = y
    return [v, (F(t) - c * v - k * x) / m]


def simulate(m, k, c, x0, v0, F0, Omega, t_end, n_points=2000):
    """Simulate free + forced response together: ICs plus an optional harmonic force."""
    t_eval = np.linspace(0, t_end, n_points)
    forcing = (lambda t: F0 * np.cos(Omega * t)) if F0 != 0 else (lambda t: 0.0)
    sol = solve_ivp(
        _eom, (t_eval[0], t_eval[-1]), [x0, v0],
        t_eval=t_eval, args=(m, k, c, forcing),
        rtol=1e-9, atol=1e-12,
    )
    return t_eval, sol.y[0], sol.y[1]


def receptance(w, k, m, zeta):
    c_i = zeta * 2 * np.sqrt(k * m)
    return 1.0 / (k - m * w**2 + 1j * c_i * w)


def frf_sweep(k, m, zeta_list, n_points=2000, w_max_ratio=3.0):
    wn = np.sqrt(k / m)
    w = np.linspace(0.01 * wn, w_max_ratio * wn, n_points)
    curves = {z: receptance(w, k, m, z) for z in zeta_list}
    return w, wn, curves
