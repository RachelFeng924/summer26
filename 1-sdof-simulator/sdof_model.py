"""Core SDOF (single-degree-of-freedom) physics: m x'' + c x' + k x = F(t)"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks


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


def free_vibration(t, wn, zeta, x0, v0):
    """Closed-form free-vibration solution x(t), one formula per damping regime."""
    if zeta <= 0:
        return x0 * np.cos(wn * t) + (v0 / wn) * np.sin(wn * t)
    elif zeta < 1:
        wd = wn * np.sqrt(1 - zeta**2)
        env = np.exp(-zeta * wn * t)
        return env * (x0 * np.cos(wd * t) + (v0 + zeta * wn * x0) / wd * np.sin(wd * t))
    elif zeta == 1:
        return np.exp(-wn * t) * (x0 + (v0 + wn * x0) * t)
    else:
        s = wn * np.sqrt(zeta**2 - 1)
        r1, r2 = -zeta * wn + s, -zeta * wn - s
        A = (v0 - r2 * x0) / (r1 - r2)
        B = (r1 * x0 - v0) / (r1 - r2)
        return A * np.exp(r1 * t) + B * np.exp(r2 * t)


def analytical_solution(t, m, k, c, x0, v0, F0, Omega):
    """Closed-form x(t). With no forcing this is just `free_vibration`; with a
    harmonic force it's the steady-state term (from the receptance) plus a
    free-vibration transient sized to match the actual initial conditions —
    exact by superposition, since the ODE is linear."""
    wn = np.sqrt(k / m)
    zeta = c / (2 * np.sqrt(k * m))
    if F0 == 0:
        return free_vibration(t, wn, zeta, x0, v0)
    denom = (k - m * Omega**2) + 1j * c * Omega
    if abs(denom) < 1e-9:
        return np.full_like(t, np.nan)  # undamped resonance: no bounded steady-state
    H = 1.0 / denom
    X, phi = F0 * abs(H), np.angle(H)
    x_ss = X * np.cos(Omega * t + phi)
    x_ss0 = X * np.cos(phi)
    v_ss0 = -X * Omega * np.sin(phi)
    x_transient = free_vibration(t, wn, zeta, x0 - x_ss0, v0 - v_ss0)
    return x_ss + x_transient


def log_decrement(t, x):
    """Recover wn and zeta from a decaying oscillation's peaks (logarithmic
    decrement), to check the numerical/analytical solution against a fully
    independent method. Returns None if there aren't enough clean peaks."""
    peaks, _ = find_peaks(x)
    if len(peaks) < 3 or np.any(x[peaks] <= 0):
        return None
    tp, xp = t[peaks], x[peaks]
    Td = np.mean(np.diff(tp))
    wd_meas = 2 * np.pi / Td
    n = len(peaks) - 1
    delta = (1.0 / n) * np.log(xp[0] / xp[-1])
    zeta_meas = delta / np.sqrt(4 * np.pi**2 + delta**2)
    if zeta_meas >= 1:
        return None
    wn_meas = wd_meas / np.sqrt(1 - zeta_meas**2)
    return dict(wn=wn_meas, zeta=zeta_meas, wd=wd_meas, n_cycles=n, peaks_t=tp, peaks_x=xp)
