"""SDOF Simulator - interactive spring-mass-damper playground (Streamlit)."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from sdof_model import derived_quantities, simulate, frf_sweep, analytical_solution, log_decrement

st.set_page_config(page_title="SDOF Simulator", layout="wide")

simulate_cached = st.cache_data(simulate)
frf_sweep_cached = st.cache_data(frf_sweep)

# ---------------------------------------------------------------- state ----
defaults = dict(m=1.0, k=400.0, c=2.0, x0=0.01, v0=0.0, F0=0.0, drive_ratio=1.0, speed=1.0)
for key, val in defaults.items():
    st.session_state.setdefault(key, val)

# --------------------------------------------------------------- sidebar ---
st.sidebar.title("System parameters")

m = st.sidebar.slider("mass  m  [kg]", 0.1, 10.0, st.session_state.m, 0.1,
                       help="The moving weight on the spring. Heavier mass → slower oscillation.")
k = st.sidebar.slider("stiffness  k  [N/m]", 10.0, 2000.0, st.session_state.k, 10.0,
                       help="Spring stiffness: how much force it takes to stretch/compress the spring "
                            "by 1 m. Stiffer spring → faster oscillation.")
c_crit = 2 * np.sqrt(k * m)
c = st.sidebar.slider("damping  c  [N·s/m]", 0.0, float(2 * c_crit), st.session_state.c, 0.5,
                       help="The dashpot's resistance to motion, proportional to velocity. It bleeds "
                            "energy out of the system, shrinking the oscillation over time.")

st.sidebar.markdown("**Presets**")
if st.sidebar.button("Undamped", use_container_width=True,
                      help="c = 0: no energy loss, oscillates forever at the same amplitude."):
    c = 0.0
if st.sidebar.button("Underdamped", use_container_width=True,
                      help="0 < c < c_crit: oscillates while decaying — the everyday case (car "
                           "suspension, guitar string)."):
    c = 0.1 * c_crit
if st.sidebar.button("Critical", use_container_width=True,
                      help="c = c_crit: returns to rest fastest without overshooting/oscillating."):
    c = c_crit
if st.sidebar.button("Overdamped", use_container_width=True,
                      help="c > c_crit: too much damping — creeps back to rest slower than critical, "
                           "with no oscillation."):
    c = 1.5 * c_crit

st.sidebar.markdown("---")
st.sidebar.markdown("**Initial conditions**")
x0 = st.sidebar.slider("initial displacement  x0  [m]", -0.05, 0.05, st.session_state.x0, 0.001,
                        help="Where the mass starts, relative to its resting position.")
v0 = st.sidebar.slider("initial velocity  v0  [m/s]", -1.0, 1.0, st.session_state.v0, 0.05,
                        help="How fast the mass is moving at the start.")

st.sidebar.markdown("---")
st.sidebar.markdown("**Harmonic forcing**")
F0 = st.sidebar.slider("force amplitude  F0  [N]", 0.0, 20.0, st.session_state.F0, 0.5,
                        help="An external push-pull force applied continuously, oscillating at the "
                             "drive ratio below. Set to 0 for free vibration only.")
drive_ratio = st.sidebar.slider("drive ratio  Ω / ωn", 0.1, 3.0, st.session_state.drive_ratio, 0.05,
                                 help="Forcing frequency relative to the system's natural frequency. "
                                      "Near 1.0 the system resonates, swinging much wider than F0 alone "
                                      "would suggest.")

st.sidebar.markdown("---")
speed = st.sidebar.slider("playback speed", 0.25, 4.0, st.session_state.speed, 0.25)

# store back
st.session_state.m, st.session_state.k, st.session_state.c = m, k, c
st.session_state.x0, st.session_state.v0 = x0, v0
st.session_state.F0, st.session_state.drive_ratio = F0, drive_ratio
st.session_state.speed = speed

st.title("Spring-Mass-Damper Simulator")

st.subheader("System components")
st.markdown(
    "- A **spring** (k) stores energy and pulls the mass back toward rest.\n"
    "- A **damper/dashpot** (c) resists motion and bleeds that energy away as heat.\n"
    "- Their balance sets how the mass settles after a disturbance:"
)
regime_cols = st.columns(4)
regime_cols[0].markdown("**Undamped**\n\nc = 0 — oscillates forever")
regime_cols[1].markdown("**Underdamped**\n\n0 < c < c_crit — oscillates, decaying")
regime_cols[2].markdown("**Critical**\n\nc = c_crit — fastest return, no overshoot")
regime_cols[3].markdown("**Overdamped**\n\nc > c_crit — slow return, no oscillation")

st.subheader("Simulation parameters + animation")
st.caption(
    "Solves m·x″ + c·x′ + k·x = F(t) numerically (solid line) and compares it against the "
    "exact closed-form solution (dotted line)."
)
with st.expander("How each solver works"):
    st.markdown(
        "**Numerical (state-space + `solve_ivp`)** — the 2nd-order ODE is rewritten as two "
        "coupled 1st-order ODEs using the state vector `[x, v]`: `x′ = v`, `v′ = (F(t) − c·v − "
        "k·x) / m`. SciPy's `solve_ivp` integrates that state vector forward in time with an "
        "adaptive Runge-Kutta method (RK45), taking smaller steps where the solution curves "
        "fastest and larger ones where it's flat, to hit tight error tolerances "
        "(`rtol=1e-9, atol=1e-12`). This works for *any* regime or forcing, but has no formula "
        "— just a dense sequence of stepped-through numbers.\n\n"
        "**Analytical (closed form)** — for free vibration (F0 = 0), each damping regime has "
        "its own exact formula from solving the ODE's characteristic equation directly (e.g. "
        "the underdamped case is a decaying cosine, `x0·e^(−ζωn·t)·cos(ωd·t) + …`). When a "
        "harmonic force is applied, the exact solution is the steady-state term (from the "
        "receptance, see below) plus a free-vibration transient — sized so the sum matches the "
        "true initial conditions. Because the ODE is linear, that superposition is exact, not "
        "an approximation."
    )
wn, zeta, wd, regime = derived_quantities(m, k, c)
info_cols = st.columns(2)
info_cols[0].metric("ωn [rad/s]", f"{wn:.2f}", help=f"fn = {wn/(2*np.pi):.2f} Hz")
info_cols[1].markdown(f"**ζ (zeta):** {zeta:.3f}   |   **regime:** {regime}")

# --------------------------------------------------------------- sim run ---
Omega = drive_ratio * wn
t_end = 8 * (2 * np.pi / wn)
n_points = 600
t, x, v = simulate_cached(m, k, c, x0, v0, F0, Omega, t_end, n_points=n_points)
x_amp = np.max(np.abs(x)) if np.max(np.abs(x)) > 0 else 1.0

x_ana = analytical_solution(t, m, k, c, x0, v0, F0, Omega)
ana_valid = not np.any(np.isnan(x_ana))

# ---------------------------------------------------- animated dashboard --
N_FRAMES = 120
frame_idx = np.linspace(0, n_points - 1, N_FRAMES).astype(int)
frame_duration_ms = max(15, (t_end / N_FRAMES) * 1000 / speed)

WALL_X = -1.1
SCALE = 0.8 / max(x_amp, 1e-6)
SPRING_Y = 0.35
DAMPER_Y = -0.35
DAMPER_CYL_FRAC = 0.4  # cylinder occupies this fraction of the wall-to-mass gap


def spring_xy(x_now):
    """Spring zigzag, offset above the centerline so it sits in parallel with the damper."""
    x_mass = x_now * SCALE
    n_zig = 8
    zig_x = np.linspace(WALL_X, x_mass - 0.3, n_zig)
    zig_y = 0.12 * (np.arange(n_zig) % 2 * 2 - 1) + SPRING_Y
    zig_y[0] = SPRING_Y
    zig_y[-1] = SPRING_Y
    return zig_x, zig_y, x_mass


def damper_xy(x_now):
    """Cylinder + piston rod that both scale with the wall-to-mass gap, so the
    dashpot visibly compresses and stretches like the spring instead of the
    rod sliding past a fixed-size cylinder (which could overlap the mass)."""
    x_mass = x_now * SCALE
    attach_x = x_mass - 0.3
    gap = attach_x - WALL_X
    cyl_x1 = WALL_X + DAMPER_CYL_FRAC * gap
    cyl_h = 0.12
    cyl_x = [WALL_X, cyl_x1, cyl_x1, WALL_X, WALL_X]
    cyl_y = [DAMPER_Y - cyl_h, DAMPER_Y - cyl_h, DAMPER_Y + cyl_h, DAMPER_Y + cyl_h, DAMPER_Y - cyl_h]
    rod_x = [cyl_x1, attach_x]
    rod_y = [DAMPER_Y, DAMPER_Y]
    return cyl_x, cyl_y, rod_x, rod_y, x_mass


def connector_xy(x_now):
    """Vertical link through the mass tying the spring and damper attachment points together."""
    x_mass = x_now * SCALE
    return [x_mass - 0.3, x_mass - 0.3], [SPRING_Y, DAMPER_Y]


fig = make_subplots(rows=1, cols=2, column_widths=[0.32, 0.68],
                     subplot_titles=("Mass-spring-damper", "Displacement vs. time"))

zx0, zy0, xm0 = spring_xy(x[0])
cylx0, cyly0, rx0, ry0, _ = damper_xy(x[0])
cx0, cy0 = connector_xy(x[0])
fig.add_trace(go.Scatter(x=zx0, y=zy0, mode="lines",
                          line=dict(color="#4C78A8", width=3), showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=[xm0], y=[0], mode="markers+text",
                          marker=dict(size=46, color="#F2B5B4", line=dict(color="#E45756", width=2), symbol="square"),
                          text=["m"], textfont=dict(size=16), showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=t, y=x * 1000, mode="lines", name="numerical",
                          line=dict(color="#4C78A8")), row=1, col=2)
if ana_valid:
    fig.add_trace(go.Scatter(x=t, y=x_ana * 1000, mode="lines", name="analytical",
                              line=dict(color="#F58518", width=2, dash="dot")), row=1, col=2)
now_marker_idx = len(fig.data)
fig.add_trace(go.Scatter(x=[t[0]], y=[x[0] * 1000], mode="markers",
                          marker=dict(size=12, color="#E45756"), name="now"), row=1, col=2)
fig.add_trace(go.Scatter(x=cylx0, y=cyly0, mode="lines", fill="toself",
                          line=dict(color="#54A24B", width=2), fillcolor="rgba(84,162,75,0.15)",
                          showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=rx0, y=ry0, mode="lines",
                          line=dict(color="#54A24B", width=4), showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=cx0, y=cy0, mode="lines",
                          line=dict(color="gray", width=2), showlegend=False), row=1, col=1)

# wall (fixed support)
fig.add_shape(type="line", x0=WALL_X, x1=WALL_X, y0=-0.5, y1=0.5,
              line=dict(color="gray", width=6), row=1, col=1)
fig.add_annotation(x=WALL_X + 0.1, y=SPRING_Y + 0.22, text="k", showarrow=False,
                    font=dict(size=13, color="#4C78A8"), row=1, col=1)
fig.add_annotation(x=WALL_X + 0.1, y=DAMPER_Y - 0.22, text="c", showarrow=False,
                    font=dict(size=13, color="#54A24B"), row=1, col=1)
# equilibrium reference (x = 0)
fig.add_shape(type="line", x0=0, x1=0, y0=-0.7, y1=0.7,
              line=dict(color="gray", width=1, dash="dot"), row=1, col=1)
fig.add_annotation(x=0, y=-0.8, text="x = 0", showarrow=False,
                    font=dict(size=11, color="gray"), row=1, col=1)
# +x direction arrow, same rightward convention as increasing values on the time plot
fig.add_annotation(x=1.2, y=-1.0, ax=0.3, ay=-1.0, xref="x1", yref="y1", axref="x1", ayref="y1",
                    showarrow=True, arrowhead=3, arrowwidth=2, arrowcolor="#4C78A8", text="")
fig.add_annotation(x=1.25, y=-1.0, text="+x", showarrow=False,
                    font=dict(size=12, color="#4C78A8"), xanchor="left", row=1, col=1)

fig.add_hline(y=0, line=dict(color="gray", width=1, dash="dot"), row=1, col=2)

fig.update_xaxes(visible=False, range=[-1.6, 1.6], row=1, col=1)
fig.update_yaxes(visible=False, range=[-1.4, 1.4], scaleanchor="x", scaleratio=1, row=1, col=1)
fig.update_xaxes(title_text="time [s]", row=1, col=2)
fig.update_yaxes(title_text="x [mm]", row=1, col=2)

frames = []
for i in frame_idx:
    zx, zy, xm = spring_xy(x[i])
    cylx, cyly, rx, ry, _ = damper_xy(x[i])
    cx, cy = connector_xy(x[i])
    frames.append(go.Frame(
        name=str(i),
        traces=[0, 1, now_marker_idx, now_marker_idx + 1, now_marker_idx + 2, now_marker_idx + 3],
        data=[
            go.Scatter(x=zx, y=zy),
            go.Scatter(x=[xm], y=[0]),
            go.Scatter(x=[t[i]], y=[x[i] * 1000]),
            go.Scatter(x=cylx, y=cyly),
            go.Scatter(x=rx, y=ry),
            go.Scatter(x=cx, y=cy),
        ],
    ))
fig.frames = frames

fig.update_layout(
    height=480,
    margin=dict(l=10, r=10, t=130, b=10),
    updatemenus=[dict(
        type="buttons", direction="left", x=0.388, y=1.28, xanchor="left", yanchor="bottom",
        buttons=[
            dict(label="▶ Play / ⏸ Pause", method="animate",
                 args=[None, dict(frame=dict(duration=frame_duration_ms, redraw=True),
                                   fromcurrent=True, transition=dict(duration=0))],
                 args2=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
        ],
    )],
    sliders=[dict(
        active=0, x=0.388, len=0.612, y=1.16, yanchor="bottom",
        steps=[dict(method="animate", label="",
                    args=[[str(i)], dict(mode="immediate",
                                          frame=dict(duration=0, redraw=True))])
               for i in frame_idx],
    )],
)

st.plotly_chart(fig, use_container_width=True)

if ana_valid:
    max_err = np.max(np.abs(x - x_ana)) * 1000
    st.caption(
        f"Numerical vs. analytical agree closely: max deviation **{max_err:.2e} mm** "
        f"(numerical tolerance is set to ~1e-9). The dotted analytical curve is the exact "
        f"closed-form solution — free-vibration formula for the active damping regime, plus "
        f"the steady-state term when a force is applied."
    )
else:
    st.caption(
        "Analytical solution is undefined here — this is exact undamped resonance "
        "(Ω = ωn with ζ = 0), where the closed-form steady-state amplitude is unbounded."
    )

# ---------------------------------------------------------- log decrement --
st.subheader("Logarithmic decrement verification")
st.caption(
    "An independent check: measure ωn and ζ purely from the peaks of the simulated decay "
    "curve, using the log-decrement method, and compare against the input values."
)
with st.expander("How log-decrement recovery works"):
    st.markdown(
        "This method never looks at m, k, or c — it only reads peaks off the decay curve, "
        "so agreement with the input values is a strong independent check that the physics "
        "(and the solver) are correct:\n\n"
        "1. **Find the peaks** of the simulated `x(t)` (SciPy's `find_peaks`).\n"
        "2. **Peak spacing → ωd → ωn.** The average time between consecutive peaks is the "
        "damped period `Td`, so `ωd = 2π / Td`.\n"
        "3. **Peak decay ratio → ζ.** How much each peak shrinks relative to the one *n* "
        "cycles later gives the logarithmic decrement `δ = (1/n)·ln(x₁/xₙ)`, which inverts "
        "exactly to `ζ = δ / √(4π² + δ²)`.\n"
        "4. Combine steps 2–3: `ωn = ωd / √(1 − ζ²)`."
    )
ld = log_decrement(t, x) if (F0 == 0 and 0 < zeta < 1) else None
if ld is not None:
    ld_cols = st.columns(2)
    ld_cols[0].markdown(
        f"**ωn:** input {wn:.3f} rad/s vs. recovered **{ld['wn']:.3f} rad/s** "
        f"({ld['n_cycles']} cycles measured)"
    )
    ld_cols[1].markdown(f"**ζ:** input {zeta:.4f} vs. recovered **{ld['zeta']:.4f}**")
else:
    st.caption(
        "Not applicable right now — this method needs a free (unforced), underdamped decay "
        "with at least 3 clean peaks to measure. Set force amplitude F0 to 0 and pick an "
        "underdamped damping level to see it."
    )

# ------------------------------------------------------------------ FRF ---
st.subheader("Frequency response function")
st.caption(
    "How strongly the system responds to a continuous push at each drive frequency. The peak "
    "near Ω/ωn = 1 is **resonance** — more damping (higher ζ) flattens and widens it."
)
st.caption(
    "Computed directly from the closed-form receptance H(ω) = 1 / (k − mω² + icω) swept across "
    "ω — an algebraic formula, not a simulated trajectory."
)
with st.expander("How the FRF is computed"):
    st.markdown(
        "For a steady harmonic force `F(t) = F0·cos(ωt)`, the long-run response is also a "
        "cosine at the same frequency, just scaled and phase-shifted: `x(t) = F0·|H(ω)|·"
        "cos(ωt + φ)`. Writing `F(t)` and `x(t)` as the real part of complex exponentials turns "
        "the ODE into plain algebra — plugging `x = X·e^(iωt)` into `m·x″ + c·x′ + k·x = F(t)` "
        "and solving for `X/F0` gives the receptance directly, no differential equation solving "
        "needed:\n\n"
        "`H(ω) = X / F0 = 1 / (k − mω² + icω)`\n\n"
        "- **`|H(ω)|`** (top plot) is how much the displacement is amplified relative to the "
        "force — the sharp peak near ω = ωn is resonance.\n"
        "- **`angle(H(ω))`** (bottom plot) is the phase lag φ between the force and the "
        "resulting motion, which swings from ~0° to ~−180° as ω crosses resonance."
    )
zetas_to_compare = sorted(set([0.05, 0.10, 0.25, 0.50, 1.00, round(zeta, 3)]))
w, wn_, curves = frf_sweep_cached(k, m, zetas_to_compare)

col_mag, col_phase = st.columns(2)
current_H = curves[round(zeta, 3)] if round(zeta, 3) in curves else None

with col_mag:
    fig_mag = go.Figure()
    for z, H in curves.items():
        fig_mag.add_trace(go.Scatter(x=w / wn_, y=np.abs(H), mode="lines",
                                      name=f"ζ={z:.2f}",
                                      line=dict(width=3 if abs(z - zeta) < 1e-6 else 1.5)))
    fig_mag.add_vline(x=drive_ratio, line=dict(color="gray", dash="dot"))
    if current_H is not None:
        idx = (np.abs(w / wn_ - drive_ratio)).argmin()
        fig_mag.add_trace(go.Scatter(x=[drive_ratio], y=[np.abs(current_H[idx])],
                                      mode="markers", marker=dict(size=12, color="black"),
                                      name="operating point"))
    fig_mag.update_layout(height=350, xaxis_title="Ω / ωn", yaxis_title="|H| [m/N]",
                          yaxis_type="log", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_mag, use_container_width=True)

with col_phase:
    fig_ph = go.Figure()
    for z, H in curves.items():
        fig_ph.add_trace(go.Scatter(x=w / wn_, y=np.degrees(np.angle(H)), mode="lines",
                                     name=f"ζ={z:.2f}",
                                     line=dict(width=3 if abs(z - zeta) < 1e-6 else 1.5)))
    fig_ph.add_vline(x=drive_ratio, line=dict(color="gray", dash="dot"))
    fig_ph.update_layout(height=350, xaxis_title="Ω / ωn", yaxis_title="phase [deg]",
                         margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_ph, use_container_width=True)
