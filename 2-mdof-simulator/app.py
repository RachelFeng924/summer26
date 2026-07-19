"""Streamlit app version of mdof.ipynb: solve, verify, and animate MDOF vibration modes."""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.linalg import eigh, LinAlgError

st.set_page_config(page_title="MDOF Simulator", layout="wide")


# ---------------------------------------------------------------------------
# Core routines (same math as mdof.ipynb)
# ---------------------------------------------------------------------------

def solve_modes(M, K):
    """Solve K phi = w^2 M phi. Returns wn [rad/s] ascending, fn [Hz], mass-normalized Phi."""
    eigvals, eigvecs = eigh(K, M)
    eigvals = np.clip(eigvals, 0, None)
    wn = np.sqrt(eigvals)
    fn = wn / (2 * np.pi)

    Phi = eigvecs.copy()
    for i in range(Phi.shape[1]):
        m_i = Phi[:, i] @ M @ Phi[:, i]
        Phi[:, i] /= np.sqrt(m_i)

    return wn, fn, Phi


def chain_matrices(masses, stiffnesses):
    """Build M, K for an n-DOF spring-mass chain fixed at both ends."""
    n = len(masses)
    M = np.diag(masses).astype(float)
    K = np.zeros((n, n))
    for i in range(n):
        K[i, i] = stiffnesses[i] + stiffnesses[i + 1]
    for i in range(n - 1):
        K[i, i + 1] = K[i + 1, i] = -stiffnesses[i + 1]
    return M, K


def frf_direct(w, M, K):
    n = M.shape[0]
    H = np.zeros((n, n, len(w)), dtype=complex)
    for k, wk in enumerate(w):
        H[:, :, k] = np.linalg.inv(K - wk**2 * M)
    return H


def frf_modal(w, wn, Phi):
    n = Phi.shape[0]
    H = np.zeros((n, n, len(w)), dtype=complex)
    for i in range(len(wn)):
        outer = np.outer(Phi[:, i], Phi[:, i])
        H += outer[:, :, None] / (wn[i]**2 - w[None, None, :]**2)
    return H


def free_response(t, wn, Phi, M, x0, v0):
    """General free response via modal superposition.

    Returns x(t) shape (n, n_t) and per-mode contributions shape (n_modes, n, n_t).
    """
    q0 = Phi.T @ M @ x0
    qdot0 = Phi.T @ M @ v0

    q = (q0[:, None] * np.cos(wn[:, None] * t[None, :])
         + (qdot0 / wn)[:, None] * np.sin(wn[:, None] * t[None, :]))

    contributions = Phi[:, :, None] * q[None, :, :]      # (n, n_modes, n_t)
    contributions = np.transpose(contributions, (1, 0, 2))  # (n_modes, n, n_t)
    x = Phi @ q
    return x, contributions


# ---------------------------------------------------------------------------
# Plotly animation builders
# ---------------------------------------------------------------------------

def _stem_xy(x_eq, y):
    """Interleave stem segments (x_eq[i],0) -> (x_eq[i], y[i]) with None breaks for one trace."""
    xs, ys = [], []
    for xi, yi in zip(x_eq, y):
        xs += [xi, xi, None]
        ys += [0, yi, None]
    return xs, ys


def mode_animation_figure(mode_index, wn, Phi, n_frames=60, amplitude_scale=0.3, spacing=1.0):
    n = Phi.shape[0]
    shape = Phi[:, mode_index]
    shape = shape / np.max(np.abs(shape))
    x_eq = np.arange(1, n + 1) * spacing

    period = 2 * np.pi / wn[mode_index]
    t = np.linspace(0, period, n_frames, endpoint=False)

    color = f"C{mode_index}"
    plotly_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                      "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    marker_color = plotly_colors[mode_index % len(plotly_colors)]

    def frame_xy(frame):
        y = amplitude_scale * shape * np.cos(wn[mode_index] * t[frame])
        return x_eq, y

    x0, y0 = frame_xy(0)
    stem_x0, stem_y0 = _stem_xy(x_eq, y0)

    fig = go.Figure(
        data=[
            go.Scatter(x=stem_x0, y=stem_y0, mode="lines", line=dict(color="gray", width=1),
                       showlegend=False),
            go.Scatter(x=x0, y=y0, mode="markers", marker=dict(size=18, color=marker_color),
                       showlegend=False),
        ]
    )

    frames = []
    for k in range(n_frames):
        xk, yk = frame_xy(k)
        stem_xk, stem_yk = _stem_xy(x_eq, yk)
        frames.append(go.Frame(
            data=[
                go.Scatter(x=stem_xk, y=stem_yk),
                go.Scatter(x=xk, y=yk),
            ],
            name=str(k),
        ))
    fig.frames = frames

    fig.update_layout(
        title=f"Mode {mode_index + 1}:  fn = {wn[mode_index] / (2*np.pi):.3f} Hz",
        xaxis=dict(range=[0, (n + 1) * spacing], title="equilibrium position along chain"),
        yaxis=dict(range=[-1.5 * amplitude_scale, 1.5 * amplitude_scale], showticklabels=False),
        height=320,
        margin=dict(t=50, b=40),
        updatemenus=[_play_pause_menu()],
        sliders=[_frame_slider(n_frames)],
    )
    return fig


def superposition_animation_figure(t, x, contributions, n_frames=150, spacing=1.0,
                                    amplitude_scale=1.0):
    n_modes, n, n_t = contributions.shape
    x_eq = np.arange(1, n + 1) * spacing
    frame_idx = np.linspace(0, n_t - 1, n_frames).astype(int)

    plotly_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                      "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

    def frame_data(k):
        y_total = amplitude_scale * x[:, k]
        stem_x, stem_y = _stem_xy(x_eq, y_total)
        data = [
            go.Scatter(x=stem_x, y=stem_y),
            go.Scatter(x=x_eq, y=y_total),
        ]
        for i in range(n_modes):
            y_mode = amplitude_scale * contributions[i, :, k]
            data.append(go.Scatter(x=x_eq, y=y_mode))
        return data

    k0 = frame_idx[0]
    y_total0 = amplitude_scale * x[:, k0]
    stem_x0, stem_y0 = _stem_xy(x_eq, y_total0)

    fig = go.Figure(
        data=[
            go.Scatter(x=stem_x0, y=stem_y0, mode="lines", line=dict(color="black", width=2),
                       showlegend=False),
            go.Scatter(x=x_eq, y=y_total0, mode="markers", marker=dict(size=18, color="black"),
                       name="superimposed"),
        ] + [
            go.Scatter(x=x_eq, y=amplitude_scale * contributions[i, :, k0], mode="markers",
                       marker=dict(size=9, color=plotly_colors[i % len(plotly_colors)]),
                       opacity=0.55, name=f"mode {i+1} contribution")
            for i in range(n_modes)
        ]
    )

    frames = []
    for k in frame_idx:
        frames.append(go.Frame(data=frame_data(k), name=str(int(k)),
                                layout=go.Layout(annotations=[dict(
                                    text=f"t = {t[k]:.2f} s", x=0.02, y=0.92,
                                    xref="paper", yref="paper", showarrow=False)])))
    fig.frames = frames

    fig.update_layout(
        title="Superposed free response — total (black) vs. each mode's contribution",
        xaxis=dict(range=[0, (n + 1) * spacing], title="equilibrium position along chain"),
        yaxis=dict(range=[-1.5 * amplitude_scale, 1.5 * amplitude_scale], showticklabels=False),
        height=420,
        margin=dict(t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        updatemenus=[_play_pause_menu()],
        sliders=[_frame_slider(len(frame_idx))],
        annotations=[dict(text=f"t = {t[k0]:.2f} s", x=0.02, y=0.92,
                           xref="paper", yref="paper", showarrow=False)],
    )
    return fig


def _play_pause_menu():
    return dict(
        type="buttons", showactive=False, x=0, y=-0.15, xanchor="left", yanchor="top",
        buttons=[
            dict(label="Play", method="animate",
                 args=[None, dict(frame=dict(duration=40, redraw=True), fromcurrent=True,
                                   transition=dict(duration=0))]),
            dict(label="Pause", method="animate",
                 args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
        ],
    )


def _frame_slider(n_frames):
    return dict(
        active=0, x=0, y=-0.05, len=1.0,
        steps=[dict(method="animate", args=[[str(k)], dict(mode="immediate",
                    frame=dict(duration=0, redraw=True))], label="")
               for k in range(n_frames)],
    )


# ---------------------------------------------------------------------------
# Sidebar: build M and K
# ---------------------------------------------------------------------------

st.title("MDOF Vibration Simulator")
st.caption("Solve the generalized eigenproblem for any mass/stiffness matrix pair, "
           "verify it, and animate the mode shapes — individually and superimposed.")

st.sidebar.header("System definition")
input_mode = st.sidebar.radio("Input mode", ["Spring-mass chain", "Custom M / K matrices"])

if input_mode == "Spring-mass chain":
    n_dof = st.sidebar.slider("Number of masses", 2, 8, 4)

    default_masses = [1.0, 1.5, 1.0, 2.0, 1.0, 1.5, 1.0, 2.0][:n_dof]
    default_k = [400.0, 300.0, 300.0, 250.0, 200.0, 300.0, 250.0, 200.0, 150.0][:n_dof + 1]

    st.sidebar.caption("Masses [kg]")
    masses_df = st.sidebar.data_editor(
        pd.DataFrame({"mass": default_masses}), num_rows="fixed", key="masses_editor",
        hide_index=True,
    )
    st.sidebar.caption("Springs [N/m], wall–m0–m1–...–wall (n+1 values)")
    springs_df = st.sidebar.data_editor(
        pd.DataFrame({"k": default_k}), num_rows="fixed", key="springs_editor",
        hide_index=True,
    )

    masses = masses_df["mass"].to_numpy(dtype=float)
    stiffnesses = springs_df["k"].to_numpy(dtype=float)

    if len(masses) != n_dof or len(stiffnesses) != n_dof + 1:
        st.sidebar.error(f"Need {n_dof} masses and {n_dof + 1} springs.")
        st.stop()

    M, K = chain_matrices(masses, stiffnesses)

else:
    n_dof = st.sidebar.slider("Degrees of freedom", 2, 8, 4)
    if "custom_M" not in st.session_state or st.session_state.get("custom_n") != n_dof:
        m0, k0 = chain_matrices(np.ones(n_dof), 200.0 * np.ones(n_dof + 1))
        st.session_state["custom_M"] = m0
        st.session_state["custom_K"] = k0
        st.session_state["custom_n"] = n_dof

    st.sidebar.caption("M matrix")
    M_df = st.sidebar.data_editor(
        pd.DataFrame(st.session_state["custom_M"],
                     columns=[str(i) for i in range(n_dof)]),
        num_rows="fixed", key="M_editor",
    )
    st.sidebar.caption("K matrix")
    K_df = st.sidebar.data_editor(
        pd.DataFrame(st.session_state["custom_K"],
                     columns=[str(i) for i in range(n_dof)]),
        num_rows="fixed", key="K_editor",
    )

    M_raw = M_df.to_numpy(dtype=float)
    K_raw = K_df.to_numpy(dtype=float)
    # symmetrize in case the user only edited one triangle
    M = 0.5 * (M_raw + M_raw.T)
    K = 0.5 * (K_raw + K_raw.T)

    if not np.allclose(M_raw, M, atol=1e-9) or not np.allclose(K_raw, K, atol=1e-9):
        st.sidebar.warning("M/K weren't symmetric — averaged with their transpose.")

# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------

try:
    wn, fn, Phi = solve_modes(M, K)
except LinAlgError:
    st.error("Eigensolve failed — check that M is symmetric positive definite "
             "(e.g. no zero or negative masses).")
    st.stop()

n = M.shape[0]

col_m, col_k = st.columns(2)
with col_m:
    st.subheader("M (mass matrix)")
    st.dataframe(pd.DataFrame(M).style.format("{:.3f}"), use_container_width=True)
with col_k:
    st.subheader("K (stiffness matrix)")
    st.dataframe(pd.DataFrame(K).style.format("{:.2f}"), use_container_width=True)

st.subheader("Natural frequencies")
freq_df = pd.DataFrame({
    "mode": [f"mode {i+1}" for i in range(n)],
    "wn [rad/s]": wn,
    "fn [Hz]": fn,
})
st.dataframe(freq_df.style.format({"wn [rad/s]": "{:.4f}", "fn [Hz]": "{:.4f}"}),
             hide_index=True, use_container_width=True)

with st.expander("Mode shapes (mass-normalized Phi)"):
    st.dataframe(pd.DataFrame(Phi, columns=[f"mode {i+1}" for i in range(n)])
                 .style.format("{:.4f}"), use_container_width=True)

with st.expander("Orthogonality check — Phi should diagonalize M and K"):
    Mr = Phi.T @ M @ Phi
    Kr = Phi.T @ K @ Phi
    off_diag_M = Mr - np.diag(np.diag(Mr))
    off_diag_K = Kr - np.diag(np.diag(Kr))
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Phi^T M Phi (should be I)")
        st.dataframe(pd.DataFrame(Mr).style.format("{:.2e}"), use_container_width=True)
    with c2:
        st.caption("Phi^T K Phi (should be diag(wn^2))")
        st.dataframe(pd.DataFrame(Kr).style.format("{:.2e}"), use_container_width=True)
    st.write(f"max |off-diagonal| in Phi^T M Phi: `{np.max(np.abs(off_diag_M)):.2e}`")
    st.write(f"max |off-diagonal| in Phi^T K Phi: `{np.max(np.abs(off_diag_K)):.2e}`")

with st.expander("FRF check — direct matrix inversion vs. modal-sum reconstruction"):
    drive = st.number_input("Drive DOF", 0, n - 1, 0)
    w_sweep = np.linspace(0.1, 1.3 * wn[-1], 1200)
    for wr in wn:
        w_sweep = w_sweep[np.abs(w_sweep - wr) > 0.5]
    H_direct = frf_direct(w_sweep, M, K)
    H_modal = frf_modal(w_sweep, wn, Phi)
    diff = np.max(np.abs(H_direct - H_modal))
    st.write(f"max |H_direct - H_modal| over the sweep: `{diff:.2e}`")

    fig = go.Figure()
    plotly_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                      "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    for i in range(n):
        fig.add_trace(go.Scatter(x=w_sweep, y=np.abs(H_direct[i, drive, :]),
                                  mode="lines", line=dict(width=3, color=plotly_colors[i % 10]),
                                  opacity=0.5, name=f"direct, DOF {i}"))
        fig.add_trace(go.Scatter(x=w_sweep, y=np.abs(H_modal[i, drive, :]),
                                  mode="lines", line=dict(width=1.5, dash="dash", color="black"),
                                  name=f"modal sum, DOF {i}", showlegend=(i == 0)))
    for wr in wn:
        fig.add_vline(x=wr, line=dict(color="gray", dash="dot", width=1))
    fig.update_layout(title=f"FRF magnitude, driven at DOF {drive}",
                       xaxis_title="frequency [rad/s]", yaxis_title="|H| [m/N]",
                       yaxis_type="log", height=450)
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Mode animations (separate)
# ---------------------------------------------------------------------------

st.header("Mode shapes — animated individually")
amp_scale = st.slider("Mode animation amplitude scale", 0.05, 1.0, 0.3, 0.05)

tabs = st.tabs([f"Mode {i+1}" for i in range(n)])
for i, tab in enumerate(tabs):
    with tab:
        fig = mode_animation_figure(i, wn, Phi, amplitude_scale=amp_scale)
        st.plotly_chart(fig, use_container_width=True, key=f"mode_anim_{i}")

# ---------------------------------------------------------------------------
# Superposition (combined)
# ---------------------------------------------------------------------------

st.header("Superimposed free response")
st.caption("Pick an initial condition; the system's free response is the sum of every mode's "
           "independently-oscillating contribution.")

ic_col1, ic_col2 = st.columns(2)
with ic_col1:
    st.caption("Initial displacement x0 [m]")
    default_x0 = np.zeros(n)
    default_x0[0] = 1.0
    x0_df = st.data_editor(pd.DataFrame({"x0": default_x0}), num_rows="fixed",
                            key="x0_editor", hide_index=True)
with ic_col2:
    st.caption("Initial velocity v0 [m/s]")
    v0_df = st.data_editor(pd.DataFrame({"v0": np.zeros(n)}), num_rows="fixed",
                            key="v0_editor", hide_index=True)

x0 = x0_df["x0"].to_numpy(dtype=float)
v0 = v0_df["v0"].to_numpy(dtype=float)

duration = st.slider("Duration [s]", 1.0, 20.0, 5.0, 0.5)
super_amp = st.slider("Superposition amplitude scale", 0.1, 3.0, 1.0, 0.1)

t_check = np.linspace(0, duration, 800)
x_super, contributions = free_response(t_check, wn, Phi, M, x0, v0)

fig_super = superposition_animation_figure(t_check, x_super, contributions,
                                            amplitude_scale=super_amp)
st.plotly_chart(fig_super, use_container_width=True, key="superposition_anim")

with st.expander("Time-series view of the superposed response"):
    fig_ts = go.Figure()
    plotly_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                      "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    for i in range(n):
        fig_ts.add_trace(go.Scatter(x=t_check, y=x_super[i], mode="lines",
                                     name=f"DOF {i}", line=dict(color=plotly_colors[i % 10])))
    fig_ts.update_layout(title="Free response from the chosen initial condition",
                         xaxis_title="time [s]", yaxis_title="displacement [m]", height=400)
    st.plotly_chart(fig_ts, use_container_width=True)
