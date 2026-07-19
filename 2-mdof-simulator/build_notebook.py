"""One-off script to assemble mdof.ipynb. Not part of the deliverable itself."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))

md("# MDOF Simulator")

md("""Objectives:

Write a reusable routine that takes any mass matrix `M` and stiffness matrix `K`, returns the
natural frequencies and mode shapes, and visualizes them — this is the workhorse for the rest
of the summer.

Assemble `M` and `K` for a 3-5 DOF spring-mass chain.

Solve the generalized eigenproblem (`scipy.linalg.eigh`) for natural frequencies and mode
shapes; mass-normalize the modes.

Verify orthogonality numerically (modes should diagonalize `M` and `K`).

Reconstruct the FRF as a modal sum and confirm it matches the direct matrix-inversion FRF.

Animate the mode shapes (`matplotlib FuncAnimation`) so you can see each mode move.""")

md("# Imports")

code("""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.linalg import eigh
""")

md("""# The workhorse routine

`solve_modes(M, K)` takes any mass matrix `M` and stiffness matrix `K` (same shape, symmetric
positive definite) and returns:

- `wn` — natural frequencies [rad/s], sorted ascending
- `fn` — natural frequencies [Hz]
- `Phi` — mass-normalized mode shape matrix, one column per mode (`Phi[:, i]` is mode i)

`scipy.linalg.eigh(K, M)` solves the generalized eigenproblem `K phi = omega^2 M phi` directly
(it's built for symmetric/Hermitian generalized problems, so no manual `M^-1 K` products or
eigenvector sorting are needed — `eigh` returns eigenvalues already sorted ascending).
Mass-normalizing means scaling each column so `phi_i^T M phi_i = 1`, which is what makes the
modes diagonalize `M` to the identity and `K` to `diag(omega_i^2)` (see the orthogonality
check below), and is the standard convention for modal superposition / FRF reconstruction.""")

code("""
def solve_modes(M, K):
    \"\"\"Solve the generalized eigenproblem K phi = w^2 M phi.

    Returns
    -------
    wn  : (n,) natural frequencies [rad/s], ascending
    fn  : (n,) natural frequencies [Hz]
    Phi : (n, n) mass-normalized mode shapes, Phi[:, i] is mode i
    \"\"\"
    eigvals, eigvecs = eigh(K, M)          # eigh handles the generalized symmetric problem
    eigvals = np.clip(eigvals, 0, None)    # guard against tiny negative numerical noise
    wn = np.sqrt(eigvals)
    fn = wn / (2 * np.pi)

    # eigh's eigenvectors already satisfy Phi^T M Phi = I when M is fed as the RHS matrix,
    # but we mass-normalize explicitly so the routine is correct even if that guarantee
    # is ever relied on loosely elsewhere.
    Phi = eigvecs.copy()
    for i in range(Phi.shape[1]):
        m_i = Phi[:, i] @ M @ Phi[:, i]
        Phi[:, i] /= np.sqrt(m_i)

    return wn, fn, Phi
""")

md("""# Assembling M and K for a spring-mass chain

A chain of `n` masses connected in series by springs, fixed at both ends:

```
wall --k0-- m0 --k1-- m1 --k2-- ... --kn-1-- m(n-1) --kn-- wall
```

Each mass only talks to its immediate neighbors, so `K` is tridiagonal: diagonal entries are
the sum of the two springs touching that mass, off-diagonal entries are `-k` for the shared
spring. `M` is diagonal since the masses aren't coupled directly.""")

code("""
def chain_matrices(masses, stiffnesses):
    \"\"\"Build M, K for an n-DOF spring-mass chain fixed at both ends.

    masses       : length-n array of point masses
    stiffnesses  : length-(n+1) array of spring constants, wall-m0-m1-...-wall
    \"\"\"
    n = len(masses)
    assert len(stiffnesses) == n + 1, "need n+1 springs for n masses fixed at both ends"

    M = np.diag(masses).astype(float)

    K = np.zeros((n, n))
    for i in range(n):
        K[i, i] = stiffnesses[i] + stiffnesses[i + 1]
    for i in range(n - 1):
        K[i, i + 1] = K[i + 1, i] = -stiffnesses[i + 1]

    return M, K


# 4-DOF example chain
masses = np.array([1.0, 1.5, 1.0, 2.0])          # [kg]
stiffnesses = np.array([400.0, 300.0, 300.0, 250.0, 200.0])  # [N/m], n+1 springs

M, K = chain_matrices(masses, stiffnesses)
print("M =\\n", M)
print("K =\\n", K)
""")

md("# Solve for modes")

code("""
wn, fn, Phi = solve_modes(M, K)

print("Natural frequencies:")
for i, (w, f) in enumerate(zip(wn, fn)):
    print(f"  mode {i+1}: wn = {w:8.3f} rad/s   fn = {f:7.3f} Hz")

print("\\nMass-normalized mode shapes (columns):")
print(Phi)
""")

md("""# Verify orthogonality

Mass-normalized modes should diagonalize `M` to the identity and `K` to `diag(omega_i^2)`:

`Phi^T M Phi = I`
`Phi^T K Phi = diag(wn^2)`

If the off-diagonal terms aren't ~0, something's wrong with the eigensolve or normalization.""")

code("""
Mr = Phi.T @ M @ Phi
Kr = Phi.T @ K @ Phi

print("Phi^T M Phi (should be identity):")
print(np.round(Mr, 8))

print("\\nPhi^T K Phi (should be diag(wn^2)):")
print(np.round(Kr, 6))

print("\\ndiag(wn^2) for comparison:")
print(np.round(wn**2, 6))

off_diag_M = Mr - np.diag(np.diag(Mr))
off_diag_K = Kr - np.diag(np.diag(Kr))
print(f"\\nmax |off-diagonal| in Phi^T M Phi: {np.max(np.abs(off_diag_M)):.2e}")
print(f"max |off-diagonal| in Phi^T K Phi: {np.max(np.abs(off_diag_K)):.2e}")
""")

md("""# FRF: modal sum vs. direct matrix inversion

The direct FRF comes from inverting the dynamic stiffness matrix at each frequency:

`H(w) = [K - w^2 M]^-1`

The modal sum reconstructs the same thing from the mode shapes and natural frequencies, with
no per-frequency matrix inversion, only a sum over modes:

`H(w) = sum_i  (phi_i phi_i^T) / (wn_i^2 - w^2)`

(undamped case — each mode contributes a rank-1 term that blows up at its own resonance). If
`solve_modes` is correct, these two should agree everywhere except exactly at resonance.""")

code("""
def frf_direct(w, M, K):
    \"\"\"H(w) = [K - w^2 M]^-1, evaluated at each frequency in w. Returns (n, n, len(w)).\"\"\"
    n = M.shape[0]
    H = np.zeros((n, n, len(w)), dtype=complex)
    for k, wk in enumerate(w):
        H[:, :, k] = np.linalg.inv(K - wk**2 * M)
    return H


def frf_modal(w, wn, Phi):
    \"\"\"Undamped modal-sum reconstruction of the same FRF.\"\"\"
    n = Phi.shape[0]
    H = np.zeros((n, n, len(w)), dtype=complex)
    for i in range(len(wn)):
        outer = np.outer(Phi[:, i], Phi[:, i])
        H += outer[:, :, None] / (wn[i]**2 - w[None, None, :]**2)
    return H


# sweep frequency, avoiding exact resonances so the direct inverse stays finite
w_sweep = np.linspace(0.1, 1.3 * wn[-1], 1500)
for wr in wn:
    w_sweep = w_sweep[np.abs(w_sweep - wr) > 0.5]

H_direct = frf_direct(w_sweep, M, K)
H_modal = frf_modal(w_sweep, wn, Phi)

diff = np.abs(H_direct - H_modal)
print(f"max |H_direct - H_modal| over the sweep: {np.max(diff):.2e}")
print(f"max |H_direct| over the sweep (for scale): {np.max(np.abs(H_direct)):.2e}")
""")

code("""
# visualize: drive point 0, measure at every DOF (H[i, 0, :] column)
drive = 0
fig, ax = plt.subplots(figsize=(9, 5))
for i in range(M.shape[0]):
    ax.semilogy(w_sweep, np.abs(H_direct[i, drive, :]), lw=2.5, alpha=0.5,
                label=f"direct, DOF {i}" if i == 0 else None, color=f"C{i}")
    ax.semilogy(w_sweep, np.abs(H_modal[i, drive, :]), "--", lw=1.2,
                label=f"modal sum, DOF {i}" if i == 0 else None, color="k")
for wr in wn:
    ax.axvline(wr, color="gray", ls=":", lw=1)
ax.set(title=f"FRF magnitude, driven at DOF {drive}: direct inverse vs. modal sum",
       xlabel="frequency [rad/s]", ylabel="|H| [m/N]")
ax.legend()
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig("mdof_frf_check.png", dpi=130)
print("Saved figure -> mdof_frf_check.png")
""")

md("""# Animate the mode shapes

Each mode is a pattern of relative motion at a single frequency: `x(t) = phi_i cos(wn_i t)`.
Plotting each mass's displacement about its equilibrium position over one period shows how the
DOFs move in and out of phase with each other for that mode.""")

code("""
def animate_mode(mode_index, wn, Phi, n_frames=60, amplitude_scale=0.3, spacing=1.0):
    \"\"\"Animate a single mode shape: masses oscillate about equally-spaced equilibrium points.\"\"\"
    n = Phi.shape[0]
    shape = Phi[:, mode_index]
    shape = shape / np.max(np.abs(shape))          # normalize for a consistent visual amplitude
    x_eq = np.arange(1, n + 1) * spacing            # equilibrium positions along the chain

    period = 2 * np.pi / wn[mode_index]
    t = np.linspace(0, period, n_frames, endpoint=False)

    fig, ax = plt.subplots(figsize=(7, 2.5))
    ax.set_xlim(0, (n + 1) * spacing)
    ax.set_ylim(-1.5, 1.5)
    ax.set_title(f"Mode {mode_index + 1}:  fn = {wn[mode_index] / (2*np.pi):.2f} Hz")
    ax.set_xlabel("equilibrium position along chain")
    ax.set_yticks([])
    ax.axhline(0, color="gray", lw=0.5)

    points, = ax.plot([], [], "o", ms=16, color=f"C{mode_index}")
    stems = [ax.plot([], [], "-", color="gray", lw=1)[0] for _ in range(n)]

    def init():
        points.set_data([], [])
        for s in stems:
            s.set_data([], [])
        return [points, *stems]

    def update(frame):
        y = amplitude_scale * shape * np.cos(wn[mode_index] * t[frame])
        points.set_data(x_eq, y)
        for i, s in enumerate(stems):
            s.set_data([x_eq[i], x_eq[i]], [0, y[i]])
        return [points, *stems]

    anim = FuncAnimation(fig, update, frames=n_frames, init_func=init,
                          interval=1000 / n_frames * period / (period / n_frames), blit=True)
    return fig, anim


fig, anim = animate_mode(0, wn, Phi)
anim.save("mdof_mode1.gif", writer=PillowWriter(fps=30))
plt.close(fig)
print("Saved animation -> mdof_mode1.gif")
""")

code("""
# save an animation for every mode
for i in range(M.shape[0]):
    fig, anim = animate_mode(i, wn, Phi)
    anim.save(f"mdof_mode{i+1}.gif", writer=PillowWriter(fps=30))
    plt.close(fig)
    print(f"Saved animation -> mdof_mode{i+1}.gif")
""")

nb['cells'] = cells
with open("mdof.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("wrote mdof.ipynb")
