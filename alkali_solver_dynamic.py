import numpy as np
import scipy.linalg

def model_potential(r, alpha_c, a_l):
    """
    Generate the l‐dependent core + polarization potential V_l(r) on a grid r.
    """
    V = np.empty_like(r)
    V[r <= a_l] = np.inf
    idx = r > a_l
    V[idx] = -1.0 / r[idx] - alpha_c / (2.0 * (r[idx]**2 + a_l**2)**2)
    return V

def build_grid(R, N, a_l):
    """
    Create a uniform radial grid from r=a_l to r=R with N steps.
    Returns the grid r and step size h.
    """
    r = np.linspace(a_l, R, N+1)
    h = r[1] - r[0]
    return r, h

def build_hamiltonian(R, N, l, alpha_c, a_l):
    """
    Assemble the tridiagonal Hamiltonian H for given parameters.
    Returns diagonal (diag) and off‐diagonal (off) arrays, plus the full grid r.
    """
    r, h = build_grid(R, N, a_l)
    r_int = r[1:-1]  # interior points only
    V = model_potential(r_int, alpha_c, a_l)
    diag = 1.0/(h*h) + l*(l+1)/(2.0*r_int**2) + V
    off  = np.full(len(r_int)-1, -1.0/(2.0*h*h))
    return diag, off, r

def solve_radial(R, N, l, alpha_c, a_l, num_states=10):
    """
    Solve H f = E f using a tridiagonal solver.
    Returns:
      E     : array of eigenvalues (length num_states)
      f     : radial functions f_n(r)=r R_n(r) (shape (N+1, num_states))
      R_n   : radial wavefunctions R_n(r) (same shape as f)
      r     : radial grid (length N+1)
    """
    diag, off, r = build_hamiltonian(R, N, l, alpha_c, a_l)
    # select the first `num_states` eigenvalues by index
    select_range = (0, num_states - 1)
    E, vecs = scipy.linalg.eigh_tridiagonal(
        diag, off,
        select='i',
        select_range=select_range
    )
    
    # Add Dirichlet boundaries (zeros at endpoints)
    f = np.zeros((len(r), num_states))
    f[1:-1, :] = vecs
    
    # Normalize each f_n so ∫|f|^2 dr = 1
    h = r[1] - r[0]
    for n in range(num_states):
        norm = np.sqrt(np.sum(f[:, n]**2) * h)
        f[:, n] /= norm
    
    # Compute R_n(r) = f_n(r) / r
    R_n = f / r[:, None]
    return E, f, R_n, r

def find_converged_R(R_list, N, l, alpha_c, a_l, tol=1e-6):
    """
    Loop over candidate R values to find convergence of the ground‐state energy.
    """
    E_prev = None
    for R in R_list:
        E, _, _, _ = solve_radial(R, N, l, alpha_c, a_l, num_states=1)
        E0 = E[0]
        print(f"R = {R:.1f} a.u.  →  E0 = {E0:.8f} au")
        if E_prev is not None and abs(E0 - E_prev) < tol:
            print(f"Converged at R = {R:.1f} a.u. within ΔE < {tol}")
            return R, E0
        E_prev = E0
    raise ValueError("No convergence in provided R_list")

def find_converged_R_dynamic(R_list, h_target, l, alpha_c, a_l, tol=1e-6):
    """
    For each R in R_list, choose N so that Δr ≈ h_target,
    then solve and check convergence of E₀.
    """
    E_prev = None
    for R in R_list:
        # choose N to keep grid spacing ~ h_target
        N = max(100, int((R - a_l) / h_target))
        E, _, _, _ = solve_radial(R, N, l, alpha_c, a_l, num_states=1)
        E0 = E[0]
        print(f"R={R:4.0f} a.u.  N={N:5d}  → E0={E0:.8f}  "
              f"{'(ΔE={:.2e})'.format(E0-E_prev) if E_prev is not None else ''}")
        if E_prev is not None and abs(E0 - E_prev) < tol:
            print(f" Converged at R={R:.0f} a.u., ΔE={abs(E0-E_prev):.2e} au")
            return R, N, E0
        E_prev = E0
    raise ValueError("Still no convergence – try larger R_list or smaller h_target")


if __name__ == "__main__":
    # Example for Lithium (l=0):
    alpha_c = 0.1915   # core polarizability (au)
    a_l     = 0.3      # cut‑off radius (au)
    R_list = [30, 40, 50, 60, 80, 100, 120, 150, 200]
    N       = 2000     # number of radial steps
    l       = 0        # azimuthal Quantum number

    print(model_potential)


    R_conv, E0 = find_converged_R(R_list, N, l, alpha_c, a_l)
    print(f"\n→ Final choice: R = {R_conv:.1f} a.u.,  E0 = {E0:.8f} au")
