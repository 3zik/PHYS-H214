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
    """Loop R to find convergence of the ground‐state energy."""
    E_prev = None
    for R in R_list:
        E, _, _, _ = solve_radial(R, N, l, alpha_c, a_l, num_states=1)
        E0 = E[0]
        print(f"R = {R:.1f} a.u.  ->  E0 = {E0:.8f} au")
        if E_prev is not None and abs(E0 - E_prev) < tol:
            print(f"Converged at R = {R:.1f} a.u. within ΔE < {tol}")
            return R, E0
        E_prev = E0
    raise ValueError("NO CONVERGE")

def find_converged_R_dynamic(R_list, h_target, l, alpha_c, a_l, tol=1e-6):
    E_prev = None
    for R in R_list:
        N = int((R - a_l) / h_target)
        E, _, _, _ = solve_radial(R, N, l, alpha_c, a_l, num_states=1)
        E0 = E[0]
        if E_prev is not None:
            dE = abs(E0 - E_prev)
            print(f"R = {R:.1f} a.u., N = {N}, E0 = {E0:.8f}, ΔE = {dE:.2e}")
            if dE < tol:
                print("YES CONVERGE")
                return R, N, E0
        else:
            print(f"R = {R:.1f} a.u., N = {N}, E0 = {E0:.8f}")
        E_prev = E0

    raise ValueError("NO CONVERGE")


if __name__ == "__main__":
    alpha_c = 0.1915
    a_l     = 0.3
    l       = 0
    R_list  = [30, 40, 50, 60, 80, 100, 120, 150, 200, 300]
    h_target = 0.01
    tol = 1e-6

    R_conv, N_conv, E0 = find_converged_R_dynamic(
        R_list, h_target, l, alpha_c, a_l, tol=tol
    )
    print(f"\nFinal converged energy:\nR = {R_conv} a.u., N = {N_conv}, E0 = {E0:.8f} au")
