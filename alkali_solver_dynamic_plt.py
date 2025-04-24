import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal

# ----------------------- PHYSICAL PARAMETERS ----------------------- #
the_Value = 0
l = the_Value # azimuthal quantum number
alpha_c = 0.192 # core polarizability
a_l = 0.833   # inner hard wall radius (a.u.)

# === Effective potential === #
def V_eff(r, l, alpha_c, a_l):
    """
    COMBINED: l-dependtn core + potential
    """
    V = np.empty_like(r)
    # avoid points exactly at a_l to prevent errors #
    V[:] = -1.0 / r - alpha_c / (2 * (r**2 + a_l**2)**2) + l*(l+1)/(2*r**2)
    return V

# === Radial solver === #
def solve_radial(R, N, l, alpha_c, a_l, num_states):
    """
    Solve radial SE on [a_l, R] with N grid points.
    Returns:
      r           : radial grid (length N)
      energies    : lowest num_states eigenvalues
      wavefns     : eigenvectors shape (N, num_states)
    """
    # define grid just above a_l to avoid boundary case #
    r = np.linspace(a_l + 1e-6, R, N)
    h = r[1] - r[0]

    # effective pot #
    V = V_eff(r, l, alpha_c, a_l)

    # build 3-diag matrix #
    main_diag = 1.0/h**2 + V
    off_diag  = -0.5 / h**2 * np.ones(N-1)

    # solve for lowest num_states eigenpairs #
    energies, wavefns = eigh_tridiagonal(
        main_diag, off_diag,
        select='i', select_range=(0, num_states-1)
    )
    return r, energies, wavefns

# === Convergence Check === #
def find_converged_R(R_list, h_target, l, alpha_c, a_l, tol=1e-6):
    """
    Increase R until ground-state energy converges within tol.
    Chooses N so that grid spacing ≈ h_target.
    Returns converged R, N, and E0.
    """
    E_prev = None
    for R in R_list:
        N = max(500, int((R - a_l)/h_target))
        r, energies, _ = solve_radial(R, N, l, alpha_c, a_l, num_states)
        E0 = energies[0]
        if E_prev is None:
            print(f"R={R:.1f} a.u., N={N}, E0={E0:.8f}")
        else:
            dE = abs(E0 - E_prev)
            print(f"R={R:.1f} a.u., N={N}, E0={E0:.8f}, ΔE={dE:.2e}")
            if dE < tol:
                print("YES CONVERGE\n")
                return R, N, E0
        E_prev = E0
    raise ValueError("NO CONVERGE")


def x_expectation2(r, f_n):


    r1 = np.trapz(r * fn**2 * r**2, r)
    r2 = np.trapz(r**2 * fn**2 * r**2, r)

    delta_r_val = np.sqrt(r2 - r1**2)
    return delta_r_val

def p_expectation2(r, f_n):
    
    #This computes the standard deviation Δp = sqrt(<p²> - <p>²) for each wavefunction
    h_bar = 1
    dR = np.gradient(f_n, r)
    dR_scaled = r**2 * dR
    d2R_term = np.gradient(dR_scaled, r) / r**2
    integrand = f_n * d2R_term * r**2
    deltap = np.sqrt(-np.trapz(integrand, r))
    return deltap

# === Main execution ===
if __name__ == "__main__":

    num_states = 5
    # 1. Convergence in R
    R_list = [100, 120, 150, 175, 200, 250, 300, 400, 500]
    h_target = 0.0001  # target grid spacing (a.u.)
    R_conv, N, E0 = find_converged_R(R_list, h_target, l, alpha_c, a_l, tol=1e-6)

    # 2. Solve for first 5 states at converged R
    r, energies, wavefns = solve_radial(R_conv, N, l, alpha_c, a_l, num_states)

    # 3. Print energies
    print("\nEnergy levels (eV):")
    for i, E in enumerate(energies):
        E = E*27.2114 # Conversion Factor from a.u. to eV
        print(f"n={i}, E={E:.8f}")

    # 4. Plot first 3 radial wavefunctions R_n(r)
    plt.figure(figsize=(10,6))
    for n in range(5):
        un = wavefns[:, n]
        fn = un / r
        # normalize: ∫|R|^2 r^2 dr = 1
        norm = np.sqrt(np.trapz(fn**2 * r**2, r))
        fn /= norm
        plt.plot(r, fn, label=f'n={n}, E={energies[n]*27.2114:.6f} eV')
        print("For E_"+str(n)+": delta x = "+str(x_expectation2(r, fn)))
        print("For E_"+str(n)+": delta p = "+str(p_expectation2(r, fn)))
        print("For E_"+str(n)+": delta x * delta p = "+str(x_expectation2(r, fn)*p_expectation2(r, fn)))

    plt.xlabel('r (a.u.)')
    plt.ylabel(r'$f_n(r)$')
    plt.title(f'Li Wavefunctions l = {the_Value}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 5. Plot radial probability density
    plt.figure(figsize=(10,6))
    for m in range(5):
        fn0 = wavefns[:,m]
        R0 = fn0 / r
        norm0 = np.sqrt(np.trapz(R0**2 * r**2, r))
        R0 /= norm0
        P_r = R0**2 * r**2
        plt.plot(r, P_r, label=f'{m+3}d')
    
    plt.xlabel('r (a.u.)')
    plt.ylabel(r'$|R_0(r)|^2 r^2$')
    plt.title('Li Radial Probability Density')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()