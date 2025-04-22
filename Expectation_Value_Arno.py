import numpy as np
import matplotlib.pyplot as plt

delxstorage = []
delpstorage = []
def x_expectation(r, wavefns):
    # This computes the standard deviation Δx = sqrt(<x²> - <x>²) for each wavefunction
    for i in range(wavefns.shape[1]):
        efunc = wavefns[:, i]
        integrand1 = r * efunc**2
        integrand2 = (r**2) * efunc**2
        x1 = np.trapz(integrand1, r) # can be trapezoid or trapz depending on numpy version
        x2 = np.trapz(integrand2, r) # can be trapezoid or trapz depending on numpy version
        deltax = np.sqrt(x2 - x1**2)
        delxstorage.append(deltax)
    return(delxstorage)

def p_expectation(r, wavefns):
    #This computes the standard deviation Δp = sqrt(<p²> - <p>²) for each wavefunction
    h_bar = 1.054571817 * 10**-34
    for i in range(wavefns.shape[1]):
        efunc = wavefns[:, i]
        integrand1 = h_bar * efunc * np.gradient(efunc) # also has a factor of 1/i
        integrand2 = -h_bar**2 * efunc * np.gradient(np.gradient(efunc))
        p1 = np.trapz(integrand1,r) # can be trapezoid or trapz depending on numpy version
        p2 = np.trapz(integrand2,r) # can be trapezoid or trapz depending on numpy version
        deltap = np.sqrt(p2 + p1**2)
        delpstorage.append(deltap)
    return(delpstorage)

def x_expectation2(r, f_n):
    integrand1 = r * f_n**2
    integrand2 = (r**2) * f_n**2
    x1 = np.trapz(integrand1, r) # can be trapezoid or trapz depending on numpy version
    x2 = np.trapz(integrand2, r) # can be trapezoid or trapz depending on numpy version
    deltax = np.sqrt(x2 - x1**2)
    return deltax


def p_expectation2(r, f_n):
    #This computes the standard deviation Δp = sqrt(<p²> - <p>²) for each wavefunction
    h_bar = 1 #Atomic Untis
    integrand1 = h_bar * f_n * np.gradient(f_n) # also has a factor of 1/i
    integrand2 = -h_bar**2 * f_n * np.gradient(np.gradient(f_n))
    p1 = np.trapz(integrand1,r) # can be trapezoid or trapz depending on numpy version
    p2 = np.trapz(integrand2,r) # can be trapezoid or trapz depending on numpy version
    deltap = np.sqrt(p2 + p1**2)
    return deltap

def uncertainty_energy(delp, delx, energies):
    uncertainty = delp*delx
    plt.plot(energies,uncertainty)
    plt.xlabel("Energies")
    plt.ylabel("Uncertainty")
    plt.title("Energies vs Uncertainty")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def normalize(wavefns):
    normalizedefunc = np.arra
    for i in range(wavefns.shape[1]):
        efunc = wavefns[:, i]
        A = np.sqrt(1/np.trapz(efunc**2))
        normalizedefunc = A*efunc
        wavefns[:,i] = normalizedefunc
    return wavefns