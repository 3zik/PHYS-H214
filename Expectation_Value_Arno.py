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
        x1 = np.trapezoid(integrand1, r)
        x2 = np.trapezoid(integrand2, r)
        deltax = np.sqrt(x2 - x1**2)
        delxstorage.append(deltax)
    print(delxstorage)


def p_expectation(r, wavefns):
    #This computes the standard deviation Δp = sqrt(<p²> - <p>²) for each wavefunction
    h = 1.054571817 * 10**-34
    for i in range(wavefns.shape[1]):
        efunc = wavefns[:, i]
        integrand1 = np.gradient(h*efunc**2)
        integrand2 = np.gradient(-h**2*efunc**2)
        p1 = np.trapezoid(integrand1,r)
        p2 = np.transpose(integrand2,r)
        deltap = np.sqrat(p2 + p1**2)
        delpstorage.append(deltap)
    print(delpstorage)

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



print("hello world")
