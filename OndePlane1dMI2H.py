import numpy as np
import matplotlib.pyplot as plt

# --- 2.a Constantes ---
HBAR = 1.0  # Constante de Planck réduite
M = 1.0     # Masse de la particule
K0 = 1.0    # Nombre d'onde central
A = 0.5     # Paramètre de largeur (à ajuster selon vos besoins)

# --- 2.b Fonction GaussWP ---
def GaussWP(k0, a, x, t):
    """
    Calcule le paquet d'ondes gaussien à une position x et un temps t.
    Utilise l'expression théorique donnée dans le document de cours.
    """
    # Calcul des termes temporels et spatiaux
    # Note: Cette formule découle de la résolution de l'équation de Schrödinger
    # pour un paquet gaussien libre.
    
    denom = (M * a**2 + 2j * HBAR * t)
    prefactor = (1 / (8 * np.pi**3))**0.25 * np.sqrt((4 * np.pi * M * a) / denom)
    
    exponent_term = (M / 4) * ((a**2 * K0 + 2j * x)**2 / denom) - (a**2 * K0**2) / 4
    
    psi = prefactor * np.exp(exponent_term)
    return psi

# --- 2.c Préparation de l'affichage ---
x = np.linspace(-50, 50, 1000)
t = 0
psi_t0 = GaussWP(K0, A, x, t)

plt.figure(figsize=(10, 5))
plt.plot(x, np.abs(psi_t0)**2, label=f"Densité de probabilité à t={t}")
plt.title("Paquet d'ondes gaussien à t=0")
plt.xlabel("x")
plt.ylabel("|ψ(x,t)|²")
plt.legend()
plt.grid(True)
plt.show()
