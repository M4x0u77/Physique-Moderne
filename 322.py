import numpy as np
import matplotlib.pyplot as plt
HBAR = 1.0
M = 1.0
K0 = 1.0
A = 0.5  # Paramètre de largeur du paquet

# --- 2. Fonction GaussWP ---
def GaussWP(k0, a, x, t):
    """
    Calcule le paquet d'ondes gaussien à une position x et un temps t.
    """
    denom = (M * a**2 + 2j * HBAR * t)
    prefactor = (1 / (8 * np.pi**3))**0.25 * np.sqrt((4 * np.pi * M * a) / denom)
    exponent_term = (M / 4) * ((a**2 * K0 + 2j * x)**2 / denom) - (a**2 * K0**2) / 4
    return prefactor * np.exp(exponent_term)

# Paramètres
nx = 2000
nt = 5000
V0 = 0

# Intervalles
x = np.linspace(-50, 50, nx)
t = np.linspace(0, 1, nt)
dx = x[1] - x[0]
dt = t[1] - t[0]

# Initialisation
psi = np.empty((nx, nt), dtype=complex)
psi[:, 0] = GaussWP(K0, A, x, 0)

# Résolution
for j in range(nt - 1):
    d2psi = (psi[2:, j] - 2*psi[1:-1, j] + psi[:-2, j]) / dx**2
    psi[1:-1, j+1] = psi[1:-1, j] - (1j * dt / HBAR) * (
        - (HBAR**2 / (2*M)) * d2psi + V0 * psi[1:-1, j]
    )
    psi[0, j+1] = 0
    psi[nx-1, j+1] = 0

# Comparaison
j_test = 100
t_test = t[j_test]
psi_num = psi[:, j_test]
psi_th = GaussWP(K0, A, x, t_test)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(x, np.abs(psi_num)**2, label="Numérique")
plt.plot(x, np.abs(psi_th)**2, label="Analytique", linestyle="--")
plt.title(f"Densité de probabilité à t={t_test:.2f}")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x, np.abs(psi_num - psi_th))
plt.title(f"Erreur absolue à t={t_test:.2f}")

plt.tight_layout()
plt.show()

erreur = np.max(np.abs(psi_num - psi_th)) / np.max(np.abs(psi_th))
print(f"Erreur relative maximale : {erreur:.4f}")
