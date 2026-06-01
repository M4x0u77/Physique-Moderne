import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# --- 1. Constantes ---
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

# --- 3. Validation Partie 1 ---
x = np.linspace(-50, 50, 2000)
t = 0
psi = GaussWP(K0, A, x, t)

# a. Normalisation (Question 1.d)
norme = np.trapz(np.abs(psi)**2, x)
print(f"Norme du paquet à t={t} : {norme:.4f}")

# b. Transformée de Fourier pour g(k) (Question 1.e)
# La FFT permet de passer de Psi(x) à g(k)
psi_fft = fft(psi)
k_freq = fftfreq(len(x), d=(x[1]-x[0])) * 2 * np.pi

plt.figure(figsize=(12, 5))

# Graphique de la fonction d'onde
plt.subplot(1, 2, 1)
plt.plot(x, np.abs(psi)**2)
plt.title("Densité de probabilité |ψ(x,0)|²")

# Graphique de g(k)
plt.subplot(1, 2, 2)
plt.plot(k_freq, np.abs(psi_fft))
plt.title("Distribution g(k) via FFT")
plt.xlim(-5, 5) # Zoom sur la zone d'intérêt
plt.show()