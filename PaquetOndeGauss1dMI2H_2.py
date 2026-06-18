# =============================================================================
# PaquetOndeGauss1dMI2H.py
# Partie 2 du projet : Paquet d'ondes gaussien - mise en evidence de la
# difficulte d'echantillonnage et sa solution
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# QUESTIONS 2.2.2.a et b : Definition des constantes et de la fonction
# =============================================================================

# Constantes physiques (unites naturelles pour simplifier)
HBAR = 1.0   # constante de Planck reduite (h-barre)
M = 1.0      # masse de la particule
K0 = 1.0     # nombre d'onde central du paquet (determine sa vitesse)
a = 0.5      # parametre de largeur du paquet (plus a est petit, plus le paquet est etroit)


def GaussWP(k0, a, x, t):
    """
    Calcule l'amplitude de probabilite psi(x,t) d'un paquet d'ondes gaussien.
    C'est la solution analytique de l'equation de Schrodinger pour une particule libre.

    Parametres :
    - k0 : nombre d'onde central (relie a l'impulsion par p = hbar*k0)
    - a  : parametre de largeur initiale du paquet
    - x  : position(s) ou on evalue la fonction d'onde
    - t  : instant auquel on evalue la fonction d'onde

    Retourne : psi(x,t) complexe
    """
    # Le denominateur contient un terme imaginaire qui depend du temps
    # C'est lui qui provoque l'etalement du paquet au cours du temps
    denom = M * a**2 + 2j * HBAR * t

    # Le prefacteur assure la normalisation (integrale de |psi|^2 = 1)
    prefactor = (1 / (8 * np.pi**3))**0.25 * np.sqrt((4 * np.pi * M * a) / denom)

    # L'exposant contient la dependance spatiale et l'impulsion
    # C'est un terme quadratique en x qui donne la forme gaussienne
    exponent_term = (M / 4) * ((a**2 * k0 + 2j * x)**2 / denom) - (a**2 * k0**2) / 4

    # psi = prefacteur * e^(exposant)
    psi = prefactor * np.exp(exponent_term)

    return psi


# =============================================================================
# QUESTIONS 2.2.2.c et d : Mise en evidence de la difficulte
# =============================================================================
# On utilise volontairement TRES PEU de points (150) pour montrer le probleme
# d'echantillonnage (aussi appele "aliasing" ou "repliement de spectre").
#
# Le paquet d'ondes oscille a la frequence spatiale k0. Pour bien representer
# ces oscillations, il faut au minimum 2 points par longueur d'onde
# (theoreme de Shannon-Nyquist). Avec seulement 150 points sur [-50, 50],
# l'ecart entre deux points est dx = 100/149 = 0.67, or la longueur d'onde
# est lambda = 2*pi/k0 = 6.28. On a donc environ 9 points par longueur d'onde,
# ce qui est a la limite : le trace apparait "hache" et peu lisible.
# =============================================================================

# Grille spatiale a FAIBLE resolution (150 points)
x = np.linspace(-50, 50, 150)

# Calcul du paquet d'ondes a t=0 sur cette grille grossiere
psi = GaussWP(K0, a, x, 0)

# --- Figure 1 : trace avec peu de points (probleme visible) ---
plt.figure()

# Partie reelle de psi : Re[psi] = amplitude * cos(k0*x) * enveloppe gaussienne
plt.plot(x, np.real(psi), label="Partie réelle", color="blue")

# Partie imaginaire de psi : Im[psi] = amplitude * sin(k0*x) * enveloppe gaussienne
plt.plot(x, np.imag(psi), label="Partie imaginaire", color="red")

plt.xlabel("Position x")
plt.ylabel("Amplitude de ψ(x, 0)")
plt.title("Paquet d'ondes gaussien - 150 points (échantillonnage insuffisant)")
plt.legend()
plt.grid(True)
plt.show()

# On observe que les courbes sont "en dents de scie" : les oscillations rapides
# de l'onde (cos et sin) ne sont pas correctement capturees par si peu de points.
# C'est la difficulte demandee par la question 2.2.2.d.


# =============================================================================
# QUESTION 2.2.2.e : La solution - augmenter le nombre de points + enveloppe
# =============================================================================
# La solution est simple : on augmente le nombre de points d'echantillonnage.
# Avec 2000 points sur [-50, 50], on a dx = 100/1999 = 0.05, soit environ
# 125 points par longueur d'onde. Les oscillations sont parfaitement resolues.
#
# De plus, on trace l'ENVELOPPE du paquet (|psi|) qui montre la forme
# gaussienne sous-jacente, independamment des oscillations rapides.
# L'enveloppe represente l'amplitude locale de la probabilite de presence.
# =============================================================================

# Grille spatiale a HAUTE resolution (2000 points)
x_fin = np.linspace(-50, 50, 2000)

# Recalcul du paquet d'ondes sur la grille fine
psi_fin = GaussWP(K0, a, x_fin, 0)

# --- Figure 2 : trace avec beaucoup de points (probleme resolu) ---
plt.figure()

# Partie reelle : maintenant les oscillations sont lisses
plt.plot(x_fin, np.real(psi_fin), label="Partie réelle", color="blue", linewidth=0.8)

# Partie imaginaire : egalement lisse
plt.plot(x_fin, np.imag(psi_fin), label="Partie imaginaire", color="red", linewidth=0.8)

# Enveloppe superieure : |psi| = sqrt(Re^2 + Im^2)
# C'est la forme gaussienne qui "contient" les oscillations
# Physiquement, |psi|^2 donne la densite de probabilite de trouver la particule
plt.plot(x_fin, np.abs(psi_fin), label="Enveloppe |ψ|", color="black", linewidth=2, linestyle="--")

# Enveloppe inferieure : -|psi| (symetrie pour visualiser le "tube" qui contient l'onde)
plt.plot(x_fin, -np.abs(psi_fin), color="black", linewidth=2, linestyle="--")

plt.xlabel("Position x")
plt.ylabel("Amplitude de ψ(x, 0)")
plt.title("Paquet d'ondes gaussien - 2000 points (résolution correcte)")
plt.legend()
plt.grid(True)
plt.show()

# =============================================================================
# EXPLICATION PHYSIQUE :
#
# Le paquet d'ondes gaussien est le produit de deux choses :
#   1) Une oscillation rapide : exp(i*k0*x) = cos(k0*x) + i*sin(k0*x)
#      -> c'est la "porteuse", elle oscille a la frequence spatiale k0
#   2) Une enveloppe gaussienne : exp(-x^2 / largeur^2)
#      -> c'est la "forme" du paquet, elle localise la particule dans l'espace
#
# Avec 150 points, on n'a pas assez de resolution pour suivre les oscillations
# rapides de la porteuse -> le graphique est "hache" et inexploitable.
#
# Avec 2000 points, chaque oscillation est tracee avec ~125 points -> le trace
# est lisse et on voit clairement :
#   - Les oscillations (partie reelle et imaginaire) contenues dans l'enveloppe
#   - L'enveloppe gaussienne (en pointilles) qui delimite l'amplitude maximale
#
# C'est exactement comme en traitement du signal : pour echantillonner un signal
# de frequence f, il faut au minimum 2*f points par seconde (Shannon-Nyquist).
# =============================================================================
