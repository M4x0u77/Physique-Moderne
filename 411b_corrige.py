import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# PARTIE 4.1.1b : Temps de traversée libre (sans barrière)
# Méthode de Saute-Mouton (Leapfrog symplectique)
# =============================================================================

# --- 1. Paramètres physiques ---
HBAR = 1.0
M = 1.0
K0 = 2.0
A = 0.5

# --- 2. Fonction GaussWP (paquet d'ondes gaussien analytique) ---
def GaussWP(k0, a, x, t):
    denom = (M * a**2 + 2j * HBAR * t)
    prefactor = (1 / (8 * np.pi**3))**0.25 * np.sqrt((4 * np.pi * M * a) / denom)
    exponent_term = (M / 4) * ((a**2 * k0 + 2j * x)**2 / denom) - (a**2 * k0**2) / 4
    return prefactor * np.exp(exponent_term)

# --- 3. Paramètres de simulation ---
nx = 600       # Nombre de points spatiaux
nt = 60000     # Beaucoup d'étapes de temps (dt petit pour la précision)

V0_hauteur = 0.0    # PAS DE BARRIÈRE (particule libre)
largeur_a = 5.0     # La "largeur" qu'on chronomètre
position_mur = 15.0  # Le point de départ du chronomètre

# Construction du potentiel (ici tout est à zéro, mais on garde la structure)
V = np.zeros(nx)
x = np.linspace(-40, 60, nx)
for i in range(nx):
    if position_mur <= x[i] <= position_mur + largeur_a:
        V[i] = V0_hauteur

t = np.linspace(0, 10, nt)
dx = x[1] - x[0]
dt = t[1] - t[0]

# --- 4. Initialisation (séparation Réel / Imaginaire) ---
# Convention : LIGNES = temps (nt), COLONNES = espace (nx)
re = np.zeros((nt, nx))  # Partie réelle de psi(x, t)
im = np.zeros((nt, nx))  # Partie imaginaire de psi(x, t)

# Condition initiale : psi(x, 0) = R(x,0) + i*I(x,0)
psi0 = GaussWP(K0, A, x, 0)
re[0, :] = np.real(psi0)
im[0, :] = np.imag(psi0)

# --- 5. Coefficient pour la boucle ---
# Coefficient devant la dérivée seconde : ℏ/(2m·dx²)
coeff = HBAR / (2 * M * dx**2)

# =============================================================================
# --- 6. BOUCLE DE RÉSOLUTION : SCHÉMA SAUTE-MOUTON VECTORISÉ ---
# =============================================================================
# L'idée du Saute-Mouton pour l'équation de Schrödinger :
#   iℏ ∂ψ/∂t = -ℏ²/(2m) ∂²ψ/∂x² + V·ψ
#
# En séparant ψ = R + iI, on obtient deux équations couplées :
#   ∂I/∂t = +(ℏ/2m) ∂²R/∂x² - (V/ℏ)·R
#   ∂R/∂t = -(ℏ/2m) ∂²I/∂x² + (V/ℏ)·I
#
# Le schéma "saute-mouton" consiste à les résoudre EN ALTERNANCE :
#   1) I^{n+1} calculé avec R^n       (le R "ancien")
#   2) R^{n+1} calculé avec I^{n+1}   (le I "nouveau", qu'on vient de trouver)
#
# Pourquoi ça marche ? R et I sont en quadrature de phase (déphasés de π/2).
# En les mettant à jour l'un après l'autre, on respecte cette oscillation
# naturelle, et la quantité R² + I² reste conservée (schéma symplectique).
# C'est exactement comme le schéma de Verlet en mécanique classique :
# la position utilise la nouvelle vitesse, la vitesse utilise l'ancienne position.
# =============================================================================

for j in range(nt - 1):

    # =========================================================================
    # ÉTAPE A : Mise à jour de I (partie imaginaire) avec R^n (l'ancien réel)
    # =========================================================================
    # Dérivée seconde spatiale de R à l'instant j :
    # Vectorisation NumPy : au lieu d'une boucle for i in range(1, nx-1),
    # on calcule TOUS les points intérieurs d'un coup avec les slices :
    #   [2:]   = tous les points de i=2 à la fin    (= voisin de droite)
    #   [1:-1] = tous les points de i=1 à i=nx-2    (= point central)
    #   [:-2]  = tous les points de i=0 à i=nx-3    (= voisin de gauche)
    d2re = re[j, 2:] - 2*re[j, 1:-1] + re[j, :-2]

    # Application de l'Eq.1 discrétisée (V=0 ici, mais on garde le terme)
    im[j+1, 1:-1] = (im[j, 1:-1]
                      + dt * coeff * d2re
                      - dt * (V[1:-1] / HBAR) * re[j, 1:-1])

    # Conditions aux limites : l'onde est nulle aux bords
    im[j+1, 0] = 0
    im[j+1, -1] = 0

    # =========================================================================
    # ÉTAPE B : Mise à jour de R (partie réelle) avec I^{n+1} (le NOUVEAU imag)
    # =========================================================================
    # C'est LE point crucial : on utilise im[j+1] qu'on vient de calculer,
    # PAS im[j]. C'est ce décalage qui stabilise le schéma.
    d2im = im[j+1, 2:] - 2*im[j+1, 1:-1] + im[j+1, :-2]

    # Application de l'Eq.2 discrétisée (attention au signe MOINS)
    re[j+1, 1:-1] = (re[j, 1:-1]
                      - dt * coeff * d2im
                      + dt * (V[1:-1] / HBAR) * im[j+1, 1:-1])

    # Conditions aux limites
    re[j+1, 0] = 0
    re[j+1, -1] = 0

# =============================================================================
# --- 7. Chronométrage du temps de traversée ---
# =============================================================================
# On cherche à quel instant le SOMMET de la densité de probabilité
# entre dans la zone [position_mur, position_mur + largeur_a] puis en sort.

t_entree = 0.0
t_sortie = 0.0

# Ces verrous (booléens) permettent de ne capturer le temps QUE la première fois
deja_entre = False
deja_sorti = False

# On parcourt le film ligne par ligne (dans le temps)
for j in range(nt):
    # Densité de probabilité à l'instant j : |psi|² = R² + I²
    proba_ligne = re[j, :]**2 + im[j, :]**2

    # On cherche la position du maximum (le "sommet" du paquet)
    indice_sommet = np.argmax(proba_ligne)
    x_sommet = x[indice_sommet]

    # Le sommet vient d'entrer dans la zone ?
    if x_sommet >= position_mur and not deja_entre:
        t_entree = t[j]
        deja_entre = True

    # Le sommet vient de sortir de la zone ?
    if x_sommet >= (position_mur + largeur_a) and not deja_sorti:
        t_sortie = t[j]
        deja_sorti = True

# Calcul de la durée numérique du voyage sur la distance 'a'
tau0_num = t_sortie - t_entree

# =============================================================================
# --- 8. Comparaison avec la théorie ---
# =============================================================================
# Vitesse de groupe théorique : v_g = ℏk₀/m
v_g_theorique = (HBAR * K0) / M

# Temps théorique pour parcourir la largeur 'a' : tau = a / v_g
tau0_th = largeur_a / v_g_theorique

# Erreur absolue entre la simulation et la théorie
erreur_absolue = abs(tau0_th - tau0_num)

# --- 9. Affichage des résultats ---
print("\n==========================================")
print("       RÉSULTATS DE LA QUESTION 1.b       ")
print("==========================================")
print(f"Entrée du sommet (x = {position_mur:.1f}) à : t = {t_entree:.4f} s")
print(f"Sortie du sommet (x = {position_mur + largeur_a:.1f}) à : t = {t_sortie:.4f} s")
print(f"--> Temps de parcours libre tau0_num = {tau0_num:.4f} s")
print(f"Temps théorique tau0_th = {tau0_th:.4f} s")
print(f"Erreur absolue  = {erreur_absolue:.4f} s")
print("==========================================\n")
