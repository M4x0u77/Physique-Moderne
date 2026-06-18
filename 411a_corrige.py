import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# PARTIE 4.1.1a : Observation visuelle de l'effet tunnel
# Méthode de Saute-Mouton (Leapfrog symplectique) avec potentiel V
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
nx = 2000
nt = 5000

V0_hauteur = 2.0    # La hauteur du mur (barrière de potentiel)
largeur_a = 5.0     # L'épaisseur du mur
position_mur = 15.0  # L'endroit où commence le mur

# Construction de la barrière de potentiel
V = np.zeros(nx)
x = np.linspace(-40, 60, nx)
for i in range(nx):
    if position_mur <= x[i] <= position_mur + largeur_a:
        V[i] = V0_hauteur

t = np.linspace(0, 10, nt)
dx = x[1] - x[0]
dt = t[1] - t[0]

# --- 4. Initialisation (séparation Réel / Imaginaire) ---
# Au lieu d'UN tableau complexe psi, on crée DEUX tableaux réels.
# Convention : les LIGNES = le temps (nt), les COLONNES = l'espace (nx)
# C'est la même organisation que dans ton code original.
re = np.zeros((nt, nx))  # Partie réelle de psi(x, t)
im = np.zeros((nt, nx))  # Partie imaginaire de psi(x, t)

# Condition initiale : on décompose psi(x, 0) = R(x,0) + i*I(x,0)
psi0 = GaussWP(K0, A, x, 0)
re[0, :] = np.real(psi0)
im[0, :] = np.imag(psi0)

# --- 5. Coefficient utile pour la boucle ---
# L'équation de Schrödinger iℏ ∂ψ/∂t = -ℏ²/(2m) ∂²ψ/∂x² + V·ψ
# se sépare en deux équations couplées (en posant ψ = R + iI) :
#   ∂I/∂t = +(ℏ/2m) ∂²R/∂x²  - (V/ℏ)·R    ... (Eq.1)
#   ∂R/∂t = -(ℏ/2m) ∂²I/∂x²  + (V/ℏ)·I    ... (Eq.2)
coeff = HBAR / (2 * M * dx**2)

# =============================================================================
# --- 6. BOUCLE DE RÉSOLUTION TEMPORELLE : SCHÉMA SAUTE-MOUTON ---
# =============================================================================
# RAPPEL : Pourquoi Euler complexe est instable ?
# Euler calcule psi^{n+1} = psi^n + dt*f(psi^n). Pour Schrödinger, le
# multiplicateur complexe fait que |psi| croît à chaque pas --> explosion.
#
# Le Saute-Mouton corrige cela en mettant à jour R et I de façon ALTERNÉE :
#   1) On calcule I^{n+1} avec l'ANCIEN R^n         (Eq.1 discrétisée)
#   2) On calcule R^{n+1} avec le NOUVEAU I^{n+1}   (Eq.2 discrétisée)
# Ce couplage croisé conserve R² + I² = |psi|² --> schéma stable !
# =============================================================================

for j in range(nt - 1):

    # =========================================================================
    # ÉTAPE A : Mise à jour de la partie IMAGINAIRE (I^{n+1} à partir de R^n)
    # =========================================================================
    # Dérivée seconde spatiale de R à l'instant j (vectorisée avec slices NumPy) :
    #   re[j, 2:]   = R aux points i+1 (tout l'espace décalé à droite)
    #   re[j, 1:-1] = R aux points i   (le centre)
    #   re[j, :-2]  = R aux points i-1 (tout l'espace décalé à gauche)
    d2re = re[j, 2:] - 2*re[j, 1:-1] + re[j, :-2]  # = dx² * ∂²R/∂x²

    # Discrétisation de l'Eq.1 : I^{n+1} = I^n + dt * [(ℏ/2m)·∂²R/∂x² - (V/ℏ)·R]
    im[j+1, 1:-1] = (im[j, 1:-1]
                      + dt * coeff * d2re
                      - dt * (V[1:-1] / HBAR) * re[j, 1:-1])

    # Conditions aux limites (murs infinis aux bords du domaine)
    im[j+1, 0] = 0
    im[j+1, -1] = 0

    # =========================================================================
    # ÉTAPE B : Mise à jour de la partie RÉELLE (R^{n+1} à partir de I^{n+1})
    # =========================================================================
    # POINT CLÉ DU SAUTE-MOUTON :
    # On utilise im[j+1, :] qu'on vient TOUT JUSTE de calculer à l'étape A,
    # PAS l'ancienne valeur im[j, :].
    # C'est ce décalage temporel qui rend le schéma symplectique (conservatif).
    d2im = im[j+1, 2:] - 2*im[j+1, 1:-1] + im[j+1, :-2]  # = dx² * ∂²I/∂x²

    # Discrétisation de l'Eq.2 : R^{n+1} = R^n - dt * [(ℏ/2m)·∂²I/∂x² - (V/ℏ)·I]
    # Attention au signe MOINS global (c'est l'autre équation couplée)
    re[j+1, 1:-1] = (re[j, 1:-1]
                      - dt * coeff * d2im
                      + dt * (V[1:-1] / HBAR) * im[j+1, 1:-1])

    # Conditions aux limites
    re[j+1, 0] = 0
    re[j+1, -1] = 0

# =============================================================================
# --- 7. Affichage graphique : densité de probabilité à 3 instants ---
# =============================================================================
plt.figure(figsize=(10, 6))

# 1. On dessine la forme du mur pour bien le voir sur le graphique
plt.plot(x, V, color='black', linewidth=2, label="Barrière de potentiel")

# 2. On choisit 3 instants de notre "film" (Début, Milieu, Fin)
instants_a_tracer = [0, 2500, 4999]

# 3. Pour chaque instant, on reconstitue la densité |psi|² = R² + I²
for j_plot in instants_a_tracer:
    densite_proba = re[j_plot, :]**2 + im[j_plot, :]**2
    plt.plot(x, densite_proba, label=f"Particule à t={t[j_plot]:.1f}")

plt.legend()
plt.title("Effet tunnel : Rendu final (méthode Saute-Mouton)")
plt.xlabel("Position x")
plt.ylabel("Densité de probabilité")
plt.show()
