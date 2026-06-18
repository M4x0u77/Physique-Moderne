import numpy as np

# --- 1. Paramètres physiques ---
HBAR = 1.0
M = 1.0
K0 = 2.0
A = 0.5

# --- 2. Paramètres de la barrière (MODIFIABLES LIBREMENT) ---
V0_hauteur = 5.0
largeur_a = 5.0
position_mur = 15.0

# --- 3. Fonction GaussWP ---
def GaussWP(k0, a, x, t):
    denom = (M * a**2 + 2j * HBAR * t)
    prefactor = (1 / (8 * np.pi**3))**0.25 * np.sqrt((4 * np.pi * M * a) / denom)
    exponent_term = (M / 4) * ((a**2 * k0 + 2j * x)**2 / denom) - (a**2 * k0**2) / 4
    return prefactor * np.exp(exponent_term)

# --- 4. Grille spatiale adaptative ---
v_groupe = HBAR * K0 / M
x_min = -40.0
x_max = position_mur + largeur_a + 40.0
nx = max(600, int((x_max - x_min) * 6))
x = np.linspace(x_min, x_max, nx)
dx = x[1] - x[0]

# --- 5. Grille temporelle (dt calé sur le cas stable connu) ---
t_max = 3.0 * (position_mur + largeur_a) / v_groupe
dt = 1.5e-4
nt = int(t_max / dt) + 1
t = np.linspace(0, t_max, nt)
dt = t[1] - t[0]

print(f"Simulation : nx={nx}, nt={nt}, dx={dx:.4f}, dt={dt:.2e}")
print(f"Domaine spatial : [{x_min}, {x_max:.1f}]")
print(f"Durée simulée : {t_max:.1f} s")
print("Calcul en cours...")

# --- 6. Potentiel ---
V = np.zeros(nx)
for i in range(nx):
    if position_mur <= x[i] <= position_mur + largeur_a:
        V[i] = V0_hauteur

# --- 7. Initialisation (méthode Saute-Mouton : séparation Réel / Imaginaire) ---
# On calcule le paquet d'ondes initial (complexe), puis on extrait R et I
psi_initial = GaussWP(K0, A, x, 0)
re = np.real(psi_initial).copy()  # Partie réelle de psi (tableau 1D)
im = np.imag(psi_initial).copy()  # Partie imaginaire de psi (tableau 1D)

# Seuil pour la détection du paquet transmis (basé sur la densité initiale)
seuil_amplitude = np.max(re**2 + im**2) * 0.01

# Coefficient qui apparaît dans les équations discrétisées : ℏ/(2m·dx²)
coeff = HBAR / (2 * M * dx**2)

# --- 8. Résolution + chronométrage en une seule passe ---
t_entree = 0.0
t_sortie = 0.0
deja_entre = False
deja_sorti = False

# =============================================================================
# BOUCLE TEMPORELLE : SCHÉMA SAUTE-MOUTON (LEAPFROG SYMPLECTIQUE)
# =============================================================================
# L'équation de Schrödinger iℏ ∂ψ/∂t = -ℏ²/(2m) ∂²ψ/∂x² + V·ψ
# se sépare en posant ψ = R + iI :
#   ∂I/∂t = +(ℏ/2m) ∂²R/∂x² - (V/ℏ)·R    ... (Eq.1)
#   ∂R/∂t = -(ℏ/2m) ∂²I/∂x² + (V/ℏ)·I    ... (Eq.2)
#
# Le Saute-Mouton met à jour I puis R de façon CROISÉE :
#   1) I^{n+1} utilise R^n       (l'ancien R)
#   2) R^{n+1} utilise I^{n+1}   (le NOUVEAU I qu'on vient de calculer)
#
# Ce décalage rend le schéma symplectique : il conserve |ψ|² = R² + I²
# à chaque pas de temps, contrairement à Euler qui fait exploser la norme.
# =============================================================================

for j in range(nt - 1):

    # =========================================================================
    # ÉTAPE A : Mise à jour de la partie IMAGINAIRE avec l'ANCIEN réel (R^n)
    # =========================================================================
    # Dérivée seconde spatiale de R (vectorisée avec slices NumPy) :
    #   re[2:]   = R[i+1]  (voisin de droite)
    #   re[1:-1] = R[i]    (point central)
    #   re[:-2]  = R[i-1]  (voisin de gauche)
    d2re = re[2:] - 2*re[1:-1] + re[:-2]  # = dx² * ∂²R/∂x²

    # Eq.1 discrétisée : I^{n+1} = I^n + dt·[(ℏ/2m)·∂²R/∂x² - (V/ℏ)·R]
    im[1:-1] = im[1:-1] + dt * coeff * d2re - dt * (V[1:-1] / HBAR) * re[1:-1]

    # Conditions aux limites (psi = 0 aux bords)
    im[0] = 0
    im[-1] = 0

    # =========================================================================
    # ÉTAPE B : Mise à jour de la partie RÉELLE avec le NOUVEAU imaginaire (I^{n+1})
    # =========================================================================
    # POINT CLÉ : on utilise le im qu'on vient TOUT JUSTE de modifier à l'étape A.
    # C'est ce "croisement" qui rend le schéma stable et conservatif.
    # Physiquement : R et I oscillent en quadrature (déphasés de π/2),
    # comme position et vitesse dans un oscillateur harmonique.
    d2im = im[2:] - 2*im[1:-1] + im[:-2]  # = dx² * ∂²I/∂x²

    # Eq.2 discrétisée : R^{n+1} = R^n - dt·[(ℏ/2m)·∂²I/∂x² - (V/ℏ)·I]
    # (Attention au signe MOINS devant : c'est l'autre équation couplée)
    re[1:-1] = re[1:-1] - dt * coeff * d2im + dt * (V[1:-1] / HBAR) * im[1:-1]

    # Conditions aux limites
    re[0] = 0
    re[-1] = 0

    # =========================================================================
    # Chronométrage à chaque pas (logique inchangée)
    # =========================================================================
    # Densité de probabilité : |ψ|² = R² + I² (remplace np.abs(psi)**2)
    proba = re**2 + im**2
    indice_max = np.argmax(proba)
    x_max_proba = x[indice_max]

    if V0_hauteur == 0.0:
        if x_max_proba >= position_mur and not deja_entre:
            t_entree = t[j+1]
            deja_entre = True
        if x_max_proba >= position_mur + largeur_a and not deja_sorti:
            t_sortie = t[j+1]
            deja_sorti = True
    else:
        if x_max_proba >= position_mur and not deja_entre:
            t_entree = t[j+1]
            deja_entre = True
        if deja_entre and not deja_sorti:
            masque_apres = x >= (position_mur + largeur_a)
            proba_apres = proba[masque_apres]
            if len(proba_apres) > 0 and np.max(proba_apres) > seuil_amplitude:
                idx_local = np.argmax(proba_apres)
                x_apres = x[masque_apres]
                if x_apres[idx_local] > position_mur + largeur_a + dx:
                    t_sortie = t[j+1]
                    deja_sorti = True

    if deja_entre and deja_sorti:
        break

# --- 9. Résultats ---
tau_t_num = t_sortie - t_entree
tau_theorique = largeur_a / v_groupe

print("\n==========================================")
print("       RÉSULTATS - Question 4.1.1.c       ")
print("==========================================")
print(f"Hauteur du mur (V0)  = {V0_hauteur}")
print(f"Épaisseur du mur (a) = {largeur_a:.1f}")
print(f"Vitesse de groupe    = {v_groupe:.2f}")
print(f"τ théorique (a/v_g)  = {tau_theorique:.4f} s")
print(f"Entrée  (x = {position_mur:.1f}) à t = {t_entree:.4f} s")
print(f"Sortie  (x = {position_mur + largeur_a:.1f}) à t = {t_sortie:.4f} s")
print(f"--> Temps de traversée (tau_num) = {tau_t_num:.4f} s")
if tau_theorique > 0:
    erreur_rel = abs(tau_t_num - tau_theorique) / tau_theorique
    print(f"--> Écart relatif = {erreur_rel:.4f}")
print("==========================================\n")
