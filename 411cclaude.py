import numpy as np

# --- 1. Paramètres physiques ---
HBAR = 1.0
M = 1.0
K0 = 2.0
A = 0.5

# --- 2. Paramètres de la barrière (MODIFIABLES LIBREMENT) ---
V0_hauteur = 0.0
largeur_a = 50.0
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

# --- 7. Initialisation (2 tranches seulement) ---
psi_curr = np.array([GaussWP(K0, A, x[i], 0) for i in range(nx)], dtype=complex)
psi_next = np.zeros(nx, dtype=complex)

seuil_amplitude = np.max(np.abs(psi_curr)**2) * 0.01

# --- 8. Résolution + chronométrage en une seule passe ---
t_entree = 0.0
t_sortie = 0.0
deja_entre = False
deja_sorti = False

for j in range(nt - 1):
    # Euler explicite
    d2psi = (psi_curr[:-2] - 2 * psi_curr[1:-1] + psi_curr[2:]) / (dx**2)
    psi_next[1:-1] = psi_curr[1:-1] - (1j * dt / HBAR) * (-(HBAR**2 / (2*M)) * d2psi + V[1:-1] * psi_curr[1:-1])
    psi_next[0] = 0
    psi_next[-1] = 0

    # Chronométrage à chaque pas
    proba = np.abs(psi_next)**2
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

    # Échanger les tranches
    psi_curr, psi_next = psi_next, psi_curr

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
