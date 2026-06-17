import numpy as np
import matplotlib.pyplot as plt

# --- 1. Paramètres physiques ---
HBAR = 1.0
M = 1.0
K0 = 2.0
A = 0.5
nx = 600      # Moins de points spatiaux (dx plus grand)
nt = 60000    # Beaucoup plus d'étapes de temps (dt minuscule)
V0_hauteur = 0.0  # La hauteur du mur
largeur_a = 10    # L'épaisseur du mur
position_mur = 15.0 # L'endroit où commence le mur

# --- 2. Fonction GaussWP ---
def GaussWP(k0, a, x, t):
    denom = (M * a**2 + 2j * HBAR * t)
    prefactor = (1 / (8 * np.pi**3))**0.25 * np.sqrt((4 * np.pi * M * a) / denom)
    exponent_term = (M / 4) * ((a**2 * k0 + 2j * x)**2 / denom) - (a**2 * k0**2) / 4
    return prefactor * np.exp(exponent_term)

# On crée une carte vide de l'espace (remplie de zéros)
V = np.zeros(nx)

x = np.linspace(-40, 60, nx)
# On y construit le mur brique par brique
for i in range(nx):
    if position_mur <= x[i] <= position_mur + largeur_a:
        V[i] = V0_hauteur



t = np.linspace(0, 10, nt)
dx = x[1] - x[0]
dt = t[1] - t[0]
# --- 4. CHANGEMENT ICI : nt lignes et nx colonnes ---
# Les lignes = le temps (nt)
# Les colonnes = l'espace (nx)
psi = np.empty((nt, nx), dtype=complex)

# "La première ligne doit contenir un paquet d'ondes" :
# On sélectionne la ligne 0, et toutes les colonnes (:) correspondant à l'espace x
for i in range(nx):
    # On calcule la valeur de la fonction d'onde à la position x[i]
    # et au temps t=0, puis on la range dans la case (0, i)
    psi[0, i] = GaussWP(K0, A, x[i], 0)

# --- 5. Résolution numérique ---

# 1ère boucle : on avance dans le TEMPS (ligne par ligne)
for j in range(nt - 1):

    # ÉTAPE A & B VECTORISÉES : On calcule tout l'espace d'un seul coup !
    # On utilise les "slices" [1:-1], [:-2] et [2:] de NumPy
    d2psi = (psi[j, :-2] - 2 * psi[j, 1:-1] + psi[j, 2:]) / (dx**2)

    psi[j+1, 1:-1] = psi[j, 1:-1] - (1j * dt / HBAR) * (- (HBAR**2 / (2*M)) * d2psi + V[1:-1] * psi[j, 1:-1])

    # ÉTAPE C : Conditions aux limites
    psi[j+1, 0] = 0           # Bord gauche
    psi[j+1, -1] = 0          # Bord droit (en Python, -1 désigne le dernier élément)




    # ÉTAPE C : Conditions aux limites
    # Une fois qu'on a calculé l'intérieur de l'espace pour le temps j+1,
    # on force explicitement les deux bords extrêmes à rester à zéro.
    psi[j+1, 0] = 0           # Bord gauche
    psi[j+1, nx - 1] = 0      # Bord droit

# --- 5. Parcours et Chronométrage (Algorithme fait main) ---

t_entree = 0.0
t_sortie = 0.0
deja_entre = False
deja_sorti = False

for j in range(nt):
    valeur_max_globale = -1.0
    indice_sommet_global = -1
    valeur_max_apres_mur = -1.0
    indice_sommet_apres_mur = -1

    for i in range(nx):
        proba_actuelle = abs(psi[j, i])**2

        if proba_actuelle > valeur_max_globale:
            valeur_max_globale = proba_actuelle
            indice_sommet_global = i

        if x[i] >= (position_mur + largeur_a):
            if proba_actuelle > valeur_max_apres_mur:
                valeur_max_apres_mur = proba_actuelle
                indice_sommet_apres_mur = i

    x_sommet_global = x[indice_sommet_global]
    x_sommet_apres_mur = x[indice_sommet_apres_mur]

    if V0_hauteur == 0.0:
        # --- MODIFICATION 2 appliquée ici : seuil_sortie au lieu de position_mur + largeur_a ---
        if x_sommet_global >= position_mur and not deja_entre:
            t_entree = t[j]
            deja_entre = True

        if x_sommet_global >= seuil_sortie and not deja_sorti:
            t_sortie = t[j]
            deja_sorti = True
    else:
        if x_sommet_global >= position_mur and not deja_entre:
            t_entree = t[j]
            deja_entre = True

        if deja_entre and not deja_sorti:
            if x_sommet_apres_mur > (position_mur + largeur_a + 2 * dx):
                t_sortie = t[j]
                deja_sorti = True

# Calcul du temps de traversée
tau_t_num = t_sortie - t_entree

print("\n==========================================")
print("       RÉSULTATS DE LA QUESTION 1.c       ")
print("==========================================")
print(f"Hauteur du mur (V0) = {V0_hauteur}")
print(f"Épaisseur du mur (a) = {largeur_a:.1f}") # Ajout de cette ligne
print(f"Entrée du sommet (x = {position_mur:.1f}) à : t = {t_entree:.4f} s")
print(f"Sortie du sommet transmis (x = {position_mur + largeur_a:.1f}) à : t = {t_sortie:.4f} s")
print(f"--> Temps de traversée (tau_t_num) = {tau_t_num:.4f} s")
print("==========================================\n")
