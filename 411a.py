import numpy as np
import matplotlib.pyplot as plt

# --- 1. Paramètres physiques ---
HBAR = 1.0
M = 1.0
K0 = 2.0
A = 0.5

# --- 2. Fonction GaussWP ---
def GaussWP(k0, a, x, t):
    denom = (M * a**2 + 2j * HBAR * t)
    prefactor = (1 / (8 * np.pi**3))**0.25 * np.sqrt((4 * np.pi * M * a) / denom)
    exponent_term = (M / 4) * ((a**2 * k0 + 2j * x)**2 / denom) - (a**2 * k0**2) / 4
    return prefactor * np.exp(exponent_term)

# --- 3. Paramètres de simulation ---
nx = 2000
nt = 5000


V0_hauteur = 2.0   # La hauteur du mur
largeur_a = 5.0    # L'épaisseur du mur
position_mur = 15.0 # L'endroit où commence le mur

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

    # 2ème boucle : on se déplace dans l'ESPACE (colonne par colonne)
    # On commence à l'indice 1 et on s'arrête à nx-2 pour ne pas toucher les bords
    for i in range(1, nx - 1):

        # ÉTAPE A : Calcul de la dérivée spatiale locale (d2psi)
        # On regarde le point à droite (i+1), le point actuel (i), et le point à gauche (i-1)
        # tout en restant sur la ligne de temps actuelle (j)
        d2psi = (psi[j, i+1] - 2 * psi[j, i] + psi[j, i-1]) / (dx**2)
        #                 calcul de la dérivé seconde spatiale avec la limite

        # ÉTAPE B : Calcul de l'état futur
        # On calcule la valeur de l'onde pour le futur immédiat (j+1) à la position (i)
        # en utilisant l'équation de Schrödinger et les valeurs du présent (j)
        psi[j+1, i] = psi[j, i] - (1j * dt / HBAR) * (- (HBAR**2 / (2*M)) * d2psi + V[i] * psi[j, i])
        #       1) isole dpsi/dt dans shrodinger




    # ÉTAPE C : Conditions aux limites
    # Une fois qu'on a calculé l'intérieur de l'espace pour le temps j+1,
    # on force explicitement les deux bords extrêmes à rester à zéro.
    psi[j+1, 0] = 0           # Bord gauche
    psi[j+1, nx - 1] = 0      # Bord droit


# --- 6. Affichage graphique ---
plt.figure(figsize=(10, 6))

# 1. On dessine la forme du mur pour bien le voir sur le graphique
plt.plot(x, V, color='black', linewidth=2, label="Barrière de potentiel")

# 2. On choisit 3 numéros d'images de notre film (Début, Milieu, Fin)
instants_a_tracer = [0, 2500, 4999]

# 3. On trace la particule pour ces 3 instants
for j_plot in instants_a_tracer:

    # AJOUT ICI : On crée le tableau vide à chaque fois qu'on change d'instant
    densite_proba = np.zeros(nx)

    for i in range(nx):
        # On calcule le module au carré pour trouver la probabilité
        densite_proba[i] = abs(psi[j_plot, i])**2

    plt.plot(x, densite_proba, label=f"Particule à t={t[j_plot]:.1f}")

plt.legend()
plt.title("Effet tunnel : Rendu final")
plt.xlabel("Position x")
plt.ylabel("Densité de probabilité")
plt.show()
