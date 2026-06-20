import numpy as np
import matplotlib.pyplot as plt

# --- 1. Paramètres (Identiques) ---
hbar= 1.0
m=1
k0=5
x_min, x_max = -20.0, 20.0
nx = 500
x_tab = np.linspace(x_min, x_max, nx)
dx = x_tab[1] - x_tab[0]
V0=20
largeur_a=4
a0=1

t_min, t_max = 0.0, 4.0
nt = 8000
t_tab = np.linspace(t_min, t_max, nt)
dt = t_tab[1] - t_tab[0]

T_entree=0
T_sortie=0

entree=False
sortie=False

Psi = np.zeros((nx, nt), dtype=complex)

def GaussWP(k0, a, x, t_val):#créer le paquet gaussien
    prefacteur = (1 / (8 * np.pi**3))**0.25
    denominateur = m * a**2 + 2j * hbar * t_val
    racine = np.sqrt((4 * np.pi * m * a) / denominateur)
    terme_exp_1 = (m / 4) * ((a**2 * k0 + 2j * x)**2 / denominateur)
    terme_exp_2 = (a**2 * k0**2) / 4
    return prefacteur * racine * np.exp(terme_exp_1 - terme_exp_2)

# --- Initialisation ---
Psi[:, 0] = GaussWP(k0, 1.0, x_tab, t_min)

V=np.zeros(nx)
for i in range (nx):
    if (x_tab[i]>=a0 and x_tab[i]<=a0+largeur_a):
        V[i]=V0


print("Début simulation Leapfrog...")

for x in range(1, nx - 1):#euler pour initialiser t=1
    d2_spatiale = (Psi[x+1, 0] - 2*Psi[x, 0] + Psi[x-1, 0]) / (dx**2)
    d_temporelle = (1j * hbar / (2 * m)) * d2_spatiale - (1j / hbar) * V[x] * Psi[x, 0]
    Psi[x, 1] = dt * d_temporelle + Psi[x,0]

for t in range(1, nt - 1):#saute mouton
    for x in range(1, nx - 1):
        d2_spatiale = (Psi[x+1, t] - 2*Psi[x, t] + Psi[x-1, t]) / (dx**2)
        d_temporelle = (1j * hbar / (2 * m)) * d2_spatiale - (1j / hbar) * V[x] * Psi[x, t]
        Psi[x, t+1] = Psi[x, t-1] + 2 * dt * d_temporelle


indice_entree = np.argmin(np.abs(x_tab - a0))
indice_sortie = np.argmin(np.abs(x_tab - (a0 + largeur_a)))

proba_entree_temps = np.abs(Psi[indice_entree, :])**2
proba_sortie_temps = np.abs(Psi[indice_sortie, :])**2


t_entree_idx = np.argmax(proba_entree_temps)
T_entree = t_tab[t_entree_idx]

t_sortie_idx = np.argmax(proba_sortie_temps)#t_sortie_idx = t_entree_idx + np.argmax(proba_sortie_temps[t_entree_idx:])
T_sortie = t_tab[t_sortie_idx]





v_groupe = hbar * k0 / m
Tnum =T_sortie - T_entree
T0 = largeur_a / v_groupe

print("==========================================")
print(f"Hauteur du mur (V0)  = {V0}")
print(f"Épaisseur du mur (a) = {largeur_a:.1f}")
print(f"Vitesse de groupe    = {v_groupe:.2f}")
print(f"τ théorique (a/v_g)  = {T0:.4f} s")
print(f"Entrée  (x = {a0:.1f}) à t = {T_entree:.4f} s")
print(f"Sortie  (x = {a0 + largeur_a:.1f}) à t = {T_sortie:.4f} s")
print(f"--> Temps de traversée (tau_num) = {Tnum:.4f} s")
if T0 > 0:
    erreur_rel = abs(Tnum - T0) / T0
    print(f"--> Écart relatif = {erreur_rel:.4f}")
print("==========================================\n")
