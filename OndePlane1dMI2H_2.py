# =============================================================================
# OndePlane1dMI2H.py
# Partie 1 du projet : Onde plane et superposition de 3 ondes planes
# =============================================================================

# --- Étape 1 : Importations ---
from numpy import pi, exp, sqrt, real, imag, linspace  # fonctions mathématiques de NumPy
import matplotlib.pyplot as plt  # bibliothèque pour tracer des graphiques


# =============================================================================
# PARTIE 1.1 : Fonction de l'onde plane
# =============================================================================

def PlaneWave(amp, k, omega, x, t):
    """
    Calcule une onde plane complexe en 1 dimension.

    En physique, une onde plane s'écrit : psi(x,t) = A * exp(i*(k*x - omega*t))
    - amp   : amplitude A de l'onde (hauteur maximale)
    - k     : nombre d'onde (relie a la longueur d'onde par k = 2*pi/lambda)
    - omega : pulsation (relie a la frequence par omega = 2*pi*f)
    - x     : position(s) ou on evalue l'onde
    - t     : instant auquel on evalue l'onde
    """
    # La phase de l'onde : phi = k*x - omega*t
    # k*x : composante spatiale (combien de cycles par unite de longueur)
    # omega*t : composante temporelle (l'onde avance dans le temps)
    phase = k * x - omega * t

    # L'onde plane complexe : A * e^(i*phase)
    # exp(i*phase) = cos(phase) + i*sin(phase) (formule d'Euler)
    # Donc la partie reelle = A*cos(phase) et la partie imaginaire = A*sin(phase)
    psi = amp * exp(1j * phase)

    return psi


# --- Parametres pour la demonstration de l'onde plane ---
A = 1.0        # amplitude de l'onde (sans unite ici)
k0 = 5.0       # nombre d'onde central [rad/m] (5 oscillations par 2*pi metres)
omega0 = 1.0   # pulsation [rad/s]
t0 = 0.0       # instant d'observation (on regarde l'onde a t=0)

# --- Grille spatiale ---
# On cree 1000 points entre -2*pi et 2*pi pour avoir un trace lisse
x_demo = linspace(-2 * pi, 2 * pi, 1000)

# --- Calcul de l'onde plane en chaque point x, a l'instant t0 ---
psi_demo = PlaneWave(A, k0, omega0, x_demo, t0)

# --- Trace de la partie reelle et imaginaire ---
fig, ax = plt.subplots()  # cree une figure et un axe

# Partie reelle : Re[psi] = A*cos(k*x - omega*t)
ax.plot(x_demo, real(psi_demo), label="Partie réelle (cos)", color="blue")

# Partie imaginaire : Im[psi] = A*sin(k*x - omega*t)
ax.plot(x_demo, imag(psi_demo), label="Partie imaginaire (sin)", color="red", linestyle="--")

ax.set_xlabel("Position x [m]")          # legende axe horizontal
ax.set_ylabel("Amplitude")                # legende axe vertical
ax.set_title("Onde plane à t = 0")        # titre du graphique
ax.legend()                               # affiche la legende (noms des courbes)
ax.grid(True)                             # affiche une grille pour la lisibilite
plt.tight_layout()                        # ajuste les marges
plt.show()                                # affiche le graphique


# =============================================================================
# PARTIE 1.2 : Superposition de 3 ondes planes
# =============================================================================

# --- Parametres physiques ---
k0 = 5.0       # nombre d'onde central [rad/m]
dk = 1.0       # ecart en nombre d'onde (Delta k)
A = 1.0        # amplitude de l'onde centrale
t = 0.0        # instant d'observation

# --- Grille spatiale ---
# L'intervalle est [-pi/dk, pi/dk] comme demande dans le sujet
# C'est l'intervalle ou l'enveloppe fait exactement un "battement" complet
x_min = -pi / dk   # borne gauche
x_max = pi / dk    # borne droite
npts = 1000        # nombre de points (pour un trace lisse)
x = linspace(x_min, x_max, npts)  # tableau de positions regulierement espacees

# --- Calcul des 3 ondes planes a t=0 ---

# Onde 1 : onde centrale, nombre d'onde k0, amplitude A
# C'est l'onde "principale" du paquet
onde1 = PlaneWave(A, k0, 0.0, x, t)

# Onde 2 : decalee vers les petits k, nombre d'onde k0 - dk/2, amplitude A/2
# Elle oscille un peu plus lentement que l'onde centrale (k plus petit = lambda plus grand)
onde2 = PlaneWave(A / 2, k0 - dk / 2, 0.0, x, t)

# Onde 3 : decalee vers les grands k, nombre d'onde k0 + dk/2, amplitude A/2
# Elle oscille un peu plus vite que l'onde centrale (k plus grand = lambda plus petit)
onde3 = PlaneWave(A / 2, k0 + dk / 2, 0.0, x, t)

# --- Superposition (somme) des 3 ondes ---
# En mecanique quantique, le principe de superposition dit que
# si psi1, psi2, psi3 sont des solutions, alors psi1+psi2+psi3 aussi
somme = onde1 + onde2 + onde3

# --- Calcul de l'enveloppe ---
# L'enveloppe est le module (valeur absolue) de l'onde complexe totale
# |psi| = sqrt(Re(psi)^2 + Im(psi)^2)
# Elle represente l'amplitude locale du paquet : la ou l'enveloppe est grande,
# les ondes s'additionnent (interference constructive)
# la ou elle est petite, elles s'annulent (interference destructive)
enveloppe = sqrt(real(somme)**2 + imag(somme)**2)

# --- Trace ---
fig2, ax2 = plt.subplots(figsize=(10, 6))  # nouvelle figure plus large

# Partie reelle de chaque onde individuelle (en traits fins)
ax2.plot(x, real(onde1), label="Onde 1 (k₀, amp=A)", color="blue", linewidth=0.8)
ax2.plot(x, real(onde2), label="Onde 2 (k₀-Δk/2, amp=A/2)", color="green", linewidth=0.8)
ax2.plot(x, real(onde3), label="Onde 3 (k₀+Δk/2, amp=A/2)", color="orange", linewidth=0.8)

# Partie reelle de la somme (en trait plus epais)
ax2.plot(x, real(somme), label="Somme (partie réelle)", color="black", linewidth=1.5)

# Enveloppe (module) en pointilles rouges epais
ax2.plot(x, enveloppe, label="Enveloppe |ψ|", color="red", linewidth=2, linestyle="--")
ax2.plot(x, -enveloppe, color="red", linewidth=2, linestyle="--")  # enveloppe inferieure (symetrie)

ax2.set_xlabel("Position x [m]")
ax2.set_ylabel("Amplitude")
ax2.set_title("Superposition de 3 ondes planes et enveloppe")
ax2.legend(loc="upper right")
ax2.grid(True)
plt.tight_layout()
plt.show()


# =============================================================================
# EXPLICATION PHYSIQUE : Pourquoi la somme de 3 ondes cree une enveloppe ?
# =============================================================================
#
# Chaque onde plane oscille a une frequence spatiale differente (k different).
#
# - Au centre (x=0), les trois ondes ont toutes la meme phase (elles sont
#   toutes au maximum en meme temps). Elles s'additionnent : c'est
#   l'INTERFERENCE CONSTRUCTIVE. L'amplitude totale est maximale.
#
# - En s'eloignant du centre, les ondes 2 et 3 (qui ont des k differents)
#   commencent a se decaler par rapport a l'onde 1. Leurs cretes ne coincident
#   plus avec celles de l'onde 1. Elles finissent par s'annuler mutuellement :
#   c'est l'INTERFERENCE DESTRUCTIVE. L'amplitude totale diminue.
#
# - Ce phenomene se repete periodiquement, creant des "battements" :
#   des zones de forte amplitude alternant avec des zones de faible amplitude.
#
# L'enveloppe visualise ce phenomene : elle montre OU l'amplitude est forte
# (interference constructive) et OU elle est faible (interference destructive).
#
# Mathematiquement, la largeur du paquet est inversement proportionnelle a dk :
#   Largeur du paquet ~ 2*pi / dk
# Plus on ajoute d'ondes avec des k proches (dk petit), plus le paquet est large.
# Plus dk est grand, plus le paquet est localise dans l'espace.
# C'est une manifestation directe de la relation d'incertitude de Heisenberg :
#   Delta_x * Delta_k >= 1/2
# =============================================================================
