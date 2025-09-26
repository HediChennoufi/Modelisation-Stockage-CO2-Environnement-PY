#Metier,Chennoufi,Muratore
import numpy as np
import matplotlib.pyplot as plt

# Temps
T_max = 50
dt = 0.1   
N = int(T_max / dt)


def S(C,alpha,K):
    return alpha * C * (1 - C/K)

def S_prime(C,alpha,K):
    return alpha * (1 - 2 * C/K)


def newton(CA_prev, CT_prev, CS_prev, dt, alpha,beta,delta,gamma,K,tol=1e-12, max_iter=10000):
    erreurs_it = []
    it = []
    #initialisation
    CA = CA_prev
    CT = CT_prev
    CS = CS_prev

    for _ in range(max_iter):
        # Calcul de F
        F1 = CA - CA_prev - dt * (-S(CT,alpha,K) + beta * CT + delta * CS)
        F2 = CT - CT_prev - dt * (S(CT,alpha,K) - beta * CT - delta * CT - gamma * CT)
        F3 = CS - CS_prev - dt * (gamma * CT - delta * CS + delta * CT)

        # Jacobienne coeff non nuls 
        a = -dt * (-S_prime(CT,alpha,K) + beta)
        b = -dt * delta
        c = 1 - dt * (S_prime(CT,alpha,K) - beta - delta - gamma)
        d = -dt * (gamma + delta)
        e = 1 + dt * delta

        # Résolution manuelle
        Delta_CT = -F2 / c
        Delta_CS = (-F3 - d * Delta_CT) / e
        Delta_CA = -F1 - a * Delta_CT - b * Delta_CS

        #calcul de l'erreur
        deltaX = np.array([Delta_CT, Delta_CS, Delta_CA])
        erreur = np.linalg.norm(deltaX,ord=np.inf) 
        erreurs_it.append(erreur)
        it.append(_)

        CA += Delta_CA
        CT += Delta_CT   
        CS += Delta_CS

        # Test de convergence
        if max(abs(Delta_CA),abs(Delta_CT), abs(Delta_CS)) < tol:
            return CA, CT, CS, erreurs_it,it

    print("Non convergence")


# Boucle de temps
def Programme(alpha=0.08,beta=0.02,delta=0.01,gamma=0.005,K=2000):
    erreurs_globales = []
    erreurs_newton_moy = []
    erreurs_newton_par_itération = []

    print("Démarrage du programme...")
    t = [0]
    C_A = [800]     # Carbone atmosphérique
    C_T = [500]     # Carbone dans les arbres
    C_S = [1500]     # Carbone dans les sols
    somme=[2800]
    erreurs = []

    print("Calculs en cours...")
    for n in range(N):
        t_n = t[-1]
        CA_n = C_A[-1]
        CT_n = C_T[-1]
        CS_n = C_S[-1]

        # Résolution implicite avec Newton
        CA_np1, CT_np1, CS_np1, erreurs_it, it = newton(CA_n, CT_n, CS_n, dt,alpha,beta,delta,gamma,K)

        t.append(t_n + dt)
        C_A.append(CA_np1)
        C_T.append(CT_np1)
        C_S.append(CS_np1)
        somme.append(CA_np1+ CT_np1+ CS_np1)

        
        #cv globale du système dans le temps(stabilité schéma implicite)
        erreurCA = CA_np1 - CA_n
        erreurCT = CT_np1 - CT_n
        erreurCS = CS_np1 - CS_n

        error_global = np.linalg.norm([erreurCA, erreurCT, erreurCS],ord=np.inf)
        erreurs_globales.append(error_global)
        
        erreurs_newton_moy.append(erreurs_it[-1])  # stocke l'erreur finale

    return(C_A,C_T,C_S,somme,t,erreurs_globales, erreurs_newton_moy, it)



# Affichage par "défaut"
C_A,C_T,C_S,somme,t,erreurs_globales, erreurs_newton_moy, it = Programme()
print("Affichage en cours...")
plt.plot(t, C_A, label="Atmosphère")
plt.plot(t, C_T, label="Arbres")
plt.plot(t, C_S, label="Sols")
plt.plot(t,somme,label="Somme")
plt.xlabel("Temps")
plt.ylabel("Concentration de carbone")
plt.title("Évolution des concentrations de carbone")
plt.grid()
plt.legend()
plt.show()


#Variation de alpha
alpha = [0.02,0.05,0.08,0.12]
k=0
for i in alpha:
    k = k + 1
    C_A,C_T,C_S,somme,t,erreurs_globales, erreurs_newton_moy, it = Programme(alpha=i)
    plt.subplot(2, 2, k)
    plt.plot(t, C_A, label="Atmosphère")
    plt.plot(t, C_T, label="Arbres")
    plt.plot(t, C_S, label="Sols")
    plt.plot(t,somme,label="Somme")
    plt.xlabel("Temps")
    plt.ylabel("Concentration de carbone")
    plt.title(f"alpha = {i}")
    plt.grid()
    plt.legend()
plt.tight_layout()
plt.show()


#Variation de beta
beta = [0.005,0.01,0.02,0.04]
k=0
for i in beta:
    k = k + 1
    C_A,C_T,C_S,somme,t,erreurs_globales, erreurs_newton_moy, it = Programme(beta=i)
    plt.subplot(2, 2, k)
    plt.plot(t, C_A, label="Atmosphère")
    plt.plot(t, C_T, label="Arbres")
    plt.plot(t, C_S, label="Sols")
    plt.plot(t,somme,label="Somme")
    plt.xlabel("Temps")
    plt.ylabel("Concentration de carbone")
    plt.title(f"beta = {i}")
    plt.grid()
    plt.legend()
plt.tight_layout()
plt.show()


#Variation de delta
delta = [0.002,0.005,0.01,0.02]
k=0
for i in delta:
    k = k + 1
    C_A,C_T,C_S,somme,t,erreurs_globales, erreurs_newton_moy, it = Programme(delta=i)
    plt.subplot(2, 2, k)
    plt.plot(t, C_A, label="Atmosphère")
    plt.plot(t, C_T, label="Arbres")
    plt.plot(t, C_S, label="Sols")
    plt.plot(t,somme,label="Somme")
    plt.xlabel("Temps")
    plt.ylabel("Concentration de carbone")
    plt.title(f"delta = {i}")
    plt.grid()
    plt.legend()
plt.tight_layout()
plt.show()


#Variation de gamma
gamma = [0.005,0.01,0.02,0.03]
k=0
for i in gamma:
    k = k + 1
    C_A,C_T,C_S,somme,t,erreurs_globales, erreurs_newton_moy, it = Programme(gamma=i)
    plt.subplot(2, 2, k)
    plt.plot(t, C_A, label="Atmosphère")
    plt.plot(t, C_T, label="Arbres")
    plt.plot(t, C_S, label="Sols")
    plt.plot(t,somme,label="Somme")
    plt.xlabel("Temps")
    plt.ylabel("Concentration de carbone")
    plt.title(f"gamma = {i}")
    plt.grid()
    plt.legend()
plt.tight_layout()
plt.show()


#Variation de K
K = [1000,2000,4000,6000]
k=0
for i in K:
    k = k + 1
    C_A,C_T,C_S,somme,t,erreurs_globales, erreurs_newton_moy, it = Programme(K=i)
    plt.subplot(2, 2, k)
    plt.plot(t, C_A, label="Atmosphère")
    plt.plot(t, C_T, label="Arbres")
    plt.plot(t, C_S, label="Sols")
    plt.plot(t,somme,label="Somme")
    plt.xlabel("Temps")
    plt.ylabel("Concentration de carbone")
    plt.title(f"K = {i}")
    plt.grid()
    plt.legend()
plt.tight_layout()
plt.show()

'''
C_A, C_T, C_S, somme, t, erreurs_globales, erreurs_newton_moy, it = Programme()
plt.figure(figsize=(8,5))
plt.plot(t[1:], erreurs_globales, marker='o', linestyle='-', color='purple')
plt.xlabel("temps")
plt.ylabel("Erreur entre 2 pas de temps")
plt.title("Erreur globale du schéma implicite")
plt.yscale("log")
plt.grid()
plt.tight_layout()
plt.show()


plt.figure(figsize=(8,5))
plt.plot(range(1, len(erreurs_newton_moy) + 1), erreurs_newton_moy, label="Erreur moyenne Newton")
plt.xlabel("Pas de temps")
plt.ylabel("Erreur L2 moyenne")
plt.title("Erreur de Newton en fonction du temps")
plt.yscale("log")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
'''

C_A, C_T, C_S, somme, t, erreurs_globales, erreurs_newton_moy, it = Programme()
plt.figure(figsize=(8,5))
plt.semilogy(range(1, len(erreurs_newton_moy) + 1), erreurs_newton_moy, label="Erreur finale Newton", color='blue')
plt.xlabel("Pas de temps")
plt.ylabel("Erreur (log)")
plt.title("Erreur de Newton à chaque pas de temps")
plt.grid(True, which="both", ls="--")
plt.legend()
plt.tight_layout()
plt.show()


