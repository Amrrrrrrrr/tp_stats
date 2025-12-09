import numpy as np
import matplotlib.pyplot as plt

plt.close('all')


# --- Question 1 ---
data1 = np.load('./data1.npy')
X1 = data1[0, :].reshape(-1, 1)
y1 = data1[1, :]

# Implémentation OLS
X1_avec_biais = np.column_stack([np.ones(len(X1)), X1])
beta = np.linalg.inv(X1_avec_biais.T @ X1_avec_biais) @ X1_avec_biais.T @ y1
beta0, beta1 = beta[0], beta[1]

# Erreur 
y_pred1 = beta0 + beta1 * X1.flatten()
erreur1 = np.mean((y1 - y_pred1)**2)

print("Question 1 - data1.npy:")
print(f"Droite: y = {beta0:.4f} + {beta1:.4f}x")
print(f"Erreur: {erreur1:.6f}\n")



# --- Question 2 ---
data2 = np.load('./data2.npy')
X2 = data2[0, :].reshape(-1, 1)
y2 = data2[1, :]

# Même méthode sur data2
X2_avec_biais = np.column_stack([np.ones(len(X2)), X2])
beta2 = np.linalg.inv(X2_avec_biais.T @ X2_avec_biais) @ X2_avec_biais.T @ y2
beta0_2, beta1_2 = beta2[0], beta2[1]

y_pred2 = beta0_2 + beta1_2 * X2.flatten()
erreur2 = np.mean((y2 - y_pred2)**2)

print("Question 2 - data2.npy:")
print(f"Droite: y = {beta0_2:.4f} + {beta1_2:.4f}x")
print(f"Erreur: {erreur2:.6f}")
print("→ Erreur beaucoup plus élevée, données non-linéaires!")


# Graphiques
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(X1, y1, alpha=0.5)
x_line = np.linspace(X1.min(), X1.max(), 100)
plt.plot(x_line, beta0 + beta1*x_line, 'r-', label='OLS')
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'data1.npy (Erreur={erreur1:.3f})')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(X2, y2, alpha=0.5)
x_line = np.linspace(X2.min(), X2.max(), 100)
plt.plot(x_line, beta0_2 + beta1_2*x_line, 'r-', label='OLS')
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'data2.npy (Erreur={erreur2:.3f})')
plt.legend()

plt.tight_layout()
plt.show()


# --- Question 3 ---

print("\nQuestion 3 - Extension OLS avec fonctions de base φ:")
print("Modèle: f(x) = Σ βi φi(x)")
print("Matrice Φ: Φ[i,j] = φj(xi)")
print("Solution: β̂ = (Φ^T Φ)^(-1) Φ^T y")
print("Hypothèse nécessaire: Φ^T Φ inversible\n")


# --- Question 4 ---

print("\nQuestion 4 - Modèle polynomial sur data2:")

x2 = X2.flatten()

degre_choisi = 3  # ← Change ici (essaye 2, 3, 4, 5, 10...)

print("Test de différents degrés:")
for k in [3, 5, 7, 9, 10, 11]:
    # Construire la matrice Φ pour degré k
    Phi_k = np.ones((len(x2), k+1))
    for i in range(k+1):
        Phi_k[:, i] = x2**i
    
    # Résolution OLS
    beta_k = np.linalg.inv(Phi_k.T @ Phi_k) @ Phi_k.T @ y2
    erreur_k = np.mean((y2 - Phi_k @ beta_k)**2)
    print(f"  Degré {k}: erreur = {erreur_k:.6f}")


degre_choisi = 10 

Phi2 = np.ones((len(x2), degre_choisi+1))
for i in range(degre_choisi+1):
    Phi2[:, i] = x2**i

beta_poly = np.linalg.inv(Phi2.T @ Phi2) @ Phi2.T @ y2
y_pred_poly = Phi2 @ beta_poly
erreur_poly = np.mean((y2 - y_pred_poly)**2)

print(f"\nErreur: {erreur_poly:.6f} avec un polynôme de degré {degre_choisi}")
print(f"Amélioration vs linéaire: {(1 - erreur_poly/erreur2)*100:.1f}%\n")


# --- Question 5 ---

print("Question 5 - data3.npy avec polynôme degré 10:")
data3 = np.load('./data3.npy')
X3 = data3[0, :].reshape(-1, 1)
y3 = data3[1, :]

# Construire Φ pour degré 10
x3 = X3.flatten()
Phi3 = np.ones((len(x3), 11))  # 11 colonnes pour degré 10
for i in range(11):
    Phi3[:, i] = x3**i

# Résolution OLS
beta3 = np.linalg.inv(Phi3.T @ Phi3) @ Phi3.T @ y3

y_pred3 = Phi3 @ beta3
erreur3 = np.mean((y3 - y_pred3)**2)

print(f"Nombre d'échantillons: {len(X3)}")
print("Nombre de coefficients: 11")
print(f"Erreur: {erreur3:.6f}")
print(f"Max |coefficient|: {np.max(np.abs(beta3)):.2e}")

if np.max(np.abs(beta3)) > 1000:
    print("→ Sur-apprentissage! Coefficients trop grands!")
    print("Trop de complexité (degré 10) pour peu de données (20 pt)")
# Graphiques
plt.figure(figsize=(10, 4))

# Graph 1: data2 linéaire vs polynomial
plt.subplot(1, 2, 1)
plt.scatter(X2, y2, alpha=0.5, label='Données')
x_plot = np.linspace(X2.min(), X2.max(), 200)
# Droite OLS
plt.plot(x_plot, beta0_2 + beta1_2*x_plot, 'r--', label='Linéaire', alpha=0.7)
# Polynôme degré choisi
Phi_plot = np.ones((len(x_plot), degre_choisi+1))
for i in range(degre_choisi+1):
    Phi_plot[:, i] = x_plot**i
y_plot = Phi_plot @ beta_poly

plt.plot(x_plot, y_plot, 'g-', label=f'Polynôme degré {degre_choisi}')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Comparaison des modèles (degré 1 vs degré 10)')
plt.legend()

# Graph 2: data3 avec degré 10
plt.subplot(1, 2, 2)
plt.scatter(X3, y3, alpha=0.5, label='Données')
x_plot3 = np.linspace(X3.min(), X3.max(), 200)

Phi_plot3 = np.ones((len(x_plot3), 11))
for i in range(11):
    Phi_plot3[:, i] = x_plot3**i
y_plot3 = Phi_plot3 @ beta3

plt.plot(x_plot3, y_plot3, 'r-', label='Polynôme degré 10')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Modèle polynomiale de degré 10 sur les data3')
plt.legend()
