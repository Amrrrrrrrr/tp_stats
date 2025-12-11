import numpy as np
import matplotlib.pyplot as plt

#sinon lasso se plaint
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)


plt.close('all')
print(60*"=")
print("TP STATISTIQUES")
print(60*"=")
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

print("-"*60)
print("Question 1 - data1.npy:")
print(f"\nDroite: y = {beta0:.4f} + {beta1:.4f}x")
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

print("-"*60)
print("Question 2 - data2.npy:")
print(f"\nDroite: y = {beta0_2:.4f} + {beta1_2:.4f}x")
print(f"Erreur: {erreur2:.6f}")
print("→ Erreur beaucoup plus élevée, données non-linéaires!")


# Graphiques
fig = plt.figure(figsize=(10, 4))
fig.suptitle("Q2 : Méthode OLS sur les data1 et data2")

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

print("-"*60)
print("Question 3 - Extension OLS avec fonctions de base φ:")
print("\nModèle: f(x) = Σ βi φi(x)")
print("Matrice Φ: Φ[i,j] = φj(xi)")
print("Solution: β̂ = (Φ^T Φ)^(-1) Φ^T y")
print("Hypothèse nécessaire: Φ^T Φ inversible\n")


# --- Question 4 ---

print("-"*60)
print("Question 4 - Modèle polynomial sur data2:")

x2 = X2.flatten()

degre_choisi = 3  # ← Change ici (essaye 2, 3, 4, 5, 10...)

print("\nTest de différents degrés:")
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

print("-"*60)
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

print(f"\nNombre d'échantillons: {len(X3)}")
print("Nombre de coefficients: 11")
print(f"Erreur: {erreur3:.6f}")
print(f"Max |coefficient|: {np.max(np.abs(beta3)):.2e}")

if np.max(np.abs(beta3)) > 1000:
    print("→ Sur-apprentissage! Coefficients trop grands!")
    print("Trop de complexité (degré 10) pour peu de données (20 pt)")
# Graphiques
fig2 = plt.figure(figsize=(10, 4))
fig2.suptitle("Q5 : Regression avec polynômes de degré 10 sur les data2 et data3")
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


# --- Question 6 ---

print("-"*60)
print("Question 6 - Résolution Ridge:")
print("\nSolution: β̂_ridge = (X^T X + λI)^(-1) X^T y")
print("(voir document LaTeX séparé)\n")

# --- Question 7 ---

print("-"*60)
print("Question 7 - Comparaison OLS vs Ridge sur data3:")

def ridge_polynomial(X, y, degree, lam):
    """Ridge regression avec polynôme"""
    x = X.flatten()
    n = len(x)
    Phi = np.ones((n, degree + 1))
    for i in range(degree + 1):
        Phi[:, i] = x**i
    
    # Solution Ridge
    p = Phi.shape[1]
    I = np.eye(p)
    beta_ridge = np.linalg.inv(Phi.T @ Phi + lam * I) @ Phi.T @ y
    
    y_pred = Phi @ beta_ridge
    erreur = np.mean((y - y_pred)**2)
    return beta_ridge, erreur

# Test avec différents λ
print("\nDegré 10 sur data3:")
lambdas = [0, 0.001, 0.01, 0.1, 1.0]
for lam in lambdas:
    beta_r, err_r = ridge_polynomial(X3, y3, 10, lam)
    print(f"λ={lam:5.3f}: erreur={err_r:.6f}, ||β||={np.linalg.norm(beta_r):.2e}")

print("\n→ λ=0 (OLS): erreur minimale MAIS coefficients énormes = sur-apprentissage")
print("→ λ>0 (Ridge): erreur légèrement plus grande MAIS modèle stable = meilleure généralisation")

# Graphique comparatif
plt.figure(figsize=(8, 8))
plt.scatter(X3, y3, alpha=0.7, s=50, label='Données') # Points
plt.plot(x_plot3, Phi_plot3 @ beta3, 'r-', label='OLS', alpha=0.7) # OLS 

# Ridge (λ=0.1)
beta_ridge, _ = ridge_polynomial(X3, y3, 10, 0.1)
plt.plot(x_plot3, Phi_plot3 @ beta_ridge, 'g--', label='Ridge λ=0.1', linewidth=2)

plt.xlabel('x')
plt.ylabel('y')
plt.title('Q7: OLS vs Ridge')
plt.legend()
plt.ylim([y3.min()-2, y3.max()+2])
plt.tight_layout()
plt.show()


# --- Question 8 ---

print("-"*60)
print("Question 8 - LASSO:")
print("\nmin (1/n) Σ(yi - β^T xi)² + λ Σ|βi|")
print("→ Pas de solution analytique (|β| non-différentiable en 0)")
print("→ Utilisation de sklearn.linear_model.Lasso\n")


from sklearn.linear_model import Lasso

# Réutiliser Phi3 de la question 5, mais sans la colonne de 1
# car Lasso gère l'intercept séparément
X3_for_lasso = Phi3[:, 1:] 

# Test avec différents α pour voir l'impact
alphas = [0.001, 0.01, 0.1, 0.5]
print("Impact de la régularisation L1:")
for alpha in alphas:
    lasso = Lasso(alpha=alpha, fit_intercept=True, max_iter=10000)
    lasso.fit(X3_for_lasso, y3)
    n_nonzero = np.sum(np.abs(lasso.coef_) > 1e-10)
    err = np.mean((y3 - lasso.predict(X3_for_lasso))**2)
    print(f"α={alpha:5.3f}: erreur={err:.4f}, coefs non-nuls={n_nonzero}/10")


print("\n- α petit : presque tous les coefficients gardés")
print("- α grand : sélection agressive, peu de coefficients")
print("- Contrairement à L2 qui réduit uniformément, L1 élimine des variables")

# On prend α=0.001 car c'est le meilleur compromis
lasso_best = Lasso(alpha=0.001, fit_intercept=True, max_iter=10000)
lasso_best.fit(X3_for_lasso, y3)

plt.figure(figsize=(8, 8))
plt.scatter(X3, y3, alpha=0.7, s=50, label='Données')
plt.plot(x_plot3, Phi_plot3 @ beta3, 'r-', label='OLS', alpha=0.7)
plt.plot(x_plot3, Phi_plot3 @ beta_ridge, 'g--', label='Ridge λ=0.1', linewidth=2)

# LASSO α=0.1
X_plot_lasso = Phi_plot3[:, 1:]
y_plot_lasso = lasso_best.predict(X_plot_lasso)
plt.plot(x_plot3, y_plot_lasso, 'b-.', label='LASSO α=0.001', linewidth=2)

plt.xlabel('x')
plt.ylabel('y')
plt.title('Q8 - Comparaison des 3 méthodes')
plt.legend()
plt.ylim([y3.min()-2, y3.max()+2])
plt.tight_layout()
plt.show()

print("-"*60)
print("\nRÉSUMÉ DES PERFORMANCES:")

print("OLS (degré 10)    : erreur = 0.0736, 11 coefficients --> Enorme !")
print("Ridge (λ=0.1)     : erreur = 0.0798, 11 coefficients mais contrôlés !")
print("LASSO (α=0.001)   : erreur = 0.0940, 10 coefficients --> sparse")

print("\nANALYSE:")

print("- OLS a Meilleure erreur sur les données, mais coefficients énormes (||β|| ≈ 10⁴) ")
print("  Le modèle colle le bruit au lieu d'apprendre la tendance.")
print(" ==>  Biais faible, Variance élevée  → sur-apprentissage")

print("\n- Ridge (L2) augmente légèrement l'erreur, mais ses coefficients")
print("  sont raisonnables.")
print(" ==> Biais modéré, Variance réduite → bon compromis")

print("\n- Lasso (L1) a l'erreur la plus élevée car il force des coefs à 0")
print(" ==> Utile pour sélectionner les variables, mais ici ça aplatit trop")



print("\nCONCLUSION:")

print("\nRidge est le meilleur choix ici : il régularise sans trop perdre en précision.")
print("Lasso est trop agressif pour ce problème (peu de données) mais il aurait été mieux s'il avait fallu supprimer beaucoup de variables")
print("Dans tous les cas, ces deux méthodes sont largement mieux que OLS qui créé du sur-apprentissage")
print("-"*60)