import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Question 1: Régression OLS sur data1.npy
data1 = np.load('./data1.npy')
X1 = data1[0, :].reshape(-1, 1)
y1 = data1[1, :]

# Implémentation manuelle OLS
X1_avec_biais = np.column_stack([np.ones(len(X1)), X1])
beta = np.linalg.inv(X1_avec_biais.T @ X1_avec_biais) @ X1_avec_biais.T @ y1
beta0, beta1 = beta[0], beta[1]

# Erreur d'apprentissage
y_pred1 = beta0 + beta1 * X1.flatten()
erreur1 = np.mean((y1 - y_pred1)**2)

print(f"Question 1 - data1.npy:")
print(f"Droite: y = {beta0:.4f} + {beta1:.4f}x")
print(f"Erreur: {erreur1:.6f}\n")

# Question 2: Régression OLS sur data2.npy
data2 = np.load('./data2.npy')
X2 = data2[0, :].reshape(-1, 1)
y2 = data2[1, :]

# Même méthode sur data2
X2_avec_biais = np.column_stack([np.ones(len(X2)), X2])
beta2 = np.linalg.inv(X2_avec_biais.T @ X2_avec_biais) @ X2_avec_biais.T @ y2
beta0_2, beta1_2 = beta2[0], beta2[1]

y_pred2 = beta0_2 + beta1_2 * X2.flatten()
erreur2 = np.mean((y2 - y_pred2)**2)

print(f"Question 2 - data2.npy:")
print(f"Droite: y = {beta0_2:.4f} + {beta1_2:.4f}x")
print(f"Erreur: {erreur2:.6f}")
print(f"→ Erreur beaucoup plus élevée, données non-linéaires!")

# Graphiques
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(X1, y1, alpha=0.5)
x_line = np.linspace(X1.min(), X1.max(), 100)
plt.plot(x_line, beta0 + beta1*x_line, 'r-', label=f'OLS')
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'data1.npy (Erreur={erreur1:.3f})')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(X2, y2, alpha=0.5)
x_line = np.linspace(X2.min(), X2.max(), 100)
plt.plot(x_line, beta0_2 + beta1_2*x_line, 'r-', label=f'OLS')
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'data2.npy (Erreur={erreur2:.3f})')
plt.legend()

plt.tight_layout()
plt.savefig('q1_q2.png')
plt.show()