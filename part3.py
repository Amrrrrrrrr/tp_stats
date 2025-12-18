###############################################################################
# QUESTION 2 - Classification OLS / Ridge
###############################################################################

import numpy as np
import matplotlib.pyplot as plt
from classif import X, y
from sklearn.linear_model import LogisticRegression

# --------------------------------------------------
# Données générées par classif.py
# X : (n,3) = [1, x1, x2]
# y : (n,)  = -1 / +1
# --------------------------------------------------

X2 = X[:, 1:3]   # pour affichage (sans intercept)

# --------------------------------------------------
# OLS
# --------------------------------------------------
beta_ols = np.linalg.inv(X.T @ X) @ (X.T @ y)

# --------------------------------------------------
# Ridge
# --------------------------------------------------
lam = 0.1
n = X.shape[0]
I = np.eye(X.shape[1])
beta_ridge = np.linalg.inv(X.T @ X + n * lam * I) @ (X.T @ y)

# --------------------------------------------------
# Prédictions + accuracy
# --------------------------------------------------
yhat_ols = np.sign(X @ beta_ols)
yhat_ridge = np.sign(X @ beta_ridge)

acc_ols = np.mean(yhat_ols == y)
acc_ridge = np.mean(yhat_ridge == y)

print("\n--- Classification ---")
print(f"Accuracy OLS   : {acc_ols:.3f}")
print(f"Accuracy Ridge : {acc_ridge:.3f}")

# --------------------------------------------------
# Tracé des frontières
# beta0 + beta1 x + beta2 y = 0
# => y = -(beta0 + beta1 x)/beta2
# --------------------------------------------------
x_min, x_max = X2[:,0].min() - 1, X2[:,0].max() + 1
xs = np.linspace(x_min, x_max, 200)

plt.figure(figsize=(7,7))

# Nuage de points
plt.scatter(X2[y==-1,0], X2[y==-1,1], marker='x', label='Classe -1')
plt.scatter(X2[y==+1,0], X2[y==+1,1], marker='x', label='Classe +1')

# Frontière OLS
ys_ols = -(beta_ols[0] + beta_ols[1]*xs) / beta_ols[2]
plt.plot(xs, ys_ols, 'r-', label='OLS')

# Frontière Ridge
ys_ridge = -(beta_ridge[0] + beta_ridge[1]*xs) / beta_ridge[2]
plt.plot(xs, ys_ridge, 'g--', label=f'Ridge λ={lam}')

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Classification OLS vs Ridge")
plt.legend()
plt.grid(True)
plt.show()

# --------------------------------------------------
# Entraînement du classifieur logistique
# - pénalité L2 par défaut
# - C = 1/λ (plus C est petit, plus la régularisation est forte)
# --------------------------------------------------

logreg = LogisticRegression(
    penalty='l2',
    C=1.0,
    fit_intercept=True,
    solver='lbfgs'
)

# sklearn attend X sans la colonne de 1
logreg.fit(X2, y)

# Coefficients
beta0_log = logreg.intercept_[0]
beta_log = logreg.coef_[0]   # [β1, β2]

# Accuracy
acc_log = logreg.score(X2, y)

print(f"Accuracy Logistic : {acc_log:.3f}")

# --------------------------------------------------
# Frontière logistique
# β0 + β1 x + β2 y = 0
# --------------------------------------------------
ys_log = -(beta0_log + beta_log[0]*xs) / beta_log[1]

# --------------------------------------------------
# Tracé comparatif
# --------------------------------------------------
plt.figure(figsize=(7,7))

plt.scatter(X2[y==-1,0], X2[y==-1,1], marker='x', label='Classe -1')
plt.scatter(X2[y==+1,0], X2[y==+1,1], marker='x', label='Classe +1')

plt.plot(xs, ys_ols, 'r-', label='OLS')
plt.plot(xs, ys_ridge, 'g--', label='Ridge')
plt.plot(xs, ys_log, 'b-.', label='Logistic')

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("OLS vs Ridge vs Logistic Regression")
plt.legend()
plt.grid(True)
plt.show()